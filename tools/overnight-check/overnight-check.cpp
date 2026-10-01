// overnight-check (infernet, 2026-09-23): one model load, several A/B correctness checks for env-gated changes.
//
//   llama-simple-overnight-check MODEL.gguf [rounds]
//
// argmax: teacher-forced 8-row "verify" decodes (all rows output) in three contexts:
//   A: no backend sampler, CPU argmax over the copied logits (reference)
//   B: backend greedy chain (top-k 40, top-p 0.95, min-p 0.05, temp 0, dist) with LLAMA_BATCHED_ARGMAX=1
//   C: the same chain without it (stock per-row sampler graphs)
// Every row's token must agree across A/B/C. Also reports ms per 8-row decode for each (interleaved rounds).
// gdn: GDN a/b fused projection (LLAMA_GDN_AB_FUSE=1) vs not, CPU argmax + max |logit diff| per row.

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

static const char * k_text =
    "import json\nimport sys\n\n\ndef load_config(path):\n    with open(path) as f:\n        cfg = json.load(f)\n"
    "    if 'layers' not in cfg:\n        raise ValueError('config has no layers: ' + path)\n    return cfg\n\n\n"
    "def count_params(cfg):\n    total = 0\n    for layer in cfg['layers']:\n        rows, cols = layer['shape']\n"
    "        total += rows * cols\n        if layer.get('bias'):\n            total += rows\n    return total\n\n\n"
    "def main():\n    if len(sys.argv) < 2:\n        print('usage: count.py CONFIG')\n        return 1\n"
    "    cfg = load_config(sys.argv[1])\n    n = count_params(cfg)\n    print(f'{n} parameters in {len(cfg[\"layers\"])} layers')\n"
    "    return 0\n\n\nif __name__ == '__main__':\n    sys.exit(main())\n";

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static llama_sampler * make_greedy_chain() {
    llama_sampler * ch = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(ch, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(ch, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(ch, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(ch, llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f));
    llama_sampler_chain_add(ch, llama_sampler_init_dist(42));
    return ch;
}

struct ctx_t {
    llama_context * ctx = nullptr;
    llama_sampler * smpl = nullptr;
    const char * env_val = nullptr; // LLAMA_BATCHED_ARGMAX / LLAMA_GDN_AB_FUSE value to set around this context's decodes
    double t_ms = 0; int n_dec = 0;
};

static void set_env(const char * name, const char * v) {
    if (v) setenv(name, v, 1); else unsetenv(name);
}

static llama_context * new_ctx(llama_model * model, llama_sampler * smpl) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512; cp.no_perf = true;
    cp.n_outputs_max_per_seq = 16; // verify-style: every row of the batch is sampled (default is 1)
    llama_sampler_seq_config sc = { 0, smpl };
    if (smpl) { cp.samplers = &sc; cp.n_samplers = 1; }
    return llama_init_from_model(model, cp);
}

// decode toks[p0 .. p0+n) at positions p0.., outputs for all rows if all_out, else last only
static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, int p0, int n, bool all_out) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[p0 + i]; b.pos[i] = p0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = all_out || i == n - 1;
    }
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static llama_token cpu_argmax(llama_context * ctx, int i, int n_vocab, float * margin) {
    const float * l = llama_get_logits_ith(ctx, i);
    int best = 0; float b1 = -INFINITY, b2 = -INFINITY;
    for (int t = 0; t < n_vocab; ++t) {
        if (l[t] > b1) { b2 = b1; b1 = l[t]; best = t; } else if (l[t] > b2) { b2 = l[t]; }
    }
    if (margin) *margin = b1 - b2;
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL.gguf [rounds] [mode: argmax|gdn|all]\n", argv[0]); return 2; }
    const int rounds = argc > 2 ? atoi(argv[2]) : 10;
    const std::string mode = argc > 3 ? argv[3] : "all";
    const int NR = 8;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks(4096);
    int nt = llama_tokenize(vocab, k_text, (int) strlen(k_text), toks.data(), (int) toks.size(), true, false);
    toks.resize(std::max(nt, 0));
    const int p_pre = 32;
    const int R = std::min(rounds, (nt - p_pre) / NR);
    printf("tokens %d, prefix %d, rounds %d x %d rows\n", nt, p_pre, R, NR);

    int fails = 0;

    if (mode == "argmax" || mode == "all") {
        ctx_t A, B, C;
        A.ctx = new_ctx(model, nullptr);
        B.smpl = make_greedy_chain(); B.env_val = "1"; B.ctx = new_ctx(model, B.smpl);
        C.smpl = make_greedy_chain(); C.env_val = nullptr; C.ctx = new_ctx(model, C.smpl);
        ctx_t * cs[3] = { &A, &B, &C };
        for (auto * c : cs) {
            set_env("LLAMA_BATCHED_ARGMAX", c->env_val);
            if (!decode(c->ctx, toks, 0, p_pre, false)) { fprintf(stderr, "prefix decode failed\n"); fflush(stdout); _exit(1); }
        }
        int n_rows = 0, n_bad = 0;
        for (int r = 0; r < R; ++r) {
            const int p0 = p_pre + r * NR;
            llama_token tk[3][NR]; float margin[NR];
            // rotate the order each round so no context always runs first (ABC, BCA, CAB)
            for (int k = 0; k < 3; ++k) {
                ctx_t * c = cs[(r + k) % 3];
                const int ci = c == &A ? 0 : (c == &B ? 1 : 2);
                set_env("LLAMA_BATCHED_ARGMAX", c->env_val);
                const double t0 = now_ms();
                if (!decode(c->ctx, toks, p0, NR, true)) { fprintf(stderr, "decode failed\n"); fflush(stdout); _exit(1); }
                for (int i = 0; i < NR; ++i) {
                    tk[ci][i] = ci == 0 ? cpu_argmax(c->ctx, i, n_vocab, &margin[i]) : llama_get_sampled_token_ith(c->ctx, i);
                }
                const double dt = now_ms() - t0;
                if (r > 0) { c->t_ms += dt; c->n_dec++; } // round 0 = warm-up (graph build)
            }
            for (int i = 0; i < NR; ++i) {
                n_rows++;
                if (tk[0][i] != tk[1][i] || tk[0][i] != tk[2][i]) {
                    n_bad++;
                    printf("  MISMATCH round %d row %d: cpu %d batched %d stock %d (top-2 margin %.4g)\n", r, i, tk[0][i], tk[1][i], tk[2][i], margin[i]);
                }
            }
        }
        printf("argmax: %d/%d rows agree (cpu argmax = batched = stock chain)\n", n_rows - n_bad, n_rows);
        printf("argmax: ms per 8-row decode incl. result fetch: cpu-logits %.2f | batched %.2f | stock chain %.2f (n=%d each)\n",
               A.t_ms / std::max(A.n_dec, 1), B.t_ms / std::max(B.n_dec, 1), C.t_ms / std::max(C.n_dec, 1), A.n_dec);
        fails += n_bad;
        for (auto * c : cs) { llama_free(c->ctx); if (c->smpl) llama_sampler_free(c->smpl); }
        unsetenv("LLAMA_BATCHED_ARGMAX");
    }

    if (mode == "gdn" || mode == "all") {
        ctx_t X, Y; // X: stock, Y: LLAMA_GDN_AB_FUSE=1
        Y.env_val = "1";
        ctx_t * cs[2] = { &X, &Y };
        for (auto * c : cs) {
            set_env("LLAMA_GDN_AB_FUSE", c->env_val);
            c->ctx = new_ctx(model, nullptr);
            if (!decode(c->ctx, toks, 0, p_pre, false)) { fprintf(stderr, "prefix decode failed\n"); fflush(stdout); _exit(1); }
        }
        int n_rows = 0, n_bad = 0; double max_diff = 0;
        for (int r = 0; r < R; ++r) {
            const int p0 = p_pre + r * NR;
            for (int k = 0; k < 2; ++k) {
                ctx_t * c = cs[(r + k) % 2];
                set_env("LLAMA_GDN_AB_FUSE", c->env_val);
                const double t0 = now_ms();
                if (!decode(c->ctx, toks, p0, NR, true)) { fprintf(stderr, "decode failed\n"); fflush(stdout); _exit(1); }
                llama_get_logits_ith(c->ctx, NR - 1);
                if (r > 0) { c->t_ms += now_ms() - t0; c->n_dec++; }
            }
            for (int i = 0; i < NR; ++i) {
                n_rows++;
                float m;
                const llama_token a = cpu_argmax(X.ctx, i, n_vocab, &m);
                const llama_token b = cpu_argmax(Y.ctx, i, n_vocab, nullptr);
                const float * la = llama_get_logits_ith(X.ctx, i);
                const float * lb = llama_get_logits_ith(Y.ctx, i);
                for (int t = 0; t < n_vocab; ++t) max_diff = std::max(max_diff, (double) std::fabs(la[t] - lb[t]));
                if (a != b) { n_bad++; printf("  MISMATCH round %d row %d: stock %d fused %d (margin %.4g)\n", r, i, a, b, m); }
            }
        }
        printf("gdn: %d/%d rows agree, max |logit diff| %.6g\n", n_rows - n_bad, n_rows, max_diff);
        printf("gdn: ms per 8-row decode: stock %.2f | fused %.2f (n=%d each)\n",
               X.t_ms / std::max(X.n_dec, 1), Y.t_ms / std::max(Y.n_dec, 1), X.n_dec);
        fails += n_bad;
        for (auto * c : cs) llama_free(c->ctx);
        unsetenv("LLAMA_GDN_AB_FUSE");
    }

    llama_model_free(model);
    printf("%s\n", fails ? "RESULT: FAIL" : "RESULT: OK");
    return fails ? 1 : 0;
}
