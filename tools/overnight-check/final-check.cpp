// final-check (infernet, 2026-09-23): the night's last whole-model run, several A/B checks from one model load.
//
//   llama-simple-overnight-final TARGET.gguf DRAFT.gguf [rounds] [cpu]
//
// Run with GGML_METAL_REGFED=1 GGML_METAL_REGFED_KSPLIT_DYNAMIC=1 (GGML_METAL_FA_GQA=1 optional).
//  1. split-K end to end: two target contexts decode the same teacher-forced 8-row "verify" batches (all rows output);
//     context A with GGML_METAL_REGFED_KSPLIT=1 (split-K off), B with the defaults (env unset), order alternating per round.
//     Reports ms per 8-row decode (wall incl. logits fetch), argmax agreement per row and max |logit diff|.
//  2. drafter graph reuse (LLAMA_GRAPH_REUSE2): two DFlash2 drafter contexts (stock / REUSE2) on target context A, fed the
//     same inject features and draft blocks; full-lattice comparison every round, host ms inside llama_decode and wall ms.

#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

static void die(const char * msg) { fprintf(stderr, "%s\n", msg); fflush(stdout); _exit(1); }

static llama_token argmax_row(const float * l, int n_vocab, float * margin) {
    int best = 0; float b1 = -INFINITY, b2 = -INFINITY;
    for (int t = 0; t < n_vocab; ++t) {
        if (l[t] > b1) { b2 = b1; b1 = l[t]; best = t; } else if (l[t] > b2) { b2 = l[t]; }
    }
    if (margin) *margin = b1 - b2;
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s TARGET.gguf DRAFT.gguf [rounds]\n", argv[0]); return 2; }
    const int rounds = argc > 3 ? atoi(argv[3]) : 20;
    const int NR = 8;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_devs[1] = { nullptr };
    if (argc > 4 && std::string(argv[4]) == "cpu") { mp.devices = no_devs; mp.n_gpu_layers = 0; } // dry run of the harness
    llama_model * model_tgt = llama_model_load_from_file(argv[1], mp);
    if (!model_tgt) die("target load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // ---- 1. split-K end to end ----
    llama_context * T[2];
    for (int c = 0; c < 2; ++c) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = 64; cp.n_ubatch = 64; cp.no_perf = true;
        T[c] = llama_init_from_model(model_tgt, cp);
        if (!T[c]) die("target ctx failed");
    }
    const char * ks_val[2] = { "1", nullptr }; // A: split-K off, B: defaults

    std::vector<llama_token> toks(4096);
    const int nt = std::max(0, llama_tokenize(vocab, k_text, (int) strlen(k_text), toks.data(), (int) toks.size(), true, false));
    toks.resize(nt);
    const int p_pre = 32;
    const int R = std::min(rounds, (nt - p_pre) / NR);
    printf("tokens %d, prefix %d, rounds %d x %d rows\n", nt, p_pre, R, NR);

    auto decode = [&](llama_context * ctx, int p0, int n, bool all_out) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = toks[p0 + i]; b.pos[i] = p0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = all_out || i == n - 1;
        }
        b.n_tokens = n;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) die("target decode failed");
    };
    auto set_ks = [&](int c) {
        if (ks_val[c]) setenv("GGML_METAL_REGFED_KSPLIT", ks_val[c], 1); else unsetenv("GGML_METAL_REGFED_KSPLIT");
    };
    for (int c = 0; c < 2; ++c) { set_ks(c); decode(T[c], 0, p_pre, false); llama_synchronize(T[c]); }

    double t_ms[2] = {0, 0}; int n_meas = 0, n_rows = 0, n_agree = 0; double max_diff = 0;
    std::vector<float> la((size_t) NR * n_vocab);
    for (int r = 0; r < R; ++r) {
        const int p0 = p_pre + r * NR;
        for (int k = 0; k < 2; ++k) {
            const int c = (r + k) % 2;
            set_ks(c);
            const double t0 = now_ms();
            decode(T[c], p0, NR, true);
            llama_get_logits_ith(T[c], NR - 1); // syncs
            if (r > 0) t_ms[c] += now_ms() - t0;
        }
        if (r > 0) n_meas++;
        for (int i = 0; i < NR; ++i) {
            const float * a = llama_get_logits_ith(T[0], i);
            const float * b = llama_get_logits_ith(T[1], i);
            float m;
            const llama_token ta = argmax_row(a, n_vocab, &m), tb = argmax_row(b, n_vocab, nullptr);
            n_rows++; n_agree += ta == tb;
            if (ta != tb) printf("  argmax differs: round %d row %d: off %d on %d (top-2 margin %.4g)\n", r, i, ta, tb, m);
            for (int t = 0; t < n_vocab; ++t) max_diff = std::max(max_diff, (double) std::fabs(a[t] - b[t]));
        }
    }
    unsetenv("GGML_METAL_REGFED_KSPLIT");
    printf("splitk: ms per 8-row decode: KSPLIT=1 %.2f | default %.2f (n=%d each, interleaved) | argmax %d/%d rows agree, max |logit diff| %.4g\n",
           t_ms[0] / n_meas, t_ms[1] / n_meas, n_meas, n_agree, n_rows, max_diff);
    fflush(stdout);
    llama_free(T[1]);

    // ---- 2. drafter graph reuse ----
    llama_context * ctx_tgt = T[0];
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) die("draft load failed");
    const int n_embd_enc = (int) llama_model_target_layer_ids_n(model) * llama_model_n_embd(model_tgt);
    const int n_embd_dec = llama_model_n_embd(model);
    const bool is_mrope  = llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE;
    const llama_token mask = llama_vocab_mask(llama_model_get_vocab(model));
    const int NV = 8, NB = 8;

    llama_context * D[2];
    for (int c = 0; c < 2; ++c) {
        if (c == 1) setenv("LLAMA_GRAPH_REUSE2", "1", 1); else unsetenv("LLAMA_GRAPH_REUSE2");
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096; cp.n_batch = 64; cp.n_ubatch = 64; cp.no_perf = true;
        cp.n_outputs_max = NB;
        cp.ctx_other = ctx_tgt;
        D[c] = llama_init_from_model(model, cp);
        if (!D[c]) die("draft ctx failed");
        llama_set_embeddings_nextn(D[c], true, /*masked*/ false);
        llama_set_causal_attn(D[c], false);
    }
    unsetenv("LLAMA_GRAPH_REUSE2");

    llama_batch bi = llama_batch_init(64, n_embd_enc, 1);
    if (is_mrope) { free(bi.pos); bi.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * 64); }
    llama_batch bd = llama_batch_init(64, 0, 1);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    double h_inj[2] = {0, 0}, h_dft[2] = {0, 0}, w_inj[2] = {0, 0}, w_dft[2] = {0, 0};
    int dm = 0, n_cmp = 0, n_diff = 0;
    llama_pos p = 0;
    llama_token id_last = 1000;
    std::vector<float> lat0((size_t) NB * n_embd_dec);
    const int DR = argc > 4 && std::string(argv[4]) == "cpu" ? 4 : 40;
    for (int r = 0; r < DR; ++r) {
        for (int i = 0; i < NV; ++i) {
            for (int k = 0; k < n_embd_enc; ++k) bi.embd[(size_t) i * n_embd_enc + k] = 0.05f * nd(rng);
        }
        const int acc = 1 + (r * 5) % NV;
        llama_token id_next = id_last;
        for (int k = 0; k < 2; ++k) {
            const int c = (r + k) % 2;
            bi.n_tokens = NV;
            for (int i = 0; i < NV; ++i) {
                bi.pos[i] = p + i;
                if (is_mrope) { bi.pos[NV + i] = p + i; bi.pos[2 * NV + i] = p + i; bi.pos[3 * NV + i] = 0; }
                bi.n_seq_id[i] = 1; bi.seq_id[i][0] = 0; bi.logits[i] = false;
            }
            double t0 = now_ms();
            if (llama_decode(D[c], bi) != 0) die("inject failed");
            double t1 = now_ms();
            llama_synchronize(D[c]);
            double t2 = now_ms();
            if (r > 1) { h_inj[c] += t1 - t0; w_inj[c] += t2 - t0; }
            llama_memory_seq_rm(llama_get_memory(D[c]), 0, p + acc, -1);

            bd.n_tokens = NB;
            for (int i = 0; i < NB; ++i) {
                bd.token[i] = i == 0 ? id_last : mask; bd.pos[i] = p + acc + i;
                bd.n_seq_id[i] = 1; bd.seq_id[i][0] = 0; bd.logits[i] = false;
            }
            t0 = now_ms();
            if (llama_decode(D[c], bd) != 0) die("draft failed");
            t1 = now_ms();
            llama_synchronize(D[c]);
            t2 = now_ms();
            if (r > 1) { h_dft[c] += t1 - t0; w_dft[c] += t2 - t0; }
            const float * lat = llama_get_embeddings_nextn(D[c]);
            if (!lat) die("no lattice");
            if (k == 0) {
                std::memcpy(lat0.data(), lat, lat0.size() * sizeof(float));
                id_next = (llama_token) (1000 + ((int) std::fabs(lat[5]) * 7919) % 50000);
            } else {
                n_cmp++;
                if (std::memcmp(lat, lat0.data(), lat0.size() * sizeof(float)) != 0) n_diff++;
            }
            llama_memory_seq_rm(llama_get_memory(D[c]), 0, p + acc, -1);
        }
        if (r > 1) dm++;
        id_last = id_next;
        p += acc;
    }
    for (int c = 0; c < 2; ++c) {
        printf("drafter %s: inject host %.3f wall %.2f | draft host %.3f wall %.2f ms/decode (n=%d)\n",
               c == 0 ? "stock " : "reuse2", h_inj[c] / dm, w_inj[c] / dm, h_dft[c] / dm, w_dft[c] / dm, dm);
    }
    printf("drafter: lattice differs in %d of %d rounds (stock vs reuse2, same inputs)\n", n_diff, n_cmp);
    fflush(stdout);
    _exit(0); // skip teardown (Metal residency-set asserts at exit seen earlier); results are printed
}
