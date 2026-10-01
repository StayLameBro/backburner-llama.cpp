// llama-phone-kv-test: correctness gate for phone-held KV (infernet, docs/phone-kv-262k.md §5 step 2).
//
// Prefills a real text on the Mac, saves the state, then decodes the same continuation four ways:
//   mac-ar      greedy, one token per step, every key on the Mac
//   mac-chunk8  the mac-ar tokens again, 8 per step (the speculative-verify shape)
//   phone-ar    greedy, one token per step, the oldest EVICT positions moved to the phone
//   phone-chunk8
// and reports whether the phone runs pick the same tokens, and how far their logits are from the Mac-only runs.
// "noise floor" (mac-ar vs mac-chunk8) is only the GPU kernel against itself; it is not the GPU's error vs exact math
// (see the gate comment below and fa-bias-test.cpp). --evict takes a list: --evict 64,2048,3968.
//
//   llama-phone-kv-test -m MODEL -f TEXT [--prompt 4096] [--evict 2048] [--gen 48] [--ctx 8192]
//                       [--remote 127.0.0.1:50062] [--kv q8_0|q4_0|f16] [--grow MAC_CELLS [--restore STATE] [--rounds N]]
// Start the phone side first: the Sidecar app on the phone, or `phone-attn/build/pa-tool serve 50062` on this Mac.
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int argmax(const float * x, int n) {
    return (int) (std::max_element(x, x + n) - x);
}

struct cmp {
    int n = 0, same_top1 = 0;
    double max_abs = 0, sum_max_abs = 0, max_kl = 0, sum_kl = 0;
    void add(const float * ref, const float * t, int nv) {
        double ma = 0, mr = -1e30, mt = -1e30;
        for (int i = 0; i < nv; i++) {
            ma = std::max(ma, (double) std::fabs(ref[i] - t[i]));
            mr = std::max(mr, (double) ref[i]);
            mt = std::max(mt, (double) t[i]);
        }
        double zr = 0, zt = 0;
        for (int i = 0; i < nv; i++) { zr += std::exp(ref[i] - mr); zt += std::exp(t[i] - mt); }
        double kl = 0;
        for (int i = 0; i < nv; i++) {
            const double lr = ref[i] - mr - std::log(zr), lt = t[i] - mt - std::log(zt);
            kl += std::exp(lr)*(lr - lt);
        }
        if (getenv("PKV_KL_DETAIL")) {
            const int a = argmax(ref, nv), b = argmax(t, nv);
            printf("    pos %3d: KL %.2e | ref top-1 %d p=%.4f | test top-1 %d p=%.4f | max|dlogit| %.3f\n", n, kl,
                   a, std::exp(ref[a] - mr)/zr, b, std::exp(t[b] - mt)/zt, ma);
        }
        n++;
        same_top1 += argmax(ref, nv) == argmax(t, nv);
        max_abs = std::max(max_abs, ma); sum_max_abs += ma;
        max_kl = std::max(max_kl, kl); sum_kl += kl;
    }
    void print(const char * name) const {
        printf("  %-26s %3d positions: same top-1 %3d/%d | max|dlogit| mean %.4f max %.4f | KL mean %.2e max %.2e\n",
               name, n, same_top1, n, n ? sum_max_abs/n : 0.0, max_abs, n ? sum_kl/n : 0.0, max_kl);
    }
};

// debug (PKV_CAPTURE=1): each attention layer's output (kqv_out-<il>) during the restore's first decodes, per run
#include <map>
#include "ggml-backend.h"
struct capture {
    int run = 0, step = -1;   // step: 0 = the restore decode, 1.. = generation steps; -1 = off
    std::map<std::string, std::vector<float>> t;   // "run/step/name"
};
static capture g_cap;
static bool cap_cb(struct ggml_tensor * t, bool ask, void *) {
    const bool want = g_cap.step >= 0 && g_cap.step <= 3 && strncmp(t->name, "kqv_out-", 8) == 0 && t->type == GGML_TYPE_F32;
    if (ask) return want;
    if (want) {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        g_cap.t[std::to_string(g_cap.run) + "/" + std::to_string(g_cap.step) + "/" + t->name] = std::move(v);
    }
    return true;
}

// --grow MAC_CELLS: the context outgrows the Mac. A Mac-only context (room for everything) and a phone-attached context
// with only MAC_CELLS cells both prefill the prompt in 512-token batches and generate greedily; the phone context moves
// its oldest pages to the phone as it fills (LLAMA_KV_REMOTE_PAGE) and then runs 8-token ubatches.
static int run_grow(llama_model * model, const std::vector<llama_token> & toks_in, int n_gen, int mac_cells,
                    const std::string & remote, const std::string & kv, const std::string & restore_path, int n_rounds) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    // --restore FILE: start from a saved server slot (llama_state_seq_save_file) instead of prefilling the prompt;
    // generation starts by feeding one fixed token ("\n") after the saved context
    std::vector<llama_token> toks = toks_in;
    if (!restore_path.empty()) {
        FILE * fp = fopen(restore_path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "cannot open %s\n", restore_path.c_str()); return 1; }
        uint32_t magic = 0, version = 0, n = 0;
        fread(&magic, 4, 1, fp); fread(&version, 4, 1, fp); fread(&n, 4, 1, fp); fclose(fp);
        toks.resize(n);   // the token count; the tokens come back from the load below
    }
    const int n_prompt = (int) toks.size();
    auto make = [&](int n_ctx) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_seq_max = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = kv == "f16" ? GGML_TYPE_F16 : kv == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
        cp.no_perf = true;
        if (getenv("PKV_CAPTURE")) { cp.cb_eval = cap_cb; cp.cb_eval_user_data = nullptr; }
        if (!restore_path.empty()) {
            cp.n_rs_replay = 8;   // the saved slots come from serve-infernet.sh (--spec-gdn-replay 8): same recurrent layout
        }
        return llama_init_from_model(model, cp);
    };
    std::vector<std::vector<float>> first_logits;   // the restore's first decode, one per run
    // --rounds N: after the greedy tokens, N speculative-verify-shaped rounds: 8 tokens per decode, logits for all 8. The tokens
    // are the Mac-only run's greedy tokens again (the same input for both runs), so the logits compare position by position.
    std::vector<llama_token> round_toks;
    std::vector<std::vector<float>> round_L(2);
    double ms_round[2] = { 0, 0 };
    auto run = [&](llama_context * ctx, std::vector<llama_token> & out, std::vector<float> & L, double & t_pf, double & ms_tok) {
        llama_batch b = llama_batch_init(512, 0, 1);
        auto t0 = std::chrono::steady_clock::now();
        if (!restore_path.empty()) {
            std::vector<llama_token> saved(n_prompt + 16);
            size_t n_saved = 0;
            if (llama_state_seq_load_file(ctx, restore_path.c_str(), 0, saved.data(), saved.size(), &n_saved) == 0 || (int) n_saved != n_prompt) {
                fprintf(stderr, "restore of %s failed (%zu tokens)\n", restore_path.c_str(), n_saved); exit(1);
            }
            if (const char * e = getenv("PKV_EVICT_AFTER_RESTORE"); e && llama_kv_remote_n(ctx) >= 0 && atoi(e) > 0) {
                const int moved = llama_kv_remote_evict(ctx, atoi(e));   // debug: evict from the Mac instead of restoring to the phone
                if (moved > 0) printf("evicted %d positions after the restore\n", moved);
            }
            llama_token nl = 0; llama_tokenize(vocab, "\n", 1, &nl, 1, false, true);
            b.n_tokens = 1; b.token[0] = nl; b.pos[0] = n_prompt; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            g_cap.step = 0;
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode after restore failed\n"); exit(1); }
            g_cap.step = -1;
            first_logits.emplace_back(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + nv);   // debug: compared below
        }
        for (int p = 0; restore_path.empty() && p < n_prompt; p += 512) {
            const int n = std::min(512, n_prompt - p);
            b.n_tokens = n;
            for (int j = 0; j < n; j++) { b.token[j] = toks[p + j]; b.pos[j] = p + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = p + j == n_prompt - 1; }
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill decode failed at %d\n", p); exit(1); }
        }
        t_pf = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const int p0 = n_prompt + (restore_path.empty() ? 0 : 1);
        llama_token t = argmax(llama_get_logits_ith(ctx, -1), nv);
        out.assign(1, t); L.assign((size_t) n_gen*nv, 0);
        auto a = std::chrono::steady_clock::now();
        for (int i = 0; i < n_gen; i++) {
            // debug: PKV_EVICT_AT_STEP=S:N moves the N oldest positions to the phone just before generation step S
            if (const char * e = getenv("PKV_EVICT_AT_STEP"); e && llama_kv_remote_n(ctx) >= 0 && atoi(e) == i && strchr(e, ':')) {
                const int moved = llama_kv_remote_evict(ctx, atoi(strchr(e, ':') + 1));
                if (moved > 0) printf("evicted %d positions before step %d\n", moved, i);
            }
            b.n_tokens = 1; b.token[0] = t; b.pos[0] = p0 + i; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            g_cap.step = restore_path.empty() ? -1 : i + 1;
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", i); exit(1); }
            g_cap.step = -1;
            const float * lg = llama_get_logits_ith(ctx, 0);
            memcpy(&L[(size_t) i*nv], lg, sizeof(float)*nv);
            t = argmax(lg, nv); out.push_back(t);
        }
        ms_tok = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count()/n_gen;
        if (n_rounds > 0) {
            if (round_toks.empty()) round_toks = out;   // the first (Mac-only) run's tokens feed both runs
            const int ri = g_cap.run;
            auto & RL = round_L[ri];
            RL.assign((size_t) n_rounds*8*nv, 0);
            const int q0 = p0 + n_gen;
            auto r0 = std::chrono::steady_clock::now();
            for (int r = 0; r < n_rounds; r++) {
                b.n_tokens = 8;
                for (int j = 0; j < 8; j++) {
                    b.token[j] = round_toks[(r*8 + j) % round_toks.size()]; b.pos[j] = q0 + r*8 + j;
                    b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1;
                }
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "round decode failed at %d\n", r); exit(1); }
                for (int j = 0; j < 8; j++) memcpy(&RL[((size_t) r*8 + j)*nv], llama_get_logits_ith(ctx, j), sizeof(float)*nv);
            }
            ms_round[ri] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count()/n_rounds;
        }
        llama_batch_free(b);
    };

    std::vector<llama_token> ta, tb; std::vector<float> La, Lb; double pa_s, pb_s, ma, mb;
    {
        llama_context * ctx = make(n_prompt + n_gen + 8*n_rounds + 512);
        run(ctx, ta, La, pa_s, ma);
        llama_free(ctx);
    }
    {
        g_cap.run = 1;
        llama_context * ctx = make(mac_cells);
        if (llama_kv_remote_attach(ctx, remote.c_str()) != 0) { fprintf(stderr, "cannot attach the phone at %s\n", remote.c_str()); return 1; }
        run(ctx, tb, Lb, pb_s, mb);
        printf("phone holds %d positions at the end, the Mac %d cells\n", llama_kv_remote_n(ctx), llama_n_ctx(ctx));
        llama_free(ctx);
    }
    int div = (int) ta.size();
    for (size_t i = 0; i < ta.size(); i++) if (ta[i] != tb[i]) { div = (int) i; break; }
    if (getenv("PKV_CAPTURE")) {
        for (int st = 0; st <= 3; st++) {
            printf("  attention outputs, step %d (0 = restore decode):", st);
            for (int il = 0; il < 64; il++) {
                const std::string k = "/" + std::to_string(st) + "/kqv_out-" + std::to_string(il);
                auto a0 = g_cap.t.find("0" + k), a1 = g_cap.t.find("1" + k);
                if (a0 == g_cap.t.end() || a1 == g_cap.t.end()) continue;
                double md = 0, mr = 1e-30;
                for (size_t q = 0; q < a0->second.size() && q < a1->second.size(); q++) {
                    md = std::max(md, (double) std::fabs(a0->second[q] - a1->second[q])); mr = std::max(mr, (double) std::fabs(a0->second[q]));
                }
                printf(" L%d %.1e", il, md/mr);
            }
            printf("\n");
        }
    }
    if (first_logits.size() == 2) {
        cmp c0; printf("  first decode after the restore (the fixed token):\n"); c0.add(first_logits[0].data(), first_logits[1].data(), nv);
        c0.print("restore decode");
    }
    cmp c;
    for (int i = 0; i < std::min(div, n_gen); i++) c.add(&La[(size_t) i*nv], &Lb[(size_t) i*nv], nv);
    printf("\ngrow: prompt %d tokens, Mac cells %d (phone takes the rest), %d generated (%s KV)\n", n_prompt, mac_cells, n_gen, kv.c_str());
    if (restore_path.empty()) printf("  prefill: Mac-only %.1f s (%.0f tok/s), with phone %.1f s (%.0f tok/s)\n", pa_s, n_prompt/pa_s, pb_s, n_prompt/pb_s);
    else printf("  restore + first token: Mac-only %.1f s, with phone %.1f s\n", pa_s, pb_s);
    printf("  decode ms/token: Mac-only %.1f, with phone %.1f\n", ma, mb);
    printf("  greedy tokens identical: %s (first difference at token %d of %zu)\n", div == (int) ta.size() ? "YES" : "NO", div, ta.size());
    c.print("with phone vs Mac-only");
    bool rounds_ok = true;
    if (n_rounds > 0) {
        cmp cr;
        for (int i = 0; i < n_rounds*8; i++) cr.add(&round_L[0][(size_t) i*nv], &round_L[1][(size_t) i*nv], nv);
        printf("  verify rounds (8 tokens each, %d rounds) ms/round: Mac-only %.1f, with phone %.1f\n", n_rounds, ms_round[0], ms_round[1]);
        cr.print("rounds: phone vs Mac-only");
        rounds_ok = cr.same_top1 >= cr.n - cr.n/32;   // near-ties may flip (the partials are fp16): allow ~3%
    }
    const bool pass = div == (int) ta.size() && rounds_ok;
    printf("GATE: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 3;
}

int main(int argc, char ** argv) {
    std::string model_path, text_path, remote = "127.0.0.1:50062", kv = "q8_0", restore_path;
    int n_prompt = 4096, n_gen = 48, n_ctx = 8192, grow = 0, n_rounds = 0;
    std::vector<int> evicts = { 2048 };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "-m") model_path = next();
        else if (a == "-f") text_path = next();
        else if (a == "--prompt") n_prompt = atoi(next().c_str());
        else if (a == "--evict") { evicts.clear(); std::stringstream es(next()); std::string x; while (std::getline(es, x, ',')) evicts.push_back(atoi(x.c_str())); }
        else if (a == "--gen") n_gen = atoi(next().c_str());
        else if (a == "--ctx") n_ctx = atoi(next().c_str());
        else if (a == "--remote") remote = next();
        else if (a == "--kv") kv = next();
        else if (a == "--grow") grow = atoi(next().c_str());
        else if (a == "--rounds") n_rounds = atoi(next().c_str());
        else if (a == "--restore") restore_path = next();
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || text_path.empty()) {
        fprintf(stderr, "usage: %s -m MODEL -f TEXT [--prompt N] [--evict N] [--gen N] [--ctx N] [--remote host:port] [--kv q8_0|q4_0|f16]\n", argv[0]);
        return 2;
    }
    n_gen = n_gen/8*8;

    // same attention kernel for 1-token and 8-token steps in every run (the phone runs force it anyway)
    setenv("GGML_METAL_FA_GQA", "1", 0);
    setenv("GGML_METAL_FA_GQA_MIN_NE01", "1", 0);

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);

    std::ifstream f(text_path);
    std::stringstream ss; ss << f.rdbuf();
    const std::string text = ss.str();
    std::vector<llama_token> toks(text.size() + 16);
    const int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
    if (nt < n_prompt) { fprintf(stderr, "text has only %d tokens, need %d\n", nt, n_prompt); return 1; }
    toks.resize(n_prompt);

    if (grow > 0) {
        const int rc = run_grow(model, toks, n_gen, grow, remote, kv, restore_path, n_rounds);
        llama_model_free(model);
        llama_backend_free();
        return rc;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = kv == "f16" ? GGML_TYPE_F16 : kv == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create context\n"); return 1; }

    if (llama_kv_remote_attach(ctx, remote.c_str()) != 0) {
        fprintf(stderr, "cannot attach the phone at %s (start it first: pa-tool serve)\n", remote.c_str());
        return 1;
    }

    // prefill (Mac only), keep the last logits
    auto t0 = std::chrono::steady_clock::now();
    llama_batch b = llama_batch_init(512, 0, 1);
    for (int p = 0; p < n_prompt; p += 512) {
        const int n = std::min(512, n_prompt - p);
        b.n_tokens = n;
        for (int j = 0; j < n; j++) {
            b.token[j] = toks[p + j]; b.pos[j] = p + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = p + j == n_prompt - 1;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill decode failed at %d\n", p); return 1; }
    }
    const double t_pf = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const llama_token g0 = argmax(llama_get_logits_ith(ctx, -1), nv);
    printf("prefill %d tokens in %.1f s (%.0f tok/s)\n", n_prompt, t_pf, n_prompt/t_pf);

    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
    state.resize(llama_state_seq_get_data(ctx, state.data(), state.size(), 0));
    auto restore = [&]() {
        if (llama_state_seq_set_data(ctx, state.data(), state.size(), 0) == 0) { fprintf(stderr, "restore failed\n"); exit(1); }
    };

    // greedy, one token per step: returns the tokens g1..gN and the logits that chose them
    auto run_ar = [&](std::vector<llama_token> & out, std::vector<float> & L, double & ms_per_tok) {
        out.clear(); L.assign((size_t) n_gen*nv, 0);
        llama_token t = g0;
        auto a = std::chrono::steady_clock::now();
        for (int i = 0; i < n_gen; i++) {
            b.n_tokens = 1;
            b.token[0] = t; b.pos[0] = n_prompt + i; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
            const float * lg = llama_get_logits_ith(ctx, 0);
            memcpy(&L[(size_t) i*nv], lg, sizeof(float)*nv);
            t = argmax(lg, nv);
            out.push_back(t);
        }
        ms_per_tok = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count()/n_gen;
    };
    // the given tokens (g0 then in[0..N-2]), 8 per step
    auto run_chunk8 = [&](const std::vector<llama_token> & in, std::vector<float> & L) {
        L.assign((size_t) n_gen*nv, 0);
        for (int i = 0; i < n_gen; i += 8) {
            b.n_tokens = 8;
            for (int j = 0; j < 8; j++) {
                b.token[j] = i + j == 0 ? g0 : in[i + j - 1]; b.pos[j] = n_prompt + i + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1;
            }
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
            for (int j = 0; j < 8; j++) memcpy(&L[(size_t) (i + j)*nv], llama_get_logits_ith(ctx, j), sizeof(float)*nv);
        }
    };
    auto evict = [&](int n_evict) {
        const int moved = llama_kv_remote_evict(ctx, n_evict);
        if (moved != n_evict/64*64) { fprintf(stderr, "evict moved %d of %d\n", moved, n_evict); exit(1); }
    };
    auto piece = [&](const std::vector<llama_token> & v) {
        std::string s; char buf[256];
        for (auto t : v) { int n = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true); if (n > 0) s.append(buf, n); }
        for (auto & c : s) if (c == '\n') c = ' ';
        return s;
    };

    std::vector<llama_token> tok_mac, tok_phone;
    std::vector<float> L_mac_ar, L_mac_c8, L_ph_ar, L_ph_c8;
    double ms_mac = 0, ms_ph = 0;

    run_ar(tok_mac, L_mac_ar, ms_mac);
    restore(); run_chunk8(tok_mac, L_mac_c8);
    cmp noise;
    for (int i = 0; i < n_gen; i++) noise.add(&L_mac_ar[(size_t) i*nv], &L_mac_c8[(size_t) i*nv], nv);
    printf("\nprompt %d tokens, %d generated (%s KV)\n", n_prompt, n_gen, kv.c_str());
    printf("mac-only  : %s\n", piece(tok_mac).c_str());
    noise.print("noise floor (ar vs chunk8)");

    bool pass = true;
    for (int n_evict : evicts) {
        restore(); evict(n_evict); run_ar(tok_phone, L_ph_ar, ms_ph);
        restore(); evict(n_evict); run_chunk8(tok_mac, L_ph_c8);
        int div = n_gen;
        for (int i = 0; i < n_gen; i++) if (tok_mac[i] != tok_phone[i]) { div = i; break; }
        cmp ar, c8;
        for (int i = 0; i < div; i++)   ar.add(&L_mac_ar[(size_t) i*nv], &L_ph_ar[(size_t) i*nv], nv);
        for (int i = 0; i < n_gen; i++) c8.add(&L_mac_c8[(size_t) i*nv], &L_ph_c8[(size_t) i*nv], nv);
        printf("\n%d of %d positions on the phone:\n", n_evict/64*64, n_prompt);
        if (div != n_gen) printf("with phone: %s\n", piece(tok_phone).c_str());
        printf("  greedy tokens identical: %s (first difference at token %d of %d)\n", div == n_gen ? "YES" : "NO", div, n_gen);
        ar.print("phone ar vs mac ar");
        c8.print("phone chunk8 vs mac chunk8");
        printf("  decode ms/token (1-token steps, informational): mac %.1f, phone %.1f\n", ms_mac, ms_ph);
        // Gate: identical greedy tokens and the same top-1 at every verify-shaped position. The logit gap is reported,
        // not gated: the Mac-only runs are not exact either. The GPU kernel keeps P and V in half precision (~2.6e-3 max
        // relative error vs fp64, fa-bias-test) while the phone's SME kernel is ~4e-4 (GGML_METAL_REMOTE_CHECK=1), so
        // moving keys to the phone shifts the logits by as much as the Mac's own SME co-attention does (KL ~5e-4 with
        // half the keys at 4k); against SME co-attention at the same split the gap is KL ~1e-5.
        pass = pass && div == n_gen && c8.same_top1 == c8.n;
    }
    printf("GATE: %s\n", pass ? "PASS" : "FAIL");

    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 3;
}
