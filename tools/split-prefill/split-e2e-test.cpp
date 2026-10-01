// split-e2e-test: end-to-end gate for split prefill INSIDE llama_decode (src/llama-split.cpp, env LLAMA_SPLIT_TAIL).
//
//   LLAMA_SPLIT_L=20 llama-split-e2e-test -m model.gguf -f text.txt --tail HOST:PORT [--min 1024 --view 2048 --ub 512
//                                          --taps 21,27 -ngl 99 --kv q8_0 --replay 8]
//
// Two contexts on one model: R (plain) and S (created with LLAMA_SPLIT_TAIL set, so its big prompt views go to the
// worker). Both run the same server-like conversation through plain llama_decode calls, in views of --view tokens:
//   1. prompt A (3 views)                 5. long turn C (2 views; the second asks for logits -> its last ubatch is local)
//   2. greedy 16                          6. greedy 16
//   3. short follow-up (below --min)      7. edit: checkpoint restore + seq_rm back to the end of step 4, long turn D, greedy 16
//   4. greedy 16
// Pass = identical greedy tokens at every step. Also reports the max |diff| of the tap layer rows (DFlash reads these)
// between R and S on the split views, and the split counters from the log (LLAMA_SPLIT_VERBOSE=1 prints them).
#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int argmax(const float * v, int n) { return (int) (std::max_element(v, v + n) - v); }

int main(int argc, char ** argv) {
    std::string model_path, text_path, tail, kv;
    int min_tok = 1024, view = 2048, ub = 512, ngl = 99, replay = 0;
    bool quick = false;   // --quick: one 2-view prompt + greedy 16 (timing + a token check in ~1 min on the 27B)
    std::vector<int> taps;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (s == "-m") model_path = nx();
        else if (s == "-f") text_path = nx();
        else if (s == "--tail") tail = nx();
        else if (s == "--min") min_tok = std::stoi(nx());
        else if (s == "--view") view = std::stoi(nx());
        else if (s == "--ub") ub = std::stoi(nx());
        else if (s == "-ngl") ngl = std::stoi(nx());
        else if (s == "--kv") kv = nx();
        else if (s == "--replay") replay = std::stoi(nx());
        else if (s == "--quick") quick = true;
        else if (s == "--taps") { std::stringstream t(nx()); std::string x; while (std::getline(t, x, ',')) taps.push_back(std::stoi(x)); }
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (model_path.empty() || text_path.empty() || tail.empty()) { fprintf(stderr, "need -m -f --tail\n"); return 2; }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab), n_embd = llama_model_n_embd(model);

    std::ifstream fs(text_path); std::stringstream ss; ss << fs.rdbuf();
    const std::string text = ss.str();
    std::vector<llama_token> T(text.size() + 16);
    T.resize(llama_tokenize(vocab, text.c_str(), (int) text.size(), T.data(), (int) T.size(), true, false));
    // segments of the conversation, cut from the text
    const int nA = 3 * view - 300, nB = 200, nC = 2 * view - 100, nD = view + view / 2;
    if ((int) T.size() < nA + nB + nC + nD + 64) { fprintf(stderr, "text too short (%zu tokens)\n", T.size()); return 1; }
    size_t cur = 0;
    auto take = [&](int n) { std::vector<llama_token> v(T.begin() + cur, T.begin() + cur + n); cur += n; return v; };
    const auto A = take(nA), B = take(nB), C = take(nC), Dd = take(nD);

    auto cp = llama_context_default_params();
    cp.n_ctx = nA + nB + nC + nD + 256; cp.n_batch = view; cp.n_ubatch = ub; cp.n_seq_max = 1; cp.no_perf = true;
    cp.n_rs_replay = replay;
    if (!kv.empty()) {
        const ggml_type t = kv == "q8_0" ? GGML_TYPE_Q8_0 : kv == "q4_0" ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
        cp.type_k = t; cp.type_v = t; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    llama_context * R = llama_init_from_model(model, cp);
    setenv("LLAMA_SPLIT_TAIL", tail.c_str(), 1);
    setenv("LLAMA_SPLIT_MIN", std::to_string(min_tok).c_str(), 1);
    llama_context * S = llama_init_from_model(model, cp);
    unsetenv("LLAMA_SPLIT_TAIL");
    if (!R || !S) return 1;
    for (int t : taps) { llama_set_embeddings_layer_inp(R, t, true); llama_set_embeddings_layer_inp(S, t, true); }

    llama_pos pos = 0;
    double tap_diff = 0, tap_ref = 0;
    double ms_r = 0, ms_s = 0; long tok_timed = 0;   // prompt views >= --min: Mac-only vs split wall time
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    // prompt tokens in views; logits on the very last token; compare tap rows view by view
    auto prompt = [&](const std::vector<llama_token> & toks) {
        for (size_t v0 = 0; v0 < toks.size(); v0 += view) {
            const int n = (int) std::min<size_t>(view, toks.size() - v0);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; i++) { b.token[i] = toks[v0 + i]; b.pos[i] = pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
            b.logits[n - 1] = v0 + n == toks.size();
            b.n_tokens = n;
            const auto t0 = now();
            if (llama_decode(R, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
            llama_synchronize(R);
            const auto t1 = now();
            if (llama_decode(S, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
            llama_synchronize(S);
            const auto t2 = now();
            if (n >= min_tok) { ms_r += ms(t0, t1); ms_s += ms(t1, t2); tok_timed += n; }
            for (int t : taps) {
                const float * a = llama_get_embeddings_layer_inp(R, t), * c = llama_get_embeddings_layer_inp(S, t);
                for (size_t i = 0; i < (size_t) n * n_embd; i++) { tap_diff = std::max(tap_diff, (double) std::fabs(a[i] - c[i])); tap_ref = std::max(tap_ref, (double) std::fabs(a[i])); }
            }
            llama_batch_free(b);
            pos += n;
        }
    };
    int step = 0, fails = 0;
    auto greedy = [&](int G) {
        step++;
        llama_token ta = argmax(llama_get_logits_ith(R, -1), n_vocab), tb = argmax(llama_get_logits_ith(S, -1), n_vocab);
        int div = -1;
        for (int i = 0; i < G; i++) {
            if (ta != tb && div < 0) div = i;
            llama_batch ba = llama_batch_get_one(&ta, 1); ba.pos = nullptr;
            llama_batch bb = llama_batch_get_one(&tb, 1);
            if (llama_decode(R, ba) || llama_decode(S, bb)) { fprintf(stderr, "decode failed\n"); exit(1); }
            ta = argmax(llama_get_logits_ith(R, -1), n_vocab); tb = argmax(llama_get_logits_ith(S, -1), n_vocab);
        }
        pos += G;
        printf("step %d: greedy %d at pos %d: %s (first diverge %d)\n", step, G, pos, div < 0 ? "IDENTICAL" : "DIFFERENT", div);
        fails += div >= 0;
    };
    if (quick) {
        prompt(std::vector<llama_token>(A.begin(), A.begin() + 2 * view)); greedy(16);
        printf("tap rows: max|R-S| %.4g (max|R| %.4g)\n", tap_diff, tap_ref);
        printf("views >= %d tokens: %ld tokens, Mac-only %.0f ms (%.1f tok/s), split %.0f ms (%.1f tok/s): %.2fx\n", min_tok, tok_timed,
               ms_r, tok_timed / ms_r * 1e3, ms_s, tok_timed / ms_s * 1e3, ms_r / ms_s);
        printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    }
    prompt(A);  greedy(16);
    prompt(B);  greedy(16);
    // checkpoint at p_edit the way the server makes one: the partial (recurrent) state
    const llama_pos p_edit = pos;
    auto ckpt = [&](llama_context * c) {
        std::vector<uint8_t> v(llama_state_seq_get_size_ext(c, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        v.resize(llama_state_seq_get_data_ext(c, v.data(), v.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        return v;
    };
    const auto ck_r = ckpt(R), ck_s = ckpt(S);
    prompt(C);  greedy(16);
    // edit: the conversation is rewound to p_edit (checkpoint restore, then the KV rows go) and continues differently
    for (auto pr : { std::make_pair(R, &ck_r), std::make_pair(S, &ck_s) }) {
        if (llama_state_seq_set_data_ext(pr.first, pr.second->data(), pr.second->size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != pr.second->size() ||
            !llama_memory_seq_rm(llama_get_memory(pr.first), 0, p_edit, -1)) { fprintf(stderr, "rewind failed\n"); return 1; }
    }
    pos = p_edit;
    prompt(Dd); greedy(16);
    printf("tap rows: max|R-S| %.4g (max|R| %.4g)\n", tap_diff, tap_ref);
    printf("views >= %d tokens: %ld tokens, Mac-only %.0f ms (%.1f tok/s), split %.0f ms (%.1f tok/s): %.2fx\n", min_tok, tok_timed,
           ms_r, tok_timed / ms_r * 1e3, ms_s, tok_timed / ms_s * 1e3, ms_r / ms_s);
    printf("%s\n", fails ? "FAIL" : "PASS");
    llama_free(R); llama_free(S); llama_model_free(model); llama_backend_free();
    return fails ? 1 : 0;
}
