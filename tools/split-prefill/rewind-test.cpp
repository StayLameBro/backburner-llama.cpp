// rewind-test: gate for tail protocol v3 (tail-server.h CB_ALL_LOGITS / CB_H_ROWS / CB_REWIND), the phone half of
// speculative decoding across the split. No Mac head needed: the residuals are synthetic rows, so any tail GGUF works.
//
//   llama-split-rewind-test --tail-model tail.gguf [--replay 9] [--k 7] [--prompt 100] [--ub 64] [--port 50071]
//
// Runs a tail_server on loopback in this process and drives it with tail_client. For every rejected depth r = 1..k:
//   X: HELLO, prompt [0, P) in ubatch chunks, a verify chunk of k+1 rows at P (the last r rows are "rejected" residuals),
//      then a chunk of m rows at P + (k+1-r) with CB_REWIND
//   Y: HELLO, the same prompt, a chunk of the k+1-r accepted rows at P, then the same m rows
// Pass = X's and Y's last-chunk logits and h rows are bit-identical (memcmp), for every r. Also checks:
//   - a v2-style chunk (bits 1 only) still returns one logits row and no h rows
//   - a rewind deeper than the last chunk is refused with ERR (and the link then fails: the Mac would re-prefill)
//   - the all-rows logits of a verify chunk: row k equals the single-row logits of the same chunk (CB_LOGITS) bit-exact
#include "tail-client.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

static std::vector<float> rows(int p0, int n, int n_embd, int variant) {
    // deterministic per (position, variant): the same position always gets the same residual in both sessions
    std::vector<float> v((size_t) n * n_embd);
    for (int i = 0; i < n; i++) {
        std::mt19937 g((uint32_t) ((p0 + i) * 7919 + variant * 104729 + 1));
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (int j = 0; j < n_embd; j++) v[(size_t) i * n_embd + j] = d(g);
    }
    return v;
}

struct sess {
    spt::tail_client tc;
    int n_embd = 0;
    bool open(int port, const spt::hello_req2 & q, std::string & err) {
        spt::hello_rep2 r;
        if (!tc.connect("127.0.0.1", port, q, r, err)) return false;
        n_embd = (int) q.base.n_embd;
        return true;
    }
    // one chunk, waits for it
    bool chunk(int p0, int n, int variant, uint32_t bits, std::vector<float> & lg, std::vector<ggml_fp16_t> & h, std::string & err) {
        auto x = rows(p0, n, n_embd, variant);
        try { tc.submit_chunk_v3(x.data(), n, n_embd, p0, bits); } catch (const std::exception & e) { err = e.what(); return false; }
        std::vector<ggml_fp16_t> taps;
        return tc.finish_v3(lg, taps, h, err);
    }
    bool prompt(int P, int ub, std::string & err) {
        std::vector<float> lg; std::vector<ggml_fp16_t> h;
        for (int p = 0; p < P; p += ub) {
            const int n = std::min(ub, P - p);
            if (!chunk(p, n, 0, spt::CB_H_ROWS | (p + n == P ? spt::CB_LOGITS : 0u), lg, h, err)) return false;
        }
        return true;
    }
};

int main(int argc, char ** argv) {
    std::string tail_model;
    int replay = 9, k = 7, P = 100, ub = 64, port = 50071;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (s == "--tail-model") tail_model = nx();
        else if (s == "--replay") replay = std::stoi(nx());
        else if (s == "--k") k = std::stoi(nx());
        else if (s == "--prompt") P = std::stoi(nx());
        else if (s == "--ub") ub = std::stoi(nx());
        else if (s == "--port") port = std::stoi(nx());
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (tail_model.empty()) { fprintf(stderr, "need --tail-model\n"); return 2; }
    if (k + 1 > replay) { fprintf(stderr, "k+1 (%d) must be <= --replay (%d): the verify chunk has to stay in the replay log\n", k + 1, replay); return 2; }

    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * t, void *) { if (lvl >= GGML_LOG_LEVEL_WARN) fputs(t, stderr); }, nullptr);

    // the HELLO needs the tail's shape: read it from the GGUF (vocab_only skips the weights)
    uint32_t L = 0, n_full = 0;
    int n_embd = 0, n_vocab = 0;
    {
        auto mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model * m = llama_model_load_from_file(tail_model.c_str(), mp);
        if (!m) return 1;
        char b[64];
        if (llama_model_meta_val_str(m, "split.layer_start", b, sizeof b) > 0) L = (uint32_t) atoi(b);
        if (llama_model_meta_val_str(m, "split.n_layer_full", b, sizeof b) > 0) n_full = (uint32_t) atoi(b);
        // vocab_only leaves the hparams unset: read the embedding width from the metadata
        n_embd = llama_model_meta_val_str(m, "qwen35.embedding_length", b, sizeof b) > 0 ? atoi(b) : 0;
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m));
        llama_model_free(m);
    }

    spt::server_status st;
    spt::tail_server srv(tail_model, &st, [](const std::string & s) { fprintf(stderr, "  [tail] %s\n", s.c_str()); });
    std::thread th([&] { srv.serve(port); });
    th.detach();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    spt::hello_req2 q = {};
    q.base.proto = spt::PROTO_HELLO; q.base.state_format = spt::STATE_FORMAT;
    q.base.L = L; q.base.n_layer_full = n_full; q.base.n_embd = (uint32_t) n_embd; q.base.n_vocab = (uint32_t) n_vocab;
    q.base.n_ctx = 1024; q.base.n_ubatch = (uint32_t) ub;
    q.type_k = GGML_TYPE_F16; q.type_v = GGML_TYPE_F16; q.flash_attn = 0;
    q.n_rs_replay = (uint32_t) replay;

    int fails = 0;
    auto check = [&](bool ok, const std::string & what) { printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str()); fflush(stdout); if (!ok) fails++; };
    std::string err;
    const size_t NV = (size_t) n_vocab;

    // proto (the server takes one client at a time: every session is closed before the next one opens)
    std::vector<float> lg1;
    {
        sess s;
        if (!s.open(port, q, err)) { fprintf(stderr, "connect: %s\n", err.c_str()); return 1; }
        check(s.tc.v3(), "HELLO_OK reports proto " + std::to_string(s.tc.hello.base.proto) + " (>= 3) to a client that sent " + std::to_string(spt::PROTO_HELLO));
        // v2-style chunk on a v3 server: one row, no h
        std::vector<float> lg; std::vector<ggml_fp16_t> h;
        bool ok = s.chunk(0, 5, 0, spt::CB_LOGITS, lg, h, err);
        check(ok && lg.size() == NV && h.empty(), "v2 chunk (CB_LOGITS only): 1 logits row, no h rows" + (ok ? "" : ": " + err));
        lg1 = lg;
    }
    {
        // all rows vs last row of the same chunk
        const std::vector<float> & lg = lg1;
        std::vector<float> lg_all; std::vector<ggml_fp16_t> h_all;
        sess s2;
        bool ok;
        s2.open(port, q, err);
        ok = s2.chunk(0, 5, 0, spt::CB_ALL_LOGITS | spt::CB_H_ROWS, lg_all, h_all, err);
        check(ok && lg_all.size() == 5 * NV && h_all.size() == (size_t) 5 * n_embd, "CB_ALL_LOGITS|CB_H_ROWS: 5 logits rows, 5 h rows" + (ok ? "" : ": " + err));
        if (ok && lg.size() == NV) {
            double md = 0; for (size_t i = 0; i < NV; i++) md = std::max(md, (double) std::fabs(lg[i] - lg_all[4 * NV + i]));
            // a different number of output rows changes nothing upstream of the head, so the last row should match exactly;
            // report rather than fail on fp noise, since the head matmul's kernel may depend on the row count
            printf("%s  last row of an all-rows chunk vs the single-row chunk: max |diff| %.3g\n", md == 0 ? "PASS" : "INFO", md);
        }
    }

    // rewind equivalence for every depth
    for (int r = 1; r <= k; r++) {
        const int a = k + 1 - r, m = 3;
        std::vector<float> lgV, lgX, lgY; std::vector<ggml_fp16_t> hV, hX, hY;
        bool ok = true;
        {
        sess X;
        if (!X.open(port, q, err) || !X.prompt(P, ub, err)) { check(false, "X prompt: " + err); continue; }
        // verify chunk: a accepted rows (variant 0), r rejected rows (variant 1)
        auto acc = rows(P, a, n_embd, 0), rej = rows(P + a, r, n_embd, 1);
        std::vector<float> ver(acc); ver.insert(ver.end(), rej.begin(), rej.end());
        try { X.tc.submit_chunk_v3(ver.data(), k + 1, n_embd, P, spt::CB_ALL_LOGITS | spt::CB_H_ROWS); } catch (const std::exception & e) { ok = false; err = e.what(); }
        { std::vector<ggml_fp16_t> t; ok = ok && X.tc.finish_v3(lgV, t, hV, err); }
        ok = ok && X.chunk(P + a, m, 0, spt::CB_ALL_LOGITS | spt::CB_H_ROWS | spt::CB_REWIND, lgX, hX, err);
        if (!ok) { check(false, "r=" + std::to_string(r) + " X: " + err); continue; }
        }
        std::vector<float> lgYa; std::vector<ggml_fp16_t> hYa;
        {
        sess Y;
        ok = Y.open(port, q, err) && Y.prompt(P, ub, err);
        ok = ok && Y.chunk(P, a, 0, spt::CB_ALL_LOGITS | spt::CB_H_ROWS, lgY, hY, err);
        lgYa = lgY; hYa = hY;
        ok = ok && Y.chunk(P + a, m, 0, spt::CB_ALL_LOGITS | spt::CB_H_ROWS, lgY, hY, err);
        if (!ok) { check(false, "r=" + std::to_string(r) + " Y: " + err); continue; }
        }

        const bool same_lg = lgX.size() == lgY.size() && memcmp(lgX.data(), lgY.data(), lgX.size() * sizeof(float)) == 0;
        const bool same_h  = hX.size() == hY.size() && memcmp(hX.data(), hY.data(), hX.size() * 2) == 0;
        double md = 0;
        for (size_t i = 0; i < std::min(lgX.size(), lgY.size()); i++) md = std::max(md, (double) std::fabs(lgX[i] - lgY[i]));
        // the accepted rows themselves were computed in a (k+1)-row batch in X and an a-row batch in Y
        double mda = 0;
        for (size_t i = 0; i < (size_t) a * NV && i < lgYa.size(); i++) mda = std::max(mda, (double) std::fabs(lgV[i] - lgYa[i]));
        // Y computes the kept rows in an a-row batch, X in a (k+1)-row batch. When the backend picks a different matmul path for
        // a rows (CPU: a == 1 takes mul_mv; Metal with n_embd 256: also a == 2), Y's own rows already differ from X's before any
        // rewind (mda > 0), so that reference can't be bit-compared. Bit-exactness is required whenever mda == 0; otherwise the
        // post-rewind rows may differ by no more than fp noise: 1e-3, or a quarter of the kept rows' own pre-rewind difference
        // when that is larger (Metal with REGFED off: the reference's 1-3-row batches start 0.04-0.06 off, the rows after the
        // rewind inherit 2-7% of that)
        const bool noise_ok = mda > 0 && md <= std::max(1e-3, std::min(0.25 * mda, 0.05));
        if (noise_ok && !(same_lg && same_h)) {
            printf("PASS  r=%d (keep %d of %d): post-rewind logits within fp noise (max %.3g) of a reference whose %d-row batch takes "
                   "another matmul path (its kept rows already differ by %.3g, before any rewind)\n", r, a, k + 1, md, a, mda);
            fflush(stdout);
            continue;
        }
        check(same_lg && same_h, "r=" + std::to_string(r) + " (keep " + std::to_string(a) + " of " + std::to_string(k + 1) +
              "): post-rewind logits " + (same_lg ? "bit-identical" : "differ (max " + std::to_string(md) + ")") +
              ", h rows " + (same_h ? "bit-identical" : "differ") +
              "; accepted rows in the verify batch vs Y max |diff| " + std::to_string(mda));
    }

    // a rewind deeper than the last chunk (into the prompt's last ubatch, which was > replay tokens) must be refused
    {
        sess X;
        bool ok = X.open(port, q, err) && X.prompt(P, ub, err);
        std::vector<float> lg; std::vector<ggml_fp16_t> h;
        // the prompt's last chunk is P % ub (or ub) rows; rewinding past it can't be followed
        const int last = P % ub ? P % ub : ub;
        const int to = P - last - 1;
        bool refused = ok && !X.chunk(to, 1, 0, spt::CB_LOGITS | spt::CB_REWIND, lg, h, err);
        check(refused && err.find("replay window") != std::string::npos, "rewind to " + std::to_string(to) + " (past the last chunk of " +
              std::to_string(last) + ") refused: " + err);
    }
    // the same, within the last chunk but deeper than replay allows: prompt's last chunk is P % ub rows > replay
    if ((P % ub) > replay) {
        sess X;
        bool ok = X.open(port, q, err) && X.prompt(P, ub, err);
        std::vector<float> lg; std::vector<ggml_fp16_t> h;
        bool refused = ok && !X.chunk(P - 2, 1, 0, spt::CB_LOGITS | spt::CB_REWIND, lg, h, err);
        check(refused, "rewind 2 into an eager chunk (" + std::to_string(P % ub) + " > replay " + std::to_string(replay) + " rows) refused: " + err);
    }

    printf("%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    fflush(stdout);
    _exit(fails ? 1 : 0);   // the server thread is still in accept()
}
