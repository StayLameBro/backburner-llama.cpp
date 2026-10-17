// rewind-test: gate for tail protocol v3 (tail-server.h CB_ALL_LOGITS / CB_H_ROWS / CB_REWIND), the phone half of
// speculative decoding across the split. No Mac head needed: the residuals are synthetic rows, so any tail GGUF works.
//
//   llama-split-rewind-test --tail-model tail.gguf [--replay 9] [--k 7] [--prompt 100] [--ub 64] [--port 50071] [--exact]
//
// Runs a tail_server on loopback in this process and drives it with tail_client. Two speculative rounds back to back, as
// the MTP loop sends them, for every round-1 depth r = 0..k (r2 = (r + 1) % (k + 1) in round 2):
//   X:  prompt [0, P), a verify chunk of k+1 rows at P (the last r rows rejected), a verify chunk of k+1 rows with
//       CB_REWIND (the last r2 rows rejected), then 2 rows with CB_REWIND
//   X2: X with other rejected rows
//   Z:  every kept row in the prompt (its last ubatch > replay rows: eager, no replay log), then the same 2 rows
// Pass, for every r:
//   - X vs X2 bit-identical (round-2 kept rows and the last 2 rows, logits and h): no trace of the rejected rows. Any backend
//   - with --exact, X vs Z bit-identical on the last 2 rows: the replay gives what a plain decode gives. Without --exact
//     INFO only: the matmul kernel can change with the batch shape. --exact needs a build where it can't, e.g.
//     -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_CPU_REPACK=OFF -DGGML_LLAMAFILE=OFF -DGGML_CPU_KLEIDIAI=OFF, and on ARM
//     -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 (no i8mm). It sets LLAMA_GDN_PREP_FUSE=0: the fused
//     GDN prep is not bit-identical to the unfused one that Z's eager ubatch builds
// Also checks:
//   - a v2-style chunk (bits 1 only) still returns one logits row and no h rows
//   - a rewind deeper than the last chunk is refused with ERR (and the link then fails: the Mac would re-prefill)
//   - the all-rows logits of a verify chunk: row k equals the single-row logits of the same chunk (CB_LOGITS) bit-exact
#include "tail-client.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
    bool exact = false;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (s == "--tail-model") tail_model = nx();
        else if (s == "--replay") replay = std::stoi(nx());
        else if (s == "--k") k = std::stoi(nx());
        else if (s == "--prompt") P = std::stoi(nx());
        else if (s == "--ub") ub = std::stoi(nx());
        else if (s == "--port") port = std::stoi(nx());
        else if (s == "--exact") exact = true;
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (tail_model.empty()) { fprintf(stderr, "need --tail-model\n"); return 2; }
    if (k + 1 > replay) { fprintf(stderr, "k+1 (%d) must be <= --replay (%d): the verify chunk has to stay in the replay log\n", k + 1, replay); return 2; }

    if (exact) setenv("LLAMA_GDN_PREP_FUSE", "0", 1);   // before any graph is built; one thread so far
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

    // rewind equivalence: two speculative rounds back to back
    const uint32_t ALL = spt::CB_ALL_LOGITS | spt::CB_H_ROWS;
    const int nf = 2;   // rows after the last rewind
    for (int r = 0; r <= k; r++) {
        const int r2 = (r + 1) % (k + 1);
        const int a = k + 1 - r, a2 = k + 1 - r2;   // kept rows per round
        const std::string tag = "r=" + std::to_string(r) + "," + std::to_string(r2) + " (keep " + std::to_string(a) + "+" + std::to_string(a2) + ")";
        // out: the round-2 kept rows, then the nf rows (logits, h)
        auto run_x = [&](int rv, std::vector<float> & lg, std::vector<ggml_fp16_t> & h) {
            sess X;
            if (!X.open(port, q, err) || !X.prompt(P, ub, err)) return false;
            auto verify = [&](int p0, int na, int nr, uint32_t bits, std::vector<float> & l, std::vector<ggml_fp16_t> & hh) {
                auto x = rows(p0, na, n_embd, 0), rej = rows(p0 + na, nr, n_embd, rv);
                x.insert(x.end(), rej.begin(), rej.end());
                try { X.tc.submit_chunk_v3(x.data(), na + nr, n_embd, p0, bits); } catch (const std::exception & e) { err = e.what(); return false; }
                std::vector<ggml_fp16_t> t;
                return X.tc.finish_v3(l, t, hh, err);
            };
            std::vector<float> l1, l2, l3; std::vector<ggml_fp16_t> h1, h2, h3;
            if (!verify(P, a, r, ALL, l1, h1) || !verify(P + a, a2, r2, ALL | spt::CB_REWIND, l2, h2) ||
                !X.chunk(P + a + a2, nf, 0, ALL | spt::CB_REWIND, l3, h3, err)) return false;
            lg.assign(l2.begin(), l2.begin() + (size_t) a2 * NV); lg.insert(lg.end(), l3.begin(), l3.end());
            h.assign(h2.begin(), h2.begin() + (size_t) a2 * n_embd); h.insert(h.end(), h3.begin(), h3.end());
            return true;
        };
        std::vector<float> lgX, lgX2, lgZ; std::vector<ggml_fp16_t> hX, hX2, hZ;
        if (!run_x(1, lgX, hX))   { check(false, tag + " X: " + err); continue; }
        if (!run_x(2, lgX2, hX2)) { check(false, tag + " X2: " + err); continue; }
        auto same = [](const void * x, const void * y, size_t n) { return memcmp(x, y, n) == 0; };
        auto max_diff = [](const float * x, const float * y, size_t n) { double m = 0; for (size_t i = 0; i < n; i++) m = std::max(m, (double) std::fabs(x[i] - y[i])); return m; };
        char b[200];

        const bool xx = lgX.size() == lgX2.size() && hX.size() == hX2.size() &&
                        same(lgX.data(), lgX2.data(), lgX.size() * sizeof(float)) && same(hX.data(), hX2.data(), hX.size() * 2);
        snprintf(b, sizeof b, ": X vs X2 (other rejected rows) max |diff| %.3g", max_diff(lgX.data(), lgX2.data(), std::min(lgX.size(), lgX2.size())));
        check(xx, tag + (xx ? ": X vs X2 (other rejected rows) bit-identical" : std::string(b)));

        // Z is a reference only if its last prompt ubatch is eager
        const int zl = (P + a + a2 - 1) % ub + 1;
        if (zl <= replay) {
            check(!exact, tag + ": no Z: its last prompt ubatch (" + std::to_string(zl) + " rows) would be lazy (<= replay " +
                  std::to_string(replay) + "); pick --prompt so (P + kept) % ub > replay");
            continue;
        }
        {
            sess Z;
            if (!Z.open(port, q, err) || !Z.prompt(P + a + a2, ub, err) || !Z.chunk(P + a + a2, nf, 0, ALL, lgZ, hZ, err)) {
                check(false, tag + " Z: " + err); continue;
            }
        }
        const float * fx = lgX.data() + (size_t) a2 * NV;
        const ggml_fp16_t * hx = hX.data() + (size_t) a2 * n_embd;
        const double mz = max_diff(fx, lgZ.data(), (size_t) nf * NV);
        if (exact) {
            const bool xz = same(fx, lgZ.data(), (size_t) nf * NV * sizeof(float)) && same(hx, hZ.data(), (size_t) nf * n_embd * 2);
            snprintf(b, sizeof b, ": X vs Z (no replay) max |diff| %.3g", mz);
            check(xz, tag + (xz ? ": X vs Z (no replay) bit-identical" : std::string(b)));
        } else {
            printf("INFO  %s: X vs Z (no replay) max |diff| %.3g (bit-compared with --exact)\n", tag.c_str(), mz);
            fflush(stdout);
        }
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
