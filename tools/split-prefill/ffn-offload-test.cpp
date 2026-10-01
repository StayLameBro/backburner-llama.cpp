// ffn-offload-test: correctness gate for llama_set_ffn_offload (the hook the phone tail uses to run the FFN on the ANE).
//
//   llama-ffn-offload-test -m model.gguf -f text.txt [--il0 20 --il1 32 -n 512 -G 32 --ub 256 -ngl 99]
//
// Two contexts on the same model and prompt: A runs normally, B replaces the FFN of layers [il0, il1) with a CPU
// reference (Accelerate sgemm on the dequantized GGUF weights, f32). Pass = the same greedy continuation; the report
// also gives max|dlogit| of the first predicted token (f32 CPU vs the GPU's quantized matmuls, so small but nonzero).
#include "llama.h"
#include "../../src/llama-ext.h"
#include "ggml.h"
#include "gguf.h"
#include "ane-ffn.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct ref_ffn {
    int n_embd = 0, n_ff = 0;
    std::vector<std::vector<float>> gate, up, down;   // per layer, row-major [n_ff][n_embd], [n_ff][n_embd], [n_embd][n_ff]
    double ms = 0; int calls = 0;
};

static std::vector<float> load_f32(gguf_context * g, FILE * f, const std::string & name) {
    const int64_t id = gguf_find_tensor(g, name.c_str());
    if (id < 0) throw std::runtime_error("missing tensor " + name);
    const ggml_type t = gguf_get_tensor_type(g, id);
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    const size_t nb = gguf_get_tensor_size(g, id);
    std::vector<uint8_t> raw(nb);
    fseeko(f, (off_t) off, SEEK_SET);
    if (fread(raw.data(), 1, nb, f) != nb) throw std::runtime_error("short read " + name);
    const size_t n = nb / ggml_type_size(t) * ggml_blck_size(t);
    std::vector<float> out(n);
    if (t == GGML_TYPE_F32) memcpy(out.data(), raw.data(), n * 4);
    else ggml_get_type_traits(t)->to_float(raw.data(), out.data(), (int64_t) n);
    return out;
}

static bool ref_fn(float * y, const float * x, int32_t n_embd, int32_t n_tok, int32_t il, void * user) {
    auto * r = (ref_ffn *) user;
    const auto t0 = std::chrono::steady_clock::now();
    const int F = r->n_ff;
    std::vector<float> g((size_t) n_tok * F), u((size_t) n_tok * F);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_tok, F, n_embd, 1.0f, x, n_embd, r->gate[il].data(), n_embd, 0.0f, g.data(), F);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_tok, F, n_embd, 1.0f, x, n_embd, r->up[il].data(), n_embd, 0.0f, u.data(), F);
    for (size_t i = 0; i < g.size(); i++) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_tok, n_embd, F, 1.0f, g.data(), F, r->down[il].data(), F, 0.0f, y, n_embd);
    r->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    r->calls++;
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, text_path, ane_dir;
    int il0 = 20, il1 = 32, N = 512, G = 32, ub = 256, ngl = 99;
    double min_agree = 101;   // --min-agree PCT: pass on top-1 agreement instead of identical greedy text (lossy paths)
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (s == "-m") model_path = nx();
        else if (s == "-f") text_path = nx();
        else if (s == "--il0") il0 = std::stoi(nx());
        else if (s == "--il1") il1 = std::stoi(nx());
        else if (s == "-n") N = std::stoi(nx());
        else if (s == "-G") G = std::stoi(nx());
        else if (s == "--ub") ub = std::stoi(nx());
        else if (s == "-ngl") ngl = std::stoi(nx());
        else if (s == "--min-agree") min_agree = std::stod(nx());
        else if (s == "--ane") ane_dir = nx();   // run the offloaded FFNs on the ANE (ffn_L<il>.mlmodelc in this dir) instead of the CPU reference
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (model_path.empty() || text_path.empty() || N % ub) { fprintf(stderr, "usage: %s -m model -f text [--il0 --il1 -n -G --ub -ngl]\n", argv[0]); return 2; }

    // reference weights
    ref_ffn R;
    if (ane_dir.empty()) {
        gguf_init_params gp = { true, nullptr };
        gguf_context * g = gguf_init_from_file(model_path.c_str(), gp);
        FILE * f = fopen(model_path.c_str(), "rb");
        if (!g || !f) { fprintf(stderr, "cannot read %s\n", model_path.c_str()); return 1; }
        R.gate.resize(il1); R.up.resize(il1); R.down.resize(il1);
        for (int il = il0; il < il1; il++) {
            const std::string p = "blk." + std::to_string(il) + ".";
            R.gate[il] = load_f32(g, f, p + "ffn_gate.weight");
            R.up[il]   = load_f32(g, f, p + "ffn_up.weight");
            R.down[il] = load_f32(g, f, p + "ffn_down.weight");
        }
        fclose(f); gguf_free(g);
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    R.n_embd = llama_model_n_embd(model);
    if (ane_dir.empty()) R.n_ff = (int) (R.gate[il0].size() / R.n_embd);
    spt::ane_ffn * ane = nullptr;
    if (!ane_dir.empty()) {
        std::string err;
        ane = spt::ane_ffn_open(ane_dir, il0, il1, 0, err);
        if (!ane) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    }

    std::ifstream fs(text_path); std::stringstream ss; ss << fs.rdbuf();
    std::vector<llama_token> P(ss.str().size() + 16);
    const int n_tok = llama_tokenize(vocab, ss.str().c_str(), (int) ss.str().size(), P.data(), (int) P.size(), true, false);
    if (n_tok < N) { fprintf(stderr, "text too short\n"); return 1; }
    P.resize(N);

    auto cp = llama_context_default_params();
    cp.n_ctx = N + G + 16; cp.n_batch = ub; cp.n_outputs_max = ub; cp.n_ubatch = ub; cp.n_seq_max = 1; cp.no_perf = true;
    llama_context * A = llama_init_from_model(model, cp);
    llama_context * B = llama_init_from_model(model, cp);
    if (ane) llama_set_ffn_offload(B, il0, il1, spt::ane_ffn_run, ane);
    else     llama_set_ffn_offload(B, il0, il1, ref_fn, &R);

    // every prompt position's logits (top-1 agreement + KL over all of them: a numeric change flips near-ties, so
    // "identical greedy text" is the wrong bar for a lossy path; this is the bar docs/kv-140k.md used for q4_0 KV)
    auto run = [&](llama_context * c, std::vector<float> & first_logits, std::vector<std::vector<float>> & all) {
        std::vector<llama_token> out;
        all.clear();
        for (int p0 = 0; p0 < N; p0 += ub) {
            llama_batch b = llama_batch_init(ub, 0, 1);
            for (int i = 0; i < ub; i++) { b.token[i] = P[p0 + i]; b.pos[i] = p0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
            b.n_tokens = ub;
            if (llama_decode(c, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
            for (int i = 0; i < ub; i++) { const float * l = llama_get_logits_ith(c, i); all.emplace_back(l, l + n_vocab); }
            llama_batch_free(b);
        }
        const float * lg = llama_get_logits_ith(c, -1);
        first_logits.assign(lg, lg + n_vocab);
        llama_token t = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
        for (int i = 0; i < G; i++) {
            out.push_back(t);
            llama_batch b = llama_batch_get_one(&t, 1);
            if (llama_decode(c, b)) { fprintf(stderr, "decode failed\n"); exit(1); }
            const float * l = llama_get_logits_ith(c, -1);
            t = (llama_token) (std::max_element(l, l + n_vocab) - l);
        }
        return out;
    };
    std::vector<float> la, lb;
    std::vector<std::vector<float>> aa, ab;
    const auto ta = run(A, la, aa);
    const auto tb = run(B, lb, ab);
    int agree = 0; double kl_sum = 0, kl_max = 0;
    for (size_t t = 0; t < aa.size(); t++) {
        const auto & p = aa[t]; const auto & q = ab[t];
        agree += std::max_element(p.begin(), p.end()) - p.begin() == std::max_element(q.begin(), q.end()) - q.begin();
        const double mp = *std::max_element(p.begin(), p.end()), mq = *std::max_element(q.begin(), q.end());
        double zp = 0, zq = 0; for (int i = 0; i < n_vocab; i++) { zp += std::exp(p[i] - mp); zq += std::exp(q[i] - mq); }
        double kl = 0;
        for (int i = 0; i < n_vocab; i++) { const double lp = p[i] - mp - std::log(zp), lq = q[i] - mq - std::log(zq); kl += std::exp(lp) * (lp - lq); }
        kl_sum += kl; kl_max = std::max(kl_max, kl);
    }
    printf("prompt positions %zu: top-1 agree %.2f%% | KL mean %.5f max %.4f\n", aa.size(), 100.0 * agree / aa.size(), kl_sum / aa.size(), kl_max);
    const bool failed = llama_ffn_offload_failed(B);
    if (ane) {
        const auto st = spt::ane_ffn_get_stats(ane);
        R.calls = (int) st.calls; R.ms = st.ms_total;
        printf("ANE: %llu calls, %llu blocks, %.2f ms/block in CoreML, %.2f ms/call total%s%s\n", (unsigned long long) st.calls,
               (unsigned long long) st.blocks, st.blocks ? st.ms_pred / st.blocks : 0.0, st.calls ? st.ms_total / st.calls : 0.0,
               failed ? " | error: " : "", failed ? spt::ane_ffn_last_error(ane).c_str() : "");
    }
    float md = 0; for (int i = 0; i < n_vocab; i++) md = std::max(md, std::fabs(la[i] - lb[i]));
    int div = -1; for (int i = 0; i < G; i++) if (ta[i] != tb[i]) { div = i; break; }
    printf("offload layers [%d, %d): %d calls, %.1f ms/call | max|dlogit| %.4f | tokens %s (first diverge %d of %d)%s\n",
           il0, il1, R.calls, R.calls ? R.ms / R.calls : 0.0, md, div < 0 ? "IDENTICAL" : "DIFFERENT", div, G, failed ? " | HOOK FAILED" : "");
    const bool pass = !failed && R.calls > 0 && (div < 0 || 100.0 * agree / aa.size() >= min_agree);
    printf("%s\n", pass ? "PASS" : "FAIL");
    llama_free(A); llama_free(B); spt::ane_ffn_close(ane); llama_model_free(model); llama_backend_free();
    return pass ? 0 : 1;
}
