// mirror-test: correctness gate for the llama_state_filter API (split prefill of an APPEND into a cached conversation).
//
// One process plays both sides, on the CPU by default (no GPU needed):
//   R  reference: the full model prefills P[0, D+N), then greedy-decodes G tokens
//   M  "Mac":     the full model prefills P[0, D) normally; then
//        PUSH  : M writes layers [L, n_layer) of positions [0, D) (in two pieces, to exercise APPEND) -> T reads them
//        SPLIT : M runs layers [0, L) of P[D, D+N) (llama_set_layer_range), T runs the tail on M's residuals
//        MERGE : T writes positions [D, D+N) (+ its recurrent state) -> M overwrites those rows of layers [L, n_layer)
//      then M decodes greedily with the whole model.
//   T  "phone":   the split TAIL GGUF (layers [L, n_layer) renumbered from 0), or the same model with a layer range.
// Gate: M's tokens == R's tokens, and M's layers-[L, n) state bytes == R's right after the prefill.
// Negative controls: --no-push (the tail has no mirror of [0, D)) and --no-merge must both fail.
//
//   llama-mirror-test -m 4B.gguf --tail-model tail-4b-L20.gguf -L 20 -f text.txt -D 1024 -N 1024 [-G 32] [--ub 256] [-ngl 0]
#include "llama.h"
#include "../../src/llama-ext.h"
#include "tail-client.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, true, true);
    std::vector<llama_token> t(n);
    llama_tokenize(vocab, text.c_str(), (int) text.size(), t.data(), n, true, true);
    return t;
}

static int argmax(const float * v, int n) { return (int) (std::max_element(v, v + n) - v); }

static void decode_tokens(llama_context * ctx, const llama_token * toks, int n, llama_pos pos0, bool logits_last) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) { b.token[i] = toks[i]; b.pos[i] = pos0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
    b.logits[n - 1] = logits_last;
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) throw std::runtime_error("decode failed: " + std::to_string(rc));
    llama_synchronize(ctx);
}

static void decode_embd(llama_context * ctx, const float * embd, int n_tok, int n_embd, int n_pos_per_embd, llama_pos pos0, bool logits_last) {
    llama_batch b = llama_batch_init(n_tok, n_embd, 1);
    free(b.pos);
    b.pos = (llama_pos *) malloc(sizeof(llama_pos) * n_tok * n_pos_per_embd);
    for (int j = 0; j < n_pos_per_embd; ++j) for (int i = 0; i < n_tok; ++i) b.pos[j*n_tok + i] = pos0 + i;
    memcpy(b.embd, embd, sizeof(float) * (size_t) n_tok * n_embd);
    for (int i = 0; i < n_tok; ++i) { b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
    b.logits[n_tok - 1] = logits_last;
    b.n_tokens = n_tok;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) throw std::runtime_error("tail decode failed: " + std::to_string(rc));
    llama_synchronize(ctx);
}

static std::vector<llama_token> greedy(llama_context * ctx, int n_vocab, llama_token first, llama_pos pos, int n_gen) {
    std::vector<llama_token> out = { first };
    while ((int) out.size() < n_gen) {
        decode_tokens(ctx, &out.back(), 1, pos++, true);
        out.push_back(argmax(llama_get_logits_ith(ctx, -1), n_vocab));
    }
    return out;
}

// seq-0 state with an optional filter (p0, p1, il0, il1); modes only matter on read
static std::vector<uint8_t> get_state(llama_context * ctx, bool filt, int p0, int p1, int il0, int il1, bool kv = true, bool rs = true) {
    if (filt) llama_state_filter_set(p0, p1, il0, il1, 0, 0, kv, rs);
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
    const size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
    if (filt) llama_state_filter_clear();
    buf.resize(n);
    return buf;
}

static bool set_state(llama_context * ctx, const std::vector<uint8_t> & b, int il0, int il1, int kv_mode, int rs_mode) {
    llama_state_filter_set(0, 0x7fffffff, il0, il1, kv_mode, rs_mode, true, true);
    const size_t n = llama_state_seq_set_data(ctx, b.data(), b.size(), 0);
    llama_state_filter_clear();
    return n == b.size();
}

int main(int argc, char ** argv) {
    std::string model_path, tail_path, text_path, tail_host;
    int L = 0, D = 1024, N = 1024, G = 32, ub = 256, ngl = 0, replay = 0;
    std::string kv_type;
    bool no_push = false, no_merge = false, resid_f32 = false, no_mmap = false, tokens_only = false;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) throw std::runtime_error("missing value for " + s); return std::string(argv[++i]); };
        if (s == "-m") model_path = nx();
        else if (s == "--tail-model") tail_path = nx();
        else if (s == "--tail-host") tail_host = nx();
        else if (s == "-f") text_path = nx();
        else if (s == "-L") L = std::stoi(nx());
        else if (s == "-D") D = std::stoi(nx());
        else if (s == "-N") N = std::stoi(nx());
        else if (s == "-G") G = std::stoi(nx());
        else if (s == "--ub") ub = std::stoi(nx());
        else if (s == "-ngl") ngl = std::stoi(nx());
        else if (s == "--replay") replay = std::stoi(nx());
        else if (s == "--kv") kv_type = nx();
        else if (s == "--resid-f32") resid_f32 = true;
        else if (s == "--no-push") no_push = true;
        else if (s == "--no-merge") no_merge = true;
        else if (s == "--no-mmap") no_mmap = true;          // wired weights: mmapped pages get evicted under pressure on the 24 GB Mac
        else if (s == "--tokens-only") tokens_only = true;  // another device's GPU computes the tail: bytes differ, tokens must not
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (model_path.empty() || text_path.empty() || L <= 0 || D % ub || N % ub) {
        fprintf(stderr, "usage: %s -m model.gguf [--tail-model tail.gguf] -L L -f text.txt [-D 1024 -N 1024 -G 32 --ub 256 -ngl 0] [--no-push|--no-merge]\n"
                        "       (D and N must be multiples of --ub, so every context sees the same ubatch shapes)\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    if (no_mmap) mp.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    llama_model * tail  = tail_path.empty() ? nullptr : llama_model_load_from_file(tail_path.c_str(), mp);
    if (!model || (!tail_path.empty() && !tail)) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab), n_embd = llama_model_n_embd(model), n_layer = llama_model_n_layer(model);
    const int n_pos_per_embd = 4;   // qwen35 M-RoPE

    std::ifstream f(text_path); std::stringstream ss; ss << f.rdbuf();
    std::vector<llama_token> P = tokenize(vocab, ss.str());
    if ((int) P.size() < D + N) { fprintf(stderr, "text has %zu tokens < D+N = %d\n", P.size(), D + N); return 1; }
    P.resize(D + N);

    auto cp = llama_context_default_params();
    cp.n_ctx = D + N + G + 16; cp.n_batch = ub; cp.n_ubatch = ub; cp.n_seq_max = 1; cp.no_perf = true;
    cp.n_rs_replay = replay;   // GDN replay rollback, as the server runs it (--spec-gdn-replay)
    if (!kv_type.empty()) {
        const ggml_type t = kv_type == "q8_0" ? GGML_TYPE_Q8_0 : kv_type == "q4_0" ? GGML_TYPE_Q4_0 : kv_type == "f16" ? GGML_TYPE_F16 : GGML_TYPE_COUNT;
        if (t == GGML_TYPE_COUNT) { fprintf(stderr, "--kv: f16 | q8_0 | q4_0\n"); return 2; }
        cp.type_k = t; cp.type_v = t;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;   // quantized V needs flash attention
    }
    llama_context * R = llama_init_from_model(model, cp);
    llama_context * M = llama_init_from_model(model, cp);
    const bool remote = !tail_host.empty();
    llama_context * T = remote ? nullptr : llama_init_from_model(tail ? tail : model, cp);
    if (!R || !M || (!remote && !T)) { fprintf(stderr, "context creation failed\n"); return 1; }
    if (T && !tail) llama_set_layer_range(T, L, -1);
    spt::tail_client tc;
    tc.resid_f32 = resid_f32;
    if (remote) {
        spt::hello_req2 q = {};
        q.base.proto = spt::PROTO_VERSION; q.base.state_format = spt::STATE_FORMAT;
        q.base.L = (uint32_t) L; q.base.n_layer_full = (uint32_t) n_layer; q.base.n_embd = (uint32_t) n_embd;
        q.base.n_vocab = (uint32_t) n_vocab; q.base.n_ctx = cp.n_ctx; q.base.n_ubatch = (uint32_t) ub;
        q.type_k = cp.type_k; q.type_v = cp.type_v;
        q.flash_attn = cp.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED ? 1 : 0;
        q.n_rs_replay = (uint32_t) replay; q.session = 1; q.keep = 0;
        spt::hello_rep2 r; std::string err;
        const size_t c = tail_host.rfind(':');
        if (!tc.connect(tail_host.substr(0, c), c == std::string::npos ? spt::DEFAULT_PORT : std::stoi(tail_host.substr(c + 1)), q, r, err)) {
            fprintf(stderr, "tail worker: %s\n", err.c_str()); return 1;
        }
        printf("tail worker %s: %s, layers [%u, %u), mirror %u\n", tail_host.c_str(), r.base.desc, r.base.layer_start, r.base.n_layer_full, r.n_valid);
    }
    // layer ids of the tail as T's model sees them
    const int t_il0 = tail ? 0 : L, t_il1 = tail ? 0x7fffffff : n_layer;

    // ---- R: reference
    for (int p0 = 0; p0 < D + N; p0 += ub) decode_tokens(R, P.data() + p0, ub, p0, p0 + ub == D + N);
    const llama_token r_first = argmax(llama_get_logits_ith(R, -1), n_vocab);
    const std::vector<uint8_t> r_tail_state = get_state(R, true, 0, 0x7fffffff, L, n_layer);
    const std::vector<llama_token> ref = greedy(R, n_vocab, r_first, D + N, G);

    // ---- M: the cached conversation [0, D)
    for (int p0 = 0; p0 < D; p0 += ub) decode_tokens(M, P.data() + p0, ub, p0, false);

    // ---- PUSH in two pieces: [0, D/2) then [D/2, D), KV appended; the recurrent state (at D) replaces
    if (!no_push) {
        const int Dh = (D / 2) / ub * ub;
        const auto b1 = get_state(M, true, 0, Dh, L, n_layer);
        const auto b2 = get_state(M, true, Dh, D, L, n_layer);
        if (remote) {
            // the protocol's way: KV pieces, then the recurrent state on its own
            const auto k1 = get_state(M, true, 0, Dh, L, n_layer, true, false);
            const auto k2 = get_state(M, true, Dh, D, L, n_layer, true, false);
            const auto rs = get_state(M, true, 0, 0x7fffffff, L, n_layer, false, true);
            tc.sync_state(k1, 1, Dh);
            tc.sync_state(k2, 1, D);
            tc.sync_state(rs, 2, D);
            printf("push: %zu + %zu KV + %zu recurrent bytes, phone mirror %u\n", k1.size(), k2.size(), rs.size(), tc.n_valid());
        } else {
        if (!set_state(T, b1, t_il0, t_il1, /*kv*/ 1, /*rs*/ 0) || !set_state(T, b2, t_il0, t_il1, 1, 0)) { fprintf(stderr, "PUSH failed\n"); return 1; }
        printf("push: %zu + %zu bytes (layers [%d, %d), positions [0, %d) + [%d, %d))\n", b1.size(), b2.size(), L, n_layer, Dh, Dh, D);
        }
    }

    // ---- SPLIT the append [D, D+N)
    llama_set_layer_range(M, 0, L);
    llama_set_embeddings_layer_inp(M, L, true);
    if (remote && no_push) {
        tc.trim(0);
        // the phone mirror is empty: CHUNK must start at its end, so give it zeros for [0, D) the honest way: it can't.
        fprintf(stderr, "--no-push with --tail-host: the phone refuses a CHUNK past its mirror end (that is the check); skipping\n");
        return 0;
    }
    llama_token m_first = 0;
    std::vector<float> last_logits;
    for (int p0 = D; p0 < D + N; p0 += ub) {
        decode_tokens(M, P.data() + p0, ub, p0, false);
        if (remote) tc.submit_chunk(llama_get_embeddings_layer_inp(M, L), ub, n_embd, p0, p0 + ub == D + N, false);
        else decode_embd(T, llama_get_embeddings_layer_inp(M, L), ub, n_embd, n_pos_per_embd, p0, p0 + ub == D + N);
    }
    if (remote) {
        std::vector<ggml_fp16_t> taps; std::string err;
        if (!tc.finish(last_logits, taps, err)) { fprintf(stderr, "tail: %s\n", err.c_str()); return 1; }
        m_first = argmax(last_logits.data(), n_vocab);
    } else {
        m_first = argmax(llama_get_logits_ith(T, -1), n_vocab);
    }
    llama_set_embeddings_layer_inp(M, L, false);
    llama_set_layer_range(M, 0, -1);

    // ---- MERGE: T's rows of [D, D+N) + its recurrent state -> overwrite M's layers [L, n)
    if (!no_merge) {
        const auto b = remote ? tc.state_range(D, D + N, 3) : get_state(T, true, D, D + N, t_il0, t_il1);
        if (!set_state(M, b, L, n_layer, /*kv*/ 2, /*rs*/ 2)) { fprintf(stderr, "MERGE failed\n"); return 1; }
        printf("merge: %zu bytes\n", b.size());
    }

    const std::vector<uint8_t> m_tail_state = get_state(M, true, 0, 0x7fffffff, L, n_layer);
    const std::vector<llama_token> got = greedy(M, n_vocab, m_first, D + N, G);

    int div = -1;
    for (int i = 0; i < G; i++) if (ref[i] != got[i]) { div = i; break; }
    size_t nd = 0;
    const bool same_size = r_tail_state.size() == m_tail_state.size();
    if (same_size) for (size_t i = 0; i < r_tail_state.size(); i++) nd += r_tail_state[i] != m_tail_state[i];
    printf("tokens: %s (first diverge %d of %d) | tail-layer state: %s (%zu vs %zu bytes, %zu differ)\n",
           div < 0 ? "IDENTICAL" : "DIFFERENT", div, G, same_size && nd == 0 ? "BYTE-IDENTICAL" : "DIFFERENT",
           r_tail_state.size(), m_tail_state.size(), nd);
    const bool pass = div < 0 && same_size && (nd == 0 || tokens_only);
    printf("%s\n", pass ? "PASS" : "FAIL");

    llama_free(R); llama_free(M); if (T) llama_free(T);
    llama_model_free(model); if (tail) llama_model_free(tail);
    llama_backend_free();
    return pass ? 0 : 1;
}
