// split-prefill: Mac + phone split-prefill engine
//
//   HEAD  = decoder layers [0, L)          run on the Mac   (llama_set_layer_range(ctx, 0, L))
//   TAIL  = decoder layers [L, n_layer) + output norm + lm_head, one of
//           - the same model with llama_set_layer_range(ctx, L, -1)                  (default, stage 1)
//           - a split TAIL GGUF (scripts/split-gguf.py, layers renumbered)            (--tail-model, stage 1)
//           - a tail worker over TCP (the iOS Sidecar app, or --serve-tail on a Mac) (--tail-host, stage 2)
//
// prefill: for each ubatch chunk: HEAD(tokens) -> residual [n_tok, n_embd] -> TAIL(embd).
//          With --tail-host the chunks are pipelined: the HEAD computes chunk c+1 on the Mac GPU
//          while the phone computes chunk c. --resid f16 halves the wire bytes.
// merge:   HEAD seq-state blob + TAIL seq-state blob -> full-model seq-state blob,
//          loaded back into the HEAD context, which is then switched to the full layer range
// decode:  greedy from the merged context; compared token-by-token with an unsplit FULL context
//
// Wire protocol and the server: tail-server.h.

#include "llama.h"
#include "../../src/llama-ext.h"
#include "tail-server.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static double now_ms() { return ggml_time_us() / 1000.0; }

// mean over full-size chunks, skipping chunk 0 (first-use warm-up) when there are others
static double steady_mean(const std::vector<double> & t, int n_prompt, int n_ub) {
    double sum = 0; int n = 0;
    for (size_t c = 0; c < t.size(); ++c) {
        const bool full = (int) ((c + 1) * n_ub) <= n_prompt;
        if (!full || (c == 0 && t.size() > 2)) continue;
        sum += t[c]; ++n;
    }
    return n ? sum / n : t[0];
}

// ---------------------------------------------------------------------------------------------
// seq-state blob layout (llama_state_seq_get_data on a hybrid KV + recurrent memory, 1 seq):
//
// [llama_context::state_seq_get_data]  u32 magic 0xaf143cd8, i32 seq_id
// [attn / llama_kv_cache::state_write]
//   u32 n_stream
//   per stream: u32 cell_count; if cell_count:
//     meta: cell_count x { i32 pos, u32 n_seq_id, [ext: i32 x, i32 y, i32 tok  (M-RoPE/PLE)], n_seq_id x i32 }
//     u32 v_trans, u32 n_layer_kv
//     K: n_layer_kv x { i32 type, u64 row_size, cell_count*row_size bytes }
//     V: n_layer_kv x { i32 type, u64 row_size, cell_count*row_size bytes }                (v_trans == 0)
//        n_layer_kv x { i32 type, u32 el_size, u32 n_embd_v, n_embd_v*cell_count*el bytes } (v_trans == 1)
// [recurrent / llama_memory_recurrent::state_write]
//   u32 cell_count
//   meta: cell_count x { i32 pos, u32 n_seq_id(=0 for a single seq), n_seq_id x i32 }
//   u32 s_trans(=0), u32 n_layer (trunk layers, incl. non-recurrent ones)
//   R: n_recr x { i32 type, u64 row_size, cell_count*row_size bytes }   (conv state)
//   S: n_recr x { i32 type, u64 row_size, cell_count*row_size bytes }   (delta-net state)
//
// Each per-layer entry is recorded as [offset, size) of the whole entry (header + payload),
// in the order the entries appear, which is ascending layer index.
// ---------------------------------------------------------------------------------------------

struct blob_entry { size_t off, size; };

struct blob_layout {
    uint32_t kv_cells  = 0;
    uint32_t v_trans   = 0;
    std::vector<blob_entry> k, v;     // per attention layer (ascending il)
    uint32_t rs_cells  = 0;
    uint32_t rs_nlayer = 0;
    std::vector<blob_entry> r, s;     // per recurrent layer (ascending il)
    std::vector<int32_t> kv_pos, rs_pos;
    size_t kv_meta_off = 0, kv_meta_size = 0;
};

struct reader {
    const uint8_t * p; size_t n; size_t o = 0;
    reader(const std::vector<uint8_t> & b) : p(b.data()), n(b.size()) {}
    template<typename T> T get() {
        if (o + sizeof(T) > n) throw std::runtime_error("blob truncated");
        T v; memcpy(&v, p + o, sizeof(T)); o += sizeof(T); return v;
    }
    void skip(size_t k) { if (o + k > n) throw std::runtime_error("blob truncated (skip)"); o += k; }
};

// n_attn / n_recr: expected number of per-layer entries (from the model's layer pattern)
static blob_layout parse_blob(const std::vector<uint8_t> & b, size_t ext_size, uint32_t n_attn, uint32_t n_recr) {
    blob_layout L;
    reader r(b);

    if (r.get<uint32_t>() != 0xaf143cd8u) throw std::runtime_error("bad state magic");
    r.get<int32_t>(); // seq_id

    const uint32_t n_stream = r.get<uint32_t>();
    if (n_stream != 1) throw std::runtime_error("expected 1 kv stream");
    L.kv_cells = r.get<uint32_t>();
    if (L.kv_cells > 0) {
        L.kv_meta_off = r.o;
        for (uint32_t i = 0; i < L.kv_cells; ++i) {
            L.kv_pos.push_back(r.get<int32_t>());
            const uint32_t nsq = r.get<uint32_t>();
            r.skip(ext_size);
            r.skip(nsq * sizeof(int32_t));
        }
        L.kv_meta_size = r.o - L.kv_meta_off;
        L.v_trans = r.get<uint32_t>();
        const uint32_t nl = r.get<uint32_t>();
        if (L.v_trans > 1 || nl != n_attn) {
            throw std::runtime_error("kv header mismatch (v_trans=" + std::to_string(L.v_trans) +
                                     ", n_layer_kv=" + std::to_string(nl) + ", expected " + std::to_string(n_attn) + ")");
        }
        for (uint32_t l = 0; l < nl; ++l) {
            const size_t o = r.o;
            r.get<int32_t>();
            const uint64_t row = r.get<uint64_t>();
            r.skip(row * L.kv_cells);
            L.k.push_back({o, r.o - o});
        }
        for (uint32_t l = 0; l < nl; ++l) {
            const size_t o = r.o;
            r.get<int32_t>();
            if (!L.v_trans) {
                const uint64_t row = r.get<uint64_t>();
                r.skip(row * L.kv_cells);
            } else {
                const uint32_t el = r.get<uint32_t>();
                const uint32_t ne = r.get<uint32_t>();
                r.skip((size_t) el * ne * L.kv_cells);
            }
            L.v.push_back({o, r.o - o});
        }
    }

    L.rs_cells = r.get<uint32_t>();
    for (uint32_t i = 0; i < L.rs_cells; ++i) {
        L.rs_pos.push_back(r.get<int32_t>());
        const uint32_t nsq = r.get<uint32_t>();
        r.skip(nsq * sizeof(int32_t));
    }
    const uint32_t s_trans = r.get<uint32_t>();
    L.rs_nlayer = r.get<uint32_t>();
    if (s_trans != 0) throw std::runtime_error("s_trans != 0 not supported");
    for (auto * vec : { &L.r, &L.s }) {
        for (uint32_t l = 0; l < n_recr; ++l) {
            const size_t o = r.o;
            r.get<int32_t>();
            const uint64_t row = r.get<uint64_t>();
            r.skip(row * L.rs_cells);
            vec->push_back({o, r.o - o});
        }
    }
    if (r.o != b.size()) {
        throw std::runtime_error("blob has " + std::to_string(b.size() - r.o) + " trailing bytes (layout mismatch)");
    }
    return L;
}

// try both cell-ext sizes; the right one is the one that parses exactly
static blob_layout parse_blob_auto(const std::vector<uint8_t> & b, uint32_t n_attn, uint32_t n_recr) {
    std::string err;
    for (size_t ext : { (size_t) 12, (size_t) 0 }) {
        try { return parse_blob(b, ext, n_attn, n_recr); } catch (const std::exception & e) { err += std::string(" [ext=") + std::to_string(ext) + "] " + e.what(); }
    }
    throw std::runtime_error("cannot parse state blob:" + err);
}

// layer bookkeeping for qwen35: full attention at (il+1) % interval == 0, gated delta-net elsewhere
struct layer_map {
    std::vector<int> attn_il, recr_il; // blob entry index -> model layer index
};

static layer_map make_layer_map(int n_layer, int interval) {
    layer_map m;
    for (int il = 0; il < n_layer; ++il) {
        if ((il + 1) % interval == 0) m.attn_il.push_back(il); else m.recr_il.push_back(il);
    }
    return m;
}

// merge: start from the HEAD blob (full layout; layers >= L untouched) and overwrite every per-layer
// entry with il >= L by the corresponding TAIL entry. tail_il_base = 0 when the TAIL blob has the full
// layout (same model, layer range), = L when the TAIL blob comes from a split model (layers renumbered).
static std::vector<uint8_t> merge_blobs(const std::vector<uint8_t> & head, const blob_layout & H,
                                        const std::vector<uint8_t> & tail, const blob_layout & T,
                                        const layer_map & full, int L, int tail_il_base,
                                        size_t * n_bytes_from_tail) {
    if (H.kv_pos != T.kv_pos) throw std::runtime_error("HEAD/TAIL kv cell positions differ");
    if (H.rs_pos != T.rs_pos) throw std::runtime_error("HEAD/TAIL recurrent cell positions differ");
    if (H.v_trans != T.v_trans) throw std::runtime_error("HEAD/TAIL v_trans differ");

    std::vector<uint8_t> out = head;
    size_t moved = 0;

    auto copy_entries = [&](const std::vector<int> & full_il, const std::vector<blob_entry> & he, const std::vector<blob_entry> & te, const char * what) {
        // tail entry index for full layer il: position of il among the tail's layers of this kind
        for (size_t k = 0; k < full_il.size(); ++k) {
            const int il = full_il[k];
            if (il < L) continue;
            // index of this layer within the tail blob
            size_t tk = k;
            if (tail_il_base > 0) {
                tk = 0;
                for (size_t j = 0; j < k; ++j) if (full_il[j] >= L) ++tk;
            }
            if (tk >= te.size()) throw std::runtime_error(std::string("tail blob missing ") + what + " layer " + std::to_string(il));
            if (he[k].size != te[tk].size) throw std::runtime_error(std::string("entry size mismatch for ") + what + " layer " + std::to_string(il));
            memcpy(out.data() + he[k].off, tail.data() + te[tk].off, he[k].size);
            moved += he[k].size;
        }
    };
    copy_entries(full.attn_il, H.k, T.k, "K");
    copy_entries(full.attn_il, H.v, T.v, "V");
    copy_entries(full.recr_il, H.r, T.r, "R");
    copy_entries(full.recr_il, H.s, T.s, "S");

    if (n_bytes_from_tail) *n_bytes_from_tail = moved;
    return out;
}

// ---------------------------------------------------------------------------------------------
// TAIL interface. submit() hands over one chunk's residual and must not block on the TAIL's compute
// (remote_tail queues it; local_tail runs it inline, so the local path is serial). finish() waits for
// every submitted chunk and returns the last-token logits of the chunk submitted with want_logits.
// ---------------------------------------------------------------------------------------------

struct tail_runner {
    virtual ~tail_runner() = default;
    virtual void reset() = 0;
    // residual: n_tok x n_embd f32 (token-major); converted to `dtype` for the wire
    virtual void submit(const float * residual, int n_tok, int n_embd, llama_pos pos0, bool want_logits) = 0;
    virtual void finish(std::vector<float> & logits_out) = 0;
    virtual std::vector<uint8_t> get_state() = 0;   // full seq-0 state blob of the tail context
    virtual int  il_base() const = 0;               // 0: blob has full-model layout, L: renumbered layers
    std::vector<double> tail_ms;                    // per chunk TAIL compute time (as the TAIL measured it)
};

// f32 -> f16 -> f32, exactly what the wire does with --resid f16
static void f16_roundtrip(const float * src, float * dst, size_t n, std::vector<ggml_fp16_t> & tmp) {
    tmp.resize(n);
    ggml_fp32_to_fp16_row(src, tmp.data(), (int64_t) n);
    ggml_fp16_to_fp32_row(tmp.data(), dst, (int64_t) n);
}

struct local_tail : tail_runner {
    llama_context * ctx;
    int n_pos_per_embd, n_vocab, il_base_;
    bool f16;
    std::vector<float> logits, buf;
    std::vector<ggml_fp16_t> tmp;

    local_tail(llama_context * ctx, int n_pos_per_embd, int n_vocab, int il_base, bool f16) :
        ctx(ctx), n_pos_per_embd(n_pos_per_embd), n_vocab(n_vocab), il_base_(il_base), f16(f16) {}

    void reset() override { llama_memory_clear(llama_get_memory(ctx), true); tail_ms.clear(); }

    void submit(const float * residual, int n_tok, int n_embd, llama_pos pos0, bool want_logits) override {
        const double t0 = now_ms();
        llama_batch b = llama_batch_init(n_tok, n_embd, 1);
        // M-RoPE: embd batches carry n_pos_per_embd positions per token, section-major;
        // text tokens use the same position in every section (what the token path broadcasts)
        free(b.pos);
        b.pos = (llama_pos *) malloc(sizeof(llama_pos) * n_tok * n_pos_per_embd);
        for (int j = 0; j < n_pos_per_embd; ++j) {
            for (int i = 0; i < n_tok; ++i) b.pos[j*n_tok + i] = pos0 + i;
        }
        if (f16) f16_roundtrip(residual, b.embd, (size_t) n_tok * n_embd, tmp);
        else     memcpy(b.embd, residual, sizeof(float) * (size_t) n_tok * n_embd);
        for (int i = 0; i < n_tok; ++i) {
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0;
        }
        b.logits[n_tok - 1] = want_logits;
        b.n_tokens = n_tok;
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) throw std::runtime_error("tail decode failed: " + std::to_string(rc));
        if (want_logits) {
            const float * lg = llama_get_logits_ith(ctx, n_tok - 1);
            logits.assign(lg, lg + n_vocab);
        } else {
            llama_synchronize(ctx);
        }
        tail_ms.push_back(now_ms() - t0);
    }

    void finish(std::vector<float> & logits_out) override { logits_out = logits; }

    std::vector<uint8_t> get_state() override {
        std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
        const size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
        buf.resize(n);
        return buf;
    }

    int il_base() const override { return il_base_; }
};

// TCP client for the phone's tail worker (tail-server.h). A writer thread streams queued chunks, a
// reader thread collects the acks, so the HEAD never waits for the phone until finish().
struct remote_tail : tail_runner {
    int fd = -1;
    int L;
    bool f16;
    spt::hello_rep hello = {};

    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::vector<uint8_t>> queue;   // CHUNK payloads (chunk_req + residual)
    int n_submitted = 0, n_acked = 0;
    bool stop = false;
    std::string error;
    std::vector<float> logits;
    std::thread writer, reader;
    double bytes_sent = 0;

    remote_tail(const std::string & hostport, const spt::hello_req & q, bool f16) : L((int) q.L), f16(f16) {
        const size_t c = hostport.rfind(':');
        const std::string host = c == std::string::npos ? hostport : hostport.substr(0, c);
        const int port = c == std::string::npos ? spt::DEFAULT_PORT : std::stoi(hostport.substr(c + 1));
        fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a = {};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t) port);
        if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) throw std::runtime_error("bad IPv4 address " + host);
        spt::tune_socket(fd);
        if (connect(fd, (sockaddr *) &a, sizeof(a)) != 0) {
            throw std::runtime_error("cannot connect to tail worker at " + hostport + ": " + strerror(errno) +
                                     " (is Sidecar open and in the foreground?)");
        }
        spt::send_msg(fd, spt::MSG_HELLO, &q, sizeof(q));
        const spt::msg_hdr h = spt::recv_hdr(fd);
        std::vector<uint8_t> p(h.len);
        if (h.len) spt::recv_all(fd, p.data(), h.len);
        if (h.type == spt::MSG_ERR) throw std::runtime_error("tail worker refused: " + std::string(p.begin(), p.end()));
        if (h.type != spt::MSG_HELLO_OK || h.len != sizeof(hello)) throw std::runtime_error("bad HELLO reply");
        memcpy(&hello, p.data(), sizeof(hello));
        writer = std::thread([this] { write_loop(); });
        reader = std::thread([this] { read_loop(); });
    }

    ~remote_tail() override {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        if (writer.joinable()) writer.join();
        if (reader.joinable()) reader.join();
        if (fd >= 0) {
            try { spt::send_msg(fd, spt::MSG_BYE, nullptr, 0); } catch (...) {}
            close(fd);
        }
    }

    void fail(const std::string & e) {
        std::lock_guard<std::mutex> lk(mu);
        if (error.empty()) error = e;
        stop = true;
        cv.notify_all();
    }

    void write_loop() {
        while (true) {
            std::vector<uint8_t> msg;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !queue.empty(); });
                if (stop) return;
                msg = std::move(queue.front());
                queue.pop_front();
            }
            try {
                spt::send_msg(fd, spt::MSG_CHUNK, msg.data(), msg.size());
            } catch (const std::exception & e) { fail(std::string("send chunk: ") + e.what()); return; }
            bytes_sent += msg.size();
        }
    }

    void read_loop() {
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || n_acked < n_submitted; });
                if (stop) return;
            }
            try {
                const spt::msg_hdr h = spt::recv_hdr(fd);
                if (h.type == spt::MSG_ERR) {
                    std::string e(h.len, '\0'); spt::recv_all(fd, e.data(), h.len);
                    fail("tail worker: " + e); return;
                }
                if (h.type != spt::MSG_CHUNK_ACK || h.len < sizeof(spt::chunk_rep)) { fail("bad CHUNK reply"); return; }
                spt::chunk_rep r; spt::recv_all(fd, &r, sizeof(r));
                std::vector<float> lg(r.n_logits);
                if (r.n_logits) spt::recv_all(fd, lg.data(), lg.size() * sizeof(float));
                std::lock_guard<std::mutex> lk(mu);
                if (r.n_logits) logits = std::move(lg);
                tail_ms.push_back(r.compute_ms);
                n_acked++;
            } catch (const std::exception & e) { fail(std::string("recv ack: ") + e.what()); return; }
            cv.notify_all();
        }
    }

    void check() { std::lock_guard<std::mutex> lk(mu); if (!error.empty()) throw std::runtime_error(error); }

    void reset() override {
        check();
        spt::send_msg(fd, spt::MSG_RESET, nullptr, 0);
        const spt::msg_hdr h = spt::recv_hdr(fd);
        if (h.type != spt::MSG_RESET_OK) throw std::runtime_error("bad RESET reply");
        std::lock_guard<std::mutex> lk(mu);
        tail_ms.clear(); logits.clear(); n_submitted = n_acked = 0;
    }

    void submit(const float * residual, int n_tok, int n_embd, llama_pos pos0, bool want_logits) override {
        check();
        spt::chunk_req q = { pos0, (uint32_t) n_tok, f16 ? spt::DT_F16 : spt::DT_F32, want_logits ? 1u : 0u };
        const size_t n = (size_t) n_tok * n_embd;
        std::vector<uint8_t> msg(sizeof(q) + n * (f16 ? 2 : 4));
        memcpy(msg.data(), &q, sizeof(q));
        if (f16) ggml_fp32_to_fp16_row(residual, (ggml_fp16_t *) (msg.data() + sizeof(q)), (int64_t) n);
        else     memcpy(msg.data() + sizeof(q), residual, n * sizeof(float));
        {
            std::lock_guard<std::mutex> lk(mu);
            queue.push_back(std::move(msg));
            n_submitted++;
        }
        cv.notify_all();
    }

    void finish(std::vector<float> & logits_out) override {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !error.empty() || n_acked == n_submitted; });
        if (!error.empty()) throw std::runtime_error(error);
        logits_out = logits;
    }

    std::vector<uint8_t> get_state() override {
        check();
        spt::send_msg(fd, spt::MSG_STATE, nullptr, 0);
        const spt::msg_hdr h = spt::recv_hdr(fd);
        std::vector<uint8_t> blob(h.len);
        if (h.len) spt::recv_all(fd, blob.data(), h.len);
        if (h.type == spt::MSG_ERR) throw std::runtime_error("tail worker: " + std::string(blob.begin(), blob.end()));
        if (h.type != spt::MSG_STATE_DATA) throw std::runtime_error("bad STATE reply");
        return blob;
    }

    int il_base() const override { return L; }
};

// ---------------------------------------------------------------------------------------------

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, true, true);
    std::vector<llama_token> t(n);
    llama_tokenize(vocab, text.c_str(), (int) text.size(), t.data(), n, true, true);
    return t;
}

static int argmax(const float * v, int n) {
    return (int) (std::max_element(v, v + n) - v);
}

static void decode_tokens(llama_context * ctx, const llama_token * toks, int n, llama_pos pos0, bool logits_last) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = toks[i]; b.pos[i] = pos0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0;
    }
    b.logits[n - 1] = logits_last;
    b.n_tokens = n;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) throw std::runtime_error("decode failed: " + std::to_string(rc));
    llama_synchronize(ctx);
}

// greedy decode n_gen tokens after `first` has been chosen; ctx memory holds positions [0, pos)
static std::vector<llama_token> greedy(llama_context * ctx, const llama_vocab * vocab, llama_token first, llama_pos pos, int n_gen) {
    std::vector<llama_token> out = { first };
    const int n_vocab = llama_vocab_n_tokens(vocab);
    while ((int) out.size() < n_gen) {
        decode_tokens(ctx, &out.back(), 1, pos++, true);
        out.push_back(argmax(llama_get_logits_ith(ctx, -1), n_vocab));
    }
    return out;
}

static std::vector<int> parse_int_list(const std::string & s) {
    std::vector<int> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) v.push_back(std::stoi(t));
    return v;
}

struct args_t {
    std::string model, tail_model, tail_host;
    std::vector<std::string> prompt_files;
    std::vector<int> Ls = { 16 };
    std::vector<int> n_prompts = { 0 };   // 0 = whole file
    int n_ub = 256, n_gen = 64, n_ctx = 4096, attn_interval = 4;
    int serve_port = 0;
    // prompts shorter than this are not split: one or two chunks cannot pipeline, and the split
    // measured 0.8x at 512 tokens (docs/split-prefill.md). They run Mac-only.
    int min_split = 1500;
    bool timing_only = false;
    bool no_merge    = false; // negative control: restore the HEAD blob without the TAIL layers
    bool f16         = false; // residual on the wire as f16
    bool warmup      = true;
};

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf -f prompt.txt [-f prompt2.txt ...] [-L 16,20] [--ub 256] [-n 64] [-c 4096]\n"
        "          [--n-prompt N[,N2,...]] [--tail-model tail.gguf | --tail-host IP[:PORT]] [--resid f32|f16]\n"
        "          [--timing-only] [--no-merge] [--no-warmup] [--min-split N (default 1500; 0 = always split)]\n"
        "       %s --serve-tail PORT --tail-model tail.gguf       (run the phone's tail worker on this Mac)\n"
        "  --tail-model: split TAIL GGUF (layers renumbered from 0); only valid with a single -L\n"
        "  --tail-host : stream chunks to a tail worker (Sidecar on the phone, default port %d); single -L\n",
        argv0, argv0, spt::DEFAULT_PORT);
}

static int serve_tail(const args_t & a) {
    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * txt, void *) { if (lvl >= GGML_LOG_LEVEL_WARN) fputs(txt, stderr); }, nullptr);
    spt::server_status st;
    spt::tail_server srv(a.tail_model, &st, [](const std::string & s) { fprintf(stderr, "%s\n", s.c_str()); });
    const std::string e = srv.load();
    if (!e.empty()) { fprintf(stderr, "%s\n", e.c_str()); return 1; }
    fprintf(stderr, "%s\n", srv.serve(a.serve_port).c_str());
    return 1;
}

int main(int argc, char ** argv) {
    args_t a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(argv[0]); exit(1); } return argv[++i]; };
        if      (s == "-m") a.model = next();
        else if (s == "--tail-model") a.tail_model = next();
        else if (s == "--tail-host") a.tail_host = next();
        else if (s == "--serve-tail") a.serve_port = std::stoi(next());
        else if (s == "-f") a.prompt_files.push_back(next());
        else if (s == "-L") a.Ls = parse_int_list(next());
        else if (s == "--ub") a.n_ub = std::stoi(next());
        else if (s == "-n") a.n_gen = std::stoi(next());
        else if (s == "-c") a.n_ctx = std::stoi(next());
        else if (s == "--n-prompt") a.n_prompts = parse_int_list(next());
        else if (s == "--resid") { const std::string v = next(); if (v != "f16" && v != "f32") { usage(argv[0]); return 1; } a.f16 = v == "f16"; }
        else if (s == "--timing-only") a.timing_only = true;
        else if (s == "--no-merge") a.no_merge = true;
        else if (s == "--no-warmup") a.warmup = false;
        else if (s == "--min-split") a.min_split = std::stoi(next());
        else { usage(argv[0]); return 1; }
    }
    if (a.serve_port > 0) {
        if (a.tail_model.empty()) { usage(argv[0]); return 1; }
        return serve_tail(a);
    }
    if (a.model.empty() || a.prompt_files.empty()) { usage(argv[0]); return 1; }
    const bool remote = !a.tail_host.empty();
    if ((!a.tail_model.empty() || remote) && a.Ls.size() != 1) { fprintf(stderr, "--tail-model / --tail-host need exactly one -L\n"); return 1; }
    if (remote && !a.tail_model.empty()) { fprintf(stderr, "--tail-model and --tail-host are exclusive\n"); return 1; }

    llama_backend_init();
    llama_log_set([](ggml_log_level lvl, const char * txt, void *) { if (lvl >= GGML_LOG_LEVEL_WARN) fputs(txt, stderr); }, nullptr);

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", a.model.c_str()); return 1; }
    llama_model * tail_model = nullptr;
    if (!a.tail_model.empty()) {
        tail_model = llama_model_load_from_file(a.tail_model.c_str(), mparams);
        if (!tail_model) { fprintf(stderr, "failed to load %s\n", a.tail_model.c_str()); return 1; }
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int rope    = llama_model_rope_type(model);
    const int n_pos_per_embd = (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
    const layer_map full_map = make_layer_map(n_layer, a.attn_interval);

    fprintf(stderr, "model: n_layer=%d n_embd=%d n_vocab=%d n_pos_per_embd=%d attn=%zu recr=%zu\n",
            n_layer, n_embd, n_vocab, n_pos_per_embd, full_map.attn_il.size(), full_map.recr_il.size());

    for (int L : a.Ls) {
        if (L <= 0 || L >= n_layer || L % a.attn_interval != 0) {
            fprintf(stderr, "invalid L=%d (must be a multiple of %d in (0, %d))\n", L, a.attn_interval, n_layer);
            return 1;
        }
    }

    // contexts must hold the longest prompt plus the greedy continuation
    const int max_np = *std::max_element(a.n_prompts.begin(), a.n_prompts.end());
    if (max_np > 0) a.n_ctx = std::max(a.n_ctx, max_np + a.n_gen + 16);

    auto cparams = llama_context_default_params();
    cparams.n_ctx     = a.n_ctx;
    cparams.n_batch   = a.n_ub;
    cparams.n_ubatch  = a.n_ub;
    cparams.n_seq_max = 1;
    cparams.no_perf   = true;

    llama_context * ctx_full = llama_init_from_model(model, cparams);
    llama_context * ctx_head = llama_init_from_model(model, cparams);
    llama_context * ctx_tail = remote ? nullptr : llama_init_from_model(tail_model ? tail_model : model, cparams);
    if (!ctx_full || !ctx_head || (!remote && !ctx_tail)) { fprintf(stderr, "context creation failed\n"); return 1; }

    int n_fail = 0;
    std::unique_ptr<tail_runner> tail;
    if (remote) {
        spt::hello_req q = {};
        q.proto = spt::PROTO_VERSION; q.state_format = spt::STATE_FORMAT;
        q.L = (uint32_t) a.Ls[0]; q.n_layer_full = (uint32_t) n_layer;
        q.n_embd = (uint32_t) n_embd; q.n_vocab = (uint32_t) n_vocab;
        q.n_ctx = (uint32_t) a.n_ctx; q.n_ubatch = (uint32_t) a.n_ub;
        try {
            auto * rt = new remote_tail(a.tail_host, q, a.f16);
            tail.reset(rt);
            fprintf(stderr, "tail worker %s: %s, layers [%u, %u), %.2f GB, n_ctx %u\n", a.tail_host.c_str(), rt->hello.desc,
                    rt->hello.layer_start, rt->hello.n_layer_full, rt->hello.file_bytes / 1e9, rt->hello.n_ctx);
        } catch (const std::exception & e) {
            fprintf(stderr, "%s\n", e.what());
            n_fail = -1;
        }
    }

    if (n_fail == 0) {
    printf("| prompt | n_prompt | L | ub | resid | token-identical | first diverge | max|dlogit| first tok | HEAD ms/chunk | TAIL ms/chunk | FULL ms/chunk "
           "| FULL tok/s | split prefill ms (to logits) | state ms | merge ms | split tok/s (incl. state+merge) | speedup | tail state bytes |\n");
    printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");

    for (const auto & pf : a.prompt_files) {
        std::ifstream f(pf);
        if (!f) { fprintf(stderr, "cannot read %s\n", pf.c_str()); return 1; }
        std::stringstream ss; ss << f.rdbuf();
        const std::vector<llama_token> prompt_all = tokenize(vocab, ss.str());

      for (int np_req : a.n_prompts) {
        std::vector<llama_token> prompt = prompt_all;
        if (np_req > 0) {
            if ((int) prompt.size() < np_req) {
                fprintf(stderr, "%s has only %zu tokens (< %d), skipping\n", pf.c_str(), prompt.size(), np_req);
                continue;
            }
            prompt.resize(np_req);
        }
        const int n_prompt = (int) prompt.size();
        const int n_chunks = (n_prompt + a.n_ub - 1) / a.n_ub;

        // ---- FULL reference ----
        if (a.warmup) {
            llama_memory_clear(llama_get_memory(ctx_full), true);
            decode_tokens(ctx_full, prompt.data(), std::min(a.n_ub, n_prompt), 0, true);
        }
        llama_memory_clear(llama_get_memory(ctx_full), true);
        std::vector<double> t_full;
        for (int c = 0; c < n_chunks; ++c) {
            const int p0 = c * a.n_ub, n = std::min(a.n_ub, n_prompt - p0);
            const double t0 = now_ms();
            decode_tokens(ctx_full, prompt.data() + p0, n, p0, c == n_chunks - 1);
            t_full.push_back(now_ms() - t0);
        }
        double t_full_total = 0; for (double t : t_full) t_full_total += t;
        std::vector<float> ref_logits(llama_get_logits_ith(ctx_full, -1), llama_get_logits_ith(ctx_full, -1) + n_vocab);
        std::vector<llama_token> ref;
        if (!a.timing_only) {
            ref = greedy(ctx_full, vocab, argmax(ref_logits.data(), n_vocab), n_prompt, a.n_gen);
        }

        if (n_prompt < a.min_split) {
            // routed: the Mac-only prefill above is the answer, so split tok/s == FULL tok/s
            const double tps = n_prompt * 1000.0 / t_full_total;
            for (int L : a.Ls) {
                printf("| %s | %d | %d | %d | %s | routed Mac-only (< --min-split %d) | - | - | - | - | %.1f | %.1f | - | - | - | %.1f | 1.00x | - |\n",
                       pf.c_str(), n_prompt, L, a.n_ub, a.f16 ? "f16" : "f32", a.min_split, t_full_total / n_chunks, tps, tps);
            }
            fflush(stdout);
            continue;
        }

        for (int L : a.Ls) {
            std::unique_ptr<tail_runner> local;
            if (!remote) {
                local.reset(new local_tail(ctx_tail, n_pos_per_embd, n_vocab, tail_model ? L : 0, a.f16));
                if (!tail_model) llama_set_layer_range(ctx_tail, L, -1);
            }
            tail_runner & tr = remote ? *tail : *local;

            llama_set_layer_range(ctx_head, 0, L);
            llama_set_embeddings_layer_inp(ctx_head, L, true);

            std::vector<float> tail_logits;
            try {
                // warm-up: one chunk through HEAD and TAIL (Metal pipelines, first-touch pages on both ends)
                if (a.warmup) {
                    tr.reset();
                    llama_memory_clear(llama_get_memory(ctx_head), true);
                    const int n = std::min(a.n_ub, n_prompt);
                    decode_tokens(ctx_head, prompt.data(), n, 0, false);
                    tr.submit(llama_get_embeddings_layer_inp(ctx_head, L), n, n_embd, 0, true);
                    tr.finish(tail_logits);
                }
                tr.reset();
                llama_memory_clear(llama_get_memory(ctx_head), true);

                // ---- split prefill: HEAD chunk c+1 runs on the Mac while the TAIL works on chunk c ----
                std::vector<double> t_head;
                const double t_p0 = now_ms();
                for (int c = 0; c < n_chunks; ++c) {
                    const int p0 = c * a.n_ub, n = std::min(a.n_ub, n_prompt - p0);
                    const bool last = c == n_chunks - 1;

                    const double t0 = now_ms();
                    decode_tokens(ctx_head, prompt.data() + p0, n, p0, false);
                    const float * resid = llama_get_embeddings_layer_inp(ctx_head, L);
                    t_head.push_back(now_ms() - t0);

                    tr.submit(resid, n, n_embd, p0, last); // copies the residual before returning
                }
                tr.finish(tail_logits);
                const double t_prefill = now_ms() - t_p0;

                // ---- state fetch + merge ----
                const double t_s0 = now_ms();
                std::vector<uint8_t> tail_blob = tr.get_state();
                const double t_state = now_ms() - t_s0;

                const double t_m0 = now_ms();
                std::vector<uint8_t> head_blob(llama_state_seq_get_size(ctx_head, 0));
                head_blob.resize(llama_state_seq_get_data(ctx_head, head_blob.data(), head_blob.size(), 0));

                const bool renumbered = tr.il_base() > 0;
                const blob_layout H = parse_blob_auto(head_blob, full_map.attn_il.size(), full_map.recr_il.size());
                const int n_tail_attn = renumbered ? (int) std::count_if(full_map.attn_il.begin(), full_map.attn_il.end(), [&](int il){ return il >= L; }) : (int) full_map.attn_il.size();
                const int n_tail_recr = renumbered ? (int) std::count_if(full_map.recr_il.begin(), full_map.recr_il.end(), [&](int il){ return il >= L; }) : (int) full_map.recr_il.size();
                const blob_layout T = parse_blob_auto(tail_blob, n_tail_attn, n_tail_recr);

                size_t n_from_tail = 0;
                std::vector<uint8_t> merged = a.no_merge ? head_blob :
                    merge_blobs(head_blob, H, tail_blob, T, full_map, L, tr.il_base(), &n_from_tail);

                llama_memory_seq_rm(llama_get_memory(ctx_head), 0, -1, -1);
                llama_set_embeddings_layer_inp(ctx_head, L, false);
                llama_set_layer_range(ctx_head, 0, -1);
                if (llama_state_seq_set_data(ctx_head, merged.data(), merged.size(), 0) != merged.size()) {
                    throw std::runtime_error("state_seq_set_data failed");
                }
                const double t_merge = now_ms() - t_m0;

                // ---- compare ----
                if ((int) tail_logits.size() != n_vocab) throw std::runtime_error("no logits from the TAIL");
                double max_diff = 0;
                for (int i = 0; i < n_vocab; ++i) max_diff = std::max(max_diff, (double) std::fabs(tail_logits[i] - ref_logits[i]));

                std::string ident = "n/a", div = "-";
                if (!a.timing_only) {
                    std::vector<llama_token> got = greedy(ctx_head, vocab, argmax(tail_logits.data(), n_vocab), n_prompt, a.n_gen);
                    int d = -1;
                    for (int i = 0; i < a.n_gen; ++i) if (got[i] != ref[i]) { d = i; break; }
                    ident = d < 0 ? "yes" : "NO";
                    if (d >= 0) { div = std::to_string(d); ++n_fail; }
                }

                const double full_tps  = n_prompt * 1000.0 / t_full_total;
                const double split_tps = n_prompt * 1000.0 / (t_prefill + t_state + t_merge);
                printf("| %s | %d | %d | %d | %s | %s | %s | %.3g | %.1f | %.1f | %.1f | %.1f | %.0f | %.0f | %.0f | %.1f | %.2fx | %zu |\n",
                       pf.substr(pf.find_last_of('/') + 1).c_str(), n_prompt, L, a.n_ub, a.f16 ? "f16" : "f32",
                       ident.c_str(), div.c_str(), max_diff,
                       steady_mean(t_head, n_prompt, a.n_ub), steady_mean(tr.tail_ms, n_prompt, a.n_ub), steady_mean(t_full, n_prompt, a.n_ub),
                       full_tps, t_prefill, t_state, t_merge, split_tps, split_tps / full_tps, tail_blob.size());
                fflush(stdout);
                fprintf(stderr, "  L=%d n=%d: blobs head=%zu tail=%zu merged=%zu (moved %zu), kv_cells=%u rs_cells=%u; prefill to logits %.0f ms (%.1f tok/s)\n",
                        L, n_prompt, head_blob.size(), tail_blob.size(), merged.size(), n_from_tail, H.kv_cells, H.rs_cells,
                        t_prefill, n_prompt * 1000.0 / t_prefill);
            } catch (const std::exception & e) {
                fprintf(stderr, "split run failed: %s\n", e.what());
                n_fail = -1;
                goto done;
            }
        }
      }
    }
    }
done:

    tail.reset();
    if (ctx_tail) llama_free(ctx_tail);
    llama_free(ctx_head);
    llama_free(ctx_full);
    if (tail_model) llama_model_free(tail_model);
    llama_model_free(model);
    llama_backend_free();

    return n_fail == 0 ? 0 : n_fail < 0 ? 1 : 2;
}
