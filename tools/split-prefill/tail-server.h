// split-prefill TAIL wire protocol and server (header-only).
//
// Used by two programs built from the same tree:
//   - llama-split-prefill --serve-tail PORT --tail-model tail.gguf   (Mac, loopback testing)
//   - the iOS Sidecar app (ios/Sidecar/Sidecar/RPCBridge.mm), which hosts it next to ggml-rpc
//
// The TAIL is a split GGUF from scripts/split-gguf.py: decoder layers [L, n_layer_full) renumbered
// from 0, plus output norm + lm_head. The client (the Mac HEAD) streams the residual entering layer L
// for each ubatch-sized chunk; the server runs its layers and, at the end, returns its seq-0 state
// blob (llama_state_seq_get_data), which the Mac splices into the full model's state.
//
// Framing: every message is  u32 magic 'SPTL' | u32 type | u64 payload_len | payload.
// All integers are little-endian (both ends are arm64). One client at a time; messages are
// strictly request -> reply, in order, but the client may pipeline several CHUNK requests.
//
//   HELLO      hello_req            -> HELLO_OK hello_rep | ERR text
//   RESET      -                    -> RESET_OK
//   CHUNK      chunk_req + residual -> CHUNK_ACK chunk_rep + n_logits f32 | ERR text
//                residual = n_tok x n_embd, dtype 0 = f32, 1 = f16 (token-major)
//   STATE      -                    -> STATE_DATA blob (llama_state_seq_get_data, seq 0)
//   BYE        -                    -> connection closed
//
// The state blob is llama.cpp's internal seq-state format, which is not a stable API. The Mac
// parses it with an exact-length check, and HELLO carries STATE_FORMAT + PROTO_VERSION, so both
// ends must be built from the same llama.cpp tree. Bump STATE_FORMAT when that format changes.

#pragma once

#include "llama.h"
#include "ggml.h"
#include "../../src/llama-ext.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace spt {

constexpr uint32_t MAGIC         = 0x4C545053u; // "SPTL"
constexpr uint32_t PROTO_VERSION = 2;   // v2 adds TRIM / SYNC / STATE_RANGE / taps; v1 clients (split-prefill) still work
constexpr uint32_t STATE_FORMAT  = 1;           // llama_state_seq_* layout as parsed by split-prefill.cpp
constexpr int      DEFAULT_PORT  = 50060;

enum msg_type : uint32_t {
    MSG_HELLO = 1, MSG_HELLO_OK, MSG_RESET, MSG_RESET_OK, MSG_CHUNK, MSG_CHUNK_ACK,
    MSG_STATE, MSG_STATE_DATA, MSG_BYE, MSG_ERR,
    // link-latency probe, no model needed: PING [u32 reply_len][any bytes] -> PONG [reply_len bytes]
    MSG_PING, MSG_PONG,
    // v2 (split prefill of APPENDS into a cached conversation; the phone keeps a mirror of its layers' state):
    //   TRIM        u32 n                          -> OK u32 n_valid   (mirror past n is dropped; v2: any trim below n_valid resets it)
    //   SYNC        sync_req + state blob          -> OK u32 n_valid   (Mac's filtered state: KV rows appended and/or recurrent replaced)
    //   STATE_RANGE range_req                      -> STATE_DATA blob  (this tail's state: KV rows [p0, p1) and/or recurrent)
    //   CHUNK       want_logits bit 1: also return the tap-layer inputs of the chunk (f16, n_taps x n_tok x n_embd) after the logits
    MSG_TRIM, MSG_SYNC, MSG_OK, MSG_STATE_RANGE,
};

enum resid_dtype : uint32_t { DT_F32 = 0, DT_F16 = 1 };

#pragma pack(push, 1)
struct msg_hdr   { uint32_t magic, type; uint64_t len; };
struct hello_req { uint32_t proto, state_format, L, n_layer_full, n_embd, n_vocab, n_ctx, n_ubatch; };
struct hello_rep {
    uint32_t proto, state_format;
    uint32_t layer_start, n_layer_full, n_layer; // from the tail GGUF (split.layer_start, split.n_layer_full)
    uint32_t n_embd, n_vocab, n_ctx;
    uint64_t file_bytes;
    char     desc[128];
};
struct chunk_req { int32_t pos0; uint32_t n_tok, dtype, want_logits; };   // want_logits: bit 0 logits, bit 1 taps (v2)
// v2 HELLO: hello_req + context settings the Mac server uses, so both caches have the same layout
struct hello_req2 {
    hello_req base;
    int32_t  type_k, type_v;       // ggml_type of the KV cache
    uint32_t flash_attn;           // 1 = on
    uint32_t n_rs_replay;          // --spec-gdn-replay
    uint32_t n_taps;               // tap layers (tail-local ids) whose inputs CHUNK returns on request
    int32_t  taps[4];
    uint64_t session;              // the mirror is kept across connections only for the same session id
    uint32_t keep;                 // 1: keep the mirror if session and settings match
};
struct hello_rep2 { hello_rep base; uint32_t n_valid; uint64_t session; };
struct sync_req   { uint32_t flags, n_valid_after; };          // flags: 1 KV rows (appended), 2 recurrent (replaced)
struct range_req  { int32_t p0, p1; uint32_t flags; };         // flags as sync_req
struct chunk_rep { uint32_t status, n_tok, n_logits; float compute_ms; };
#pragma pack(pop)

// ---- socket helpers ---------------------------------------------------------------------------

inline void send_all(int fd, const void * p, size_t n) {
    const char * c = (const char *) p;
    while (n > 0) {
#ifdef MSG_NOSIGNAL
        ssize_t w = ::send(fd, c, n, MSG_NOSIGNAL);   // a dead peer is an error, not a SIGPIPE that kills the process
#else
        ssize_t w = ::send(fd, c, n, 0);              // SO_NOSIGPIPE is set in tune_socket (macOS/iOS)
#endif
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) throw std::runtime_error("send: timed out (peer not reading)");
        if (w <= 0) throw std::runtime_error(std::string("send: ") + strerror(errno));
        c += w; n -= (size_t) w;
    }
}

inline void recv_all(int fd, void * p, size_t n) {
    char * c = (char *) p;
    while (n > 0) {
        ssize_t r = ::recv(fd, c, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r == 0) throw std::runtime_error("connection closed");
        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) throw std::runtime_error("recv: timed out (peer stopped sending)");
        if (r < 0) throw std::runtime_error(std::string("recv: ") + strerror(errno));
        c += r; n -= (size_t) r;
    }
}

inline void send_msg(int fd, uint32_t type, const void * a, size_t na, const void * b = nullptr, size_t nb = 0) {
    msg_hdr h = { MAGIC, type, (uint64_t) (na + nb) };
    send_all(fd, &h, sizeof(h));
    if (na) send_all(fd, a, na);
    if (nb) send_all(fd, b, nb);
}

inline msg_hdr recv_hdr(int fd) {
    msg_hdr h;
    recv_all(fd, &h, sizeof(h));
    if (h.magic != MAGIC) throw std::runtime_error("bad message magic (not a split-prefill tail peer?)");
    return h;
}

inline void tune_socket(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    for (int sz : { 8 << 20, 4 << 20, 2 << 20 }) {
        if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz)) == 0) break;
    }
    for (int sz : { 8 << 20, 4 << 20, 2 << 20 }) {
        if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz)) == 0) break;
    }
}

// Who may connect. The protocol has no authentication, so by default only this machine may (loopback: the e2e tests, a tail
// on the same Mac); SPT_ALLOW_REMOTE=1 lifts that for a deliberate setup. The Backburner app replaces it with its USB-cable
// check (set_accept_filter). Called after accept(), before anything is read; `why` says who it was and why.
using accept_filter_fn = std::function<bool(int fd, std::string & why)>;
inline bool accept_loopback_only(int fd, std::string & why) {
    sockaddr_storage p = {};
    socklen_t pl = sizeof p;
    if (getpeername(fd, (sockaddr *) &p, &pl) != 0) { why = "no peer address"; return false; }
    bool lo = false;
    char ip[INET6_ADDRSTRLEN] = "?";
    if (p.ss_family == AF_INET) {
        const in_addr a = ((sockaddr_in *) &p)->sin_addr;
        lo = (ntohl(a.s_addr) >> 24) == 127;
        inet_ntop(AF_INET, &a, ip, sizeof ip);
    } else if (p.ss_family == AF_INET6) {
        const in6_addr & a = ((sockaddr_in6 *) &p)->sin6_addr;
        lo = IN6_IS_ADDR_LOOPBACK(&a) || (IN6_IS_ADDR_V4MAPPED(&a) && a.s6_addr[12] == 127);
        inet_ntop(AF_INET6, &a, ip, sizeof ip);
    }
    const char * e = getenv("SPT_ALLOW_REMOTE");
    if (lo || (e && atoi(e) != 0)) { why = std::string(ip) + (lo ? ": loopback" : ": SPT_ALLOW_REMOTE"); return true; }
    why = std::string(ip) + ": not loopback (SPT_ALLOW_REMOTE=1 allows it)";
    return false;
}

// ---- server ----------------------------------------------------------------------------------

// Live status for a UI (the Sidecar screen). Plain fields guarded by `mu`.
struct server_status {
    std::mutex  mu;
    std::string state = "starting";  // starting | no model | loading | ready | connected | busy | error
    std::string detail;
    std::string model_desc;
    uint64_t    chunks = 0, tokens = 0, sessions = 0;
    double      last_chunk_ms = 0, last_tok_s = 0;
    float       load_progress = 0;   // 0..1 while the weights load (state "loading"), 1 after
    uint64_t    model_bytes = 0;
    std::string last_sync;           // the last mirror SYNC's timing (recv / on_work / state load)

    void set(const std::string & s, const std::string & d = "") { std::lock_guard<std::mutex> lk(mu); state = s; detail = d; }
};

class tail_server {
public:
    using log_fn = std::function<void(const std::string &)>;

    // called on every HELLO once the context is ready (e.g. to switch the FFN offload on or off for the next chunks);
    // returns a note appended to the model description the client sees ("" = none)
    std::function<std::string(llama_context *, int n_layer_tail)> on_hello;

    // who may connect (default: loopback only, see accept_loopback_only)
    void set_accept_filter(accept_filter_fn f) { accept_ok_ = std::move(f); }

    // called before every message that runs or loads this tail's memory (SYNC, CHUNK, STATE, STATE_RANGE): lets the host make
    // room first (Sidecar unloads its ANE page models: the tail's weights are wired again by the next graph, and wired memory
    // is what iOS kills on)
    std::function<void()> on_work;

    // tmp_dir: where the state blob is staged before it is sent (see send_state)
    tail_server(std::string model_path, server_status * st, log_fn log, std::string tmp_dir = "/tmp")
        : path_(std::move(model_path)), tmp_dir_(std::move(tmp_dir)), st_(st), log_(std::move(log)) {}

    ~tail_server() {
        if (ctx_)   llama_free(ctx_);
        if (model_) llama_model_free(model_);
    }

    // Loads the model (once). Returns an error string or "".
    std::string load() {
        if (model_) return "";
        FILE * f = fopen(path_.c_str(), "rb");
        if (!f) { status("no model", path_); return "no tail model at " + path_; }
        fseeko(f, 0, SEEK_END); file_bytes_ = (uint64_t) ftello(f); fclose(f);

        {
            std::lock_guard<std::mutex> lk(st_->mu);
            st_->load_progress = 0; st_->model_bytes = file_bytes_;
        }
        status("loading", path_);
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        mp.progress_callback_user_data = st_;
        mp.progress_callback = [](float p, void * ud) {   // for the Sidecar screen's load animation
            auto * st = (server_status *) ud;
            std::lock_guard<std::mutex> lk(st->mu);
            st->load_progress = p;
            return true;
        };
        model_ = llama_model_load_from_file(path_.c_str(), mp);
        if (!model_) { status("error", "model load failed"); return "failed to load " + path_; }

        char buf[256];
        layer_start_  = llama_model_meta_val_str(model_, "split.layer_start",  buf, sizeof(buf)) > 0 ? (uint32_t) atoi(buf) : 0;
        no_head_      = llama_model_meta_val_str(model_, "split.no_head",      buf, sizeof(buf)) > 0;
        n_layer_full_ = llama_model_meta_val_str(model_, "split.n_layer_full", buf, sizeof(buf)) > 0 ? (uint32_t) atoi(buf) : 0;
        llama_model_desc(model_, buf, sizeof(buf));
        desc_ = buf;
        if (llama_model_meta_val_str(model_, "general.name", buf, sizeof(buf)) > 0) desc_ = std::string(buf) + " | " + desc_;
        if (layer_start_ == 0) { status("error", "not a split TAIL GGUF (no split.layer_start)"); return "not a split TAIL GGUF"; }
        {
            std::lock_guard<std::mutex> lk(st_->mu);
            st_->model_desc = desc_;
        }
        status("ready", desc_);
        log_("tail model: " + desc_ + ", layers [" + std::to_string(layer_start_) + ", " + std::to_string(n_layer_full_) + ")");
        return "";
    }

    // Blocking accept loop on 0.0.0.0:port (each connection then passes the accept filter). Returns only on a listen error.
    std::string serve(int port) {
        int srv = socket(AF_INET, SOCK_STREAM, 0);
        if (srv < 0) return std::string("socket: ") + strerror(errno);
        int one = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((uint16_t) port);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(srv, (sockaddr *) &addr, sizeof(addr)) != 0 || listen(srv, 2) != 0) {
            std::string e = std::string("tail listen :") + std::to_string(port) + ": " + strerror(errno);
            close(srv);
            return e;
        }
        log_("tail worker listening on :" + std::to_string(port));
        while (true) {
            int fd = accept(srv, nullptr, nullptr);
            if (fd < 0) {   // a USB replug fails accept: keep listening (breaking here left the port dead until relaunch)
                if (errno != EINTR) { log_(std::string("tail accept: ") + strerror(errno) + " (retrying)"); usleep(100000); }
                continue;
            }
            {
                std::string why;
                if (!(accept_ok_ ? accept_ok_(fd, why) : accept_loopback_only(fd, why))) {
                    log_("refused a connection from " + why);
                    close(fd);
                    continue;
                }
            }
            tune_socket(fd);
            {
                std::lock_guard<std::mutex> lk(st_->mu);
                st_->sessions++;
            }
            status("connected");
            try {
                handle(fd);
            } catch (const std::exception & e) {
                log_(std::string("tail session ended: ") + e.what());
            }
            close(fd);
            status(model_ ? "ready" : "no model", model_ ? desc_ : path_);
        }
        close(srv);
        return "tail accept failed";
    }

private:
    std::string path_, tmp_dir_;
    server_status * st_;
    log_fn log_;
    llama_model   * model_ = nullptr;
    llama_context * ctx_   = nullptr;
    uint32_t ctx_n_ctx_ = 0, ctx_n_ub_ = 0;
    int32_t  ctx_type_k_ = -1, ctx_type_v_ = -1;
    uint32_t ctx_fa_ = 0, ctx_replay_ = 0;
    uint64_t session_ = 0;
    uint32_t n_valid_ = 0;            // the mirror holds positions [0, n_valid_) of the Mac's conversation
    std::vector<int32_t> taps_;
    uint32_t layer_start_ = 0, n_layer_full_ = 0;
    accept_filter_fn accept_ok_;
    bool     no_head_ = false;        // split-gguf.py --no-head: no token_embd / output, never logits
    uint64_t file_bytes_ = 0;
    std::string desc_;
    std::vector<uint8_t>     in_;
    std::vector<float>       f32_;

    void status(const std::string & s, const std::string & d = "") { st_->set(s, d); }

    static void send_err(int fd, const std::string & e) { send_msg(fd, MSG_ERR, e.data(), e.size()); }

    void handle(int fd) {
        while (true) {
            msg_hdr h = recv_hdr(fd);
            const auto t_hdr = std::chrono::steady_clock::now();
            in_.resize(h.len);
            if (h.len) recv_all(fd, in_.data(), h.len);
            const auto t_in = std::chrono::steady_clock::now();
            auto ms_since = [](std::chrono::steady_clock::time_point a) {
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(); };

            switch (h.type) {
            case MSG_HELLO: {
                if (h.len != sizeof(hello_req) && h.len != sizeof(hello_req2)) { send_err(fd, "bad HELLO size"); return; }
                const bool v2 = h.len == sizeof(hello_req2);
                hello_req2 q2 = {};
                q2.type_k = GGML_TYPE_F16; q2.type_v = GGML_TYPE_F16;
                memcpy(&q2, in_.data(), h.len);
                const hello_req & q = q2.base;
                std::string err = load();
                if (err.empty() && ((q.proto != PROTO_VERSION && q.proto != 1) || q.state_format != STATE_FORMAT)) {
                    err = "version mismatch: client proto " + std::to_string(q.proto) + "/state " + std::to_string(q.state_format) +
                          ", phone proto " + std::to_string(PROTO_VERSION) + "/state " + std::to_string(STATE_FORMAT) +
                          " - rebuild both from the same llama.cpp tree";
                }
                if (err.empty() && (q.L != layer_start_ || q.n_layer_full != n_layer_full_ ||
                                    (int) q.n_embd != llama_model_n_embd(model_) ||
                                    (int) q.n_vocab != llama_vocab_n_tokens(llama_model_get_vocab(model_)))) {
                    err = "tail model mismatch: phone has layers [" + std::to_string(layer_start_) + ", " + std::to_string(n_layer_full_) +
                          ") n_embd " + std::to_string(llama_model_n_embd(model_)) + ", client wants L=" + std::to_string(q.L) +
                          " of " + std::to_string(q.n_layer_full) + " n_embd " + std::to_string(q.n_embd);
                }
                const bool same_ctx = ctx_ && ctx_n_ctx_ == q.n_ctx && ctx_n_ub_ == q.n_ubatch && ctx_type_k_ == q2.type_k &&
                                      ctx_type_v_ == q2.type_v && ctx_fa_ == q2.flash_attn && ctx_replay_ == q2.n_rs_replay;
                if (err.empty() && !same_ctx) {
                    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
                    auto cp = llama_context_default_params();
                    cp.n_ctx = q.n_ctx; cp.n_batch = q.n_ubatch; cp.n_ubatch = q.n_ubatch;
                    cp.n_seq_max = 1; cp.no_perf = true;
                    cp.type_k = (ggml_type) q2.type_k; cp.type_v = (ggml_type) q2.type_v;
                    cp.n_rs_replay = q2.n_rs_replay;
                    if (q2.flash_attn) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                    ctx_ = llama_init_from_model(model_, cp);
                    if (!ctx_) err = "tail context creation failed (n_ctx " + std::to_string(q.n_ctx) + ", out of memory?)";
                    else { ctx_n_ctx_ = q.n_ctx; ctx_n_ub_ = q.n_ubatch; ctx_type_k_ = q2.type_k; ctx_type_v_ = q2.type_v;
                           ctx_fa_ = q2.flash_attn; ctx_replay_ = q2.n_rs_replay; }
                }
                if (!err.empty()) { status("error", err); send_err(fd, err); return; }
                // the mirror survives a reconnect only for the same session with the same settings
                if (!(v2 && q2.keep && same_ctx && q2.session == session_)) {
                    llama_memory_clear(llama_get_memory(ctx_), true);
                    n_valid_ = 0;
                }
                session_ = v2 ? q2.session : 0;
                for (int32_t t : taps_) llama_set_embeddings_layer_inp(ctx_, (uint32_t) t, false);
                taps_.clear();
                if (v2) {
                    for (uint32_t i = 0; i < std::min<uint32_t>(q2.n_taps, 4); i++) {
                        taps_.push_back(q2.taps[i]);
                        llama_set_embeddings_layer_inp(ctx_, (uint32_t) q2.taps[i], true);
                    }
                }

                const std::string note = on_hello ? on_hello(ctx_, llama_model_n_layer(model_)) : std::string();
                if (!note.empty() && note[0] == '!') {   // the hook refuses this session
                    status("error", note.substr(1)); send_err(fd, note.substr(1)); return;
                }
                hello_rep r = {};
                r.proto = PROTO_VERSION; r.state_format = STATE_FORMAT;
                r.layer_start = layer_start_; r.n_layer_full = n_layer_full_; r.n_layer = (uint32_t) llama_model_n_layer(model_);
                r.n_embd = (uint32_t) llama_model_n_embd(model_);
                r.n_vocab = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_));
                r.n_ctx = ctx_n_ctx_; r.file_bytes = file_bytes_;
                snprintf(r.desc, sizeof(r.desc), "%s%s", desc_.c_str(), note.c_str());
                if (v2) {
                    hello_rep2 r2 = { r, n_valid_, session_ };
                    send_msg(fd, MSG_HELLO_OK, &r2, sizeof(r2));
                } else {
                    send_msg(fd, MSG_HELLO_OK, &r, sizeof(r));
                }
                status("connected", "n_ctx " + std::to_string(ctx_n_ctx_) + ", ubatch " + std::to_string(ctx_n_ub_));
                break;
            }
            case MSG_RESET: {
                if (ctx_) llama_memory_clear(llama_get_memory(ctx_), true);
                n_valid_ = 0;
                send_msg(fd, MSG_RESET_OK, nullptr, 0);
                break;
            }
            case MSG_CHUNK: {
                if (!ctx_) { send_err(fd, "CHUNK before HELLO"); return; }
                if (on_work) on_work();
                if (h.len < sizeof(chunk_req)) { send_err(fd, "bad CHUNK size"); return; }
                chunk_req q; memcpy(&q, in_.data(), sizeof(q));
                const int n_embd = llama_model_n_embd(model_);
                const size_t el = q.dtype == DT_F16 ? 2 : 4;
                if (q.n_tok == 0 || q.n_tok > ctx_n_ub_ || h.len != sizeof(q) + (size_t) q.n_tok * n_embd * el) {
                    send_err(fd, "bad CHUNK payload"); return;
                }
                status("busy");
                const auto t0 = std::chrono::steady_clock::now();
                const float * resid = nullptr;
                const uint8_t * data = in_.data() + sizeof(q);
                if (q.dtype == DT_F16) {
                    f32_.resize((size_t) q.n_tok * n_embd);
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *) data, f32_.data(), (int64_t) f32_.size());
                    resid = f32_.data();
                } else {
                    f32_.resize((size_t) q.n_tok * n_embd);
                    memcpy(f32_.data(), data, f32_.size() * sizeof(float));
                    resid = f32_.data();
                }
                std::string err;
                if ((uint32_t) q.pos0 != n_valid_) {
                    // v1 clients prefill from 0 after RESET; v2 clients chunk right after the mirror
                    if (q.pos0 != 0 || n_valid_ != 0) { send_err(fd, "CHUNK pos0 " + std::to_string(q.pos0) + " != mirror end " + std::to_string(n_valid_)); return; }
                }
                // a head-less tail (split-gguf.py --no-head) has no logits to give: the chunk is acked without them
                const float * lg = run_chunk(resid, (int) q.n_tok, n_embd, q.pos0, (q.want_logits & 1) != 0 && !no_head_, err);
                if (!err.empty()) { status("error", err); send_err(fd, err); return; }
                n_valid_ = (uint32_t) q.pos0 + q.n_tok;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

                chunk_rep r = {};
                r.status = 0; r.n_tok = q.n_tok; r.compute_ms = (float) ms;
                r.n_logits = lg ? (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model_)) : 0;
                std::vector<ggml_fp16_t> tap16;
                if ((q.want_logits & 2) && !taps_.empty()) {
                    const size_t per = (size_t) q.n_tok * n_embd;
                    tap16.resize(per * taps_.size());
                    for (size_t t = 0; t < taps_.size(); t++) {
                        const float * rows = llama_get_embeddings_layer_inp(ctx_, (uint32_t) taps_[t]);
                        if (!rows) { send_err(fd, "tap layer " + std::to_string(taps_[t]) + " not captured"); return; }
                        ggml_fp32_to_fp16_row(rows, tap16.data() + t * per, (int64_t) per);
                    }
                }
                if (lg) {
                    std::vector<uint8_t> tail_payload((size_t) r.n_logits * sizeof(float) + tap16.size() * 2);
                    memcpy(tail_payload.data(), lg, (size_t) r.n_logits * sizeof(float));
                    if (!tap16.empty()) memcpy(tail_payload.data() + (size_t) r.n_logits * sizeof(float), tap16.data(), tap16.size() * 2);
                    send_msg(fd, MSG_CHUNK_ACK, &r, sizeof(r), tail_payload.data(), tail_payload.size());
                } else {
                    send_msg(fd, MSG_CHUNK_ACK, &r, sizeof(r), tap16.data(), tap16.size() * 2);
                }
                {
                    std::lock_guard<std::mutex> lk(st_->mu);
                    st_->chunks++; st_->tokens += q.n_tok; st_->last_chunk_ms = ms;
                    st_->last_tok_s = ms > 0 ? q.n_tok * 1000.0 / ms : 0;
                    st_->state = "busy"; st_->detail = "";
                }
                break;
            }
            case MSG_STATE: {
                if (on_work) on_work();
                if (!ctx_) { send_err(fd, "STATE before HELLO"); return; }
                send_state(fd);
                break;
            }
            case MSG_TRIM: {
                if (!ctx_ || h.len != 4) { send_err(fd, "bad TRIM"); return; }
                uint32_t n; memcpy(&n, in_.data(), 4);
                if (n < n_valid_) {
                    // the recurrent state can't be rewound: v2 resets the mirror and the client pushes it again
                    llama_memory_clear(llama_get_memory(ctx_), true);
                    n_valid_ = 0;
                }
                send_msg(fd, MSG_OK, &n_valid_, 4);
                break;
            }
            case MSG_SYNC: {
                const double recv_ms = std::chrono::duration<double, std::milli>(t_in - t_hdr).count();
                if (on_work) on_work();
                const double work_ms = ms_since(t_in);
                if (!ctx_ || h.len < sizeof(sync_req)) { send_err(fd, "bad SYNC"); return; }
                sync_req q; memcpy(&q, in_.data(), sizeof(q));
                status("busy", "sync");
                llama_state_filter_set(0, 0x7fffffff, 0, 0x7fffffff, /*kv append*/ 1, /*rs replace*/ 0, (q.flags & 1) != 0, (q.flags & 2) != 0);
                const size_t n = llama_state_seq_set_data(ctx_, in_.data() + sizeof(q), h.len - sizeof(q), 0);
                llama_state_filter_clear();
                if (n != h.len - sizeof(q)) {
                    llama_memory_clear(llama_get_memory(ctx_), true);
                    n_valid_ = 0;
                    send_err(fd, "SYNC: state load failed (mirror reset)");
                    return;
                }
                n_valid_ = q.n_valid_after;
                send_msg(fd, MSG_OK, &n_valid_, 4);
                char tb[160];
                snprintf(tb, sizeof tb, " (%.1f MB: recv %.0f ms, on_work %.0f ms, load %.0f ms)", (h.len - sizeof(q)) / 1e6, recv_ms,
                         work_ms, ms_since(t_in) - work_ms);
                status("connected", "mirror " + std::to_string(n_valid_) + " tokens" + tb);
                { std::lock_guard<std::mutex> lk(st_->mu); st_->last_sync = tb + 2; }
                break;
            }
            case MSG_STATE_RANGE: {
                if (on_work) on_work();
                if (!ctx_ || h.len != sizeof(range_req)) { send_err(fd, "bad STATE_RANGE"); return; }
                range_req q; memcpy(&q, in_.data(), sizeof(q));
                llama_state_filter_set(q.p0, q.p1, 0, 0x7fffffff, 0, 0, (q.flags & 1) != 0, (q.flags & 2) != 0);
                try { send_state(fd); } catch (...) { llama_state_filter_clear(); throw; }
                llama_state_filter_clear();
                break;
            }
            case MSG_PING: {
                uint32_t n_rep = 0;
                if (h.len >= 4) memcpy(&n_rep, in_.data(), 4);
                if (n_rep > (64u << 20)) { send_err(fd, "PING reply too large"); return; }
                if (in_.size() < n_rep) in_.resize(n_rep);
                send_msg(fd, MSG_PONG, in_.data(), n_rep);
                break;
            }
            case MSG_BYE:
                return;
            default:
                send_err(fd, "unknown message type " + std::to_string(h.type));
                return;
            }
        }
    }

    // The seq-state blob (27B, L=40, 8k tokens: 260 MB) is staged in a file instead of a heap buffer:
    // on the phone the tail weights already sit near the per-app memory limit, and a 260 MB
    // std::vector there ends in bad_alloc. llama_state_seq_save_file streams layer by layer; its
    // file is [u32 magic, u32 version, u32 n_token=0] + exactly what llama_state_seq_get_data writes
    // after its own [u32 io magic, i32 seq_id] prefix, so the wire blob is identical.
    void send_state(int fd) {
        const std::string tmp = tmp_dir_ + "/spt-state.bin";
        const size_t n = llama_state_seq_save_file(ctx_, tmp.c_str(), 0, nullptr, 0);
        FILE * f = n > 12 ? fopen(tmp.c_str(), "rb") : nullptr;
        if (!f) { unlink(tmp.c_str()); send_err(fd, "state save failed (" + tmp + ")"); throw std::runtime_error("state save failed"); }
        fseeko(f, 12, SEEK_SET);
        const uint32_t io_magic = 0xaf143cd8u;
        const int32_t  seq_id   = 0;
        msg_hdr h = { MAGIC, MSG_STATE_DATA, (uint64_t) (n - 12 + 8) };
        send_all(fd, &h, sizeof(h));
        send_all(fd, &io_magic, 4);
        send_all(fd, &seq_id, 4);
        std::vector<char> buf(4 << 20);
        size_t left = n - 12;
        while (left > 0) {
            const size_t r = fread(buf.data(), 1, std::min(left, buf.size()), f);
            if (r == 0) { fclose(f); unlink(tmp.c_str()); throw std::runtime_error("state file short read"); }
            send_all(fd, buf.data(), r);
            left -= r;
        }
        fclose(f);
        unlink(tmp.c_str());
        status("connected", "sent state " + std::to_string(n >> 20) + " MB");
    }

    // same batch construction as split-prefill.cpp local_tail
    const float * run_chunk(const float * resid, int n_tok, int n_embd, llama_pos pos0, bool want_logits, std::string & err) {
        const int rope = llama_model_rope_type(model_);
        const int n_pos = (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
        llama_batch b = llama_batch_init(n_tok, n_embd, 1);
        free(b.pos);
        b.pos = (llama_pos *) malloc(sizeof(llama_pos) * n_tok * n_pos);
        for (int j = 0; j < n_pos; ++j) for (int i = 0; i < n_tok; ++i) b.pos[j*n_tok + i] = pos0 + i;
        memcpy(b.embd, resid, sizeof(float) * (size_t) n_tok * n_embd);
        for (int i = 0; i < n_tok; ++i) { b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0; }
        b.logits[n_tok - 1] = want_logits;
        b.n_tokens = n_tok;
        const int rc = llama_decode(ctx_, b);
        llama_batch_free(b);
        if (rc != 0) { err = "tail llama_decode failed: " + std::to_string(rc); return nullptr; }
        if (llama_ffn_offload_failed(ctx_)) { err = "tail FFN offload failed (see the phone log)"; return nullptr; }
        if (!want_logits) { llama_synchronize(ctx_); return nullptr; }
        return llama_get_logits_ith(ctx_, n_tok - 1);
    }
};

} // namespace spt
