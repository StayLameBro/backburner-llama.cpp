// tail-client.h - Mac-side client of the split-prefill TAIL worker, protocol v2 (tail-server.h).
//
// The phone keeps a MIRROR of its layers' state for positions [0, n_valid) of the Mac's conversation. To split-prefill an
// APPEND of N tokens at depth D:
//   1. trim(D) if the Mac's conversation changed below the mirror end (v2: resets the mirror), then
//      sync_state(...) the Mac's filtered state for [n_valid, D): KV pieces (flags 1), then the recurrent state (flags 2)
//   2. submit_chunk(...) the residual entering layer L for each ubatch of the append (pipelined, returns at once)
//   3. finish(...): waits for every ack; returns the last chunk's logits and the tap-layer rows
//   4. state_range(D, D+N, 3): the phone's rows of the append + its recurrent state, for the Mac to overwrite in place
// Every call throws std::runtime_error on a link or phone error; the caller falls back to Mac-only.
#pragma once

#include "tail-server.h"

#include <condition_variable>
#include <deque>
#include <thread>

namespace spt {

class tail_client {
public:
    hello_rep2 hello = {};
    bool resid_f32 = false;           // f16 halves the link bytes and is token-identical; f32 is bit-exact (gates)
    std::vector<double> tail_ms;      // per chunk phone compute time
    double bytes_sent = 0;
    int recv_timeout_s = 0;           // > 0: a reply (HELLO, ack, logits) slower than this fails the link (0 = wait forever)

    ~tail_client() { close_link(); }

    bool connect(const std::string & host, int port, const hello_req2 & q, hello_rep2 & rep, std::string & err) {
        try {
            fd_ = socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in a = {};
            a.sin_family = AF_INET;
            a.sin_port = htons((uint16_t) port);
            if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) throw std::runtime_error("bad IPv4 address " + host);
            tune_socket(fd_);
            if (recv_timeout_s > 0) {
                timeval tv = { recv_timeout_s, 0 };
                setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            }
            if (::connect(fd_, (sockaddr *) &a, sizeof(a)) != 0) {
                throw std::runtime_error("cannot connect to tail worker at " + host + ":" + std::to_string(port) + ": " + strerror(errno));
            }
            send_msg(fd_, MSG_HELLO, &q, sizeof(q));
            const auto p = recv_reply(MSG_HELLO_OK);
            if (p.size() != sizeof(hello)) throw std::runtime_error("bad HELLO reply (phone runs protocol v1? rebuild Sidecar)");
            memcpy(&hello, p.data(), sizeof(hello));
            rep = hello;
            n_valid_ = hello.n_valid;
            n_taps_ = q.n_taps;
            n_embd_ = (int) q.base.n_embd;
            stop_ = false;
            writer_ = std::thread([this] { write_loop(); });
            reader_ = std::thread([this] { read_loop(); });
            return true;
        } catch (const std::exception & e) {
            err = e.what();
            close_link();
            return false;
        }
    }

    uint32_t n_valid() const { return n_valid_; }

    void trim(uint32_t n) {
        idle_check();
        send_msg(fd_, MSG_TRIM, &n, 4);
        n_valid_ = u32_reply();
    }

    void sync_state(const std::vector<uint8_t> & blob, uint32_t flags, uint32_t n_valid_after) {
        idle_check();
        sync_req q = { flags, n_valid_after };
        send_msg(fd_, MSG_SYNC, &q, sizeof(q), blob.data(), blob.size());
        bytes_sent += (double) blob.size();
        n_valid_ = u32_reply();
    }

    void submit_chunk(const float * resid, int n_tok, int n_embd, llama_pos pos0, bool want_logits, bool want_taps) {
        check();
        chunk_req q = { pos0, (uint32_t) n_tok, resid_f32 ? DT_F32 : DT_F16, (want_logits ? 1u : 0u) | (want_taps ? 2u : 0u) };
        const size_t n = (size_t) n_tok * n_embd;
        std::vector<uint8_t> msg(sizeof(q) + n * (resid_f32 ? 4 : 2));
        memcpy(msg.data(), &q, sizeof(q));
        if (resid_f32) memcpy(msg.data() + sizeof(q), resid, n * sizeof(float));
        else ggml_fp32_to_fp16_row(resid, (ggml_fp16_t *) (msg.data() + sizeof(q)), (int64_t) n);
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push_back({ std::move(msg), n_tok, want_logits, want_taps });
            n_submitted_++;
        }
        n_valid_ = (uint32_t) pos0 + n_tok;
        cv_.notify_all();
    }

    bool finish(std::vector<float> & last_logits, std::vector<ggml_fp16_t> & taps, std::string & err) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !error_.empty() || n_acked_ == n_submitted_; });
        if (!error_.empty()) { err = error_; return false; }
        last_logits = std::move(logits_);
        taps = std::move(taps_);
        logits_.clear(); taps_.clear();
        return true;
    }

    // drop the link now, chunks in flight or not: shutdown() wakes a reader blocked in recv
    void abort_link() {
        if (fd_ >= 0) shutdown(fd_, SHUT_RDWR);
        fail("link aborted");
        close_link();
    }

    std::vector<uint8_t> state_range(int p0, int p1, uint32_t flags) {
        idle_check();
        range_req q = { p0, p1, flags };
        send_msg(fd_, MSG_STATE_RANGE, &q, sizeof(q));
        return recv_reply(MSG_STATE_DATA);
    }

private:
    struct item { std::vector<uint8_t> msg; int n_tok; bool logits, taps; };

    int fd_ = -1;
    uint32_t n_valid_ = 0, n_taps_ = 0;
    int n_embd_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<item> queue_;          // waiting for the writer
    std::deque<item> inflight_;       // sent, waiting for the ack (the reader needs n_tok / taps to size the reply)
    int n_submitted_ = 0, n_acked_ = 0;
    bool stop_ = true;
    std::string error_;
    std::vector<float> logits_;
    std::vector<ggml_fp16_t> taps_;
    std::thread writer_, reader_;

    void close_link() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (writer_.joinable()) writer_.join();
        if (reader_.joinable()) reader_.join();
        if (fd_ >= 0) {
            try { send_msg(fd_, MSG_BYE, nullptr, 0); } catch (...) {}
            close(fd_);
            fd_ = -1;
        }
    }

    void fail(const std::string & e) {
        std::lock_guard<std::mutex> lk(mu_);
        if (error_.empty()) error_ = e;
        stop_ = true;
        cv_.notify_all();
    }

    void check() { std::lock_guard<std::mutex> lk(mu_); if (!error_.empty()) throw std::runtime_error(error_); }

    // synchronous calls share the socket with the pipelined chunks: only between finish() and the next submit
    void idle_check() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!error_.empty()) throw std::runtime_error(error_);
        if (n_acked_ != n_submitted_ || !queue_.empty()) throw std::runtime_error("tail_client: sync call while chunks are in flight");
    }

    std::vector<uint8_t> recv_reply(uint32_t want) {
        const msg_hdr h = recv_hdr(fd_);
        std::vector<uint8_t> p(h.len);
        if (h.len) recv_all(fd_, p.data(), h.len);
        if (h.type == MSG_ERR) throw std::runtime_error("tail worker: " + std::string(p.begin(), p.end()));
        if (h.type != want) throw std::runtime_error("unexpected reply type " + std::to_string(h.type));
        return p;
    }

    uint32_t u32_reply() {
        const auto p = recv_reply(MSG_OK);
        uint32_t v = 0;
        if (p.size() >= 4) memcpy(&v, p.data(), 4);
        return v;
    }

    void write_loop() {
        while (true) {
            item it;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
                if (stop_) return;
                it = std::move(queue_.front());
                queue_.pop_front();
            }
            try {
                send_msg(fd_, MSG_CHUNK, it.msg.data(), it.msg.size());
            } catch (const std::exception & e) { fail(std::string("send chunk: ") + e.what()); return; }
            bytes_sent += (double) it.msg.size();
            {
                std::lock_guard<std::mutex> lk(mu_);
                it.msg.clear();
                inflight_.push_back(std::move(it));
            }
            cv_.notify_all();
        }
    }

    void read_loop() {
        while (true) {
            item it;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || !inflight_.empty(); });
                if (stop_) return;
                it = std::move(inflight_.front());
                inflight_.pop_front();
            }
            try {
                const msg_hdr h = recv_hdr(fd_);
                if (h.type == MSG_ERR) {
                    std::string e(h.len, '\0'); recv_all(fd_, e.data(), h.len);
                    fail("tail worker: " + e); return;
                }
                if (h.type != MSG_CHUNK_ACK || h.len < sizeof(chunk_rep)) { fail("bad CHUNK reply"); return; }
                chunk_rep r; recv_all(fd_, &r, sizeof(r));
                std::vector<float> lg(r.n_logits);
                if (r.n_logits) recv_all(fd_, lg.data(), lg.size() * sizeof(float));
                const size_t n_tap = h.len - sizeof(r) - (size_t) r.n_logits * sizeof(float);
                std::vector<ggml_fp16_t> tp(n_tap / 2);
                if (n_tap) recv_all(fd_, tp.data(), n_tap);
                if (it.taps && tp.size() != (size_t) n_taps_ * it.n_tok * n_embd_) { fail("tap rows missing in CHUNK reply"); return; }
                std::lock_guard<std::mutex> lk(mu_);
                if (r.n_logits) logits_ = std::move(lg);
                taps_.insert(taps_.end(), tp.begin(), tp.end());
                tail_ms.push_back(r.compute_ms);
                n_acked_++;
            } catch (const std::exception & e) { fail(std::string("recv ack: ") + e.what()); return; }
            cv_.notify_all();
        }
    }
};

} // namespace spt
