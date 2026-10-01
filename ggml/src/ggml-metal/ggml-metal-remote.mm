// Phone-held KV (infernet): see ggml-metal-remote.h and docs/phone-kv-262k.md.
//
// One connection to the phone-attn server (the real phone over USB, or `pa-tool serve` on this Mac as a loopback).
// More than one phone (LLAMA_KV_REMOTE=ip1:port,ip2:port): the old keys are dealt to the phones in blocks of
// GGML_METAL_REMOTE_BLOCK positions (default 4096) in turn, so each holds an equal share at any depth. Every attention call
// goes to all phones at once and their partials are merged here before the GPU sees them (exact: softmax partials combine
// by their log-sum-exp). The phones and the GPU side don't change.
// The KV cache sends old keys with _append between graphs; during a graph, one thread waits for each tagged attention
// op's "Q is ready" event value, sends Q, receives the phone's partial and signals "partial is ready". Jobs run in
// event-value order (= graph node order = GPU execution order), strictly one at a time.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ggml-metal-remote.h"
#include "ggml-metal-coattn.h"   // GGML_METAL_COATTN_SM / _FLOATS: the partial layout the scatter kernel reads
#include "phone-attn.h"

#include <pthread.h>
#include <pthread/qos.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::mutex              r_mu;                 // the connection: one request at a time
pa::client *            r_cli    = nullptr;
int                     r_nlayer = 0;
int                     r_nkv    = 0;
size_t                  r_rs     = 0;
std::vector<int32_t>    r_il;                 // phone layer i -> model layer
std::vector<int32_t>    r_map;                // model layer -> phone layer (-1)
std::atomic<uint32_t>   r_held{0};

id<MTLSharedEvent>      r_ev[2] = { nil, nil };   // [1]: the second half of a pipelined prefill ubatch
id<MTLBuffer>           r_part[2] = { nil, nil };
id<MTLBuffer>           r_big[4]  = { nil, nil, nil, nil };   // ATTN_BIG partials: layer parity | half << 1
uint32_t                r_version = 0;              // the phone's protocol version (3: ATTN_BIG)

std::vector<pa::client *> r_more;          // phones 2..N (r_cli is phone 1)
std::vector<uint32_t>   r_held_p;          // keys each phone holds (every layer)
uint32_t                r_block = 4096;

size_t n_phones() { return r_cli ? 1 + r_more.size() : 0; }
pa::client * cli(size_t p) { return p == 0 ? r_cli : r_more[p - 1]; }

// how many of the positions [0, n) phone p holds
uint32_t local_count(size_t p, uint32_t n) {
    const uint32_t cyc = r_block*(uint32_t) n_phones();
    const uint32_t rem = n % cyc;
    const uint32_t lo  = (uint32_t) p*r_block;
    return (n/cyc)*r_block + (rem > lo ? std::min(rem - lo, r_block) : 0);
}

std::mutex              r_qmu;
std::vector<ggml_metal_remote_job> r_q;
pthread_once_t          r_once = PTHREAD_ONCE_INIT;

uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

// GGML_METAL_REMOTE_CHECK=1 (debug): keep a dequantized copy of what the phone holds and check each partial of KV head 0
// against an fp64 reference computed from the f32 Q
int check_on() {
    static int v = -1;
    if (v < 0) v = getenv("GGML_METAL_REMOTE_CHECK") ? atoi(getenv("GGML_METAL_REMOTE_CHECK")) : 0;
    return v;
}
int r_is_q8 = 1;
size_t r_hb = 0;
std::vector<std::vector<float>> r_chk_k, r_chk_v;   // per layer: [key][256] of KV head 0

float deq_head0(const uint8_t * row, int d) {
    if (r_is_q8 == 0) return pa::h2f(((const uint16_t *) row)[d]);
    if (r_is_q8 == 2) {
        const uint8_t * blk = row + (d/32)*18;
        uint16_t dh; memcpy(&dh, blk, 2);
        const uint8_t b = blk[2 + (d%32)%16];
        return pa::h2f(dh)*((d%32 < 16 ? (b & 15) : (b >> 4)) - 8);
    }
    const uint8_t * blk = row + (d/32)*34;
    uint16_t dh; memcpy(&dh, blk, 2);
    return pa::h2f(dh)*(float) (int8_t) blk[2 + d%32];
}

void check_job(const ggml_metal_remote_job & j, const std::vector<uint16_t> & O, const std::vector<float> & lse) {
    const auto & K = r_chk_k[j.layer];
    const auto & V = r_chk_v[j.layer];
    const size_t nk = K.size()/pa::HD;
    double e_o = 0, e_l = 0;
    std::vector<double> sc(nk), acc(pa::HD);
    for (int t = 0; t < j.n_tok; t++) {
        const float * q = (const float *) ((const char *) j.q + (size_t) t*j.q_nb1);   // head 0 = KV head 0, g = 0
        double mx = -1e300;
        for (size_t k = 0; k < nk; k++) {
            double a = 0;
            for (int d = 0; d < pa::HD; d++) a += (double) q[d]*K[k*pa::HD + d];
            sc[k] = a*j.scale; mx = std::max(mx, sc[k]);
        }
        double L = 0; std::fill(acc.begin(), acc.end(), 0.0);
        for (size_t k = 0; k < nk; k++) { const double p = exp(sc[k] - mx); L += p; for (int d = 0; d < pa::HD; d++) acc[d] += p*V[k*pa::HD + d]; }
        double om = 1e-30; for (int d = 0; d < pa::HD; d++) om = std::max(om, fabs(acc[d]/L));
        const int r = t;   // g = 0
        for (int d = 0; d < pa::HD; d++) e_o = std::max(e_o, fabs(pa::h2f(O[(size_t) r*pa::HD + d]) - acc[d]/L)/om);
        e_l = std::max(e_l, fabs(lse[r] - (mx + log(L))));
    }
    static double w_o = 0, w_l = 0; static uint64_t n = 0;
    w_o = std::max(w_o, e_o); w_l = std::max(w_l, e_l); n++;
    if (n <= 4 || n % 64 == 0) {
        fprintf(stderr, "phone-kv check: layer %d, %zu keys, %d tokens: max|O-ref|/max|ref| %.2e, |lse-ref| %.2e (worst so far over %llu calls: %.2e, %.2e)\n",
                j.layer, nk, j.n_tok, e_o, e_l, (unsigned long long) n, w_o, w_l);
    }
}

// One attention call (ATTN or ATTN_BIG) on every phone that holds keys. Of, lse: the merged partial, normalized (O = sum of
// w_p O_p, w_p = exp(lse_p - lse), lse = log sum exp(lse_p)); O also keeps phone 1's raw f16 output for the debug check.
bool attn_all(bool big, const ggml_metal_remote_job & j, uint32_t n_tok, const std::vector<uint16_t> & Q, size_t qn,
              std::vector<uint16_t> & O, std::vector<float> & Of, std::vector<float> & lse, pa::attn_rep & rep) {
    const size_t rows = qn/pa::HD;
    O.resize(qn);
    lse.resize(rows);
    Of.resize(qn);
    std::lock_guard<std::mutex> lk(r_mu);
    if (!r_cli) {
        return false;
    }
    auto call = [&](pa::client * c, uint16_t * o, float * l, pa::attn_rep & r) {
        return big ? c->attn_big((uint32_t) j.layer, n_tok, 0, j.scale, Q.data(), qn, o, l, r)
                   : c->attn((uint32_t) j.layer, n_tok, 0, j.scale, Q.data(), qn, o, l, r);
    };
    if (r_more.empty()) {
        if (!call(r_cli, O.data(), lse.data(), rep)) {
            fprintf(stderr, "phone-kv: %s layer %d failed: %s\n", big ? "ATTN_BIG" : "ATTN", j.layer, r_cli->last_err.c_str());
            return false;
        }
        for (size_t i = 0; i < qn; i++) {
            Of[i] = pa::h2f(O[i]);
        }
        return true;
    }
    const size_t np = n_phones();
    std::vector<std::vector<uint16_t>> Op(np);
    std::vector<std::vector<float>>    Lp(np);
    std::vector<pa::attn_rep>          Rp(np);
    std::vector<char>                  ok(np, 1), used(np, 0);
    std::vector<std::thread>           th;
    for (size_t p = 0; p < np; p++) {
        if (r_held_p[p] == 0) {
            continue;   // nothing there yet (the first block is still filling)
        }
        used[p] = 1;
        Op[p].resize(qn);
        Lp[p].resize(rows);
        if (p == 0) {
            continue;
        }
        th.emplace_back([&, p]() { ok[p] = call(cli(p), Op[p].data(), Lp[p].data(), Rp[p]); });
    }
    if (used[0]) {
        ok[0] = call(r_cli, Op[0].data(), Lp[0].data(), Rp[0]);
    }
    for (auto & t : th) {
        t.join();
    }
    for (size_t p = 0; p < np; p++) {
        if (used[p] && !ok[p]) {
            fprintf(stderr, "phone-kv: %s layer %d failed on phone %zu: %s\n", big ? "ATTN_BIG" : "ATTN", j.layer, p + 1, cli(p)->last_err.c_str());
            return false;
        }
    }
    rep = {};
    for (size_t p = 0; p < np; p++) {
        if (used[p]) {
            rep.phone_ms = std::max(rep.phone_ms, Rp[p].phone_ms);
            rep.nk += Rp[p].nk;
        }
    }
    if (used[0]) {
        O = Op[0];
    }
    for (size_t r = 0; r < rows; r++) {
        float m = -INFINITY;
        for (size_t p = 0; p < np; p++) {
            if (used[p] && std::isfinite(Lp[p][r])) m = std::max(m, Lp[p][r]);
        }
        float * o = Of.data() + r*pa::HD;
        std::fill(o, o + pa::HD, 0.0f);
        if (!std::isfinite(m)) {
            lse[r] = -INFINITY;
            continue;
        }
        float sum = 0;
        for (size_t p = 0; p < np; p++) {
            if (!used[p] || !std::isfinite(Lp[p][r])) continue;
            const float w = expf(Lp[p][r] - m);
            sum += w;
            const uint16_t * op = Op[p].data() + r*pa::HD;
            for (int d = 0; d < pa::HD; d++) o[d] += w*pa::h2f(op[d]);
        }
        for (int d = 0; d < pa::HD; d++) o[d] /= sum;
        lse[r] = m + logf(sum);
    }
    return true;
}

// ATTN_BIG (protocol v3): group 0 of a prefill ubatch sends every token's Q in one call; the phone reads its keys for all 8-token
// groups in one request and the partials of the whole ubatch land in j.part (a big buffer) in the usual row layout
// rid = t*n_head + head, with (S, M) at GGML_METAL_REMOTE_BIG_SM. The followers (groups > 0) have nothing left to do.
void run_big(const ggml_metal_remote_job & j, std::vector<uint16_t> & Q, std::vector<uint16_t> & O, std::vector<float> & lse) {
    const int G  = j.n_head/j.n_head_kv;
    const int nt = j.n_tok_all;
    const int ng = (nt + 7)/8;
    const size_t qn = (size_t) j.n_head_kv*pa::NR*pa::HD;   // one group
    Q.assign((size_t) ng*qn, 0);
    for (int gr = 0; gr < ng; gr++) {
        for (int h = 0; h < j.n_head_kv; h++) {
            for (int g = 0; g < G; g++) {
                for (int t = 0; t < 8 && gr*8 + t < nt; t++) {
                    const float * src = (const float *) ((const char *) j.q + (size_t) (gr*8 + t)*j.q_nb1 + (size_t) (h*G + g)*j.q_nb2);
                    uint16_t * dst = Q.data() + (size_t) gr*qn + ((size_t) h*pa::NR + g*8 + t)*pa::HD;
                    for (int d = 0; d < pa::HD; d++) {
                        dst[d] = pa::f2h(src[d]);
                    }
                }
            }
        }
    }
    pa::attn_rep rep = {};
    const uint64_t t0 = now_ns();
    std::vector<float> Of;
    if (!attn_all(true, j, (uint32_t) nt, Q, (size_t) ng*qn, O, Of, lse, rep)) {
        abort();   // never merge a missing partial silently
    }
    const uint64_t t1 = now_ns();
    for (int gr = 0; gr < ng; gr++) {
        for (int h = 0; h < j.n_head_kv; h++) {
            for (int g = 0; g < G; g++) {
                for (int t = 0; t < 8 && gr*8 + t < nt; t++) {
                    const int r = g*8 + t;
                    const int64_t rid = (int64_t) (gr*8 + t)*j.n_head + (h*G + g);
                    const float * o = Of.data() + (size_t) gr*qn + ((size_t) h*pa::NR + r)*pa::HD;
                    float * dst = j.part + rid*pa::HD;
                    for (int d = 0; d < pa::HD; d++) {
                        dst[d] = o[d];
                    }
                    const float l = lse[(size_t) gr*(qn/pa::HD) + h*pa::NR + r];
                    j.part[GGML_METAL_REMOTE_BIG_SM + 2*rid + 0] = std::isfinite(l) ? 1.0f : 0.0f;
                    j.part[GGML_METAL_REMOTE_BIG_SM + 2*rid + 1] = std::isfinite(l) ? l : -FLT_MAX/2;
                }
            }
        }
    }
    static int stats = -1;
    static uint64_t n = 0;
    static double sum_rt = 0, sum_ph = 0;
    if (stats < 0) {
        stats = getenv("GGML_METAL_REMOTE_STATS") ? atoi(getenv("GGML_METAL_REMOTE_STATS")) : 0;
    }
    n++; sum_rt += (t1 - t0)*1e-6; sum_ph += rep.phone_ms;
    if (stats > 0 && n % (uint64_t) stats == 0) {
        fprintf(stderr, "phone-kv: %llu ATTN_BIG calls (%d tokens), mean round trip %.2f ms (phone compute %.2f ms), %u keys per layer\n",
                (unsigned long long) n, nt, sum_rt/n, sum_ph/n, rep.nk);
    }
}

void run_job(const ggml_metal_remote_job & j, std::vector<uint16_t> & Q, std::vector<uint16_t> & O, std::vector<float> & lse) {
    if (j.n_tok_all > 0) {
        if (j.grp == 0) {
            run_big(j, Q, O, lse);
        }
        return;   // a follower: group 0 already put its rows in the big buffer
    }
    const int G   = j.n_head/j.n_head_kv;
    const size_t qn = (size_t) j.n_head_kv*pa::NR*pa::HD;
    static int delay_us = -1;
    if (delay_us < 0) delay_us = getenv("GGML_METAL_REMOTE_DELAY_US") ? atoi(getenv("GGML_METAL_REMOTE_DELAY_US")) : 0;
    if (delay_us > 0) {   // debug: does reading Q later change the answer? (a race with the GPU writing Q)
        const uint64_t t = now_ns();
        while (now_ns() - t < (uint64_t) delay_us*1000) __builtin_arm_yield();
    }
    auto qsum = [&]() {
        double s = 0;
        for (int t = 0; t < j.n_tok; t++) for (int h = 0; h < j.n_head; h++) {
            const float * src = (const float *) ((const char *) j.q + (size_t) t*j.q_nb1 + (size_t) h*j.q_nb2);
            for (int d = 0; d < pa::HD; d++) s += src[d]*(1.0 + 1e-3*d);
        }
        return s;
    };
    const double qs0 = check_on() ? qsum() : 0.0;
    Q.assign(qn, 0);
    // Q rows per KV head: r = g*8 + t (GQA group member g, token t), unscaled; rows t >= n_tok stay zero
    for (int h = 0; h < j.n_head_kv; h++) {
        for (int g = 0; g < G; g++) {
            for (int t = 0; t < j.n_tok; t++) {
                const float * src = (const float *) ((const char *) j.q + (size_t) t*j.q_nb1 + (size_t) (h*G + g)*j.q_nb2);
                uint16_t * dst = Q.data() + ((size_t) h*pa::NR + g*8 + t)*pa::HD;
                for (int d = 0; d < pa::HD; d++) {
                    dst[d] = pa::f2h(src[d]);
                }
            }
        }
    }

    pa::attn_rep rep = {};
    const uint64_t t0 = now_ns();
    std::vector<float> Of;
    if (!attn_all(false, j, (uint32_t) j.n_tok, Q, qn, O, Of, lse, rep)) {
        // never merge a missing partial silently: the output would be wrong
        abort();
    }
    const uint64_t t1 = now_ns();
    if (check_on() && r_more.empty()) {
        const double qs1 = qsum();
        if (qs1 != qs0) {
            static int nq = 0;
            if (nq++ < 10) fprintf(stderr, "phone-kv check: Q CHANGED during the call (layer %d): %.6g -> %.6g\n", j.layer, qs0, qs1);
        }
        check_job(j, O, lse);
    }

    // the phone's O is normalized: O_unnorm = O, S = 1, M = lse (the reduce weighs it by exp(M - max M))
    for (int h = 0; h < j.n_head_kv; h++) {
        for (int g = 0; g < G; g++) {
            for (int t = 0; t < j.n_tok; t++) {
                const int r = g*8 + t;
                const int64_t rid = (int64_t) t*j.n_head + (h*G + g);
                const float * o = Of.data() + ((size_t) h*pa::NR + r)*pa::HD;
                float * dst = j.part + rid*pa::HD;
                for (int d = 0; d < pa::HD; d++) {
                    dst[d] = o[d];
                }
                const float l = lse[h*pa::NR + r];
                j.part[GGML_METAL_COATTN_SM + 2*rid + 0] = std::isfinite(l) ? 1.0f : 0.0f;
                j.part[GGML_METAL_COATTN_SM + 2*rid + 1] = std::isfinite(l) ? l : -FLT_MAX/2;
            }
        }
    }

    // GGML_METAL_REMOTE_TRACE=N (debug): one line for each of the first N calls
    static int trace = -1;
    static uint64_t n_trace = 0;
    if (trace < 0) {
        trace = getenv("GGML_METAL_REMOTE_TRACE") ? atoi(getenv("GGML_METAL_REMOTE_TRACE")) : 0;
    }
    if (n_trace < (uint64_t) trace) {
        n_trace++;
        fprintf(stderr, "phone-kv trace %3llu: layer %2d, n_tok %d, nrows %lld, phone keys %u, v_start %llu v_done %llu event %llu, part %p, lse[h0,r0] %.4f\n",
                (unsigned long long) n_trace, j.layer, j.n_tok, (long long) j.nrows, rep.nk, (unsigned long long) j.v_start,
                (unsigned long long) j.v_done, (unsigned long long) r_ev[j.ev].signaledValue, (void *) j.part, lse[0]);
    }

    static int stats = -1;
    static uint64_t n = 0;
    static double sum_rt = 0, sum_ph = 0;
    if (stats < 0) {
        stats = getenv("GGML_METAL_REMOTE_STATS") ? atoi(getenv("GGML_METAL_REMOTE_STATS")) : 0;
    }
    n++;
    sum_rt += (t1 - t0)*1e-6;
    sum_ph += rep.phone_ms;
    if (stats > 0 && n % (uint64_t) stats == 0) {
        fprintf(stderr, "phone-kv: %llu ATTN calls, mean round trip %.3f ms (phone compute %.3f ms), %u keys per layer\n",
                (unsigned long long) n, sum_rt/n, sum_ph/n, rep.nk);
    }
}

void * phone_thread(void *) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    std::vector<uint16_t> Q, O;
    std::vector<float> lse;
    uint64_t t_idle = now_ns();
    for (;;) {
        ggml_metal_remote_job j;
        bool have = false;
        {
            std::lock_guard<std::mutex> lk(r_qmu);
            // the oldest job whose Q is ready (its own event reached v_start); graph order across both events
            int best = -1;
            bool pending = false;
            for (int i = 0; i < (int) r_q.size(); i++) {
                pending = true;
                if (r_ev[r_q[i].ev].signaledValue < r_q[i].v_start) {
                    continue;
                }
                if (best < 0 || r_q[i].v_start < r_q[best].v_start) {
                    best = i;
                }
            }
            if (best >= 0) {
                j = r_q[best];
                r_q[best] = r_q.back();
                r_q.pop_back();
                have = true;
            } else if (pending) {
                t_idle = now_ns();   // a job is pending: keep spinning, the GPU is about to reach it
            }
        }
        if (!have) {
            if (now_ns() - t_idle < 5000000ull) {
                __builtin_arm_yield();
            } else {
                struct timespec ts = { 0, 50000 };
                nanosleep(&ts, nullptr);
            }
            continue;
        }
        run_job(j, Q, O, lse);
        __sync_synchronize();
        if (r_ev[j.ev].signaledValue < j.v_done) {
            r_ev[j.ev].signaledValue = j.v_done;
        }
        t_idle = now_ns();
    }
    return nullptr;
}

void start_thread() {
    pthread_t th;
    pthread_create(&th, nullptr, phone_thread, nullptr);
    pthread_detach(th);
}

} // namespace

extern "C" {

int ggml_backend_metal_remote_attach(const char * host_port, int n_layer, const int32_t * il, int n_head_kv, size_t rs, size_t hb, int is_q8) {
    std::lock_guard<std::mutex> lk(r_mu);
    if (n_head_kv <= 0 || n_head_kv > 16 || n_layer <= 0) {
        return 1;
    }
    std::string all = host_port ? host_port : "";
    for (auto * c : r_more) {
        delete c;
    }
    r_more.clear();
    if (const char * b = getenv("GGML_METAL_REMOTE_BLOCK")) {
        r_block = std::max(256, atoi(b));
    }
    // more phones after a comma: attach the first below, the rest the same way after it
    std::vector<std::string> extra;
    for (size_t c; (c = all.rfind(',')) != std::string::npos; all = all.substr(0, c)) {
        extra.insert(extra.begin(), all.substr(c + 1));
    }
    std::string hp = all;
    std::string host = hp;
    int port = pa::DEFAULT_PORT;
    if (auto c = hp.rfind(':'); c != std::string::npos) {
        host = hp.substr(0, c);
        port = atoi(hp.c_str() + c + 1);
    }
    delete r_cli;
    r_cli = new pa::client();
    pa::hello_rep hr = {};
    if (!r_cli->connect(host, port) || !r_cli->hello(hr)) {
        fprintf(stderr, "phone-kv: cannot reach the phone at %s:%d: %s\n", host.c_str(), port, r_cli->last_err.c_str());
        delete r_cli; r_cli = nullptr;
        return 1;
    }
    if (hr.version < 2 || hr.version > pa::VERSION) {
        fprintf(stderr, "phone-kv: phone speaks protocol v%u, this build v%u\n", hr.version, pa::VERSION);
        delete r_cli; r_cli = nullptr;
        return 1;
    }
    r_version = hr.version;
    if (r_version < 3) {
        fprintf(stderr, "phone-kv: the phone speaks protocol v%u (no ATTN_BIG): prefill with phone-held keys runs 8 tokens at a time; "
                        "rebuild Sidecar for v3\n", r_version);
    }
    pa::config_req cfg = {};
    cfg.n_layer     = (uint32_t) n_layer;
    cfg.n_head_kv   = (uint32_t) n_head_kv;
    cfg.rs          = (uint32_t) rs;
    cfg.hb          = (uint32_t) hb;
    cfg.is_q8       = (uint32_t) is_q8;
    cfg.sme_workers = getenv("GGML_METAL_REMOTE_SME_WORKERS") ? (uint32_t) atoi(getenv("GGML_METAL_REMOTE_SME_WORKERS")) : 2;
    cfg.sme_helpers = getenv("GGML_METAL_REMOTE_SME_HELPERS") ? (uint32_t) atoi(getenv("GGML_METAL_REMOTE_SME_HELPERS")) : 1;
    // A19 Pro (2026-09-26, pa-tool, q8_0 pages): SME only 3.02 ms / GPU only 2.98 ms at 16k keys; 75% of the pages on the GPU
    // (neural accelerators) and the rest on SME at the same time: 2.41 ms (16k), 8.88 vs 11.47 ms GPU-only (64k). A phone without
    // a GPU engine ignores the share.
    cfg.gpu_permille = getenv("GGML_METAL_REMOTE_GPU_PERMILLE") ? (uint32_t) atoi(getenv("GGML_METAL_REMOTE_GPU_PERMILLE")) : 700;
    cfg.gpu_chunk   = 1024;
    if (!r_cli->config(cfg)) {
        fprintf(stderr, "phone-kv: CONFIG failed: %s\n", r_cli->last_err.c_str());
        delete r_cli; r_cli = nullptr;
        return 1;
    }
    r_nlayer = n_layer;
    r_nkv    = n_head_kv;
    r_is_q8  = is_q8;
    r_hb     = hb;
    r_rs     = rs;
    r_il.assign(il, il + n_layer);
    int32_t mx = 0;
    for (int i = 0; i < n_layer; i++) mx = std::max(mx, il[i]);
    r_map.assign(mx + 1, -1);
    for (int i = 0; i < n_layer; i++) r_map[il[i]] = i;
    r_held = 0;
    fprintf(stderr, "phone-kv: attached to '%s' at %s:%d (%d layers, %d KV heads, %zu B rows, %s)\n",
            hr.device, host.c_str(), port, n_layer, n_head_kv, rs, is_q8 == 2 ? "q4_0" : is_q8 ? "q8_0" : "f16");
    for (const std::string & e : extra) {
        std::string h2 = e;
        int p2 = pa::DEFAULT_PORT;
        if (auto c = e.rfind(':'); c != std::string::npos) {
            h2 = e.substr(0, c);
            p2 = atoi(e.c_str() + c + 1);
        }
        auto * c2 = new pa::client();
        pa::hello_rep h = {};
        if (!c2->connect(h2, p2) || !c2->hello(h) || h.version < 2 || h.version > pa::VERSION || !c2->config(cfg)) {
            fprintf(stderr, "phone-kv: cannot use the phone at %s:%d: %s\n", h2.c_str(), p2, c2->last_err.c_str());
            delete c2;
            for (auto * c : r_more) delete c;
            r_more.clear();
            delete r_cli; r_cli = nullptr;
            return 1;
        }
        r_version = std::min(r_version, h.version);
        r_more.push_back(c2);
        fprintf(stderr, "phone-kv: attached to '%s' at %s:%d as phone %zu\n", h.device, h2.c_str(), p2, r_more.size() + 1);
    }
    r_held_p.assign(n_phones(), 0);
    if (!r_more.empty()) {
        fprintf(stderr, "phone-kv: %zu phones, old keys dealt in blocks of %u positions\n", n_phones(), r_block);
    }
    return 0;
}

int ggml_backend_metal_remote_append(int i, uint32_t pos0, uint32_t n, const void * k_, const void * v_) {
    std::lock_guard<std::mutex> lk(r_mu);
    if (!r_cli || i < 0 || i >= r_nlayer) {
        return 1;
    }
    if (r_more.empty()) {
        if (!r_cli->append((uint32_t) i, pos0, n, k_, v_, r_rs)) {
            fprintf(stderr, "phone-kv: APPEND layer %d failed: %s\n", i, r_cli->last_err.c_str());
            return 1;
        }
    } else {
        // in pieces at block boundaries, each to the phone that owns the block (at that phone's own next position)
        for (uint32_t a = pos0; a < pos0 + n; ) {
            const uint32_t b = std::min(pos0 + n, (a/r_block + 1)*r_block);
            const size_t   p = (a/r_block) % n_phones();
            const size_t off = (size_t) (a - pos0)*r_rs;
            if (!cli(p)->append((uint32_t) i, local_count(p, a), b - a, (const uint8_t *) k_ + off, (const uint8_t *) v_ + off, r_rs)) {
                fprintf(stderr, "phone-kv: APPEND layer %d failed on phone %zu: %s\n", i, p + 1, cli(p)->last_err.c_str());
                return 1;
            }
            a = b;
        }
    }
    if (check_on()) {
        if ((int) r_chk_k.size() < r_nlayer) { r_chk_k.resize(r_nlayer); r_chk_v.resize(r_nlayer); }
        r_chk_k[i].resize((size_t) pos0*pa::HD); r_chk_v[i].resize((size_t) pos0*pa::HD);
        for (uint32_t k = 0; k < n; k++) {
            for (int d = 0; d < pa::HD; d++) {
                r_chk_k[i].push_back(deq_head0((const uint8_t *) k_ + (size_t) k*r_rs, d));
                r_chk_v[i].push_back(deq_head0((const uint8_t *) v_ + (size_t) k*r_rs, d));
            }
        }
    }
    if (i == r_nlayer - 1) {
        r_held = pos0 + n;   // every layer has it once the last layer does
        for (size_t p = 0; p < r_held_p.size(); p++) {
            r_held_p[p] = local_count(p, pos0 + n);
        }
    }
    return 0;
}

int ggml_backend_metal_remote_truncate(uint32_t n) {
    std::lock_guard<std::mutex> lk(r_mu);
    if (!r_cli) {
        return n == 0 ? 0 : 1;
    }
    for (size_t p = 0; p < n_phones(); p++) {
        const uint32_t np_n = r_more.empty() ? n : local_count(p, n);
        if (!cli(p)->truncate(np_n)) {
            fprintf(stderr, "phone-kv: TRUNCATE failed on phone %zu: %s\n", p + 1, cli(p)->last_err.c_str());
            return 1;
        }
        if (p < r_held_p.size()) {
            r_held_p[p] = std::min(r_held_p[p], np_n);
        }
    }
    r_held = std::min<uint32_t>(r_held, n);
    return 0;
}

uint32_t ggml_backend_metal_remote_held(void) {
    return r_held.load();
}

int ggml_backend_metal_remote_big(void) {
    static const int off = getenv("GGML_METAL_REMOTE_BIG") && atoi(getenv("GGML_METAL_REMOTE_BIG")) == 0;   // =0: force 8-token prefill
    return r_cli && r_version >= 3 && !off ? 1 : 0;
}

int ggml_metal_remote_layer(const int32_t * op_params) {
    const int32_t tag = op_params[GGML_METAL_REMOTE_OP_PARAM];
    if ((tag & 0xFFFF0000) != GGML_METAL_REMOTE_TAG || r_held.load() == 0) {
        return -1;
    }
    const int32_t il = (tag & 0xFFFF) - 1;
    return il >= 0 && il < (int32_t) r_map.size() ? r_map[il] : -1;
}

void * ggml_metal_remote_event_half(void * mtl_device, int half) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    half &= 1;
    if (r_ev[half] == nil) {
        r_ev[half] = [(id<MTLDevice>) mtl_device newSharedEvent];
        r_ev[half].signaledValue = 0;
    }
    return (void *) r_ev[half];
}

void * ggml_metal_remote_event(void * mtl_device) {
    return ggml_metal_remote_event_half(mtl_device, 0);
}

void * ggml_metal_remote_part_buffer(void * mtl_device, int i, float ** host) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (r_part[0] == nil) {
        for (int k = 0; k < 2; k++) {
            r_part[k] = [(id<MTLDevice>) mtl_device newBufferWithLength:GGML_METAL_COATTN_FLOATS*sizeof(float)
                                                                options:MTLResourceStorageModeShared];
        }
    }
    *host = (float *) [r_part[i & 1] contents];
    return (void *) r_part[i & 1];
}

void * ggml_metal_remote_big_buffer(void * mtl_device, int i, float ** host) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (r_big[0] == nil) {
        for (int k = 0; k < 4; k++) {
            r_big[k] = [(id<MTLDevice>) mtl_device newBufferWithLength:(size_t) GGML_METAL_REMOTE_BIG_ROWS*258*sizeof(float)
                                                               options:MTLResourceStorageModeShared];
        }
    }
    *host = (float *) [r_big[i & 3] contents];
    return (void *) r_big[i & 3];
}

void ggml_metal_remote_submit(const struct ggml_metal_remote_job * job) {
    pthread_once(&r_once, start_thread);
    std::lock_guard<std::mutex> lk(r_qmu);
    r_q.push_back(*job);
}

} // extern "C"

int ggml_metal_remote_defer(void) {
    static const int v = getenv("GGML_METAL_REMOTE_DEFER") ? atoi(getenv("GGML_METAL_REMOTE_DEFER")) : 1;
    return v;
}
