// SME co-attention (infernet): see ggml-metal-coattn.h.
//
// Threads: 2 SME workers (2 KV heads each), each with 2 NEON pack helpers and 1 NEON softmax helper (the pipelined
// kernel of scripts/sme/sme_attn.c, docs/sme-coattention.md), plus one coordinator that waits for the GPU's "Q is
// ready" event value, stages Q, starts the job and signals "partial is ready". The 8 compute threads take their
// roles per job by the core they are on, so the two SME workers sit in different P-clusters (GGML_METAL_FA_SME_PLACE).
// Jobs are taken in event-value order (= graph node order = GPU execution order).
//
// Compiled with -mcpu=apple-m4 (SME2 intrinsics); nothing here runs unless ggml_metal_coattn_fraction() > 0,
// which requires FEAT_SME2 with 512-bit streaming vectors.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ggml-metal-coattn.h"
#include "ggml-metal-coattn-sme.inc"

#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <sys/sysctl.h>
#include <time.h>

#define CO_MAXQ 4096
#define CO_NW   2                    // SME workers (pipes)
#define CO_NH   2                    // pack helpers per worker
#define CO_TPW  (CO_NH + 2)          // threads per worker: worker, softmax helper, pack helpers
#define CO_NT   (CO_NW*CO_TPW)
#define CO_BK   512                  // keys per online-softmax block
#define CO_SPIN_NS_DEFAULT 5000000ull  // spin this long before backing off to short sleeps (GGML_METAL_FA_SME_SPIN_US)

static id<MTLSharedEvent> co_event = nil;
static id<MTLSharedEvent> co_event_gpu = nil;
static _Atomic float co_frac_now = -1.0f;
static _Atomic uint64_t co_n_jobs = 0, co_n_gpu_first = 0;

static pthread_mutex_t co_mu = PTHREAD_MUTEX_INITIALIZER;
static struct ggml_metal_coattn_job co_q[CO_MAXQ];
static int co_nq = 0;

static _Atomic uint64_t co_next = 1;  // next free event value
static _Atomic uint64_t co_base = 0;  // current graph's base

static struct ggml_metal_coattn_job co_cur;
static sme_pipe * co_pipe[CO_NW];
static float * co_Q;                  // [n_head_kv][48][256] staged queries (rows g*8 + t)
static float * co_O;                  // [n_head_kv][48][256]
static float * co_m;                  // [n_head_kv][48]
static float * co_l;

static _Atomic uint64_t co_go = 0;    // job sequence number published to the compute threads
static _Atomic int co_helpers_done[CO_NW];
static _Atomic int co_workers_done;
static _Atomic uint64_t co_t_seen[CO_NT]; // _STATS: when each compute thread picked up the current job
static _Atomic uint64_t co_cpu_hist[16];
static _Atomic int co_wcpu0[CO_NW], co_wcpu1[CO_NW]; // _STATS: core of each SME worker at pickup and at the end

// Cluster-aware roles (GGML_METAL_FA_SME_PLACE, default 1). The M4 Pro has one SME unit per P-cluster (cores 4-8, 9-13) and
// macOS can't pin threads, so fixed roles put both SME workers in the same cluster in ~45% of jobs (4/9 at random), where
// they share one unit: 1.55x slower (2026-09-25, 51k live server). Instead every compute thread reports its core when it
// picks up a job, and the coordinator hands out roles: one SME worker per P-cluster, each worker's helpers in its cluster.
static _Atomic int      co_core[CO_NT];
static _Atomic int      co_role[CO_NT];
static _Atomic int      co_arrived = 0;
static _Atomic uint64_t co_role_seq = 0;

static int co_place(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = getenv("GGML_METAL_FA_SME_PLACE");
        v = s ? atoi(s) : 1;
    }
    return v;
}

// 0 / 1 = P-cluster, 2 = E-cluster or unknown (M4 Pro layout: cpu0-3 E, cpu4-8 and cpu9-13 P)
static int co_cluster(int cpu) {
    return cpu >= 4 && cpu < 14 ? (cpu - 4)/5 : 2;
}

// roles: role = w*CO_TPW + r (r 0 = SME worker, 1 = softmax helper, 2.. = pack helpers)
static void co_assign_roles(void) {
    int L[3][CO_NT], nL[3] = { 0, 0, 0 }, iL[3] = { 0, 0, 0 };
    for (int t = 0; t < CO_NT; t++) {
        const int c = co_cluster(atomic_load_explicit(&co_core[t], memory_order_relaxed));
        L[c][nL[c]++] = t;
    }
    // pipe w prefers its own cluster, then E/unknown, then the other P-cluster
    #define CO_POP(w) (iL[(w)] < nL[(w)] ? L[(w)][iL[(w)]++] : iL[2] < nL[2] ? L[2][iL[2]++] : L[1 - (w)][iL[1 - (w)]++])
    int role_of[CO_NT];
    const int w0 = CO_POP(0);
    const int w1 = CO_POP(1);
    role_of[w0] = 0*CO_TPW;
    role_of[w1] = 1*CO_TPW;
    for (int r = 1; r < CO_TPW; r++) {
        for (int w = 0; w < CO_NW; w++) {
            role_of[CO_POP(w)] = w*CO_TPW + r;
        }
    }
    #undef CO_POP
    for (int t = 0; t < CO_NT; t++) {
        atomic_store_explicit(&co_role[t], role_of[t], memory_order_relaxed);
    }
}  // _STATS: which core (pthread_cpu_number_np) compute threads start jobs on

static pthread_once_t co_once = PTHREAD_ONCE_INIT;

static uint64_t co_now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

// spin, then back off to short sleeps (so idle threads don't burn the P-cores between verifies)
static uint64_t co_spin_ns(void) {
    static uint64_t v = 0;
    if (v == 0) {
        const char * s = getenv("GGML_METAL_FA_SME_SPIN_US");
        v = s ? (uint64_t) atoll(s)*1000ull + 1 : CO_SPIN_NS_DEFAULT;
    }
    return v;
}

// jobs submitted (at encode time) and not yet taken by the coordinator. While > 0 the compute threads keep spinning
// (GGML_METAL_FA_SME_SPIN_PENDING, default 1): a verify's jobs are all queued when it is encoded, so the threads stay
// awake for the whole verify and sleep between verifies. With the timed spin alone they fell asleep between attention
// layers at depth (layers ~10 ms apart at 140k > the 5 ms spin): ~0.9 ms mean wake-up, the GPU then waited on the CPU.
static _Atomic int co_pending = 0;

// Predictive wake (GGML_METAL_FA_SME_LEAD_US=<us>, default 0 = off: spin whenever a job is queued): the coordinator tracks
// when each job's Q became ready and the mean gap between attention layers; while a job is queued the compute threads sleep
// until LEAD before the next one is due, then spin. Meant to save the CPU power that pulls the GPU clock down (1578 -> 1492 MHz
// seen at 51k with 8 spinning cores), but at 140k LEAD 1500 got only half the gain (verify 151.2 -> 146.7 ms vs 142.0 with
// plain spinning, 2 runs each, 2026-09-25), so it is off.
static _Atomic uint64_t co_t_last_ready = 0, co_gap_ema = 0;

static uint64_t co_lead_ns(void) {
    static int64_t v = -1;
    if (v < 0) {
        const char * s = getenv("GGML_METAL_FA_SME_LEAD_US");
        v = s ? (int64_t) atoll(s)*1000 : 0;
    }
    return (uint64_t) v;
}

static int co_spin_pending(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = getenv("GGML_METAL_FA_SME_SPIN_PENDING");
        v = s ? atoi(s) : 1;
    }
    return v;
}

// Low-power spin (GGML_METAL_FA_SME_WFE, default 1): the compute threads wait for a new job with WFE on the job counter's
// cache line (armed by a load-exclusive) instead of a yield loop. The CPU and GPU share one power budget: 8 cores in a
// yield loop through the whole verify pulled the GPU from 1578 to 1492 MHz and ate co-attention's gain (51k, 2026-09-25).
// A store to the counter wakes WFE in ~0.1 us (measured in user space on this Mac).
static int co_wfe(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = getenv("GGML_METAL_FA_SME_WFE");
        v = s ? atoi(s) : 1;
    }
    return v;
}

static inline uint64_t co_ldx(_Atomic uint64_t * p) {
    uint64_t v;
    __asm__ volatile("ldaxr %0, [%1]" : "=r"(v) : "r"(p) : "memory");
    return v;
}

static void co_backoff(uint64_t t_idle0);

// spin now? queued job: yes once it is due within LEAD (or no prediction yet); nothing queued: within the timed spin window
static int co_should_spin(uint64_t t_idle0) {
    const uint64_t now = co_now_ns();
    if (co_spin_pending() && atomic_load_explicit(&co_pending, memory_order_relaxed) > 0) {
        const uint64_t lead = co_lead_ns();
        const uint64_t tl = atomic_load_explicit(&co_t_last_ready, memory_order_relaxed);
        const uint64_t gap = atomic_load_explicit(&co_gap_ema, memory_order_relaxed);
        return lead == 0 || gap == 0 || tl == 0 || now + lead >= tl + gap;
    }
    return now - t_idle0 < co_spin_ns();
}

// wait until *p != seen: spin (WFE or yield) while a job is pending or within the spin window, then short sleeps
static void co_wait_change(_Atomic uint64_t * p, uint64_t seen) {
    const uint64_t t0 = co_now_ns();
    for (;;) {
        if (co_wfe()) {
            if (co_ldx(p) != seen) {
                __asm__ volatile("clrex" ::: "memory");
                return;
            }
            if (co_should_spin(t0)) {
                __asm__ volatile("wfe" ::: "memory");
                continue;
            }
            __asm__ volatile("clrex" ::: "memory");
        } else if (atomic_load_explicit(p, memory_order_acquire) != seen) {
            return;
        }
        co_backoff(t0);
    }
}

static void co_backoff(uint64_t t_idle0) {
    if (co_should_spin(t_idle0)) {
        __builtin_arm_yield();
    } else {
        struct timespec ts = { 0, 50000 };
        nanosleep(&ts, NULL);
    }
}

static int co_sme2_available(void) {
    int v = 0, svl = 0;
    size_t n = sizeof v, n2 = sizeof svl;
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &v, &n, NULL, 0) != 0 || !v) {
        return 0;
    }
    if (sysctlbyname("hw.optional.arm.sme_max_svl_b", &svl, &n2, NULL, 0) != 0) {
        return 0;
    }
    return svl == 64;
}

float ggml_metal_coattn_fraction(void) {
    static float frac = -1.0f;
    if (frac < 0.0f) {
        const char * s = getenv("GGML_METAL_FA_SME");
        float f = s ? (float) atof(s) : 0.0f;
        if (f > 0.0f && !co_sme2_available()) {
            fprintf(stderr, "%s: GGML_METAL_FA_SME set but SME2 (512-bit) is not available: disabled\n", __func__);
            f = 0.0f;
        }
        frac = f < 0.0f ? 0.0f : (f > 0.9f ? 0.9f : f);
    }
    return frac;
}

float ggml_metal_coattn_fraction_now(void) {
    float f = atomic_load(&co_frac_now);
    if (f < 0.0f) {
        f = ggml_metal_coattn_fraction();
        atomic_store(&co_frac_now, f);
    }
    return f;
}

void * ggml_metal_coattn_event_gpu(void * mtl_device) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (co_event_gpu == nil) {
        co_event_gpu = [(id<MTLDevice>) mtl_device newSharedEvent];
        co_event_gpu.signaledValue = 0;
    }
    pthread_mutex_unlock(&mu);
    return (void *) co_event_gpu;
}

int ggml_metal_coattn_min_kv(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = getenv("GGML_METAL_FA_SME_MIN_KV");
        v = s ? atoi(s) : 8192;
        if (v < 256) {
            v = 256;
        }
    }
    return v;
}

void * ggml_metal_coattn_event(void * mtl_device) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (co_event == nil) {
        co_event = [(id<MTLDevice>) mtl_device newSharedEvent];
        co_event.signaledValue = 0;
    }
    pthread_mutex_unlock(&mu);
    return (void *) co_event;
}

void ggml_metal_coattn_graph_begin(int n_nodes) {
    const uint64_t base = atomic_fetch_add(&co_next, 2ull*(uint64_t) n_nodes + 4);
    atomic_store(&co_base, base);
}

uint64_t ggml_metal_coattn_graph_base(void) {
    return atomic_load(&co_base);
}

static id<MTLBuffer> co_part[2] = { nil, nil };

void * ggml_metal_coattn_part_buffer(void * mtl_device, int i, float ** host) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (co_part[0] == nil) {
        for (int k = 0; k < 2; k++) {
            co_part[k] = [(id<MTLDevice>) mtl_device newBufferWithLength:GGML_METAL_COATTN_FLOATS*sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        }
    }
    pthread_mutex_unlock(&mu);
    *host = (float *) [co_part[i & 1] contents];
    return (void *) co_part[i & 1];
}

static void co_write_output(const struct ggml_metal_coattn_job * j, int h0, int h1) {
    const int G = j->n_head/j->n_head_kv;
    for (int h = h0; h < h1; h++) {
        for (int g = 0; g < G; g++) {
            for (int t = 0; t < j->n_tok; t++) {
                const int r = g*8 + t;
                const int64_t rid = (int64_t) t*j->n_head + (h*G + g);
                memcpy(j->part + rid*HD, co_O + ((size_t) h*NR + r)*HD, HD*sizeof(float));
                j->part[GGML_METAL_COATTN_SM + 2*rid + 0] = co_l[h*NR + r];
                j->part[GGML_METAL_COATTN_SM + 2*rid + 1] = co_m[h*NR + r];
            }
        }
    }
}

// GGML_METAL_FA_SME_CHECK=1: compare the SME partial of KV head 0 (rows g = 0) against a scalar fp32 reference
static float co_ld(const struct ggml_metal_coattn_job * j, const uint8_t * base, int key, int h, int d) {
    const uint8_t * row = base + (size_t) key*j->rs + (size_t) h*j->hb;
    if (!j->is_q8) {
        return (float) ((const __fp16 *) row)[d];
    }
    if (j->is_q8 == 2) {   // q4_0: 18-byte blocks, x[i] = lo nibble - 8, x[i+16] = hi nibble - 8
        const uint8_t * blk = row + (d/32)*18;
        const uint8_t b = blk[2 + (d%32)%16];
        const int v = ((d%32) < 16 ? (b & 0x0F) : (b >> 4)) - 8;
        return (float) *(const __fp16 *) blk * (float) v;
    }
    const uint8_t * blk = row + (d/32)*34;
    return (float) *(const __fp16 *) blk * (float) ((const int8_t *) (blk + 2))[d%32];
}

static void co_check(const struct ggml_metal_coattn_job * j) {
    double e_o = 0, e_m = 0, e_l = 0;
    float * s = malloc(sizeof(float)*j->nk);
    for (int t = 0; t < j->n_tok; t++) {
        const float * q = (const float *) ((const char *) j->q + (size_t) t*j->q_nb1);
        float m = -INFINITY;
        for (int k = 0; k < j->nk; k++) {
            float acc = 0;
            for (int d = 0; d < HD; d++) acc += q[d]*co_ld(j, j->k, k, 0, d);
            acc *= j->scale;
            if (j->mask) acc += (float) ((const __fp16 *) j->mask)[(size_t) t*j->mask_rs + k];
            s[k] = acc;
            m = fmaxf(m, acc);
        }
        double l = 0;
        for (int k = 0; k < j->nk; k++) l += expf(s[k] - m);
        const int r = t; // g = 0
        e_m = fmax(e_m, fabs(co_m[r] - m));
        e_l = fmax(e_l, fabs(co_l[r] - l)/fmax(l, 1e-30));
        for (int d = 0; d < HD; d += 37) {
            double o = 0;
            for (int k = 0; k < j->nk; k++) o += expf(s[k] - m)*co_ld(j, j->v, k, 0, d);
            e_o = fmax(e_o, fabs(co_O[(size_t) r*HD + d] - o)/fmax(fabs(o), 1e-3*l));
        }
    }
    free(s);
    fprintf(stderr, "%s: nk %d n_tok %d is_q8 %d: max |dm| %.3g, rel dl %.3g, rel dO %.3g (m sme %.4g)\n",
            __func__, j->nk, j->n_tok, j->is_q8, e_m, e_l, e_o, co_m[0]);
}

static void * co_compute_thread(void * arg) {
    const int t = (int) (intptr_t) arg;

    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    uint64_t seen = 0;
    for (;;) {
        co_wait_change(&co_go, seen);
        seen = atomic_load_explicit(&co_go, memory_order_acquire);
        atomic_store_explicit(&co_t_seen[t], co_now_ns(), memory_order_relaxed);

        size_t cpu = 99;
        if (pthread_cpu_number_np(&cpu) != 0 || cpu >= 16) {
            cpu = 99;
        }
        int role = t;
        if (co_place()) {
            atomic_store_explicit(&co_core[t], (int) cpu, memory_order_relaxed);
            atomic_fetch_add_explicit(&co_arrived, 1, memory_order_release);
            while (atomic_load_explicit(&co_role_seq, memory_order_acquire) != seen) {
                __builtin_arm_yield();
            }
            role = atomic_load_explicit(&co_role[t], memory_order_relaxed);
        }
        const int w = role/CO_TPW;
        const int r = role%CO_TPW;
        if (cpu < 16) {
            atomic_fetch_add_explicit(&co_cpu_hist[cpu], 1, memory_order_relaxed);
        }
        if (r == 0) {
            atomic_store_explicit(&co_wcpu0[w], (int) cpu, memory_order_relaxed);
        }

        sme_pipe * p = co_pipe[w];
        if (r == 0) {
            sme_pipe_worker2(p);
            while (atomic_load_explicit(&co_helpers_done[w], memory_order_acquire) < CO_TPW - 1) {
                __builtin_arm_yield();
            }
            {
                size_t cpu = 0;
                pthread_cpu_number_np(&cpu);
                atomic_store_explicit(&co_wcpu1[w], (int) cpu, memory_order_relaxed);
            }
            co_write_output(&co_cur, p->h0, p->h1);
            atomic_fetch_add_explicit(&co_workers_done, 1, memory_order_release);
        } else if (r == 1) {
            sme_pipe_softmax_helper(p);
            atomic_fetch_add_explicit(&co_helpers_done[w], 1, memory_order_release);
        } else {
            sme_pipe_helper(p, r - 2);
            atomic_fetch_add_explicit(&co_helpers_done[w], 1, memory_order_release);
        }
    }
    return NULL;
}

static void * co_coord_thread(void * arg) {
    (void) arg;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    for (;;) {
        // the pending job with the smallest start value runs next; wait until the GPU has reached it
        struct ggml_metal_coattn_job j;
        uint64_t t0 = co_now_ns();
        for (;;) {
            int best = -1;
            pthread_mutex_lock(&co_mu);
            for (int i = 0; i < co_nq; i++) {
                if (best < 0 || co_q[i].v_start < co_q[best].v_start) {
                    best = i;
                }
            }
            if (best >= 0 && co_event.signaledValue >= co_q[best].v_start) {
                j = co_q[best];
                co_q[best] = co_q[--co_nq];
                atomic_fetch_sub_explicit(&co_pending, 1, memory_order_relaxed);
                pthread_mutex_unlock(&co_mu);
                break;
            }
            pthread_mutex_unlock(&co_mu);
            if (best >= 0) {
                t0 = co_now_ns(); // a job is pending: keep spinning, the GPU is about to reach it
                __builtin_arm_yield();
            } else {
                co_backoff(t0);
            }
        }

        const uint64_t t_ready = co_now_ns();
        {
            // gap between consecutive attention layers of one verify (> 40 ms = a new verify: not a layer gap)
            const uint64_t tl = atomic_load_explicit(&co_t_last_ready, memory_order_relaxed);
            if (tl != 0 && t_ready - tl < 40000000ull) {
                const uint64_t g = atomic_load_explicit(&co_gap_ema, memory_order_relaxed);
                atomic_store_explicit(&co_gap_ema, g == 0 ? t_ready - tl : (4*g + (t_ready - tl))/5, memory_order_relaxed);
            }
            atomic_store_explicit(&co_t_last_ready, t_ready, memory_order_relaxed);
        }

        // stage Q as [head_kv][48 rows = g*8 + t][256], padding rows t >= n_tok with zeros
        const int G = j.n_head/j.n_head_kv;
        for (int h = 0; h < j.n_head_kv; h++) {
            for (int g = 0; g < G; g++) {
                for (int t = 0; t < 8; t++) {
                    float * dst = co_Q + ((size_t) h*NR + g*8 + t)*HD;
                    if (t < j.n_tok) {
                        memcpy(dst, (const char *) j.q + (size_t) t*j.q_nb1 + (size_t) (h*G + g)*j.q_nb2, HD*sizeof(float));
                    } else {
                        memset(dst, 0, HD*sizeof(float));
                    }
                }
            }
        }

        const uint64_t t_start = co_now_ns();
        co_cur = j;
        for (int w = 0; w < CO_NW; w++) {
            sme_pipe * p = co_pipe[w];
            sme_pipe_reset(p);
            p->Q = co_Q; p->qrs = HD; p->qhs = NR*HD;
            p->K = j.k; p->V = j.v; p->rs = j.rs; p->hb = j.hb; p->is_q8 = j.is_q8;
            p->nk = j.nk; p->bk = CO_BK;
            p->h0 = w*j.n_head_kv/CO_NW; p->h1 = (w + 1)*j.n_head_kv/CO_NW;
            p->scale = j.scale; p->O = co_O; p->m = co_m; p->l = co_l;
            p->nh = CO_NH; p->sm_off = 1;
            p->mask = (const f16 *) j.mask; p->mrs = j.mask_rs; p->ntok = j.n_tok;
            atomic_store(&co_helpers_done[w], 0);
        }
        atomic_store(&co_workers_done, 0);
        atomic_store(&co_arrived, 0);
        const uint64_t go = atomic_fetch_add_explicit(&co_go, 1, memory_order_release) + 1;
        if (co_place()) {
            // every compute thread reports its core, then gets its role for this job
            while (atomic_load_explicit(&co_arrived, memory_order_acquire) < CO_NT) {
                __builtin_arm_yield();
            }
            co_assign_roles();
            atomic_store_explicit(&co_role_seq, go, memory_order_release);
        }

        // wait for the workers; note when the GPU finishes its own share (event 2) to measure both rates
        uint64_t t_gpu = 0;
        while (atomic_load_explicit(&co_workers_done, memory_order_acquire) < CO_NW) {
            if (t_gpu == 0 && co_event_gpu != nil && co_event_gpu.signaledValue >= j.v_start) {
                t_gpu = co_now_ns();
            }
            __builtin_arm_yield();
        }
        const uint64_t t_cpu = co_now_ns();
        __sync_synchronize();
        static int check = -1;
        if (check < 0) {
            check = getenv("GGML_METAL_FA_SME_CHECK") ? atoi(getenv("GGML_METAL_FA_SME_CHECK")) : 0;
        }
        if (check) {
            co_check(&j);
        }
        // adapt the split: did the GPU finish its share before the CPU did?
        {
            static int adapt = -1, stats = -1;
            if (adapt < 0) {
                adapt = getenv("GGML_METAL_FA_SME_ADAPT") ? atoi(getenv("GGML_METAL_FA_SME_ADAPT")) : 1;
                stats = getenv("GGML_METAL_FA_SME_STATS") ? atoi(getenv("GGML_METAL_FA_SME_STATS")) : 0;
            }
            const bool gpu_first = t_gpu != 0;
            const uint64_t n = atomic_fetch_add(&co_n_jobs, 1) + 1;
            const uint64_t ng = gpu_first ? atomic_fetch_add(&co_n_gpu_first, 1) + 1 : atomic_load(&co_n_gpu_first);
            if (adapt) {
                // the CPU was done first: signal right away, then learn when the GPU finishes (nothing else to do
                // until the GPU reaches the next layer anyway)
                if (!gpu_first) {
                    __sync_synchronize();
                    co_event.signaledValue = j.v_done;
                    while (co_event_gpu.signaledValue < j.v_start) {
                        __builtin_arm_yield();
                    }
                    t_gpu = co_now_ns();
                }
                // keys per ns on each side, then the split where both finish together (5% margin toward the GPU)
                const double dc = (double) (t_cpu - t_start), dg = (double) (t_gpu - t_start);
                const double n_gpu = j.nk/(double) j.frac_used - j.nk; // the GPU's keys for this job
                if (dc > 0 && dg > 0 && n_gpu > 0) {
                    const double rc = j.nk/dc, rg = n_gpu/dg;
                    float target = (float) (0.95*rc/(rc + rg));
                    float f = ggml_metal_coattn_fraction_now();
                    f = 0.8f*f + 0.2f*target;
                    f = f < 0.05f ? 0.05f : (f > 0.6f ? 0.6f : f);
                    atomic_store(&co_frac_now, f);
                }
            }
            if (stats) {
                // per-window means (us): Q staging, compute-thread wake lag (last thread to pick the job up), CPU share,
                // GPU share (both from t_start), GPU wait on the CPU (t_cpu - t_gpu when the CPU was late), keys per us
                static double w_stage, w_wake, w_cpu, w_gpu, w_late, w_kc, w_kg; static int w_n, w_late_n;
                static float w_cpu_us[256];
                uint64_t t_wake = 0;
                for (int i = 0; i < CO_NT; i++) {
                    const uint64_t ts = atomic_load_explicit(&co_t_seen[i], memory_order_relaxed);
                    t_wake = ts > t_wake ? ts : t_wake;
                }
                const double n_gpu = j.nk/(double) j.frac_used - j.nk;
                w_stage += (t_start - t_ready)*1e-3;
                w_wake  += t_wake > t_start ? (t_wake - t_start)*1e-3 : 0.0;
                w_cpu   += (t_cpu - t_start)*1e-3;
                w_gpu   += t_gpu > t_start ? (t_gpu - t_start)*1e-3 : 0.0;
                if (t_cpu > t_gpu && t_gpu != 0) { w_late += (t_cpu - t_gpu)*1e-3; w_late_n++; }
                // P-clusters: cores 4-8 and 9-13 on the M4 Pro (0-3 are E-cores); one SME unit per P-cluster
                {
                    static double w_same, w_split; static int w_same_n, w_split_n, w_moved;
                    const int a0 = atomic_load(&co_wcpu0[0]), b0 = atomic_load(&co_wcpu0[1]);
                    const int a1 = atomic_load(&co_wcpu1[0]), b1 = atomic_load(&co_wcpu1[1]);
                    const int ca = a0 < 4 ? -1 : (a0 - 4)/5, cb = b0 < 4 ? -2 : (b0 - 4)/5;
                    const double us1k = (t_cpu - t_start)*1e-3/fmax(j.nk, 1)*1e3;
                    if (ca == cb) { w_same += us1k; w_same_n++; } else { w_split += us1k; w_split_n++; }
                    w_moved += (a0 != a1) + (b0 != b1);
                    if (n % 256 == 0) {
                        fprintf(stderr, "%s:   SME workers in the same P-cluster: %d jobs at %.1f us/1k keys; split: %d jobs at %.1f; "
                                "worker core changed during the job: %d\n", __func__, w_same_n, w_same_n ? w_same/w_same_n : 0.0,
                                w_split_n, w_split_n ? w_split/w_split_n : 0.0, w_moved);
                        w_same = w_split = 0; w_same_n = w_split_n = w_moved = 0;
                    }
                }
                w_kc += j.nk; w_kg += n_gpu;
                w_cpu_us[w_n % 256] = (float) ((t_cpu - t_start)*1e-3/fmax(j.nk, 1)*1e3); // us per 1000 CPU keys
                w_n++;
                if (n % 256 == 0) {
                    fprintf(stderr, "%s: %llu jobs, GPU finished first in %.1f%%, fraction now %.3f | last %d: stage %.1f wake %.1f "
                            "cpu %.1f gpu %.1f us, CPU late in %d by %.1f us, keys cpu %.0f gpu %.0f, keys/us cpu %.1f gpu %.1f\n",
                            __func__, (unsigned long long) n, 100.0*ng/n, ggml_metal_coattn_fraction_now(), w_n,
                            w_stage/w_n, w_wake/w_n, w_cpu/w_n, w_gpu/w_n, w_late_n, w_late_n ? w_late/w_late_n : 0.0,
                            w_kc/w_n, w_kg/w_n, w_kc/w_cpu, w_kg/w_gpu);
                    // spread of the CPU's time per 1000 keys (p10 / p50 / p90), and the cores the compute threads ran on
                    const int nw = w_n < 256 ? w_n : 256;
                    float srt[256];
                    memcpy(srt, w_cpu_us, nw*sizeof(float));
                    for (int a = 1; a < nw; a++) { float x = srt[a]; int b = a - 1; while (b >= 0 && srt[b] > x) { srt[b + 1] = srt[b]; b--; } srt[b + 1] = x; }
                    char hist[256]; int o = 0;
                    for (int c = 0; c < 16; c++) {
                        const uint64_t h = atomic_exchange(&co_cpu_hist[c], 0);
                        if (h) o += snprintf(hist + o, sizeof(hist) - o, " c%d:%llu", c, (unsigned long long) h);
                    }
                    fprintf(stderr, "%s:   us per 1k CPU keys p10 %.1f p50 %.1f p90 %.1f | job pickups by core:%s\n", __func__,
                            srt[nw/10], srt[nw/2], srt[(nw*9)/10], hist);
                    w_stage = w_wake = w_cpu = w_gpu = w_late = w_kc = w_kg = 0; w_n = w_late_n = 0;
                }
            }
        }
        if (co_event.signaledValue < j.v_done) {
            co_event.signaledValue = j.v_done;
        }
    }
    return NULL;
}

static void co_start(void) {
    co_Q = aligned_alloc(128, (size_t) 4*NR*HD*sizeof(float));
    co_O = aligned_alloc(128, (size_t) 4*NR*HD*sizeof(float));
    co_m = aligned_alloc(128, (size_t) 4*NR*sizeof(float));
    co_l = aligned_alloc(128, (size_t) 4*NR*sizeof(float));
    for (int w = 0; w < CO_NW; w++) {
        co_pipe[w] = sme_pipe_new(CO_NH);
    }
    for (int t = 0; t < CO_NT; t++) {
        pthread_t th;
        pthread_create(&th, NULL, co_compute_thread, (void *) (intptr_t) t);
        pthread_detach(th);
    }
    pthread_t th;
    pthread_create(&th, NULL, co_coord_thread, NULL);
    pthread_detach(th);
    fprintf(stderr, "%s: SME co-attention started: %d SME workers x (1 + %d pack + 1 softmax), fraction %.2f\n",
            __func__, CO_NW, CO_NH, ggml_metal_coattn_fraction());
}

void ggml_metal_coattn_submit(const struct ggml_metal_coattn_job * job) {
    pthread_once(&co_once, co_start);
    pthread_mutex_lock(&co_mu);
    if (co_nq >= CO_MAXQ) {
        pthread_mutex_unlock(&co_mu);
        fprintf(stderr, "%s: job queue full\n", __func__);
        abort();
    }
    co_q[co_nq++] = *job;
    atomic_fetch_add_explicit(&co_pending, 1, memory_order_relaxed);
    pthread_mutex_unlock(&co_mu);
}
