// SME prefill matmuls (infernet): see ggml-metal-mmsme.h.
//
// The kernel is scripts/sme/sme_ffn.c (infernet, 2026-09-23), with persistent threads: per SME worker (one per P-cluster
// SME unit) H NEON helpers dequantize 64 weight rows at a time into an fp16 panel ring, the worker multiplies each panel
// with the packed fp16 activations (FMOPA fp16 -> fp32, 4 ZA tiles = 16 tokens x 64 outputs) and stores into dst.
// Same precision class as the GPU's kernel_mul_mm (fp16 weights and activations, fp32 accumulation); only the
// summation order differs.
//
// Compiled with -mcpu=apple-m4 (SME2 intrinsics); nothing here runs unless ggml_metal_mmsme_fraction() > 0.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ggml-metal-mmsme.h"

#include <arm_neon.h>
#include <arm_sme.h>
#include <math.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>

typedef __fp16 f16;

#define MS_QK_K   256
#define MS_PN     64            // output rows per panel (4 ZA tiles x 16)
#define MS_RING   4             // panels in flight per worker
#define MS_NW     2             // max SME workers (ms_nw: one per P-cluster SME unit)
#define MS_MAXH   6             // max helpers per worker
#define MS_KMAX   32768         // largest K (the 27B's ffn_down: 17408)
#define MS_TMAX   512           // largest CPU token share
#define MS_MAXQ   4096

typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[MS_QK_K / 64];
    uint8_t  qs[MS_QK_K / 2];
} ms_block_iq4_xs;              // = ggml block_iq4_xs, 136 bytes / 256 weights

static const int8_t ms_kvalues[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

static inline float ms_h2f(uint16_t h) { f16 x; memcpy(&x, &h, 2); return (float) x; }

// rows [n0, n0 + 64) of W (K per row) -> P[K/2][64][2] fp16 (rows >= nrows zero).
// Weights exactly as the GPU's dequantize_iq4_xs: (float) d * (ls - 32) * kvalue in fp32, rounded to fp16 once. Per
// 32-weight sub-block the 16 possible values are built that way and the nibbles look them up (two byte tables).

// one row's 32 weights of sub-block ib of block x as 16 (k, k+1) fp16 pairs: u[0..3] = pairs 0-3, 4-7, 8-11, 12-15
static inline void ms_dq32(const ms_block_iq4_xs * x, int ib, const float32x4_t kvf[4], uint32x4_t u[4]) {
    const float d = ms_h2f(x->d);
    const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
    const float dl = d * (float) (ls - 32);
    const float16x8_t t0 = vcombine_f16(vcvt_f16_f32(vmulq_n_f32(kvf[0], dl)), vcvt_f16_f32(vmulq_n_f32(kvf[1], dl)));
    const float16x8_t t1 = vcombine_f16(vcvt_f16_f32(vmulq_n_f32(kvf[2], dl)), vcvt_f16_f32(vmulq_n_f32(kvf[3], dl)));
    const uint8x16x2_t tb = vuzpq_u8(vreinterpretq_u8_f16(t0), vreinterpretq_u8_f16(t1));  // [0] low bytes, [1] high
    const uint8x16_t q  = vld1q_u8(x->qs + 16 * ib);
    const uint8x16_t lo = vandq_u8(q, vdupq_n_u8(0xf));   // weights 0..15
    const uint8x16_t hi = vshrq_n_u8(q, 4);                // weights 16..31
    const uint8x16x2_t wlo = vzipq_u8(vqtbl1q_u8(tb.val[0], lo), vqtbl1q_u8(tb.val[1], lo));
    const uint8x16x2_t whi = vzipq_u8(vqtbl1q_u8(tb.val[0], hi), vqtbl1q_u8(tb.val[1], hi));
    u[0] = vreinterpretq_u32_u8(wlo.val[0]); u[1] = vreinterpretq_u32_u8(wlo.val[1]);
    u[2] = vreinterpretq_u32_u8(whi.val[0]); u[3] = vreinterpretq_u32_u8(whi.val[1]);
}

// 4x4 transpose of uint32 lanes
static inline void ms_tr4(uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t o[4]) {
    const uint32x4x2_t ab = vtrnq_u32(a, b), cd = vtrnq_u32(c, d);
    o[0] = vcombine_u32(vget_low_u32 (ab.val[0]), vget_low_u32 (cd.val[0]));
    o[1] = vcombine_u32(vget_low_u32 (ab.val[1]), vget_low_u32 (cd.val[1]));
    o[2] = vcombine_u32(vget_high_u32(ab.val[0]), vget_high_u32(cd.val[0]));
    o[3] = vcombine_u32(vget_high_u32(ab.val[1]), vget_high_u32(cd.val[1]));
}

static void ms_dequant_panel(const ms_block_iq4_xs * W, int K, int n0, int nrows, f16 * P) {
    const int nb = K / MS_QK_K;
    float32x4_t kvf[4];
    for (int i = 0; i < 4; i++) {
        kvf[i] = (float32x4_t) { ms_kvalues[4*i + 0], ms_kvalues[4*i + 1], ms_kvalues[4*i + 2], ms_kvalues[4*i + 3] };
    }
    uint32_t * P32 = (uint32_t *) P;
    const uint32x4_t zero = vdupq_n_u32(0);
    // 4 rows at a time: each k pair's 4 values are one 16-byte store (instead of 4 scattered 4-byte stores)
    for (int r = 0; r < MS_PN; r += 4) {
        const ms_block_iq4_xs * row[4];
        for (int i = 0; i < 4; i++) row[i] = r + i < nrows ? W + (size_t) (n0 + r + i) * nb : NULL;
        if (!row[0]) {
            for (int kp = 0; kp < K / 2; kp++) vst1q_u32(P32 + (size_t) kp * MS_PN + r, zero);
            continue;
        }
        for (int b = 0; b < nb; b++) {
            for (int ib = 0; ib < MS_QK_K / 32; ib++) {
                uint32x4_t u[4][4];
                for (int i = 0; i < 4; i++) {
                    if (row[i]) ms_dq32(row[i] + b, ib, kvf, u[i]);
                    else { u[i][0] = u[i][1] = u[i][2] = u[i][3] = zero; }
                }
                uint32_t * dst = P32 + (size_t) ((b * MS_QK_K + ib * 32) / 2) * MS_PN + r;
                for (int j = 0; j < 4; j++) {          // pairs 4j .. 4j+3
                    uint32x4_t o[4];
                    ms_tr4(u[0][j], u[1][j], u[2][j], u[3][j], o);
                    vst1q_u32(dst + (size_t) (4 * j + 0) * MS_PN, o[0]);
                    vst1q_u32(dst + (size_t) (4 * j + 1) * MS_PN, o[1]);
                    vst1q_u32(dst + (size_t) (4 * j + 2) * MS_PN, o[2]);
                    vst1q_u32(dst + (size_t) (4 * j + 3) * MS_PN, o[3]);
                }
            }
        }
    }
}

// 4x4 transpose of uint32 lanes (defined below the dequant helpers' users; forward)
static inline void ms_tr4(uint32x4_t a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t o[4]);

// tokens [t0, t1) of X[T][K] f32 (row stride ldx) -> A[K/2][Tp][2] fp16; tokens >= T zero.
// 16 tokens at a time: per k pair one 64-byte store of the 16 tokens' (k, k+1) pairs (whole cache lines; one 4-byte store per
// token and k pair left the phone's pack of a 256 x 17408 X at ~4 ms, 2026-09-27)
static void ms_pack_x(const float * X, int T, int K, size_t ldx, int Tp, int t0, int t1, f16 * A) {
    uint32_t * A32 = (uint32_t *) A;
    const uint32x4_t zero = vdupq_n_u32(0);
    int t = t0;
    for (; t + 16 <= t1 && t < T; t += 16) {
        const float * x[16];
        for (int i = 0; i < 16; i++) x[i] = t + i < T ? X + (size_t) (t + i) * ldx : NULL;
        for (int k = 0; k < K; k += 8) {
            uint32x4_t o[4][4];
            for (int g = 0; g < 4; g++) {
                uint32x4_t u[4];
                for (int i = 0; i < 4; i++) {
                    const float * xi = x[4*g + i];
                    u[i] = xi ? vreinterpretq_u32_f16(vcombine_f16(vcvt_f16_f32(vld1q_f32(xi + k)), vcvt_f16_f32(vld1q_f32(xi + k + 4)))) : zero;
                }
                ms_tr4(u[0], u[1], u[2], u[3], o[g]);
            }
            for (int j = 0; j < 4; j++) {
                uint32_t * d = A32 + (size_t) (k/2 + j) * Tp + t;
                vst1q_u32(d, o[0][j]); vst1q_u32(d + 4, o[1][j]); vst1q_u32(d + 8, o[2][j]); vst1q_u32(d + 12, o[3][j]);
            }
        }
    }
    for (; t < t1; t++) {
        if (t >= T) { for (int kp = 0; kp < K / 2; kp++) A32[(size_t) kp * Tp + t] = 0; continue; }
        const float * x = X + (size_t) t * ldx;
        for (int k = 0; k < K; k += 8) {
            const float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(x + k)), vcvt_f16_f32(vld1q_f32(x + k + 4)));
            const uint32x4_t u = vreinterpretq_u32_f16(h);
            uint32_t * d = A32 + (size_t) (k / 2) * Tp + t;
            d[0] = vgetq_lane_u32(u, 0); d[Tp] = vgetq_lane_u32(u, 1); d[2 * Tp] = vgetq_lane_u32(u, 2); d[3 * Tp] = vgetq_lane_u32(u, 3);
        }
    }
}

// Y[t][n0 + 0..63] for all 16-token groups: A[K/2][Tp][2], P[K/2][64][2]
__arm_locally_streaming __arm_new("za")
static void ms_sme_panel(const f16 * A, int Tp, const f16 * P, int K, float * Y, size_t ldy, int T, int n0, int nvalid) {
    const svbool_t  ph = svptrue_b16();
    const svcount_t pc = svptrue_c16();
    const int KP = K / 2;
    for (int tg = 0; tg < Tp / 16; tg++) {
        svzero_za();
        const f16 * a = A + tg * 32;
        for (int kp = 0; kp < KP; kp++) {
            svfloat16_t   x = svld1_f16(ph, (const float16_t *) (a + (size_t) kp * Tp * 2));
            svfloat16x4_t w = svld1_f16_x4(pc, (const float16_t *) (P + (size_t) kp * MS_PN * 2));
            svmopa_za32_f16_m(0, ph, ph, x, svget4_f16(w, 0));
            svmopa_za32_f16_m(1, ph, ph, x, svget4_f16(w, 1));
            svmopa_za32_f16_m(2, ph, ph, x, svget4_f16(w, 2));
            svmopa_za32_f16_m(3, ph, ph, x, svget4_f16(w, 3));
        }
        const int nt = T - tg * 16 < 16 ? T - tg * 16 : 16;
        for (int r = 0; r < nt; r++) {
            float * y = Y + (size_t) (tg * 16 + r) * ldy + n0;
            for (int j = 0; j < 4; j++) {
                const int c0 = 16 * j;
                if (c0 >= nvalid) break;
                const svbool_t pw = svwhilelt_b32(0, nvalid - c0);
                if (j == 0) svst1_hor_za32(0, r, pw, y + c0);
                if (j == 1) svst1_hor_za32(1, r, pw, y + c0);
                if (j == 2) svst1_hor_za32(2, r, pw, y + c0);
                if (j == 3) svst1_hor_za32(3, r, pw, y + c0);
            }
        }
    }
}

// ---------------------------------------------------------------- persistent thread pool

typedef struct {
    int p_begin, p_end;
    f16 * ring[MS_RING];
    _Atomic int ready[MS_RING];     // panel index held by the slot, -1 = empty
    _Atomic int free_for[MS_RING];  // the panel this slot may be filled with next
    _Atomic int next;               // next panel for the helpers to claim
} ms_worker;

static ms_worker ms_w[MS_NW];
static int ms_nw = 2;
static int ms_nh = 3;
static f16 * ms_A;                  // packed activations [K/2][Tp][2]
static struct ggml_metal_mmsme_job ms_cur;
static int ms_Tp;

static _Atomic uint64_t ms_gen = 0;       // job generation published to the pool
// Cluster-aware roles (as ggml-metal-coattn.m): the M4 Pro has one SME unit per P-cluster (cores 4-8, 9-13) and macOS can't
// pin threads; with fixed roles both SME workers shared one unit in a fraction of the jobs (10-15 ms instead of ~8.4 for a
// 17408x5120 share, 2026-09-27). Every thread reports its core when it picks up a job; the coordinator hands out roles:
// one SME worker per P-cluster, each worker's helpers in its cluster.
#define MS_NTMAX (MS_NW*(1 + MS_MAXH))
static _Atomic int      ms_core[MS_NTMAX];
static _Atomic int      ms_role[MS_NTMAX];
static _Atomic int      ms_arrived = 0;
static _Atomic uint64_t ms_role_seq = 0;

// core topology from sysctl (E-cores first, then the P-clusters): M4 Pro 4 E + 2 x 5 P, A19 Pro 4 E + 1 x 2 P
static int ms_ne = 4, ms_np = 10, ms_per = 5;
static void ms_topology(void) {
    int v = 0; size_t n = sizeof v;
    if (sysctlbyname("hw.perflevel1.logicalcpu", &v, &n, NULL, 0) == 0) ms_ne = v;
    n = sizeof v;
    if (sysctlbyname("hw.perflevel0.logicalcpu", &v, &n, NULL, 0) == 0 && v > 0) ms_np = v;
    n = sizeof v;
    if (sysctlbyname("hw.perflevel0.cpusperl2", &v, &n, NULL, 0) == 0 && v > 0) ms_per = v;
}
// P-cluster index, or MS_NW for an E-core / unknown core
static int ms_cluster(int cpu) { return cpu >= ms_ne && cpu < ms_ne + ms_np ? (cpu - ms_ne)/ms_per : MS_NW; }

static void ms_assign_roles(int nt) {
    // bucket w < ms_nw: cores of worker w's P-cluster; bucket ms_nw: everything else (E-cores, P-clusters without a worker)
    int L[MS_NW + 1][MS_NTMAX], nL[MS_NW + 1] = { 0 }, iL[MS_NW + 1] = { 0 };
    for (int t = 0; t < nt; t++) {
        int c = ms_cluster(atomic_load_explicit(&ms_core[t], memory_order_relaxed));
        if (c >= ms_nw) c = ms_nw;
        L[c][nL[c]++] = t;
    }
    // worker w's threads: its own cluster first, then the others bucket, then another worker's cluster
    #define MS_POP(w, out) do { \
        if (iL[(w)] < nL[(w)]) { (out) = L[(w)][iL[(w)]++]; break; } \
        if (iL[ms_nw] < nL[ms_nw]) { (out) = L[ms_nw][iL[ms_nw]++]; break; } \
        for (int b_ = 0; b_ < ms_nw; b_++) if (iL[b_] < nL[b_]) { (out) = L[b_][iL[b_]++]; break; } \
    } while (0)
    const int tpw = nt/ms_nw;
    int role_of[MS_NTMAX];
    for (int r = 0; r < tpw; r++) {
        for (int w = 0; w < ms_nw; w++) { int t = 0; MS_POP(w, t); role_of[t] = w*tpw + r; }
    }
    #undef MS_POP
    for (int t = 0; t < nt; t++) atomic_store_explicit(&ms_role[t], role_of[t], memory_order_relaxed);
}
static _Atomic int ms_done = 0;           // threads finished with the current generation
static _Atomic int ms_packed = 0;         // pool threads done packing X for the current generation
static pthread_mutex_t ms_pmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  ms_pcv = PTHREAD_COND_INITIALIZER;

static id<MTLSharedEvent> ms_event = nil;
static id<MTLSharedEvent> ms_event_gpu = nil;

static pthread_mutex_t ms_mu = PTHREAD_MUTEX_INITIALIZER;
static struct ggml_metal_mmsme_job ms_q[MS_MAXQ];
static int ms_nq = 0;
static _Atomic int ms_pending = 0;
static pthread_cond_t ms_qcv = PTHREAD_COND_INITIALIZER;   // signalled on submit (the coordinator sleeps on it when idle)
static pthread_once_t ms_once = PTHREAD_ONCE_INIT;

static uint64_t ms_now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static int ms_nt(void) { return ms_nw * (1 + ms_nh); }

// wait for a new generation. While jobs are queued (a prefill ubatch's matmuls are all queued when it is encoded) the threads
// wait with WFE on ms_gen's cache line (low power, woken in ~0.1 us by the coordinator's store) so they stay hot between
// layers: sleeping on the condition variable in the ~2-5 ms of attention between two layers' FFNs left the next job on
// clocked-down cores (the first matmul of each FFN pair took 11-21 ms instead of ~8.6, 2026-09-27). With nothing queued:
// the same WFE wait for 2 ms, then the condition variable.
static inline uint64_t ms_ldx(_Atomic uint64_t * p) {
    uint64_t v;
    __asm__ volatile("ldaxr %0, [%1]" : "=r"(v) : "r"(p) : "memory");
    return v;
}

static int ms_spin_pending(void) {
    static int v = -1;
    if (v < 0) v = getenv("GGML_METAL_MM_SME_SPIN_PENDING") ? atoi(getenv("GGML_METAL_MM_SME_SPIN_PENDING")) : 1;
    return v;
}

static uint64_t ms_wait_gen(uint64_t seen) {
    uint64_t t0 = ms_now_ns();
    for (;;) {
        const uint64_t g = ms_ldx(&ms_gen);
        if (g != seen) { __asm__ volatile("clrex" ::: "memory"); return g; }
        if (ms_spin_pending() && atomic_load_explicit(&ms_pending, memory_order_relaxed) > 0) {
            t0 = ms_now_ns();
            __asm__ volatile("wfe" ::: "memory");
            continue;
        }
        if (ms_now_ns() - t0 < 2000000ull) { __asm__ volatile("wfe" ::: "memory"); continue; }
        __asm__ volatile("clrex" ::: "memory");
        pthread_mutex_lock(&ms_pmu);
        while (atomic_load_explicit(&ms_gen, memory_order_acquire) == seen &&
               (!ms_spin_pending() || atomic_load_explicit(&ms_pending, memory_order_relaxed) == 0)) {
            pthread_cond_wait(&ms_pcv, &ms_pmu);
        }
        pthread_mutex_unlock(&ms_pmu);
        t0 = ms_now_ns();
    }
}

static void ms_helper(ms_worker * w, const struct ggml_metal_mmsme_job * j) {
    for (;;) {
        const int p = atomic_fetch_add(&w->next, 1);
        if (p >= w->p_end) return;
        const int slot = (p - w->p_begin) % MS_RING;
        while (atomic_load_explicit(&w->free_for[slot], memory_order_acquire) != p) { __builtin_arm_yield(); }
        const int n0 = p * MS_PN;
        ms_dequant_panel((const ms_block_iq4_xs *) j->w, j->k, n0, j->n - n0 < MS_PN ? j->n - n0 : MS_PN, w->ring[slot]);
        atomic_store_explicit(&w->ready[slot], p, memory_order_release);
    }
}

static void ms_worker_run(ms_worker * w, const struct ggml_metal_mmsme_job * j) {
    for (int p = w->p_begin; p < w->p_end; p++) {
        const int slot = (p - w->p_begin) % MS_RING;
        while (atomic_load_explicit(&w->ready[slot], memory_order_acquire) != p) { __builtin_arm_yield(); }
        const int n0 = p * MS_PN;
        ms_sme_panel(ms_A, ms_Tp, w->ring[slot], j->k, j->y, j->ldy, j->t, n0, j->n - n0 < MS_PN ? j->n - n0 : MS_PN);
        atomic_store_explicit(&w->ready[slot], -1, memory_order_relaxed);
        atomic_store_explicit(&w->free_for[slot], p + MS_RING, memory_order_release);
    }
}

// thread t: w = t / (1 + nh), role r = t % (1 + nh) (0 = SME worker, else helper). Every thread first packs its slice of X.
static void * ms_thread(void * arg) {
    const int t = (int) (intptr_t) arg;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    uint64_t seen = 0;
    for (;;) {
        seen = ms_wait_gen(seen);
        const struct ggml_metal_mmsme_job * j = &ms_cur;
        const int nt = ms_nt();
        // pack tokens in slices; the worker waits for all slices before its first panel
        const int per = (ms_Tp/16 + nt - 1) / nt * 16;
        const int t0 = t * per < ms_Tp ? t * per : ms_Tp, t1 = t0 + per < ms_Tp ? t0 + per : ms_Tp;
        if (t0 < t1) ms_pack_x(j->x, j->t, j->k, j->ldx, ms_Tp, t0, t1, ms_A);
        atomic_fetch_add_explicit(&ms_packed, 1, memory_order_acq_rel);
        size_t cpu = 99;
        if (pthread_cpu_number_np(&cpu) != 0 || cpu >= 16) cpu = 99;
        atomic_store_explicit(&ms_core[t], (int) cpu, memory_order_relaxed);
        atomic_fetch_add_explicit(&ms_arrived, 1, memory_order_release);
        while (atomic_load_explicit(&ms_role_seq, memory_order_acquire) != seen) { __builtin_arm_yield(); }
        const int role = atomic_load_explicit(&ms_role[t], memory_order_relaxed);
        const int w = role / (1 + ms_nh), r = role % (1 + ms_nh);
        if (r == 0) {
            while (atomic_load_explicit(&ms_packed, memory_order_acquire) < nt) { __builtin_arm_yield(); }
            ms_worker_run(&ms_w[w], j);
        } else {
            ms_helper(&ms_w[w], j);
        }
        atomic_fetch_add_explicit(&ms_done, 1, memory_order_acq_rel);
    }
    return NULL;
}

// returns when the CPU's share is written; *pack_ns: until every thread had packed X (all arrived)
static void ms_run(const struct ggml_metal_mmsme_job * j, uint64_t * pack_ns) {
    const uint64_t t0 = ms_now_ns();
    ms_cur = *j;
    ms_Tp = (j->t + 15) / 16 * 16;
    const int np = (j->n + MS_PN - 1) / MS_PN;
    for (int i = 0; i < ms_nw; i++) {
        ms_worker * w = &ms_w[i];
        w->p_begin = np * i / ms_nw; w->p_end = np * (i + 1) / ms_nw;
        for (int s = 0; s < MS_RING; s++) {
            atomic_store(&w->ready[s], -1);
            atomic_store(&w->free_for[s], w->p_begin + s);
        }
        atomic_store(&w->next, w->p_begin);
    }
    atomic_store(&ms_done, 0);
    atomic_store(&ms_packed, 0);
    atomic_store(&ms_arrived, 0);
    pthread_mutex_lock(&ms_pmu);
    const uint64_t gen = atomic_fetch_add_explicit(&ms_gen, 1, memory_order_release) + 1;
    pthread_cond_broadcast(&ms_pcv);
    pthread_mutex_unlock(&ms_pmu);
    const int nt = ms_nt();
    while (atomic_load_explicit(&ms_arrived, memory_order_acquire) < nt) { __builtin_arm_yield(); }
    *pack_ns = ms_now_ns() - t0;
    ms_assign_roles(nt);
    atomic_store_explicit(&ms_role_seq, gen, memory_order_release);
    while (atomic_load_explicit(&ms_done, memory_order_acquire) < nt) { __builtin_arm_yield(); }
}

// ---------------------------------------------------------------- adaptive split (GGML_METAL_MM_SME_ADAPT, default 1)
// Per shape (n, k, t_all): CPU time ~ a + b*t (dequant is a fixed cost per matmul, the multiply scales with tokens), GPU
// time ~ c*(t_all - t). Each job updates EMAs of the CPU's time at the t it ran and the GPU's time per token; the next t
// is the largest tile multiple where the CPU still finishes 5% before the GPU (the GPU never waits on it).
typedef struct { int n, k, t_all, rows; _Atomic int t; double cpu_ms, gpu_ms_tok, pack_ms; int t_meas; int cnt; } ms_shape;
#define MS_NSHAPE 32
static ms_shape ms_sh[MS_NSHAPE];
static int ms_nsh = 0;
static pthread_mutex_t ms_shmu = PTHREAD_MUTEX_INITIALIZER;

static int ms_stats_on(void);
static int ms_cluster(int cpu);
static int ms_nt(void);

static int ms_adapt_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("GGML_METAL_MM_SME_ADAPT") ? atoi(getenv("GGML_METAL_MM_SME_ADAPT")) : 1;
    return v;
}

static ms_shape * ms_shape_get(int n, int k, int t_all, int rows, int t0) {
    for (int i = 0; i < ms_nsh; i++) {
        if (ms_sh[i].n == n && ms_sh[i].k == k && ms_sh[i].t_all == t_all && ms_sh[i].rows == rows) return &ms_sh[i];
    }
    if (ms_nsh == MS_NSHAPE) return NULL;
    ms_shape * s = &ms_sh[ms_nsh++];
    memset(s, 0, sizeof(*s));
    s->n = n; s->k = k; s->t_all = t_all; s->rows = rows; atomic_store(&s->t, t0);
    return s;
}

int ggml_metal_mmsme_tokens(int n, int k, int t_all, int tile) {
    const float frac = ggml_metal_mmsme_fraction();
    int t0 = ((int) (frac*t_all)/tile)*tile;
    if (!ms_adapt_on()) return t0;
    pthread_mutex_lock(&ms_shmu);
    ms_shape * s = ms_shape_get(n, k, t_all, 0, t0);
    const int t = s ? atomic_load(&s->t) : t0;
    pthread_mutex_unlock(&ms_shmu);
    return t;
}

int ggml_metal_mmsme_rows_mode(void) {
    static int v = -1;
    if (v < 0) v = getenv("GGML_METAL_MM_SME_ROWS") ? atoi(getenv("GGML_METAL_MM_SME_ROWS")) : 0;
    return v;
}

int ggml_metal_mmsme_rows(int n, int k, int t_all, int tile) {
    const float frac = ggml_metal_mmsme_fraction();
    const int r0 = ((int) (frac*n)/tile)*tile;
    if (!ms_adapt_on()) return r0;
    pthread_mutex_lock(&ms_shmu);
    ms_shape * s = ms_shape_get(n, k, t_all, 1, r0);
    const int r = s ? atomic_load(&s->t) : r0;
    pthread_mutex_unlock(&ms_shmu);
    return r;
}

// rows mode: CPU time = pack (measured) + b*rows, GPU time = c*(n_all - rows); the next share is the largest tile multiple
// where the CPU finishes 5% before the GPU. Steps are a quarter of the distance (at least a tile), so it converges within
// a ubatch but one noisy job can't swing it.
static void ms_adapt_rows(const struct ggml_metal_mmsme_job * j, double cpu_ms, double pack_ms, double gpu_ms, int gpu_exact) {
    ms_shape * s = ms_shape_get(j->n_all, j->k, j->t_all, 1, j->n);
    if (!s) return;
    const double g = gpu_ms/(j->n_all - j->n);
    if (s->cnt == 0) { s->gpu_ms_tok = g; }
    else if (gpu_exact || g < s->gpu_ms_tok) { s->gpu_ms_tok = 0.8*s->gpu_ms_tok + 0.2*g; }
    const double b = (cpu_ms - pack_ms)/j->n;
    s->cpu_ms  = s->cnt == 0 ? b       : 0.8*s->cpu_ms  + 0.2*b;     // per row
    s->pack_ms = s->cnt == 0 ? pack_ms : 0.8*s->pack_ms + 0.2*pack_ms;
    s->cnt++;
    if (s->cnt < 3) return;
    const int tile = j->tile;
    int best = 0;
    for (int r = tile; r <= j->n_all - tile; r += tile) {
        if (s->pack_ms + s->cpu_ms*r <= 0.95*s->gpu_ms_tok*(j->n_all - r)) best = r;
    }
    const int cur = atomic_load(&s->t);
    int step = abs(best - cur)/4/tile*tile;
    if (step < tile) step = tile;
    int next = best > cur ? (cur + step < best ? cur + step : best) : (best < cur ? (cur - step > best ? cur - step : best) : cur);
    if (!gpu_exact && j->n == cur) {
        // the CPU finished last: the GPU's time is only a bound, so the model can't see how early it was. Back off 1/8.
        const int back = (cur/8/tile)*tile;
        if (cur - (back > tile ? back : tile) < next) next = cur - (back > tile ? back : tile);
        if (next < 0) next = 0;
    }
    if (ms_stats_on() >= 2) {
        char cores[96]; int o = 0;
        for (int t = 0; t < ms_nt() && o < (int) sizeof(cores) - 8; t++) {
            o += snprintf(cores + o, sizeof(cores) - o, "%d%s", atomic_load(&ms_core[t]), atomic_load(&ms_role[t]) % (1 + ms_nh) == 0 ? "W " : " ");
        }
        fprintf(stderr, "mmsme rows %dx%d t %d: ran %d rows cpu %.2f ms (pack %.2f) gpu %.2f ms (%s) -> best %d, next %d | cores %s\n",
                j->n_all, j->k, j->t_all, j->n, cpu_ms, pack_ms, gpu_ms, gpu_exact ? "exact" : "bound", best, next, cores);
    }
    atomic_store(&s->t, next);
}

static void ms_adapt(const struct ggml_metal_mmsme_job * j, double cpu_ms, double gpu_ms, int gpu_exact, int tile) {
    pthread_mutex_lock(&ms_shmu);
    ms_shape * s = ms_shape_get(j->n, j->k, j->t_all, 0, j->t);
    if (s) {
        const double g = gpu_ms/(j->t_all - j->t);
        // the GPU's time is exact when the CPU finished first (the coordinator then waits for the GPU's event); when the
        // CPU finished last it is only an upper bound (the GPU was done by then): then only trust it downward
        if (s->cnt == 0) { s->gpu_ms_tok = g; }
        else if (gpu_exact || g < s->gpu_ms_tok) { s->gpu_ms_tok = 0.8*s->gpu_ms_tok + 0.2*g; }
        s->cpu_ms = s->cnt == 0 || s->t_meas != j->t ? cpu_ms : 0.8*s->cpu_ms + 0.2*cpu_ms;
        s->t_meas = j->t;
        s->cnt++;
        if (s->cnt >= 3) {
            // CPU time per extra token from the dequant-fixed model: a = 45% of the measured time at t (measured 2026-09-27:
            // 17408x5120 t 96 -> 8.2 ms, t 160 -> 8.8 ms at 3 helpers: the fixed part dominates), b = rest / t
            const double a = 0.45*s->cpu_ms, b = (s->cpu_ms - a)/j->t;
            int best = 0;
            for (int t = tile; t <= j->t_all - tile && t <= 512; t += tile) {
                if (a + b*t <= 0.95*s->gpu_ms_tok*(j->t_all - t)) best = t;
            }
            // move one tile per update, so a noisy job can't swing the split
            const int cur = atomic_load(&s->t);
            if (ms_stats_on() >= 2 && j->n == 17408) {
                // cores at pickup: per-worker cluster of the worker, and how many threads were on E-cores
                char cores[64]; int o = 0, ne = 0;
                for (int t = 0; t < ms_nt(); t++) {
                    const int c = atomic_load(&ms_core[t]);
                    ne += ms_cluster(c) == MS_NW;
                    o += snprintf(cores + o, sizeof(cores) - o, "%d%s", c, atomic_load(&ms_role[t]) % (1 + ms_nh) == 0 ? "W " : " ");
                }
                fprintf(stderr, "mmsme adapt %dx%d t_all %d: ran t %d cpu %.2f ms gpu %.2f ms (%s) -> best %d (cur %d) | E-core threads %d | cores %s\n",
                        j->n, j->k, j->t_all, j->t, cpu_ms, gpu_ms, gpu_exact ? "exact" : "bound", best, cur, ne, cores);
            }
            atomic_store(&s->t, best > cur ? cur + tile : (best < cur ? cur - tile : cur));
        }
    }
    pthread_mutex_unlock(&ms_shmu);
}

// ---------------------------------------------------------------- stats (GGML_METAL_MM_SME_STATS=1)

static int ms_stats_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("GGML_METAL_MM_SME_STATS") ? atoi(getenv("GGML_METAL_MM_SME_STATS")) : 0;
    return v;
}

typedef struct { int n, k, t, t_all; double cpu_ms, gpu_ms, late_ms; int cnt, late_cnt; } ms_stat;
static ms_stat ms_st[16];
static int ms_nst = 0;
static uint64_t ms_jobs = 0;

static void ms_stat_add(const struct ggml_metal_mmsme_job * j, double cpu_ms, double gpu_ms) {
    int i = 0;
    for (; i < ms_nst; i++) if (ms_st[i].n == j->n && ms_st[i].k == j->k && ms_st[i].t == j->t && ms_st[i].t_all == j->t_all) break;
    if (i == ms_nst) {
        if (ms_nst == 16) return;
        ms_nst++;
        memset(&ms_st[i], 0, sizeof(ms_st[i]));
        ms_st[i].n = j->n; ms_st[i].k = j->k; ms_st[i].t = j->t; ms_st[i].t_all = j->t_all;
    }
    ms_st[i].cpu_ms += cpu_ms; ms_st[i].gpu_ms += gpu_ms; ms_st[i].cnt++;
    if (cpu_ms > gpu_ms) { ms_st[i].late_ms += cpu_ms - gpu_ms; ms_st[i].late_cnt++; }
    if (++ms_jobs % 240 == 0) {
        for (int s = 0; s < ms_nst; s++) {
            const ms_stat * q = &ms_st[s];
            if (!q->cnt) continue;
            fprintf(stderr, "mmsme: %5d x %5d, cpu %3d of %4d tokens: %4d jobs, cpu %.2f ms (%.2f TFLOPS), gpu share %.2f ms, "
                    "GPU waited in %d by %.2f ms\n", q->n, q->k, q->t, q->t_all, q->cnt, q->cpu_ms / q->cnt,
                    2.0 * q->n * q->k * q->t / (q->cpu_ms / q->cnt * 1e-3) * 1e-12, q->gpu_ms / q->cnt, q->late_cnt,
                    q->late_cnt ? q->late_ms / q->late_cnt : 0.0);
        }
        ms_nst = 0;
    }
}

// GGML_METAL_MM_SME_CHECK=1 (debug): compare sampled outputs of the CPU's rows against a scalar reference that
// dequantizes like the GPU (fp32 product, one fp16 rounding) and rounds X to fp16, accumulating in double
// gpu_rows (rows mode, after the GPU's share is done): check the GPU's rows [0, n_all - n) instead, which sit before the
// CPU's in W and dst (the GPU kernel's strided dst store)
static void ms_check(const struct ggml_metal_mmsme_job * j, int gpu_rows) {
    const int nb = j->k / MS_QK_K;
    const int off = gpu_rows ? j->n_all - j->n : 0;
    const int nr  = gpu_rows ? off : j->n;
    double e_max = 0, e_sum = 0; int cnt = 0;
    for (int t = 0; t < j->t; t += 7) {
        for (int n = 0; n < nr; n += 997) {
            const ms_block_iq4_xs * row = (const ms_block_iq4_xs *) j->w + ((ptrdiff_t) n - off) * nb;
            double acc = 0, mag = 0;
            for (int b = 0; b < nb; b++) {
                const ms_block_iq4_xs * x = row + b;
                const float d = ms_h2f(x->d);
                for (int ib = 0; ib < 8; ib++) {
                    const int ls = ((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4);
                    const float dl = d * (float) (ls - 32);
                    for (int i = 0; i < 32; i++) {
                        const uint8_t q = x->qs[16 * ib + (i % 16)];
                        const int v = ms_kvalues[i < 16 ? (q & 0xf) : (q >> 4)];
                        const float w = (float) (f16) (dl * (float) v);
                        const float xv = (float) (f16) j->x[(size_t) t * j->ldx + b * MS_QK_K + ib * 32 + i];
                        acc += (double) w * xv; mag += fabs((double) w * xv);
                    }
                }
            }
            const double e = fabs(j->y[(ptrdiff_t) t * j->ldy + n - off] - acc) / fmax(mag, 1e-20);
            e_max = fmax(e_max, e); e_sum += e; cnt++;
        }
    }
    fprintf(stderr, "mmsme check (%s): %d x %d, t %d: max rel err %.3g, mean %.3g (relative to sum |w x|, %d samples)\n",
            gpu_rows ? "GPU rows" : "CPU rows", nr, j->k, j->t, e_max, e_sum / (cnt ? cnt : 1), cnt);
}

// ---------------------------------------------------------------- coordinator

static void * ms_coord(void * arg) {
    (void) arg;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (;;) {
        struct ggml_metal_mmsme_job j;
        uint64_t t_idle = ms_now_ns();
        for (;;) {
            int best = -1;
            pthread_mutex_lock(&ms_mu);
            for (int i = 0; i < ms_nq; i++) {
                if (best < 0 || ms_q[i].v_start < ms_q[best].v_start) best = i;
            }
            if (best >= 0 && ms_event.signaledValue >= ms_q[best].v_start) {
                j = ms_q[best];
                ms_q[best] = ms_q[--ms_nq];
                atomic_fetch_sub_explicit(&ms_pending, 1, memory_order_relaxed);
                pthread_mutex_unlock(&ms_mu);
                break;
            }
            if (best >= 0 && ms_stats_on() >= 3) {
                static uint64_t t_best = 0, v_best = 0;
                if (v_best != ms_q[best].v_start) { v_best = ms_q[best].v_start; t_best = ms_now_ns(); }
                if (ms_now_ns() - t_best > 1000000000ull) {
                    fprintf(stderr, "mmsme: coordinator has waited 1 s for job v_start %llu (event at %llu, %d queued)\n",
                            (unsigned long long) ms_q[best].v_start, (unsigned long long) ms_event.signaledValue, ms_nq);
                    t_best = ms_now_ns();
                }
            }
            if (best < 0 && ms_now_ns() - t_idle >= 2000000ull) {
                // idle (decode, or between requests): sleep until a job is queued. No polling: a busy CPU core pulls the
                // shared power budget and the GPU clock down (seen with co-attention, 1578 -> 1492 MHz)
                pthread_cond_wait(&ms_qcv, &ms_mu);
                pthread_mutex_unlock(&ms_mu);
                t_idle = ms_now_ns();
                continue;
            }
            pthread_mutex_unlock(&ms_mu);
            if (best >= 0) t_idle = ms_now_ns();   // a job is queued: spin, the GPU is about to reach it
            __builtin_arm_yield();
        }
        const uint64_t t0 = ms_now_ns();
        // diagnostic: was the GPU's own share already done when the CPU saw the start? (then the two ran one after the other)
        const int gpu_done_at_start = ms_event_gpu.signaledValue >= j.v_start;
        static uint64_t n_late_start = 0, n_starts = 0;
        n_starts++; n_late_start += gpu_done_at_start;
        uint64_t pack_ns = 0;
        ms_run(&j, &pack_ns);
        __sync_synchronize();
        const uint64_t t1 = ms_now_ns();
        static int check = -1;
        if (check < 0) check = getenv("GGML_METAL_MM_SME_CHECK") ? atoi(getenv("GGML_METAL_MM_SME_CHECK")) : 0;
        const int check_now = check > 0;
        if (check_now) { check--; ms_check(&j, 0); }
        // signal first (the GPU may already be waiting), then learn when the GPU finished its own share
        const int gpu_first = ms_event_gpu.signaledValue >= j.v_start;
        if (check_now && j.rows) {
            // before v_done: dst can't be consumed (and its memory reused) while it is checked
            while (ms_event_gpu.signaledValue < j.v_start) { __builtin_arm_yield(); }
            ms_check(&j, 1);
        }
        if (ms_event.signaledValue < j.v_done) {
            ms_event.signaledValue = j.v_done;
        }
        if (ms_adapt_on() || ms_stats_on()) {
            while (ms_event_gpu.signaledValue < j.v_start) { __builtin_arm_yield(); }
            const uint64_t t2 = ms_now_ns();
            const double cpu_ms = (t1 - t0) * 1e-6, gpu_ms = gpu_first ? cpu_ms : (t2 - t0) * 1e-6;
            if (ms_adapt_on()) {
                if (j.rows) {
                    pthread_mutex_lock(&ms_shmu);
                    ms_adapt_rows(&j, cpu_ms, pack_ns*1e-6, gpu_ms, !gpu_first);
                    pthread_mutex_unlock(&ms_shmu);
                    if (ms_stats_on() >= 2 && n_starts % 50 == 0) {
                        fprintf(stderr, "mmsme: %llu of %llu jobs started after the GPU's share was already done\n",
                                (unsigned long long) n_late_start, (unsigned long long) n_starts);
                    }
                } else {
                    ms_adapt(&j, cpu_ms, gpu_ms, !gpu_first, j.tile);
                }
            }
            if (ms_stats_on()) ms_stat_add(&j, cpu_ms, gpu_ms);
        }
    }
    return NULL;
}

static void ms_start(void) {
    ms_topology();
    const int npc = (ms_np + ms_per - 1)/ms_per;
    const char * sw = getenv("GGML_METAL_MM_SME_WORKERS");
    ms_nw = sw ? atoi(sw) : npc;
    ms_nw = ms_nw < 1 ? 1 : (ms_nw > MS_NW ? MS_NW : ms_nw);
    const char * s = getenv("GGML_METAL_MM_SME_HELPERS");
    ms_nh = s ? atoi(s) : 3;
    ms_nh = ms_nh < 1 ? 1 : (ms_nh > MS_MAXH ? MS_MAXH : ms_nh);
    for (int i = 0; i < ms_nw; i++) {
        for (int r = 0; r < MS_RING; r++) {
            ms_w[i].ring[r] = aligned_alloc(128, (size_t) MS_KMAX * MS_PN * sizeof(f16));
        }
    }
    ms_A = aligned_alloc(128, (size_t) MS_KMAX * MS_TMAX * sizeof(f16));
    for (int t = 0; t < ms_nt(); t++) {
        pthread_t th;
        pthread_create(&th, NULL, ms_thread, (void *) (intptr_t) t);
        pthread_detach(th);
    }
    pthread_t th;
    pthread_create(&th, NULL, ms_coord, NULL);
    pthread_detach(th);
    fprintf(stderr, "%s: SME prefill matmuls started: %d SME workers x %d dequant helpers (cores: %d E, %d P in clusters of %d), "
            "%s split, fraction %.2f of ubatches >= %d tokens\n", __func__, ms_nw, ms_nh, ms_ne, ms_np, ms_per,
            ggml_metal_mmsme_rows_mode() ? "rows" : "tokens", (double) ggml_metal_mmsme_fraction(), ggml_metal_mmsme_min_tokens());
}

// ---------------------------------------------------------------- API

static int ms_sme2_available(void) {
    int v = 0, svl = 0;
    size_t n = sizeof v, n2 = sizeof svl;
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &v, &n, NULL, 0) != 0 || !v) return 0;
    if (sysctlbyname("hw.optional.arm.sme_max_svl_b", &svl, &n2, NULL, 0) != 0) return 0;
    return svl == 64;
}

float ggml_metal_mmsme_fraction(void) {
    static float frac = -1.0f;
    if (frac < 0.0f) {
        const char * s = getenv("GGML_METAL_MM_SME");
        float f = s ? (float) atof(s) : 0.0f;
        if (f > 0.0f && !ms_sme2_available()) {
            fprintf(stderr, "%s: GGML_METAL_MM_SME set but SME2 (512-bit) is not available: disabled\n", __func__);
            f = 0.0f;
        }
        frac = f < 0.0f ? 0.0f : (f > 0.75f ? 0.75f : f);
    }
    return frac;
}

int ggml_metal_mmsme_min_tokens(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = getenv("GGML_METAL_MM_SME_MIN");
        v = s ? atoi(s) : 256;
        if (v < 32) v = 32;
    }
    return v;
}

void * ggml_metal_mmsme_event(void * mtl_device) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (ms_event == nil) {
        ms_event = [(id<MTLDevice>) mtl_device newSharedEvent];
        ms_event.signaledValue = 0;
    }
    pthread_mutex_unlock(&mu);
    return (void *) ms_event;
}

void * ggml_metal_mmsme_event_gpu(void * mtl_device) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (ms_event_gpu == nil) {
        ms_event_gpu = [(id<MTLDevice>) mtl_device newSharedEvent];
        ms_event_gpu.signaledValue = 0;
    }
    pthread_mutex_unlock(&mu);
    return (void *) ms_event_gpu;
}

void ggml_metal_mmsme_submit(const struct ggml_metal_mmsme_job * job) {
    pthread_once(&ms_once, ms_start);
    if (job->k > MS_KMAX || job->t > MS_TMAX || job->k % MS_QK_K != 0) {
        fprintf(stderr, "%s: unsupported job (k %d, t %d)\n", __func__, job->k, job->t);
        abort();
    }
    pthread_mutex_lock(&ms_mu);
    if (ms_nq >= MS_MAXQ) {
        pthread_mutex_unlock(&ms_mu);
        fprintf(stderr, "%s: job queue full\n", __func__);
        abort();
    }
    ms_q[ms_nq++] = *job;
    atomic_fetch_add_explicit(&ms_pending, 1, memory_order_relaxed);
    pthread_cond_signal(&ms_qcv);
    pthread_mutex_unlock(&ms_mu);
    pthread_mutex_lock(&ms_pmu);
    pthread_cond_broadcast(&ms_pcv);
    pthread_mutex_unlock(&ms_pmu);
}
