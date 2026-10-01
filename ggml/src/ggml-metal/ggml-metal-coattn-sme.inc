// sme_attn.c - SME2 "co-attention" partial for the Qwen3.8-27B verify shape (infernet, 2026-09-23).
//
// One KV head: 48 query rows (8 tokens x GQA 6) x head dim 256 against keys [k0, k1).
// Output: the unnormalised partial (O[48][256], m[48], l[48]) that a log-sum-exp merge takes:
//   O = sum_j exp(s_j - m) v_j,  l = sum_j exp(s_j - m),  m = max_j s_j,  s_j = scale * q . k_j
//
// Split of work inside one thread:
//   NEON (non-streaming): pack K into the "16 keys x d-pair" layout FMOPA wants (fp16 copy or q8_0
//                         dequant), online softmax on S^T (rows in lanes), pack P and V (key-pair zip).
//   SME  (streaming)    : S^T = Q K^T with FMOPA fp16->fp32 (4 ZA tiles = 16 rows x 64 keys),
//                         O_blk = P V with FMOPA fp16->fp32 (4 ZA tiles = 16 rows x 64 d).
//
// K/V cache layout = llama.cpp's with flash attention (not transposed): row per key, the 4 KV heads
// next to each other, head dim contiguous. fp16: 256 halfs/head. q8_0: 8 blocks x 34 B/head.
//
// Build (macOS):  clang -O3 -mcpu=apple-m4 -DSME_BENCH_MAIN sme_attn.c -o sme_attn
// Build (iOS)  :  xcrun --sdk iphoneos clang -O3 -arch arm64 -mcpu=apple-m4 -DSME_BENCH_MAIN \
//                   -miphoneos-version-min=18.0 sme_attn.c -o sme_attn_ios
// Without -DSME_BENCH_MAIN it is a library file: call sme_attn_partial() (see the header comment on it).
#include <arm_neon.h>
#include <arm_sme.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NR 48       // query rows per KV head
#define HD 256      // head dim
#define BK_MAX 1024
#ifndef PFD
#define PFD 16      // prefetch distance in keys
#endif // keys per online-softmax block (multiple of 64)

typedef __fp16 f16;

typedef struct {
    f16   *Qp;               // [128 d-pairs][48 rows][2]
    f16   *Kp;               // [BK/64][128][64 keys][2]
    f16   *Vp;               // [BK/2 key-pairs][256 d][2]
    f16   *Pp;               // [BK/2][48][2]
    float *St;               // [BK][48]  S transposed
    float *Ot;               // [48][256]
} sme_ws;

// ---------------------------------------------------------------- SME (streaming) parts

__arm_locally_streaming __arm_new("za")
static void sme_qk(const f16 *Qp, const f16 *Kp, float *St, int bk) {
    const svbool_t  ph = svptrue_b16();
    const svbool_t  ps = svptrue_b32();
    const svcount_t pc = svptrue_c16();
    for (int g = 0; g < bk / 64; g++) {
        const f16 *kg = Kp + (size_t)g * 128 * 128;
        for (int qt = 0; qt < 3; qt++) {
            svzero_za();
            const f16 *qq = Qp + qt * 32;
            for (int p = 0; p < 128; p++) {
                svfloat16_t   q = svld1_f16(ph, (const float16_t *)(qq + p * 96));
                svfloat16x4_t k = svld1_f16_x4(pc, (const float16_t *)(kg + p * 128));
                svmopa_za32_f16_m(0, ph, ph, q, svget4_f16(k, 0));
                svmopa_za32_f16_m(1, ph, ph, q, svget4_f16(k, 1));
                svmopa_za32_f16_m(2, ph, ph, q, svget4_f16(k, 2));
                svmopa_za32_f16_m(3, ph, ph, q, svget4_f16(k, 3));
            }
            // tile t, column c = key g*64 + t*16 + c; the column holds 16 query rows -> St[key][qt*16..]
            float *so = St + (size_t)g * 64 * NR + qt * 16;
            for (int c = 0; c < 16; c++) {
                svst1_ver_za32(0, c, ps, so + (0 * 16 + c) * NR);
                svst1_ver_za32(1, c, ps, so + (1 * 16 + c) * NR);
                svst1_ver_za32(2, c, ps, so + (2 * 16 + c) * NR);
                svst1_ver_za32(3, c, ps, so + (3 * 16 + c) * NR);
            }
        }
    }
}

__arm_locally_streaming __arm_new("za")
static void sme_pv(const f16 *Pp, const f16 *Vp, float *Ot, int bk) {
    const svbool_t  ph = svptrue_b16();
    const svbool_t  ps = svptrue_b32();
    const svcount_t pc = svptrue_c16();
    for (int qt = 0; qt < 3; qt++) {
        for (int dg = 0; dg < 4; dg++) {
            svzero_za();
            const f16 *pp = Pp + qt * 32;
            const f16 *vv = Vp + dg * 128;
            for (int kp = 0; kp < bk / 2; kp++) {
                svfloat16_t   p = svld1_f16(ph, (const float16_t *)(pp + kp * 96));
                svfloat16x4_t v = svld1_f16_x4(pc, (const float16_t *)(vv + kp * 512));
                svmopa_za32_f16_m(0, ph, ph, p, svget4_f16(v, 0));
                svmopa_za32_f16_m(1, ph, ph, p, svget4_f16(v, 1));
                svmopa_za32_f16_m(2, ph, ph, p, svget4_f16(v, 2));
                svmopa_za32_f16_m(3, ph, ph, p, svget4_f16(v, 3));
            }
            float *oo = Ot + (size_t)qt * 16 * HD + dg * 64;
            for (int r = 0; r < 16; r++) {
                svst1_hor_za32(0, r, ps, oo + r * HD + 0);
                svst1_hor_za32(1, r, ps, oo + r * HD + 16);
                svst1_hor_za32(2, r, ps, oo + r * HD + 32);
                svst1_hor_za32(3, r, ps, oo + r * HD + 48);
            }
        }
    }
}

// ---------------------------------------------------------------- NEON parts

static inline float32x4_t v_expf(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    float32x4_t t = vmulq_n_f32(x, 1.4426950409f);
    float32x4_t n = vrndnq_f32(t);
    float32x4_t f = vsubq_f32(t, n);                  // 2^f, |f| <= 0.5, Taylor deg 6 (~1e-7)
    float32x4_t p = vdupq_n_f32(1.5403530e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3333558e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), p, f);
    p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, f);
    p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, f);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, f);
    int32x4_t e = vshlq_n_s32(vcvtq_s32_f32(n), 23);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), e));
}

// 4x4 transpose of 32-bit words
static inline void tr4(uint32x4_t *a, uint32x4_t *b, uint32x4_t *c, uint32x4_t *d) {
    uint32x4_t t0 = vtrn1q_u32(*a, *b), t1 = vtrn2q_u32(*a, *b);
    uint32x4_t t2 = vtrn1q_u32(*c, *d), t3 = vtrn2q_u32(*c, *d);
    *a = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    *b = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
    *c = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    *d = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
}

// K rows (fp16, head dim contiguous) -> Kp[g][p][64][2]: word (key j, d-pair p) to Kp32[(g*128+p)*64 + jj]
static void pack_k_f16(const uint8_t *K, size_t rs, int bk, uint32_t *Kp32) {
    for (int j = 0; j < bk; j += 4) {
        const uint32_t *r0 = (const uint32_t *)(K + (size_t)(j + 0) * rs);
        const uint32_t *r1 = (const uint32_t *)(K + (size_t)(j + 1) * rs);
        const uint32_t *r2 = (const uint32_t *)(K + (size_t)(j + 2) * rs);
        const uint32_t *r3 = (const uint32_t *)(K + (size_t)(j + 3) * rs);
        uint32_t *o = Kp32 + (size_t)(j / 64) * 128 * 64 + (j % 64);
        for (int i = 0; i < 4; i++) for (int c = 0; c < 512; c += 128) __builtin_prefetch(K + (size_t)(j + PFD + i) * rs + c);
        for (int p = 0; p < 128; p += 4) {
            uint32x4_t a = vld1q_u32(r0 + p), b = vld1q_u32(r1 + p), c = vld1q_u32(r2 + p), d = vld1q_u32(r3 + p);
            tr4(&a, &b, &c, &d);
            vst1q_u32(o + (p + 0) * 64, a); vst1q_u32(o + (p + 1) * 64, b);
            vst1q_u32(o + (p + 2) * 64, c); vst1q_u32(o + (p + 3) * 64, d);
        }
    }
}

// dequant one q8_0 block (2 B fp16 scale + 32 int8) to 32 fp16 (as 4 x float16x8)
static inline void deq_q8(const uint8_t *blk, float16x8_t o[4]) {
    f16 d; memcpy(&d, blk, 2);
    int8x16_t q0 = vld1q_s8((const int8_t *)blk + 2), q1 = vld1q_s8((const int8_t *)blk + 18);
    o[0] = vmulq_n_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(q0))), d);
    o[1] = vmulq_n_f16(vcvtq_f16_s16(vmovl_high_s8(q0)), d);
    o[2] = vmulq_n_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(q1))), d);
    o[3] = vmulq_n_f16(vcvtq_f16_s16(vmovl_high_s8(q1)), d);
}

static void pack_k_q8(const uint8_t *K, size_t rs, int bk, uint32_t *Kp32) {
    for (int j = 0; j < bk; j += 4) {
        uint32_t *o = Kp32 + (size_t)(j / 64) * 128 * 64 + (j % 64);
        for (int i = 0; i < 4; i++) for (int c = 0; c < 272; c += 128) __builtin_prefetch(K + (size_t)(j + PFD + i) * rs + c);
        __builtin_prefetch(K + (size_t)(j + PFD) * rs + 271); __builtin_prefetch(K + (size_t)(j + PFD + 3) * rs + 271);
        for (int b = 0; b < 8; b++) {              // 8 blocks of 32 d = 16 d-pairs each
            float16x8_t x[4][4];
            for (int i = 0; i < 4; i++) deq_q8(K + (size_t)(j + i) * rs + b * 34, x[i]);
            for (int w = 0; w < 4; w++) {          // 4 words-groups of 4 d-pairs
                uint32x4_t a = vreinterpretq_u32_f16(x[0][w]), bb = vreinterpretq_u32_f16(x[1][w]);
                uint32x4_t c = vreinterpretq_u32_f16(x[2][w]), d = vreinterpretq_u32_f16(x[3][w]);
                tr4(&a, &bb, &c, &d);
                int p = b * 16 + w * 4;
                vst1q_u32(o + (p + 0) * 64, a); vst1q_u32(o + (p + 1) * 64, bb);
                vst1q_u32(o + (p + 2) * 64, c); vst1q_u32(o + (p + 3) * 64, d);
            }
        }
    }
}

// V rows -> Vp[kp][d][2] (zip of rows 2kp, 2kp+1)
static void pack_v_f16(const uint8_t *V, size_t rs, int bk, f16 *Vp) {
    for (int kp = 0; kp < bk / 2; kp++) {
        const f16 *a = (const f16 *)(V + (size_t)(2 * kp) * rs), *b = (const f16 *)(V + (size_t)(2 * kp + 1) * rs);
        f16 *o = Vp + (size_t)kp * 512;
        for (int c = 0; c < 512; c += 128) { __builtin_prefetch((const uint8_t *)a + PFD * rs + c); __builtin_prefetch((const uint8_t *)b + PFD * rs + c); }
        for (int d = 0; d < HD; d += 8) {
            float16x8_t x = vld1q_f16(a + d), y = vld1q_f16(b + d);
            vst1q_f16(o + 2 * d, vzip1q_f16(x, y));
            vst1q_f16(o + 2 * d + 8, vzip2q_f16(x, y));
        }
    }
}

static void pack_v_q8(const uint8_t *V, size_t rs, int bk, f16 *Vp) {
    for (int kp = 0; kp < bk / 2; kp++) {
        const uint8_t *a = V + (size_t)(2 * kp) * rs, *b = V + (size_t)(2 * kp + 1) * rs;
        f16 *o = Vp + (size_t)kp * 512;
        for (int c = 0; c < 272; c += 128) { __builtin_prefetch(a + PFD * rs + c); __builtin_prefetch(b + PFD * rs + c); }
        __builtin_prefetch(a + PFD * rs + 271); __builtin_prefetch(b + PFD * rs + 271);
        for (int blk = 0; blk < 8; blk++) {
            float16x8_t x[4], y[4];
            deq_q8(a + blk * 34, x); deq_q8(b + blk * 34, y);
            for (int w = 0; w < 4; w++) {
                int d = blk * 32 + w * 8;
                vst1q_f16(o + 2 * d, vzip1q_f16(x[w], y[w]));
                vst1q_f16(o + 2 * d + 8, vzip2q_f16(x[w], y[w]));
            }
        }
    }
}

// dequant one q4_0 block (2 B fp16 scale + 16 B of nibbles: x[i] = lo(i) - 8, x[i+16] = hi(i) - 8) to 32 fp16 (4 x float16x8)
static inline void deq_q4(const uint8_t *blk, float16x8_t o[4]) {
    f16 d; memcpy(&d, blk, 2);
    const uint8x16_t q = vld1q_u8(blk + 2);
    const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, vdupq_n_u8(0x0F))), vdupq_n_s8(8));
    const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), vdupq_n_s8(8));
    o[0] = vmulq_n_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(lo))), d);
    o[1] = vmulq_n_f16(vcvtq_f16_s16(vmovl_high_s8(lo)), d);
    o[2] = vmulq_n_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(hi))), d);
    o[3] = vmulq_n_f16(vcvtq_f16_s16(vmovl_high_s8(hi)), d);
}

static void pack_k_q4(const uint8_t *K, size_t rs, int bk, uint32_t *Kp32) {
    for (int j = 0; j < bk; j += 4) {
        uint32_t *o = Kp32 + (size_t)(j / 64) * 128 * 64 + (j % 64);
        for (int i = 0; i < 4; i++) { __builtin_prefetch(K + (size_t)(j + PFD + i) * rs); __builtin_prefetch(K + (size_t)(j + PFD + i) * rs + 143); }
        for (int b = 0; b < 8; b++) {              // 8 blocks of 32 d = 16 d-pairs each
            float16x8_t x[4][4];
            for (int i = 0; i < 4; i++) deq_q4(K + (size_t)(j + i) * rs + b * 18, x[i]);
            for (int w = 0; w < 4; w++) {
                uint32x4_t a = vreinterpretq_u32_f16(x[0][w]), bb = vreinterpretq_u32_f16(x[1][w]);
                uint32x4_t c = vreinterpretq_u32_f16(x[2][w]), d = vreinterpretq_u32_f16(x[3][w]);
                tr4(&a, &bb, &c, &d);
                int p = b * 16 + w * 4;
                vst1q_u32(o + (p + 0) * 64, a); vst1q_u32(o + (p + 1) * 64, bb);
                vst1q_u32(o + (p + 2) * 64, c); vst1q_u32(o + (p + 3) * 64, d);
            }
        }
    }
}

static void pack_v_q4(const uint8_t *V, size_t rs, int bk, f16 *Vp) {
    for (int kp = 0; kp < bk / 2; kp++) {
        const uint8_t *a = V + (size_t)(2 * kp) * rs, *b = V + (size_t)(2 * kp + 1) * rs;
        f16 *o = Vp + (size_t)kp * 512;
        __builtin_prefetch(a + PFD * rs); __builtin_prefetch(a + PFD * rs + 143);
        __builtin_prefetch(b + PFD * rs); __builtin_prefetch(b + PFD * rs + 143);
        for (int blk = 0; blk < 8; blk++) {
            float16x8_t x[4], y[4];
            deq_q4(a + blk * 18, x); deq_q4(b + blk * 18, y);
            for (int w = 0; w < 4; w++) {
                int d = blk * 32 + w * 8;
                vst1q_f16(o + 2 * d, vzip1q_f16(x[w], y[w]));
                vst1q_f16(o + 2 * d + 8, vzip2q_f16(x[w], y[w]));
            }
        }
    }
}

// online softmax on one block. St[bk][48] (already scaled). Updates m,l; writes Pp; returns alpha[48].
static void softmax_blk(const float *St, int bk, float *m, float *l, float *alpha, f16 *Pp) {
    float32x4_t mx[12], mn[12], ls[12];
    for (int i = 0; i < 12; i++) mx[i] = vdupq_n_f32(-INFINITY);
    for (int j = 0; j < bk; j++)
        for (int i = 0; i < 12; i++) mx[i] = vmaxq_f32(mx[i], vld1q_f32(St + j * NR + 4 * i));
    for (int i = 0; i < 12; i++) {
        float32x4_t mo = vld1q_f32(m + 4 * i);
        mn[i] = vmaxq_f32(mo, mx[i]);
        vst1q_f32(alpha + 4 * i, v_expf(vsubq_f32(mo, mn[i])));
        vst1q_f32(m + 4 * i, mn[i]);
        ls[i] = vdupq_n_f32(0);
    }
    for (int kp = 0; kp < bk / 2; kp++) {
        const float *s0 = St + (2 * kp) * NR, *s1 = s0 + NR;
        f16 *o = Pp + kp * 96;
        for (int i = 0; i < 12; i++) {
            float32x4_t p0 = v_expf(vsubq_f32(vld1q_f32(s0 + 4 * i), mn[i]));
            float32x4_t p1 = v_expf(vsubq_f32(vld1q_f32(s1 + 4 * i), mn[i]));
            ls[i] = vaddq_f32(ls[i], vaddq_f32(p0, p1));
            float16x4_t h0 = vcvt_f16_f32(p0), h1 = vcvt_f16_f32(p1);
            vst1q_f16(o + 8 * i, vcombine_f16(vzip1_f16(h0, h1), vzip2_f16(h0, h1)));
        }
    }
    for (int i = 0; i < 12; i++) {
        float32x4_t lo = vld1q_f32(l + 4 * i);
        vst1q_f32(l + 4 * i, vfmaq_f32(ls[i], lo, vld1q_f32(alpha + 4 * i)));
    }
}

static void o_update(float *O, const float *Ot, const float *alpha) {
    for (int r = 0; r < NR; r++) {
        float32x4_t a = vdupq_n_f32(alpha[r]);
        float *o = O + r * HD; const float *t = Ot + r * HD;
        for (int d = 0; d < HD; d += 4) vst1q_f32(o + d, vfmaq_f32(vld1q_f32(t + d), vld1q_f32(o + d), a));
    }
}

// ---------------------------------------------------------------- public API

sme_ws *sme_ws_new(void) {
    sme_ws *w = calloc(1, sizeof *w);
    w->Qp = aligned_alloc(128, 128 * NR * 2 * sizeof(f16));
    w->Kp = aligned_alloc(128, (size_t)BK_MAX * HD * sizeof(f16));
    w->Vp = aligned_alloc(128, (size_t)BK_MAX * HD * sizeof(f16));
    w->Pp = aligned_alloc(128, (size_t)BK_MAX * NR * sizeof(f16));
    w->St = aligned_alloc(128, (size_t)BK_MAX * NR * sizeof(float));
    w->Ot = aligned_alloc(128, (size_t)NR * HD * sizeof(float));
    return w;
}

double sme_t_pack, sme_t_qk, sme_t_sm, sme_t_pv, sme_t_upd; // phase timers (only if sme_timing)
int sme_timing = 0;
#include <time.h>
static inline double now_s(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

// One KV head. Q: fp32 [48][256] (row stride qrs floats). K,V: pointer to key k0's row for this head
// (i.e. base + k0*rs + head_offset), rs = row stride in bytes, is_q8: 0 fp16, 1 q8_0, 2 q4_0.
// nk = number of keys, multiple of 64. Writes O[48][256] (unnormalised), m[48], l[48].
void sme_attn_partial(sme_ws *w, const float *Q, int qrs, const uint8_t *K, const uint8_t *V, size_t rs,
                      int is_q8, int nk, float scale, int bk, float *O, float *m, float *l) {
    for (int p = 0; p < 128; p++)
        for (int r = 0; r < NR; r++) {
            w->Qp[p * 96 + r * 2 + 0] = (f16)(Q[r * qrs + 2 * p + 0] * scale);
            w->Qp[p * 96 + r * 2 + 1] = (f16)(Q[r * qrs + 2 * p + 1] * scale);
        }
    for (int r = 0; r < NR; r++) { m[r] = -1e30f; l[r] = 0; }
    memset(O, 0, sizeof(float) * NR * HD);
    float alpha[NR];
    for (int j0 = 0; j0 < nk; j0 += bk) {
        int b = nk - j0 < bk ? nk - j0 : bk;
        double t0 = sme_timing ? now_s() : 0;
        if (is_q8 == 2)  pack_k_q4(K + (size_t)j0 * rs, rs, b, (uint32_t *)w->Kp);
        else if (is_q8)  pack_k_q8(K + (size_t)j0 * rs, rs, b, (uint32_t *)w->Kp);
        else             pack_k_f16(K + (size_t)j0 * rs, rs, b, (uint32_t *)w->Kp);
        double t1 = sme_timing ? now_s() : 0;
        sme_qk(w->Qp, w->Kp, w->St, b);
        double t2 = sme_timing ? now_s() : 0;
        softmax_blk(w->St, b, m, l, alpha, w->Pp);
        double t3 = sme_timing ? now_s() : 0;
        if (is_q8 == 2)  pack_v_q4(V + (size_t)j0 * rs, rs, b, w->Vp);
        else if (is_q8)  pack_v_q8(V + (size_t)j0 * rs, rs, b, w->Vp);
        else             pack_v_f16(V + (size_t)j0 * rs, rs, b, w->Vp);
        double t4 = sme_timing ? now_s() : 0;
        sme_pv(w->Pp, w->Vp, w->Ot, b);
        double t5 = sme_timing ? now_s() : 0;
        o_update(O, w->Ot, alpha);
        if (sme_timing) {
            double t6 = now_s();
            sme_t_pack += (t1 - t0) + (t4 - t3); sme_t_qk += t2 - t1; sme_t_sm += t3 - t2;
            sme_t_pv += t5 - t4; sme_t_upd += t6 - t5;
        }
    }
}


// ---------------------------------------------------------------- pipelined version: helpers pack, SME thread computes
// One "SME worker" (ideally one per P-cluster) + nh NEON helper threads. Work items = (head, block of bk keys).
// Helper t packs items t, t+nh, ... into a ring of NSLOT slots; the SME worker consumes them in order.
#include <stdatomic.h>
#define NSLOT 8
typedef struct {
    const float *Q; int qrs, qhs;            // Q for head h at Q + h*qhs
    const uint8_t *K, *V; size_t rs, hb;     // head h base = K + h*hb
    int is_q8, nk, bk, h0, h1; float scale;
    float *O, *m, *l;                        // head h at O + h*48*256 etc.
    int nh;
    f16 *Kp[NSLOT], *Vp[NSLOT];
    _Atomic int ready[NSLOT];
    _Atomic int consumed;
    sme_ws *w;
    // softmax offload (sm_off=1): QK(i) -> St[i%2] -> softmax helper -> Pp[i%2], alpha[i%2]; worker does PV(i-1) meanwhile
    int sm_off;
    float *St2[2]; f16 *Pp2[2]; float alpha2[2][NR];
    _Atomic int st_ready, sm_done;
    // infernet: optional additive mask (f16, row per query token, mrs elements per row, key 0 = this pipe's key 0);
    // query row r = g*8 + t takes mask row t. NULL = no mask.
    const f16 *mask; size_t mrs; int ntok;
} sme_pipe;

static int pipe_nitems(const sme_pipe *p) { return (p->h1 - p->h0) * ((p->nk + p->bk - 1) / p->bk); }

void sme_pipe_helper(sme_pipe *p, int t) {
    int nb = (p->nk + p->bk - 1) / p->bk, n = pipe_nitems(p);
    for (int i = t; i < n; i += p->nh) {
        int s = i % NSLOT, h = p->h0 + i / nb, j0 = (i % nb) * p->bk;
        int b = p->nk - j0 < p->bk ? p->nk - j0 : p->bk;
        while (atomic_load_explicit(&p->consumed, memory_order_acquire) < i - NSLOT + 1) __builtin_arm_yield();
        const uint8_t *K = p->K + h * p->hb + (size_t)j0 * p->rs, *V = p->V + h * p->hb + (size_t)j0 * p->rs;
        if (p->is_q8 == 2) { pack_k_q4(K, p->rs, b, (uint32_t *)p->Kp[s]); pack_v_q4(V, p->rs, b, p->Vp[s]); }
        else if (p->is_q8) { pack_k_q8(K, p->rs, b, (uint32_t *)p->Kp[s]); pack_v_q8(V, p->rs, b, p->Vp[s]); }
        else               { pack_k_f16(K, p->rs, b, (uint32_t *)p->Kp[s]); pack_v_f16(V, p->rs, b, p->Vp[s]); }
        atomic_store_explicit(&p->ready[s], i + 1, memory_order_release);
    }
}

void sme_pipe_worker(sme_pipe *p) {
    sme_ws *w = p->w;
    int nb = (p->nk + p->bk - 1) / p->bk, n = pipe_nitems(p);
    float alpha[NR];
    for (int i = 0; i < n; i++) {
        int s = i % NSLOT, h = p->h0 + i / nb, j0 = (i % nb) * p->bk;
        int b = p->nk - j0 < p->bk ? p->nk - j0 : p->bk;
        float *O = p->O + h * NR * HD, *m = p->m + h * NR, *l = p->l + h * NR;
        if (j0 == 0) {
            const float *Q = p->Q + h * p->qhs;
            for (int q = 0; q < 128; q++)
                for (int r = 0; r < NR; r++) {
                    w->Qp[q * 96 + r * 2 + 0] = (f16)(Q[r * p->qrs + 2 * q + 0] * p->scale);
                    w->Qp[q * 96 + r * 2 + 1] = (f16)(Q[r * p->qrs + 2 * q + 1] * p->scale);
                }
            for (int r = 0; r < NR; r++) { m[r] = -1e30f; l[r] = 0; }
            memset(O, 0, sizeof(float) * NR * HD);
        }
        while (atomic_load_explicit(&p->ready[s], memory_order_acquire) != i + 1) __builtin_arm_yield();
        sme_qk(w->Qp, p->Kp[s], w->St, b);
        softmax_blk(w->St, b, m, l, alpha, w->Pp);
        sme_pv(w->Pp, p->Vp[s], w->Ot, b);
        atomic_store_explicit(&p->consumed, i + 1, memory_order_release);
        o_update(O, w->Ot, alpha);
    }
}

static void pipe_item(const sme_pipe *p, int i, int *h, int *j0, int *b) {
    int nb = (p->nk + p->bk - 1) / p->bk;
    *h = p->h0 + i / nb; *j0 = (i % nb) * p->bk; *b = p->nk - *j0 < p->bk ? p->nk - *j0 : p->bk;
}

// add the mask to a block of S^T (St[key][48 rows], row r = g*8 + t); zero entries are skipped
static void add_mask_blk(float *St, int b, const f16 *mask, size_t mrs, int ntok) {
    for (int t = 0; t < ntok; t++) {
        const f16 *mr = mask + (size_t)t * mrs;
        for (int j = 0; j < b; j++) {
            const float mv = (float)mr[j];
            if (mv != 0.0f)
                for (int g = 0; g < NR / 8; g++) St[(size_t)j * NR + g * 8 + t] += mv;
        }
    }
}

void sme_pipe_softmax_helper(sme_pipe *p) {
    int n = pipe_nitems(p);
    for (int i = 0; i < n; i++) {
        int h, j0, b; pipe_item(p, i, &h, &j0, &b);
        float *m = p->m + h * NR, *l = p->l + h * NR;
        if (j0 == 0) for (int r = 0; r < NR; r++) { m[r] = -1e30f; l[r] = 0; }
        while (atomic_load_explicit(&p->st_ready, memory_order_acquire) < i + 1) __builtin_arm_yield();
        if (p->mask) add_mask_blk(p->St2[i & 1], b, p->mask + j0, p->mrs, p->ntok);
        softmax_blk(p->St2[i & 1], b, m, l, p->alpha2[i & 1], p->Pp2[i & 1]);
        atomic_store_explicit(&p->sm_done, i + 1, memory_order_release);
    }
}

void sme_pipe_worker2(sme_pipe *p) {
    sme_ws *w = p->w;
    int n = pipe_nitems(p);
    for (int i = 0; i <= n; i++) {
        if (i < n) {
            int h, j0, b; pipe_item(p, i, &h, &j0, &b);
            if (j0 == 0) {
                const float *Q = p->Q + h * p->qhs;
                for (int q = 0; q < 128; q++)
                    for (int r = 0; r < NR; r++) {
                        w->Qp[q * 96 + r * 2 + 0] = (f16)(Q[r * p->qrs + 2 * q + 0] * p->scale);
                        w->Qp[q * 96 + r * 2 + 1] = (f16)(Q[r * p->qrs + 2 * q + 1] * p->scale);
                    }
                memset(p->O + h * NR * HD, 0, sizeof(float) * NR * HD);
            }
            while (atomic_load_explicit(&p->ready[i % NSLOT], memory_order_acquire) != i + 1) __builtin_arm_yield();
            while (atomic_load_explicit(&p->sm_done, memory_order_acquire) < i - 1) __builtin_arm_yield(); // St[i&1] free
            sme_qk(w->Qp, p->Kp[i % NSLOT], p->St2[i & 1], b);
            atomic_store_explicit(&p->st_ready, i + 1, memory_order_release);
        }
        if (i >= 1) {
            int k = i - 1, h, j0, b; pipe_item(p, k, &h, &j0, &b);
            while (atomic_load_explicit(&p->sm_done, memory_order_acquire) < k + 1) __builtin_arm_yield();
            sme_pv(p->Pp2[k & 1], p->Vp[k % NSLOT], w->Ot, b);
            atomic_store_explicit(&p->consumed, k + 1, memory_order_release);
            o_update(p->O + h * NR * HD, w->Ot, p->alpha2[k & 1]);
        }
    }
}

sme_pipe *sme_pipe_new(int nh) {
    sme_pipe *p = calloc(1, sizeof *p);
    p->nh = nh; p->w = sme_ws_new();
    for (int s = 0; s < NSLOT; s++) {
        p->Kp[s] = aligned_alloc(128, (size_t)BK_MAX * HD * sizeof(f16));
        p->Vp[s] = aligned_alloc(128, (size_t)BK_MAX * HD * sizeof(f16));
    }
    for (int s = 0; s < 2; s++) {
        p->St2[s] = aligned_alloc(128, (size_t)BK_MAX * NR * sizeof(float));
        p->Pp2[s] = aligned_alloc(128, (size_t)BK_MAX * NR * sizeof(f16));
    }
    return p;
}
static void sme_pipe_reset(sme_pipe *p) {
    for (int s = 0; s < NSLOT; s++) atomic_store(&p->ready[s], 0);
    atomic_store(&p->consumed, 0); atomic_store(&p->st_ready, 0); atomic_store(&p->sm_done, 0);
}

// ---------------------------------------------------------------- compressed-domain int8 QK (q8_0 K, Q quantised per 32-block)
// T[g][b][key 64][48] int32 = per-block partial dot products; the fp32 score needs sum_b dq[r,b]*dk[key,b]*T.
// Qi: [64 quads][48 rows][4] int8 ; Ki: [g][64 quads][64 keys][4] int8
__arm_locally_streaming __arm_new("za")
static void sme_qk_i8blk(const int8_t *Qi, const int8_t *Ki, int32_t *T, int bk) {
    const svbool_t pb = svptrue_b8(), ps = svptrue_b32();
    const svcount_t pc = svptrue_c8();
    for (int g = 0; g < bk / 64; g++)
        for (int qt = 0; qt < 3; qt++)
            for (int b = 0; b < 8; b++) {
                svzero_za();
                for (int qd = 0; qd < 8; qd++) {
                    int quad = b * 8 + qd;
                    svint8_t   q = svld1_s8(pb, Qi + quad * 192 + qt * 64);
                    svint8x4_t k = svld1_s8_x4(pc, Ki + ((size_t)g * 64 + quad) * 256);
                    svmopa_za32_s8_m(0, pb, pb, q, svget4_s8(k, 0));
                    svmopa_za32_s8_m(1, pb, pb, q, svget4_s8(k, 1));
                    svmopa_za32_s8_m(2, pb, pb, q, svget4_s8(k, 2));
                    svmopa_za32_s8_m(3, pb, pb, q, svget4_s8(k, 3));
                }
                int32_t *to = T + (((size_t)g * 8 + b) * 64) * NR + qt * 16;
                for (int c = 0; c < 16; c++) {
                    svst1_ver_za32(0, c, ps, to + (0 * 16 + c) * NR);
                    svst1_ver_za32(1, c, ps, to + (1 * 16 + c) * NR);
                    svst1_ver_za32(2, c, ps, to + (2 * 16 + c) * NR);
                    svst1_ver_za32(3, c, ps, to + (3 * 16 + c) * NR);
                }
            }
}
// NEON: St[key][r] = sum_b dq[b][r] * dk[key][b] * T[g][b][key][r]
static void i8_combine(const int32_t *T, const float *dq /*[8][48]*/, const float *dk /*[bk][8]*/, float *St, int bk) {
    for (int j = 0; j < bk; j++) {
        int g = j / 64, jj = j % 64;
        float32x4_t acc[12];
        for (int i = 0; i < 12; i++) acc[i] = vdupq_n_f32(0);
        for (int b = 0; b < 8; b++) {
            const int32_t *t = T + (((size_t)g * 8 + b) * 64 + jj) * NR;
            float s = dk[j * 8 + b];
            for (int i = 0; i < 12; i++)
                acc[i] = vfmaq_f32(acc[i], vcvtq_f32_s32(vld1q_s32(t + 4 * i)), vmulq_n_f32(vld1q_f32(dq + b * NR + 4 * i), s));
        }
        for (int i = 0; i < 12; i++) vst1q_f32(St + j * NR + 4 * i, acc[i]);
    }
}

// ================================================================ bench
