// SME prefill matmuls (infernet): the CPU's SME2 units compute the first tokens of a prefill ubatch's IQ4_XS matmul
// while the GPU computes the rest. Split by tokens, so each side writes whole contiguous rows of dst.
//
// Enable with GGML_METAL_MM_SME=<fraction of the ubatch's tokens for the CPU>, e.g. 0.25 (0 or unset = off).
// Rounded down to the GPU kernel's token tile (so the GPU's compiled bounds-check path is unchanged).
// GGML_METAL_MM_SME_MIN (default 256): only ubatches with at least this many tokens (decode/verify never use it).
// GGML_METAL_MM_SME_HELPERS (default 3): NEON dequant helpers per SME worker (2 workers, one per P-cluster SME unit).
// GGML_METAL_MM_SME_STATS=1: per-shape CPU vs GPU time. GGML_METAL_MM_SME_ADAPT=0: keep the fraction fixed per shape.
// Rows (n) below GGML_METAL_MM_SME_MIN_N (default 2048) stay on the GPU: the CPU's fixed dequant cost buys nothing there.
// GGML_METAL_MM_SME_ROWS=1: split by ROWS instead: the CPU takes the last rows (multiple of 64) for all tokens and dequantizes
// only those (tokens mode dequantizes the whole matrix per matmul: a fixed cost a phone's few cores can't pay); the GPU writes
// its rows with the full dst row stride (ggml_metal_kargs_mul_mm.ldd). Adaptive per shape from CPU (pack + per-row) and GPU
// (per-row) times. GGML_METAL_MM_SME_WORKERS (default: one per P-cluster, from sysctl: M4 Pro 2, A19 Pro 1).
// Measured 2026-09-27: Mac (M4 Pro) pp2048 ub256 121.6 off -> 148.2 tokens -> 157-166 rows. iPhone 17 Pro Max (A19 Pro, Sidecar
// split-prefill tail): a NET LOSS, keep it off there: the A19 GPU (neural accelerators) runs these matmuls at ~5 TFLOPS vs SME2's
// ~1.7 achieved, and SME at full load takes the phone's shared power budget: 12% of rows -> phone chunks 1.15-1.35 s -> 1.7-3.1 s.
// Needs FEAT_SME2 with 512-bit streaming vectors and shared buffers.
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_mmsme_job {
    uint64_t v_start;       // the GPU signals this once src1 (X) is written: the CPU may start
    uint64_t v_done;        // the CPU signals this once its rows of dst are written

    const void  * w;        // IQ4_XS weights, N rows of K
    int           n;        // rows of W = dst columns (ne01)
    int           k;        // ne00, multiple of 256
    const float * x;        // X: t rows of k floats, row stride ldx floats
    size_t        ldx;
    int           t;        // tokens for the CPU (the first t rows of X / dst)
    float       * y;        // dst: t rows of n floats, row stride ldy floats
    size_t        ldy;
    int           t_all;    // tokens of the whole op
    int           tile;     // the GPU kernel's token tile (t is a multiple of it); rows mode: the row tile
    int           n_all;    // rows of the whole op (rows mode: the CPU has the last n of them)
    int           rows;     // 1 = rows mode (t == t_all)
};

// fraction of tokens for the CPU (GGML_METAL_MM_SME), 0 when disabled or SME2 is unavailable
float ggml_metal_mmsme_fraction(void);

// the CPU's tokens for this shape: GGML_METAL_MM_SME's fraction, then adapted per shape from measured CPU and GPU times
// (GGML_METAL_MM_SME_ADAPT=0: fixed fraction). A multiple of tile.
int ggml_metal_mmsme_tokens(int n, int k, int t_all, int tile);

// rows mode (GGML_METAL_MM_SME_ROWS=1): the CPU's rows for this shape (a multiple of tile, the last rows of the op)
int ggml_metal_mmsme_rows_mode(void);
int ggml_metal_mmsme_rows(int n, int k, int t_all, int tile);

// minimum ubatch tokens (GGML_METAL_MM_SME_MIN)
int ggml_metal_mmsme_min_tokens(void);

// the shared events (id<MTLSharedEvent>): the CPU's (start / done) and the GPU's "my share is done" (stats)
void * ggml_metal_mmsme_event(void * mtl_device);
void * ggml_metal_mmsme_event_gpu(void * mtl_device);

// queue a job; the CPU runs it when the event reaches v_start and signals v_done
void ggml_metal_mmsme_submit(const struct ggml_metal_mmsme_job * job);

#ifdef __cplusplus
}
#endif
