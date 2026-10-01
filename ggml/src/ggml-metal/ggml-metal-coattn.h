// SME co-attention (infernet): the CPU's SME2 units compute the oldest keys of a GQA verify flash attention
// while the GPU computes the rest; the CPU's (O, m, l) partial is merged by the GPU's split-K reduce.
//
// Enable with GGML_METAL_FA_SME=<fraction of keys for the CPU>, e.g. 0.35 (0 or unset = off).
// Needs GGML_METAL_FA_GQA=1 (the verify kernel), FEAT_SME2 with 512-bit streaming vectors, shared buffers.
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// one flash-attention op's CPU share: keys [0, nk) of every KV head
struct ggml_metal_coattn_job {
    uint64_t v_start;       // the GPU signals this once Q is written: the CPU may start
    uint64_t v_done;        // the CPU signals this once its partial is in tmp: the GPU's reduce may start

    const float * q;        // Q [256, n_tok, n_head]: token stride q_nb1, head stride q_nb2 (bytes)
    size_t q_nb1;
    size_t q_nb2;
    int    n_tok;           // query tokens (<= 8)
    int    n_head;          // query heads (= 6 * n_head_kv)
    int    n_head_kv;       // 4

    const uint8_t * k;      // K / V rows: key j of head h at k + j*rs + h*hb
    const uint8_t * v;
    size_t rs;
    size_t hb;
    int    is_q8;           // 0: f16, 1: q8_0, 2: q4_0

    const uint16_t * mask;  // f16 KQ mask (NULL = none): row t at mask + t*mask_rs elements, key 0 first
    size_t mask_rs;

    int    nk;              // keys for the CPU (multiple of 64)
    float  frac_used;       // nk / total keys (for the adaptive split)
    float  scale;

    float * part;           // the CPU's partial: O [nrows][256], then (S, M) per row at GGML_METAL_COATTN_SM
    int64_t nrows;          // ne1*ne2*ne3 of the FA output (<= GGML_METAL_COATTN_MAX_ROWS)
};

#define GGML_METAL_COATTN_MAX_ROWS 192                           // 8 tokens x 24 query heads
#define GGML_METAL_COATTN_SM       (GGML_METAL_COATTN_MAX_ROWS*256)
#define GGML_METAL_COATTN_FLOATS   (GGML_METAL_COATTN_SM + 2*GGML_METAL_COATTN_MAX_ROWS)

// fraction of keys for the CPU (GGML_METAL_FA_SME), 0 when disabled or SME2 is unavailable
float ggml_metal_coattn_fraction(void);

// the adaptive fraction currently in use (starts at ggml_metal_coattn_fraction(); GGML_METAL_FA_SME_ADAPT=0 keeps it fixed):
// after each job the CPU checks whether the GPU had already finished its share (then the CPU was the bottleneck:
// shrink the CPU's share) or not (grow it)
float ggml_metal_coattn_fraction_now(void);

// the second event (id<MTLSharedEvent>): the GPU signals v_start on it once its own share is done
void * ggml_metal_coattn_event_gpu(void * mtl_device);

// minimum number of keys before co-attention is used (GGML_METAL_FA_SME_MIN_KV, default 8192)
int ggml_metal_coattn_min_kv(void);

// the shared event (id<MTLSharedEvent>), created on first use for this device
void * ggml_metal_coattn_event(void * mtl_device);

// reserve a range of event values for one graph (2 per node); call before encoding it
void ggml_metal_coattn_graph_begin(int n_nodes);

// the current graph's base value: node idx uses base + 2*idx + 1 (start) and base + 2*idx + 2 (done)
uint64_t ggml_metal_coattn_graph_base(void);

// the CPU partial buffers (id<MTLBuffer>, shared storage), used alternately by consecutive jobs; *host = contents
void * ggml_metal_coattn_part_buffer(void * mtl_device, int i, float ** host);

// queue a job; the CPU workers run it when the event reaches v_start and signal v_done
void ggml_metal_coattn_submit(const struct ggml_metal_coattn_job * job);

#ifdef __cplusplus
}
#endif
