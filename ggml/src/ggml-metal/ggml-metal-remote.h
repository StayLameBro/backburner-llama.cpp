// Phone-held KV (infernet, docs/phone-kv-262k.md): the phone holds the OLDEST keys of every full-attention layer and
// computes their share of each attention op; its partial (O, lse) goes into one extra split-K slot of the GQA verify
// flash attention and is merged by the same reduce the SME co-attention partial uses.
//
// The KV cache (libllama) owns the policy: it attaches the phone, moves old cells there (append) and truncates it.
// It reaches these functions through ggml_backend_reg_get_proc_address(metal_reg, "<name>"):
//   ggml_backend_metal_remote_attach, _append, _truncate, _held
// and tags each attention op it owns with GGML_METAL_REMOTE_TAG in op_params[GGML_METAL_REMOTE_OP_PARAM].
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_METAL_REMOTE_OP_PARAM 5            // op_params slot of GGML_OP_FLASH_ATTN_EXT (0-4 are used by ggml)
#define GGML_METAL_REMOTE_TAG      0x50480000   // 'PH' << 16 | (model layer + 1)

// connect to the phone-attn server at "host:port" and configure it for n_layer layers (model layer ids il[i]; the phone
// numbers them 0..n_layer-1), rows of rs bytes per key (n_head_kv heads of hb bytes), is_q8: 0 f16, 1 q8_0, 2 q4_0
// returns 0 on success
typedef int      (*ggml_backend_metal_remote_attach_t)(const char * host_port, int n_layer, const int32_t * il,
                                                        int n_head_kv, size_t rs, size_t hb, int is_q8);
// keys [pos0, pos0 + n) of layer i (0..n_layer-1): n rows of K, then n rows of V; returns 0 on success
typedef int      (*ggml_backend_metal_remote_append_t)(int i, uint32_t pos0, uint32_t n, const void * k, const void * v);
// every layer keeps its first n keys (0 = drop everything); returns 0 on success
typedef int      (*ggml_backend_metal_remote_truncate_t)(uint32_t n);
// keys held per layer (0 when not attached)
typedef uint32_t (*ggml_backend_metal_remote_held_t)(void);

int      ggml_backend_metal_remote_attach(const char * host_port, int n_layer, const int32_t * il, int n_head_kv, size_t rs, size_t hb, int is_q8);
int      ggml_backend_metal_remote_append(int i, uint32_t pos0, uint32_t n, const void * k, const void * v);
int      ggml_backend_metal_remote_truncate(uint32_t n);
uint32_t ggml_backend_metal_remote_held(void);
// 1 if the attached phone speaks protocol v3 (ATTN_BIG: a whole prefill ubatch per layer in one call), else 0
typedef int      (*ggml_backend_metal_remote_big_t)(void);
int      ggml_backend_metal_remote_big(void);

// phone ATTN_BIG (prefill ubatch split into 8-token groups by the graph, op_params[6] = group, [7] = ubatch tokens):
// group 0's job carries every token of the ubatch; its partials land in a big buffer that the other groups read
#define GGML_METAL_REMOTE_BIG_ROWS (512*24)                          // 512 tokens x 24 query heads
#define GGML_METAL_REMOTE_BIG_SM   (GGML_METAL_REMOTE_BIG_ROWS*256)  // (S, M) pairs after the O rows

// ---- Metal backend side

// the phone's layer index for this flash-attention op, or -1 if the op is not tagged or the phone holds no keys
int ggml_metal_remote_layer(const int32_t * op_params);

struct ggml_metal_remote_job {
    uint64_t v_start;       // the GPU signals this on ggml_metal_remote_event() once Q is written
    uint64_t v_done;        // the phone thread signals this once the partial is in part

    const float * q;        // Q [256, n_tok, n_head]: token stride q_nb1, head stride q_nb2 (bytes)
    size_t q_nb1;
    size_t q_nb2;
    int    n_tok;           // <= 8 (this op's tokens)
    int    n_tok_all;       // ATTN_BIG: tokens of the whole ubatch (group 0 sends them all); 0 = plain job
    int    grp;             // ATTN_BIG: 8-token group index (> 0: follower, its rows are already in the big buffer)
    int    n_head;          // 6 * n_head_kv
    int    n_head_kv;
    int    layer;           // phone layer index
    float  scale;

    float * part;           // O [nrows][256], then (S, M) per row at GGML_METAL_COATTN_SM (same layout as co-attention)
    int64_t nrows;
    int     ev;             // which shared event carries v_start / v_done: 0, or 1 for the second half of a pipelined ubatch
};

// the shared event (id<MTLSharedEvent>) the phone jobs wait on / signal
void * ggml_metal_remote_event(void * mtl_device);
// pipelined prefill (two halves of a ubatch in flight): each half has its own event, since event values only grow and one
// half's "Q ready" would otherwise pass the other half's "phone done" before the phone answered
void * ggml_metal_remote_event_half(void * mtl_device, int half);

// the partial buffers (id<MTLBuffer>, shared), used alternately by consecutive jobs; *host = contents
void * ggml_metal_remote_part_buffer(void * mtl_device, int i, float ** host);

// ATTN_BIG partial buffers (id<MTLBuffer>, shared; GGML_METAL_REMOTE_BIG_ROWS rows): i = phone layer parity | half << 1
void * ggml_metal_remote_big_buffer(void * mtl_device, int i, float ** host);

// queue a job: the phone thread runs it once the event reaches v_start and signals v_done
void ggml_metal_remote_submit(const struct ggml_metal_remote_job * job);

// prefill (ATTN_BIG): defer every group's merge to after the ubatch's last group, so the Mac's own attention for all groups
// runs while the phone computes (GGML_METAL_REMOTE_DEFER=0: wait at group 0 as before)
int ggml_metal_remote_defer(void);

#ifdef __cplusplus
}
#endif
