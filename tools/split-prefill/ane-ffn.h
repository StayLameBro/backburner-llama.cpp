// ane-ffn.h - run the FFN of a qwen35 layer on the Apple Neural Engine (CoreML), as a llama_set_ffn_offload callback.
//
// One compiled CoreML model per layer, built from the GGUF's own weights by scripts/ane-ffn-real.py:
//   <dir>/ffn_L<layer>.mlmodelc, input "x" fp16 (1, n_embd, 1, S) (the ANE layout; transposed here), output "y" the same
//   shape. The older token-major (1, S, 1, n_embd) form also loads, but on the A19 it cost ~670 MB of app memory and 37 s
//   to load per layer (parts of it left the ANE).
//   (int8 per-output-channel weights, SwiGLU; the RMSNorm before it and the residual add after it stay in the graph).
// A ubatch of n_tokens is run as ceil(n_tokens / S) calls (S = 128 keeps the ANE's tile on-chip).
// <layer> is the model's global layer id: tail-GGUF layer il is global layer il + il_offset.
// Works on macOS and iOS (implementation: ane-ffn.mm, Objective-C++, CoreML + Accelerate).
#pragma once

#include <cstdint>
#include <string>

namespace spt {

struct ane_ffn;

struct ane_ffn_stats {
    uint64_t calls = 0, blocks = 0;
    double   ms_total = 0;   // inside the callback
    double   ms_pred  = 0;   // inside CoreML predictions
};

// Loads the models of tail layers [il0, il1). Returns nullptr and sets err on failure (e.g. a missing file).
ane_ffn *     ane_ffn_open(const std::string & dir, int il0, int il1, int il_offset, std::string & err);
void          ane_ffn_close(ane_ffn * a);
// llama_ffn_offload_fn: user = the ane_ffn *
bool          ane_ffn_run(float * y, const float * x, int32_t n_embd, int32_t n_tokens, int32_t il, void * user);
ane_ffn_stats ane_ffn_get_stats(const ane_ffn * a);
std::string   ane_ffn_last_error(const ane_ffn * a);

} // namespace spt
