// llama-kvstate-convert: re-type the attention KV of a llama_state_seq_save_file() file.
//
//   llama-kvstate-convert IN.bin OUT.bin q8_0|f16 [ROT_K [ROT_V]]
//
// Quantized KV caches are stored Hadamard-ROTATED (src/llama-kv-cache.cpp attn_rot_k/v,
// llama-graph.cpp build_attn: K in blocks of ROT_K, V in blocks of ROT_V; Q and the output are
// rotated to match). An f16 cache is not rotated. So f16 -> quantized must rotate before
// quantizing and quantized -> f16 must rotate back (H is its own inverse). Defaults: ROT_K = the
// largest power of two >= 64 that divides the head size (256 for Qwen3.x, head 256), ROT_V = 64.
// Pass 0 0 for a cache built with LLAMA_ATTN_ROT_DISABLE=1. (v1 of this tool skipped the
// rotation: its q8_0 files were NOT what a q8_0 prefill stores; attention read garbage keys.)
//
// Exists so a long prefill is paid once: the server's slot state is saved with an f16 KV
// cache, and this rewrites it as q8_0 (or back) so a -ctk/-ctv q8_0 server can restore it.
// Layout handled (src/llama-context.cpp state_seq_save_file + llama_kv_cache::state_write):
//   u32 magic, u32 version, u32 n_token, i32 tokens[n_token],
//   attention cache: u32 n_stream; per stream: u32 cell_count, meta[cell_count], data
//     meta:  i32 pos, u32 n_seq_id, [cell_ext], i32 seq_id[n_seq_id]
//     data:  u32 v_trans, u32 n_layer, per layer K {i32 type, u64 row, rows},
//            per layer V {i32 type, u64 row, rows} (v_trans == 0 only, i.e. flash attention)
//   everything after (hybrid recurrent state) is copied verbatim.
// Assumes a hybrid (attention first) or pure-attention memory. The quantized KV is what a
// q8_0 prefill would have stored for the same tokens, except that later tokens were computed
// against f16 rather than q8_0 history (checked at 2k against a native q8_0 prefill).
#include "ggml.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>

static std::vector<uint8_t> buf;

// in-place orthonormal Walsh-Hadamard transform of consecutive blocks of n (Sylvester order,
// identical to ggml_gen_hadamard's matrix applied by llama_mul_mat_hadamard)
static void fwht_blocks(float * x, int64_t len, int n) {
    const float scale = 1.0f / sqrtf((float) n);
    for (int64_t b = 0; b + n <= len; b += n) {
        float * v = x + b;
        for (int h = 1; h < n; h *= 2) {
            for (int i = 0; i < n; i += 2*h) {
                for (int j = i; j < i + h; ++j) {
                    const float a = v[j], c = v[j + h];
                    v[j] = a + c; v[j + h] = a - c;
                }
            }
        }
        for (int i = 0; i < n; ++i) v[i] *= scale;
    }
}
static size_t off = 0;

template <typename T> static T rd() {
    if (off + sizeof(T) > buf.size()) throw std::runtime_error("truncated");
    T v; memcpy(&v, buf.data() + off, sizeof(T)); off += sizeof(T); return v;
}

// try to parse meta with a given ext size; returns offset after meta or 0 if the data header is implausible
static size_t try_meta(size_t o, uint32_t cell_count, size_t ext) {
    for (uint32_t i = 0; i < cell_count; ++i) {
        if (o + 8 > buf.size()) return 0;
        uint32_t n_seq; memcpy(&n_seq, buf.data() + o + 4, 4);
        if (n_seq > 64) return 0;
        o += 8 + ext + 4ull * n_seq;
    }
    if (o + 16 > buf.size()) return 0;
    uint32_t v_trans, n_layer; int32_t k_type;
    memcpy(&v_trans, buf.data() + o, 4); memcpy(&n_layer, buf.data() + o + 4, 4); memcpy(&k_type, buf.data() + o + 8, 4);
    if (v_trans > 1 || n_layer == 0 || n_layer > 1024 || k_type < 0 || k_type >= GGML_TYPE_COUNT) return 0;
    return o;
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 6) { fprintf(stderr, "usage: %s IN OUT q8_0|f16 [ROT_K [ROT_V]]\n", argv[0]); return 2; }
    const int rot_k_arg = argc > 4 ? atoi(argv[4]) : -1; // -1: derive from the row size (head 256 assumed)
    const int rot_v     = argc > 5 ? atoi(argv[5]) : 64;
    const std::string tname = argv[3];
    const ggml_type dst_type = tname == "q8_0" ? GGML_TYPE_Q8_0 : tname == "f16" ? GGML_TYPE_F16 : GGML_TYPE_COUNT;
    if (dst_type == GGML_TYPE_COUNT) { fprintf(stderr, "type must be q8_0 or f16\n"); return 2; }

    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); buf.resize(ftell(f)); fseek(f, 0, SEEK_SET);
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) { perror("read"); return 1; }
    fclose(f);

    std::vector<uint8_t> out; out.reserve(buf.size());
    auto put = [&](const void * p, size_t n) { out.insert(out.end(), (const uint8_t *) p, (const uint8_t *) p + n); };

    try {
        rd<uint32_t>(); rd<uint32_t>();
        const uint32_t n_token = rd<uint32_t>();
        off += 4ull * n_token;
        const uint32_t n_stream = rd<uint32_t>();
        put(buf.data(), off);

        std::vector<float> tmp;
        for (uint32_t s = 0; s < n_stream; ++s) {
            const uint32_t cell_count = rd<uint32_t>();
            put(&cell_count, 4);
            if (cell_count == 0) continue;
            size_t end = 0;
            for (size_t ext : { (size_t) 12, (size_t) 0, (size_t) 8, (size_t) 4, (size_t) 16 }) {
                if ((end = try_meta(off, cell_count, ext))) { fprintf(stderr, "cell_ext = %zu bytes\n", ext); break; }
            }
            if (!end) throw std::runtime_error("cannot parse cell metadata");
            put(buf.data() + off, end - off); off = end;

            const uint32_t v_trans = rd<uint32_t>(), n_layer = rd<uint32_t>();
            if (v_trans) throw std::runtime_error("v_trans=1 (no flash attention) is not supported");
            put(&v_trans, 4); put(&n_layer, 4);
            for (int kv = 0; kv < 2; ++kv) {
                for (uint32_t il = 0; il < n_layer; ++il) {
                    const ggml_type src_type = (ggml_type) rd<int32_t>();
                    const uint64_t src_row = rd<uint64_t>();
                    const int64_t n_embd = (int64_t) (src_row / ggml_type_size(src_type)) * ggml_blck_size(src_type);
                    const uint8_t * src = buf.data() + off;
                    off += src_row * cell_count;
                    if (off > buf.size()) throw std::runtime_error("truncated layer data");

                    const int32_t t_i = dst_type;
                    const uint64_t dst_row = ggml_row_size(dst_type, n_embd);
                    put(&t_i, 4); put(&dst_row, 8);
                    if (src_type == dst_type) { put(src, src_row * cell_count); continue; }

                    const auto * tr_src = ggml_get_type_traits(src_type);
                    if (!tr_src->to_float) throw std::runtime_error("no dequantizer for source type");
                    tmp.resize((size_t) n_embd * cell_count);
                    tr_src->to_float(src, tmp.data(), (int64_t) n_embd * cell_count);
                    // rotation state differs between the two types -> apply H (self-inverse)
                    const bool rot_src = ggml_is_quantized(src_type), rot_dst = ggml_is_quantized(dst_type);
                    int nrot = kv ? rot_v : rot_k_arg;
                    if (!kv && nrot < 0) { nrot = 64; while (256 % (nrot*2) == 0 && n_embd % (nrot*2) == 0) nrot *= 2; }
                    if (rot_src != rot_dst && nrot > 0) {
                        if (n_embd % nrot) throw std::runtime_error("row size not a multiple of the rotation size");
                        fwht_blocks(tmp.data(), (int64_t) n_embd * cell_count, nrot);
                        if (il == 0) fprintf(stderr, "%s: Hadamard rotation, block %d\n", kv ? "V" : "K", nrot);
                    }
                    const size_t o0 = out.size();
                    out.resize(o0 + dst_row * cell_count);
                    ggml_quantize_chunk(dst_type, tmp.data(), out.data() + o0, 0, cell_count, n_embd, nullptr);
                    if (il == 0) fprintf(stderr, "%s: %s -> %s, n_embd=%lld, cells=%u\n", kv ? "V" : "K",
                                         ggml_type_name(src_type), ggml_type_name(dst_type), (long long) n_embd, cell_count);
                }
            }
        }
        put(buf.data() + off, buf.size() - off); // recurrent state etc.
    } catch (const std::exception & e) {
        fprintf(stderr, "error at offset %zu: %s\n", off, e.what());
        return 1;
    }

    FILE * fo = fopen(argv[2], "wb");
    if (!fo || fwrite(out.data(), 1, out.size(), fo) != out.size()) { perror(argv[2]); return 1; }
    fclose(fo);
    fprintf(stderr, "wrote %s: %zu bytes (from %zu)\n", argv[2], out.size(), buf.size());
    return 0;
}
