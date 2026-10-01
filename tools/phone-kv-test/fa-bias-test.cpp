// fa-bias-test: is the Metal flash-attention output biased when most keys score far below the best one?
// (infernet phone-kv, 2026-09-25) The GPU kernels keep P = exp(s - max) in half precision for P*V but sum the
// normaliser in fp32; if tiny P values are flushed to zero in half, every such key still counts in the normaliser
// but not in the output, and the output shrinks by that key's share.
//
// One attention op of the Qwen3.8-27B shape (24 query heads, 4 KV heads, head dim 256, q8_0 K/V), 8 query tokens,
// N keys: key 0 is a "sink" (score +SINK over the rest), the others have scores ~ N(0, SPREAD^2).
// Compares the Metal result against an fp64 reference on the same dequantized K/V.
//   fa-bias-test [N=4096] [SPREAD=3] [SINK=12]    (set GGML_METAL_FA_GQA=1 to test the verify kernel)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const int N = argc > 1 ? atoi(argv[1]) : 4096;
    const float spread = argc > 2 ? atof(argv[2]) : 3.0f;
    const float sink = argc > 3 ? atof(argv[3]) : 12.0f;
    const int D = 256, NH = 24, NKV = 4, T = 8, G = NH/NKV;
    const float scale = 1.0f/16;

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0, 1);

    // K rows ~ N(0,1); Q rows scaled so that s = scale*q.k ~ N(0, spread^2); key 0 = sink along each query
    std::vector<float> K((size_t) N*NKV*D), V((size_t) N*NKV*D), Q((size_t) T*NH*D);
    for (auto & x : K) x = nd(rng);
    for (auto & x : V) x = nd(rng);
    for (auto & x : Q) x = nd(rng)*spread/(scale*std::sqrt((float) D));
    // sink: make key 0 of each KV head aligned with the mean query of its group
    for (int h = 0; h < NKV; h++) {
        std::vector<double> qm(D, 0.0);
        for (int t = 0; t < T; t++) for (int g = 0; g < G; g++) for (int d = 0; d < D; d++) qm[d] += Q[((size_t) t*NH + h*G + g)*D + d];
        double nq = 0; for (double x : qm) nq += x*x; nq = std::sqrt(nq);
        for (int d = 0; d < D; d++) K[(size_t) h*D + d] = (float) (qm[d]/nq)*sink/(scale*std::sqrt((float) D))*0.35f + K[(size_t) h*D + d]*0.1f;
    }

    // quantize K/V to q8_0 and dequantize for the reference
    const size_t rs = ggml_row_size(GGML_TYPE_Q8_0, D);
    std::vector<uint8_t> Kq((size_t) N*NKV*rs), Vq((size_t) N*NKV*rs);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, K.data(), Kq.data(), 0, (int64_t) N*NKV, D, nullptr);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, V.data(), Vq.data(), 0, (int64_t) N*NKV, D, nullptr);
    const auto * tr = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<float> Kd(K.size()), Vd(V.size());
    tr->to_float(Kq.data(), Kd.data(), (int64_t) K.size());
    tr->to_float(Vq.data(), Vd.data(), (int64_t) V.size());

    ggml_backend_t be = ggml_backend_metal_init();
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    // layouts as llama.cpp passes them: q [D, T, NH], k/v [D, N, NKV] (views of [D*NKV, N] rows), mask [N, T_pad]
    ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NH, T);
    ggml_tensor * kc = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, D*NKV, N);
    ggml_tensor * vc = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, D*NKV, N);
    ggml_tensor * m  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, T);
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * k  = ggml_view_3d(ctx, kc, D, NKV, N, ggml_row_size(GGML_TYPE_Q8_0, D), rs*NKV, 0);
    ggml_tensor * v  = ggml_view_3d(ctx, vc, D, NKV, N, ggml_row_size(GGML_TYPE_Q8_0, D), rs*NKV, 0);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, qp, k, v, m, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    (void) buf;

    // Q as [D, NH, T]: head-major within a token
    std::vector<float> Qt((size_t) T*NH*D);
    for (int t = 0; t < T; t++) for (int h = 0; h < NH; h++) memcpy(&Qt[((size_t) t*NH + h)*D], &Q[((size_t) t*NH + h)*D], D*sizeof(float));
    ggml_backend_tensor_set(q, Qt.data(), 0, Qt.size()*sizeof(float));
    ggml_backend_tensor_set(kc, Kq.data(), 0, Kq.size());
    ggml_backend_tensor_set(vc, Vq.data(), 0, Vq.size());
    std::vector<ggml_fp16_t> mz((size_t) ggml_nelements(m), ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(m, mz.data(), 0, mz.size()*sizeof(ggml_fp16_t));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> O((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, O.data(), 0, O.size()*sizeof(float));   // [D, NH, T]

    // fp64 reference (Q rounded to half like both kernels do) + an emulation of "P flushed to zero below 2^-14 in half"
    double e_ref = 0, e_ftz = 0, shrink = 0, dropped = 0; int nrow = 0;
    for (int t = 0; t < T; t++) for (int h = 0; h < NH; h++) {
        const int hk = h/G;
        std::vector<double> s(N); double mx = -1e300;
        for (int j = 0; j < N; j++) {
            double a = 0;
            for (int d = 0; d < D; d++) a += (double) ggml_fp16_to_fp32(ggml_fp32_to_fp16(Q[((size_t) t*NH + h)*D + d]))*Kd[((size_t) j*NKV + hk)*D + d];
            s[j] = a*scale; mx = std::max(mx, s[j]);
        }
        double L = 0, Lf = 0, drop = 0; std::vector<double> o(D, 0.0), of(D, 0.0);
        for (int j = 0; j < N; j++) {
            const double p = std::exp(s[j] - mx); L += p;
            const double pf = p < 6.103515625e-05 ? 0.0 : p; drop += p - pf;
            for (int d = 0; d < D; d++) { o[d] += p*Vd[((size_t) j*NKV + hk)*D + d]; of[d] += pf*Vd[((size_t) j*NKV + hk)*D + d]; }
        }
        double on = 0, dr = 0, df = 0, dot = 0, nn = 0;
        for (int d = 0; d < D; d++) {
            const double ref = o[d]/L, ftz = of[d]/L, gpu = O[((size_t) t*NH + h)*D + d];
            on = std::max(on, std::fabs(ref)); dr = std::max(dr, std::fabs(gpu - ref)); df = std::max(df, std::fabs(gpu - ftz));
            dot += gpu*ref; nn += ref*ref;
        }
        e_ref = std::max(e_ref, dr/on); e_ftz = std::max(e_ftz, df/on); shrink += 1.0 - dot/nn; dropped += drop/L; nrow++;
    }
    printf("N %d keys, score spread %.1f, sink +%.1f: GPU vs fp64 max rel err %.2e | GPU vs 'fp64 with P<2^-14 dropped' %.2e | "
           "GPU output shrink vs fp64 %.3f%% | attention mass below 2^-14 %.3f%%\n",
           N, spread, sink, e_ref, e_ftz, 100*shrink/nrow, 100*dropped/nrow);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
