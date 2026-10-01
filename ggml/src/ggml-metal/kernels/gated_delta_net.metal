#include "common.h"

constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];

#if 1
template<short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        device const int32_t * s_ids,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B (n_seqs)
    const uint i21 = tgpig.y; // H (head)
    const uint i20 = tgpig.x*NSG + ty; // row within S_v

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    // input state layout [S_v, S_v, H, n_seqs] (s0 only): per-seq stride is H*D.
    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous
    // s_rows (ggml_gated_delta_net_replay_rows): s0 is read straight from cache row s_ids[i23]; the output may be that
    // same row (GDN_CACHE fusion): each thread reads the values it later overwrites before writing any of them
    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const float * s_ptr = args.s_rows
        ? (device const float *) (s + (uint64_t) s_ids[i23]*args.nb_s1) + (i21*S_v*S_v + i20*S_v)
        : (device const float *) (s) + state_in_base;

    float ls[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        ls[j] = s_ptr[is];
    }

    // infernet replay mode (args.replay, K == 1): the first n_replay tokens only update the state (no attention row),
    // a replay token with beta == 0 and g == 0 is an identity pad and is skipped (uniform per threadgroup: per head),
    // and the one state written out is the one after commit_at tokens
    const bool replay = args.replay != 0;
    const int  n_rp   = args.n_replay;
    const int  n_out  = args.ne22 - n_rp; // attention rows per (seq, head)

    device float * dst_attn = (device float *) (dst) + (i23*n_out*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    // args.l2 with up to 32 tokens: the per-token norm scales of q and k are computed up front (they do not depend on the
    // recurrence), one token per simdgroup at a time, so they stay off the sequential token loop
    threadgroup float l2s[64]; // [t]: k scale, [32 + t]: q scale
    const bool l2_pre = NSG == 4 && args.l2 && args.ne22 <= 32;
    if (l2_pre) {
        for (int t = ty; t < args.ne22; t += NSG) {
            const float4 k4 = *((device const float4 *) (k_ptr + t*args.ns12) + tx);
            const float4 q4 = *((device const float4 *) (q_ptr + t*args.ns02) + tx);

            float sk = 0.0f;
            sk += dot(k4, k4);
            sk = simd_sum(sk);
            float sq = 0.0f;
            sq += dot(q4, q4);
            sq = simd_sum(sq);

            if (tx == 0) {
                const float mean_k = sk/args.l2_n;
                const float mean_q = sq/args.l2_n;
                l2s[t]      = 1.0f/sqrt(mean_k + args.l2_eps);
                l2s[32 + t] = 1.0f/sqrt(mean_q + args.l2_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K, only slots 0..n_tokens-1 are written; older slots are caller-owned.

    // output state base offset: after attention scores
    const uint attn_size = n_out * args.ne21 * S_v * args.ne23;
    // output state per-slot size: S_v * S_v * H * n_seqs
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    // per-(seq,head) offset within a slot
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + attn_size;
    const uint slot_stride = fused ? (uint)args.nb_out : state_size_per_snap;

    device float * dst_commit = (device float *)state_out + state_out_base; // replay mode: the one state (slot 0)

    if (replay && args.commit_at == 0) {
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            dst_commit[tx*NSG + j] = ls[j];
        }
    }

    for (short t = 0; t < args.ne22; t++) {
        const bool rp = t < n_rp;

        if (!(rp && G == 1 && b_ptr[0] == 0.0f && g_ptr[0] == 0.0f)) {
        // this lane's NSG elements of the q and k rows (L2-normalized here when args.l2)
        float kr[NSG];
        float qr[NSG];

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            kr[j] = k_ptr[tx*NSG + j];
            qr[j] = q_ptr[tx*NSG + j];
        }

        if (NSG == 4 && args.l2) {
            const float4 k4 = float4(kr[0], kr[1], kr[2], kr[3]);
            const float4 q4 = float4(qr[0], qr[1], qr[2], qr[3]);

            float scale_k;
            float scale_q;

            if (l2_pre) {
                scale_k = l2s[t];
                scale_q = l2s[32 + t];
            } else {
            // exactly kernel_rms_norm_mul_f32_4 with use_scale at 128 wide (32 threads, one float4 each, simd_sum; the second
            // reduction over the zero-initialized shared slots adds only zeros), then * 1/sqrt(128): bit-identical to the
            // unfused RMS_NORM + SCALE of build_gdn_l2_norm (same expressions, same fast-math flags)
            float sk = 0.0f;
            sk += dot(k4, k4);
            sk = simd_sum(sk);
            float sq = 0.0f;
            sq += dot(q4, q4);
            sq = simd_sum(sq);

            const float mean_k  = sk/args.l2_n;
            scale_k = 1.0f/sqrt(mean_k + args.l2_eps);
            const float mean_q  = sq/args.l2_n;
            scale_q = 1.0f/sqrt(mean_q + args.l2_eps);
            }

            const float4 kn = (k4*scale_k)*args.l2_scale;
            const float4 qn = (q4*scale_q)*args.l2_scale;

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                kr[j] = kn[j];
                qr[j] = qn[j];
            }
        }

        float s_k = 0.0f;

        if (G == 1) {
            const float g_exp = exp(g_ptr[0]);

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                ls[j] *= g_exp;

                s_k += ls[j]*kr[j];
            }
        } else {
            // KDA
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= exp(g_ptr[is]);

                s_k += ls[j]*kr[j];
            }
        }

        s_k = simd_sum(s_k);

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        float y = 0.0f;

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            ls[j] += kr[j]*d;

            y += ls[j]*qr[j];
        }

        if (!rp) {
            y = simd_sum(y);

            if (tx == 0) {
                dst_attn[(t - n_rp)*args.ne21*S_v] = y*scale;
            }
        }
        } // !pad

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (replay && t + 1 == args.commit_at) {
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                dst_commit[tx*NSG + j] = ls[j];
            }
        }

        if (K > 1) {
            const int target_slot = (int)args.ne22 - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                device float * dst_state = (device float *)state_out + (uint)target_slot * slot_stride + state_out_base;
                FOR_UNROLL (short j = 0; j < NSG; j++) {
                    const short is = tx*NSG + j;
                    dst_state[is] = ls[j];
                }
            }
        }
    }

    if (K == 1 && !replay) {
        device float * dst_state = (device float *)state_out + state_out_base;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            dst_state[is] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
}

typedef decltype(kernel_gated_delta_net_impl<4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<4>;

#else
// a simplified version of the above
// no performance improvement, so keep the above version for now

template<typename T, short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        device const int32_t * s_ids,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B
    const uint i21 = tgpig.y; // H
    const uint i20 = tgpig.x*NSG + ty;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    device const float * s_ptr = (device const float *) (s) + (i23*args.ne21 + i21)*S_v*S_v + i20;

    float lsf[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        lsf[j] = s_ptr[is*S_v];
    }

    thread T * ls = (thread T *) (lsf);

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr  = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr  = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    for (short t = 0; t < args.ne22; t++) {
        device const T * qt_ptr = (device const T *) (q_ptr);
        device const T * kt_ptr = (device const T *) (k_ptr);
        device const T * gt_ptr = (device const T *) (g_ptr);

        if (G == 1) {
            *ls *= exp(g_ptr[0]);
        } else {
            // KDA
            *ls *= exp(gt_ptr[tx]);
        }

        const float s_k = simd_sum(dot(*ls, kt_ptr[tx]));

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        *ls += kt_ptr[tx]*d;

        const float y = simd_sum(dot(*ls, qt_ptr[tx]));

        if (tx == 0) {
            *dst_attn = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        dst_attn += args.ne21*S_v;
    }

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + args.ne23*args.ne22*args.ne21*S_v;
    const uint slot_stride = fused ? (uint)args.nb_out : S_v*S_v;

    device float * dst_state  = (device float *)state_out + (i23*args.ne21 + i21)*slot_stride + i20;
    device T     * dstt_state = (device T     *) (dst_state);

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        dst_state[is*S_v] = lsf[j];
    }

#undef S_v
#undef G
}

typedef decltype(kernel_gated_delta_net_impl<float4, 4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float,  1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float2, 2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float4, 4>;
#endif
