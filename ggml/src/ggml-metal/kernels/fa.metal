#include "common.h"
#include "dequantize.h"

// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
// - one thread per block; dispatched separately for K and V
// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
template <
    typename block_t,
    short QK,
    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
kernel void kernel_flash_attn_ext_kv_f16(
        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
        device const char * x,
        device       half * x_dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.nblocks) {
        return;
    }

    const uint nb = args.ne0/QK;
    const uint i0 = gid%nb;
    uint ib       = gid/nb;
    const uint i1 = ib%args.ne1;
    ib /= args.ne1;
    const uint i2 = ib%args.ne2;
    const uint i3 = ib/args.ne2;

    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;

    device const block_t * src = (device const block_t *) (x + offs);
    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;

    for (short i = 0; i < QK/16; ++i) {
        float4x4 reg;
        deq_t4x4(src, i, reg);
        dst[4*i + 0] = (half4) reg[0];
        dst[4*i + 1] = (half4) reg[1];
        dst[4*i + 2] = (half4) reg[2];
        dst[4*i + 3] = (half4) reg[3];
    }
}

typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;

template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
template [[host_name("kernel_flash_attn_ext_kv_q4_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_1, 32, dequantize_q4_1>;
template [[host_name("kernel_flash_attn_ext_kv_q5_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_0, 32, dequantize_q5_0>;
template [[host_name("kernel_flash_attn_ext_kv_q5_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_1, 32, dequantize_q5_1>;
template [[host_name("kernel_flash_attn_ext_kv_q8_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>;

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];

constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// pad the last chunk of C elements of k and v into a an extra pad buffer
kernel void kernel_flash_attn_ext_pad(
        constant ggml_metal_kargs_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            // here it is not important the exact value that will be used as we rely on masking out the scores in the attention
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// scan the blocks of the mask that are not masked
// 0 -     masked (i.e. full of -INF, skip)
// 1 - not masked (i.e. at least one element of the mask is not -INF)
// 2 - all zero
kernel void kernel_flash_attn_ext_blk(
        constant ggml_metal_kargs_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    // block size C x Q
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 ? 1 : 0;

    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;

    // detailed check of the elements of the block
    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }

            mask_src += args.nb31/2;
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}

constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];

constant bool FC_flash_attn_ext_bc_mask [[function_constant(FC_FLASH_ATTN_EXT + 10)]];

//constant float FC_flash_attn_ext_scale         [[function_constant(FC_FLASH_ATTN_EXT + 10)]];
//constant float FC_flash_attn_ext_max_bias      [[function_constant(FC_FLASH_ATTN_EXT + 11)]];
//constant float FC_flash_attn_ext_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT + 12)]];

constant int32_t FC_flash_attn_ext_ns10 [[function_constant(FC_FLASH_ATTN_EXT + 20)]];
constant int32_t FC_flash_attn_ext_ns20 [[function_constant(FC_FLASH_ATTN_EXT + 21)]];
constant int32_t FC_flash_attn_ext_nsg  [[function_constant(FC_FLASH_ATTN_EXT + 22)]];
constant int32_t FC_flash_attn_ext_nwg  [[function_constant(FC_FLASH_ATTN_EXT + 23)]];
constant int32_t FC_flash_attn_ext_ncpsg [[function_constant(FC_FLASH_ATTN_EXT + 24)]];

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head size
    short DV,         // V head size
    short Q,          // queries per threadgroup
    short C,          // cache items per threadgroup
    short NSG>        // number of simd groups
void kernel_flash_attn_ext_impl(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
#define NS10 (FC_flash_attn_ext_ns10)
#define NS20 (FC_flash_attn_ext_ns20)
#define NWG  (FC_flash_attn_ext_nwg)

    // split-K over the KV cache: NWG workgroups each walk every NWG-th chunk of C
    // cache items, then kernel_flash_attn_ext_vec_reduce merges the partials.
    // NWG == 1 reproduces the original single-workgroup-per-(q-tile, head) dispatch.
    const short  iwg = tgpig[2]%NWG;
    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

    // note: I had some concerns that using this instead of the ugly macros above was affecting performance
    //       need to re-check carefully and if no regressions are observerd - remove the macros
    //       the concerns is that maybe using const variables requires extra registers? but not sure if the compiler
    //         is clever enough to avoid this. unfortunately, using constexpr is not possible with FC
    //const short NS10 = FC_flash_attn_ext_ns10;
    //const short NS20 = FC_flash_attn_ext_ns20;

    constexpr short KV   = 8;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
  //constexpr short DV8  = DV/8;
    constexpr short DV16 = DV/16;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;
  //constexpr short PV16 = PV/16;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C; // shared memory per simdgroup (s_t == float)

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV; // shared memory size per query in (half)

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T); // holds the query data
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T); // same as above but in q4_t
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK); // the result for all queries in 8x8 matrices (the O matrix from the paper)
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T); // scratch buffer for attention, mask and diagonal matrix
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T); // same as above but in s2_t

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load K in shared memory
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in k4x4_t

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // scratch buffer to load V in shared memory
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T + Q*TS); // same as above but in v4x4_t

    // mask storage in shared mem
    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    // per-query mask pointers
    device const half2 * pm2[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm2[jj] = (device const half2 *) ((device const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
        pm2[jj] += iwg*(C/2);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load heads from Q to shared memory
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        device const float4 * q4 = (device const float4 *) ((device const char *) q + j*args.nb01);

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    // zero out
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };
    float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

    {

        float slope = 1.0f;

        // ALiBi
        if (FC_flash_attn_ext_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = iwg; ; ic0 += NWG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            // the last partial chunk uses the pad buffer as source
            if (FC_flash_attn_ext_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_has_mask) {
                    threadgroup half * sm = (threadgroup half *) (sm2);

                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) {
                                sm[2*j*SH + i] = -MAXHALF;
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        pm2[jj] = (device const half2 *) ((device const half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }

                ic = 0;
            }

            char blk_cur = 1;

            // read the mask into shared mem
            if (FC_flash_attn_ext_has_mask) {
                blk_cur = blk[ic0];

                if (blk_cur == 0) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NWG*(C/2);
                    }

                    continue;
                }

                if (blk_cur == 1) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;

                        if (2*tiisg < C) {
                            if (FC_flash_attn_ext_bc_mask) {
                                sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF, -MAXHALF);
                            } else {
                                sm2[j*SH + tiisg] = pm2[jj][tiisg];
                            }
                        }

                        pm2[jj] += NWG*(C/2);
                    }
                } else if (blk_cur == 2) {
                    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                        pm2[jj] += NWG*(C/2);
                    }
                }

#if 0
                // note: old -INF block optimization - obsoleted by pre-computing non-masked blocks

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // used to detect blocks full of -INF
                // skip only when the entire threadgroup is masked
                half2 smax2(-MAXHALF/2, -MAXHALF/2);

                FOR_UNROLL (short j = 0; j < Q; ++j) {
                    smax2 = max(smax2, sm2[j*SH + tiisg]);
                }

                smax2 = simd_max(smax2);

                if (max(smax2[0], smax2[1]) <= -MAXHALF/2) {
                    // this barrier is important
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    continue;
                }
#endif
            }

            // Q*K^T
            // this is compile-time check, so it does not have runtime overhead
            if (is_same<kd4x4_t, k4x4_t>::value) {
                // we can read directly from global memory
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                static_assert((C/8) % NSG == 0, "");

                constexpr short NC = (C/8)/NSG;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    if (DK % 16 != 0) {
                        k8x8_t mk;
                        q8x8_t mq;

                        FOR_UNROLL (short i = 0; i < DK8; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mk, pk + 8*i, NS10, 0, true);
                            simdgroup_load(mq, pq + 8*i, DK);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        k8x8_t mk[2];
                        q8x8_t mq[2];

                        // note: too much unroll can tank the performance for large heads
                        #pragma unroll (MIN(DK8/2, 4*NSG))
                        for (short i = 0; i < DK8/2; ++i) {
                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                            simdgroup_barrier(mem_flags::mem_none);

                            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                        }
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            } else {
                // TODO: this is the quantized K cache branch - not optimized yet
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    for (short ii = 0; ii < DK16; ii += 4) {
                        device const kd4x4_t * pk4x4 = (device const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));

                        if (DK16%4 == 0) {
                            // the head is evenly divisible by 4*16 = 64, so no need for bound checks
                            {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            FOR_UNROLL (short k = 0; k < 4; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        } else {
                            if (ii + tx < DK16) {
                                k4x4_t tmp;
                                deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                sk4x4[4*ty + tx] = tmp;
                            }

                            simdgroup_barrier(mem_flags::mem_threadgroup);

                            for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                k8x8_t mk;
                                q8x8_t mq;

                                simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                            }
                        }
                    }

                    simdgroup_store(mqk, ss + 8*cc, SH, 0, false);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                // each lane owns one float2 of the score row, so the loop covers 2*NW = 64
                // columns. That is exactly C when C == 64, but over-runs it when C < 64: the
                // surplus lanes would read stale exp() values left in ss by the previous
                // iteration and fold them into M and S. Mask them to -inf instead.
                // Compile-time, so C == 64 folds this away entirely.
                const bool lane_ok = (2*tiisg) < C;

                // scale and apply the logitcap / mask
                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                if (FC_flash_attn_ext_has_scap) {
                    s2 = args.logit_softcap*precise::tanh(s2);
                }

                // mqk = mqk + slope*mask
                if (blk_cur != 2) {
                    if (FC_flash_attn_ext_has_bias) {
                        s2 += s2_t(sm2[j*SH + tiisg])*slope;
                    } else {
                        s2 += s2_t(sm2[j*SH + tiisg]);
                    }
                }

                if (!lane_ok) {
                    s2 = float2(-FLT_MAX/2, -FLT_MAX/2);
                }

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = lane_ok ? exp(s2 - M[jj]) : float2(0.0f, 0.0f);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                // the P matrix from the paper (Q rows, C columns). Zero past C so the O = P*V
                // simdgroup_multiply_accumulate below cannot pick up a surplus column either.
                ss2[j*SH/2 + tiisg] = vs2;

                if (DV4 % NW == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                        const short i = ii*NW + tiisg;

                        so4[j*PV4 + i] *= ms;
                    }
                } else {
                    for (short i = tiisg; i < DV4; i += NW) {
                        so4[j*PV4 + i] *= ms;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                // we can read directly from global memory
                if (is_same<vd4x4_t, v4x4_t>::value) {
                    static_assert(PV8 % NSG == 0, "");

                    constexpr short NO = PV8/NSG;

                    o8x8_t lo[NO];

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_load(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }

                    {
                        device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                        pv += 8*sgitg;

                        if (DV <= 64) {
                            FOR_UNROLL (short cc = 0; cc < C/8; ++cc) {
                                s8x8_t vs;
                                simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[2];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs, mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs, mv[1], lo[2*ii + 1]);
                                }

                                pv  += 8*NS20;
                            }
                        } else {
                            constexpr short NC = (C/8)/2;

                            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                                s8x8_t vs[2];

                                simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                                simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                                FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                                    v8x8_t mv[4];

                                    simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                                    simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                                }

                                pv  += 2*8*NS20;
                            }
                        }
                    }

                    {
                        auto sot = so + 8*sgitg;

                        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                            simdgroup_store(lo[ii], sot, PV, 0, false);

                            sot += 8*NSG;
                        }
                    }
                } else {
                    // TODO: this is the quantized V cache branch - not optimized yet

                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    for (short cc = 0; cc < C/8; ++cc) {
                        s8x8_t vs;
                        simdgroup_load(vs, ss + 8*cc, SH, 0, false);

                        for (short ii = 4*sgitg; ii < DV16; ii += 4*NSG) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) (v + ((ic + 8*cc + ty)*args.nb21));

                            if (DV16%4 == 0) {
                                // no need for bound checks
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                FOR_UNROLL (short k = 0; k < 4; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv[2];
                                    o8x8_t lo[2];

                                    simdgroup_load(mv[0], sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_load(mv[1], sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_load(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_load(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);

                                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);

                                    simdgroup_store(lo[0], so + 8*(2*(ii + k) + 0), PV, 0, false);
                                    simdgroup_store(lo[1], so + 8*(2*(ii + k) + 1), PV, 0, false);
                                }
                            }
                        }
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (FC_flash_attn_ext_has_sinks && iwg == 0) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = simd_max(max(M[jj], s));

                const float ms = exp(m - M[jj]);
                const float vs = exp(s - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs);

                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }
    }

    // store to global memory
    if (NWG == 1) {
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;
            if (iq1 + j >= args.ne01) {
                break;
            }

            device float4 * dst4 = (device float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;

            const float scale = S[jj] == 0.0 ? 0.0f : 1.0f/S[jj];

            if (DV4 % NW == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                    const short i = ii*NW + tiisg;

                    dst4[i] = (float4) so4[j*PV4 + i]*scale;
                }
            } else {
                for (short i = tiisg; i < DV4; i += NW) {
                    dst4[i] = (float4) so4[j*PV4 + i]*scale;
                }
            }
        }
    } else {
        // partial results, in the layout kernel_flash_attn_ext_vec_reduce expects:
        // dst4[rid*DV4*NWG + NWG*i + iwg]  then  S/M at nrows*DV*NWG + rid*2*NWG
        const uint64_t nrows = (uint64_t)args.ne3*args.ne2*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;
            if (iq1 + j >= args.ne01) {
                break;
            }

            const uint64_t rid = (uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1;

            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[j*PV4 + i];
            }

            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = S[jj];
                dst1[rid*(2*NWG) + 2*iwg + 1] = M[jj];
            }
        }
    }

#undef NS10
#undef NS20
#undef NWG
}

template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s2_t,
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // value type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,         // K head size
    short DV,         // V head size
    short Q  = OP_FLASH_ATTN_EXT_NQPSG, // queries per threadgroup
    short C  = OP_FLASH_ATTN_EXT_NCPSG> // cache items per threadgroup
kernel void kernel_flash_attn_ext(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device const char * blk,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#define FWD_TMPL_NC q_t, q4_t, q8x8_t, k_t, k4x4_t, k8x8_t, v_t, v4x4_t, v8x8_t, qk_t, qk8x8_t, s_t, s2_t, s8x8_t, o_t, o4_t, o8x8_t, kd4x4_t, nl_k, deq_k, vd4x4_t, nl_v, deq_v, DK, DV, Q
#define FWD_ARGS args, q, k, v, mask, sinks, pad, blk, dst, shmem_f16, tgpig, tiisg, sgitg
    // (nsg, ncpsg) pairs. FATTN_SMEM is
    //   (nqptg*(ne00 + 2*PAD(ne20,64) + 2*(2*ncpsg)) + is_q*(16*32*nsg)) * 2 bytes
    // so at ne00=ne20=256, nqptg=8, is_q=1 (quantized KV, which is what a verify at ne01<32
    // takes since use_kv_f16() declines to pre-dequant there):
    //   ncpsg=64 nsg=4 -> 20.0 KB -> ONE resident threadgroup per core on a 32 KB limit
    //   ncpsg=32 nsg=2 -> 16.0 KB -> TWO
    // Split-K raises threadgroup COUNT; this raises RESIDENCY. They are independent and stack.
    // (C/8) % NSG == 0 holds for both new pairs: 32/8 = 4, 4%2 == 0, 4%4 == 0.
    switch (FC_flash_attn_ext_nsg*1000 + FC_flash_attn_ext_ncpsg) {
        case 4*1000 + 64: kernel_flash_attn_ext_impl<FWD_TMPL_NC, 64, 4>(FWD_ARGS); break;
        case 8*1000 + 64: kernel_flash_attn_ext_impl<FWD_TMPL_NC, 64, 8>(FWD_ARGS); break;
        case 2*1000 + 32: kernel_flash_attn_ext_impl<FWD_TMPL_NC, 32, 2>(FWD_ARGS); break;
        case 4*1000 + 32: kernel_flash_attn_ext_impl<FWD_TMPL_NC, 32, 4>(FWD_ARGS); break;
        default:
            // An unhandled (nsg, ncpsg) pair used to fall straight through, leaving dst
            // unwritten: fast, silent, plausible-looking garbage. That shape of bug has hit this
            // switch twice -- once via GGML_METAL_FA_NSG=1|2 (never instantiated) and once via
            // (nsg=8, ncpsg=32), which cannot exist because (C/8) % NSG == 4 % 8. Fail loudly.
            if (tiisg == 0 && sgitg == 0) {
                ((device float *) dst)[0] = NAN;
            }
            break;
    }
#undef FWD_TMPL
#undef FWD_ARGS
}

// TODO: this is quite ugly. in the future these types will be hardcoded in the kernel, but for now keep them as
//       template to be able to explore different combinations
//
#define FA_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

#define FA_TYPES_BF \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    half,   half4,     simdgroup_half8x8
    //float,  float4,    simdgroup_float8x8

#define FA_TYPES_F32 \
    half,   half4,     simdgroup_half8x8,  \
    float,  float4x4,  simdgroup_float8x8, \
    float,  float4x4,  simdgroup_float8x8, \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_f32_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  32,  32>;
template [[host_name("kernel_flash_attn_ext_f32_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  40,  40>;
template [[host_name("kernel_flash_attn_ext_f32_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  48,  48>;
template [[host_name("kernel_flash_attn_ext_f32_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f32_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  72,  72>;
template [[host_name("kernel_flash_attn_ext_f32_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f32_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f32_dk96_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  96,  64>;
template [[host_name("kernel_flash_attn_ext_f32_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f32_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f32_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  192, 192>;
template [[host_name("kernel_flash_attn_ext_f32_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f32_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f32_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  320, 256>;
template [[host_name("kernel_flash_attn_ext_f32_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  512, 512>;
template [[host_name("kernel_flash_attn_ext_f32_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  576, 512>;

template [[host_name("kernel_flash_attn_ext_f16_dk32_dv32"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  32,  32>;
template [[host_name("kernel_flash_attn_ext_f16_dk40_dv40"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  40,  40>;
template [[host_name("kernel_flash_attn_ext_f16_dk48_dv48"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  48,  48>;
template [[host_name("kernel_flash_attn_ext_f16_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f16_dk72_dv72"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  72,  72>;
template [[host_name("kernel_flash_attn_ext_f16_dk80_dv80"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f16_dk96_dv96"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f16_dk96_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  64>;
template [[host_name("kernel_flash_attn_ext_f16_dk112_dv112")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f16_dk128_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv192")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 192>;
template [[host_name("kernel_flash_attn_ext_f16_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f16_dk256_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk320_dv256")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  320, 256>;
template [[host_name("kernel_flash_attn_ext_f16_dk512_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  512, 512>;
template [[host_name("kernel_flash_attn_ext_f16_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  576, 512>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_bf16_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 32,  32>;
template [[host_name("kernel_flash_attn_ext_bf16_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 40,  40>;
template [[host_name("kernel_flash_attn_ext_bf16_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 48,  48>;
template [[host_name("kernel_flash_attn_ext_bf16_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 64,  64>;
template [[host_name("kernel_flash_attn_ext_bf16_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 72,  72>;
template [[host_name("kernel_flash_attn_ext_bf16_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 80,  80>;
template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  96>;
template [[host_name("kernel_flash_attn_ext_bf16_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 96,  64>;
template [[host_name("kernel_flash_attn_ext_bf16_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 112, 112>;
template [[host_name("kernel_flash_attn_ext_bf16_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 128, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 192>;
template [[host_name("kernel_flash_attn_ext_bf16_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 192, 128>;
template [[host_name("kernel_flash_attn_ext_bf16_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 256, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 320, 256>;
template [[host_name("kernel_flash_attn_ext_bf16_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 512, 512>;
template [[host_name("kernel_flash_attn_ext_bf16_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_BF, bfloat4x4,  1, dequantize_bf16, bfloat4x4,  1, dequantize_bf16, 576, 512>;
#endif

template [[host_name("kernel_flash_attn_ext_q4_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 96,  64>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q4_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_0, 2, dequantize_q4_0, block_q4_0, 2, dequantize_q4_0, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q4_1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 96,  64>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q4_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q4_1, 2, dequantize_q4_1, block_q4_1, 2, dequantize_q4_1, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q5_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 96,  64>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q5_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_0, 2, dequantize_q5_0, block_q5_0, 2, dequantize_q5_0, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q5_1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96,  64>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 576, 512>;

template [[host_name("kernel_flash_attn_ext_q8_0_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96,  64>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q8_0_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 576, 512>;

#undef FA_TYPES
#undef FA_TYPES_BF
#undef FA_TYPES_F32

constant bool FC_flash_attn_ext_vec_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 0)]];
constant bool FC_flash_attn_ext_vec_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_VEC + 1)]];
constant bool FC_flash_attn_ext_vec_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 2)]];
constant bool FC_flash_attn_ext_vec_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 3)]];
constant bool FC_flash_attn_ext_vec_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT_VEC + 4)]];

//constant float FC_flash_attn_ext_vec_scale         [[function_constant(FC_FLASH_ATTN_EXT_VEC + 10)]];
//constant float FC_flash_attn_ext_vec_max_bias      [[function_constant(FC_FLASH_ATTN_EXT_VEC + 11)]];
//constant float FC_flash_attn_ext_vec_logit_softcap [[function_constant(FC_FLASH_ATTN_EXT_VEC + 12)]];

constant int32_t FC_flash_attn_ext_vec_ns10 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 20)]];
constant int32_t FC_flash_attn_ext_vec_ns20 [[function_constant(FC_FLASH_ATTN_EXT_VEC + 21)]];
constant int32_t FC_flash_attn_ext_vec_nsg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 22)]];
constant int32_t FC_flash_attn_ext_vec_nwg  [[function_constant(FC_FLASH_ATTN_EXT_VEC + 23)]];
constant bool    FC_flash_attn_ext_vec_has_sparse [[function_constant(FC_FLASH_ATTN_EXT_VEC + 5)]];

// compress the finite entries of each KQ mask row into a list of KV indices (ascending order),
// padded with -1 up to n_kv_max_padded (a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG)
// one threadgroup per mask row; the mask remains the single source of truth for the values
kernel void kernel_flash_attn_ext_vec_idx(
        constant ggml_metal_kargs_flash_attn_ext_vec_idx & args,
        device const half * mask,
        device       int  * idx,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NLOCAL = 32; // max finite positions kept in registers per thread

    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];

    device const half * pm  = (device const half *) ((device const char *) mask + i1*args.nb31 + i2*args.nb32 + i3*args.nb33);
    device int * pidx = idx + (((int64_t)i3*args.ne32 + i2)*args.ne31 + i1)*args.n_kv_max_padded;

    const int n  = args.ne30;
    const int q  = n/ntg.x;
    const int r  = n%ntg.x;

    // each thread handles a contiguous slice of the mask row
    const int r0 = q*tiitg + min((int) tiitg, r);
    const int r1 = r0 + q + (tiitg < r ? 1 : 0);

    // count the finite entries in the slice and keep their positions in registers (single mask read)
    int cnt = 0;  // total finite entries in the slice
    int nloc = 0; // finite entries kept in registers
    int local[NLOCAL];
    for (int i = r0; i < r1; ++i) {
        if (isfinite((float) pm[i])) {
            if (nloc < NLOCAL) {
                local[nloc] = i;
                nloc++;
            }
            cnt++;
        }
    }

    const short sgitg = tiitg/NW;
    const short tiisg = tiitg%NW;

    threadgroup int tcount[8];

    // simd_sum is a collective: all lanes must evaluate it
    const int sg_sum = simd_sum(cnt);
    if (tiisg == 0) {
        tcount[sgitg] = sg_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int total = 0;
    for (short s = 0; s < ntg.x/NW; ++s) {
        total += tcount[s];
    }

    // base offset of this thread's slice in the output list (exclusive scan within the simdgroup)
    int sg_base = 0;
    for (short s = 0; s < sgitg; ++s) {
        sg_base += tcount[s];
    }

    // exclusive prefix scan of the per-thread counts within the simdgroup
    int incl = cnt;
    for (int d = 1; d < NW; d <<= 1) {
        const int v = simd_shuffle_up(incl, d);
        if (tiisg >= d) {
            incl += v;
        }
    }
    const int base = sg_base + (incl - cnt);

    // write the finite positions in order; if the hint is violated, keep only the first n_kv_max entries
    int j = 0;
    for (; j < nloc && base + j < args.n_kv_max; ++j) {
        pidx[base + j] = local[j];
    }

    // a dense mask may have more than NLOCAL finite entries in a slice; re-read the mask to write the rest
    if (cnt > nloc && base + nloc < args.n_kv_max) {
        int j2 = 0;
        for (int i = r0; i < r1; ++i) {
            if (isfinite((float) pm[i])) {
                if (j2 >= nloc) {
                    pidx[base + j2] = i;
                }
                j2++;
                if (base + j2 >= args.n_kv_max) {
                    break;
                }
            }
        }
    }

    // pad the tail of the list with -1
    const int count = min(total, args.n_kv_max);
    for (int i = count + tiitg; i < args.n_kv_max_padded; i += ntg.x) {
        pidx[i] = -1;
    }
}

template<
    typename q4_t,  // query types in shared memory
    typename k4_t,  // key types in shared memory
    typename v4_t,  // value types in shared memory
    typename qk_t,  // Q*K types
    typename s_t,   // soft-max types
    typename s4_t,
    typename o4_t,  // attention accumulation types
    typename kd4_t, // key type in device memory
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t, // value type in device memory
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,       // K head size
    short DV,       // V head size
    short NE = 4,   // head elements per thread
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,  // queries per threadgroup
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>  // cache items per threadgroup

kernel void kernel_flash_attn_ext_vec(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        device const char * idx,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)

#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;

    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE; // note: this can be adjusted to support different head sizes and simdgroup work loads
    constexpr short SH  = 4*Q*C; // shared memory per simdgroup

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

  //const short T = PK + NSG*SH; // shared memory size per query in (half)

  //threadgroup q_t   * sq  = (threadgroup q_t   *) (shmem_f16 +                          0*PK); // holds the query data
    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                          0*PK); // same as above but in q4_t
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH         + Q*NSG*PK); // scratch buffer for attention
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH         + Q*NSG*PK); // same as above but in s4_t
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*Q*C + Q*NSG*PK); // scratch buffer for mask
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*Q*PV       + Q*NSG*PK + NSG*SH); // scratch buffer for the results

    // store the result for all queries in shared memory (the O matrix from the paper)
    so4 += tiisg;

    {
        q += iq1*Q*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load Q query rows to shared memory
    {
        for (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            device const float4 * q4 = (device const float4 *) ((device const char *) q + qq*args.nb01);
            if (iq1_q < args.ne01) {
                for (short i = tiisg; i < PK4; i += NW) {
                    if (i < DK4) {
                        sq4[qq*PK4 + i] = (q4_t) q4[i];
                    } else {
                        sq4[qq*PK4 + i] = (q4_t) 0.0f;
                    }
                }
            } else {
                for (short i = tiisg; i < PK4; i += NW) {
                    sq4[qq*PK4 + i] = (q4_t) 0.0f;
                }
            }
        }
    }

    // zero out so
    for (short qq = 0; qq < Q; ++qq) {
        for (short i = 0; i < DV4/NL; ++i) {
            so4[qq*DV4 + i*NL] = (o4_t) 0.0f;
        }
    }

    // zero out shared memory SH
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S[Q];
        float M[Q];
        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            S[qq] = 0.0f;
            M[qq] = -FLT_MAX/2;
        }

        // thread indices inside the simdgroup
        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        // pointer to the mask
        device const half * pm_base = (device const half *) (mask + iq1*Q*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        // sparse indices: the list of finite mask entries per query row
        // the sparse path requires Q == 1 (enforced by the host)
        device const int * pidx = nullptr;
        if (FC_flash_attn_ext_vec_has_sparse) {
            pidx = (device const int *) idx +
                ((int64_t)(iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*args.ne31*args.n_kv_max_padded + (iq1%args.ne31)*args.n_kv_max_padded;
        }

        float slope = 1.0f;

        // ALiBi
        if (FC_flash_attn_ext_vec_has_bias) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            device const half * pm[Q];
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                // padded query rows clamp to row 0 of the mask to avoid OOB; their scores
                // are forced to -inf below, so the values never affect the result.
                pm[qq] = pm_base + ((iq1*Q + qq) < args.ne01 ? qq*(args.nb31/sizeof(half)) : -iq1*Q*(args.nb31/sizeof(half)));
            }

            // the last partial chunk uses the pad buffer as source
            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if (ic + tiisg >= args.ne11) {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                } else {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        pm[qq] = (device const half *) (mask) +
                            (iq1*Q + qq)*C +
                            (iq2%args.ne32)*(C*args.ne31) +
                            (iq3%args.ne33)*(C*args.ne31*args.ne32);
                    }
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                if (FC_flash_attn_ext_vec_has_sparse) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        const int i11 = pidx[ic + tiisg];
                        if ((iq1*Q + qq) < args.ne01 && i11 >= 0) {
                            sm[qq*C + tiisg] = pm[qq][i11];
                        } else {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                } else {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if ((iq1*Q + qq) < args.ne01) {
                            sm[qq*C + tiisg] = pm[qq][ic + tiisg];
                        } else {
                            sm[qq*C + tiisg] = -MAXHALF;
                        }
                    }
                }
            } else {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if ((iq1*Q + qq) >= args.ne01) {
                        sm[qq*C + tiisg] = -MAXHALF;
                    }
                }
            }

            // skip -INF mask
            {
                bool any_finite = false;
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if (simd_max(sm[qq*C + tiisg]) > -MAXHALF) {
                        any_finite = true;
                    }
                }
                if (!any_finite) {
                    continue;
                }
            }

            // Q*K^T
            {
                device      const k4_t * pk4 = nullptr;

                if (!FC_flash_attn_ext_vec_has_sparse) {
                    pk4 = (device const k4_t *) (k + ic*args.nb11);

                    pk4 += ty*NS10/4 + tx;
                }

                qk_t mqk[Q][C/NE];
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        mqk[qq][cc] = 0.0f;
                    }
                }

                // each simdgroup processes Q queries and NE (NW/NL) cache elements
                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (FC_flash_attn_ext_vec_has_sparse) {
                        // the KV rows are gathered from the index list; -1 entries are padding
                        const int i11 = pidx[ic + NE*cc + ty];
                        if (i11 >= 0) {
                            if (is_same<kd4_t, k4_t>::value) {
                                device const k4_t * pk4s = (device const k4_t *) (k + i11*args.nb11) + tx;
                                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                                    const k4_t k_elem = pk4s[ii*NL];
                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
                                    }
                                }
                            } else {
                                device const kd4_t * pk = (device const kd4_t *) (k + i11*args.nb11);

                                k4_t mk;

                                FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                                    const short i = ii*NL + tx;

                                    deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq*PK4 + i]);
                                    }
                                }
                            }
                        }
                    } else if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const k4_t k_elem = pk4[cc*NE*NS10/4 + ii*NL];
                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                mqk[qq][cc] += dot((float4) k_elem, (float4) sq4[qq*PK4 + ii*NL + tx]);
                            }
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                mqk[qq][cc] += dot((float4) mk, (float4) sq4[qq*PK4 + i]);
                            }
                        }
                    }

                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        if (NE == 1) {
                            mqk[qq][cc] = simd_sum(mqk[qq][cc]);
                        } else {
                            // simdgroup reduce (NE = 4)
                            // [ 0 ..  7] -> [ 0]
                            // [ 8 .. 15] -> [ 8]
                            // [16 .. 23] -> [16]
                            // [24 .. 31] -> [24]
                            if (NE <= 1) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc], 16);
                            }
                            if (NE <= 2) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  8);
                            }
                            if (NE <= 4) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  4);
                            }
                            if (NE <= 8) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  2);
                            }
                            if (NE <= 16) {
                                mqk[qq][cc] += simd_shuffle_down(mqk[qq][cc],  1);
                            }

                            // broadcast
                            mqk[qq][cc] = simd_shuffle(mqk[qq][cc], NL*ty);
                        }
                    }
                }

                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    if (FC_flash_attn_ext_vec_has_mask &&
                       !FC_flash_attn_ext_vec_has_scap &&
                       !FC_flash_attn_ext_vec_has_bias) {
                        ss[qq*C + NE*tx + ty] = fma(mqk[qq][tx], args.scale, (qk_t) sm[qq*C + NE*tx + ty]);
                    } else {
                        mqk[qq][tx] *= args.scale;

                        if (FC_flash_attn_ext_vec_has_scap) {
                            mqk[qq][tx] = args.logit_softcap*precise::tanh(mqk[qq][tx]);
                        }

                        if (FC_flash_attn_ext_vec_has_bias) {
                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty]*slope;
                        } else {
                            mqk[qq][tx] += (qk_t) sm[qq*C + NE*tx + ty];
                        }

                        ss[qq*C + NE*tx + ty] = mqk[qq][tx];
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    const float m = M[qq];
                    const float s = ss[qq*C + tiisg];

                    M[qq] = simd_max(max(M[qq], s));

                    const float ms = exp(m - M[qq]);
                    const float vs = exp(s - M[qq]);

                    S[qq] = S[qq]*ms + simd_sum(vs);

                    // the P matrix from the paper (Q rows, C columns)
                    ss[qq*C + tiisg] = vs;

                    // O = diag(ms)*O
                    if ((DV4/NL % NW == 0) || ty == 0) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] *= ms;
                        }
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                o4_t lo[Q][DV4/NL];
                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[qq][ii] = 0.0f;
                    }
                }

                if (FC_flash_attn_ext_vec_has_sparse) {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        // the KV rows are gathered from the index list; -1 entries are padding
                        const int i11 = pidx[ic + NE*cc + ty];
                        if (i11 >= 0) {
                            if (is_same<vd4_t, v4_t>::value) {
                                device const v4_t * pv4 = (device const v4_t *) (v + i11*args.nb21);

                                pv4 += tx;

                                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                                    const v4_t v_elem = pv4[ii*NL];
                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
                                    }
                                }
                            } else {
                                device const vd4_t * pv4 = (device const vd4_t *) (v + i11*args.nb21);

                                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                                    const short i = ii*NL + tx;

                                    v4_t mv;

                                    deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                        lo[qq][ii] += o4_t(float4(mv)*float4(ss[qq*C + cc*NE + ty]));
                                    }
                                }
                            }
                        }
                    }
                } else if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const v4_t v_elem = pv4[cc*NE*NS20/4 + ii*NL];
                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                lo[qq][ii] += o4_t(float4(v_elem)*float4(ss[qq*C + cc*NE + ty]));
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                                lo[qq][ii] += o4_t(float4(mv)*float4(ss[qq*C + NE*cc + ty]));
                            }
                        }
                    }
                }

                FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        if (NE > 1) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0], 16);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1], 16);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2], 16);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3], 16);
                        }

                        if (NE > 2) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  8);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  8);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  8);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  8);
                        }

                        if (NE > 4) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  4);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  4);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  4);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  4);
                        }

                        if (NE > 8) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  2);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  2);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  2);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  2);
                        }

                        if (NE > 16) {
                            lo[qq][ii][0] += simd_shuffle_down(lo[qq][ii][0],  1);
                            lo[qq][ii][1] += simd_shuffle_down(lo[qq][ii][1],  1);
                            lo[qq][ii][2] += simd_shuffle_down(lo[qq][ii][2],  1);
                            lo[qq][ii][3] += simd_shuffle_down(lo[qq][ii][3],  1);
                        }
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4[qq*DV4 + ii*NL] += lo[qq][ii];
                        }
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const float m = M[qq];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2] : -FLT_MAX/2;

                M[qq] = simd_max(max(M[qq], s));

                const float ms = exp(m - M[qq]);
                const float vs = exp(s - M[qq]);

                S[qq] = S[qq]*ms + simd_sum(vs);

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[qq*DV4 + ii*NL] *= ms;
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                ss[2*qq + 0] = (s_t) S[qq];
                ss[2*qq + 1] = (s_t) M[qq];
            }
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
                const float S0 = ss[                2*qq + 0];
                const float S1 = ss[r*(SH/2) +      2*qq + 0];

                const float M0 = ss[                2*qq + 1];
                const float M1 = ss[r*(SH/2) +      2*qq + 1];

                const float Mx  = max(M0, M1);

                const float ms0 = exp(M0 - Mx);
                const float ms1 = exp(M1 - Mx);

                const float Sx  = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[2*qq + 0] = Sx;
                    ss[2*qq + 1] = Mx;
                }

                // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
                for (short i = tiisg; i < DV4; i += NW) {
                    so4[qq*DV4 + i] = so4[qq*DV4 + i]*ms0 + so4[qq*DV4 + i + r*Q*PV4]*ms1;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG; // the S and M are stored after the results

        FOR_UNROLL (short qq = 0; qq < Q; ++qq) {
            const int iq1_q = iq1*Q + qq;
            if (iq1_q >= args.ne01) {
                continue;
            }

            const int64_t rid = iq3*args.ne2*args.ne1 + iq2 + iq1_q*args.ne1;

            const float Sval = NWG == 1 ? (ss[2*qq + 0] == 0.0f ? 0.0f : 1.0f/ss[2*qq + 0]) : 1.0f;

            // interleave the workgroup data
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[qq*DV4 + i]*Sval;
            }

            // store S and M
            if (NWG > 1) {
                if (tiisg == 0) {
                    dst1[rid*(2*NWG) + 2*iwg + 0] = ss[2*qq + 0];
                    dst1[rid*(2*NWG) + 2*iwg + 1] = ss[2*qq + 1];
                }
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

// note: I think the s_t can be half instead of float, because the Q*K scaling is done before storing to shared mem
//       in the other (non-vec) kernel, we need s_t to also be float because we scale during the soft_max
//
#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  32, 32, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 32, 32, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 64, 64, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 96, 96, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk96_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 64, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 96, 64, 4>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 96, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 96, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 96, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 96, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 128, 128, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 192, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 128, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 4>;
// G11: Q=8 (one speculative verify block per threadgroup). Shares each K/V read across all 8 queries.
template [[host_name("kernel_flash_attn_ext_vec_f16_dk256_dv256_q8_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4, 8>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 256, 256, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk256_dv256_q8_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 4, 8>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q8_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 8>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 320, 256, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 512, 512, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f32_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_f16_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4, 4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_bf16_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 576, 512, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q4_1_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q5_1_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 4>;


#undef FA_TYPES
#undef FA_TYPES_F32

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

kernel void kernel_flash_attn_ext_vec_reduce(
        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    // the simd reductions below span all 32 lanes, so lanes past NWG must contribute
    // identity values rather than read the next row's partials. NWG == 32 is the only
    // case the vec path ever dispatched, which is why this was previously unguarded.
    const bool active = iwg < NWG;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = active ? ss[rid*(2*NWG) + 2*iwg + 0] : 0.0f;
    float M = active ? ss[rid*(2*NWG) + 2*iwg + 1] : -FLT_MAX/2;

    const float m  = simd_max(M);
    const float ms = active ? exp(M - m) : 0.0f;

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(active ? htmp4[i*NWG + iwg]*ms : float4(0.0f));

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}

// infernet SME co-attention: the CPU writes its partial into its own shared buffer (O [nrows][256], then (S, M) per row
// at src_sm); this copies it into slot iwg of the split-K partials, so only the GPU ever writes the partials buffer
// (CPU and GPU writing neighbouring bytes of the same cache lines at the same time lost updates).
kernel void kernel_flash_attn_ext_coattn_scatter(
        constant ggml_metal_kargs_flash_attn_ext_coattn_scatter & args,
        device const float * src,
        device       float * htmp,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    const uint64_t rid = tgpig;
    const uint64_t sid = rid + (uint64_t) args.row0;   // source row (phone ATTN_BIG: this group's rows of the ubatch)
    const int      NWG = args.nwg;

    device const float4 * s4 = (device const float4 *) (src + sid*256);
    device       float4 * d4 = (device       float4 *) htmp;

    d4[rid*64*NWG + (uint64_t) NWG*tiitg + args.iwg] = s4[tiitg];

    if (tiitg == 0) {
        device float * d1 = htmp + (uint64_t) args.nrows*256*NWG;
        d1[rid*(2*NWG) + 2*args.iwg + 0] = src[(uint64_t) args.src_sm + 2*sid + 0];
        d1[rid*(2*NWG) + 2*args.iwg + 1] = src[(uint64_t) args.src_sm + 2*sid + 1];
    }
}

template<
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread half4x4 &)>
kernel void kernel_lightning_indexer(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK    = OP_LIGHTNING_INDEXER_DK;
    constexpr short NH    = OP_LIGHTNING_INDEXER_NH;
    constexpr short NHPTG = OP_LIGHTNING_INDEXER_NHPTG;
    constexpr short NKPSG = OP_LIGHTNING_INDEXER_NKPSG;
    constexpr short NSG   = OP_LIGHTNING_INDEXER_NSG;
    constexpr short NBPTG = OP_LIGHTNING_INDEXER_NBPTG;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;

    constexpr short NK  = NKPSG*NSG; // keys    per threadgroup
    constexpr short NTG = 32*NSG;    // threads per threadgroup

    const int i_stream = tgpig.z;
    const int i_kv_0   = tgpig.x*NK;            // first key of this threadgroup
    const int i_kv     = i_kv_0 + sgitg*NKPSG;  // first key of this simdgroup

    threadgroup half sk[NK * DK16 * 16];
    threadgroup half4x4 * sk4x4 = (threadgroup half4x4 *) sk;

    for (short i = tiitg; i < NK*DK16; i += NTG) {
        const short ik  = i/DK16;
        const short i16 = i%DK16;

        half4x4 tmp;

        if (i_kv_0 + ik < args.n_kv) {
            device const kd4x4_t * kr = (device const kd4x4_t *) (k + (i_kv_0 + ik)*args.nbk2 + i_stream*args.nbk3);

            deq_k(kr + i16/nl_k, i16%nl_k, tmp);
        } else {
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                tmp[j] = half4(0.0h);
            }
        }

        sk4x4[i] = tmp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // K tile of this simdgroup, transposed to [DK, NKPSG]
    simdgroup_half8x8 mk[DK8];

    FOR_UNROLL (short i = 0; i < DK8; ++i) {
        simdgroup_load(mk[i], sk + sgitg*NKPSG*DK + 8*i, DK, 0, true);
    }

    threadgroup half4   sq4[NHPTG*DK4];
    threadgroup half  * sq = (threadgroup half *) sq4;

    threadgroup float sw [NHPTG];
    threadgroup float sqk[NSG*NHPTG*NKPSG];

    const int i_batch_0 = tgpig.y*NBPTG;
    const int n_batch   = min((int) NBPTG, args.n_batch - i_batch_0);

    for (short ib = 0; ib < n_batch; ++ib) {
        const int i_batch = i_batch_0 + ib;

        device const char * pq = q + i_batch*args.nbq2 + i_stream*args.nbq3;
        device const char * pw = w + i_batch*args.nbw1 + i_stream*args.nbw3;

        float score = 0.0f;

        FOR_UNROLL (short i_head = 0; i_head < NH; i_head += NHPTG) {
            // stage the Q tile [DK, NHPTG] and the (prescaled) head weights
            for (short i = tiitg; i < NHPTG*DK4; i += NTG) {
                const short ih = i/DK4;
                const short i4 = i%DK4;

                device const float4 * q4 = (device const float4 *) (pq + (i_head + ih)*args.nbq1);

                sq4[ih*DK4 + i4] = half4(q4[i4]);
            }

            if (tiitg < NHPTG) {
                sw[tiitg] = ((device const float *) pw)[i_head + tiitg];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            FOR_UNROLL (short i = 0; i < DK8; ++i) {
                simdgroup_half8x8 mq;

                simdgroup_load(mq, sq + 8*i, DK, 0, false);
                simdgroup_multiply_accumulate(mqk, mq, mk[i], mqk);
            }

            threadgroup float * pqk = sqk + sgitg*NHPTG*NKPSG;

            simdgroup_store(mqk, pqk, NKPSG, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // one lane per key: ReLU, apply the head weight and accumulate over the head tile
            if (tiisg < NKPSG) {
                FOR_UNROLL (short ih = 0; ih < NHPTG; ++ih) {
                    score += max(pqk[ih*NKPSG + tiisg], 0.0f)*sw[ih];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiisg < NKPSG) {
            const int ik = i_kv + tiisg;
            if (ik < args.n_kv) {
                device const half  * pm = (device const half  *) (m   + i_batch*args.nbm1 + (i_stream % args.mask_ne3)*args.nbm3);
                device       float * pd = (device       float *) (dst + i_batch*args.nb1  + i_stream*args.nb3);

                pd[ik] = score + (float) pm[ik];
            }
        }
    }
}

typedef decltype(kernel_lightning_indexer<half4x4, 1, dequantize_f16>) kernel_lightning_indexer_t;

template [[host_name("kernel_lightning_indexer_f32")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<float4x4, 1, dequantize_f32>;
template [[host_name("kernel_lightning_indexer_f16")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<half4x4,  1, dequantize_f16>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_lightning_indexer_bf16")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<bfloat4x4, 1, dequantize_bf16>;
#endif

template [[host_name("kernel_lightning_indexer_q4_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_lightning_indexer_q4_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_lightning_indexer_q5_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_lightning_indexer_q5_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_lightning_indexer_q8_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q8_0, 2, dequantize_q8_0>;

// =============================================================================================
// GQA verify flash attention (GGML_METAL_FA_GQA). docs/fa-gqa-verify.md in the infernet repo.
//
// A speculative verify runs ne01 <= 8 query tokens, and GQA puts G = ne02/ne12 query heads on
// each KV head, so R = G*ne01 rows (48 for Qwen3.x 27B: 6 heads x 8 tokens) all read the SAME
// K/V. The stock kernels keep one query head per threadgroup, so the KV is streamed G times
// (tiled) or G*ne01 times (vec). Here:
//   threadgroup = (KV split iwg, KV head ikv2, seq iq3); the R rows are the flattened
//   (head, token) pairs r = h*ne01 + t, simdgroup s owns rows 8s..8s+7 as the 8 fragment rows.
//   The host launches max(ceil(R/8), 4) simdgroups; simdgroups without rows only help staging.
//   Q is held in registers as half fragments, O (8 x DV) in float fragments.
//   K/V tile of C=32 keys is staged ONCE per threadgroup in threadgroup memory (dequantized
//   there). The staging is software-pipelined through registers: the raw (still quantized)
//   V tile is loaded while Q K^T runs, the next K tile while P V runs.
//   S = Q K^T and O += P V as 8x8 MMAs; online softmax entirely in registers: in the Apple
//   8x8 layout row fm is owned by the 4 lanes that differ in lane bits 0 and 3, so a row
//   max/sum is two simd_shuffle_xor (1, 8).
//   split-K over the KV (contiguous ranges) -> partials in kernel_flash_attn_ext_vec_reduce's
//   layout, merged by that kernel.
// no ALiBi, softcap or sinks (the host gate excludes them).
// =============================================================================================

// infernet timing ablations (GGML_METAL_FA_GQA_ABL, WRONG MATH): 1 no K/V device loads (staging stores kept), 2 no staging
// stores (tiles keep stale data), 3 no in-loop barriers (racy), 4 no Q K^T MMAs, 5 no P V MMAs, 6 no softmax exp
constant int  FC_fa_gqa_abl [[function_constant(FC_FA_GQA_ABL)]];
constant bool FC_fa_gqa_abl_def = is_function_constant_defined(FC_fa_gqa_abl);
constant int  FC_fa_gqa_qks [[function_constant(FC_FA_GQA_ABL + 1)]];
constant bool FC_fa_gqa_qks_def = is_function_constant_defined(FC_fa_gqa_qks);

// raw 16-element pieces of a K/V row, loaded early and dequantized when stored to threadgroup memory
struct gqa_raw_f16 { half4x4 v; };
struct gqa_raw_q   { half d; ushort q[8]; };

void gqa_load_f16(device const char * row, short ch, thread gqa_raw_f16 & r) {
    r.v = ((device const half4x4 *) row)[ch];
}
void gqa_deq_f16(thread const gqa_raw_f16 & r, short ch, thread half4x4 & o) {
    o = r.v;
}

// q8_0: piece ch = half block ch%2 of block ch/2 (16 int8)
struct gqa_raw_q8 { half d; char4 q[4]; };

void gqa_load_q8_0(device const char * row, short ch, thread gqa_raw_q8 & r) {
    device const block_q8_0 * b = (device const block_q8_0 *) row + ch/2;
    device const packed_char4 * qs = (device const packed_char4 *) b->qs + 4*(ch%2);
    r.d = b->d;
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        r.q[i] = qs[i];
    }
}
void gqa_deq_q8_0(thread const gqa_raw_q8 & r, short ch, thread half4x4 & o) {
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        o[i] = half4(r.q[i])*r.d;
    }
}

// q8_0, default since 2026-09-23 (docs/q8-fa-verify.md in the infernet repo): the block is only 2-byte
// aligned, so the qs are read as 8 aligned ushorts packed into uints, and turned into halves with the
// int8 -> half magic number ((b ^ 0x80) | 0x6400 = 1024 + 128 + b, minus 1152; the trick of
// kernel_mul_mm_rf_q8_0_f32). Same math as gqa_deq_q8_0 (exact int8 -> half, one rounding in the *d).
// Neither half alone helps (magic + packed_char4 loads, or ushort loads + char4 -> half4 conversion);
// together ~-15% for the whole verify kernel at 64k keys (see the doc for the arms).
struct gqa_raw_q8u { half d; uint q[4]; };

void gqa_load_q8_0_u(device const char * row, short ch, thread gqa_raw_q8u & r) {
    device const ushort * p = (device const ushort *) (row + (ch/2)*34);
    r.d = as_type<half>(p[0]);
    const short o = 1 + 8*(ch%2);
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        r.q[i] = uint(p[o + 2*i]) | (uint(p[o + 2*i + 1]) << 16);
    }
}
void gqa_deq_q8_0_u(thread const gqa_raw_q8u & r, short ch, thread half4x4 & o) {
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const uint u = r.q[i] ^ 0x80808080u;
        const half2 lo = (as_type<half2>(( u       & 0x00FF00FFu) | 0x64006400u) - half2(1152.0h))*r.d; // bytes 0, 2
        const half2 hi = (as_type<half2>(((u >> 8) & 0x00FF00FFu) | 0x64006400u) - half2(1152.0h))*r.d; // bytes 1, 3
        o[i] = half4(lo[0], hi[0], lo[1], hi[1]);
    }
}

// q4_0: piece ch = low (ch%2 == 0) or high nibbles of block ch/2, element e <- byte e
void gqa_load_q4_0(device const char * row, short ch, thread gqa_raw_q & r) {
    device const block_q4_0 * b = (device const block_q4_0 *) row + ch/2;
    device const ushort * qs = (device const ushort *) b->qs;
    r.d = b->d;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        r.q[i] = qs[i];
    }
}
void gqa_deq_q4_0(thread const gqa_raw_q & r, short ch, thread half4x4 & o) {
    const short sh = (ch%2) ? 4 : 0;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        const ushort w = r.q[i] >> sh;
        o[i/2][2*(i%2) + 0] = ((half) (short) ((w     ) & 0xF) - 8.0h)*r.d;
        o[i/2][2*(i%2) + 1] = ((half) (short) ((w >> 8) & 0xF) - 8.0h)*r.d;
    }
}

template<
    typename raw_t,
    void (*load_raw)(device const char *, short, thread raw_t &),
    void (*deq_raw)(thread const raw_t &, short, thread half4x4 &),
    short DK, short DV, short C, bool PF, short PAD = 8, bool KVS = false>
kernel void kernel_flash_attn_ext_gqa(
        constant ggml_metal_kargs_flash_attn_ext_gqa & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        device const char * blk,
        threadgroup  half * shmem [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiitg [[thread_index_in_threadgroup]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    static_assert(DK == DV, "the K and V tiles share the staging loop");
    const int ABL = FC_fa_gqa_abl_def ? FC_fa_gqa_abl : 0;
    // Q K^T accumulation split (GGML_METAL_FA_GQA_QKS, 1/2/4), compile-time via the ablation constant's high digits
    constexpr short QKS_MAX = 4;
    const short QKS = FC_fa_gqa_qks_def ? (short) FC_fa_gqa_qks : 1;

    constexpr short SK  = DK + PAD;     // padded threadgroup row, in halves (PAD 0: a 32-key tile is exactly 16 KB)
    constexpr short NI  = C*(DK/16);    // 16-element pieces per tile
    constexpr short MIT = 4;            // pieces per thread: the host launches >= 128 threads
    constexpr short DV4 = DV/4;

    const int NWG  = args.nwg;
    const int iwg  = tgpig.x % NWG;
    const int itl  = tgpig.x / NWG;     // query-token tile (prefill); 0 for a verify
    const int NTOK = args.ntok;         // tokens per tile (verify: ne01)
    const int it0  = itl*NTOK;
    const int ikv2 = tgpig.y;
    const int iq3  = tgpig.z;
    const int G    = args.ne02/args.ne12;
    const int R    = G*NTOK;
    const int NT   = ntg.x*ntg.y*ntg.z;
    const int ikv3 = iq3/(args.ne03/args.ne13);

    const short qid = tiisg/4;
    const short fm  = (qid & 4) + ((tiisg/2)%4);
    const short fn  = (qid & 2)*2 + (tiisg%2)*2;

    // simdgroup has rows at all (uniform across the simdgroup)
    const bool sg_active = 8*sgitg < R;

    // this lane's fragment row -> (query head, query token)
    const int  r      = 8*sgitg + fm;
    const int  rr     = r < R ? r : R - 1;
    const bool rvalid = r < R && it0 + rr % NTOK < args.ne01;
    const int  it     = min(it0 + rr % NTOK, args.ne01 - 1);
    const int  iq2    = ikv2*G + rr/NTOK;

    k += ikv2*args.nb12 + ikv3*args.nb13;
    v += ikv2*args.nb22 + ikv3*args.nb23;

    // Q: 8 rows x DK, half fragments in registers
    simdgroup_half8x8 Qf[DK/8];
    {
        device const float * pq = (device const float *) (q + it*args.nb01 + iq2*args.nb02 + iq3*args.nb03);
        FOR_UNROLL (short kk = 0; kk < DK/8; ++kk) {
            const float2 x = rvalid ? *((device const float2 *) (pq + 8*kk + fn)) : float2(0.0f);
            Qf[kk].thread_elements()[0] = (half) x[0];
            Qf[kk].thread_elements()[1] = (half) x[1];
        }
    }

    device const half * pm = (device const half *) (mask + it*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

    simdgroup_float8x8 O[DV/8];
    FOR_UNROLL (short df = 0; df < DV/8; ++df) {
        O[df] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    float M = -FLT_MAX/2;
    float S = 0.0f;

    const int nchunk = (args.ne11 + C - 1)/C;
    const int cpw    = (nchunk + NWG - 1)/NWG;
    const int c0     = iwg*cpw;
    const int c1     = min(c0 + cpw, nchunk);

    raw_t pf[MIT];
    // KVS: separate K and V tiles (2 x 16 KB at PAD 0): one store phase and 2 barriers per chunk instead of 4,
    // the next chunk's K and V loads in flight during the whole Q K^T + softmax + P V
    raw_t pfv[KVS ? MIT : 1];
    threadgroup half * sk = shmem;
    threadgroup half * sv = KVS ? shmem + C*SK : shmem;

    // prefill: per (token tile, C-key chunk) mask class from kernel_flash_attn_ext_blk
    // 0 = all -inf (skip the chunk), 1 = mixed, 2 = all zero (skip the mask reads).
    // uniform across the threadgroup (all G heads of the tile share the mask, host requires ne32 == 1)
    device const char * pblk = blk;
    if (args.use_blk) {
        const int nblk1 = (args.ne01 + NTOK - 1)/NTOK;
        const int nblk0 = (args.ne30 + C - 1)/C;
        pblk += ((iq3%args.ne33)*nblk1 + itl)*nblk0;
    }

// issue the loads of one tile (C keys starting at key i0) into pf
#define GQA_ISSUE_TO(arr, src, nb, i0)                                                \
    FOR_UNROLL (short ii = 0; ii < MIT; ++ii) {                                       \
        const int i = tiitg + ii*NT;                                                  \
        if (i < NI) {                                                                 \
            const int key = min((i0) + i/(DK/16), args.ne11 - 1);                     \
            if (ABL != 1) load_raw((src) + (uint64_t) key*(nb), i%(DK/16), arr[ii]);  \
        }                                                                             \
    }

// dequantize pf into the threadgroup tile
#define GQA_STORE_TO(arr, base)                                                       \
    FOR_UNROLL (short ii = 0; ii < MIT; ++ii) {                                       \
        const int i = tiitg + ii*NT;                                                  \
        if (i < NI) {                                                                 \
            half4x4 tmp;                                                              \
            deq_raw(arr[ii], i%(DK/16), tmp);                                         \
            threadgroup half4 * sd = (threadgroup half4 *) ((base) + (i/(DK/16))*SK + (i%(DK/16))*16); \
            if (ABL != 2) { sd[0] = tmp[0]; sd[1] = tmp[1]; sd[2] = tmp[2]; sd[3] = tmp[3]; } \
        }                                                                             \
    }

#define GQA_ISSUE(src, nb, i0) GQA_ISSUE_TO(pf, src, nb, i0)
#define GQA_STORE() GQA_STORE_TO(pf, shmem)

    int ic = c0;
    if (args.use_blk) {
        while (ic < c1 && pblk[ic] == 0) {
            ++ic;
        }
    }

    if (PF && ic < c1) {
        GQA_ISSUE(k, args.nb11, ic*C);
        if (KVS) {
            GQA_ISSUE_TO(pfv, v, args.nb21, ic*C);
        }
    }

    while (ic < c1) {
        const int ic0 = ic*C;

        const char blk_cur = args.use_blk ? pblk[ic] : 1;

        // next chunk to process
        int nx = ic + 1;
        if (args.use_blk) {
            while (nx < c1 && pblk[nx] == 0) {
                ++nx;
            }
        }

        if (KVS) {
            // K and V tiles of this chunk, then the next chunk's loads
            if (ABL != 3) threadgroup_barrier(mem_flags::mem_threadgroup);
            GQA_STORE_TO(pf,  sk);
            GQA_STORE_TO(pfv, sv);
            if (ABL != 3) threadgroup_barrier(mem_flags::mem_threadgroup);
            if (nx < c1) {
                GQA_ISSUE_TO(pf,  k, args.nb11, nx*C);
                GQA_ISSUE_TO(pfv, v, args.nb21, nx*C);
            }
        } else {
        // K tile
        if (ABL != 3) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!PF) {
            GQA_ISSUE(k, args.nb11, ic0);
        }
        GQA_STORE();
        if (ABL != 3) threadgroup_barrier(mem_flags::mem_threadgroup);

        // PF: V tile loads in flight during Q K^T
        if (PF) {
            GQA_ISSUE(v, args.nb21, ic0);
        }
        }

        simdgroup_half8x8 Pf[C/8];

        if (sg_active) {
            // S = Q K^T  (8 rows x C keys)
            simdgroup_float8x8 Sf[C/8];
            FOR_UNROLL (short kf = 0; kf < C/8; ++kf) {
                Sf[kf] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            if (QKS > 1) {
                // QKS independent partial sums per key fragment: the plain loop is QKS x fewer, QKS x longer MMA dependency
                // chains (4 accumulators x 32 MMAs each), which leaves the MMA pipe waiting on its own results
                simdgroup_float8x8 Sp[QKS_MAX - 1][C/8];
                FOR_UNROLL (short u = 0; u < QKS - 1; ++u) {
                    FOR_UNROLL (short kf = 0; kf < C/8; ++kf) {
                        Sp[u][kf] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                    }
                }
                FOR_UNROLL (short kk = 0; kk < DK/8; kk += QKS) {
                    FOR_UNROLL (short u = 0; u < QKS; ++u) {
                        FOR_UNROLL (short kf = 0; kf < C/8; ++kf) {
                            simdgroup_half8x8 Kb;
                            simdgroup_load(Kb, sk + 8*kf*SK + 8*(kk + u), SK, 0, true);
                            if (u == 0) {
                                simdgroup_multiply_accumulate(Sf[kf], Qf[kk + u], Kb, Sf[kf]);
                            } else {
                                simdgroup_multiply_accumulate(Sp[u - 1][kf], Qf[kk + u], Kb, Sp[u - 1][kf]);
                            }
                        }
                    }
                }
                FOR_UNROLL (short u = 0; u < QKS - 1; ++u) {
                    FOR_UNROLL (short kf = 0; kf < C/8; ++kf) {
                        Sf[kf].thread_elements()[0] += Sp[u][kf].thread_elements()[0];
                        Sf[kf].thread_elements()[1] += Sp[u][kf].thread_elements()[1];
                    }
                }
            } else {
            FOR_UNROLL (short kk = 0; kk < DK/8; ++kk) {
                FOR_UNROLL (short kf = 0; kf < C/8; ++kf) {
                    simdgroup_half8x8 Kb;
                    if (ABL == 7) simdgroup_load(Kb, sk + 8*kf*SK + 8*kk, SK, 0, false); else
                    simdgroup_load(Kb, sk + 8*kf*SK + 8*kk, SK, 0, true);
                    if (ABL != 4) simdgroup_multiply_accumulate(Sf[kf], Qf[kk], Kb, Sf[kf]);
                }
            }
            }

            // online softmax, in registers
            float2 sv[C/8];
            float mloc = -FLT_MAX/2;
            FOR_UNROLL (short kf = 0; kf < C/8; ++kf) {
                const int key = ic0 + 8*kf + fn;
                float2 s = float2(Sf[kf].thread_elements()[0], Sf[kf].thread_elements()[1])*args.scale;
                if (args.has_mask && blk_cur == 2) {
                    s[0] = key + 0 < args.ne11 ? s[0] : -INFINITY;
                    s[1] = key + 1 < args.ne11 ? s[1] : -INFINITY;
                } else if (args.has_mask) {
                    s[0] += key + 0 < args.ne11 ? (float) pm[key + 0] : -INFINITY;
                    s[1] += key + 1 < args.ne11 ? (float) pm[key + 1] : -INFINITY;
                } else {
                    s[0] = key + 0 < args.ne11 ? s[0] : -INFINITY;
                    s[1] = key + 1 < args.ne11 ? s[1] : -INFINITY;
                }
                sv[kf] = s;
                mloc = max(mloc, max(s[0], s[1]));
            }
            mloc = max(mloc, simd_shuffle_xor(mloc, 1));
            mloc = max(mloc, simd_shuffle_xor(mloc, 8));

            const float Mn = max(M, mloc);
            const float ms = exp(M - Mn);

            float sl = 0.0f;
            FOR_UNROLL (short kf = 0; kf < C/8; ++kf) {
                const float p0 = ABL == 6 ? sv[kf][0] - Mn : exp(sv[kf][0] - Mn);
                const float p1 = ABL == 6 ? sv[kf][1] - Mn : exp(sv[kf][1] - Mn);
                sl += p0 + p1;
                Pf[kf].thread_elements()[0] = (half) p0;
                Pf[kf].thread_elements()[1] = (half) p1;
            }
            sl += simd_shuffle_xor(sl, 1);
            sl += simd_shuffle_xor(sl, 8);

            S = S*ms + sl;
            M = Mn;

            FOR_UNROLL (short df = 0; df < DV/8; ++df) {
                O[df].thread_elements()[0] *= ms;
                O[df].thread_elements()[1] *= ms;
            }
        }

        if (!KVS) {
        // V tile
        if (ABL != 3) threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!PF) {
            GQA_ISSUE(v, args.nb21, ic0);
        }
        GQA_STORE();
        if (ABL != 3) threadgroup_barrier(mem_flags::mem_threadgroup);

        // PF: next K tile loads in flight during P V
        if (PF && nx < c1) {
            GQA_ISSUE(k, args.nb11, nx*C);
        }
        }

        if (sg_active) {
            // O += P V  (8 rows x DV)
            FOR_UNROLL (short df = 0; df < DV/8; ++df) {
                FOR_UNROLL (short kf = 0; kf < C/8; ++kf) {
                    simdgroup_half8x8 Vb;
                    simdgroup_load(Vb, sv + 8*kf*SK + 8*df, SK, 0, false);
                    if (ABL != 5) simdgroup_multiply_accumulate(O[df], Pf[kf], Vb, O[df]);
                }
            }
        }

        ic = nx;
    }

#undef GQA_ISSUE
#undef GQA_STORE
#undef GQA_ISSUE_TO
#undef GQA_STORE_TO

    if (!rvalid) {
        return;
    }

    // dst is [DV, ne1 = heads, ne2 = tokens, ne3]
    const uint64_t rid = (uint64_t) iq3*args.ne2*args.ne1 + (uint64_t) it*args.ne1 + iq2;

    // partial slot stride: nwg_out > nwg leaves slots [nwg, nwg_out) to another producer (the CPU's SME co-attention)
    const int NWGO = args.nwg_out > 0 ? args.nwg_out : NWG;

    if (NWGO == 1) {
        const float sc = S == 0.0f ? 0.0f : 1.0f/S;
        device float * d = (device float *) dst + rid*DV;
        FOR_UNROLL (short df = 0; df < DV/8; ++df) {
            *((device float2 *) (d + 8*df + fn)) = float2(O[df].thread_elements()[0], O[df].thread_elements()[1])*sc;
        }
    } else {
        // partials: dst4[rid*DV4*NWG + NWG*i + iwg], then S/M at nrows*DV*NWG + rid*2*NWG
        const uint64_t nrows = (uint64_t) args.ne3*args.ne2*args.ne1;
        device float * d = (device float *) dst;
        FOR_UNROLL (short df = 0; df < DV/8; ++df) {
            const short e = 8*df + fn; // e%4 is 0 or 2, so the pair stays in one float4
            *((device float2 *) (d + (rid*DV4*NWGO + (uint64_t) NWGO*(e/4) + iwg)*4 + (e%4))) =
                float2(O[df].thread_elements()[0], O[df].thread_elements()[1]);
        }
        if ((tiisg & 9) == 0) {
            device float * d1 = d + nrows*DV*NWGO;
            d1[rid*(2*NWGO) + 2*iwg + 0] = S;
            d1[rid*(2*NWGO) + 2*iwg + 1] = M;
        }
    }
}

typedef decltype(kernel_flash_attn_ext_gqa<gqa_raw_f16, gqa_load_f16, gqa_deq_f16, 256, 256, 32, true>) flash_attn_ext_gqa_t;

template [[host_name("kernel_flash_attn_ext_gqa_f16_dk256_dv256_c32_pf0")]]  kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_f16, gqa_load_f16,  gqa_deq_f16,  256, 256, 32, false>;
template [[host_name("kernel_flash_attn_ext_gqa_f16_dk256_dv256_c32_pf1")]]  kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_f16, gqa_load_f16,  gqa_deq_f16,  256, 256, 32, true>;
template [[host_name("kernel_flash_attn_ext_gqa_q8_0_dk256_dv256_c32_pf0")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_q8u, gqa_load_q8_0_u, gqa_deq_q8_0_u, 256, 256, 32, false>;
template [[host_name("kernel_flash_attn_ext_gqa_q8_0_dk256_dv256_c32_pf1")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_q8u, gqa_load_q8_0_u, gqa_deq_q8_0_u, 256, 256, 32, true>;
template [[host_name("kernel_flash_attn_ext_gqa_q4_0_dk256_dv256_c32_pf0")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_q,   gqa_load_q4_0, gqa_deq_q4_0, 256, 256, 32, false>;
template [[host_name("kernel_flash_attn_ext_gqa_q4_0_dk256_dv256_c32_pf1")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_q,   gqa_load_q4_0, gqa_deq_q4_0, 256, 256, 32, true>;
template [[host_name("kernel_flash_attn_ext_gqa_f16_dk256_dv256_c32_pf1_np")]]  kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_f16, gqa_load_f16,  gqa_deq_f16,  256, 256, 32, true, 0>;
template [[host_name("kernel_flash_attn_ext_gqa_q8_0_dk256_dv256_c32_pf1_np")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_q8u, gqa_load_q8_0_u, gqa_deq_q8_0_u, 256, 256, 32, true, 0>;
template [[host_name("kernel_flash_attn_ext_gqa_q4_0_dk256_dv256_c32_pf1_np")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_q,   gqa_load_q4_0, gqa_deq_q4_0, 256, 256, 32, true, 0>;
template [[host_name("kernel_flash_attn_ext_gqa_f16_dk256_dv256_c32_pf1_kvs")]]  kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_f16, gqa_load_f16,  gqa_deq_f16,  256, 256, 32, true, 0, true>;
template [[host_name("kernel_flash_attn_ext_gqa_q8_0_dk256_dv256_c32_pf1_kvs")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_q8u, gqa_load_q8_0_u, gqa_deq_q8_0_u, 256, 256, 32, true, 0, true>;
template [[host_name("kernel_flash_attn_ext_gqa_q4_0_dk256_dv256_c32_pf1_kvs")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_q,   gqa_load_q4_0, gqa_deq_q4_0, 256, 256, 32, true, 0, true>;
// the previous q8_0 staging (packed_char4 loads + char4 -> half4), for A/B: GGML_METAL_FA_GQA_Q8_OLD=1
template [[host_name("kernel_flash_attn_ext_gqa_q8_0_dk256_dv256_c32_pf1_old")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<gqa_raw_q8,  gqa_load_q8_0, gqa_deq_q8_0, 256, 256, 32, true>;

// =============================================================================================
// GQA verify flash attention v2 (GGML_METAL_FA_GQA=2): barrier-free, register-fed.
// docs/fa-gqa-verify.md ("v2") in the infernet repo.
//
// Same grid and row packing as v1 (threadgroup = (KV split, KV head, seq), simdgroup s owns the
// 8 rows 8s..8s+7 of the flattened (head, token) rows), but there is no threadgroup memory and
// no barrier: every simdgroup streams K and V from device memory straight into its own
// registers (the simdgroups of a threadgroup read the same bytes, so they share them via L1).
// Everything is transposed so the streamed operand is the MMA A operand:
//   S^T (keys x rows) = K Q^T      A = K fragment,   B = Q^T (held in registers)
//   O^T (dv x rows)  += V^T P^T    A = V^T fragment, B = P^T = the S^T accumulator's own layout
// Apple 8x8 layout: lane holds (fm, fn), (fm, fn+1), fm = 4*b4 + 2*b2 + b1, fn = 4*b3 + 2*b0.
// dim permutations (applied to both operands, undone at the store):
//   K: fragment kk = 4g + p, column c -> dim 32g + 8(c/2) + 2p + c%2, so a lane's A elements
//      across p = 0..3 are the 8 contiguous dims 32g + 8j .. +7 (j = fn/2) of key fm:
//      one 16-byte load (f16) or 8 bytes + scale (q8_0).
//   V/O: fragment df = 8G + p, row m -> dim 64G + 8m + p, so a lane's V^T elements across
//      p = 0..7 are the 8 contiguous dims 64G + 8fm .. +7 of keys fn and fn+1.
// Softmax is per column of S^T (a row of S): the 8 lanes that differ in bits 1, 2, 4 hold one
// column, so the max is 3 simd_shuffle_xor (2, 4, 16); the sum stays per lane until the end.
// C = 8*NB keys per step (NB = 1 was the first version).
// =============================================================================================

struct gqa2_raw_f16 { uint4 u; };
struct gqa2_raw_q8  { half d; char4 a; char4 b; };

// 8 contiguous elements starting at element off (off % 8 == 0) of a row
inline void gqa2_load_f16(device const char * row, int off, thread gqa2_raw_f16 & r) {
    r.u = *((device const uint4 *) ((device const half *) row + off));
}
inline void gqa2_deq_f16(thread const gqa2_raw_f16 & r, thread half2 * o) {
    o[0] = as_type<half2>(r.u.x);
    o[1] = as_type<half2>(r.u.y);
    o[2] = as_type<half2>(r.u.z);
    o[3] = as_type<half2>(r.u.w);
}

inline void gqa2_load_q8_0(device const char * row, int off, thread gqa2_raw_q8 & r) {
    device const block_q8_0 * b = (device const block_q8_0 *) row + off/32;
    device const packed_char4 * qs = (device const packed_char4 *) (b->qs + off%32);
    r.d = b->d;
    r.a = qs[0];
    r.b = qs[1];
}
inline void gqa2_deq_q8_0(thread const gqa2_raw_q8 & r, thread half2 * o) {
    const half4 a = half4(r.a)*r.d;
    const half4 b = half4(r.b)*r.d;
    o[0] = a.xy;
    o[1] = a.zw;
    o[2] = b.xy;
    o[3] = b.zw;
}

// Tuning variants (template args, host env in ggml-metal-device.cpp / ggml-metal-ops.cpp):
//   PF  0: K for the next step is prefetched across steps, V issued after Q K^T
//       1: as 0, but V issued before Q K^T
//       2: no cross-step prefetch (K issued at the top of the step)
//   NB  key blocks of 8 per step (C = 8*NB): NB independent S^T accumulators, one softmax update
//   NA  independent accumulators per S^T block along the 256-deep dk chain (summed after)
//   SG1 1 = one simdgroup per threadgroup (rows split across threadgroups, no L1 sharing needed)
template<
    typename raw_t,
    void (*load_raw)(device const char *, int, thread raw_t &),
    void (*deq_raw)(thread const raw_t &, thread half2 *),
    short DK, short DV, short PF, short NB, short NA, short SG1>
kernel void kernel_flash_attn_ext_gqa2(
        constant ggml_metal_kargs_flash_attn_ext_gqa & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort  tiisg [[thread_index_in_simdgroup]],
        ushort  sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short C   = 8*NB;
    constexpr short NGK = DK/32;  // K pieces per lane per block
    constexpr short NGV = DV/64;  // V dim groups (2 pieces each: keys fn, fn+1)
    constexpr short DV4 = DV/4;
    static_assert(32 % C == 0, "split-K chunks are multiples of 32 keys");

    const int NWG  = args.nwg;
    const int G    = args.ne02/args.ne12;
    const int R    = G*args.ne01;
    const int NSG  = (R + 7)/8;
    const int iwg  = SG1 ? (int) tgpig.x / NSG : (int) tgpig.x;
    const int isg  = SG1 ? (int) tgpig.x % NSG : (int) sgitg;
    const int ikv2 = tgpig.y;
    const int iq3  = tgpig.z;
    const int ikv3 = iq3/(args.ne03/args.ne13);

    const short qid = tiisg/4;
    const short fm  = (qid & 4) + ((tiisg/2)%4);
    const short fn  = (qid & 2)*2 + (tiisg%2)*2;
    const short j   = fn/2;

    // this lane's two columns = rows n = fn + e of the simdgroup
    bool rval[2];
    int  rit[2];
    int  riq2[2];
    device const half * pm[2];
    FOR_UNROLL (short e = 0; e < 2; ++e) {
        const int r  = 8*isg + fn + e;
        rval[e]      = r < R;
        const int rr = rval[e] ? r : R - 1;
        rit[e]  = rr % args.ne01;
        riq2[e] = ikv2*G + rr/args.ne01;
        pm[e]   = (device const half *) (mask + rit[e]*args.nb31 + (riq2[e]%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    // Q^T: fragment kk = 4g + p, element (fm, fn + e) = Q[row fn + e][32g + 8(fm/2) + 2p + fm%2]
    simdgroup_half8x8 QT[DK/8];
    FOR_UNROLL (short e = 0; e < 2; ++e) {
        device const float * pq = (device const float *) (q + rit[e]*args.nb01 + riq2[e]*args.nb02 + iq3*args.nb03);
        FOR_UNROLL (short g = 0; g < NGK; ++g) {
            FOR_UNROLL (short p = 0; p < 4; ++p) {
                const float x = rval[e] ? pq[32*g + 8*(fm/2) + 2*p + fm%2] : 0.0f;
                QT[4*g + p].thread_elements()[e] = (half) x;
            }
        }
    }

    simdgroup_float8x8 O[DV/8];
    FOR_UNROLL (short df = 0; df < DV/8; ++df) {
        O[df] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    float2 M = float2(-FLT_MAX/2);
    float2 S = float2(0.0f);

    k += ikv2*args.nb12 + ikv3*args.nb13;
    v += ikv2*args.nb22 + ikv3*args.nb23;

    // split-K in units of 32 keys (the host sizes nwg on those), stepped by C
    const int n32  = (args.ne11 + 31)/32;
    const int cpw  = (n32 + NWG - 1)/NWG;
    const int kb0  = iwg*cpw*32;
    const int kb1  = min((iwg + 1)*cpw*32, args.ne11);

    raw_t kr[NB][NGK];
    raw_t vr[NB][NGV][2];

#define GQA2_ISSUE_K(i0)                                                                  \
    FOR_UNROLL (short b = 0; b < NB; ++b) {                                               \
        const int key = min((i0) + 8*b + fm, args.ne11 - 1);                              \
        device const char * row = k + (uint64_t) key*args.nb11;                           \
        FOR_UNROLL (short g = 0; g < NGK; ++g) {                                          \
            load_raw(row, 32*g + 8*j, kr[b][g]);                                          \
        }                                                                                 \
    }

#define GQA2_ISSUE_V(i0)                                                                  \
    FOR_UNROLL (short b = 0; b < NB; ++b) {                                               \
    FOR_UNROLL (short e = 0; e < 2; ++e) {                                                \
        const int key = min((i0) + 8*b + fn + e, args.ne11 - 1);                          \
        device const char * row = v + (uint64_t) key*args.nb21;                           \
        FOR_UNROLL (short gv = 0; gv < NGV; ++gv) {                                       \
            load_raw(row, 64*gv + 8*fm, vr[b][gv][e]);                                    \
        }                                                                                 \
    }                                                                                     \
    }

    if (PF != 2 && kb0 < kb1) {
        GQA2_ISSUE_K(kb0);
    }

    for (int i0 = kb0; i0 < kb1; i0 += C) {
        if (PF == 2) {
            GQA2_ISSUE_K(i0);
        }
        if (PF == 1) {
            GQA2_ISSUE_V(i0);
        }

        // S^T = K Q^T (NB blocks of 8 keys x 8 rows), NA independent chains per block
        float2 s[NB];
        FOR_UNROLL (short b = 0; b < NB; ++b) {
            simdgroup_float8x8 Sf[NA];
            FOR_UNROLL (short a = 0; a < NA; ++a) {
                Sf[a] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
            FOR_UNROLL (short g = 0; g < NGK; ++g) {
                half2 x[4];
                deq_raw(kr[b][g], x);
                FOR_UNROLL (short p = 0; p < 4; ++p) {
                    simdgroup_half8x8 Kf;
                    Kf.thread_elements()[0] = x[p][0];
                    Kf.thread_elements()[1] = x[p][1];
                    simdgroup_multiply_accumulate(Sf[(4*g + p)%NA], Kf, QT[4*g + p], Sf[(4*g + p)%NA]);
                }
            }
            s[b] = float2(Sf[0].thread_elements()[0], Sf[0].thread_elements()[1]);
            FOR_UNROLL (short a = 1; a < NA; ++a) {
                s[b] += float2(Sf[a].thread_elements()[0], Sf[a].thread_elements()[1]);
            }
        }

        if (PF != 1) {
            GQA2_ISSUE_V(i0);
        }

        // online softmax per column (lane rows fn, fn + 1; keys i0 + 8b + fm)
        float2 mx = float2(-INFINITY);
        FOR_UNROLL (short b = 0; b < NB; ++b) {
            const int key = i0 + 8*b + fm;
            s[b] *= args.scale;
            if (args.has_mask) {
                s[b][0] += key < args.ne11 ? (float) pm[0][key] : -INFINITY;
                s[b][1] += key < args.ne11 ? (float) pm[1][key] : -INFINITY;
            } else {
                s[b] = key < args.ne11 ? s[b] : float2(-INFINITY);
            }
            mx = max(mx, s[b]);
        }

        mx = max(mx, simd_shuffle_xor(mx, 2));
        mx = max(mx, simd_shuffle_xor(mx, 4));
        mx = max(mx, simd_shuffle_xor(mx, 16));

        const float2 Mn = max(M, mx);

        if (simd_any(Mn[0] > M[0] || Mn[1] > M[1])) {
            const float2 ms = exp(M - Mn);
            S *= ms;
            FOR_UNROLL (short df = 0; df < DV/8; ++df) {
                O[df].thread_elements()[0] *= ms[0];
                O[df].thread_elements()[1] *= ms[1];
            }
            M = Mn;
        }

        simdgroup_half8x8 PT[NB];
        FOR_UNROLL (short b = 0; b < NB; ++b) {
            const float2 pp = exp(s[b] - Mn);
            PT[b].thread_elements()[0] = (half) pp[0];
            PT[b].thread_elements()[1] = (half) pp[1];
            S += pp;
        }

        if (PF != 2 && i0 + C < kb1) {
            GQA2_ISSUE_K(i0 + C);
        }

        // O^T += V^T P^T
        FOR_UNROLL (short b = 0; b < NB; ++b) {
            FOR_UNROLL (short gv = 0; gv < NGV; ++gv) {
                half2 x0[4];
                half2 x1[4];
                deq_raw(vr[b][gv][0], x0);
                deq_raw(vr[b][gv][1], x1);
                FOR_UNROLL (short p = 0; p < 8; ++p) {
                    simdgroup_half8x8 Vf;
                    Vf.thread_elements()[0] = x0[p/2][p%2];
                    Vf.thread_elements()[1] = x1[p/2][p%2];
                    simdgroup_multiply_accumulate(O[8*gv + p], Vf, PT[b], O[8*gv + p]);
                }
            }
        }
    }

#undef GQA2_ISSUE_K
#undef GQA2_ISSUE_V

    // column sums: the 8 lanes that differ in bits 1, 2, 4
    S += simd_shuffle_xor(S, 2);
    S += simd_shuffle_xor(S, 4);
    S += simd_shuffle_xor(S, 16);

    const uint64_t nrows = (uint64_t) args.ne3*args.ne2*args.ne1;

    FOR_UNROLL (short e = 0; e < 2; ++e) {
        if (!rval[e]) {
            continue;
        }
        // dst is [DV, ne1 = heads, ne2 = tokens, ne3]
        const uint64_t rid = (uint64_t) iq3*args.ne2*args.ne1 + (uint64_t) rit[e]*args.ne1 + riq2[e];

        if (NWG == 1) {
            const float sc = S[e] == 0.0f ? 0.0f : 1.0f/S[e];
            device float4 * d = (device float4 *) ((device float *) dst + rid*DV);
            FOR_UNROLL (short gv = 0; gv < NGV; ++gv) {
                FOR_UNROLL (short h = 0; h < 2; ++h) {
                    d[16*gv + 2*fm + h] = float4(
                        O[8*gv + 4*h + 0].thread_elements()[e],
                        O[8*gv + 4*h + 1].thread_elements()[e],
                        O[8*gv + 4*h + 2].thread_elements()[e],
                        O[8*gv + 4*h + 3].thread_elements()[e])*sc;
                }
            }
        } else {
            // partials: dst4[rid*DV4*NWG + NWG*i + iwg], then S/M at nrows*DV*NWG + rid*2*NWG
            device float4 * d = (device float4 *) dst;
            FOR_UNROLL (short gv = 0; gv < NGV; ++gv) {
                FOR_UNROLL (short h = 0; h < 2; ++h) {
                    d[rid*DV4*NWG + (uint64_t) NWG*(16*gv + 2*fm + h) + iwg] = float4(
                        O[8*gv + 4*h + 0].thread_elements()[e],
                        O[8*gv + 4*h + 1].thread_elements()[e],
                        O[8*gv + 4*h + 2].thread_elements()[e],
                        O[8*gv + 4*h + 3].thread_elements()[e]);
                }
            }
            if (fm == 0) {
                device float * d1 = (device float *) dst + nrows*DV*NWG;
                d1[rid*(2*NWG) + 2*iwg + 0] = S[e];
                d1[rid*(2*NWG) + 2*iwg + 1] = M[e];
            }
        }
    }
}

typedef decltype(kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 1, 1, 0>) flash_attn_ext_gqa2_t;

template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb1_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 1, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf1_nb1_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 1, 1, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb1_na2_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 1, 2, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb1_na4_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 1, 4, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb2_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 2, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb2_na2_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 2, 2, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb4_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 4, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf2_nb4_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 2, 4, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf2_nb4_na2_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 2, 4, 2, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb1_na1_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 1, 1, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb2_na2_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 2, 2, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb4_na1_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 4, 1, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf2_nb4_na1_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 2, 4, 1, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb1_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 1, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf1_nb1_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 1, 1, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb1_na2_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 1, 2, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb1_na4_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 1, 4, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb2_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 2, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb2_na2_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 2, 2, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb4_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 4, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf2_nb4_na1_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 2, 4, 1, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf2_nb4_na2_sg0")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 2, 4, 2, 0>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb1_na1_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 1, 1, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb2_na2_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 2, 2, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb4_na1_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 4, 1, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf2_nb4_na1_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 2, 4, 1, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb1_na4_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 1, 4, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf0_nb1_na2_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 0, 1, 2, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf2_nb1_na4_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 2, 1, 4, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_f16_dk256_dv256_pf2_nb1_na1_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_f16, gqa2_load_f16, gqa2_deq_f16, 256, 256, 2, 1, 1, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb1_na4_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 1, 4, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf0_nb1_na2_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 0, 1, 2, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf2_nb1_na4_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 2, 1, 4, 1>;
template [[host_name("kernel_flash_attn_ext_gqa2_q8_0_dk256_dv256_pf2_nb1_na1_sg1")]] kernel flash_attn_ext_gqa2_t kernel_flash_attn_ext_gqa2<gqa2_raw_q8, gqa2_load_q8_0, gqa2_deq_q8_0, 256, 256, 2, 1, 1, 1>;

#ifdef GGML_METAL_HAS_TENSOR
// ---------------------------------------------------------------------------------------------------------------------
// Prefill flash attention on the GPU's matrix units (Metal 4 tensor ops; the A19's neural accelerators).
// GGML_METAL_FA_PREFILL_NA=1. One threadgroup = 64 query tokens of one query head; K and V are f16 in device memory
// (a quantized cache is dequantized to f16 scratch first) and fed to matmul2d directly, like pa_attn_na in
// phone-attn/pa-metal.mm. Per 64-key block: S = Q K^T (matmul2d) -> threadgroup, online softmax (2 threads per row),
// O = O*alpha + P V (matmul2d, O stays in a cooperative tensor). Q is f16 from kernel_flash_attn_ext_pna_q16.
// ---------------------------------------------------------------------------------------------------------------------
#define PNA_D   256

// Q f32 [D][ne01][ne02] (strided) -> f16 [ne02][ne01][D], one threadgroup per (token, head), 64 threads x 4 elements
kernel void kernel_flash_attn_ext_pna_q16(
        constant ggml_metal_kargs_flash_attn_ext_pna & args,
        device const char * q,
        device       half * q16,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    const int t = tgpig.x, h = tgpig.y;
    device const float4 * src = (device const float4 *) (q + t*args.nb01 + h*args.nb02);
    device       half4  * dst = (device half4 *) (q16 + ((uint64_t) h*args.ne01 + t)*PNA_D);
    dst[tiitg] = (half4) src[tiitg];
}

// TM query tokens per threadgroup, NB keys per block, NSG simdgroups, RLX: matmul2d relaxed precision.
// Threadgroup memory: TM*NB*6 + TM*4 bytes. The host picks the instance (GGML_METAL_FA_PNA=TMxNBxNSG[r]).
template<short PNA_TM, short PNA_NB, short PNA_NSG, bool RLX>
kernel void kernel_flash_attn_ext_pna(
        constant ggml_metal_kargs_flash_attn_ext_pna & args,
        device const half  * q16,
        device const half  * k,
        device const half  * v,
        device const half  * mask,
        device       float * dst,
        device const char  * blk,
        threadgroup  char  * shm [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    using namespace mpp::tensor_ops;

    const int itl = tgpig.x, h = tgpig.y;
    const int t0  = itl*PNA_TM;
    const int nr  = min((int) PNA_TM, args.ne01 - t0);   // valid rows in this tile
    const int hk  = h/(args.ne02/args.ne12);

    threadgroup float * S  = (threadgroup float *) shm;              // [TM][NB]
    threadgroup half  * P  = (threadgroup half *) (S + PNA_TM*PNA_NB); // [TM][NB]
    threadgroup float * AL = (threadgroup float *) (P + PNA_TM*PNA_NB); // [TM]

    constexpr auto dS = matmul2d_descriptor(PNA_TM, PNA_NB, PNA_D, false, true,  RLX, matmul2d_descriptor::mode::multiply);
    constexpr auto dO = matmul2d_descriptor(PNA_TM, PNA_D, PNA_NB, false, false, RLX, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS, execution_simdgroups<PNA_NSG>> mmS;
    matmul2d<dO, execution_simdgroups<PNA_NSG>> mmO;

    // rows past the last token read as zero (extent nr) and are not stored
    auto tQ = tensor((device half *) (q16 + ((uint64_t) h*args.ne01 + t0)*PNA_D), dextents<int32_t, 2>(PNA_D, nr), array<int, 2>({1, PNA_D}));
    auto tS = tensor(S, dextents<int32_t, 2>(PNA_NB, PNA_TM), array<int, 2>({1, PNA_NB}));
    auto tP = tensor(P, dextents<int32_t, 2>(PNA_NB, PNA_TM), array<int, 2>({1, PNA_NB}));

    device const half * kh = k + (uint64_t) hk*args.nhk;
    device const half * vh = v + (uint64_t) hk*args.nhv;

    auto tK0 = tensor((device half *) kh, dextents<int32_t, 2>(PNA_D, PNA_NB), array<int, 2>({1, (int) args.nsk}));
    auto tV0 = tensor((device half *) vh, dextents<int32_t, 2>(PNA_D, PNA_NB), array<int, 2>({1, (int) args.nsv}));

    auto oT = mmO.template get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    FOR_UNROLL (uint16_t i = 0; i < oT.get_capacity(); ++i) {
        if (oT.is_valid_element(i)) oT[i] = 0.0f;
    }

    constexpr int TPR = PNA_NSG*32/PNA_TM;   // threads per row
    constexpr int KPT = PNA_NB/TPR;          // keys per thread
    const int row = tid/TPR, hf = tid%TPR;
    const int tok = t0 + min(row, nr - 1);

    device const half * pm = mask + (uint64_t) tok*args.nm1;

    const int nblk0 = (args.ne30 + PNA_NB - 1)/PNA_NB;
    device const char * pblk = blk + (uint64_t) itl*nblk0;

    // softmax in base 2: scores pre-multiplied by log2(e) (with the scale folded in), exp2 instead of exp
    const float scale2 = args.scale*M_LOG2E_F;
    float m_row = -FLT_MAX/2, l_row = 0.0f;

    const int nb = (args.ne11 + PNA_NB - 1)/PNA_NB;
    for (int ib = 0; ib < nb; ++ib) {
        const char cls = args.use_blk ? pblk[ib] : (args.has_mask ? 1 : 2);
        if (cls == 0) {
            continue;   // every key of the block is masked for every row of the tile (uniform across the threadgroup)
        }
        const int j0 = ib*PNA_NB;
        const int nk = min((int) PNA_NB, args.ne11 - j0);

        auto tK = tensor((device half *) (kh + (uint64_t) j0*args.nsk), dextents<int32_t, 2>(PNA_D, nk), array<int, 2>({1, (int) args.nsk}));
        auto sT = mmS.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
        mmS.run(tQ, tK, sT);
        sT.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[KPT];
        float mx = -FLT_MAX/2;
        FOR_UNROLL (int i = 0; i < KPT; ++i) {
            const int c = hf*KPT + i;
            float x = S[row*PNA_NB + c]*scale2;
            if (c >= nk) {
                x = -INFINITY;
            } else if (cls == 1) {
                x += (float) pm[j0 + c]*M_LOG2E_F;
            }
            s[i] = x;
            mx = max(mx, x);
        }
        FOR_UNROLL (int o = 1; o < TPR; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, o));
        const float m_new = max(m_row, mx);
        const float alpha = exp2(m_row - m_new);
        float sum = 0.0f;
        FOR_UNROLL (int i = 0; i < KPT; ++i) {
            const float p = exp2(s[i] - m_new);
            sum += p;
            P[row*PNA_NB + hf*KPT + i] = (half) p;
        }
        FOR_UNROLL (int o = 1; o < TPR; o <<= 1) sum += simd_shuffle_xor(sum, o);
        l_row = l_row*alpha + sum;
        m_row = m_new;
        if (hf == 0) AL[row] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (uint16_t i = 0; i < oT.get_capacity(); ++i) {
            if (oT.is_valid_element(i)) { auto ids = oT.get_multidimensional_index(i); oT[i] *= AL[ids[1]]; }
        }
        auto tV = tensor((device half *) (vh + (uint64_t) j0*args.nsv), dextents<int32_t, 2>(PNA_D, nk), array<int, 2>({1, (int) args.nsv}));
        mmO.run(tP, tV, oT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (hf == 0) AL[row] = l_row == 0.0f ? 0.0f : 1.0f/l_row;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (uint16_t i = 0; i < oT.get_capacity(); ++i) {
        if (oT.is_valid_element(i)) { auto ids = oT.get_multidimensional_index(i); oT[i] *= AL[ids[1]]; }
    }

    // dst [DV][n_head][n_tok] f32: this head's rows are nd2 floats apart; the store clips to the nr valid rows
    auto tO = tensor(dst + (uint64_t) t0*args.nd2 + (uint64_t) h*PNA_D, dextents<int32_t, 2>(PNA_D, nr), array<int, 2>({1, (int) args.nd2}));
    oT.store(tO);
}

typedef decltype(kernel_flash_attn_ext_pna<64, 64, 4, false>) kernel_flash_attn_ext_pna_t;

template [[host_name("kernel_flash_attn_ext_pna_64x64x4")]]   kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<64, 64, 4, false>;
template [[host_name("kernel_flash_attn_ext_pna_64x64x8")]]   kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<64, 64, 8, false>;
template [[host_name("kernel_flash_attn_ext_pna_64x32x4")]]   kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<64, 32, 4, false>;
template [[host_name("kernel_flash_attn_ext_pna_128x32x4")]]  kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<128, 32, 4, false>;
template [[host_name("kernel_flash_attn_ext_pna_128x32x8")]]  kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<128, 32, 8, false>;
template [[host_name("kernel_flash_attn_ext_pna_32x64x4")]]   kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<32, 64, 4, false>;
template [[host_name("kernel_flash_attn_ext_pna_64x64x4r")]]  kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<64, 64, 4, true>;
template [[host_name("kernel_flash_attn_ext_pna_128x32x4r")]] kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<128, 32, 4, true>;
template [[host_name("kernel_flash_attn_ext_pna_64x64x16")]]  kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<64, 64, 16, false>;
template [[host_name("kernel_flash_attn_ext_pna_32x64x8")]]   kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<32, 64, 8, false>;
template [[host_name("kernel_flash_attn_ext_pna_16x64x4")]]   kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<16, 64, 4, false>;
template [[host_name("kernel_flash_attn_ext_pna_32x128x8")]]  kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<32, 128, 8, false>;
template [[host_name("kernel_flash_attn_ext_pna_32x128x4")]]  kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<32, 128, 4, false>;
template [[host_name("kernel_flash_attn_ext_pna_64x64x8r")]]  kernel kernel_flash_attn_ext_pna_t kernel_flash_attn_ext_pna<64, 64, 8, true>;
#endif // GGML_METAL_HAS_TENSOR
