#include "common.h"
#include "dequantize.h"

// simdgroup-GEMM tile geometry (shared by the kernel and its shared-memory sizing)
#define NR0_MM 64
#define NK_MM  32

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12  [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13  [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2    [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3    [[function_constant(FC_MUL_MM + 5)]];

// each block_q contains 16*nl weights
#ifdef GGML_METAL_HAS_TENSOR
template<
    typename SA, typename SA_4x4, typename SA_8x8,
    typename SB, typename SB_2x4, typename SB_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // Batch dimension handling
    const int im = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    // Batch offsets for srcA and srcB
    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // Tile dimensions
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;

    // Tile offsets in output matrix
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // Threadgroup memory for dequantized A tile only
    threadgroup SA * sa = (threadgroup SA *)(shmem);

    // Work-item count for A loading
    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    // tA wraps threadgroup memory
    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // tB wraps device memory directly
    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    // Configure matmul operation
    // note: K is dynamic_extent (clamped to the valid range in PHASE 2), since a static
    //       N_MM_NK_TOTAL K tile would read src1 out of bounds when K % N_MM_NK_TOTAL != 0
    // ref: https://github.com/ggml-org/llama.cpp/pull/27064
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Accumulate partial results over K dimension
    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        // === PHASE 1: Dequantization of A into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            // Bounds check: skip device read if row is out of matrix bounds
            if (ra + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).
                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,
                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.
                    // Mirrors the legacy kernel's existing guard.
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos / (16 * nl);
                    const short il = (k_pos / 16) % nl;

                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        // Zero-pad A for K positions beyond valid range (handles partial K iterations)
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                // Zero-pad rows beyond matrix bounds
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Tensor matmul ===
        // Clamp the K extent of both operand tensors to the remaining valid K range so
        // the dynamic-K op never reads past the K extent of src1 (or the staged A tile).
        const int kExt = min(N_MM_NK_TOTAL, K - loop_k);

        auto tAv = tensor(sa, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, N_MM_NK_TOTAL}));
        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(kExt, N - rb), array<int, 2>({1, strideB}));

        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result tile to output matrix (with batch offset)
    // cT.store handles bounds checking via tD's extents (M, N)
    device float * dstBatch = (device float *)dst + im * N * args.ldd;

    auto tD = tensor(dstBatch, dextents<int32_t, 2>(M, N), array<int, 2>({1, args.ldd}));
    cT.store(tD.slice(ra, rb));
}

#endif // GGML_METAL_HAS_TENSOR — tensor kernel only. The simdgroup kernel below is
           // always built: A19's tensor tile is 64 columns wide, so a short draft
           // batch has to take the narrow simdgroup tile instead.

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4,
    short NR1_T, short SG_M_T, short NR0_T, short NSG_T, short NK_T, short NBUF_T>
kernel void kernel_mul_mm_simd(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    // NBUF_T == 2 double buffers: iteration i+1 fills the half that iteration i is not
    // reading, so the "wait for all reads before overwriting" barrier at the top of the loop
    // goes away (one threadgroup_barrier per k-chunk instead of two). That costs 2x the
    // threadgroup memory, which at NK=64 can cost more occupancy than the barrier is worth.
    threadgroup S0 * const sa_all = (threadgroup S0 *)(shmem);
    threadgroup S1 * const sb_all = (threadgroup S1 *)(shmem + NBUF_T*NR0_T*NK_T*sizeof(S0));

    short sbuf = 0;

    constexpr int NR0 = NR0_T;
    constexpr int NR1 = NR1_T;

    constexpr int NK  = NK_T;

    // NL0 threads cooperate on one src0 row; each covers NCA 16-element chunks per k-window.
    // NL1 threads cooperate on one src1 row, 8 elements each.
    constexpr int NL0 = 2;
    constexpr int NCA = NK/16/NL0;
    constexpr int NL1 = NK/8;

    static_assert(NR1*NK <= 8*1024, "src1 tile needs more than one 8-element chunk per thread");

    // NR1_T src1 columns per tile, SG_M_T of the 4 simdgroups along M.
    // a 64x32 tile costs the same at ne11=8 as at ne11=32 because the padded columns are
    // still multiplied - at ne11=32 this kernel already runs at ~6 TFLOPS (hardware peak on
    // M4 Pro), so a narrower tile is the only way to make a short speculative verify cheap.
    constexpr short SG_M = SG_M_T;        // simdgroups along M
    constexpr short SG_N = NSG_T/SG_M_T;  // simdgroups along N
    constexpr short MPSG = NR0/SG_M;    // src0 rows    per simdgroup
    constexpr short NPSG = NR1/SG_N;    // src1 columns per simdgroup
    constexpr short NMA  = MPSG/8;      // 8x8 tiles along M
    constexpr short NMB  = NPSG/8;      // 8x8 tiles along N

    const short sg_m = sgitg % SG_M;
    const short sg_n = sgitg / SG_M;

    const int im = tgpig.z;
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // base of this thread's src0 row. the block and the chunk-within-block are derived from
    // the absolute chunk index below, which is correct for every nl (1, 2, 4, 8, 16); the
    // old incremental il/x advance only worked for NK == 32.
    device const block_q * const x0 = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0);

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    S0_8x8 ma[NMA];
    S1_8x8 mb[NMB];

    simdgroup_float8x8 mc[NMA*NMB];

    for (short i = 0; i < NMA*NMB; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        threadgroup S0 * sa = sa_all + sbuf*(NR0*NK);
        threadgroup S1 * sb = sb_all + sbuf*(NR1*NK);

        if (NBUF_T == 2) {
            sbuf ^= 1;
        } else {
            // single buffered: everyone must be done reading before we overwrite
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // load data and store to threadgroup memory.
        //
        // sa is row-major [NR0][NK] and the A operand is read back with simdgroup_load's
        // transpose flag. Upstream instead stores A pre-transposed as 8x8 blocks, which makes
        // each thread issue 16 scattered 2-byte stores; this makes it one 32-byte vector
        // store. Decode here is per-element bound, not byte bound (a 63% larger model costs
        // 2% more time), so that store traffic is the thing worth removing.
        FOR_UNROLL (short c = 0; c < NCA; ++c) {
            const short ic  = il0 + NL0*c;          // 16-element chunk within this k-window
            const int   ica = (loop_k >> 4) + ic;   // ... and in the whole row

            device const block_q * x = x0 + ica/nl;

            threadgroup S0 * dst = sa + lr0*NK + 16*ic;

            if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                // no need for dequantization
                for (short i = 0; i < 16; i++) {
                    dst[i] = 16*ica + i < args.ne00 ? *((device T0 *) x + i) : 0;
                }
            } else {
                S0_4x4 temp_a;
                dequantize_func(x, ica%nl, temp_a);

                *(threadgroup S0_4x4 *) dst = temp_a;
            }
        }

        // with a narrow tile only the first NR1 src1 lanes have anywhere to go
        if ((short)(tiitg/NL1) < NR1) {
            if (FC_mul_mm_bc_inp) {
                for (short i = 0; i < 8; ++i) {
                    const short sx = (tiitg%NL1);
                    const short sy = (tiitg/NL1)/8;

                    const short lx = i;
                    const short ly = (tiitg/NL1)%8;

                    const short ib = (NR1/8)*sx + sy;

                    *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
                }
            } else {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short ly = (tiitg/NL1)%8;

                const short ib = (NR1/8)*sx + sy;

                *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
            }
        }

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + (MPSG*sg_m)*NK);
        threadgroup const S1 * lsmb = (sb + NMB*64*sg_n);

        FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < NMA; i++) {
                simdgroup_load(ma[i], lsma + (8*i)*NK + 8*ik, NK, 0, true);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < NMB; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            FOR_UNROLL (short i = 0; i < NMA*NMB; i++){
                simdgroup_multiply_accumulate(mc[i], mb[i/NMA], ma[i%NMA], mc[i]);
            }

            lsmb += (NR1/8)*64;
        }
    }

    if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        // if no bounds checks on the output are needed, we can directly write to device memory
        device float * C = (device float *) dst +
            (r0 + MPSG*sg_m) + \
            (r1 + NPSG*sg_n) * args.ldd + im*args.ne1*args.ldd;

        for (short i = 0; i < NMA*NMB; i++) {
            simdgroup_store(mc[i], C + 8*(i%NMA) + 8*args.ldd*(i/NMA), args.ldd, 0, false);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + MPSG*sg_m + (NPSG*sg_n)*NR0;

        for (short i = 0; i < NMA*NMB; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%NMA) + 8*NR0*(i/NMA), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (r1 + j)*args.ldd + im*args.ne1*args.ldd;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

template<short ne20> // n_expert_used
kernel void kernel_mul_mm_id_map0(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg; // expert id

    uint32_t n_all = 0;

    device int32_t * ids_i32 = (device int32_t *) hids + ide*args.ne21;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            threadgroup uint16_t * sids = (threadgroup uint16_t *) shmem + tpitg*ne20;

            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sids[i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids = (threadgroup const uint16_t *) shmem + t*ne20;

            short sel = 0;
            #pragma unroll(ne20)
            for (short i20 = 0; i20 < ne20; i20++) {
                sel += (sids[i20] == ide)*(i20 + 1);
            }

            ids_i32[n_all] = (i21 + t)*ne20 + sel - 1;

            n_all += sel > 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device uint32_t * tpe_u32 = (device uint32_t *) (htpe);
    tpe_u32[ide] = n_all;
}

kernel void kernel_mul_mm_id_amax_part_f32(
        constant ggml_metal_kargs_mul_mm_id_amax & args,
        device   const char * src1,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const int nrow = args.ne01*args.ne02;

    float lmax = 0.0f;

    for (int ir = tgpig; ir < nrow; ir += N_MM_NPART_AMAX) {
        const int i01 = ir % args.ne01;
        const int i02 = ir / args.ne01;

        device const float * row = (device const float *) (src1 + i02*args.nb02 + i01*args.nb01);

        for (int i00 = tiitg; i00 < args.ne00; i00 += ntg) {
            lmax = max(lmax, fabs(row[i00]));
        }
    }

    float amax = simd_max(lmax);

    threadgroup float * shared_amax = (threadgroup float *) shmem;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_amax[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_amax[sgitg] = amax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        amax = shared_amax[tiisg];
        amax = simd_max(amax);
    }

    if (tiitg == 0) {
        ((device float *) (dst + 8))[tgpig] = amax;
    }
}

kernel void kernel_mul_mm_id_amax_f32(
        device char * dst,
        ushort tiitg[[thread_index_in_threadgroup]]) {
    device const float * part = (device const float *) (dst + 8);

    float amax = 0.0f;

    for (int i = tiitg; i < N_MM_NPART_AMAX; i += N_SIMDWIDTH) {
        amax = max(amax, part[i]);
    }

    amax = simd_max(amax);

    if (tiitg == 0) {
        // leave a comfortable margin below the f16 max of 65504
        float scale = 1.0f;

        // isfinite: src1 already inf/nan is not ours to fix - keep the
        // scale at 1.0 instead of turning it into a different failure
        if (isfinite(amax) && amax > 32768.0f) {
            scale = exp2(ceil(log2(amax)) - 15.0f);
        }

        device float * d = (device float *) dst;

        d[0] = 1.0f/scale; // exact: scale is a power of two
        d[1] = scale;
    }
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;

template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;

template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4>
kernel void kernel_mul_mm_id(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        device const char * amax,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);

#ifdef GGML_METAL_HAS_TENSOR
    threadgroup float * sc = (threadgroup float *)(shmem);
#endif

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int im = tgpig.z; // expert
    const int r0 = tgpig.y*NR0;
    const int r1 = tgpig.x*NR1;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    const int32_t neh1 = tpe_u32[im];

    if (r1 >= neh1) {
        return;
    }

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id = ids_i32[im*args.ne21 + r1 + lr1];

    const short i11 = (id % args.ne20) % args.ne11;
    const short i12 = (id / args.ne20);
    const short i13 = 0;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    // skip the upper half of the token tile when the expert did not fill it
    constexpr short NR1H = NR1/2;

    const bool has_hi = nr1 > NR1H;

    const short lb1 = (short) tiitg/NL1; // 0 .. NR1-1, this thread's row of the B tile

    // power-of-two rescaling
    const float s1_inv   = ((device const float *) amax)[0];
    const float s1_scale = ((device const float *) amax)[1];

#ifndef GGML_METAL_HAS_TENSOR
    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; i++){
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // simdgroups 2,3 own rows NR1H..NR1-1
    const bool sg_active = has_hi || sgitg < 2;
#else
    auto tA  = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));

    // sb is [NR1][NK] row-major
    auto tB0 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb,             dextents<int32_t, 2>(NK, NR1H));
    auto tB1 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb + NR1H*NK,   dextents<int32_t, 2>(NK, NR1H));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1H, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT0 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto cT1 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();
#endif

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
#ifndef GGML_METAL_HAS_TENSOR
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                *(sa + 64*ib + 8*ly + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // NOTE: this is massively slower.. WTF?
                //sa[64*ib + 8*ly + lx] = temp_a[i/4][i%4];

                *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
              //const short lx = (tiitg/NL1)%8;
              //const short ly = i;

                const short ib = 4*sx + sy;

                *(sb + 64*ib + 8*ly + lx) = loop_k + iy + i < args.ne00 ? (S1) (*((device T1 *) y + i) * (T1) s1_inv) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

          //const short dx = sx;
          //const short dy = sy;

            const short ly = (tiitg/NL1)%8;

            const short ib = 4*sx + sy;

            *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)((*((device T1_2x4 *) y)) * (T1) s1_inv);
        }
#else
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                //const short lx = (tiitg/NL1)%8;
                //const short ly = i;

                *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) (*((device T1 *) y + i) * (T1) s1_inv) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            //const short lx = i;
            const short ly = (tiitg/NL1)%8;
            //const short lx = (tiitg/NL1)%8;
            //const short ly = i;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)((*((device T1_2x4 *) y)) * (T1) s1_inv);
        }
#endif

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifndef GGML_METAL_HAS_TENSOR
        if (sg_active) {
            // load matrices from threadgroup memory and conduct outer products
            threadgroup const S0 * lsma = (sa + 4*64*(sgitg%2));
            threadgroup const S1 * lsmb = (sb + 2*64*(sgitg/2));

            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += 8*64;
                lsmb += 4*64;
            }
        }
#else
        auto sA  = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);

        mm.run(sB0, sA, cT0);

        if (has_hi) {
            auto sB1 = tB1.slice(0, 0);

            mm.run(sB1, sA, cT1);
        }
#endif
    }

    // block is smaller than 64x32, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef GGML_METAL_HAS_TENSOR
    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc,             dextents<int32_t, 2>(NR0, NR1H));
    cT0.store(tC0);

    if (has_hi) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + NR1H*NR0, dextents<int32_t, 2>(NR0, NR1H));
        cT1.store(tC1);
    }
#else
    if (sg_active) {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
    }
#endif

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = sgitg; j < nr1; j += 4) {
        const int id = ids_i32[im*args.ne21 + r1 + j];

        const short ide = id % args.ne20;
        const short idt = id / args.ne20;

        device float  * D  = (device float  *) dst + r0 + ide*args.ne0 + idt*args.ne1*args.ne0;
        device float4 * D4 = (device float4 *) D;

        threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
        threadgroup float4 * C4 = (threadgroup float4 *) C;

        int i = tiisg;
        for (; i < nr0/4; i += 32) {
            *(D4 + i) = *(C4 + i) * s1_scale;
        }

        i = (4*(nr0/4)) + tiisg;
        for (; i < nr0; i += 32) {
            *(D + i) = *(C + i) * s1_scale;
        }
    }
}

//
// matrix-matrix multiplication
//

#ifdef GGML_METAL_HAS_TENSOR
typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;


template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_q1_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

typedef decltype(kernel_mul_mm_simd<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4, 32, 2, 64, 4, 32, 2>) mul_mm_simd_t;
#else
typedef decltype(kernel_mul_mm_simd<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4, 32, 2, 64, 4, 32, 2>) mul_mm_simd_t;
typedef mul_mm_simd_t mul_mm_t;


template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4, 32, 2, 64, 4, 32, 2>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4, 32, 2, 64, 4, 32, 2>;
#endif
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_tq2_0_f32")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4, 32, 2, 64, 4, 32, 2>;

template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q1_0_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q2_0_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_tq2_0_f16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4, 32, 2, 64, 4, 32, 2>;

#endif
//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_id;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q1_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;

template [[host_name("kernel_mul_mm_id_f32_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_f16_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

// narrow-tile GEMM variants: ne11 <= 8 and ne11 <= 16 (speculative verify batches)
template [[host_name("kernel_mul_mm_f32_f32_nc8")]]     kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_f32_f32_nc16")]]     kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_f16_f32_nc8")]]     kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_f16_f32_nc16")]]     kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_bf16_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_bf16_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q1_0_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q1_0_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q2_0_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q2_0_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_0_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_0_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_1_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_1_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_0_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_0_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_1_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_1_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q8_0_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q8_0_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_mxfp4_f32_nc8")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_mxfp4_f32_nc16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q2_K_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q2_K_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q3_K_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q3_K_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_K_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q4_K_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_K_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q5_K_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q6_K_f32_nc8")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_q6_K_f32_nc16")]]    kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32_nc8")]] kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32_nc16")]] kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_xs_f32_nc8")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_xs_f32_nc16")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32_nc8")]] kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32_nc16")]] kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq3_s_f32_nc8")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq3_s_f32_nc16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_s_f32_nc8")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq2_s_f32_nc16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq1_s_f32_nc8")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq1_s_f32_nc16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq1_m_f32_nc8")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq1_m_f32_nc16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq4_nl_f32_nc8")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq4_nl_f32_nc16")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq4_xs_f32_nc8")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_iq4_xs_f32_nc16")]]  kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_tq2_0_f32_nc8")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4, 8, 4, 64, 4, 32, 2>;
template [[host_name("kernel_mul_mm_tq2_0_f32_nc16")]]   kernel mul_mm_simd_t kernel_mul_mm_simd<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4, 16, 4, 64, 4, 32, 2>;

// ---------------------------------------------------------------------------------------------
// Register-fed MMA for short speculative-verify batches (ne11 = 2..8), GGML_METAL_REGFED=1.
//
// Every other quantized GEMM here dequantizes weights into threadgroup memory and simdgroup_loads
// them back. Here each lane dequantizes exactly the two A-fragment elements it owns and writes them
// with thread_elements(): device -> registers -> simdgroup MMA. Weights are the native GGUF blocks,
// src1 is f32 (converted to half in-kernel), accumulation is f32. Generated from the tuned kernels
// in infernet scripts/regfed-gguf.py (BEST variants), measurements in infernet docs/regfed-kernel.md.
//
// Apple 8x8 fragment layout: lane owns (fm, fn) and (fm, fn+1). A = src0 rows x k, B = src1^T
// (k x n), C = rows x n. RF_R 8-row tiles per simdgroup. One 256-superblock per iteration in 4
// chunks of 8 fragments; weights for the block are loaded together and the next block's loads are
// issued before the back-edge. Per format, fragment t / column c = 2j+e maps to a k chosen so a
// lane's two elements come from one loaded byte/word and one scale; B is read at the transposed k.
// IQ grids/signs and IQ4 values are expanded to half2 pairs in threadgroup memory (tables only).
// Columns n >= ne11 are zero-padded and never written.
// ---------------------------------------------------------------------------------------------

// split-K pipelines are compiled with this constant true ("_ks" names); in the others S folds to 1 and the
// split-K code (incl. rf_red) is dead, so the S = 1 kernels compile as before split-K existed
constant bool FC_mul_mm_rf_ks [[function_constant(FC_MUL_MM_RF + 0)]];
// infernet timing ablations (GGML_METAL_REGFED_ABL, iq4_xs only, WRONG MATH): 1 no weight loads, 2 no codebook lookup,
// 3 no MMA (elementwise stand-in), 4 no x loads
constant int  FC_mul_mm_rf_abl [[function_constant(FC_MUL_MM_RF + 1)]];
constant bool FC_mul_mm_rf_abl_def = is_function_constant_defined(FC_mul_mm_rf_abl);
// infernet GGML_METAL_FUSION_FFN_SWIGLU (iq4_xs only): tiles 0..R/2-1 are gate rows (src0), tiles R/2..R-1 the same rows of
// up (src0b); after the (split-K) reduction the kernel writes silu(gate)*up (kernel_swiglu's expression) for 8*R/2 rows
constant bool FC_mul_mm_rf_sw [[function_constant(FC_MUL_MM_RF + 2)]];
constant bool FC_mul_mm_rf_sw_def = is_function_constant_defined(FC_mul_mm_rf_sw);

#define RF_R 4
#define RF_UNROLL _Pragma("clang loop unroll(full)")

// split-K reduce: simdgroups 1..S-1 hand their partial C to simdgroup 0 through threadgroup memory
#define RF_KREDUCE \
    if (S > 1) { \
        if (sgitg > 0) { RF_UNROLL for (short r = 0; r < R; ++r) { \
            rf_red[((sgitg - 1) * R + r) * 64 + 2 * lane]     = C[r].thread_elements()[0]; \
            rf_red[((sgitg - 1) * R + r) * 64 + 2 * lane + 1] = C[r].thread_elements()[1]; } } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (sgitg > 0) return; \
        for (uint s = 1; s < S; ++s) { RF_UNROLL for (short r = 0; r < R; ++r) { \
            C[r].thread_elements()[0] += rf_red[((s - 1) * R + r) * 64 + 2 * lane]; \
            C[r].thread_elements()[1] += rf_red[((s - 1) * R + r) * 64 + 2 * lane + 1]; } } \
    }

kernel void kernel_mul_mm_rf_iq4_xs_f32(
        constant ggml_metal_kargs_mul_mm_rf & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * src0b,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort ntg  [[simdgroups_per_threadgroup]]) {
    // iq4_xs_i from infernet scripts/regfed-gguf.py
    constexpr short R = RF_R;
    const int ABL = FC_mul_mm_rf_abl_def ? FC_mul_mm_rf_abl : 0;
    const int  N = args.ne11;
    const uint K = args.ne00;
    const uint64_t RB = args.nb01;
    const device uchar * w = (const device uchar *) src0;

    const uint lane = tiisg;
    const uint S    = FC_mul_mm_rf_ks && args.ksplit > 1 ? uint(args.ksplit) : 1u;
    const uint sg   = S > 1 ? tgpig : tgpig * ntg + sgitg;
    const short qid = lane / 4;
    const short fm  = (qid & 4) + ((lane / 2) % 4);
    const short fn  = (qid & 2) * 2 + (lane % 2) * 2;
    const short j   = fn / 2;          // A side: lane's column pair (k ownership is per format)
    const short eb  = fm & 1;          // B side: lane's k row fm = (quarter fm>>1, parity eb)
    const bool SW = FC_mul_mm_rf_sw_def && FC_mul_mm_rf_sw;
    const uint row_base = sg * (SW ? 8 * (R / 2) : 8 * R);
    const uint NBLK = K / 256;
    const uint kb0  = S > 1 ? (sgitg * NBLK) / S : 0, kb1 = S > 1 ? ((sgitg + 1) * NBLK) / S : NBLK;
    threadgroup float rf_red[3 * RF_R * 64];
    threadgroup uint tq[256];
    for (uint i = tiitg; i < 256; i += 32 * ntg) tq[i] = as_type<uint>(half2(half(kvalues_iq4nl_f[i & 15]), half(kvalues_iq4nl_f[i >> 4])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_base >= (uint) args.ne01) {
        return;
    }

    simdgroup_matrix<float, 8, 8> C[R];
    RF_UNROLL for (short r = 0; r < R; ++r) C[r] = simdgroup_matrix<float, 8, 8>(0);
    const device uchar *wb = w + (row_base + fm) * RB;
    const device uchar *wbu = (const device uchar *) src0b + (row_base + fm) * RB;
    #define RF_WROW(r) (SW ? ((r) < R / 2 ? wb + (r) * 8 * RB : wbu + ((r) - R / 2) * 8 * RB) : wb + (r) * 8 * RB)
    const bool v0 = fn < N, v1 = fn + 1 < N;

    const device float *x0 = (const device float *)(src1 + min(int(fn), N - 1) * args.nb11) + (4 * (fm >> 1) + 16 * eb);
    const device float *x1 = (const device float *)(src1 + min(int(fn) + 1, N - 1) * args.nb11) + (4 * (fm >> 1) + 16 * eb);
    uint2 H[R]; uint Q[R][8];
    #define LOADW(b) if (ABL == 1) { RF_UNROLL for (short r = 0; r < R; ++r) { H[r] = uint2(0x2c00u | (lane << 16), 0x55555555u ^ (b)); RF_UNROLL for (short u = 0; u < 8; ++u) Q[r][u] = (lane * 2654435761u) ^ (u * 97u + r + (b)); } } else { RF_UNROLL for (short r = 0; r < R; ++r) { const device uchar *p = RF_WROW(r) + (b) * 136; H[r] = *(const device uint2 *)p; const device uint *qp = (const device uint *)(p + 8 + 4 * j); RF_UNROLL for (short u = 0; u < 8; ++u) Q[r][u] = qp[4 * u]; } }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;

        half dl[R][8];
        RF_UNROLL for (short r = 0; r < R; ++r) {
            const float d = float(as_type<half2>(H[r].x).x);
            const uint sh = H[r].x >> 16, sl = H[r].y;
            RF_UNROLL for (short ib = 0; ib < 8; ++ib)
                dl[r][ib] = half(d * (float(((sl >> (4 * ib)) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)) - 32.0f));
        }
        RF_UNROLL for (short q = 0; q < 4; ++q) {
            half2 bx[8];
            RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                const uint o = kb + 64 * q + 32 * gi;
                const float4 u = ABL == 4 ? float4(o * 1e-6f + lane) : *(const device float4 *)(x0 + o);
                const float4 v = ABL == 4 ? float4(o * 2e-6f - lane) : *(const device float4 *)(x1 + o);
                RF_UNROLL for (short i = 0; i < 4; ++i) bx[4 * gi + i] = half2(half(u[i]), half(v[i]));
            }

            if (N < 8) { RF_UNROLL for (short i = 0; i < 8; ++i) bx[i] = half2(v0 ? bx[i].x : 0.0h, v1 ? bx[i].y : 0.0h); }
            simdgroup_matrix<half, 8, 8> B[8];
            RF_UNROLL for (short i = 0; i < 8; ++i) { B[i].thread_elements()[0] = bx[i].x; B[i].thread_elements()[1] = bx[i].y; }
            RF_UNROLL for (short r = 0; r < R; ++r) {
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    half2 a4[4];
                    RF_UNROLL for (short tt = 0; tt < 4; ++tt)
                        a4[tt] = (ABL == 2 ? half2(half((Q[r][2 * q + gi] >> (8 * tt)) & 0xF), half((Q[r][2 * q + gi] >> (8 * tt + 4)) & 0xF))
                                           : as_type<half2>(tq[(Q[r][2 * q + gi] >> (8 * tt)) & 0xFF])) * dl[r][2 * q + gi];
                    RF_UNROLL for (short tt = 0; tt < 4; ++tt) {
                        simdgroup_matrix<half, 8, 8> A;
                        A.thread_elements()[0] = a4[tt].x;
                        A.thread_elements()[1] = a4[tt].y;
                        if (ABL == 3) {
                            C[r].thread_elements()[0] += float(A.thread_elements()[0] * B[4 * gi + tt].thread_elements()[0]);
                            C[r].thread_elements()[1] += float(A.thread_elements()[1] * B[4 * gi + tt].thread_elements()[1]);
                        } else {
                            simdgroup_multiply_accumulate(C[r], A, B[4 * gi + tt], C[r]);
                        }
                    }
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)

    }
    RF_KREDUCE
    if (SW) {
        RF_UNROLL for (short r = 0; r < R / 2; ++r) {
            const uint row = row_base + 8 * r + fm;
            RF_UNROLL for (short e = 0; e < 2; ++e) {
                const float x0 = C[r].thread_elements()[e];
                const float x1 = C[r + R / 2].thread_elements()[e];

                const float silu = x0 / (1.0f + exp(-x0));

                if (e == 0 ? v0 : v1) *((device float *)(dst + (fn + e) * args.nb1) + row) = silu*x1;
            }
        }
        return;
    }
    RF_UNROLL for (short r = 0; r < R; ++r) {
        const uint row = row_base + 8 * r + fm;
        if (v0) *((device float *)(dst + fn * args.nb1) + row) = C[r].thread_elements()[0];
        if (v1) *((device float *)(dst + (fn + 1) * args.nb1) + row) = C[r].thread_elements()[1];
    }
}
#undef RF_WROW
#undef LOADW

kernel void kernel_mul_mm_rf_q4_K_f32(
        constant ggml_metal_kargs_mul_mm_rf & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort ntg  [[simdgroups_per_threadgroup]]) {
    // q4_k_f from infernet scripts/regfed-gguf.py
    constexpr short R = RF_R;
    const int  N = args.ne11;
    const uint K = args.ne00;
    const uint64_t RB = args.nb01;
    const device uchar * w = (const device uchar *) src0;

    const uint lane = tiisg;
    const uint S    = FC_mul_mm_rf_ks && args.ksplit > 1 ? uint(args.ksplit) : 1u;
    const uint sg   = S > 1 ? tgpig : tgpig * ntg + sgitg;
    const short qid = lane / 4;
    const short fm  = (qid & 4) + ((lane / 2) % 4);
    const short fn  = (qid & 2) * 2 + (lane % 2) * 2;
    const short j   = fn / 2;          // A side: lane's column pair (k ownership is per format)
    const short eb  = fm & 1;          // B side: lane's k row fm = (quarter fm>>1, parity eb)
    const uint row_base = sg * (8 * R);
    const uint NBLK = K / 256;
    const uint kb0  = S > 1 ? (sgitg * NBLK) / S : 0, kb1 = S > 1 ? ((sgitg + 1) * NBLK) / S : NBLK;
    threadgroup float rf_red[3 * RF_R * 64];
    if (row_base >= (uint) args.ne01) {
        return;
    }

    simdgroup_matrix<float, 8, 8> C[R];
    RF_UNROLL for (short r = 0; r < R; ++r) C[r] = simdgroup_matrix<float, 8, 8>(0);
    const device uchar *wb = w + (row_base + fm) * RB;
    const bool v0 = fn < N, v1 = fn + 1 < N;

    const device float *x0 = (const device float *)(src1 + min(int(fn), N - 1) * args.nb11) + (8 * (fm >> 1) + 2 * eb);
    const device float *x1 = (const device float *)(src1 + min(int(fn) + 1, N - 1) * args.nb11) + (8 * (fm >> 1) + 2 * eb);
    uint4 H[R]; uint Q[R][8];
    #define LOADW(b) RF_UNROLL for (short r = 0; r < R; ++r) { const device uchar *p = wb + r * 8 * RB + (b) * 144; H[r] = *(const device uint4 *)p; const device uint2 *qp = (const device uint2 *)(p + 16 + 8 * j); RF_UNROLL for (short u = 0; u < 4; ++u) { const uint2 t = qp[4 * u]; Q[r][2 * u] = t.x; Q[r][2 * u + 1] = t.y; } }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;


        RF_UNROLL for (short q = 0; q < 4; ++q) {
            half2 bx[8];
            { const float2 u = *(const device float2 *)(x0 + kb + 64 * q); const float2 v = *(const device float2 *)(x1 + kb + 64 * q);
              bx[0] = half2(half(u.x), half(v.x)); bx[1] = half2(half(u.y), half(v.y)); }
            { const float2 u = *(const device float2 *)(x0 + kb + 64 * q + 32); const float2 v = *(const device float2 *)(x1 + kb + 64 * q + 32);
              bx[2] = half2(half(u.x), half(v.x)); bx[3] = half2(half(u.y), half(v.y)); }
            { const float2 u = *(const device float2 *)(x0 + kb + 64 * q + 4); const float2 v = *(const device float2 *)(x1 + kb + 64 * q + 4);
              bx[4] = half2(half(u.x), half(v.x)); bx[5] = half2(half(u.y), half(v.y)); }
            { const float2 u = *(const device float2 *)(x0 + kb + 64 * q + 36); const float2 v = *(const device float2 *)(x1 + kb + 64 * q + 36);
              bx[6] = half2(half(u.x), half(v.x)); bx[7] = half2(half(u.y), half(v.y)); }

            if (N < 8) { RF_UNROLL for (short i = 0; i < 8; ++i) bx[i] = half2(v0 ? bx[i].x : 0.0h, v1 ? bx[i].y : 0.0h); }
            simdgroup_matrix<half, 8, 8> B[8];
            RF_UNROLL for (short i = 0; i < 8; ++i) { B[i].thread_elements()[0] = bx[i].x; B[i].thread_elements()[1] = bx[i].y; }
            RF_UNROLL for (short r = 0; r < R; ++r) {
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    half2 a4[4];
                    {
                        const uint Hs[3] = { H[r].y, H[r].z, H[r].w };
                        const half2 dd = as_type<half2>(H[r].x);
                        float ds[2], dm[2];
                        RF_UNROLL for (short s = 0; s < 2; ++s) {
                            const short ib = 2 * q + s;
                            #define QB(i) ((Hs[(i) >> 2] >> (8 * ((i) & 3))) & 0xFF)
                            const uint sc = ib < 4 ? (QB(ib) & 63) : ((QB(ib + 4) & 0xF) | ((QB(ib - 4) >> 6) << 4));
                            const uint mn = ib < 4 ? (QB(ib + 4) & 63) : ((QB(ib + 4) >> 4) | ((QB(ib) >> 6) << 4));
                            ds[s] = float(dd.x) * float(sc);
                            dm[s] = -float(dd.y) * float(mn);
                        }
                        RF_UNROLL for (short tt = 0; tt < 4; ++tt) {
                            const uint sh = (tt & 1) * 8 + (tt >> 1) * 4;
                            const half2 qq = as_type<half2>(((Q[r][2 * q + gi] >> sh) & 0x000F000Fu) | 0x64006400u) - half2(1024.0h);
                            a4[tt] = half2(fma(float2(qq), float2(ds[tt >> 1]), float2(dm[tt >> 1])));
                        }
                    }
                    RF_UNROLL for (short tt = 0; tt < 4; ++tt) {
                        simdgroup_matrix<half, 8, 8> A;
                        A.thread_elements()[0] = a4[tt].x;
                        A.thread_elements()[1] = a4[tt].y;
                        simdgroup_multiply_accumulate(C[r], A, B[4 * gi + tt], C[r]);
                    }
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)

    }
    RF_KREDUCE
    RF_UNROLL for (short r = 0; r < R; ++r) {
        const uint row = row_base + 8 * r + fm;
        if (v0) *((device float *)(dst + fn * args.nb1) + row) = C[r].thread_elements()[0];
        if (v1) *((device float *)(dst + (fn + 1) * args.nb1) + row) = C[r].thread_elements()[1];
    }
}
#undef LOADW
#undef QB

kernel void kernel_mul_mm_rf_iq3_xxs_f32(
        constant ggml_metal_kargs_mul_mm_rf & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort ntg  [[simdgroups_per_threadgroup]]) {
    // iq3_xxs from infernet scripts/regfed-gguf.py
    constexpr short R = RF_R;
    const int  N = args.ne11;
    const uint K = args.ne00;
    const uint64_t RB = args.nb01;
    const device uchar * w = (const device uchar *) src0;

    const uint lane = tiisg;
    const uint S    = FC_mul_mm_rf_ks && args.ksplit > 1 ? uint(args.ksplit) : 1u;
    const uint sg   = S > 1 ? tgpig : tgpig * ntg + sgitg;
    const short qid = lane / 4;
    const short fm  = (qid & 4) + ((lane / 2) % 4);
    const short fn  = (qid & 2) * 2 + (lane % 2) * 2;
    const short j   = fn / 2;          // A side: lane's column pair (k ownership is per format)
    const short eb  = fm & 1;          // B side: lane's k row fm = (quarter fm>>1, parity eb)
    const uint row_base = sg * (8 * R);
    const uint NBLK = K / 256;
    const uint kb0  = S > 1 ? (sgitg * NBLK) / S : 0, kb1 = S > 1 ? ((sgitg + 1) * NBLK) / S : NBLK;
    threadgroup float rf_red[3 * RF_R * 64];
    threadgroup uint2 tgrid[256];
    for (uint i = tiitg; i < 256; i += 32 * ntg) {
        const uint g = iq3xxs_grid[i];
        tgrid[i] = uint2(as_type<uint>(half2(half(g & 0xFF), half((g >> 16) & 0xFF))),
                        as_type<uint>(half2(half((g >> 8) & 0xFF), half(g >> 24))));
    }
    threadgroup uint4 tsign[128];
    for (uint i = tiitg; i < 128; i += 32 * ntg) { const uint s = ksigns_iq2xs[i]; tsign[i] = (uint4(s, s >> 1, s >> 4, s >> 5) & 5u) * 0x20008000u & 0x80008000u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_base >= (uint) args.ne01) {
        return;
    }

    simdgroup_matrix<float, 8, 8> C[R];
    RF_UNROLL for (short r = 0; r < R; ++r) C[r] = simdgroup_matrix<float, 8, 8>(0);
    const device uchar *wb = w + (row_base + fm) * RB;
    const bool v0 = fn < N, v1 = fn + 1 < N;

    const device float *x0 = (const device float *)(src1 + min(int(fn), N - 1) * args.nb11) + (64 * (fm >> 1) + 2 * eb * 1);
    const device float *x1 = (const device float *)(src1 + min(int(fn) + 1, N - 1) * args.nb11) + (64 * (fm >> 1) + 2 * eb * 1);
    ushort Dd[R], QS[R][8], AX[R][4];
    #define LOADW(b) RF_UNROLL for (short r = 0; r < R; ++r) { const device uchar *p = wb + r * 8 * RB + (b) * 98; const device ushort *ps = (const device ushort *)p; Dd[r] = ps[0]; RF_UNROLL for (short g = 0; g < 8; ++g) QS[r][g] = ps[1 + 8 * j + g]; RF_UNROLL for (short u = 0; u < 4; ++u) AX[r][u] = ps[33 + 4 * j + u]; }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;

        half db[R][2]; uint aux[R][2];
        RF_UNROLL for (short r = 0; r < R; ++r) {
            const float d = float(as_type<half>(Dd[r]));
            RF_UNROLL for (short s = 0; s < 2; ++s) {
                aux[r][s] = uint(AX[r][2 * s]) | (uint(AX[r][2 * s + 1]) << 16);
                db[r][s] = half(d * (0.5f + float(aux[r][s] >> 28)) * 0.5f);
            }
        }
        RF_UNROLL for (short q = 0; q < 4; ++q) {
            half2 bx[8];
            { const float2 u = *(const device float2 *)(x0 + kb + 16 * q); const float2 v = *(const device float2 *)(x1 + kb + 16 * q);
              bx[0] = half2(half(u.x), half(v.x)); bx[1] = half2(half(u.y), half(v.y)); }
            { const float2 u = *(const device float2 *)(x0 + kb + 16 * q + 4); const float2 v = *(const device float2 *)(x1 + kb + 16 * q + 4);
              bx[2] = half2(half(u.x), half(v.x)); bx[3] = half2(half(u.y), half(v.y)); }
            { const float2 u = *(const device float2 *)(x0 + kb + 16 * q + 8); const float2 v = *(const device float2 *)(x1 + kb + 16 * q + 8);
              bx[4] = half2(half(u.x), half(v.x)); bx[5] = half2(half(u.y), half(v.y)); }
            { const float2 u = *(const device float2 *)(x0 + kb + 16 * q + 12); const float2 v = *(const device float2 *)(x1 + kb + 16 * q + 12);
              bx[6] = half2(half(u.x), half(v.x)); bx[7] = half2(half(u.y), half(v.y)); }

            if (N < 8) { RF_UNROLL for (short i = 0; i < 8; ++i) bx[i] = half2(v0 ? bx[i].x : 0.0h, v1 ? bx[i].y : 0.0h); }
            simdgroup_matrix<half, 8, 8> B[8];
            RF_UNROLL for (short i = 0; i < 8; ++i) { B[i].thread_elements()[0] = bx[i].x; B[i].thread_elements()[1] = bx[i].y; }
            RF_UNROLL for (short r = 0; r < R; ++r) {
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    half2 a4[4];
                    {
                        const short G = 2 * q + gi;
                        const uint qs = QS[r][G];
                        const uint2 g0 = tgrid[qs & 0xFF], g1 = tgrid[qs >> 8];
                        const uint4 m = tsign[(aux[r][G >> 2] >> (7 * (G & 3))) & 127];
                        const half s = db[r][G >> 2];
                        a4[0] = as_type<half2>(g0.x ^ m.x) * s; a4[1] = as_type<half2>(g0.y ^ m.y) * s;
                        a4[2] = as_type<half2>(g1.x ^ m.z) * s; a4[3] = as_type<half2>(g1.y ^ m.w) * s;
                    }
                    RF_UNROLL for (short tt = 0; tt < 4; ++tt) {
                        simdgroup_matrix<half, 8, 8> A;
                        A.thread_elements()[0] = a4[tt].x;
                        A.thread_elements()[1] = a4[tt].y;
                        simdgroup_multiply_accumulate(C[r], A, B[4 * gi + tt], C[r]);
                    }
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)

    }
    RF_KREDUCE
    RF_UNROLL for (short r = 0; r < R; ++r) {
        const uint row = row_base + 8 * r + fm;
        if (v0) *((device float *)(dst + fn * args.nb1) + row) = C[r].thread_elements()[0];
        if (v1) *((device float *)(dst + (fn + 1) * args.nb1) + row) = C[r].thread_elements()[1];
    }
}
#undef LOADW

// single-block IQ3_S, used when K/256 is odd (rows then only 2-byte aligned); see _2b below
kernel void kernel_mul_mm_rf_iq3_s_1b_f32(
        constant ggml_metal_kargs_mul_mm_rf & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort ntg  [[simdgroups_per_threadgroup]]) {
    // iq3_s_v2 from infernet scripts/regfed-gguf.py
    constexpr short R = RF_R;
    const int  N = args.ne11;
    const uint K = args.ne00;
    const uint64_t RB = args.nb01;
    const device uchar * w = (const device uchar *) src0;

    const uint lane = tiisg;
    const uint S    = FC_mul_mm_rf_ks && args.ksplit > 1 ? uint(args.ksplit) : 1u;
    const uint sg   = S > 1 ? tgpig : tgpig * ntg + sgitg;
    const short qid = lane / 4;
    const short fm  = (qid & 4) + ((lane / 2) % 4);
    const short fn  = (qid & 2) * 2 + (lane % 2) * 2;
    const short j   = fn / 2;          // A side: lane's column pair (k ownership is per format)
    const short eb  = fm & 1;          // B side: lane's k row fm = (quarter fm>>1, parity eb)
    const uint row_base = sg * (8 * R);
    const uint NBLK = K / 256;
    const uint kb0  = S > 1 ? (sgitg * NBLK) / S : 0, kb1 = S > 1 ? ((sgitg + 1) * NBLK) / S : NBLK;
    threadgroup float rf_red[3 * RF_R * 64];
    threadgroup uint2 tgrid[512];
    for (uint i = tiitg; i < 512; i += 32 * ntg) {
        const uint g = iq3s_grid[i];
        tgrid[i] = uint2(as_type<uint>(half2(half(g & 0xFF), half((g >> 16) & 0xFF))),
                        as_type<uint>(half2(half((g >> 8) & 0xFF), half(g >> 24))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_base >= (uint) args.ne01) {
        return;
    }

    simdgroup_matrix<float, 8, 8> C[R];
    RF_UNROLL for (short r = 0; r < R; ++r) C[r] = simdgroup_matrix<float, 8, 8>(0);
    const device uchar *wb = w + (row_base + fm) * RB;
    const bool v0 = fn < N, v1 = fn + 1 < N;

    const device float *x0 = (const device float *)(src1 + min(int(fn), N - 1) * args.nb11) + (64 * (fm >> 1) + 2 * eb * 1);
    const device float *x1 = (const device float *)(src1 + min(int(fn) + 1, N - 1) * args.nb11) + (64 * (fm >> 1) + 2 * eb * 1);
    ushort Dd[R], QH[R], QS[R][8], SG[R][4]; uchar SC[R];
    #define LOADW(b) RF_UNROLL for (short r = 0; r < R; ++r) { const device uchar *p = wb + r * 8 * RB + (b) * 110; const device ushort *ps = (const device ushort *)p; Dd[r] = ps[0]; QH[r] = ps[33 + j]; SC[r] = p[106 + j]; RF_UNROLL for (short g = 0; g < 8; ++g) QS[r][g] = ps[1 + 8 * j + g]; RF_UNROLL for (short u = 0; u < 4; ++u) SG[r][u] = ps[37 + 4 * j + u]; }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;

        half db[R][2];
        RF_UNROLL for (short r = 0; r < R; ++r) {
            const float d = float(as_type<half>(Dd[r]));
            db[r][0] = half(d * float(1 + 2 * (SC[r] & 0xF))); db[r][1] = half(d * float(1 + 2 * (SC[r] >> 4)));
        }
        RF_UNROLL for (short q = 0; q < 4; ++q) {
            half2 bx[8];
            { const float2 u = *(const device float2 *)(x0 + kb + 16 * q); const float2 v = *(const device float2 *)(x1 + kb + 16 * q);
              bx[0] = half2(half(u.x), half(v.x)); bx[1] = half2(half(u.y), half(v.y)); }
            { const float2 u = *(const device float2 *)(x0 + kb + 16 * q + 4); const float2 v = *(const device float2 *)(x1 + kb + 16 * q + 4);
              bx[2] = half2(half(u.x), half(v.x)); bx[3] = half2(half(u.y), half(v.y)); }
            { const float2 u = *(const device float2 *)(x0 + kb + 16 * q + 8); const float2 v = *(const device float2 *)(x1 + kb + 16 * q + 8);
              bx[4] = half2(half(u.x), half(v.x)); bx[5] = half2(half(u.y), half(v.y)); }
            { const float2 u = *(const device float2 *)(x0 + kb + 16 * q + 12); const float2 v = *(const device float2 *)(x1 + kb + 16 * q + 12);
              bx[6] = half2(half(u.x), half(v.x)); bx[7] = half2(half(u.y), half(v.y)); }

            if (N < 8) { RF_UNROLL for (short i = 0; i < 8; ++i) bx[i] = half2(v0 ? bx[i].x : 0.0h, v1 ? bx[i].y : 0.0h); }
            simdgroup_matrix<half, 8, 8> B[8];
            RF_UNROLL for (short i = 0; i < 8; ++i) { B[i].thread_elements()[0] = bx[i].x; B[i].thread_elements()[1] = bx[i].y; }
            RF_UNROLL for (short r = 0; r < R; ++r) {
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    half2 a4[4];
                    {
                        const short G = 2 * q + gi;
                        const uint qs = QS[r][G], qh = QH[r];
                        const uint2 g0 = tgrid[(qs & 0xFF) | (((qh >> (2 * G)) & 1) << 8)];
                        const uint2 g1 = tgrid[(qs >> 8) | (((qh >> (2 * G + 1)) & 1) << 8)];
                        const uint sb = SG[r][G >> 1] >> (8 * (G & 1)); const uint4 m = (uint4(sb, sb >> 1, sb >> 4, sb >> 5) & 5u) * 0x20008000u & 0x80008000u;
                        const half s = db[r][G >> 2];
                        a4[0] = as_type<half2>(g0.x ^ m.x) * s; a4[1] = as_type<half2>(g0.y ^ m.y) * s;
                        a4[2] = as_type<half2>(g1.x ^ m.z) * s; a4[3] = as_type<half2>(g1.y ^ m.w) * s;
                    }
                    RF_UNROLL for (short tt = 0; tt < 4; ++tt) {
                        simdgroup_matrix<half, 8, 8> A;
                        A.thread_elements()[0] = a4[tt].x;
                        A.thread_elements()[1] = a4[tt].y;
                        simdgroup_multiply_accumulate(C[r], A, B[4 * gi + tt], C[r]);
                    }
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)

    }
    RF_KREDUCE
    RF_UNROLL for (short r = 0; r < R; ++r) {
        const uint row = row_base + 8 * r + fm;
        if (v0) *((device float *)(dst + fn * args.nb1) + row) = C[r].thread_elements()[0];
        if (v1) *((device float *)(dst + (fn + 1) * args.nb1) + row) = C[r].thread_elements()[1];
    }
}
#undef LOADW

// ---------------------------------------------------------------------------------------------
// regfed v2: IQ2_S, IQ2_XS, IQ2_XXS, Q2_K, Q6_K. Same skeleton as above (R 8-row tiles, one
// 256-block per iteration, next block loaded before the back-edge), written with shared macros.
// Every format here yields 8 consecutive-ish elements per (lane, 8-group G) as 4 half2 pairs
// (e0,e2) (e1,e3) (e4,e6) (e5,e7), so fragment tt, column c = 2j+e maps to
//   IQ map: k = 64j + 8G + 4(tt>>1) + (tt&1) + 2e        (lane j owns sub-blocks 2j, 2j+1)
//   K  map: k = 32G + 8j + 4(tt>>1) + (tt&1) + 2e        (lane j owns bytes 8j..8j+7 of each 32)
// and B row fm is read at the same k with j -> fm>>1, e -> fm&1.
// ---------------------------------------------------------------------------------------------

#define RF_ARGS \
        constant ggml_metal_kargs_mul_mm_rf & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        uint   tgpig[[threadgroup_position_in_grid]], \
        ushort tiitg[[thread_index_in_threadgroup]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]], \
        ushort ntg  [[simdgroups_per_threadgroup]]

#define RF_HEAD \
    constexpr short R = RF_R; \
    const int  N = args.ne11; \
    const uint K = args.ne00; \
    const uint64_t RB = args.nb01; \
    const device uchar * w = (const device uchar *) src0; \
    const uint lane = tiisg; \
    const uint sg   = tgpig * ntg + sgitg; \
    const short qid = lane / 4; \
    const short fm  = (qid & 4) + ((lane / 2) % 4); \
    const short fn  = (qid & 2) * 2 + (lane % 2) * 2; \
    const short j   = fn / 2; \
    const short eb  = fm & 1; \
    const uint row_base = sg * (8 * R); \
    const uint NBLK = K / 256; \
    (void) j;

// RF_HEAD with split-K (args.ksplit = S > 1): the S simdgroups of a threadgroup share one 32-row tile (sg from
// the threadgroup alone) and each takes superblocks [kb0, kb1); RF_KREDUCE sums them before RF_STORE
#define RF_HEAD_KS \
    constexpr short R = RF_R; \
    const int  N = args.ne11; \
    const uint K = args.ne00; \
    const uint64_t RB = args.nb01; \
    const device uchar * w = (const device uchar *) src0; \
    const uint lane = tiisg; \
    const uint S    = FC_mul_mm_rf_ks && args.ksplit > 1 ? uint(args.ksplit) : 1u; \
    const uint sg   = S > 1 ? tgpig : tgpig * ntg + sgitg; \
    const short qid = lane / 4; \
    const short fm  = (qid & 4) + ((lane / 2) % 4); \
    const short fn  = (qid & 2) * 2 + (lane % 2) * 2; \
    const short j   = fn / 2; \
    const short eb  = fm & 1; \
    const uint row_base = sg * (8 * R); \
    const uint NBLK = K / 256; \
    const uint kb0  = S > 1 ? (sgitg * NBLK) / S : 0, kb1 = S > 1 ? ((sgitg + 1) * NBLK) / S : NBLK; \
    threadgroup float rf_red[3 * RF_R * 64]; \
    (void) j; (void) kb0; (void) kb1;

#define RF_SETUP(XOFF) \
    simdgroup_matrix<float, 8, 8> C[R]; \
    RF_UNROLL for (short r = 0; r < R; ++r) C[r] = simdgroup_matrix<float, 8, 8>(0); \
    const device uchar *wb = w + (row_base + fm) * RB; \
    const bool v0 = fn < N, v1 = fn + 1 < N; \
    const device float *x0 = (const device float *)(src1 + min(int(fn), N - 1) * args.nb11) + (XOFF); \
    const device float *x1 = (const device float *)(src1 + min(int(fn) + 1, N - 1) * args.nb11) + (XOFF);

// B fragments for one q (two 8-groups): four float2 reads per x row at offsets o0..o3
#define RF_BX(o0, o1, o2, o3) \
    simdgroup_matrix<half, 8, 8> B[8]; \
    { \
        const uint oo[4] = { (o0), (o1), (o2), (o3) }; \
        half2 bx[8]; \
        RF_UNROLL for (short i = 0; i < 4; ++i) { \
            const float2 u = *(const device float2 *)(x0 + oo[i]); const float2 v = *(const device float2 *)(x1 + oo[i]); \
            bx[2 * i] = half2(half(u.x), half(v.x)); bx[2 * i + 1] = half2(half(u.y), half(v.y)); \
        } \
        if (N < 8) { RF_UNROLL for (short i = 0; i < 8; ++i) bx[i] = half2(v0 ? bx[i].x : 0.0h, v1 ? bx[i].y : 0.0h); } \
        RF_UNROLL for (short i = 0; i < 8; ++i) { B[i].thread_elements()[0] = bx[i].x; B[i].thread_elements()[1] = bx[i].y; } \
    }

#define RF_MMA(r, gi) \
    RF_UNROLL for (short tt = 0; tt < 4; ++tt) { \
        simdgroup_matrix<half, 8, 8> A; \
        A.thread_elements()[0] = a4[tt].x; \
        A.thread_elements()[1] = a4[tt].y; \
        simdgroup_multiply_accumulate(C[r], A, B[4 * (gi) + tt], C[r]); \
    }

#define RF_STORE \
    RF_UNROLL for (short r = 0; r < R; ++r) { \
        const uint row = row_base + 8 * r + fm; \
        if (v0) *((device float *)(dst + fn * args.nb1) + row) = C[r].thread_elements()[0]; \
        if (v1) *((device float *)(dst + (fn + 1) * args.nb1) + row) = C[r].thread_elements()[1]; \
    }

// sign byte -> XOR masks for the pairs (e0,e2) (e1,e3) (e4,e6) (e5,e7)
#define RF_SMASK(s) ((uint4((s), (s) >> 1, (s) >> 4, (s) >> 5) & 5u) * 0x20008000u & 0x80008000u)
// two bytes (b0,b2) of a word -> half2 of their small unsigned values (exact)
#define RF_MAGIC(u) (as_type<half2>((u) | 0x64006400u) - half2(1024.0h))

// grid entry (8 byte values) -> (e0,e2) (e1,e3) (e4,e6) (e5,e7) as half2 pairs
#define RF_GRID8(g) uint4(as_type<uint>(half2(half((g) & 0xFF), half(((g) >> 16) & 0xFF))), \
                          as_type<uint>(half2(half(((g) >> 8) & 0xFF), half(((g) >> 24) & 0xFF))), \
                          as_type<uint>(half2(half(((g) >> 32) & 0xFF), half(((g) >> 48) & 0xFF))), \
                          as_type<uint>(half2(half(((g) >> 40) & 0xFF), half(((g) >> 56) & 0xFF))))

kernel void kernel_mul_mm_rf_iq2_xxs_f32(RF_ARGS) {
    RF_HEAD_KS
    threadgroup uint4 tgrid[256];
    for (uint i = tiitg; i < 256; i += 32 * ntg) { const uint64_t g = iq2xxs_grid[i]; tgrid[i] = RF_GRID8(g); }
    threadgroup uint4 tsign[128];
    for (uint i = tiitg; i < 128; i += 32 * ntg) { const uint s = ksigns_iq2xs[i]; tsign[i] = RF_SMASK(s); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_base >= (uint) args.ne01) {
        return;
    }
    RF_SETUP(64 * (fm >> 1) + 2 * eb)
    // block 66 B: d, then per sub-block 4 grid-index bytes + (4 x 7-bit sign index | 4-bit scale)
    ushort Dd[R], Q[R][8];
    #define LOADW(b) RF_UNROLL for (short r = 0; r < R; ++r) { const device ushort *ps = (const device ushort *)(wb + r * 8 * RB + (b) * 66); Dd[r] = ps[0]; RF_UNROLL for (short u = 0; u < 8; ++u) Q[r][u] = ps[1 + 8 * j + u]; }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;
        half db[R][2]; uint aux[R][2];
        RF_UNROLL for (short r = 0; r < R; ++r) {
            const float d = float(as_type<half>(Dd[r]));
            RF_UNROLL for (short s = 0; s < 2; ++s) {
                aux[r][s] = uint(Q[r][4 * s + 2]) | (uint(Q[r][4 * s + 3]) << 16);
                db[r][s] = half(d * (0.5f + float(aux[r][s] >> 28)) * 0.25f);
            }
        }
        RF_UNROLL for (short q = 0; q < 4; ++q) {
            RF_BX(kb + 16 * q, kb + 16 * q + 4, kb + 16 * q + 8, kb + 16 * q + 12)
            RF_UNROLL for (short r = 0; r < R; ++r) {
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    const short G = 2 * q + gi, s = G >> 2, l = G & 3;
                    const uint4 g = tgrid[(Q[r][4 * s + (l >> 1)] >> (8 * (l & 1))) & 0xFF];
                    const uint4 m = tsign[(aux[r][s] >> (7 * l)) & 127];
                    const half sc = db[r][s];
                    half2 a4[4];
                    a4[0] = as_type<half2>(g.x ^ m.x) * sc; a4[1] = as_type<half2>(g.y ^ m.y) * sc;
                    a4[2] = as_type<half2>(g.z ^ m.z) * sc; a4[3] = as_type<half2>(g.w ^ m.w) * sc;
                    RF_MMA(r, gi)
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)
    }
    RF_KREDUCE
    RF_STORE
}
#undef LOADW

kernel void kernel_mul_mm_rf_iq2_xs_f32(RF_ARGS) {
    RF_HEAD_KS
    threadgroup uint4 tgrid[512];
    for (uint i = tiitg; i < 512; i += 32 * ntg) { const uint64_t g = iq2xs_grid[i]; tgrid[i] = RF_GRID8(g); }
    threadgroup uint4 tsign[128];
    for (uint i = tiitg; i < 128; i += 32 * ntg) { const uint s = ksigns_iq2xs[i]; tsign[i] = RF_SMASK(s); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_base >= (uint) args.ne01) {
        return;
    }
    RF_SETUP(64 * (fm >> 1) + 2 * eb)
    // block 74 B: d, qs[32] (9-bit grid index | 7-bit sign index), scales[8] (nibble per 16)
    ushort Dd[R], Q[R][8], SC[R];
    #define LOADW(b) RF_UNROLL for (short r = 0; r < R; ++r) { const device ushort *ps = (const device ushort *)(wb + r * 8 * RB + (b) * 74); Dd[r] = ps[0]; SC[r] = ps[33 + j]; RF_UNROLL for (short u = 0; u < 8; ++u) Q[r][u] = ps[1 + 8 * j + u]; }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;
        half db[R][4];
        RF_UNROLL for (short r = 0; r < R; ++r) {
            const float d = float(as_type<half>(Dd[r]));
            RF_UNROLL for (short h = 0; h < 4; ++h) db[r][h] = half(d * (0.5f + float((SC[r] >> (4 * h)) & 0xF)) * 0.25f);
        }
        RF_UNROLL for (short q = 0; q < 4; ++q) {
            RF_BX(kb + 16 * q, kb + 16 * q + 4, kb + 16 * q + 8, kb + 16 * q + 12)
            RF_UNROLL for (short r = 0; r < R; ++r) {
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    const short G = 2 * q + gi;
                    const uint qv = Q[r][G];
                    const uint4 g = tgrid[qv & 511];
                    const uint4 m = tsign[qv >> 9];
                    const half sc = db[r][G >> 1];
                    half2 a4[4];
                    a4[0] = as_type<half2>(g.x ^ m.x) * sc; a4[1] = as_type<half2>(g.y ^ m.y) * sc;
                    a4[2] = as_type<half2>(g.z ^ m.z) * sc; a4[3] = as_type<half2>(g.w ^ m.w) * sc;
                    RF_MMA(r, gi)
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)
    }
    RF_KREDUCE
    RF_STORE
}
#undef LOADW

kernel void kernel_mul_mm_rf_iq2_s_f32(RF_ARGS) {
    RF_HEAD_KS
    // 1024 entries expanded to half pairs (16 KB). Raw bytes (8 KB) + RF_MAGIC decode: 163 us vs 137 us
    // (6144x5120, n=8); reading iq2s_grid from constant memory: 152 us
    threadgroup uint4 tgrid[1024];
    for (uint i = tiitg; i < 1024; i += 32 * ntg) { const uint64_t g = iq2s_grid[i]; tgrid[i] = RF_GRID8(g); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_base >= (uint) args.ne01) {
        return;
    }
    RF_SETUP(64 * (fm >> 1) + 2 * eb)
    // block 82 B: d, qs[32] grid index low bytes, signs[32], qh[8] (2 bits per index), scales[8]
    ushort Dd[R], QS[R][4], SG[R][4], QH[R], SC[R];
    #define LOADW(b) RF_UNROLL for (short r = 0; r < R; ++r) { const device ushort *ps = (const device ushort *)(wb + r * 8 * RB + (b) * 82); Dd[r] = ps[0]; QH[r] = ps[33 + j]; SC[r] = ps[37 + j]; RF_UNROLL for (short u = 0; u < 4; ++u) { QS[r][u] = ps[1 + 4 * j + u]; SG[r][u] = ps[17 + 4 * j + u]; } }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;
        half db[R][4];
        RF_UNROLL for (short r = 0; r < R; ++r) {
            const float d = float(as_type<half>(Dd[r]));
            RF_UNROLL for (short h = 0; h < 4; ++h) db[r][h] = half(d * (0.5f + float((SC[r] >> (4 * h)) & 0xF)) * 0.25f);
        }
        RF_UNROLL for (short q = 0; q < 4; ++q) {
            RF_BX(kb + 16 * q, kb + 16 * q + 4, kb + 16 * q + 8, kb + 16 * q + 12)
            RF_UNROLL for (short r = 0; r < R; ++r) {
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    const short G = 2 * q + gi;
                    const uint qsb = (QS[r][G >> 1] >> (8 * (G & 1))) & 0xFF;
                    const uint hi  = (QH[r] >> (8 * (G >> 2) + 2 * (G & 3))) & 3;
                    const uint4 g = tgrid[qsb | (hi << 8)];
                    const uint sb = SG[r][G >> 1] >> (8 * (G & 1));
                    const uint4 m = RF_SMASK(sb);
                    const half sc = db[r][G >> 1];
                    half2 a4[4];
                    a4[0] = as_type<half2>(g.x ^ m.x) * sc; a4[1] = as_type<half2>(g.y ^ m.y) * sc;
                    a4[2] = as_type<half2>(g.z ^ m.z) * sc; a4[3] = as_type<half2>(g.w ^ m.w) * sc;
                    RF_MMA(r, gi)
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)
    }
    RF_KREDUCE
    RF_STORE
}
#undef LOADW

kernel void kernel_mul_mm_rf_q2_K_f32(RF_ARGS) {
    RF_HEAD_KS
    if (row_base >= (uint) args.ne01) {
        return;
    }
    RF_SETUP(8 * (fm >> 1) + 2 * eb)
    // block 84 B (4-aligned): scales[16] (scale | min << 4), qs[64] (2 bits, 4 shifts), d, dmin
    uint SCL[R][4], Q[R][4], DM[R];
    #define LOADW(b) RF_UNROLL for (short r = 0; r < R; ++r) { const device uint *pu = (const device uint *)(wb + r * 8 * RB + (b) * 84); DM[r] = pu[20]; RF_UNROLL for (short u = 0; u < 4; ++u) SCL[r][u] = pu[u]; RF_UNROLL for (short n = 0; n < 2; ++n) { Q[r][2 * n] = pu[4 + 8 * n + 2 * j]; Q[r][2 * n + 1] = pu[5 + 8 * n + 2 * j]; } }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;
        RF_UNROLL for (short q = 0; q < 4; ++q) {
            RF_BX(kb + 64 * q, kb + 64 * q + 4, kb + 64 * q + 32, kb + 64 * q + 36)
            RF_UNROLL for (short r = 0; r < R; ++r) {
                const float2 dd = float2(as_type<half2>(DM[r]));
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    const short G = 2 * q + gi, n = G >> 2, sh = 2 * (G & 3);
                    const uint sc = (SCL[r][G >> 1] >> (8 * (2 * (G & 1)) + 8 * (j >> 1))) & 0xFF;
                    const float2 ds = float2(dd.x * float(sc & 0xF)), dm = float2(-dd.y * float(sc >> 4));
                    half2 a4[4];
                    a4[0] = half2(fma(float2(RF_MAGIC((Q[r][2 * n]     >> sh)       & 0x00030003u)), ds, dm));
                    a4[1] = half2(fma(float2(RF_MAGIC((Q[r][2 * n]     >> (sh + 8)) & 0x00030003u)), ds, dm));
                    a4[2] = half2(fma(float2(RF_MAGIC((Q[r][2 * n + 1] >> sh)       & 0x00030003u)), ds, dm));
                    a4[3] = half2(fma(float2(RF_MAGIC((Q[r][2 * n + 1] >> (sh + 8)) & 0x00030003u)), ds, dm));
                    RF_MMA(r, gi)
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)
    }
    RF_KREDUCE
    RF_STORE
}
#undef LOADW

kernel void kernel_mul_mm_rf_q6_K_f32(RF_ARGS) {
    RF_HEAD_KS
    if (row_base >= (uint) args.ne01) {
        return;
    }
    RF_SETUP(8 * (fm >> 1) + 2 * eb)
    // block 210 B (2-aligned): ql[128] (low nibbles), qh[64] (2 bits, 4 shifts), int8 scales[16], d
    uint L[R][8], H[R][4]; ushort Dd[R]; char SC[R][8];
    #define LOADW(b) RF_UNROLL for (short r = 0; r < R; ++r) { const device ushort *ps = (const device ushort *)(wb + r * 8 * RB + (b) * 210); Dd[r] = ps[104]; \
        RF_UNROLL for (short u = 0; u < 8; ++u) { const short n = u >> 2, h = (u >> 1) & 1, v = u & 1; L[r][u] = uint(ps[32 * n + 16 * h + 4 * j + 2 * v]) | (uint(ps[32 * n + 16 * h + 4 * j + 2 * v + 1]) << 16); } \
        RF_UNROLL for (short u = 0; u < 4; ++u) { const short n = u >> 1, v = u & 1; H[r][u] = uint(ps[64 + 16 * n + 4 * j + 2 * v]) | (uint(ps[64 + 16 * n + 4 * j + 2 * v + 1]) << 16); } \
        RF_UNROLL for (short G = 0; G < 8; ++G) SC[r][G] = as_type<char>(uchar(ps[96 + G] >> (8 * (j >> 1)))); }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;
        RF_UNROLL for (short q = 0; q < 4; ++q) {
            RF_BX(kb + 64 * q, kb + 64 * q + 4, kb + 64 * q + 32, kb + 64 * q + 36)
            RF_UNROLL for (short r = 0; r < R; ++r) {
                const float d = float(as_type<half>(Dd[r]));
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    const short G = 2 * q + gi, n = G >> 2, jj = G & 3, h = jj & 1, nb = 4 * (jj >> 1);
                    const half sc = half(d * float(SC[r][G]));
                    half2 a4[4];
                    RF_UNROLL for (short v = 0; v < 2; ++v) {
                        const uint lo = L[r][4 * n + 2 * h + v], hq = H[r][2 * n + v];
                        a4[2 * v]     = (as_type<half2>(((lo >> nb) & 0x000F000Fu)       | (((hq >> (2 * jj)) & 0x00030003u) << 4)       | 0x64006400u) - half2(1056.0h)) * sc;
                        a4[2 * v + 1] = (as_type<half2>(((lo >> (nb + 8)) & 0x000F000Fu) | (((hq >> (2 * jj + 8)) & 0x00030003u) << 4) | 0x64006400u) - half2(1056.0h)) * sc;
                    }
                    RF_MMA(r, gi)
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)
    }
    RF_KREDUCE
    RF_STORE
}
#undef LOADW

// Q8_0 (K map): block 34 B (2-aligned) = half d + int8 qs[32]; a 256-superblock is 8 blocks, one per 8-group G.
// Lane j owns qs bytes 8j..8j+7 of each block: two uints (b0..b3) (b4..b7) from four ushort loads.
// int8 -> half exactly: (b ^ 0x80) | 0x6400 is 1024 + b + 128, minus 1152.
kernel void kernel_mul_mm_rf_q8_0_f32(RF_ARGS) {
    // RF_HEAD, with split-K: sg / row_base from the threadgroup alone when ksplit > 1
    constexpr short R = RF_R;
    const int  N = args.ne11;
    const uint K = args.ne00;
    const uint64_t RB = args.nb01;
    const device uchar * w = (const device uchar *) src0;
    const uint lane = tiisg;
    const uint S    = FC_mul_mm_rf_ks && args.ksplit > 1 ? uint(args.ksplit) : 1u;
    const uint sg   = S > 1 ? tgpig : tgpig * ntg + sgitg;
    const short qid = lane / 4;
    const short fm  = (qid & 4) + ((lane / 2) % 4);
    const short fn  = (qid & 2) * 2 + (lane % 2) * 2;
    const short j   = fn / 2;
    const short eb  = fm & 1;
    const uint row_base = sg * (8 * R);
    const uint NBLK = K / 256;
    const uint kb0  = S > 1 ? (sgitg * NBLK) / S : 0, kb1 = S > 1 ? ((sgitg + 1) * NBLK) / S : NBLK;
    threadgroup float rf_red[3 * RF_R * 64];
    (void) tiitg;
    if (row_base >= (uint) args.ne01) {
        return;
    }
    RF_SETUP(8 * (fm >> 1) + 2 * eb)
    uint Q[R][8][2]; ushort Dd[R][8];
    #define LOADW(b) RF_UNROLL for (short r = 0; r < R; ++r) { RF_UNROLL for (short G = 0; G < 8; ++G) { \
        const device ushort *ps = (const device ushort *)(wb + r * 8 * RB + ((b) * 8 + G) * 34); Dd[r][G] = ps[0]; \
        Q[r][G][0] = (uint(ps[1 + 4 * j]) | (uint(ps[2 + 4 * j]) << 16)) ^ 0x80808080u; \
        Q[r][G][1] = (uint(ps[3 + 4 * j]) | (uint(ps[4 + 4 * j]) << 16)) ^ 0x80808080u; } }
    LOADW(kb0)
    for (uint blk = kb0; blk < kb1; ++blk) {
        const uint kb = blk * 256;
        RF_UNROLL for (short q = 0; q < 4; ++q) {
            RF_BX(kb + 64 * q, kb + 64 * q + 4, kb + 64 * q + 32, kb + 64 * q + 36)
            RF_UNROLL for (short r = 0; r < R; ++r) {
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) {
                    const short G = 2 * q + gi;
                    const half d = as_type<half>(Dd[r][G]);
                    half2 a4[4];
                    RF_UNROLL for (short v = 0; v < 2; ++v) {
                        const uint u = Q[r][G][v];
                        a4[2 * v]     = (as_type<half2>((u & 0x00FF00FFu)        | 0x64006400u) - half2(1152.0h)) * d;
                        a4[2 * v + 1] = (as_type<half2>(((u >> 8) & 0x00FF00FFu) | 0x64006400u) - half2(1152.0h)) * d;
                    }
                    RF_MMA(r, gi)
                }
            }
        }
        const uint bn = min(blk + 1, kb1 - 1);
        LOADW(bn)
    }
    RF_KREDUCE
    RF_STORE
}
#undef LOADW

// IQ3_S, two 110-byte blocks per iteration. With K/256 even, rows and block pairs are 4-byte
// aligned, so block A (even) and block B (odd) each have a fixed alignment and can use uint
// loads: 20 loads per pair instead of 30 ushort/byte loads. Block B's loads are issued before
// block A's MMA, block A of the next pair before block B's MMA.
#define RF_IQ3S_BLOCK(kb, Dd, QH, QS, SG, SC) { \
        half db[R][2]; \
        RF_UNROLL for (short r = 0; r < R; ++r) { \
            const float d = float(as_type<half>(Dd[r])); \
            db[r][0] = half(d * float(1 + 2 * (SC[r] & 0xF))); db[r][1] = half(d * float(1 + 2 * (SC[r] >> 4))); \
        } \
        RF_UNROLL for (short q = 0; q < 4; ++q) { \
            RF_BX((kb) + 16 * q, (kb) + 16 * q + 4, (kb) + 16 * q + 8, (kb) + 16 * q + 12) \
            RF_UNROLL for (short r = 0; r < R; ++r) { \
                RF_UNROLL for (short gi = 0; gi < 2; ++gi) { \
                    const short G = 2 * q + gi; \
                    const uint qs = QS[r][G], qh = QH[r]; \
                    const uint2 g0 = tgrid[(qs & 0xFF) | (((qh >> (2 * G)) & 1) << 8)]; \
                    const uint2 g1 = tgrid[(qs >> 8) | (((qh >> (2 * G + 1)) & 1) << 8)]; \
                    const uint sb = SG[r][G >> 1] >> (8 * (G & 1)); const uint4 m = RF_SMASK(sb); \
                    const half s = db[r][G >> 2]; \
                    half2 a4[4]; \
                    a4[0] = as_type<half2>(g0.x ^ m.x) * s; a4[1] = as_type<half2>(g0.y ^ m.y) * s; \
                    a4[2] = as_type<half2>(g1.x ^ m.z) * s; a4[3] = as_type<half2>(g1.y ^ m.w) * s; \
                    RF_MMA(r, gi) \
                } \
            } \
        } \
    }

kernel void kernel_mul_mm_rf_iq3_s_f32(RF_ARGS) {
    RF_HEAD_KS
    threadgroup uint2 tgrid[512];
    for (uint i = tiitg; i < 512; i += 32 * ntg) {
        const uint g = iq3s_grid[i];
        tgrid[i] = uint2(as_type<uint>(half2(half(g & 0xFF), half((g >> 16) & 0xFF))),
                        as_type<uint>(half2(half((g >> 8) & 0xFF), half(g >> 24))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_base >= (uint) args.ne01) {
        return;
    }
    RF_SETUP(64 * (fm >> 1) + 2 * eb)
    const uint NP = NBLK / 2;
    const uint pp0 = S > 1 ? (sgitg * NP) / S : 0, pp1 = S > 1 ? ((sgitg + 1) * NP) / S : NP;
    // block: d@0, qs[64]@2, qh[8]@66, signs[32]@74, scales[4]@106; pair base 4-aligned
    ushort aD[R], aQH[R], aQS[R][8], aSG[R][4]; uchar aSC[R];
    ushort bD[R], bQH[R], bQS[R][8], bSG[R][4]; uchar bSC[R];
    #define LOADA(pp) RF_UNROLL for (short r = 0; r < R; ++r) { const device uchar *p = wb + r * 8 * RB + (pp) * 220; \
        aD[r] = ((const device ushort *)p)[0]; aQH[r] = ((const device ushort *)p)[33 + j]; aSC[r] = p[106 + j]; \
        const device uint *pq = (const device uint *)(p + 16 * j); uint W[5]; RF_UNROLL for (short i = 0; i < 5; ++i) W[i] = pq[i]; \
        RF_UNROLL for (short u = 0; u < 8; ++u) aQS[r][u] = ushort(W[(u + 1) >> 1] >> (16 * ((u + 1) & 1))); \
        const device uint *pg = (const device uint *)(p + 72 + 8 * j); uint V[3]; RF_UNROLL for (short i = 0; i < 3; ++i) V[i] = pg[i]; \
        RF_UNROLL for (short u = 0; u < 4; ++u) aSG[r][u] = ushort(V[(u + 1) >> 1] >> (16 * ((u + 1) & 1))); }
    #define LOADB(pp) RF_UNROLL for (short r = 0; r < R; ++r) { const device uchar *p = wb + r * 8 * RB + (pp) * 220 + 110; \
        bD[r] = ((const device ushort *)p)[0]; bQH[r] = ((const device ushort *)p)[33 + j]; bSC[r] = p[106 + j]; \
        const device uint *pq = (const device uint *)(p + 2 + 16 * j); RF_UNROLL for (short i = 0; i < 4; ++i) { const uint t = pq[i]; bQS[r][2 * i] = ushort(t); bQS[r][2 * i + 1] = ushort(t >> 16); } \
        const device uint *pg = (const device uint *)(p + 74 + 8 * j); RF_UNROLL for (short i = 0; i < 2; ++i) { const uint t = pg[i]; bSG[r][2 * i] = ushort(t); bSG[r][2 * i + 1] = ushort(t >> 16); } }
    LOADA(pp0)
    for (uint pp = pp0; pp < pp1; ++pp) {
        LOADB(pp)
        RF_IQ3S_BLOCK(pp * 512, aD, aQH, aQS, aSG, aSC)
        const uint pn = min(pp + 1, pp1 - 1);
        LOADA(pn)
        RF_IQ3S_BLOCK(pp * 512 + 256, bD, bQH, bQS, bSG, bSC)
    }
    RF_KREDUCE
    RF_STORE
}
#undef LOADA
#undef LOADB
