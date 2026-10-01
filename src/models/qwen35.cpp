#include "models.h"
#include "../llama-kv-cache.h"
#include "llama-memory-recurrent.h"

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    // a head-less split tail (split-gguf.py --no-head) has neither token_embd nor output: residual in, final norm out
    const bool no_head = ml.get_weight(tn(LLM_TENSOR_TOKEN_EMBD, "weight").str().c_str()) == nullptr;
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, no_head ? TENSOR_NOT_REQUIRED : 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL && !no_head) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        // a split tail GGUF made with split-gguf.py --no-ffn has no FFN weights: the worker runs the FFN elsewhere
        // (llama_set_ffn_offload); build_layer_ffn refuses to run such a layer without the offload
        const int ffn_flags = ml.get_weight(tn(LLM_TENSOR_FFN_GATE, "weight", il).str().c_str()) ? flags : (flags | TENSOR_NOT_REQUIRED);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, ffn_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, ffn_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, ffn_flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params, int half) :
    llm_build_delta_net_base(params), model(model) {
    remote_half = half;
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();

    // split prefill: optionally run only layers [il_start, il_end) (see llama_set_layer_range)
    // il_start > 0: inpL is the residual stream fed through batch.embd
    const int il_start = cparams.layer_start;
    const int il_end   = cparams.layer_end < 0 ? (int) n_layer : std::min<int>(cparams.layer_end, n_layer);

    // a head-only graph has no output rows to gather (and an unused input would have no buffer)
    ggml_tensor * inp_out_ids = il_end == n_layer ? build_inp_out_ids() : nullptr;

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    ggml_tensor * piped = (il_start == 0 && il_end == (int) n_layer) ? build_pipelined(params, inp, inpL, inp_pos, sections) : nullptr;
    if (piped) {
        inpL = piped;
    } else {
        for (int il = il_start; il < il_end; ++il) {
            res->t_layer_inp[il] = inpL;
            inpL = build_block(inp->get_recr(), inp->get_attn(), inpL, inp_pos, sections, il, inp_out_ids);
        }
    }

    if (il_end < n_layer) {
        // head of a split: expose the raw residual leaving layer il_end-1, skip output norm + lm_head
        res->t_layer_inp[il_end] = inpL;
        ggml_build_forward_expand(gf, inpL);
        return;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (!model.output) {
        // head-less split tail: no logits
        ggml_build_forward_expand(gf, cur);
        return;
    }

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_qwen35::graph::build_block(
         llm_graph_input_rs * inp_recr,
    llm_graph_input_attn_kv * inp_attn,
                ggml_tensor * inpL,
                ggml_tensor * inp_pos,
                        int * sections,
                        int   il,
                ggml_tensor * inp_out_ids) {
    ggml_tensor * cur;
    ggml_tensor * inpSA = inpL;

    cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "attn_norm", il);

    ggml_build_forward_expand(gf, cur);

    // Determine layer type and build appropriate attention mechanism
    if (hparams.is_recr(il)) {
        // Linear attention layer (gated delta net)
        cur = build_layer_attn_linear(inp_recr, cur, il);
    } else {
        // Full attention layer
        cur = build_layer_attn(inp_attn, cur, inp_pos, sections, il);
    }

    if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
        cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
        inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
    }

    // Residual connection
    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "attn_residual", il);

    // Save the tensor before post-attention norm for residual connection
    ggml_tensor * ffn_residual = cur;

    // Post-attention norm
    ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(attn_post_norm, "attn_post_norm", il);

    // Dense FFN layer - without residual connection
    cur = build_layer_ffn(attn_post_norm, il);
    cb(cur, "ffn_out", il);

    // Residual connection for FFN - add to the tensor from before post_attention_layernorm
    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "post_ffn", il);

    cur = build_cvec(cur, il);
    cb(cur, "l_out", il);

    return cur;
}

// infernet pipelined prefill (LLAMA_REMOTE_PIPE=1, phone-held KV): the ubatch runs as two halves A and B, staggered by one
// attention layer. Each half's attention layer sends its whole half to the phone (ATTN_BIG) and the half stops there; the
// other half's layers are built next, and a half's phone merge waits until a node reads that half's attention output
// (Metal backend, op_params[8]). So the Mac computes one half's delta-net / FFN layers while the phone computes the other
// half's old keys, instead of idling. Same per-half math as a ubatch of n/2 tokens (B's attention and delta-net read A's
// KV and state, written first in graph order).
ggml_tensor * llama_model_qwen35::graph::build_pipelined(
    const llm_graph_params & params,
   llm_graph_input_mem_hybrid * inp,
                ggml_tensor * inpL,
                ggml_tensor * inp_pos,
                        int * sections) {
    static const bool on = getenv("LLAMA_REMOTE_PIPE") && atoi(getenv("LLAMA_REMOTE_PIPE")) != 0;
    llm_graph_input_attn_kv * ia = inp->get_attn();
    if (!on || !ia || !ia->mctx || !ia->mctx->remote_big() || !ia->mctx->remote_tag_any() || cparams.embeddings_nextn_masked ||
        ubatch.n_seqs != 1 || !ubatch.equal_seqs() || n_tokens < 64 || n_tokens % 16 != 0 || n_tokens > 1024) {
        return nullptr;
    }
    const int64_t H = n_tokens / 2;
    if (ia->self_v_idxs->ne[0] != n_tokens || ia->self_k_idxs->ne[0] != n_tokens || ia->get_kq_mask()->ne[2] != 1) {
        return nullptr;   // transposed V / multi-stream: not handled
    }

    llm_graph_input_rs * ir = inp->get_recr();
    int64_t R = 0, W = 0;
    if (ir->rp_idx_conv) {   // GDN replay rollback: each half gets its own gather indices (eager halves: H > R)
        R = ir->mctx->get_rp_cap();
        W = ir->rp_idx_conv->ne[0] - R - n_tokens;
        if (H <= R) {
            return nullptr;
        }
    }

    struct half_state {
        std::unique_ptr<graph> g;
        std::unique_ptr<llm_graph_input_attn_kv> attn;
        std::unique_ptr<llm_graph_input_rs> recr;
        ggml_tensor * cur = nullptr;
        ggml_tensor * pos = nullptr;
        int il = 0;
    };
    // the half builders keep references into their params (the ubatch): these outlive them (both die with this call)
    std::unique_ptr<llm_graph_params> hp[2];
    half_state hs[2];
    for (int h = 0; h < 2; h++) {
        hp[h] = std::make_unique<llm_graph_params>(params);
        hp[h]->ubatch.n_tokens     = (uint32_t) H;
        hp[h]->ubatch.n_seq_tokens = (uint32_t) H;
        hp[h]->n_outputs           = 0;
        hs[h].g = std::make_unique<graph>(model, *hp[h], h);

        auto a = std::make_unique<llm_graph_input_attn_kv>(*ia);
        a->self_k_idxs = ggml_view_1d(ctx0, ia->self_k_idxs, H, h*H*ia->self_k_idxs->nb[0]);
        a->self_v_idxs = ggml_view_1d(ctx0, ia->self_v_idxs, H, h*H*ia->self_v_idxs->nb[0]);
        ggml_tensor * m = ia->get_kq_mask();
        a->self_kq_mask_cnv = ggml_view_4d(ctx0, m, m->ne[0], H, 1, 1, m->nb[1], m->nb[2], m->nb[3], h*H*m->nb[1]);
        a->self_kq_mask     = a->self_kq_mask_cnv;
        hs[h].attn = std::move(a);

        hs[h].recr = std::make_unique<llm_graph_input_rs>(*ir);
        if (ir->rp_idx_conv) {
            auto rp = std::make_unique<llm_graph_input_rp_half>(ir->mctx, h == 0);
            rp->rp_idx_conv = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, W + R + H);
            ggml_set_input(rp->rp_idx_conv);
            ggml_set_name(rp->rp_idx_conv, h == 0 ? "rp_idx_conv_a" : "rp_idx_conv_b");
            rp->rp_idx_gb = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, R + H);
            ggml_set_input(rp->rp_idx_gb);
            ggml_set_name(rp->rp_idx_gb, h == 0 ? "rp_idx_gb_a" : "rp_idx_gb_b");
            hs[h].recr->rp_idx_conv = rp->rp_idx_conv;
            hs[h].recr->rp_idx_gb   = rp->rp_idx_gb;
            res->add_input(std::move(rp));
        }

        hs[h].cur = ggml_view_2d(ctx0, inpL, inpL->ne[0], H, inpL->nb[1], h*H*inpL->nb[1]);
        const int64_t npe = inp_pos->ne[0] / n_tokens;   // positions per token (M-RoPE: 4 blocks of n_tokens)
        ggml_tensor * pv = ggml_view_2d(ctx0, inp_pos, H, npe, n_tokens*ggml_element_size(inp_pos), h*H*ggml_element_size(inp_pos));
        hs[h].pos = ggml_reshape_1d(ctx0, ggml_cont(ctx0, pv), H*npe);
    }
    res->set_params(params);   // the half builders' constructors set theirs

    // one stage of a half: its layers up to and including the next attention layer (whose phone call then runs while the
    // other half's stage is computed)
    // layer-input taps (the drafter reads target hidden states): each half's input to a tapped layer, joined at the end
    const auto & taps = cparams.embeddings_layer_inp;
    std::vector<ggml_tensor *> tap[2];
    tap[0].assign(n_layer, nullptr); tap[1].assign(n_layer, nullptr);
    auto step = [&](half_state & x) {
        while (x.il < (int) n_layer) {
            if ((size_t) x.il < taps.size() && taps[x.il]) {
                tap[&x - hs][x.il] = x.cur;
            }
            const bool attn = !hparams.is_recr(x.il);
            x.cur = x.g->build_block(x.recr.get(), x.attn.get(), x.cur, x.pos, sections, x.il, nullptr);
            x.il++;
            if (attn) {
                break;
            }
        }
    };
    while (hs[0].il < (int) n_layer || hs[1].il < (int) n_layer) {
        step(hs[0]);
        step(hs[1]);
    }
    for (int il = 0; il < (int) n_layer; il++) {
        if (tap[0][il] && tap[1][il]) {
            res->t_layer_inp[il] = ggml_concat(ctx0, tap[0][il], tap[1][il], 1);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }
    }
    return ggml_concat(ctx0, hs[0].cur, hs[1].cur, 1);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    // infernet round-cost: put the z projection next to the other in-proj matmuls in graph order; otherwise its only use
    // (the output gate silu) pulls it in between the output norm and the gate, and Metal cannot fuse the gated norm
    // (GGML_METAL_FUSION_NORM_GATE). Node order only.
    ggml_build_forward_expand(gf, z);

    // infernet: LLAMA_GDN_AB_FUSE=1 - when the ssm_beta and ssm_alpha weights (f32 in IQ4_XS, bf16 in IQ3_S) [n_embd, n_v_heads] sit back to back
    // in one weight buffer (consecutive tensors, no padding), one mul_mat against an alias tensor spanning both
    // replaces two (96 -> 48 small launch-bound matmuls per 8-token verify). Same dot products; stock path unchanged.
    ggml_tensor * beta  = nullptr;
    ggml_tensor * alpha = nullptr;
    ggml_tensor * beta_raw = nullptr; // the beta projection before the sigmoid (GDN replay prep takes it raw)
    {
        const char * fuse_env = getenv("LLAMA_GDN_AB_FUSE");
        const ggml_tensor * wb = model.layers[il].ssm_beta;
        const ggml_tensor * wa = model.layers[il].ssm_alpha;
        const bool can_fuse = fuse_env != nullptr && atoi(fuse_env) != 0 &&
            wb->type == wa->type && ggml_are_same_shape(wb, wa) &&
            ggml_is_contiguous(wb) && ggml_is_contiguous(wa) && wb->buffer != nullptr && wb->buffer == wa->buffer &&
            wb->view_src == nullptr && wa->view_src == nullptr &&
            model.layers[il].ssm_beta_s == nullptr && model.layers[il].ssm_alpha_s == nullptr &&
            (loras == nullptr || loras->empty());
        const bool b_first = can_fuse && (const char *) wb->data + ggml_nbytes(wb) == (const char *) wa->data;
        const bool a_first = can_fuse && (const char *) wa->data + ggml_nbytes(wa) == (const char *) wb->data;
        if (can_fuse && !b_first && !a_first) {
            static bool logged_nf = false;
            if (!logged_nf) {
                logged_nf = true;
                LLAMA_LOG_WARN("%s: LLAMA_GDN_AB_FUSE: ssm_beta/ssm_alpha not adjacent, not fused\n", __func__);
            }
        }
        if (b_first || a_first) {
            const ggml_tensor * w0 = b_first ? wb : wa;
            // alias [n_embd, 2*n_v_heads] over both weights (pre-allocated: the graph allocator leaves it alone)
            ggml_tensor * w_ab = ggml_new_tensor_2d(ctx0, wb->type, wb->ne[0], 2 * wb->ne[1]);
            w_ab->data   = w0->data;
            w_ab->buffer = w0->buffer;
            ggml_format_name(w_ab, "ssm_ab_alias-%d", il);
            static bool logged = false;
            if (!logged) {
                logged = true;
                LLAMA_LOG_INFO("%s: LLAMA_GDN_AB_FUSE: fused ssm_beta/ssm_alpha projection (%s first in memory)\n", __func__, b_first ? "beta" : "alpha");
            }

            ggml_tensor * ab = ggml_mul_mat(ctx0, w_ab, cur); // [2*n_v_heads, n_tokens]
            cb(ab, "beta_alpha", il);
            const size_t off_b = b_first ? 0 : num_v_heads * ggml_element_size(ab);
            const size_t off_a = b_first ? num_v_heads * ggml_element_size(ab) : 0;

            beta = ggml_view_2d(ctx0, ab, num_v_heads, ab->ne[1], ab->nb[1], off_b);
            beta_raw = beta;
            beta = ggml_sigmoid(ctx0, beta); // strided rows in, contiguous out
            beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
            cb(beta, "beta_sigmoid", il);

            // strided view; its only consumer (the ssm_dt add below) writes a contiguous result
            alpha = ggml_view_3d(ctx0, ab, num_v_heads, n_seq_tokens, n_seqs, ab->nb[1], ab->nb[1] * n_seq_tokens, off_a);
            cb(alpha, "alpha", il);
        }
    }
    if (beta == nullptr) {
    beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta_raw = beta;
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);
    }

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * output = nullptr;

    if (mctx_cur->get_rp_cap() > 0) {
        // infernet GDN replay rollback (docs/replay-rollback.md)
        output = build_layer_attn_linear_replay(inp, qkv_mixed, gate, beta, alpha, beta_raw, il);
    } else {
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);
    }

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

// infernet GDN replay rollback (docs/replay-rollback.md §3.3)
//
// The cell keeps one committed state X, the conv window before X and a log of the last lazy ubatch's raw in-proj
// output, g and beta. The first rp_c logged tokens are committed but not yet applied to X: they are replayed here,
// in front of the N new tokens, from the logged values (no matmul is recomputed, so the replay is bit-exact).
// To keep one graph shape for any rp_c, the replay part always has R tokens: (R - rp_c) identity pads (g = beta = 0)
// then the rp_c logged ones. The gathers are driven by the rp_idx_* inputs.
//   lazy  (N <= R): X := state after the replay; the N new tokens go to the log (rolled back by seq_rm, or replayed next)
//   eager (N >  R): X := state after all tokens; nothing is logged
ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear_replay(
        llm_graph_input_rs * inp,
        ggml_tensor *        qkv_mixed,
        ggml_tensor *        gate,
        ggml_tensor *        beta,
        ggml_tensor *        alpha_raw,
        ggml_tensor *        beta_raw,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner     = hparams.ssm_d_inner;
    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = d_inner / num_v_heads;

    const int64_t N = ubatch.n_seq_tokens;
    const int64_t R = mctx_cur->get_rp_cap();
    const int64_t T = R + N; // tokens through the delta net: R replayed (incl. pads) + N new
    const bool lazy = N <= R;

    GGML_ASSERT(ubatch.n_seqs == 1 && "GDN replay rollback supports one sequence per ubatch");

    ggml_tensor * conv_kernel = model.layers[il].ssm_conv1d;

    const int64_t W = conv_kernel->ne[0] - 1;                           // conv window
    const int64_t C = d_inner + 2 * num_k_heads * head_k_dim;           // conv channels
    const int64_t H = num_v_heads;

    ggml_tensor * r_all = mctx_cur->get_r_l(il);
    ggml_tensor * s_all = mctx_cur->get_s_l(il);

    GGML_ASSERT(inp->rp_idx_conv != nullptr && inp->rp_idx_conv->ne[0] == W + R + N);
    GGML_ASSERT(inp->rp_idx_gb   != nullptr && inp->rp_idx_gb->ne[0]   == R + N);
    GGML_ASSERT(r_all->type == GGML_TYPE_F32 && qkv_mixed->type == GGML_TYPE_F32);
    GGML_ASSERT(r_all->ne[0] == (W + R) * C + 2 * H * (R + 1));
    GGML_ASSERT(qkv_mixed->ne[0] == C);

    const auto   kv_head = mctx_cur->get_head();
    const size_t fs      = sizeof(float);

    // the cell's r row (conv window + logs), read from its source cell like any recurrent state; this is a copy,
    // so the log writes below cannot race with the reads
    ggml_tensor * rrow = build_rs(inp, r_all, (int32_t) r_all->ne[0], 1);
    cb(rrow, "rp_row", il);

    ggml_tensor * pool_old = ggml_view_2d(ctx0, rrow, C, W + R, C * fs, 0);
    ggml_tensor * g_log    = ggml_view_2d(ctx0, rrow, H, R + 1, H * fs, ((W + R) * C) * fs);
    ggml_tensor * b_log    = ggml_view_2d(ctx0, rrow, H, R + 1, H * fs, ((W + R) * C + H * (R + 1)) * fs);

    ggml_tensor * x_new = ggml_reshape_2d(ctx0, qkv_mixed, C, N);
    ggml_tensor * g_new = nullptr;
    ggml_tensor * b_new = nullptr;

    const int64_t commit_at = lazy ? R : T;

    // infernet round-cost: LLAMA_GDN_PREP_FUSE (default on) replaces the ~12 small launches below (concat, gather, transpose,
    // conv, silu, the g/beta activations and their concat + gather) with one GGML_OP_GDN_REPLAY_PREP. Same arithmetic, same
    // result bits on Metal; =0 builds the unfused graph.
    static const bool prep_fuse = [] {
        const char * e = getenv("LLAMA_GDN_PREP_FUSE");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool use_prep = prep_fuse && alpha_raw != nullptr && beta_raw != nullptr && conv_kernel->ne[0] == 4 &&
        W + T <= 48 && ggml_nelements(alpha_raw) == H * N && ggml_nelements(beta_raw) == H * N &&
        ggml_is_contiguous(alpha_raw) && // the LLAMA_GDN_AB_FUSE alpha is a strided view: reshape would assert

        ggml_is_contiguous(qkv_mixed) && model.layers[il].ssm_dt != nullptr && model.layers[il].ssm_a != nullptr;

    ggml_tensor * conv_out = nullptr;
    ggml_tensor * gath     = nullptr; // unfused only
    ggml_tensor * win      = nullptr; // the conv window before X (W gathered columns from commit_at)
    ggml_tensor * g_all    = nullptr;
    ggml_tensor * b_all    = nullptr;

    if (use_prep) {
        ggml_tensor * prep = ggml_gdn_replay_prep(ctx0, rrow, x_new,
                ggml_reshape_2d(ctx0, alpha_raw, H, N), beta_raw,
                model.layers[il].ssm_dt, model.layers[il].ssm_a, conv_kernel, inp->rp_idx_conv, inp->rp_idx_gb, R, commit_at);
        cb(prep, "gdn_replay_prep", il);
        conv_out = ggml_view_3d(ctx0, prep, C, T, 1, C * fs, C * T * fs, 0);
        g_all    = ggml_view_2d(ctx0, prep, H, T, H * fs, (C * T) * fs);
        b_all    = ggml_view_2d(ctx0, prep, H, T, H * fs, (C * T + H * T) * fs);
        win      = ggml_view_2d(ctx0, prep, C, W, C * fs, (C * T + 2 * H * T) * fs);
        // the N new rows are the last N rows of g_all / b_all (rp_idx_gb ends with R + 1 .. R + N)
        g_new    = ggml_view_2d(ctx0, prep, H, N, H * fs, (C * T + H * (T - N)) * fs);
        b_new    = ggml_view_2d(ctx0, prep, H, N, H * fs, (C * T + H * T + H * (T - N)) * fs);
    } else {
    g_new = ggml_reshape_2d(ctx0, gate, H, N);
    b_new = ggml_reshape_2d(ctx0, beta, H, N);

    // conv input: [(R - c) pads | window (W) | c logged | N new], time-major -> [W + T, C]
    ggml_tensor * pool = ggml_concat(ctx0, pool_old, x_new, 1);             // [C, W + R + N]
    gath = ggml_get_rows(ctx0, pool, inp->rp_idx_conv);                      // [C, W + T]
    cb(gath, "rp_conv_gather", il);

    ggml_tensor * conv_in = ggml_cont(ctx0, ggml_transpose(ctx0, gath));    // [W + T, C]
    conv_in = ggml_reshape_3d(ctx0, conv_in, W + T, C, 1);

    conv_out = ggml_ssm_conv(ctx0, conv_in, conv_kernel);                    // [C, T, 1]
    cb(conv_out, "conv_output_raw", il);
    conv_out = ggml_silu(ctx0, conv_out);
    cb(conv_out, "conv_output_silu", il);

    // g / beta: [(R - c) zero rows | c logged | N new]
    g_all = ggml_get_rows(ctx0, ggml_concat(ctx0, g_log, g_new, 1), inp->rp_idx_gb); // [H, T]
    b_all = ggml_get_rows(ctx0, ggml_concat(ctx0, b_log, b_new, 1), inp->rp_idx_gb); // [H, T]

    win = ggml_view_2d(ctx0, gath, C, W, gath->nb[1], commit_at * gath->nb[1]);
    }

    const size_t nb1_qkv = ggml_row_size(conv_out->type, C);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, T, 1,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * T, 0);
    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_out, head_k_dim, num_k_heads, T, 1,
            ggml_row_size(conv_out->type, head_k_dim), nb1_qkv, nb1_qkv * T,
            head_k_dim * num_k_heads * ggml_element_size(conv_out));
    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_out, head_v_dim, num_v_heads, T, 1,
            ggml_row_size(conv_out->type, head_v_dim), nb1_qkv, nb1_qkv * T,
            ggml_row_size(conv_out->type, 2 * head_k_dim * num_k_heads));

    // infernet round-cost: LLAMA_GDN_L2_FUSE=1 (default OFF: measured loss) - the recurrence L2-normalizes q and k on load
    // (ggml_gated_delta_net_set_l2, bit-identical to build_gdn_l2_norm): -96 launches, -47 barriers per verify, but the
    // verify gets ~2 ms SLOWER (89.2 -> 91.2 ms, 3 ABAB; GDN kernel alone 3.02 -> 3.64 ms serialized, more in the graph)
    static const bool l2_fuse = [] {
        const char * e = getenv("LLAMA_GDN_L2_FUSE");
        return e != nullptr && atoi(e) != 0;
    }();
    const bool use_l2 = l2_fuse && head_k_dim == 128 && head_v_dim == 128;

    const float eps_norm = hparams.f_norm_rms_eps;
    if (!use_l2) {
    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);
    }

    g_all = ggml_reshape_4d(ctx0, g_all, 1, H, T, 1);
    b_all = ggml_reshape_4d(ctx0, b_all, 1, H, T, 1);

    // infernet round-cost: LLAMA_GDN_STATE_ROWS (default on) lets the recurrence read the state straight from its cache
    // cell (ggml_gated_delta_net_replay_rows) instead of from a gathered 3.1 MB copy: one launch + barrier less per GDN layer.
    // Only with one cell in the ubatch (n_rs == 1): then build_rs copies no extra cells, so nothing else writes the cache
    // between the zeroing of a fresh cell and the recurrence's read. Same arithmetic, same result bits.
    static const bool state_rows = [] {
        const char * e = getenv("LLAMA_GDN_STATE_ROWS");
        return e == nullptr || atoi(e) != 0;
    }();

    ggml_tensor * gdn = nullptr;
    if (state_rows && mctx_cur->get_n_rs() == 1) {
        ggml_tensor * state_ids = nullptr;
        ggml_tensor * states = build_rs(inp, s_all, hparams.n_embd_s(), 1,
                [&state_ids](ggml_context *, ggml_tensor * st, ggml_tensor * ids) { state_ids = ids; return st; });
        gdn = ggml_gated_delta_net_replay_rows(ctx0, q_conv, k_conv, v_conv, g_all, b_all, states, state_ids, R, commit_at);
    } else {
    ggml_tensor * state = build_rs(inp, s_all, hparams.n_embd_s(), 1);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, 1);
    cb(state, "state_predelta", il);

    gdn = ggml_gated_delta_net_replay(ctx0, q_conv, k_conv, v_conv, g_all, b_all, state, R, commit_at);
    }
    if (use_l2) {
        ggml_gated_delta_net_set_l2(gdn, eps_norm / (float) head_k_dim);
    }
    res->add_fused_node({LLM_FUSED_OP_GDN_CH, gdn, il});

    ggml_tensor * output = ggml_view_4d(ctx0, gdn, head_v_dim, num_v_heads, N, 1,
            ggml_row_size(gdn->type, head_v_dim),
            ggml_row_size(gdn->type, head_v_dim * num_v_heads),
            ggml_row_size(gdn->type, head_v_dim * num_v_heads * N), 0);
    cb(output, "attn_output", il);

    const size_t row_r = r_all->nb[1];

    // X: the same [D, n_seqs, 1] views as build_recurrent_attn, so Metal's GDN_CACHE fusion can write it straight to the cache
    const int64_t D        = hparams.n_embd_s();
    const size_t  row_s    = s_all->nb[1];
    const auto    mem_size = mctx_cur->get_size();
    ggml_tensor * new_state = ggml_view_3d(ctx0, gdn, D, 1, 1,
            ggml_row_size(gdn->type, D), ggml_row_size(gdn->type, D),
            ggml_row_size(gdn->type, head_v_dim * num_v_heads * N));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state,
            ggml_view_3d(ctx0, s_all, D, 1, 1, row_s, (size_t) mem_size * row_s, kv_head * row_s)));

    // the conv window before X: the W inputs in front of token commit_at
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, win,
            ggml_view_2d(ctx0, r_all, C, W, C * fs, kv_head * row_r)));

    if (lazy) {
        // log the new tokens: raw in-proj output, g, beta (rows beyond N and the zero rows are left alone)
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, x_new,
                ggml_view_2d(ctx0, r_all, C, N, C * fs, kv_head * row_r + (W * C) * fs)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, g_new,
                ggml_view_2d(ctx0, r_all, H, N, H * fs, kv_head * row_r + ((W + R) * C) * fs)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, b_new,
                ggml_view_2d(ctx0, r_all, H, N, H * fs, kv_head * row_r + ((W + R) * C + H * (R + 1)) * fs)));
    }

    return output;
}

// FFN offload (llama_set_ffn_offload): runs on a CPU thread; the scheduler moves the input off the GPU and back
static void llama_ffn_offload_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    const auto * s = (const llama_ffn_offload_slot *) userdata;
    GGML_ASSERT(a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(a) && ggml_is_contiguous(dst));
    if (!s->fn) {
        // a layer without FFN weights and without an offload: no result (llama_ffn_offload_failed reports it)
        memset(dst->data, 0, ggml_nbytes(dst));
        *s->failed = true;
        return;
    }
    if (!s->fn((float *) dst->data, (const float *) a->data, (int32_t) a->ne[0], (int32_t) ggml_nrows(a), s->il, s->user)) {
        *s->failed = true;
    }
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    // offloaded layers, and layers of a GGUF without FFN weights (split-gguf.py --no-ffn), which must be offloaded
    const bool no_ffn = model.layers[il].ffn_up == nullptr;
    if (cparams.ffn_off_slots && ((il >= cparams.ffn_off_il0 && il < cparams.ffn_off_il1) || no_ffn)) {
        cur = ggml_map_custom1(ctx0, cur, llama_ffn_offload_op, 1, &cparams.ffn_off_slots[il]);
        cb(cur, "ffn_out", il);
        return cur;
    }

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
