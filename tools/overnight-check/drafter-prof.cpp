// drafter-prof (infernet, 2026-09-23): host-side cost of the DFlash2 drafter's alternating inject / draft decodes,
// CPU backend only (no GPU). Mirrors common/speculative.cpp's DFlash2 context setup: embeddings_nextn on (unmasked),
// non-causal attention, inject batches of target features [n_embd_enc] and draft blocks [id_last, mask x n].
//
//   drafter-prof DRAFT.gguf TARGET.gguf [rounds] [cpu|gpu]
// (the target is only mmap'd for its tok_embd / lm_head, which DFlash shares via ctx_other; it never decodes)
//
// Prints ms per inject / draft decode (wall, incl. compute) and a checksum of the draft lattice so two builds
// (e.g. LLAMA_GRAPH_REUSE2=0/1) can be compared for identical output. Use LLAMA_DECODE_PROF=1 for the host split.

#include "llama.h"
#include "../../src/llama-ext.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s DRAFT.gguf TARGET.gguf [rounds]\n", argv[0]); return 2; }
    const int rounds     = argc > 3 ? atoi(argv[3]) : 40;
    const int NV = 8;  // verify rows injected per round
    const int NB = 8;  // draft block: id_last + 7 masks

    const bool gpu = argc > 4 && std::string(argv[4]) == "gpu";

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_devs[1] = { nullptr };
    if (!gpu) {
        mp.devices = no_devs; // CPU only
        mp.n_gpu_layers = 0;
    }
    llama_model * model_tgt = llama_model_load_from_file(argv[2], mp);
    if (!model_tgt) { fprintf(stderr, "target load failed\n"); return 1; }
    llama_context_params cpt = llama_context_default_params();
    cpt.n_ctx = 256; cpt.n_batch = 64; cpt.n_ubatch = 64; cpt.no_perf = true;
    llama_context * ctx_tgt = llama_init_from_model(model_tgt, cpt);
    if (!ctx_tgt) { fprintf(stderr, "target ctx failed\n"); return 1; }
    const int n_embd_tgt = llama_model_n_embd(model_tgt);

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

    const int n_layers_tgt = (int) llama_model_target_layer_ids_n(model);
    const int n_embd_enc   = n_layers_tgt * n_embd_tgt;
    const int n_embd_dec   = llama_model_n_embd(model);
    const bool is_mrope    = llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE;
    const llama_token mask = llama_vocab_mask(llama_model_get_vocab(model));

    // two drafter contexts on the same target: [0] stock, [1] LLAMA_GRAPH_REUSE2=1 (read at context creation)
    llama_context * ctx[2];
    for (int c = 0; c < 2; ++c) {
        if (c == 1) setenv("LLAMA_GRAPH_REUSE2", "1", 1); else unsetenv("LLAMA_GRAPH_REUSE2");
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096; cp.n_batch = 64; cp.n_ubatch = 64; cp.no_perf = true;
        cp.n_outputs_max = NB;
        cp.ctx_other = ctx_tgt;
        ctx[c] = llama_init_from_model(model, cp);
        if (!ctx[c]) { fprintf(stderr, "draft ctx failed\n"); return 1; }
        llama_set_embeddings_nextn(ctx[c], true, /*masked*/ false);
        llama_set_causal_attn(ctx[c], false);
    }
    unsetenv("LLAMA_GRAPH_REUSE2");
    printf("n_embd_enc %d (%d target layers), n_embd_dec %d, mrope %d, mask %d, %s\n", n_embd_enc, n_layers_tgt, n_embd_dec, is_mrope, mask, gpu ? "GPU" : "CPU");

    llama_batch bi = llama_batch_init(64, n_embd_enc, 1);
    if (is_mrope) { free(bi.pos); bi.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * 64); }
    llama_batch bd = llama_batch_init(64, 0, 1);

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // per context: host ms inside llama_decode (pre + build + inputs + encode/submit), wall ms incl. sync
    double h_inj[2] = {0, 0}, h_dft[2] = {0, 0}, w_inj[2] = {0, 0}, w_dft[2] = {0, 0};
    int n_meas = 0, n_diff = 0;
    double checksum[2] = {0, 0};
    llama_pos p = 0;
    llama_token id_last = 1000;
    std::vector<float> lat0((size_t) NB * n_embd_dec);
    for (int r = 0; r < rounds; ++r) {
        for (int i = 0; i < NV; ++i) {
            for (int k = 0; k < n_embd_enc; ++k) bi.embd[(size_t) i * n_embd_enc + k] = 0.05f * nd(rng);
        }
        const int acc = 1 + (r * 5) % NV;
        llama_token id_next = id_last;
        for (int k = 0; k < 2; ++k) {
            const int c = (r + k) % 2; // alternate which context goes first
            // inject NV rows at p .. p+NV-1
            bi.n_tokens = NV;
            for (int i = 0; i < NV; ++i) {
                bi.pos[i] = p + i;
                if (is_mrope) { bi.pos[NV + i] = p + i; bi.pos[2 * NV + i] = p + i; bi.pos[3 * NV + i] = 0; }
                bi.n_seq_id[i] = 1; bi.seq_id[i][0] = 0; bi.logits[i] = false;
            }
            double t0 = now_ms();
            if (llama_decode(ctx[c], bi) != 0) { fprintf(stderr, "inject failed\n"); return 1; }
            double t1 = now_ms();
            llama_synchronize(ctx[c]);
            double t2 = now_ms();
            if (r > 1) { h_inj[c] += t1 - t0; w_inj[c] += t2 - t0; }

            llama_memory_seq_rm(llama_get_memory(ctx[c]), 0, p + acc, -1); // rejected verify rows

            bd.n_tokens = NB;
            for (int i = 0; i < NB; ++i) {
                bd.token[i] = i == 0 ? id_last : mask; bd.pos[i] = p + acc + i;
                bd.n_seq_id[i] = 1; bd.seq_id[i][0] = 0; bd.logits[i] = false;
            }
            t0 = now_ms();
            if (llama_decode(ctx[c], bd) != 0) { fprintf(stderr, "draft failed\n"); return 1; }
            t1 = now_ms();
            llama_synchronize(ctx[c]);
            t2 = now_ms();
            if (r > 1) { h_dft[c] += t1 - t0; w_dft[c] += t2 - t0; }
            const float * lat = llama_get_embeddings_nextn(ctx[c]);
            if (!lat) { fprintf(stderr, "no lattice\n"); return 1; }
            for (int i = 0; i < NB * n_embd_dec; i += 97) checksum[c] += lat[i];
            if (c == 0) {
                std::memcpy(lat0.data(), lat, lat0.size() * sizeof(float));
                id_next = (llama_token) (1000 + ((int) std::fabs(lat[5]) * 7919) % 50000);
            }
            if (k == 1) { // both done: compare the full lattice
                const float * other = c == 0 ? nullptr : lat;
                if (other && std::memcmp(other, lat0.data(), lat0.size() * sizeof(float)) != 0) n_diff++;
            }
            llama_memory_seq_rm(llama_get_memory(ctx[c]), 0, p + acc, -1); // drop the draft block
        }
        if (r > 1) n_meas++;
        id_last = id_next;
        p += acc;
    }
    for (int c = 0; c < 2; ++c) {
        printf("%s: inject host %.3f wall %.2f | draft host %.3f wall %.2f ms/decode (n=%d) checksum %.9g\n",
               c == 0 ? "stock " : "reuse2", h_inj[c] / n_meas, w_inj[c] / n_meas, h_dft[c] / n_meas, w_dft[c] / n_meas, n_meas, checksum[c]);
    }
    printf("lattice differs in %d rounds (compared when ctx 1 ran second)\n", n_diff);
    fflush(stdout);
    llama_batch_free(bi); llama_batch_free(bd);
    for (int c = 0; c < 2; ++c) llama_free(ctx[c]);
    llama_model_free(model);
    llama_free(ctx_tgt); llama_model_free(model_tgt);
    return 0;
}
