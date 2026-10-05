// infernet split prefill inside llama_decode: see llama-split.h
#include "llama-split.h"

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"
#include "llama-ext.h"

#if defined(__APPLE__) || defined(__linux__)
#define LLAMA_SPLIT_HAVE_SOCKETS 1
#include "../tools/split-prefill/tail-client.h"
#endif

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>

static std::mutex g_split_mu;
static std::map<const llama_memory_i *, llama_split_state *> g_split_reg;

void llama_split_register(const llama_memory_i * mem, llama_split_state * st) {
    std::lock_guard<std::mutex> lk(g_split_mu);
    g_split_reg[mem] = st;
}

void llama_split_unregister(const llama_memory_i * mem) {
    std::lock_guard<std::mutex> lk(g_split_mu);
    g_split_reg.erase(mem);
}

void llama_split_mem_event(const llama_memory_i * mem, llama_split_event ev, llama_seq_id seq, llama_pos p0) {
    std::lock_guard<std::mutex> lk(g_split_mu);
    auto it = g_split_reg.find(mem);
    if (it == g_split_reg.end()) {
        return;
    }
    llama_split_state * st = it->second;
    if (st->sd) {
        // split decode: the worker holds the only copy of layers >= L, which can't be rewound. Any edit below its end makes
        // the next decode TRIM it to 0, and that decode must then start at position 0 (a full re-prefill)
        const llama_pos W_new = ev == LLAMA_SPLIT_EV_RM ? (seq <= 0 ? std::min(st->W, std::max<llama_pos>(0, p0)) : st->W) : 0;
        if (W_new < st->W) {
            static const char * names[] = { "seq_rm", "clear / seq_add / seq_div / state load", "recurrent state load" };
            LLAMA_LOG_WARN("%s: split decode: %s at position %d, the phone holds [0, %d): both sides reset, the next decode re-prefills from 0\n",
                           __func__, names[ev], ev == LLAMA_SPLIT_EV_RM ? std::max<llama_pos>(0, p0) : 0, st->W);
            st->W = W_new;
            st->sd_n_resets++;
        }
        return;
    }
    if (st->active) {
        return;   // our own state reads/writes during a split call
    }
    switch (ev) {
        case LLAMA_SPLIT_EV_RM:
            if (seq <= 0) {
                st->W = std::min(st->W, std::max<llama_pos>(0, p0));
            }
            st->rs_clean = false;
            break;
        case LLAMA_SPLIT_EV_RESET:
            st->W = 0;
            st->rs_clean = false;
            break;
        case LLAMA_SPLIT_EV_RS_DIRTY:
            st->rs_clean = false;
            break;
    }
}

llama_split_state::~llama_split_state() {
    if (push_thread.joinable()) {
        push_thread.join();   // before tc goes away
    }
    if (warm_thread.joinable()) {
        { std::lock_guard<std::mutex> lk(warm_mu); warm_quit = true; }
        warm_cv.notify_one();
        warm_thread.join();
    }
}

// split decode GPU warm (LLAMA_SPLIT_GPU_WARM_US). The A18 Pro GPU clocks down within a few ms of idle, and the ~160 ms per token
// it waits for the phone is plenty: the head pass then runs at ~1.1 GHz on average instead of ~1.4. Measured (Mac head-only bench,
// L=20 IQ2_XS, 160 ms gap per token like the phone's, 200 tokens): no warm 98-101 ms mean / 135-142 p95; a dispatch every 1 ms
// 88.5 / 95; every 5 ms 127 / 168 (worse: the governor reads that as low load). 2026-10-03.
static void split_warm_start(llama_split_state & st, int us) {
    void (*warm)(void) = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_count() && !warm; ++i) {
        warm = (void (*)(void)) ggml_backend_reg_get_proc_address(ggml_backend_reg_get(i), "ggml_backend_metal_gpu_warm");
    }
    if (!warm) {
        LLAMA_LOG_WARN("%s: LLAMA_SPLIT_GPU_WARM_US needs the Metal backend: GPU warm off\n", __func__);
        return;
    }
    st.warm_thread = std::thread([&st, warm, us] {
        std::unique_lock<std::mutex> lk(st.warm_mu);
        while (!st.warm_quit) {
            if (!st.warm_on) {
                st.warm_cv.wait(lk);
                continue;
            }
            lk.unlock();
            warm();
            std::this_thread::sleep_for(std::chrono::microseconds(us));
            lk.lock();
        }
    });
    LLAMA_LOG_INFO("%s: split decode: GPU warm every %d us while the worker computes\n", __func__, us);
}

static void split_warm(llama_split_state & st, bool on) {
    if (!st.warm_thread.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(st.warm_mu);
        if (st.warm_on == on) {
            return;
        }
        st.warm_on = on;
    }
    st.warm_cv.notify_one();
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// memory the OS can hand out right now without swapping (free + inactive + purgeable pages); SIZE_MAX if unknown
static size_t split_mem_available() {
#ifdef __APPLE__
    vm_statistics64_data_t vs;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &n) == KERN_SUCCESS) {
        return (size_t) (vs.free_count + vs.inactive_count + vs.purgeable_count) * (size_t) vm_kernel_page_size;
    }
#endif
    return SIZE_MAX;
}

// wait for the async mirror push (if any); throws its error so the caller falls back like any other worker failure
static void split_push_join(llama_split_state & st) {
    if (st.push_thread.joinable()) {
        const auto t0 = std::chrono::steady_clock::now();
        st.push_thread.join();
        st.push_join_ms = ms_since(t0);
    }
    if (!st.push_err.empty()) {
        const std::string e = st.push_err;
        st.push_err.clear();
        throw std::runtime_error("mirror push: " + e);
    }
}

void llama_context::split_init(ggml_type type_k, ggml_type type_v) {
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    if (const char * sd = getenv("LLAMA_SPLIT_DECODE"); sd && atoi(sd) != 0) {
        split_sd_init(type_k, type_v);
        return;
    }
    const char * env = getenv("LLAMA_SPLIT_TAIL");
    if (!env || !*env || !memory || model.arch != LLM_ARCH_QWEN35 || cparams.n_seq_max != 1 ||
        cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
        return;
    }
    split.reset(new llama_split_state());
    std::string s = env;
    const size_t c = s.rfind(':');
    split->host = s.substr(0, c);
    if (c != std::string::npos) split->port = atoi(s.c_str() + c + 1);
    if (const char * m = getenv("LLAMA_SPLIT_MIN")) split->min_tokens = std::max(1, atoi(m));
    split->verbose = getenv("LLAMA_SPLIT_VERBOSE") != nullptr;
    split_type_k = type_k;
    split_type_v = type_v;
    llama_split_register(memory.get(), split.get());
    LLAMA_LOG_WARN("%s: split prefill to %s:%d for batches >= %d tokens (LLAMA_SPLIT_TAIL)\n", __func__,
                   split->host.c_str(), split->port, split->min_tokens);
    // the worker gets every ubatch of a batch but the last, so it never gets work when a whole batch minus one ubatch is
    // under the minimum (e.g. -b 128 -ub 64 with the default minimum: found by @dre4moff, StayLameBro/backburner discussion #6)
    if ((int) cparams.n_batch - (int) cparams.n_ubatch < split->min_tokens) {
        LLAMA_LOG_WARN("%s: with -b %u and -ub %u at most %u tokens of a batch can go to the worker, under LLAMA_SPLIT_MIN=%d: "
                       "split prefill will never run (raise -b, or lower LLAMA_SPLIT_MIN)\n", __func__,
                       cparams.n_batch, cparams.n_ubatch, cparams.n_batch - cparams.n_ubatch, split->min_tokens);
    }
#else
    GGML_UNUSED(type_k); GGML_UNUSED(type_v);
#endif
}

void llama_context::split_free() {
    if (split) {
        llama_split_unregister(memory.get());
        split.reset();
    }
}

void llama_context::split_reset_graphs() {
    for (auto & r : gf_res_prev) {
        if (r) {
            r->reset();
        }
    }
    gf_res_prev_active = nullptr;
}

// our own filtered state read (not through the public API, so no mirror events)
std::vector<uint8_t> llama_context::split_get_state(llama_pos p0, llama_pos p1, int32_t il0, int32_t il1, bool kv, bool rs) {
    synchronize();
    llama_state_filter_set(p0, p1, il0, il1, 0, 0, kv, rs);
    std::vector<uint8_t> buf(state_seq_get_size(0, 0));
    const size_t n = state_seq_get_data(0, buf.data(), buf.size(), 0);
    llama_state_filter_clear();
    buf.resize(n);
    return buf;
}

bool llama_context::split_set_state(const std::vector<uint8_t> & b, int32_t il0, int32_t il1, int32_t kv_mode, int32_t rs_mode, bool kv, bool rs) {
    synchronize();
    llama_state_filter_set(0, 0x7fffffff, il0, il1, kv_mode, rs_mode, kv, rs);
    const size_t n = state_seq_set_data(0, b.data(), b.size(), 0);
    llama_state_filter_clear();
    return n == b.size();
}

// decide whether this batch is split; if so push the mirror delta, snapshot the recurrent state and switch the graph to
// the head layers. Returns the number of leading ubatches whose tail layers run on the worker (0 = run locally).
int llama_context::split_begin(uint32_t n_tokens_all, uint32_t n_outputs_all) {
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    auto & st = *split;
    st.active = false;
    const llama_batch & b = balloc->get_batch();
    const int ub = (int) cparams.n_ubatch;
    const int n  = (int) n_tokens_all;
    const int n_ub = (n + ub - 1) / ub;
    if (!b.token || b.embd || n < st.min_tokens) {
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (b.n_seq_id[i] != 1 || b.seq_id[i][0] != 0 || b.pos[i] != b.pos[0] + i) {
            return 0;
        }
    }
    const llama_pos D = b.pos[0];
    if (D != memory->seq_pos_max(0) + 1) {
        return 0;   // not an append at the end of the sequence
    }
    // The tail worker has no remote KV. Use full-model prefill past its local context.
    if ((uint64_t) D + n_tokens_all > cparams.n_ctx) {
        return 0;
    }
    // the worker gets every ubatch but the last; the last runs here "head-first" (layers [0, L) while the worker
    // finishes, then [L, n) after the merge), which hides the worker's last chunk and keeps outputs local.
    // Outputs are allowed only in that last ubatch.
    if (n_outputs_all > 0) {
        for (int i = 0; i < (n_ub - 1) * ub; i++) {
            if (b.logits[i]) {
                return 0;
            }
        }
    }
    GGML_UNUSED(n_outputs_all);
    const int n_split_ub = n_ub - 1;
    const int n_split = std::min(n, n_split_ub * ub);
    if (n_split < st.min_tokens) {
        return 0;
    }
    if (ggml_time_us() < st.t_retry_us) {
        return 0;
    }

    const int n_layer = (int) model.hparams.n_layer();
    try {
        // ---- connect (once; again after a failure)
        if (!st.tc) {
            char desc[256];
            llama_model_desc(&model, desc, sizeof(desc));
            spt::hello_req2 q = {};
            q.base.proto = spt::PROTO_VERSION; q.base.state_format = spt::STATE_FORMAT;
            q.base.n_layer_full = (uint32_t) n_layer; q.base.n_embd = (uint32_t) model.hparams.n_embd;
            q.base.n_vocab = (uint32_t) model.vocab.n_tokens(); q.base.n_ctx = cparams.n_ctx; q.base.n_ubatch = cparams.n_ubatch;
            q.type_k = split_type_k; q.type_v = split_type_v;
            q.flash_attn = cparams.flash_attn ? 1 : 0;
            q.n_rs_replay = cparams.n_rs_replay;
            q.session = (uint64_t) ggml_time_us(); q.keep = 0;
            // the worker checks L against its tail GGUF (LLAMA_SPLIT_L; unset: the L the worker reported last time, else 44)
            const char * env_L = getenv("LLAMA_SPLIT_L");
            q.base.L = (uint32_t) (env_L ? atoi(env_L) : (st.L_learned ? st.L_learned : 44));
            // taps the context wants at or above L (tail-local ids); the protocol carries at most 4
            for (uint32_t il = q.base.L; il < (uint32_t) n_layer && il < cparams.embeddings_layer_inp.size(); il++) {
                if (cparams.embeddings_layer_inp[il]) {
                    if (q.n_taps == 4) {
                        throw std::runtime_error("more than 4 tap layers at or above L = " + std::to_string(q.base.L) + " (raise L)");
                    }
                    q.taps[q.n_taps++] = (int32_t) (il - q.base.L);
                }
            }
            auto tc = std::make_unique<spt::tail_client>();
            // f32 residual by default: at layer ~44 the residual reaches |x| ~ 258, where f16's step is 0.125, and the
            // error grows over the worker's layers into the DFlash tap rows (-5% drafter acceptance measured with f16).
            // LLAMA_SPLIT_RESID_F16=1 halves the link bytes (~10 ms per 512-token chunk on USB).
            tc->resid_f32 = getenv("LLAMA_SPLIT_RESID_F16") == nullptr;
            spt::hello_rep2 r; std::string err;
            if (!tc->connect(st.host, st.port, q, r, err)) {
                // "tail model mismatch: phone has layers [52, 64) ...": without LLAMA_SPLIT_L, take the phone's L and retry
                unsigned phone_L = 0;
                const size_t at = err.find("phone has layers [");
                if (!env_L && at != std::string::npos && sscanf(err.c_str() + at, "phone has layers [%u", &phone_L) == 1 &&
                    phone_L > 0 && phone_L != q.base.L) {
                    LLAMA_LOG_WARN("%s: split prefill: the worker's tail starts at layer %u (not %u): reconnecting with L=%u\n",
                                   __func__, phone_L, q.base.L, phone_L);
                    st.L_learned = phone_L;
                    return split_begin(n_tokens_all, n_outputs_all);
                }
                throw std::runtime_error(err);
            }
            // the worker's weights must be this model's (same quant): its desc carries the same "<ftype>" text
            std::string mine = desc;
            const size_t sp = mine.find(' ', mine.find(' ') + 1);
            if (sp != std::string::npos) mine = mine.substr(sp + 1);
            if (std::string(r.base.desc).find(mine) == std::string::npos) {
                throw std::runtime_error(std::string("worker model '") + r.base.desc + "' is not this model ('" + desc + "')");
            }
            st.L = r.base.layer_start; st.n_layer = (uint32_t) n_layer;
            st.taps.clear();
            for (uint32_t i = 0; i < q.n_taps; i++) st.taps.push_back(q.taps[i] + (int32_t) st.L);
            st.tc = std::move(tc);
            st.W = 0; st.rs_clean = false;
            LLAMA_LOG_WARN("%s: split prefill: worker %s:%d = %s, layers [%u, %d) there, %zu tap layer(s) returned\n", __func__,
                           st.host.c_str(), st.port, r.base.desc, st.L, n_layer, st.taps.size());
        }
        // a tap turned on after the HELLO (the drafter attaches later) needs a new HELLO
        for (uint32_t il = st.L; il < (uint32_t) n_layer && il < cparams.embeddings_layer_inp.size(); il++) {
            if (cparams.embeddings_layer_inp[il] && std::find(st.taps.begin(), st.taps.end(), (int32_t) il) == st.taps.end()) {
                st.tc.reset();
                return split_begin(n_tokens_all, n_outputs_all);
            }
        }

        const auto t0 = std::chrono::steady_clock::now();
        st.active = true;
        st.D = D; st.n_split = n_split;
        st.chunks.clear();

        // ---- fallback snapshot: the recurrent state at D, all layers
        st.rs_snapshot = split_get_state(0, 0x7fffffff, 0, 0x7fffffff, false, true);

        // ---- mirror: [W, D) KV rows (in pieces) + the recurrent state unless the worker already has it
        auto & tc = *st.tc;
        if (tc.n_valid() > (uint32_t) st.W) {
            tc.trim((uint32_t) st.W);   // v2 resets the worker's mirror below its end
        }
        st.W = std::min<llama_pos>(st.W, (llama_pos) tc.n_valid());
        const bool need_rs = !(st.rs_clean && st.W == D);
        // read every piece now (fast: ~34 ms for 665 MB at 51k, L=40), send them on a thread: the transfer (~1 s over USB)
        // then overlaps the first head ubatch instead of delaying it; the first submit joins it
        // holding every piece at once costs ~665 MB at 51k (L=40): only when the machine has that to spare plus 1.5 GB,
        // else the old one-piece-at-a-time push (a 24 GB Mac at 64k context is at its memory edge: two hangs / kernel
        // panics on 2026-09-25/26 during split runs there)
        const size_t kv_row_bytes = D > st.W ? split_get_state(st.W, st.W + 1, (int32_t) st.L, n_layer, true, false).size() : 0;
        const size_t push_bytes   = kv_row_bytes * (size_t) (D - st.W);
        static const bool push_async_env = getenv("LLAMA_SPLIT_PUSH_ASYNC") == nullptr || atoi(getenv("LLAMA_SPLIT_PUSH_ASYNC")) != 0;
        const size_t mem_avail = split_mem_available();
        // the 1.5 GB margin guards the big first push after a restore (~665 MB at 51k); an incremental push in a continuing
        // session (the last message's new rows, ~25 MB) only needs the push itself + 256 MB free, or it blocks ~0.2-0.3 s per
        // message for nothing (2026-09-27, prefill-bench --chain)
        const size_t margin = push_bytes < ((size_t) 128 << 20) ? ((size_t) 256 << 20) : ((size_t) 3 << 29);
        const bool push_async = push_async_env && (mem_avail == SIZE_MAX || mem_avail > push_bytes + margin);
        if (push_async_env && !push_async && st.verbose) {
            LLAMA_LOG_WARN("%s: mirror push on this thread: %.0f MB available, the push needs %.0f MB\n", __func__, mem_avail/1e6, push_bytes/1e6);
        }
        struct piece { std::vector<uint8_t> blob; uint32_t flags, n_valid; };
        std::vector<piece> pieces;
        double mb = 0;
        for (llama_pos p = st.W; p < D; p += 8192) {
            const llama_pos p1 = std::min<llama_pos>(D, p + 8192);
            auto blob = split_get_state(p, p1, (int32_t) st.L, n_layer, true, false);
            mb += blob.size()/1e6;
            if (push_async) {
                pieces.push_back({ std::move(blob), 1, (uint32_t) p1 });
            } else {
                tc.sync_state(blob, 1, (uint32_t) p1);
            }
        }
        if (need_rs) {
            pieces.push_back({ split_get_state(0, 0x7fffffff, (int32_t) st.L, n_layer, false, true), 2, (uint32_t) D });
        }
        st.push_join_ms = 0; st.push_thread_ms = 0;
        if (push_async && !pieces.empty()) {
            split_push_join(st);   // never more than one in flight
            st.push_thread = std::thread([&st, tcp = st.tc.get(), pieces = std::move(pieces)]() mutable {
                const auto tt = std::chrono::steady_clock::now();
                try {
                    for (auto & pc : pieces) {
                        tcp->sync_state(pc.blob, pc.flags, pc.n_valid);
                        std::vector<uint8_t>().swap(pc.blob);   // free each piece once it is on the phone
                    }
                } catch (const std::exception & e) {
                    st.push_err = e.what();
                }
                st.push_thread_ms = ms_since(tt);
            });
        } else {
            for (auto & pc : pieces) {
                tc.sync_state(pc.blob, pc.flags, pc.n_valid);
            }
        }
        if (st.verbose && mb > 0) {
            LLAMA_LOG_WARN("%s: mirror push %.0f MB%s\n", __func__, mb, push_async ? " (sending while the first head ubatch runs)" : "");
        }
        st.W = D;
        st.ms_push += ms_since(t0);

        // ---- head graph: layers [0, L), expose the residual entering L
        st.prev_layer_inp_L = cparams.embeddings_layer_inp[st.L];
        cparams.embeddings_layer_inp[st.L] = true;
        cparams.layer_start = 0;
        cparams.layer_end   = (int32_t) st.L;
        split_reset_graphs();
        if (st.verbose || ms_since(t0) > 250) {   // a big push (after a restore / rewind) is always worth a line
            LLAMA_LOG_WARN("%s: split %d of %d tokens at D=%d (%d ubatches), push %.0f ms (rs %s)\n", __func__, n_split, n, D,
                           n_split_ub, ms_since(t0), need_rs ? "sent" : "kept");
        }
        return n_split_ub;
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: split prefill off for 60 s: %s\n", __func__, e.what());
        st.tc.reset();
        st.W = 0; st.rs_clean = false;
        st.t_retry_us = ggml_time_us() + 60 * 1000000LL;
        st.active = false;   // the graph switch comes after the last step that can throw
        return 0;
    }
#else
    GGML_UNUSED(n_tokens_all); GGML_UNUSED(n_outputs_all);
    return 0;
#endif
}

void llama_context::split_restore_graph() {
    auto & st = *split;
    cparams.embeddings_layer_inp[st.L] = st.prev_layer_inp_L;
    cparams.layer_start = 0;
    cparams.layer_end   = -1;
    split_reset_graphs();
}

// after a head ubatch: ship the residual entering L (rows [tok_off, tok_off + n_tok) of this batch)
bool llama_context::split_submit(int tok_off, int n_tok, llama_pos pos0) {
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    auto & st = *split;
    try {
        synchronize();
        split_push_join(st);
        const int n_embd = (int) model.hparams.n_embd;
        st.tc->submit_chunk(embd_layer_inp[st.L].data + (size_t) tok_off * n_embd, n_tok, n_embd, pos0, false, !st.taps.empty());
        st.chunks.emplace_back(tok_off, n_tok);
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: %s\n", __func__, e.what());
        return false;
    }
#else
    GGML_UNUSED(tok_off); GGML_UNUSED(n_tok); GGML_UNUSED(pos0);
    return false;
#endif
}

// wait for the worker, fill the tap rows, merge its rows of [D, D + n_split) and restore the full graph
bool llama_context::split_finish() {
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    auto & st = *split;
    const int n_embd = (int) model.hparams.n_embd;
    const int n_layer = (int) model.hparams.n_layer();
    try {
        auto t0 = std::chrono::steady_clock::now();
        const double wait0 = st.ms_wait;
        std::vector<float> logits;
        std::vector<ggml_fp16_t> taps;
        std::string err;
        split_push_join(st);
        if (!st.tc->finish(logits, taps, err)) {
            throw std::runtime_error(err);
        }
        st.ms_wait += ms_since(t0);
        // taps come per chunk as [n_taps][n_tok][n_embd] f16
        size_t off = 0;
        for (const auto & ch : st.chunks) {
            for (size_t t = 0; t < st.taps.size(); t++) {
                const size_t n = (size_t) ch.second * n_embd;
                if (off + n > taps.size()) throw std::runtime_error("short tap rows");
                auto & dst = embd_layer_inp[st.taps[t]];
                if (dst.has_data()) {
                    ggml_fp16_to_fp32_row(taps.data() + off, dst.data + (size_t) ch.first * n_embd, (int64_t) n);
                }
                off += n;
            }
        }
        t0 = std::chrono::steady_clock::now();
        const auto blob = st.tc->state_range(st.D, st.D + st.n_split, 3);
        split_restore_graph();
        // the head-first ubatch's head pass already advanced the recurrent cell: keep its pos / rp_c
        synchronize();
        llama_state_filter_set(0, 0x7fffffff, (int32_t) st.L, n_layer, 2, 2, true, true);
        llama_state_filter_cur().rs_keep_meta = true;
        const bool merged = state_seq_set_data(0, blob.data(), blob.size(), 0) == blob.size();
        llama_state_filter_clear();
        if (!merged) {
            throw std::runtime_error("merge of the worker's state failed");
        }
        st.ms_merge += ms_since(t0);
        st.W = st.D + st.n_split;
        st.rs_clean = true;
        st.n_calls++;
        st.n_tokens += st.n_split;
        st.active = false;
        if (st.verbose) {
            // the phone's own compute time per chunk of this call (the tail server's chunk_rep.compute_ms)
            std::string ph;
            const auto & tm = st.tc->tail_ms;
            for (size_t i = tm.size() - std::min(tm.size(), st.chunks.size()); i < tm.size(); i++) {
                ph += (ph.empty() ? "" : " ") + std::to_string((int) tm[i]);
            }
            if (st.push_thread_ms > 0) {
                LLAMA_LOG_WARN("%s: mirror push took %.0f ms on its thread, the Mac waited %.0f ms for it\n", __func__, st.push_thread_ms, st.push_join_ms);
            }
            LLAMA_LOG_WARN("%s: split %d tokens done: wait %.0f ms, merge %.0f ms, phone ms/chunk [%s]; totals %llu calls, %llu tokens, push %.0f, wait %.0f, merge %.0f ms\n",
                           __func__, st.n_split, st.ms_wait - wait0, ms_since(t0), ph.c_str(), (unsigned long long) st.n_calls, (unsigned long long) st.n_tokens,
                           st.ms_push, st.ms_wait, st.ms_merge);
        }
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: %s\n", __func__, e.what());
        return false;
    }
#else
    return false;
#endif
}

// the head-first ubatch: wait for the worker's earlier chunks + merge, then run layers [L, n) here on the residual
llm_graph_result * llama_context::split_tail_local(const llama_ubatch & ubatch, int tok_off, llama_memory_context_i * mctx, ggml_status & status) {
    auto & st = *split;
    synchronize();
    const int n_embd = (int) model.hparams.n_embd;
    const float * r = embd_layer_inp[st.L].data + (size_t) tok_off * n_embd;
    split_resid.assign(r, r + (size_t) ubatch.n_tokens * n_embd);
    if (!split_finish()) {          // restores the full graph on success
        return nullptr;
    }
    llama_ubatch ub = ubatch;
    ub.token = nullptr;
    ub.embd  = split_resid.data();
    cparams.layer_start = (int32_t) st.L;
    cparams.layer_end   = -1;
    split_reset_graphs();
    split_skip_apply = true;
    llm_graph_result * res = process_ubatch(ub, LLM_GRAPH_TYPE_DEFAULT, mctx, status);   // split only runs on default contexts
    split_skip_apply = false;
    // layer_start stays L until decode() has read this result's outputs (resetting the graphs would free them);
    // decode() restores the full range after its ubatch loop
    return res;
}

// undo this call's memory changes (the batch then reruns locally) and drop the worker for 60 s
void llama_context::split_abort() {
    auto & st = *split;
    LLAMA_LOG_WARN("%s: split prefill failed at D=%d; rerunning the batch locally, worker off for 60 s\n", __func__, st.D);
    synchronize();
    if (cparams.layer_end >= 0 || cparams.layer_start != 0) {
        split_restore_graph();
    }
    // recurrent state back to D first (then the recurrent seq_rm below is a no-op and the KV rows [D, ...) go)
    split_set_state(st.rs_snapshot, 0, 0x7fffffff, 0, /*rs replace*/ 0, false, true);
    memory->seq_rm(0, st.D, -1);
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    try { split_push_join(st); } catch (const std::exception &) {}
    st.tc.reset();
#endif
    st.W = 0; st.rs_clean = false;
    st.t_retry_us = ggml_time_us() + 60 * 1000000LL;
    st.n_fallbacks++;
    st.active = false;
}

// ---- split decode (LLAMA_SPLIT_DECODE=1): see llama-split.h ------------------------------------------------------------

int32_t llama_split_decode_L(const llama_model & model, const llama_cparams & cparams) {
    const char * sd = getenv("LLAMA_SPLIT_DECODE");
    if (!sd || atoi(sd) == 0 || model.arch != LLM_ARCH_QWEN35 || cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
        return -1;   // the target model's own context only (not a drafter / MTP context)
    }
    const char * env_L = getenv("LLAMA_SPLIT_L");
    const int L = env_L ? atoi(env_L) : 0;
    return L > 0 && L < (int) model.hparams.n_layer() ? L : 0;
}

void llama_context::split_sd_init(ggml_type type_k, ggml_type type_v) {
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    const int32_t L = llama_split_decode_L(model, cparams);
    if (!memory || L < 0) {
        return;
    }
    if (L == 0) {
        throw std::runtime_error("split decode: LLAMA_SPLIT_L must be the tail's first layer, 0 < L < " +
                                 std::to_string(model.hparams.n_layer()));
    }
    if (model.hparams.no_alloc) {
        // common_fit's no-alloc probe: no link, but the same head graph (its memory has no layers >= L either)
        cparams.layer_start = 0;
        cparams.layer_end   = L;
        cparams.embeddings_layer_inp[L] = true;
        return;
    }
    const char * env = getenv("LLAMA_SPLIT_TAIL");
    if (!env || !*env) {
        throw std::runtime_error("split decode (LLAMA_SPLIT_DECODE=1) needs LLAMA_SPLIT_TAIL=ip:port, the phone's tail");
    }
    if (cparams.n_seq_max != 1) {
        throw std::runtime_error("split decode supports one sequence only (-np 1), this context has n_seq_max = " +
                                 std::to_string(cparams.n_seq_max));
    }
    if (cparams.embeddings) {
        throw std::runtime_error("split decode returns logits only: embeddings contexts are not supported");
    }
    split.reset(new llama_split_state());
    split->sd = true;
    std::string s = env;
    const size_t c = s.rfind(':');
    split->host = s.substr(0, c);
    if (c != std::string::npos) split->port = atoi(s.c_str() + c + 1);
    split->verbose = getenv("LLAMA_SPLIT_VERBOSE") != nullptr;
    split_type_k = type_k;
    split_type_v = type_v;
    std::string err;
    if (!split_sd_connect(err)) {
        split.reset();
        throw std::runtime_error("split decode: " + err);
    }
    llama_split_register(memory.get(), split.get());
    if (const char * w = getenv("LLAMA_SPLIT_GPU_WARM_US"); w && atoi(w) > 0) {
        split_warm_start(*split, atoi(w));
    }
    // every graph of this context is the head: layers [0, L), the residual entering L read back for the worker
    cparams.layer_start = 0;
    cparams.layer_end   = (int32_t) split->L;
    cparams.embeddings_layer_inp[split->L] = true;
    LLAMA_LOG_WARN("%s: split decode: layers [0, %u) here, [%u, %u) + output on %s:%d for every token (LLAMA_SPLIT_DECODE)\n",
                   __func__, split->L, split->L, split->n_layer, split->host.c_str(), split->port);
#else
    GGML_UNUSED(type_k); GGML_UNUSED(type_v);
    if (memory && llama_split_decode_L(model, cparams) >= 0) {
        throw std::runtime_error("split decode (LLAMA_SPLIT_DECODE=1) needs a build with sockets");
    }
#endif
}

// connect + HELLO; the worker must hold exactly this model's layers [LLAMA_SPLIT_L, n_layer) with the head.
// The HELLO makes the worker (re)create its context (KV mirror at this n_ctx) and clears its mirror.
bool llama_context::split_sd_connect(std::string & err) {
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    auto & st = *split;
    try {
        const int n_layer = (int) model.hparams.n_layer();
        const char * env_L = getenv("LLAMA_SPLIT_L");
        const int L = env_L ? atoi(env_L) : 0;
        if (L <= 0 || L >= n_layer) {
            throw std::runtime_error("LLAMA_SPLIT_L must be the tail's first layer, 0 < L < " + std::to_string(n_layer) +
                                     " (got '" + std::string(env_L ? env_L : "") + "')");
        }
        for (int il = L + 1; il < (int) cparams.embeddings_layer_inp.size(); il++) {
            if (cparams.embeddings_layer_inp[il]) {
                throw std::runtime_error("tap layer " + std::to_string(il) + " is on the worker: not supported in split decode");
            }
        }
        char desc[256];
        llama_model_desc(&model, desc, sizeof(desc));
        spt::hello_req2 q = {};
        q.base.proto = spt::PROTO_VERSION; q.base.state_format = spt::STATE_FORMAT;
        q.base.L = (uint32_t) L;
        q.base.n_layer_full = (uint32_t) n_layer; q.base.n_embd = (uint32_t) model.hparams.n_embd;
        q.base.n_vocab = (uint32_t) model.vocab.n_tokens(); q.base.n_ctx = cparams.n_ctx; q.base.n_ubatch = cparams.n_ubatch;
        q.type_k = split_type_k; q.type_v = split_type_v;
        q.flash_attn = cparams.flash_attn ? 1 : 0;
        q.n_rs_replay = cparams.n_rs_replay;
        q.session = (uint64_t) ggml_time_us(); q.keep = 0;
        auto tc = std::make_unique<spt::tail_client>();
        tc->resid_f32 = getenv("LLAMA_SPLIT_RESID_F16") == nullptr;
        // a phone that accepts but never answers (app suspended) fails here instead of hanging the server
        tc->recv_timeout_s = getenv("LLAMA_SPLIT_TIMEOUT_S") ? atoi(getenv("LLAMA_SPLIT_TIMEOUT_S")) : 120;
        spt::hello_rep2 r; std::string e;
        const auto t0 = std::chrono::steady_clock::now();
        if (!tc->connect(st.host, st.port, q, r, e)) {
            throw std::runtime_error(e);
        }
        if (r.base.layer_start != (uint32_t) L || r.base.n_layer_full != (uint32_t) n_layer || r.base.n_embd != q.base.n_embd ||
            r.base.n_vocab != q.base.n_vocab) {
            throw std::runtime_error("worker tail is layers [" + std::to_string(r.base.layer_start) + ", " + std::to_string(r.base.n_layer_full) +
                                     ") n_embd " + std::to_string(r.base.n_embd) + " n_vocab " + std::to_string(r.base.n_vocab) +
                                     ", this model wants [" + std::to_string(L) + ", " + std::to_string(n_layer) + ") n_embd " +
                                     std::to_string(q.base.n_embd) + " n_vocab " + std::to_string(q.base.n_vocab));
        }
        // same quant: the worker's desc carries this model's "<ftype>" text
        std::string mine = desc;
        const size_t sp = mine.find(' ', mine.find(' ') + 1);
        if (sp != std::string::npos) mine = mine.substr(sp + 1);
        if (std::string(r.base.desc).find(mine) == std::string::npos) {
            throw std::runtime_error(std::string("worker model '") + r.base.desc + "' is not this model's quant ('" + desc + "')");
        }
        if (r.n_valid != 0) {
            throw std::runtime_error("worker kept a mirror of " + std::to_string(r.n_valid) + " tokens after a fresh HELLO");
        }
        st.L = (uint32_t) L; st.n_layer = (uint32_t) n_layer;
        st.tc = std::move(tc);
        st.W = 0;
        LLAMA_LOG_WARN("%s: split decode: worker %s:%d = %s, layers [%u, %d) + head there, n_ctx %u, HELLO %.0f ms\n", __func__,
                       st.host.c_str(), st.port, r.base.desc, st.L, n_layer, r.base.n_ctx, ms_since(t0));
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        st.tc.reset();
        st.W = 0;
        return false;
    }
#else
    err = "no sockets in this build";
    return false;
#endif
}

// before the ubatch loop: check the batch is one this mode computes exactly, and that the worker's mirror ends where it starts.
// Returns 0, or decode()'s error code (-1 invalid batch, -3 worker down / out of sync).
int llama_context::split_sd_begin(uint32_t n_tokens_all, uint32_t n_outputs_all) {
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    auto & st = *split;
    const llama_batch & b = balloc->get_batch();
    const int n = (int) n_tokens_all;
    auto bad = [&](const std::string & why) {
        LLAMA_LOG_ERROR("%s: split decode can't run this batch: %s\n", __func__, why.c_str());
        return -1;
    };
    if (!b.token || b.embd) return bad("token batches only");
    if (cparams.embeddings || cparams.embeddings_nextn) return bad("embeddings are not returned by the worker");
    if (!sampling.samplers.empty()) return bad("backend samplers need the logits on this device");
    for (int il = (int) st.L + 1; il < (int) cparams.embeddings_layer_inp.size(); il++) {
        if (cparams.embeddings_layer_inp[il]) return bad("tap layer " + std::to_string(il) + " is on the worker");
    }
    for (int i = 0; i < n; i++) {
        if (b.n_seq_id[i] != 1 || b.seq_id[i][0] != 0 || b.pos[i] != b.pos[0] + i) {
            return bad("one sequence (seq 0) with consecutive positions only");
        }
    }
    if (n_outputs_all > 1 || (n_outputs_all == 1 && !b.logits[n - 1])) {
        return bad(std::to_string(n_outputs_all) + " outputs requested: only the batch's last token can have logits");
    }
    const llama_pos D = b.pos[0];
    if (D != memory->seq_pos_max(0) + 1) {
        return bad("batch at position " + std::to_string(D) + " is not an append (memory ends at " + std::to_string(memory->seq_pos_max(0)) + ")");
    }
    if (!st.tc) {
        if (D != 0) {
            LLAMA_LOG_ERROR("%s: split decode: the worker link is down and this batch continues at position %d: the conversation must restart from 0\n", __func__, D);
            return -3;
        }
        std::string err;
        if (!split_sd_connect(err)) {
            LLAMA_LOG_ERROR("%s: split decode: worker reconnect failed: %s\n", __func__, err.c_str());
            return -3;
        }
    }
    try {
        auto & tc = *st.tc;
        if (tc.n_valid() > (uint32_t) st.W) {
            LLAMA_LOG_WARN("%s: split decode: resetting the worker's mirror (it holds [0, %u), this context kept [0, %d))\n", __func__, tc.n_valid(), st.W);
            tc.trim((uint32_t) st.W);   // below the end: the worker resets to 0
        }
        st.W = (llama_pos) tc.n_valid();
    } catch (const std::exception & e) {
        split_sd_fail(std::string("trim: ") + e.what());
        return -3;
    }
    if (st.W != D) {
        LLAMA_LOG_ERROR("%s: split decode: the worker holds positions [0, %d) but this batch starts at %d: re-prefill from 0 needed\n", __func__, st.W, D);
        return -3;
    }
    st.sd_D = D; st.sd_n = n;
    st.sd_t0_us = ggml_time_us();
    return 0;
#else
    GGML_UNUSED(n_tokens_all); GGML_UNUSED(n_outputs_all);
    return -3;
#endif
}

// after a ubatch's head pass: ship its residual entering L (pipelined; the next ubatch's head pass overlaps the worker)
bool llama_context::split_sd_submit(int tok_off, int n_tok, llama_pos pos0, bool want_logits) {
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    auto & st = *split;
    try {
        synchronize();
        const int n_embd = (int) model.hparams.n_embd;
        st.tc->submit_chunk(embd_layer_inp[st.L].data + (size_t) tok_off * n_embd, n_tok, n_embd, pos0, want_logits, false);
        split_warm(st, true);
        return true;
    } catch (const std::exception & e) {
        split_sd_fail(std::string("submit: ") + e.what());
        return false;
    }
#else
    GGML_UNUSED(tok_off); GGML_UNUSED(n_tok); GGML_UNUSED(pos0); GGML_UNUSED(want_logits);
    return false;
#endif
}

// after the ubatch loop: wait for every chunk; the output's logits row goes where a local decode would have put it (row 0:
// the batch's only output)
bool llama_context::split_sd_finish(uint32_t n_outputs_all) {
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    auto & st = *split;
    try {
        const int64_t t_sub = ggml_time_us();
        std::vector<float> lg;
        std::vector<ggml_fp16_t> taps;
        std::string err;
        const bool ok = st.tc->finish(lg, taps, err);
        split_warm(st, false);
        if (!ok) {
            throw std::runtime_error(err);
        }
        if (n_outputs_all == 1) {
            const size_t n_vocab = (size_t) model.vocab.n_tokens();
            if (lg.size() != n_vocab) {
                throw std::runtime_error("worker returned " + std::to_string(lg.size()) + " logits, want " + std::to_string(n_vocab) +
                                         " (a head-less tail?)");
            }
            if (!logits.data || logits.size < n_vocab) {
                throw std::runtime_error("no logits buffer for the output");
            }
            memcpy(logits.data, lg.data(), n_vocab * sizeof(float));
        }
        st.W = st.sd_D + st.sd_n;
        st.n_calls++;
        st.n_tokens += st.sd_n;
        if (st.verbose) {
            const auto & tm = st.tc->tail_ms;
            LLAMA_LOG_WARN("%s: split decode %d tokens at %d: Mac %.1f ms, wait %.1f ms (phone %.1f ms last chunk), %s\n", __func__,
                           st.sd_n, st.sd_D, (t_sub - st.sd_t0_us) / 1e3, (ggml_time_us() - t_sub) / 1e3, tm.empty() ? 0.0 : tm.back(),
                           n_outputs_all ? "logits" : "no output");
        }
        return true;
    } catch (const std::exception & e) {
        split_sd_fail(std::string("finish: ") + e.what());
        return false;
    }
#else
    GGML_UNUSED(n_outputs_all);
    return false;
#endif
}

// any worker error: the decode fails (the server then clears the conversation), the link is dropped and reopened (fresh
// HELLO, empty mirror) by the next batch that starts at position 0. Layers >= L never run here.
void llama_context::split_sd_fail(const std::string & why) {
    auto & st = *split;
    split_warm(st, false);
    LLAMA_LOG_ERROR("%s: split decode failed: %s; decode fails, worker link dropped (reconnects at the next batch from position 0)\n",
                    __func__, why.c_str());
#ifdef LLAMA_SPLIT_HAVE_SOCKETS
    if (st.tc) {
        st.tc->abort_link();
        st.tc.reset();
    }
#endif
    st.W = 0;
    st.n_fallbacks++;
}
