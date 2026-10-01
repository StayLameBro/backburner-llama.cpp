#pragma once

// infernet split prefill inside llama_decode (env LLAMA_SPLIT_TAIL=ip:port; off otherwise).
//
// A large prompt batch on a qwen35 context runs layers [0, L) here and layers [L, n_layer) on a TAIL worker (the iPhone's
// Sidecar, tools/split-prefill/tail-server.h), pipelined per ubatch. The worker keeps a MIRROR of its layers' memory for
// positions [0, W): only rows the mirror lacks are pushed, and the worker's rows for the new tokens are merged back in
// place (llama_state_filter OVERWRITE), so after the call this context holds the same state as a Mac-only decode would
// (token-identical within fp noise; see tools/split-prefill/mirror-test.cpp for the gate).
//
// Engaged only for: a token batch of one sequence (seq 0) appended at the end of its memory, >= LLAMA_SPLIT_MIN tokens
// (default 2048) in split ubatches, outputs (logits) only in the last ubatch. The last ubatch never goes to the worker: it
// runs "head-first" here (layers [0, L) while the worker finishes the previous chunk, merge, then [L, n) here), so the
// worker's last chunk is hidden and logits/samplers/outputs stay local. Tap layers
// (llama_set_embeddings_layer_inp, DFlash) at or above L come back from the worker for every split row.
// Any worker error: the call's memory changes are undone (GDN state snapshot + seq_rm) and the batch reruns locally.
//
// The mirror watermark W is lowered by the public memory/state API (seq_rm below W, seq_add/div/clear, full state set);
// see llama_split_mem_event.

#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace spt { class tail_client; }

// llama_context::decode return code: the split failed and was undone; llama_decode reruns the batch locally
#define LLAMA_SPLIT_RERUN (-100)

struct llama_memory_i;

enum llama_split_event {
    LLAMA_SPLIT_EV_RM,        // seq_rm(seq, p0, ...) : W = min(W, p0)
    LLAMA_SPLIT_EV_RESET,     // the mirror no longer matches: W = 0
    LLAMA_SPLIT_EV_RS_DIRTY,  // the recurrent state changed (not the KV rows)
};

// called by the public llama_memory_* / llama_state_* entry points
void llama_split_mem_event(const llama_memory_i * mem, llama_split_event ev, llama_seq_id seq, llama_pos p0);

struct llama_split_state {
    std::string host;
    int         port       = 50060;
    int         min_tokens = 2048;
    bool        verbose    = false;

    std::unique_ptr<spt::tail_client> tc;
    int64_t  t_retry_us = 0;          // after a failure, reconnect no earlier than this
    uint32_t L = 0;                   // first layer on the worker
    uint32_t L_learned = 0;           // the worker's L from its mismatch reply (used when LLAMA_SPLIT_L is unset)
    uint32_t n_layer = 0;

    // mirror: the worker holds this context's tail-layer KV rows for [0, W) and, if rs_clean, the same recurrent state
    llama_pos W = 0;
    bool      rs_clean = false;

    // the current call
    bool      active = false;
    llama_pos D = 0;                  // first position of the batch
    int       n_split = 0;            // tokens whose tail layers run on the worker
    bool      prev_layer_inp_L = false;
    std::vector<int32_t> taps;        // global tap layers >= L the context wants
    std::vector<std::pair<int, int>> chunks;   // (token offset in the batch, n_tokens) per submitted ubatch
    std::vector<uint8_t> rs_snapshot; // recurrent state at D (all layers), for the fallback

    // the mirror push runs on its own thread while the Mac computes the first head ubatch (LLAMA_SPLIT_PUSH_ASYNC, default on);
    // everything else that talks to the worker joins it first (split_push_join)
    std::thread push_thread;
    std::string push_err;
    double      push_thread_ms = 0;   // how long the thread took
    double      push_join_ms   = 0;   // how long the Mac waited for it (0 = fully hidden)

    // stats
    uint64_t n_calls = 0, n_fallbacks = 0, n_tokens = 0;
    double   ms_push = 0, ms_wait = 0, ms_merge = 0;

    ~llama_split_state();
};

// registry memory -> split state (for llama_split_mem_event)
void llama_split_register(const llama_memory_i * mem, llama_split_state * st);
void llama_split_unregister(const llama_memory_i * mem);
