#pragma once

#include "flashnext-layer-transaction.h"

#include <cstdint>
#include <vector>

// Private experimental contract. Boundary fields own values only; no GGML,
// backend, scheduler, or graph pointer may cross a transaction boundary.
struct flashnext_layer_range {
    uint32_t start_layer = 0;
    uint32_t end_layer_exclusive = 1;
    int64_t position = 0;
    uint32_t sequence_id = 0;
    std::vector<float> input_hidden;
    std::vector<float> output_hidden;
    std::vector<uint8_t> recurrent_state;
    std::vector<uint8_t> attention_kv_state;
    uint64_t graph_generation = 0;
    uint64_t scheduler_generation = 0;
    bool graph_reserved = false;
    bool graph_allocated = false;
    bool executed = false;
    bool synchronized = false;
    bool state_committed = false;

    bool valid() const {
        const bool final_head = std::getenv("CORTEX_FLASHNEXT_FINAL_HEAD") != nullptr;
        return (final_head && start_layer == 48 && end_layer_exclusive == 48) ||
            (start_layer < end_layer_exclusive && end_layer_exclusive <= 48);
    }
};
