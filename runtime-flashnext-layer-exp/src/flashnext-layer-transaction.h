#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace flashnext_profile {
void record_pread(size_t bytes);
uint64_t pread_calls();
uint64_t pread_bytes();
}

// Bounded, model-weight-free transaction for one qwen4exp layer position.
// The vectors own their storage; no member points into a ggml graph arena or
// a temporary serialized artifact.  Model tensors remain owned by the
// provider and are streamed independently.
struct flashnext_layer_transaction {
    static constexpr size_t hidden = 2560;
    static constexpr size_t streams = 4;
    static constexpr size_t experts_used = 10;

    std::array<float, hidden> pre_moe_hidden{};
    std::array<float, hidden * streams> residual_hc{};
    std::array<float, streams> inject{};

    std::array<int32_t, experts_used> selected_expert_ids{};
    std::array<double, experts_used> routing_weights{};

    std::array<float, hidden> selected_moe{};
    std::array<float, hidden> shared_moe{};
    std::array<float, hidden> complete_moe{};
    std::array<float, hidden * streams> layer_output{};

    struct io_accounting {
        uint64_t selected_expert_bytes_read = 0;
        uint64_t unselected_expert_reads = 0;
        uint64_t tensor_window_bytes = 16 * 1024;
        uint64_t pread_calls = 0;
    } io;

    int layer = 0;
    int64_t position = 0;
    bool state_committed = false;

    // State is deliberately opaque at this first transaction boundary.  The
    // layer runner owns the concrete recurrent/conv buffers and records their
    // committed byte snapshots here only when the state API is connected.
    std::vector<uint8_t> committed_state;
};
