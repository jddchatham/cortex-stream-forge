#pragma once

#include "flashnext-layer-transaction.h"
#include "ggml-backend.h"
#include "flashnext-tensor-index.h"

#include <array>
#include <cstdint>

struct flashnext_selected_expert_slice {
    std::array<int, 3> fds{{-1, -1, -1}};
    std::array<uint64_t, 3> offsets{}; // gate, up, down; layer-local slices
};

// Executes the already-validated qwen4exp selected-expert path using the
// transaction's dynamic IDs and routing weights. The caller supplies only
// the selected slice offsets; no complete expert collection is constructed.
bool flashnext_execute_selected_moe(
        const std::array<flashnext_selected_expert_slice, flashnext_layer_transaction::experts_used> & slices,
        flashnext_layer_transaction & tx,
        size_t arena_bytes = 0);

bool flashnext_execute_selected_moe(
        const flashnext_tensor_index & index,
        const std::string & index_path,
        flashnext_layer_transaction & tx,
        size_t arena_bytes = 0);

// Independent stock-compatible validation path. It uses the same GGML
// quantized matvec semantics but performs its own expert accumulation.
bool flashnext_execute_selected_moe_reference(
        const flashnext_tensor_index & index,
        const std::string & index_path,
        flashnext_layer_transaction & tx,
        size_t arena_bytes = 0);
