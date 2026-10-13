#pragma once

#include "flashnext-layer-transaction.h"
#include "ggml-backend.h"
#include "flashnext-tensor-index.h"

#include <array>
#include <cstdint>
#include <vector>

struct flashnext_selected_expert_slice {
    std::array<int, 3> fds{{-1, -1, -1}};
    std::array<uint64_t, 3> offsets{}; // gate, up, down; layer-local slices
};

// Stock Qwen4Exp routing contract for softmax-gated MoE.  Logits are laid out
// [n_expert, n_tokens], matching GGML; ids and weights are row-major by token.
// Ties are resolved by the lower expert id, matching deterministic argsort.
bool flashnext_route_softmax_topk(
        const float * logits,
        size_t n_expert,
        size_t n_tokens,
        size_t k,
        float weight_scale,
        std::vector<int32_t> & selected_ids,
        std::vector<double> & normalized_weights);

// Executes the already-validated qwen4exp selected-expert path using the
// transaction's dynamic IDs and routing weights. The caller supplies only
// the selected slice offsets; no complete expert collection is constructed.
bool flashnext_execute_selected_moe(
        const std::array<flashnext_selected_expert_slice, flashnext_layer_transaction::experts_used> & slices,
        flashnext_layer_transaction & tx,
        size_t arena_bytes = 16 * 1024);

bool flashnext_execute_selected_moe(
        const flashnext_tensor_index & index,
        const std::string & index_path,
        flashnext_layer_transaction & tx,
        size_t arena_bytes = 16 * 1024);

// Independent stock-compatible validation path. It uses the same GGML
// quantized matvec semantics but performs its own expert accumulation.
bool flashnext_execute_selected_moe_reference(
        const flashnext_tensor_index & index,
        const std::string & index_path,
        flashnext_layer_transaction & tx,
        size_t arena_bytes = 16 * 1024);

// Bounded multi-token adapter.  Each row uses the existing one-token
// selected-MoE executor, so expert payloads are never accumulated into a
// resident collection.  ids/weights are row-major [n_tokens, experts_used].
bool flashnext_execute_selected_moe_rows(
        const flashnext_tensor_index & index,
        const std::string & index_path,
        const float * pre_moe,
        const int32_t * selected_expert_ids,
        const double * routing_weights,
        size_t n_tokens,
        float * selected_moe,
        size_t arena_bytes = 16 * 1024);

bool flashnext_execute_selected_moe_rows_layer(
        const flashnext_tensor_index & index, int layer,
        const float * pre_moe, const int32_t * selected_expert_ids,
        const double * routing_weights, size_t n_tokens, float * selected_moe,
        size_t arena_bytes = 16 * 1024);

bool flashnext_execute_selected_gate_rows_layer(
        const flashnext_tensor_index & index, int layer,
        const float * activation, const int32_t * selected_expert_ids,
        size_t n_tokens, float * gate_output,
        size_t arena_bytes = 16 * 1024);

bool flashnext_execute_selected_projection_rows_layer(
        const flashnext_tensor_index & index, int layer, flashnext_projection projection,
        const float * activation, const int32_t * selected_expert_ids,
        size_t n_tokens, float * output, size_t n_rows, size_t input_width,
        size_t arena_bytes = 16 * 1024);

bool flashnext_execute_selected_moe_synthetic_rows(
        const float * pre_moe, const int32_t * selected_expert_ids,
        const double * routing_weights, size_t n_tokens, float * selected_moe,
        size_t arena_bytes = 16 * 1024);
