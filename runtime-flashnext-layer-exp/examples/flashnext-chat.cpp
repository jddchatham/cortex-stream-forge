#include "llama.h"
#include "llama-ext.h"
#include "/home/user/bitnet-cortex/cortex-small-model/runtime-flashnext-layer-exp/src/llama-context.h"
#include "/home/user/bitnet-cortex/cortex-small-model/runtime-flashnext-layer-exp/src/llama-model.h"
#include "/home/user/bitnet-cortex/cortex-small-model/runtime-flashnext-layer-exp/src/flashnext-layer-transaction.h"
#include "/home/user/bitnet-cortex/cortex-small-model/runtime-flashnext-layer-exp/src/flashnext-selected-moe.h"
#include "/home/user/bitnet-cortex/cortex-small-model/runtime-flashnext-layer-exp/src/flashnext-shared-expert.h"
#include "/home/user/bitnet-cortex/cortex-small-model/runtime-flashnext-layer-exp/src/flashnext-segment-b.h"
#include "/home/user/bitnet-cortex/cortex-small-model/runtime-flashnext-layer-exp/src/flashnext-tensor-index.h"
#include "/home/user/bitnet-cortex/cortex-small-model/runtime-flashnext-layer-exp/ggml/src/ggml-quants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

static uint64_t hash_f32(const float * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { uint32_t u; std::memcpy(&u, p+i, 4); h ^= u; h *= 1099511628211ull; }
    return h;
}

static bool flashnext_stream_output_q6k(const flashnext_tensor_descriptor & d,
        const std::vector<float> & x, std::vector<float> & logits) {
    if (d.type != GGML_TYPE_Q6_K || d.shape0 != 2560 || d.shape1 != 248320 || x.size() != 2560) return false;
    const int fd = ::open(d.path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    constexpr size_t row_bytes = 10 * sizeof(block_q6_K);
    std::vector<uint8_t> row(row_bytes);
    std::vector<float> wf(2560);
    logits.resize((size_t) d.shape1);
    for (size_t r = 0; r < logits.size(); ++r) {
        size_t done = 0;
        while (done < row.size()) {
            const ssize_t n = ::pread(fd, row.data() + done, row.size() - done,
                    (off_t) (d.offset + r * row_bytes + done));
            if (n <= 0) { ::close(fd); return false; }
            done += (size_t) n;
        }
        dequantize_row_q6_K((const block_q6_K *) row.data(), wf.data(), 2560);
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) sum += x[i] * wf[i];
        logits[r] = sum;
    }
    ::close(fd);
    return true;
}

struct layer_transaction_result {
    int decode_status = -1;
    bool graph_a_ok = false;
    bool sparse_moe_ok = false;
    bool graph_b_ok = false;
    bool state_committed = false;
    bool rollback_restored = false;
    const char * failure_stage = nullptr;
    std::array<float, 2560 * 4> output{};
    uint64_t output_hash = 0;
    uint64_t sparse_output_hash = 0;
    std::array<int32_t, 10> selected_ids{};
    uint64_t unselected_expert_reads = 0;
    std::vector<uint8_t> state_before;
};

static std::vector<uint8_t> capture_seq_state(llama_context * ctx) {
    const size_t n = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> state(n);
    if (n != 0 && llama_state_seq_get_data(ctx, state.data(), n, 0) != n) state.clear();
    return state;
}

static layer_transaction_result run_layer_decode(
        llama_context * ctx, int position, const std::vector<float> & owned_hidden) {
    layer_transaction_result result;
    llama_batch batch = llama_batch_init(1, (int) owned_hidden.size(), 1);
    if (!batch.embd) { result.failure_stage = "batch_init"; return result; }
    std::memcpy(batch.embd, owned_hidden.data(), owned_hidden.size() * sizeof(float));
    batch.pos[0] = position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    batch.n_tokens = 1;
    result.state_before = capture_seq_state(ctx);
    result.decode_status = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    llama_batch_free(batch);
    return result;
}

struct flashnext_layer_descriptor {
    uint32_t index = 0;
    bool recurrent = false;
    bool ple = false;
    bool supported = false;
    const llama_layer * model_layer = nullptr;
    ggml_tensor * router = nullptr;
    ggml_tensor * gate_experts = nullptr;
    ggml_tensor * up_experts = nullptr;
    ggml_tensor * down_experts = nullptr;
    ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_up = nullptr;
    ggml_tensor * shared_down = nullptr;
};

struct flashnext_transaction_context {
    llama_context * ctx = nullptr;
    flashnext_tensor_index * tensor_index = nullptr;
    flashnext_layer_descriptor layer_desc;
    std::vector<uint8_t> staged_state;
    flashnext_layer_transaction values;
    std::array<float, 2560 * 4> output{};
    std::array<int32_t, 10> selected_ids{};
    std::array<float, 2560> moe_input{};
    std::array<float, 512> router_logits{};
    std::array<float, 2560 * 4> residual_hc{};
    std::array<float, 4> injection{};
    bool state_committed = false;
};

static bool resolve_flashnext_layer_descriptor(
        llama_context * ctx, uint32_t layer, flashnext_layer_descriptor & out, std::string & error) {
    if (!ctx) { error = "missing context"; return false; }
    const llama_model & model = ctx->get_model();
    if (layer >= model.hparams.n_layer()) { error = "layer index out of range"; return false; }

    const llama_layer & ml = model.layers[layer];
    out.index = layer;
    out.recurrent = model.hparams.is_recr(layer);
    out.ple = model.hparams.is_ple(layer);
    out.model_layer = &ml;
    out.router = ml.ffn_gate_inp;
    out.gate_experts = ml.ffn_gate_exps;
    out.up_experts = ml.ffn_up_exps;
    out.down_experts = ml.ffn_down_exps;
    out.shared_gate = ml.ffn_gate_inp_shexp;
    out.shared_up = ml.ffn_up_shexp;
    out.shared_down = ml.ffn_down_shexp;

    if (!out.router || !out.gate_experts || !out.up_experts || !out.down_experts) {
        error = "layer is missing sparse-MoE tensors"; return false;
    }
    if (!out.shared_gate || !out.shared_up || !out.shared_down) {
        error = "layer is missing shared-expert tensors"; return false;
    }
    out.supported = true;
    return true;
}

static bool capture_transaction_router(
        flashnext_transaction_context & txc, llama_context * ctx, const flashnext_layer_descriptor & layer) {
    const std::string suffix = "-" + std::to_string(layer.index);
    const size_t moe_bytes = ctx->debug_copy_flashnext_tensor(("diag_moe_input" + suffix).c_str(),
        txc.moe_input.data(), sizeof(txc.moe_input));
    const size_t router_bytes = ctx->debug_copy_flashnext_tensor(("diag_router_logits" + suffix).c_str(),
        txc.router_logits.data(), sizeof(txc.router_logits));
    const size_t residual_bytes = ctx->debug_copy_flashnext_tensor(("diag_hc_residual" + suffix).c_str(),
        txc.residual_hc.data(), sizeof(txc.residual_hc));
    const size_t inject_bytes = ctx->debug_copy_flashnext_tensor(("diag_hc_inject" + suffix).c_str(),
        txc.injection.data(), sizeof(txc.injection));
    if (moe_bytes != sizeof(txc.moe_input) || router_bytes != sizeof(txc.router_logits) ||
        residual_bytes != sizeof(txc.residual_hc) || inject_bytes != sizeof(txc.injection)) return false;
    for (float x : txc.moe_input) if (!std::isfinite(x)) return false;
    for (float x : txc.router_logits) if (!std::isfinite(x)) return false;
    for (float x : txc.residual_hc) if (!std::isfinite(x)) return false;
    for (float x : txc.injection) if (!std::isfinite(x)) return false;
    return true;
}

static bool execute_transaction_sparse_moe(
        flashnext_transaction_context & txc, const std::string & index_path, std::string & error) {
    if (!txc.tensor_index) { error = "missing tensor index"; return false; }
    std::array<int, 512> order{};
    for (int i = 0; i < 512; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 10,
        order.end(), [&](int a, int b) { return txc.router_logits[a] > txc.router_logits[b]; });
    double z = 0.0, ex[10];
    const float mx = txc.router_logits[order[0]];
    for (size_t i = 0; i < 10; ++i) {
        ex[i] = std::exp((double) txc.router_logits[order[i]] - mx); z += ex[i];
    }
    txc.values.layer = (int) txc.layer_desc.index;
    for (size_t i = 0; i < 10; ++i) {
        txc.values.selected_expert_ids[i] = order[i];
        txc.values.routing_weights[i] = ex[i] / z;
        txc.selected_ids[i] = order[i];
    }
    std::memcpy(txc.values.pre_moe_hidden.data(), txc.moe_input.data(), sizeof(txc.moe_input));
    std::memcpy(txc.values.residual_hc.data(), txc.residual_hc.data(), sizeof(txc.residual_hc));
    std::memcpy(txc.values.inject.data(), txc.injection.data(), sizeof(txc.injection));
    if (!txc.tensor_index->load(index_path, &error)) return false;
    if (!flashnext_execute_selected_moe(*txc.tensor_index, index_path, txc.values)) return false;
    return true;
}

static bool execute_transaction_graph_b(
        flashnext_transaction_context & txc, const std::string & index_path, std::string & error) {
    if (!txc.tensor_index) { error = "missing tensor index"; return false; }
    flashnext_tensor_descriptor sg, su, sd, sc;
    const int layer = (int) txc.layer_desc.index;
    if (!txc.tensor_index->resolve_shared(layer, flashnext_projection::gate, sg, &error) ||
        !txc.tensor_index->resolve_shared(layer, flashnext_projection::up, su, &error) ||
        !txc.tensor_index->resolve_shared(layer, flashnext_projection::down, sd, &error) ||
        !txc.tensor_index->resolve_shared(layer, flashnext_projection::shared_control, sc, &error)) return false;

    const std::array<flashnext_tensor_descriptor, 4> descs{{sg, su, sd, sc}};
    std::array<int, 4> fds{{-1, -1, -1, -1}};
    for (size_t p = 0; p < 4; ++p) {
        fds[p] = ::open(descs[p].path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fds[p] < 0) {
            for (size_t c = 0; c < p; ++c) if (fds[c] >= 0) ::close(fds[c]);
            error = "failed to open shared expert shard";
            return false;
        }
    }
    flashnext_shared_expert_slices shared{fds, sg.offset, su.offset, sd.offset, sc.offset};
    const bool shared_ok = flashnext_execute_shared_expert(shared, txc.values);
    for (int & fd : fds) { if (fd >= 0) { ::close(fd); fd = -1; } }
    if (!shared_ok) { error = "shared expert execution failed"; return false; }

    for (size_t i = 0; i < txc.values.complete_moe.size(); ++i) {
        txc.values.complete_moe[i] = txc.values.selected_moe[i] + txc.values.shared_moe[i];
    }
    if (!flashnext_execute_segment_b(txc.values)) { error = "segment B execution failed"; return false; }
    txc.output = txc.values.layer_output;
    txc.state_committed = true;
    return true;
}

static layer_transaction_result run_layer_transaction(
        llama_context * ctx, uint32_t layer, int position, const std::vector<float> & owned_hidden,
        const std::string & index_path) {
    layer_transaction_result result = run_layer_decode(ctx, position, owned_hidden);
    if (result.decode_status != 0) { result.failure_stage = "graph_a_decode"; return result; }
    result.graph_a_ok = true;

    flashnext_transaction_context txc;
    txc.ctx = ctx;
    txc.staged_state = result.state_before;
    std::string error;
    if (!resolve_flashnext_layer_descriptor(ctx, layer, txc.layer_desc, error)) {
        result.failure_stage = "layer_descriptor";
        return result;
    }
    txc.values.layer = (int) txc.layer_desc.index;
    txc.values.position = position;
    flashnext_tensor_index index;
    txc.tensor_index = &index;

    const bool router_capture_ok = capture_transaction_router(txc, ctx, txc.layer_desc);
    const bool sparse_moe_ok = router_capture_ok && execute_transaction_sparse_moe(txc, index_path, error);
    if (!sparse_moe_ok) {
        if (!result.state_before.empty()) llama_state_seq_set_data(ctx, result.state_before.data(), result.state_before.size(), 0);
        result.failure_stage = "capture_or_sparse_moe";
        result.rollback_restored = true;
        return result;
    }
    result.sparse_moe_ok = true;
    result.sparse_output_hash = hash_f32(txc.values.selected_moe.data(), txc.values.selected_moe.size());
    if (!execute_transaction_graph_b(txc, index_path, error)) {
        if (!result.state_before.empty()) llama_state_seq_set_data(ctx, result.state_before.data(), result.state_before.size(), 0);
        result.failure_stage = "graph_b";
        result.rollback_restored = true;
        return result;
    }
    result.graph_b_ok = true;
    result.output = txc.output;
    result.output_hash = hash_f32(result.output.data(), result.output.size());
    result.selected_ids = txc.selected_ids;
    result.unselected_expert_reads = txc.values.io.unselected_expert_reads;
    result.state_committed = true;
    return result;
}

static bool forward_step(
        const std::string & model_path,
        const std::string & index_path,
        int position,
        llama_token token_id,
        std::vector<std::vector<uint8_t>> & layer_states,
        std::vector<float> & out_logits,
        bool compute_logits) {

    // Clean any residual environment variables
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD");
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD_NORM_ONLY");
    unsetenv("CORTEX_FLASHNEXT_LAYER_RANGE");
    unsetenv("CORTEX_FLASHNEXT_PRE_MOE");
    unsetenv("CORTEX_FLASHNEXT_LAYER_START");
    unsetenv("CORTEX_FLASHNEXT_LAYER_END");
    unsetenv("CORTEX_QWEN35_SEGMENT_END");

    // 1. Layer 0 execution
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1);
    setenv("CORTEX_FLASHNEXT_SEGMENT_A", "1", 1);
    setenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER", "1", 1);

    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model0 = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model0) return false;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 1;
    llama_context * ctx0 = llama_init_from_model(model0, cp);
    if (!ctx0) { llama_model_free(model0); return false; }

    // Restore layer 0 state if non-empty
    if (!layer_states[0].empty()) {
        llama_state_seq_set_data(ctx0, layer_states[0].data(), layer_states[0].size(), 0);
    }

    llama_batch batch0 = llama_batch_init(1, 0, 1);
    batch0.token[0] = token_id; batch0.pos[0] = position; batch0.n_seq_id[0] = 1;
    batch0.seq_id[0][0] = 0; batch0.logits[0] = 1; batch0.n_tokens = 1;
    if (llama_decode(ctx0, batch0) != 0) {
        std::fprintf(stderr, "Layer 0 decode failed at pos %d\n", position);
        llama_batch_free(batch0); llama_free(ctx0); llama_model_free(model0); return false;
    }
    llama_synchronize(ctx0);
    layer_states[0] = capture_seq_state(ctx0);
    llama_batch_free(batch0);

    const float * artifact = llama_get_logits_ith(ctx0, 0);
    if (!artifact) { llama_free(ctx0); llama_model_free(model0); return false; }

    flashnext_layer_transaction tx0;
    tx0.layer = 0;
    tx0.position = position;
    std::memcpy(tx0.pre_moe_hidden.data(), artifact, tx0.pre_moe_hidden.size()*sizeof(float));
    std::memcpy(tx0.residual_hc.data(), artifact + 2560, tx0.residual_hc.size()*sizeof(float));
    std::memcpy(tx0.inject.data(), artifact + 2560 + tx0.residual_hc.size(), tx0.inject.size()*sizeof(float));
    std::array<int, 512> order0{};
    for (int i = 0; i < 512; ++i) order0[i] = i;
    const float * route0 = artifact + 12804;
    std::partial_sort(order0.begin(), order0.begin()+10, order0.end(), [route0](int a, int b) { return route0[a] > route0[b]; });
    double z0 = 0.0; double ex0[10]; const float mx0 = route0[order0[0]];
    for (int i = 0; i < 10; ++i) { ex0[i] = std::exp(static_cast<double>(route0[order0[i]]) - static_cast<double>(mx0)); z0 += ex0[i]; }
    for (int i = 0; i < 10; ++i) { tx0.selected_expert_ids[i] = order0[i]; tx0.routing_weights[i] = ex0[i] / z0; }

    flashnext_tensor_index index; std::string index_error;
    if (!index.load(index_path, &index_error)) { llama_free(ctx0); llama_model_free(model0); return false; }
    if (!flashnext_execute_selected_moe(index, index_path, tx0)) { llama_free(ctx0); llama_model_free(model0); return false; }

    flashnext_tensor_descriptor sg, su, sd, sc;
    if (!index.resolve_shared(0, flashnext_projection::gate, sg, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::up, su, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::down, sd, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::shared_control, sc, &index_error)) {
        llama_free(ctx0); llama_model_free(model0); return false;
    }
    const std::array<flashnext_tensor_descriptor, 4> descs0{{sg, su, sd, sc}};
    std::array<int, 4> fds0{{-1, -1, -1, -1}};
    for (size_t p = 0; p < 4; ++p) {
        fds0[p] = ::open(descs0[p].path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fds0[p] < 0) {
            for (size_t c = 0; c < p; ++c) if (fds0[c] >= 0) ::close(fds0[c]);
            llama_free(ctx0); llama_model_free(model0); return false;
        }
    }
    flashnext_shared_expert_slices shared0{fds0, sg.offset, su.offset, sd.offset, sc.offset};
    const bool shared0_ok = flashnext_execute_shared_expert(shared0, tx0);
    for (int & fd : fds0) { if (fd >= 0) { ::close(fd); fd = -1; } }
    if (!shared0_ok) { llama_free(ctx0); llama_model_free(model0); return false; }

    for (size_t i = 0; i < tx0.complete_moe.size(); ++i) tx0.complete_moe[i] = tx0.selected_moe[i] + tx0.shared_moe[i];
    if (!flashnext_execute_segment_b(tx0)) { llama_free(ctx0); llama_model_free(model0); return false; }

    std::vector<float> current_output(tx0.layer_output.begin(), tx0.layer_output.end());
    llama_free(ctx0);
    llama_model_free(model0);

    // 2. Layers 1..47
    unsetenv("CORTEX_FLASHNEXT_ROUTER_ONLY");
    unsetenv("CORTEX_FLASHNEXT_SEGMENT_A");
    unsetenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER");
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
    setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER3_TRANSACTION", "1", 1);

    for (uint32_t il = 1; il < 48; ++il) {
        setenv("CORTEX_LOW_RESIDENCY_LAYER", std::to_string(il).c_str(), 1);
        setenv("CORTEX_FLASHNEXT_LAYER_START", std::to_string(il).c_str(), 1);
        setenv("CORTEX_FLASHNEXT_LAYER_END", std::to_string(il + 1).c_str(), 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", std::to_string(il + 1).c_str(), 1);

        llama_model * next_model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!next_model) return false;
        llama_context_params next_cp = cp;
        next_cp.embeddings = true;
        llama_context * next_ctx = llama_init_from_model(next_model, next_cp);
        if (!next_ctx) { llama_model_free(next_model); return false; }

        if (!layer_states[il].empty()) {
            llama_state_seq_set_data(next_ctx, layer_states[il].data(), layer_states[il].size(), 0);
        }

        layer_transaction_result res = run_layer_transaction(next_ctx, il, position, current_output, index_path);
        if (!res.state_committed) {
            std::fprintf(stderr, "Layer %u transaction failed at pos %d\n", il, position);
            llama_free(next_ctx); llama_model_free(next_model); return false;
        }
        layer_states[il] = capture_seq_state(next_ctx);
        current_output.assign(res.output.begin(), res.output.end());
        llama_free(next_ctx);
        llama_model_free(next_model);
    }

    if (!compute_logits) return true;

    // 3. Final Head Projection
    unsetenv("CORTEX_FLASHNEXT_PRE_MOE");
    unsetenv("CORTEX_FLASHNEXT_LAYER_END");
    unsetenv("CORTEX_QWEN35_SEGMENT_END");
    setenv("CORTEX_FLASHNEXT_LAYER_START", "48", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_END", "48", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD", "1", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD_NORM_ONLY", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "47", 1);

    llama_model * head_model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!head_model) return false;
    llama_context_params head_cp = cp;
    head_cp.embeddings = true;
    llama_context * head_ctx = llama_init_from_model(head_model, head_cp);
    if (!head_ctx) { llama_model_free(head_model); return false; }

    llama_batch head_batch = llama_batch_init(1, (int) current_output.size(), 1);
    std::memcpy(head_batch.embd, current_output.data(), current_output.size() * sizeof(float));
    head_batch.pos[0] = 0; // head has no KV state, pos 0 is clean
    head_batch.n_seq_id[0] = 1;
    head_batch.seq_id[0][0] = 0;
    head_batch.logits[0] = 1;
    head_batch.n_tokens = 1;
    const int head_rc = llama_decode(head_ctx, head_batch);
    llama_synchronize(head_ctx);
    llama_batch_free(head_batch);

    if (head_rc != 0) {
        std::fprintf(stderr, "Head decode failed at pos %d\n", position);
        llama_free(head_ctx); llama_model_free(head_model); return false;
    }

    std::vector<uint8_t> norm_bytes;
    std::array<int64_t, 4> norm_ne{};
    std::array<size_t, 4> norm_nb{};
    enum ggml_type norm_type = GGML_TYPE_COUNT;
    std::string norm_name;
    const bool norm_read = head_ctx->debug_copy_final_head_output(norm_bytes, norm_ne, norm_nb, norm_type, norm_name);
    llama_free(head_ctx);
    llama_model_free(head_model);

    if (!norm_read || norm_type != GGML_TYPE_F32 || norm_bytes.size() != 2560 * sizeof(float)) {
        std::fprintf(stderr, "Head norm copy failed\n");
        return false;
    }

    const float * norm_p = (const float *) norm_bytes.data();
    std::vector<float> norm_owned(norm_p, norm_p + 2560);

    flashnext_tensor_descriptor out_desc;
    if (!index.resolve_named("output.weight", out_desc, &index_error)) return false;
    if (!flashnext_stream_output_q6k(out_desc, norm_owned, out_logits)) return false;

    return true;
}


int main(int argc, char ** argv) {
    std::string model_path = "/home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf";
    if (argc >= 2) {
        model_path = argv[1];
    }
    const std::string index_path = "/home/user/bitnet-cortex/cortex-small-model/flashnext-tensor-index.json";
    int max_tokens = 32;
    if (argc >= 3) {
        max_tokens = std::atoi(argv[2]);
    }

    llama_backend_init();

    llama_model_params v_mp = llama_model_default_params();
    v_mp.vocab_only = true;
    llama_model * v_model = llama_model_load_from_file(model_path.c_str(), v_mp);
    if (!v_model) {
        std::fprintf(stderr, "Failed to load vocabulary\n");
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(v_model);

    std::vector<std::vector<uint8_t>> layer_states(48);
    int current_pos = 0;

    std::cout << "\n======================================================\n";
    std::cout << "  Cortex FlashNext 90.78 GiB Chat Runtime (48 Layers)  \n";
    std::cout << "  Model: Qwen 3.8 Flash Next Abliterated (IQ4_XS)     \n";
    std::cout << "  Hardware: AMD Athlon Silver 3050U (2 Cores, AVX2)   \n";
    std::cout << "======================================================\n";
    std::cout << "Type /exit to quit.\n\n" << std::flush;

    std::string user_input;
    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, user_input)) break;
        if (user_input == "/exit") break;
        if (user_input.empty()) continue;

        std::string turn_prompt = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
        int n_toks = -llama_tokenize(vocab, turn_prompt.c_str(), (int32_t) turn_prompt.size(), nullptr, 0, false, true);
        std::vector<llama_token> prompt_toks(n_toks);
        llama_tokenize(vocab, turn_prompt.c_str(), (int32_t) turn_prompt.size(), prompt_toks.data(), n_toks, false, true);

        std::cout << "Cortex FlashNext: " << std::flush;

        // Prefill prompt tokens
        std::vector<float> logits;
        for (size_t i = 0; i < prompt_toks.size(); ++i) {
            const bool is_last = (i + 1 == prompt_toks.size());
            if (!forward_step(model_path, index_path, current_pos, prompt_toks[i], layer_states, logits, is_last)) {
                std::fprintf(stderr, "\n[ERROR] Prefill step failed at pos %d\n", current_pos);
                break;
            }
            current_pos++;
        }

        // Autoregressive generation
        for (int gen_step = 0; gen_step < max_tokens; ++gen_step) {
            if (logits.empty()) break;

            llama_token next_token = (llama_token) std::distance(logits.begin(),
                std::max_element(logits.begin(), logits.end()));

            // Stop tokens
            if (next_token == 248046 || next_token == 248044) {
                break;
            }

            char piece[128] = {0};
            int piece_len = llama_token_to_piece(vocab, next_token, piece, sizeof(piece), 0, false);
            if (piece_len > 0) {
                std::cout.write(piece, piece_len);
                std::cout.flush();
            }

            if (!forward_step(model_path, index_path, current_pos, next_token, layer_states, logits, true)) {
                std::fprintf(stderr, "\n[ERROR] Generation step failed at pos %d\n", current_pos);
                break;
            }
            current_pos++;
        }
        std::cout << "\n\n" << std::flush;
    }

    llama_model_free(v_model);
    llama_backend_free();
    return 0;
}
