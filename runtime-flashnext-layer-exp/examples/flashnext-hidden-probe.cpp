#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "../src/flashnext-layer-transaction.h"

static uint64_t fnv1a(const float * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        uint32_t u; std::memcpy(&u, p + i, sizeof(u));
        h ^= u; h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 7) return 2;
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    const bool segment_a = argc >= 4 && (std::strcmp(argv[3], "segment-a") == 0 || std::strcmp(argv[3], "segment-a-router") == 0);
    const bool segment_a_router = argc >= 4 && std::strcmp(argv[3], "segment-a-router") == 0;
    const bool segment_b = argc >= 5 && std::strcmp(argv[3], "segment-b") == 0;
    const bool control_copy = segment_b && argc >= 6 && std::strcmp(argv[5], "control-copy") == 0;
    const bool control_res = segment_b && argc >= 6 && std::strcmp(argv[5], "control-res") == 0;
    if (!segment_b) {
        setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1);
        setenv("CORTEX_FLASHNEXT_EXPORT_HIDDEN", "1", 1);
    }
    if (segment_a) {
        setenv("CORTEX_FLASHNEXT_SEGMENT_A", "1", 1);
        if (segment_a_router) setenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER", "1", 1);
    }
    if (segment_b) {
        setenv("CORTEX_FLASHNEXT_SEGMENT_B", "1", 1);
        setenv("CORTEX_FLASHNEXT_SEGMENT_B_INPUT_WIDTH", "12804", 1);
        setenv("CORTEX_FLASHNEXT_SEGMENT_B_DEBUG_INPUT", "1", 1);
        if (control_copy) setenv("CORTEX_FLASHNEXT_SEGMENT_B_CONTROL_COPY", "1", 1);
        if (control_res) setenv("CORTEX_FLASHNEXT_SEGMENT_B_CONTROL_RES", "1", 1);
        if (argc >= 7) setenv("CORTEX_FLASHNEXT_SEGMENT_B_HC_CONTROL", argv[6], 1);
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 256; cp.n_batch = 1;
    if (segment_b) cp.embeddings = true;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return 4;
    const int embd_width = segment_b ? 2560 * 5 + 4 : 0;
    llama_batch b = llama_batch_init(1, embd_width, 1);
    if (segment_b) {
        FILE * in = std::fopen(argv[4], "rb"); if (!in) return 9;
        if (std::fread(b.embd, sizeof(float), embd_width, in) != (size_t) embd_width) return 10;
        std::fclose(in);
    } else {
        b.token[0] = 248045;
    }
    b.pos[0] = 0; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0;
    b.logits[0] = 1; b.n_tokens = 1;
    if (llama_decode(ctx, b) != 0) return 5;
    // The segmented probe reads an asynchronously scheduled graph result;
    // force the backend completion before copying the transaction output.
    llama_synchronize(ctx);
    const float * all = segment_b ? llama_get_embeddings(ctx) : llama_get_logits_ith(ctx, 0); if (!all) return 6;
    if (segment_a_router) {
        // Own the Segment-A/router boundary in-process.  This is the first
        // transaction stage; no temporary artifact is needed to preserve the
        // values for subsequent transaction stages.
        flashnext_layer_transaction tx;
        std::memcpy(tx.pre_moe_hidden.data(), all, tx.hidden * sizeof(float));
        std::memcpy(tx.residual_hc.data(), all + tx.hidden, tx.residual_hc.size() * sizeof(float));
        std::memcpy(tx.inject.data(), all + tx.hidden + tx.residual_hc.size(), tx.inject.size() * sizeof(float));
        const float * route = all + tx.hidden + tx.residual_hc.size() + tx.inject.size();
        std::array<int, 512> order{};
        for (int i = 0; i < 512; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + tx.experts_used, order.end(),
            [route](int a, int b) { return route[a] > route[b]; });
        float mx = route[order[0]], z = 0.0f;
        std::array<double, tx.experts_used> ex{};
        for (size_t i = 0; i < tx.experts_used; ++i) { ex[i] = std::exp(route[order[i]] - mx); z += ex[i]; }
        for (size_t i = 0; i < tx.experts_used; ++i) {
            tx.selected_expert_ids[i] = order[i];
            tx.routing_weights[i] = ex[i] / z;
            std::printf("CORTEX_TX_ROUTE slot=%zu expert=%d weight=%.12g\n", i, order[i], tx.routing_weights[i]);
        }
        std::printf("CORTEX FLASHNEXT OWNED PREFIX: PASS pre_moe=%zu residual=%zu inject=%zu\n",
            tx.pre_moe_hidden.size(), tx.residual_hc.size(), tx.inject.size());
    }
    const size_t hidden = segment_b ? 2560 * 4 : (segment_a_router ? 248320 : (segment_a ? 2560 * 5 + 4 : 2560));
    FILE * f = std::fopen(argv[2], "wb"); if (!f) return 7;
    if (std::fwrite(all, sizeof(float), hidden, f) != hidden) return 8;
    std::fclose(f);
    double sum = 0, norm = 0; float mn = all[0], mx = all[0];
    for (size_t i = 0; i < hidden; ++i) { mn = std::min(mn, all[i]); mx = std::max(mx, all[i]); sum += all[i]; norm += (double) all[i] * all[i]; }
    std::printf("FLASHNEXT_HIDDEN shape=[%zu] dtype=f32 hash=%016llx min=%.9g max=%.9g mean=%.9g l2=%.9g output=%s\n",
        hidden, (unsigned long long) fnv1a(all, hidden), mn, mx, sum/hidden, std::sqrt(norm), argv[2]);
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
}
