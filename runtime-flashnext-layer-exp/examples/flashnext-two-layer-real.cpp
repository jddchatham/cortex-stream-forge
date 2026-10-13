#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "llama-context.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

static bool read_exact(int fd, void * dst, size_t bytes, off_t off) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, (uint8_t *) dst + done, bytes - done, off + (off_t) done);
        if (n <= 0) return false;
        done += (size_t) n;
    }
    return true;
}

static bool project_vocab(const cortex_output_weight_descriptor & d, size_t tile_rows,
        const std::vector<float> & hidden, int & best, float & best_logit, size_t & bytes_read) {
    if (d.file_descriptor < 0 || d.ne[0] != (int64_t) hidden.size() || tile_rows == 0 ||
            tile_rows > SIZE_MAX / d.row_bytes || tile_rows * d.row_bytes > 16u * 1024u * 1024u) return false;
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return false;
    std::vector<uint8_t> scratch(tile_rows * d.row_bytes);
    best = -1; best_logit = -INFINITY; bytes_read = 0;
    for (int64_t row = 0; row < d.ne[1]; row += (int64_t) tile_rows) {
        const int64_t nr = std::min<int64_t>((int64_t) tile_rows, d.ne[1] - row);
        const size_t bytes = (size_t) nr * d.row_bytes;
        if (!read_exact(d.file_descriptor, scratch.data(), bytes, (off_t) d.absolute_offset + row * (off_t) d.row_bytes)) { ggml_backend_free(backend); return false; }
        bytes_read += bytes;
        ggml_context * c = ggml_init({ 64u * 1024u * 1024u, nullptr, true });
        if (!c) { ggml_backend_free(backend); return false; }
        ggml_tensor * w = ggml_new_tensor_2d(c, d.type, d.ne[0], nr);
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, d.ne[0], 1);
        ggml_tensor * y = ggml_mul_mat(c, w, x);
        ggml_cgraph * g = ggml_new_graph(c); ggml_build_forward_expand(g, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, backend);
        bool ok = buf != nullptr;
        if (ok) {
            ggml_backend_tensor_set(w, scratch.data(), 0, bytes);
            ggml_backend_tensor_set(x, hidden.data(), 0, hidden.size() * sizeof(float));
            ok = ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(backend);
        }
        std::vector<float> logits((size_t) nr);
        if (ok) ggml_backend_tensor_get(y, logits.data(), 0, logits.size() * sizeof(float));
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(c);
        if (!ok) { ggml_backend_free(backend); return false; }
        for (int64_t i = 0; i < nr; ++i) if (logits[(size_t) i] > best_logit) { best_logit = logits[(size_t) i]; best = (int) (row + i); }
    }
    ggml_backend_free(backend);
    return best >= 0;
}

static bool set_layer_env(int layer) {
    char s[32], e[32];
    std::snprintf(s, sizeof(s), "%d", layer);
    std::snprintf(e, sizeof(e), "%d", layer + 1);
    return setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1) == 0 &&
        setenv("CORTEX_FLASHNEXT_LAYER_START", s, 1) == 0 &&
        setenv("CORTEX_FLASHNEXT_LAYER_END", e, 1) == 0;
}

static llama_batch make_batch(const std::vector<float> & embd, int n_embd, int hc, llama_token token) {
    const int width = hc > 1 ? n_embd * hc : n_embd;
    llama_batch b = llama_batch_init(1, width, 1);
    b.token = (llama_token *) std::malloc(sizeof(llama_token));
    if (!b.token || !b.embd) return b;
    b.token[0] = token;
    b.embd[0] = 0.0f;
    for (size_t i = 0; i < embd.size(); ++i) b.embd[i] = embd[i];
    b.pos[0] = 0;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    // Intermediate layer execution consumes the HC embedding output; asking
    // for vocabulary logits also creates the out_ids graph input, which is
    // unnecessary before the final head.
    b.logits[0] = 0;
    b.n_tokens = 1;
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 4) return 2;
    const llama_token token = argc == 3 ? (llama_token) std::strtol(argv[2], nullptr, 10) : 1;
    const int requested_layers = argc == 4 ? std::atoi(argv[3]) : 2;
    setenv("CORTEX_FLASHNEXT_EXTERNAL_EMBEDDING", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    // Retain the small real final-HC parameter set during model loading;
    // final-head graph mode is disabled immediately afterward for layers.
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD", "1", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY", "1", 1);
    set_layer_env(0);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "TWO_LAYER stage=MODEL_INITIALIZED result=FAIL\n"); return 3; }
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD");
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY");
    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int hc = 4;
    if (n_layer < 2) { std::fprintf(stderr, "TWO_LAYER result=FAIL reason=layers=%d\n", n_layer); llama_model_free(model); return 4; }
    std::fprintf(stderr, "TWO_LAYER stage=MODEL_INITIALIZED layers=%d n_embd=%d hc=%d\n", n_layer, n_embd, hc);

    std::vector<int32_t> ids{ token };
    std::vector<float> embd((size_t) n_embd);
    if (!llama_cortex_external_token_rows(model, ids.data(), ids.size(), embd.data())) {
        std::fprintf(stderr, "TWO_LAYER stage=REAL_EMBEDDING result=FAIL\n"); llama_model_free(model); return 5;
    }
    std::fprintf(stderr, "TWO_LAYER stage=REAL_EMBEDDING floats=%zu\n", embd.size());

    std::vector<uint8_t> state;
    std::vector<float> hc_out;
    const int run_layers = std::max(1, std::min(requested_layers, n_layer));
    for (int layer = 0; layer < run_layers; ++layer) {
        set_layer_env(layer);
        const bool active = layer == 0 ? llama_cortex_rebind_layer(model, layer) : llama_cortex_activate_layer(model, layer);
        if (!active) { std::fprintf(stderr, "TWO_LAYER stage=LAYER_BOUND layer=%d result=FAIL\n", layer); llama_model_free(model); return 6; }
        std::fprintf(stderr, "TWO_LAYER stage=LAYER_BOUND layer=%d result=PASS\n", layer);
        llama_context_params cp = llama_context_default_params(); cp.n_ctx = 4; cp.n_batch = 1;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { std::fprintf(stderr, "TWO_LAYER stage=CONTEXT layer=%d result=FAIL\n", layer); llama_model_free(model); return 7; }
        const bool terminal_layer = layer + 1 == run_layers;
        if (terminal_layer) llama_set_embeddings(ctx, true);
        else llama_set_embeddings_layer_inp(ctx, layer + 1, true);
        // Each physical layer owns a distinct recurrent-memory slot.  The
        // fresh layer-1 context must not import layer-0's serialized position
        // metadata; layer 1 starts its own state transaction at position 0.
        if (layer == 0 && !state.empty() && llama_state_set_data(ctx, state.data(), state.size()) == 0) {
            std::fprintf(stderr, "TWO_LAYER stage=STATE_RESTORE layer=%d result=FAIL\n", layer); llama_free(ctx); llama_model_free(model); return 8;
        }
        llama_batch b = make_batch(embd, n_embd, layer == 0 ? 1 : hc, token);
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) { std::fprintf(stderr, "TWO_LAYER stage=DECODE layer=%d result=FAIL rc=%d\n", layer, rc); llama_free(ctx); llama_model_free(model); return 9; }
        std::fprintf(stderr, "TWO_LAYER stage=REAL_LAYER_EXECUTED layer=%d result=PASS\n", layer);
        const size_t state_size = llama_state_get_size(ctx);
        state.resize(state_size);
        if (llama_state_get_data(ctx, state.data(), state.size()) != state.size()) {
            std::fprintf(stderr, "TWO_LAYER stage=STATE_CAPTURE layer=%d result=FAIL\n", layer); llama_free(ctx); llama_model_free(model); return 10;
        }
        {
            float * out = terminal_layer ? llama_get_embeddings(ctx) : llama_get_embeddings_layer_inp(ctx, layer + 1);
            if (!out) { std::fprintf(stderr, "TWO_LAYER stage=HC_HANDOFF layer=%d result=FAIL\n", layer); llama_free(ctx); llama_model_free(model); return 11; }
            hc_out.assign(out, out + (size_t) n_embd * hc);
            size_t finite = 0;
            for (float v : hc_out) if (std::isfinite(v)) ++finite;
            if (finite != hc_out.size()) { std::fprintf(stderr, "TWO_LAYER stage=HC_HANDOFF layer=%d finite=%zu/%zu result=FAIL\n", layer, finite, hc_out.size()); llama_free(ctx); llama_model_free(model); return 11; }
            embd = hc_out;
            std::fprintf(stderr, "TWO_LAYER stage=HC_HANDOFF layer=%d shape=[%d,%d,1] floats=%zu finite=%zu\n", layer, n_embd, hc, embd.size(), finite);
        }
        llama_free(ctx);
    }
    std::fprintf(stderr, "TWO_LAYER stage=COMPLETE layers=0..%d count=%d real_payloads=1\n", run_layers - 1, run_layers);
    if (run_layers == n_layer) {
        // The final-head-only graph has no segmented layer-input output.  Do
        // not carry the layer-range extraction mode into its context
        // reservation; the model remains low-residency and no layer loop is
        // re-entered.
        unsetenv("CORTEX_FLASHNEXT_LAYER_RANGE");
        unsetenv("CORTEX_FLASHNEXT_LAYER_START");
        unsetenv("CORTEX_FLASHNEXT_LAYER_END");
        setenv("CORTEX_FLASHNEXT_SEGMENT_B", "1", 1);
        setenv("CORTEX_FLASHNEXT_FINAL_HEAD", "1", 1);
        setenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY", "1", 1);
        llama_context_params cp = llama_context_default_params(); cp.n_ctx = 1; cp.n_batch = 1;
        llama_context * head_ctx = llama_init_from_model(model, cp);
        cortex_post_moe_result normalized;
        bool ok = head_ctx != nullptr && reinterpret_cast<llama_context *>(head_ctx)->cortex_execute_final_head_only(embd.data(), embd.size(), normalized);
        if (ok) {
            size_t finite = 0, nonzero = 0; float mn = INFINITY, mx = -INFINITY; double ss = 0;
            for (float v : normalized.hc_output) { finite += std::isfinite(v); nonzero += v != 0.0f; mn = std::min(mn, v); mx = std::max(mx, v); ss += double(v) * v; }
            std::fprintf(stderr, "TWO_LAYER stage=FINAL_HEAD shape=[%lld,%lld,%lld] floats=%zu finite=%zu nonzero=%zu min=%g max=%g rms=%g\n", (long long) normalized.ne[0], (long long) normalized.ne[1], (long long) normalized.ne[2], normalized.hc_output.size(), finite, nonzero, mn, mx, std::sqrt(ss / std::max<size_t>(1, normalized.hc_output.size())));
            cortex_output_weight_descriptor d;
            ok = reinterpret_cast<llama_model_base *>(model)->get_cortex_output_weight_descriptor(d);
            int best = -1; float logit = 0; size_t bytes = 0;
            ok = ok && project_vocab(d, 1024, normalized.hc_output, best, logit, bytes);
            std::fprintf(stderr, "TWO_LAYER stage=VOCAB_PROJECTOR argmax=%d logit=%g bytes_read=%zu scratch_bytes=%zu result=%s\n", best, logit, bytes, 1024 * d.row_bytes, ok ? "PASS" : "FAIL");
        } else std::fprintf(stderr, "TWO_LAYER stage=FINAL_HEAD result=FAIL\n");
        if (head_ctx) llama_free(head_ctx);
        if (!ok) { llama_model_free(model); llama_backend_free(); return 12; }
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
