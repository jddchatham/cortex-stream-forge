#include "../src/llama-graph.h"
#include "../src/llama-batch.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

static bool equal(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i] || !std::isfinite(a[i])) return false;
    }
    return true;
}

int main() {
    constexpr size_t width = 10240;
    constexpr size_t hc_width = 2560;

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return 1;

    ggml_init_params params = { 4*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { ggml_backend_free(backend); return 2; }

    ggml_tensor * input_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
    ggml_tensor * hc = ggml_reshape_3d(ctx, input_tensor, hc_width, 4, 1);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!input_tensor || !hc || !buffer) return 3;

    std::vector<float> first(width), second(width), received(width);
    for (size_t i = 0; i < width; ++i) {
        first[i] = (float) i * 0.25f - 17.0f;
        second[i] = (float) (width - i) * 0.5f + 3.0f;
    }

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->embd = first;
    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    ubatch.n_seq_tokens = 1;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.embd = data->embd.data();
    ubatch.data = data;

    llm_graph_input_embd input((int64_t) width);
    input.embd = input_tensor;
    if (!llm_graph_validate_flashnext_range_input(&ubatch, width)) return 4;
    input.set_input(&ubatch);
    ggml_backend_tensor_get(input_tensor, received.data(), 0, received.size()*sizeof(float));
    if (!equal(first, received)) return 5;
    if (hc->ne[0] != (int64_t) hc_width || hc->ne[1] != 4 || hc->ne[2] != 1) return 6;

    data->embd = second;
    ubatch.embd = data->embd.data();
    input.set_input(&ubatch);
    ggml_backend_tensor_get(input_tensor, received.data(), 0, received.size()*sizeof(float));
    if (!equal(second, received)) return 7;

    llama_ubatch missing{};
    if (llm_graph_validate_flashnext_range_input(&missing, width)) return 8;
    auto undersized = std::make_shared<llama_ubatch::data_t>();
    undersized->embd.resize(width - 1);
    llama_ubatch short_batch{};
    short_batch.data = undersized;
    short_batch.embd = undersized->embd.data();
    if (llm_graph_validate_flashnext_range_input(&short_batch, width)) return 9;

    // The production binder requires an F32 destination; an I32 destination
    // is rejected by its GGML assertion and is intentionally not executed here.
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::puts("flashnext activation binding: PASS");
    return 0;
}
