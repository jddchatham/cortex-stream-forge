#include "../src/llama-context.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr size_t count = 10240;
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return 1;
    ggml_init_params params = { 2*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return 2;
    ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 4);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!output || !buffer) return 3;

    std::vector<float> expected(count), captured(count);
    for (size_t i = 0; i < count; ++i) expected[i] = (float) i * 0.125f - 9.0f;
    ggml_backend_tensor_set(output, expected.data(), 0, expected.size()*sizeof(float));
    if (!llama_copy_graph_output_f32(backend, output, captured.data(), captured.size(), count)) return 4;
    for (size_t i = 0; i < count; ++i) if (captured[i] != expected[i]) return 5;
    if (output->ne[0] != 2560 || output->ne[1] != 4 || output->ne[2] != 1) return 6;

    if (llama_copy_graph_output_f32(backend, output, captured.data(), count - 1, count)) return 7;
    if (llama_copy_graph_output_f32(backend, output, captured.data(), count, count + 1)) return 8;
    ggml_tensor * missing = nullptr;
    if (llama_copy_graph_output_f32(backend, missing, captured.data(), count, count)) return 9;

    // A non-F32 output is rejected before any copy, preserving the production
    // contract instead of relying on an unsafe reinterpretation.
    ggml_tensor * wrong_type = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2560, 4);
    ggml_backend_buffer_t wrong_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!wrong_type || !wrong_buffer) return 10;
    if (llama_copy_graph_output_f32(backend, wrong_type, captured.data(), count, count)) return 11;

    ggml_backend_buffer_free(wrong_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::puts("flashnext output capture: PASS");
    return 0;
}
