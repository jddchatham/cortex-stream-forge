#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t hash_f32(const float * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { uint32_t u; std::memcpy(&u, p + i, 4); h ^= u; h *= 1099511628211ull; }
    return h;
}

int main() {
    constexpr int64_t N = 16, STREAMS = 4;
    constexpr float eps = 1e-5f;
    std::vector<float> input(N*STREAMS), weight(N*STREAMS), expected(N*STREAMS), actual(N*STREAMS);
    for (int s = 0; s < STREAMS; ++s) for (int i = 0; i < N; ++i) {
        input[s*N+i] = (float) (s + 1) * (0.25f + (i % 5) * 0.125f);
        weight[s*N+i] = 0.5f + 0.25f*s + 0.03125f*i;
        expected[s*N+i] = input[s*N+i];
    }
    for (int s = 0; s < STREAMS; ++s) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) sum += (double) input[s*N+i] * input[s*N+i];
        const float scale = 1.0f / std::sqrt((float) (sum/N) + eps);
        for (int i = 0; i < N; ++i) expected[s*N+i] = input[s*N+i] * scale * weight[s*N+i];
    }
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_context * ctx = backend ? ggml_init({4ull*1024*1024, nullptr, true}) : nullptr;
    if (!ctx) return 2;
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, STREAMS);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, STREAMS);
    ggml_tensor * y = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
    ggml_set_name(x, "micro_input"); ggml_set_name(w, "micro_weight"); ggml_set_name(y, "micro_weighted");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 3;
    ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
    ggml_backend_tensor_set(w, weight.data(), 0, weight.size()*sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 4;
    ggml_backend_tensor_get(y, actual.data(), 0, actual.size()*sizeof(float));
    size_t exact = 0, first = actual.size(); float max_abs = 0.0f; long double ss = 0.0L;
    for (size_t i = 0; i < actual.size(); ++i) { const float d = std::fabs(expected[i]-actual[i]); exact += d == 0.0f; max_abs = std::max(max_abs,d); ss += (long double)d*d; if (d && first == actual.size()) first=i; }
    std::printf("MICROTEST_FUSION_REFERENCE_PASS=%s EXACT=%zu TOTAL=%zu MAX_ABS=%.9g RMS=%.9g FIRST=%zu EXPECTED_HASH=%016llx ACTUAL_HASH=%016llx\n",
        exact == actual.size() ? "YES" : "NO", exact, actual.size(), max_abs, std::sqrt((double)(ss/actual.size())), first,
        (unsigned long long) hash_f32(expected.data(), expected.size()), (unsigned long long) hash_f32(actual.data(), actual.size()));
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend); return exact == actual.size() ? 0 : 5;
}
