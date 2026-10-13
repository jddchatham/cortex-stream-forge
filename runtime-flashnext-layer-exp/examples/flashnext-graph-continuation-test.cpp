#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {
constexpr size_t MAX_TOKENS = 64;
constexpr size_t WIDTH = 2560;

struct boundary {
    size_t width = 0;
    size_t tokens = 0;
    ggml_type type = GGML_TYPE_COUNT;
    std::vector<float> data;
    bool valid = false;

    bool allocate(size_t w, size_t n, ggml_type t = GGML_TYPE_F32) {
        valid = false;
        data.clear();
        if (t != GGML_TYPE_F32 || w == 0 || n == 0 || n > MAX_TOKENS ||
            w > std::numeric_limits<size_t>::max() / n ||
            w * n > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
        width = w; tokens = n; type = t;
        data.resize(w * n);
        valid = true;
        return true;
    }

    bool copy_from(const float * src, size_t bytes) {
        if (!valid || !src || bytes != data.size() * sizeof(float)) {
            valid = false;
            return false;
        }
        std::copy(src, src + data.size(), data.begin());
        return true;
    }
};

bool close_enough(const std::vector<float> & a, const std::vector<float> & b, double & max_abs, double & max_rel) {
    if (a.size() != b.size()) return false;
    max_abs = max_rel = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double da = std::abs((double) a[i] - b[i]);
        const double dr = da / std::max(1.0, std::abs((double) b[i]));
        max_abs = std::max(max_abs, da);
        max_rel = std::max(max_rel, dr);
    }
    return max_abs <= 1e-5 && max_rel <= 1e-5;
}

bool graph_a(ggml_backend_t backend, const std::vector<float> & input, boundary & out) {
    ggml_init_params p { 8 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;
    const size_t n = input.size();
    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, WIDTH, n / WIDTH);
    ggml_tensor * one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * y = ggml_add(ctx, ggml_scale(ctx, in, 2.0f), one);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(in, input.data(), 0, input.size() * sizeof(float));
    float one_value = 1.0f;
    ggml_backend_tensor_set(one, &one_value, 0, sizeof(one_value));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> tmp(n);
    if (ok) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(y, tmp.data(), 0, tmp.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok && out.allocate(WIDTH, n / WIDTH) && out.copy_from(tmp.data(), tmp.size() * sizeof(float));
}

bool graph_b(ggml_backend_t backend, const boundary & in_boundary, std::vector<float> & output) {
    if (!in_boundary.valid) return false;
    ggml_init_params p { 8 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;
    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_boundary.width, in_boundary.tokens);
    ggml_tensor * half = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * y = ggml_sub(ctx, ggml_scale(ctx, in, 3.0f), half);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(in, in_boundary.data.data(), 0, in_boundary.data.size() * sizeof(float));
    float half_value = 0.5f;
    ggml_backend_tensor_set(half, &half_value, 0, sizeof(half_value));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    output.resize(in_boundary.data.size());
    if (ok) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return 2;

    for (size_t tokens : { size_t(1), size_t(3), MAX_TOKENS }) {
        std::vector<float> input(WIDTH * tokens), reference(input.size()), result;
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::sin((float) i * 0.013f);
            reference[i] = (input[i] * 2.0f + 1.0f) * 3.0f - 0.5f;
        }
        boundary owned;
        if (!graph_a(backend, input, owned)) return 3;
        if (!graph_b(backend, owned, result)) return 4;
        double abs_error = 0, rel_error = 0;
        if (!close_enough(result, reference, abs_error, rel_error)) return 5;
        std::printf("tokens=%zu elements=%zu max_abs=%g max_rel=%g PASS\n", tokens, result.size(), abs_error, rel_error);
    }

    boundary bad;
    if (bad.allocate(WIDTH, MAX_TOKENS + 1)) return 6;
    if (!bad.allocate(WIDTH, 2) || bad.copy_from(nullptr, 0)) return 7;
    std::vector<float> output;
    if (graph_b(backend, bad, output)) return 8;

    ggml_backend_free(backend);
    std::puts("FLASHNEXT_GRAPH_CONTINUATION PASS");
    std::puts("Graph A synchronized and destroyed before Graph B creation");
    std::puts("owned boundary, multi-token limit, invalid transfer and stale-input rejection: PASS");
    return 0;
}
