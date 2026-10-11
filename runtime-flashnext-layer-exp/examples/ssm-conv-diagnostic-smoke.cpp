#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool run_case(bool nan_input, bool nan_weight, bool diagnostics, std::vector<float> & out) {
    if (diagnostics) setenv("CORTEX_FLASHNEXT_SSM_CONV_OPERAND_DIAGNOSTICS", "1", 1);
    else unsetenv("CORTEX_FLASHNEXT_SSM_CONV_OPERAND_DIAGNOSTICS");

    ggml_init_params ip{};
    ip.mem_size = 1u << 20;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    // ggml_ssm_conv expects src0=[d_conv-1+n_tokens,d_inner,n_seq]
    // and weights=[d_conv,d_inner].
    constexpr int64_t d_conv = 3;
    constexpr int64_t d_inner = 4;
    constexpr int64_t n_tokens = 2;
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_conv - 1 + n_tokens, d_inner, 1);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
    ggml_tensor * result = ggml_ssm_conv(ctx, input, weight);
    const size_t in_count = ggml_nelements(input);
    const size_t wt_count = ggml_nelements(weight);
    std::vector<float> in(in_count), wt(wt_count);
    for (size_t i = 0; i < in.size(); ++i) in[i] = 0.25f + 0.01f * (float) i;
    for (size_t i = 0; i < wt.size(); ++i) wt[i] = 0.5f - 0.02f * (float) i;
    if (nan_input) in[1] = NAN;
    if (nan_weight) wt[2] = NAN;

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) { ggml_free(ctx); return false; }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    std::vector<ggml_backend_buffer_t> buffers;
    for (ggml_tensor * t : { input, weight, result }) {
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(t));
        if (!b || ggml_backend_tensor_alloc(b, t, ggml_backend_buffer_get_base(b)) != GGML_STATUS_SUCCESS) {
            for (auto * old : buffers) ggml_backend_buffer_free(old);
            if (b) ggml_backend_buffer_free(b);
            ggml_backend_free(backend); ggml_free(ctx); return false;
        }
        buffers.push_back(b);
    }
    ggml_backend_tensor_set(input, in.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(weight, wt.data(), 0, ggml_nbytes(weight));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, result);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        for (auto * b : buffers) ggml_backend_buffer_free(b);
        ggml_backend_free(backend); ggml_free(ctx); return false;
    }
    out.resize(ggml_nelements(result));
    ggml_backend_tensor_get(result, out.data(), 0, ggml_nbytes(result));
    bool finite = true;
    for (float v : out) finite = finite && std::isfinite(v);
    std::fprintf(stderr, "SSM_CONV_SMOKE input_nan=%d weight_nan=%d diagnostics=%d output_finite=%d output_elements=%zu\n",
        nan_input, nan_weight, diagnostics, finite, out.size());
    for (auto * b : buffers) ggml_backend_buffer_free(b);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

int main() {
    ggml_backend_load_all();
    std::vector<float> baseline, enabled, nan_in, nan_w;
    if (!run_case(false, false, false, baseline) || !run_case(false, false, true, enabled)) return 2;
    if (baseline.size() != enabled.size()) return 3;
    for (size_t i = 0; i < baseline.size(); ++i) if (baseline[i] != enabled[i]) return 4;
    if (!run_case(true, false, true, nan_in)) return 5;
    if (!run_case(false, true, true, nan_w)) return 6;
    std::puts("SSM_CONV_SMOKE PASS");
    return 0;
}
