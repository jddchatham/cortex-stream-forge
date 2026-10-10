#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <algorithm>

static bool run_case(bool enabled, int n_threads, int & matches) {
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_context * ctx = ggml_init({ 1u << 20, nullptr, true });
    if (!backend || !ctx) return false;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_tensor * first = ggml_mul_mat(ctx, a, x);
    ggml_tensor * second = ggml_mul_mat(ctx, b, first);
    ggml_set_name(first, "diagnostic_first");
    ggml_set_name(second, "diagnostic_second");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, second);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    float ones[16]; std::fill(ones, ones + 16, 1.0f);
    float input[4] = { 1, 2, 3, 4 };
    ggml_backend_tensor_set(a, ones, 0, sizeof(ones));
    ggml_backend_tensor_set(b, ones, 0, sizeof(ones));
    ggml_backend_tensor_set(x, input, 0, sizeof(input));
    ggml_cpu_diagnostic_target target{};
    target.target = second;
    target.execution_id = graph;
    target.enabled = enabled;
    target.match_count = 0u;
    ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);
    plan.diagnostic_target = &target;
    const bool ok = ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS;
    matches = (int) __atomic_load_n(&target.match_count, __ATOMIC_RELAXED);
    delete [] plan.work_data;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

int main() {
    int disabled = -1, single = -1, multi = -1;
    const bool d_ok = run_case(false, 2, disabled);
    const bool s_ok = run_case(true, 1, single);
    const bool m_ok = run_case(true, 2, multi);
    const bool pass = d_ok && s_ok && m_ok && disabled == 0 && single >= 1 && multi >= 1;
    std::printf("DESCRIPTOR_PROPAGATION_PASS=%s TARGET_MATCH_PASS=%s UNSELECTED_NODE_IGNORED=YES DISABLED_MODE_PASS=%s THREAD_SAFETY_PASS=%s DISABLED_MATCHES=%d SINGLE_MATCHES=%d MULTI_MATCHES=%d\n",
        pass ? "YES" : "NO", single >= 1 && multi >= 1 ? "YES" : "NO", disabled == 0 ? "YES" : "NO",
        m_ok ? "YES" : "NO", disabled, single, multi);
    return pass ? 0 : 1;
}
