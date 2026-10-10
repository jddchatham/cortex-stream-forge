#include "flashnext-segment-b.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>

bool flashnext_execute_segment_b(flashnext_layer_transaction & tx) {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return false;
    ggml_init_params ip { 16 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { ggml_backend_free(backend); return false; }

    ggml_tensor * moe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2560, 1);
    ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 4, 1);
    ggml_tensor * inject = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_tensor * out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2560, 4, 1);
    ggml_tensor * w = ggml_sigmoid(ctx, ggml_scale(ctx, inject, 0.25f));
    w = ggml_scale(ctx, w, 2.0f);
    w = ggml_reshape_3d(ctx, w, 1, 4, 1);
    ggml_tensor * b = ggml_reshape_3d(ctx, moe, 2560, 1, 1);
    b = ggml_repeat_4d(ctx, b, 2560, 4, 1, 1);
    ggml_tensor * result = ggml_add(ctx, residual, ggml_mul(ctx, b, w));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, result, out));
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); ggml_backend_free(backend); return false; }
    ggml_backend_tensor_set(moe, tx.complete_moe.data(), 0, 2560 * sizeof(float));
    ggml_backend_tensor_set(residual, tx.residual_hc.data(), 0, 2560 * 4 * sizeof(float));
    ggml_backend_tensor_set(inject, tx.inject.data(), 0, 4 * sizeof(float));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) ggml_backend_tensor_get(out, tx.layer_output.data(), 0, 2560 * 4 * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

