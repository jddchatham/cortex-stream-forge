#include "ggml.h"
#include "ggml-backend.h"
#include "flashnext-selected-moe.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Small executable contract test for the exact Qwen4Exp post-MoE equations:
// shared SwiGLU, sigmoided shared gate, selected-MoE addition, then HC combine.
// It intentionally uses synthetic tensors only; no GGUF or model payloads.
static bool run_case(int64_t nt) {
    constexpr int64_t H = 8, F = 4, HC = 4;
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) return false;
    ggml_init_params p{ 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    if (!ctx) { ggml_backend_free(be); return false; }

    ggml_tensor * pre = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nt);
    ggml_tensor * selected = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nt);
    ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, HC, nt);
    ggml_tensor * inject = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HC, nt);
    ggml_tensor * up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, F);
    ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, F);
    ggml_tensor * down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, H);
    ggml_tensor * gate_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
    ggml_tensor * ext_selected = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nt);
    ggml_tensor * ref = nullptr;
    ggml_tensor * part = nullptr;
    if (!pre || !selected || !residual || !inject || !up || !gate || !down || !gate_inp || !ext_selected) return false;
    auto fill = [](ggml_tensor * t, float scale) {
        float * d = static_cast<float *>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = scale * float((i % 11) - 5);
    };

    // Shared expert exactly as build_layer_ffn(): SwiGLU -> down -> sigmoid gate.
    auto shared_path = [&](ggml_tensor * selected_input) {
        ggml_tensor * g = ggml_mul_mat(ctx, gate, pre);
        ggml_tensor * u = ggml_mul_mat(ctx, up, pre);
        ggml_tensor * swiglu = ggml_silu(ctx, g);
        swiglu = ggml_mul(ctx, swiglu, u);
        ggml_tensor * sh = ggml_mul_mat(ctx, down, swiglu);
        ggml_tensor * sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, gate_inp, pre));
        sh = ggml_mul(ctx, sh, sg);
        return ggml_add(ctx, selected_input, sh);
    };
    ggml_tensor * ref_ffn = shared_path(selected);
    ggml_tensor * part_ffn = shared_path(ext_selected);
    auto hc = [&](ggml_tensor * ffn) {
        ggml_tensor * w = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f/float(HC))), 2.0f);
        w = ggml_reshape_3d(ctx, w, 1, HC, nt);
        ggml_tensor * b = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, ffn, H, 1, nt), H, HC, nt, 1);
        return ggml_add(ctx, residual, ggml_mul(ctx, b, w));
    };
    ref = hc(ref_ffn); part = hc(part_ffn);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, part);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); ggml_backend_free(be); return false; }
    fill(pre, .07f); fill(selected, .11f); fill(residual, .03f); fill(inject, .05f);
    fill(up, .09f); fill(gate, .08f); fill(down, .06f); fill(gate_inp, .04f);
    std::copy(static_cast<float *>(selected->data), static_cast<float *>(selected->data) + ggml_nelements(selected),
              static_cast<float *>(ext_selected->data));
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be); return false;
    }
    std::vector<float> a(ggml_nelements(ref)), b(ggml_nelements(part));
    ggml_backend_tensor_get(ref, a.data(), 0, a.size()*sizeof(float));
    ggml_backend_tensor_get(part, b.data(), 0, b.size()*sizeof(float));
    float max_abs = 0;
    for (size_t i = 0; i < a.size(); ++i) max_abs = std::max(max_abs, std::fabs(a[i]-b[i]));
    std::printf("POST_MOE_CASE tokens=%lld elements=%zu max_abs=%g\n", (long long) nt, a.size(), max_abs);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    return max_abs == 0.0f;
}

int main() {
    const float logits[] = { 1, 2, 3, 4, 4, 3, 2, 1 };
    std::vector<int32_t> ids;
    std::vector<double> weights;
    const bool routed = flashnext_route_softmax_topk(logits, 4, 2, 2, 1.0f, ids, weights);
    const bool route_ok = routed && ids.size() == 4 && weights.size() == 4 &&
        ids[0] == 3 && ids[1] == 2 && ids[2] == 0 && ids[3] == 1 &&
        std::fabs((weights[0] + weights[1]) - 1.0) < 1e-12 &&
        std::fabs((weights[2] + weights[3]) - 1.0) < 1e-12;
    std::printf("POST_MOE_ROUTING %s\n", route_ok ? "PASS" : "FAIL");
    return route_ok && run_case(1) && run_case(3) ? 0 : 1;
}
