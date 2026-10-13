#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto n = f.tellg(); if (n < 0) return false;
    out.resize((size_t) n); f.seekg(0); f.read((char *) out.data(), n);
    return (bool) f;
}

static void compare(const char * tag, const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    const size_t n = std::min(a.size(), b.size()) / sizeof(float);
    const float * x = (const float *) a.data(); const float * y = (const float *) b.data();
    double ss = 0; float mx = 0; size_t bad = 0, nonfinite = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) ++nonfinite;
        const float d = std::abs(x[i] - y[i]); mx = std::max(mx, d); ss += (double) d * d;
        if (d > 1e-5f) ++bad;
    }
    std::printf("GDN_REPLAY compare=%s elements=%zu max_abs=%g rms=%g mismatches=%zu nonfinite=%zu\n",
        tag, n, mx, std::sqrt(ss / std::max<size_t>(1, n)), bad, nonfinite);
}

int main(int argc, char ** argv) {
    const std::string p = argc > 1 ? argv[1] : "/tmp/cortex-iter100-gdn";
    std::vector<uint8_t> q, k, v, g, beta, s, expected, expected_state;
    if (!read_file(p + "-q.bin", q) || !read_file(p + "-k.bin", k) || !read_file(p + "-v.bin", v) ||
        !read_file(p + "-g.bin", g) || !read_file(p + "-b.bin", beta) || !read_file(p + "-s_before.bin", s) ||
        !read_file(p + "-result.bin", expected) || !read_file(p + "-state_after.bin", expected_state)) {
        std::fprintf(stderr, "GDN_REPLAY fixture=FAIL prefix=%s\n", p.c_str()); return 2;
    }
    // Dimensions are taken from the exported manifest for the real layer-0 invocation.
    constexpr int64_t DQ = 128, HQ = 16, DV = 128, HV = 48;
    const int64_t K = (int64_t) q.size() / (DQ * HQ * sizeof(float));
    const size_t expected_state_bytes = (size_t) DV * DV * HV * sizeof(float);
    const size_t expected_v_bytes = (size_t) 40960 * (K - 1) + 24576;
    if (K <= 0 || q.size() != (size_t) DQ*HQ*K*4 || k.size() != q.size() ||
        v.size() != expected_v_bytes ||
        g.size() != (size_t) HV*K*4 || beta.size() != g.size() || s.size() != expected_state_bytes ||
        expected_state.size() != expected_state_bytes) {
        std::fprintf(stderr, "GDN_REPLAY fixture=BAD_SIZES q=%zu k=%zu v=%zu g=%zu b=%zu s=%zu result=%zu state=%zu\n",
            q.size(), k.size(), v.size(), g.size(), beta.size(), s.size(), expected.size(), expected_state.size()); return 3;
    }
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_init_params ip{ .mem_size = 64u*1024u*1024u, .mem_buffer = nullptr, .no_alloc = true };
    ggml_context * ctx = ggml_init(ip);
    if (!backend || !ctx) return 4;
    ggml_tensor * tq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DQ, HQ, K);
    ggml_tensor * tk = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DQ, HQ, K);
    ggml_tensor * tv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DV, HV, K);
    ggml_tensor * tg = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, HV, K);
    ggml_tensor * tb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, HV, K);
    ggml_tensor * ts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DV, DV, HV);
    // Preserve the graph-local strides exported by the production tensors.
    tq->nb[1] = 512; tq->nb[2] = 8192;
    tk->nb[1] = 512; tk->nb[2] = 8192;
    tv->nb[1] = 512; tv->nb[2] = 40960;
    tg->nb[1] = 4;   tg->nb[2] = 192;
    tb->nb[1] = 4;   tb->nb[2] = 192;
    ts->nb[1] = 512; ts->nb[2] = 65536;
    ggml_tensor * out = ggml_gated_delta_net(ctx, tq, tk, tv, tg, tb, ts, K);
    ggml_set_name(out, "gdn_replay_output");
    ggml_cgraph * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 5;
    ggml_backend_tensor_set(tq, q.data(), 0, q.size()); ggml_backend_tensor_set(tk, k.data(), 0, k.size());
    ggml_backend_tensor_set(tv, v.data(), 0, v.size()); ggml_backend_tensor_set(tg, g.data(), 0, g.size());
    ggml_backend_tensor_set(tb, beta.data(), 0, beta.size()); ggml_backend_tensor_set(ts, s.data(), 0, s.size());
    const ggml_status status = ggml_backend_graph_compute(backend, graph); ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "GDN_REPLAY compute=FAIL status=%d\n", (int) status); return 6; }
    std::vector<uint8_t> actual(ggml_nbytes(out)); ggml_backend_tensor_get(out, actual.data(), 0, actual.size());
    // The final DV*DV*HV region is the updated-state snapshot in the K=9 fused output.
    std::printf("GDN_REPLAY output_shape=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] bytes=%zu\n",
        (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3],
        out->nb[0], out->nb[1], out->nb[2], out->nb[3], actual.size());
    if (actual.size() < expected.size()) { std::fprintf(stderr, "GDN_REPLAY output_bytes=%zu expected=%zu\n", actual.size(), expected.size()); return 7; }
    compare("fused_output", actual, expected);
    const size_t state_offset = (size_t) DV * HV * K * sizeof(float);
    if (state_offset + expected_state.size() > actual.size()) return 8;
    std::vector<uint8_t> actual_state(actual.begin() + state_offset, actual.begin() + state_offset + expected_state.size());
    compare("updated_state", actual_state, expected_state);
    std::printf("GDN_REPLAY PASS q=%zu k=%zu v=%zu gate=%zu beta=%zu state=%zu output=%zu K=%lld\n",
        q.size(), k.size(), v.size(), g.size(), beta.size(), s.size(), actual.size(), (long long) K);
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend); return 0;
}
