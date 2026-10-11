#include "flashnext-shared-expert.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {
bool rd(int fd, void * p, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) { ssize_t r = ::pread(fd, static_cast<uint8_t *>(p) + done, n - done, off + done); if (r <= 0) return false; done += r; }
    return true;
}
}

bool flashnext_execute_shared_expert(const flashnext_shared_expert_slices & s,
                                     flashnext_layer_transaction & tx) {
    for (int fd : s.fds) if (fd < 0) return false;
    std::array<uint8_t, 870400> gate{}, up{};
    std::array<uint8_t, 921600> down{};
    std::array<float, 2560> sigmoid{};
    if (!rd(s.fds[0], gate.data(), gate.size(), s.gate_offset) ||
        !rd(s.fds[1], up.data(), up.size(), s.up_offset) ||
        !rd(s.fds[2], down.data(), down.size(), s.down_offset) ||
        !rd(s.fds[3], sigmoid.data(), sigmoid.size() * sizeof(float), s.sigmoid_offset)) return false;
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) return false;
    ggml_init_params ip { 64 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip); if (!c) { ggml_backend_free(be); return false; }
    ggml_tensor * wg = ggml_new_tensor_2d(c, (ggml_type) 23, 2560, 640);
    ggml_tensor * wu = ggml_new_tensor_2d(c, (ggml_type) 23, 2560, 640);
    ggml_tensor * wd = ggml_new_tensor_2d(c, (ggml_type) 20, 640, 2560);
    ggml_tensor * ws = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2560, 1);
    ggml_tensor * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2560, 1);
    ggml_tensor * g = ggml_mul_mat(c, wg, in);
    ggml_tensor * u = ggml_mul_mat(c, wu, in);
    ggml_tensor * a = ggml_mul(c, ggml_silu(c, g), u);
    ggml_tensor * y = ggml_mul_mat(c, wd, a);
    ggml_tensor * sg = ggml_sigmoid(c, ggml_mul_mat(c, ws, in));
    ggml_tensor * out = ggml_mul(c, y, sg);
    ggml_cgraph * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!buf) { ggml_free(c); ggml_backend_free(be); return false; }
    wg->data = gate.data(); wu->data = up.data(); wd->data = down.data(); ws->data = sigmoid.data();
    ggml_backend_tensor_set(in, tx.pre_moe_hidden.data(), 0, 2560 * sizeof(float));
    bool ok = ggml_backend_graph_compute(be, graph) == GGML_STATUS_SUCCESS;
    if (ok) ggml_backend_tensor_get(out, tx.shared_moe.data(), 0, 2560 * sizeof(float));
    ggml_backend_buffer_free(buf); ggml_free(c); ggml_backend_free(be);
    return ok;
}

