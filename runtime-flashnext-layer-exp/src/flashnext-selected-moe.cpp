#include "flashnext-selected-moe.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

namespace {
static constexpr int64_t K = 2560;
static constexpr int64_t F = 640;
static constexpr ggml_type GATE_UP_TYPE = (ggml_type) 23;
static constexpr ggml_type DOWN_TYPE = (ggml_type) 20;

bool pread_all(int fd, void * dst, size_t n, uint64_t off, uint64_t & calls) {
    size_t done = 0;
    while (done < n) {
        ++calls;
        const ssize_t got = ::pread(fd, static_cast<uint8_t *>(dst) + done, n - done,
                                    static_cast<off_t>(off + done));
        if (got <= 0) return false;
        done += static_cast<size_t>(got);
    }
    return true;
}

bool tiled_matvec(ggml_backend_t backend, int fd, ggml_type type, int64_t rows,
                  uint64_t offset, const float * input, float * output,
                  void * arena, size_t arena_bytes, uint64_t & bytes,
                  uint64_t & calls) {
    const size_t row_bytes = ggml_row_size(type, K == rows ? F : K);
    const int64_t width = type == DOWN_TYPE ? F : K;
    const size_t legal_row_bytes = ggml_row_size(type, width);
    const int64_t rows_per_tile = static_cast<int64_t>(arena_bytes / legal_row_bytes);
    if (rows_per_tile < 1) return false;

    for (int64_t row = 0; row < rows; row += rows_per_tile) {
        const int64_t nr = std::min<int64_t>(rows_per_tile, rows - row);
        const size_t tile_bytes = static_cast<size_t>(nr) * legal_row_bytes;
        if (!pread_all(fd, arena, tile_bytes, offset + static_cast<uint64_t>(row) * legal_row_bytes, calls)) return false;
        bytes += tile_bytes;

        ggml_init_params ip { 32 * 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;
        ggml_tensor * w = ggml_new_tensor_2d(ctx, type, width, nr);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) { ggml_free(ctx); return false; }
        w->data = arena;
        ggml_backend_tensor_set(x, input, 0, static_cast<size_t>(width) * sizeof(float));
        const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (ok) ggml_backend_tensor_get(y, output + row, 0, static_cast<size_t>(nr) * sizeof(float));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        if (!ok) return false;
    }
    return true;
}
}

bool flashnext_execute_selected_moe(
        const std::array<flashnext_selected_expert_slice, flashnext_layer_transaction::experts_used> & slices,
        flashnext_layer_transaction & tx, size_t arena_bytes) {
    tx.selected_moe.fill(0.0f);
    std::array<double, K> accumulated{};
    tx.io.selected_expert_bytes_read = 0;
    tx.io.unselected_expert_reads = 0;
    tx.io.tensor_window_bytes = arena_bytes;
    tx.io.pread_calls = 0;

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return false;
    void * arena = nullptr;
    if (posix_memalign(&arena, 64, arena_bytes) != 0) { ggml_backend_free(backend); return false; }

    std::array<float, F> gate{}, up{}, act{};
    std::array<float, K> down{};
    bool ok = true;
    const size_t gate_bytes = static_cast<size_t>(F) * ggml_row_size(GATE_UP_TYPE, K);
    const size_t down_bytes = static_cast<size_t>(K) * ggml_row_size(DOWN_TYPE, F);
    for (size_t i = 0; i < tx.experts_used && ok; ++i) {
        const auto & slice = slices[i];
        if (slice.fds[0] < 0 || slice.fds[1] < 0 || slice.fds[2] < 0) { ok = false; break; }
        ok = tiled_matvec(backend, slice.fds[0], GATE_UP_TYPE, F, slice.offsets[0],
                          tx.pre_moe_hidden.data(), gate.data(), arena, arena_bytes,
                          tx.io.selected_expert_bytes_read, tx.io.pread_calls);
        ok = ok && tiled_matvec(backend, slice.fds[1], GATE_UP_TYPE, F, slice.offsets[1],
                                tx.pre_moe_hidden.data(), up.data(), arena, arena_bytes,
                                tx.io.selected_expert_bytes_read, tx.io.pread_calls);
        if (!ok) break;
        for (size_t j = 0; j < F; ++j) act[j] = (gate[j] / (1.0f + std::exp(-gate[j]))) * up[j];
        ok = tiled_matvec(backend, slice.fds[2], DOWN_TYPE, K, slice.offsets[2],
                          act.data(), down.data(), arena, arena_bytes,
                          tx.io.selected_expert_bytes_read, tx.io.pread_calls);
        if (!ok) break;
        if (std::getenv("CORTEX_FLASHNEXT_SELECTED_DEBUG") != nullptr) {
            uint64_t h = 1469598103934665603ull;
            for (float v : down) { uint32_t u; std::memcpy(&u, &v, 4); h ^= u; h *= 1099511628211ull; }
            fprintf(stderr, "CORTEX_SELECTED_DEBUG expert=%d down_hash=%016llx first=%g\n",
                         tx.selected_expert_ids[i], (unsigned long long) h, down[0]);
        }
        for (size_t j = 0; j < K; ++j) accumulated[j] += static_cast<double>(tx.routing_weights[i]) * down[j];
    }
    for (size_t j = 0; j < K; ++j) tx.selected_moe[j] = static_cast<float>(accumulated[j]);
    std::free(arena);
    ggml_backend_free(backend);
    return ok;
}

bool flashnext_execute_selected_moe(const flashnext_tensor_index & index,
        const std::string &, flashnext_layer_transaction & tx, size_t arena_bytes) {
    std::array<flashnext_selected_expert_slice, flashnext_layer_transaction::experts_used> slices{};
    std::array<int, 3> fds{{-1, -1, -1}};
    for (size_t i=0; i<slices.size(); ++i) {
        flashnext_tensor_descriptor g,u,d;
        if (!index.resolve_expert(tx.layer, tx.selected_expert_ids[i], flashnext_projection::gate, g) ||
            !index.resolve_expert(tx.layer, tx.selected_expert_ids[i], flashnext_projection::up, u) ||
            !index.resolve_expert(tx.layer, tx.selected_expert_ids[i], flashnext_projection::down, d)) return false;
        const std::string paths[3] = { g.path, u.path, d.path };
        for (int p = 0; p < 3; ++p) {
            if (fds[p] < 0) fds[p] = ::open(paths[p].c_str(), O_RDONLY|O_CLOEXEC);
            if (fds[p] < 0) { for (int q=0;q<3;++q) if(fds[q]>=0)::close(fds[q]); return false; }
        }
        slices[i].fds=fds; slices[i].offsets={g.offset,u.offset,d.offset};
    }
    const bool ok=flashnext_execute_selected_moe(slices,tx,arena_bytes);
    for (int fd : fds) if (fd >= 0) ::close(fd); return ok;
}

bool flashnext_execute_selected_moe_reference(const flashnext_tensor_index & index,
        const std::string &, flashnext_layer_transaction & tx, size_t arena_bytes) {
    std::array<flashnext_selected_expert_slice, flashnext_layer_transaction::experts_used> slices{};
    std::array<int, 3> fds{{-1, -1, -1}};
    for (size_t i = 0; i < slices.size(); ++i) {
        flashnext_tensor_descriptor g,u,d;
        if (!index.resolve_expert(tx.layer, tx.selected_expert_ids[i], flashnext_projection::gate, g) ||
            !index.resolve_expert(tx.layer, tx.selected_expert_ids[i], flashnext_projection::up, u) ||
            !index.resolve_expert(tx.layer, tx.selected_expert_ids[i], flashnext_projection::down, d)) return false;
        const std::string paths[3] = { g.path, u.path, d.path };
        for (int p = 0; p < 3; ++p) {
            if (fds[p] < 0) fds[p] = ::open(paths[p].c_str(), O_RDONLY | O_CLOEXEC);
            if (fds[p] < 0) { for (int q=0;q<3;++q) if(fds[q]>=0)::close(fds[q]); return false; }
        }
        slices[i].fds = fds; slices[i].offsets = { g.offset, u.offset, d.offset };
    }
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    void * arena = nullptr;
    if (!backend || posix_memalign(&arena, 64, arena_bytes) != 0) { for (int fd : fds) if(fd>=0)::close(fd); if (backend) ggml_backend_free(backend); return false; }
    std::array<float, 640> gate{}, up{}, act{};
    std::array<float, 2560> down{};
    std::array<float, 2560> sum{};
    bool ok = true;
    uint64_t ignored_bytes = 0, ignored_calls = 0;
    for (size_t i = 0; i < slices.size() && ok; ++i) {
        ok = tiled_matvec(backend, slices[i].fds[0], GATE_UP_TYPE, 640, slices[i].offsets[0], tx.pre_moe_hidden.data(), gate.data(), arena, arena_bytes, ignored_bytes, ignored_calls);
        ok = ok && tiled_matvec(backend, slices[i].fds[1], GATE_UP_TYPE, 640, slices[i].offsets[1], tx.pre_moe_hidden.data(), up.data(), arena, arena_bytes, ignored_bytes, ignored_calls);
        if (!ok) break;
        for (size_t j = 0; j < act.size(); ++j) act[j] = gate[j] / (1.0f + std::exp(-gate[j])) * up[j];
        ok = tiled_matvec(backend, slices[i].fds[2], DOWN_TYPE, 2560, slices[i].offsets[2], act.data(), down.data(), arena, arena_bytes, ignored_bytes, ignored_calls);
        if (!ok) break;
        for (size_t j = 0; j < sum.size(); ++j) sum[j] += tx.routing_weights[i] * down[j];
    }
    tx.selected_moe = sum;
    std::free(arena); ggml_backend_free(backend); for (int fd : fds) if (fd >= 0) ::close(fd);
    return ok;
}
