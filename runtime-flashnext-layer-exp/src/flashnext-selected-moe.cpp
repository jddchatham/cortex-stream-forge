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
#include <limits>

namespace {
static constexpr int64_t K = 2560;
static constexpr int64_t F = 640;
static constexpr ggml_type GATE_UP_TYPE = (ggml_type) 23;
static constexpr ggml_type DOWN_TYPE = (ggml_type) 20;
static constexpr size_t EXPERT_ARENA_MIN = 16 * 1024;
static constexpr size_t EXPERT_ARENA_MAX = 1024 * 1024;

bool configure_expert_arena(size_t & arena_bytes) {
    const char * value = std::getenv("CORTEX_FLASHNEXT_EXPERT_ARENA_BYTES");
    if (value == nullptr) return true;

    auto reject = [](const char * reason) {
        std::fprintf(stderr, "CORTEX_FLASHNEXT_EXPERT_ARENA_BYTES invalid: %s\n", reason);
        return false;
    };
    if (*value == '\0') return reject("empty value; expected a decimal byte count from 16384 through 1048576");

    uint64_t parsed = 0;
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
        if (*p < '0' || *p > '9') return reject("expected unsigned decimal digits only; signs and whitespace are not allowed");
        const uint64_t digit = *p - '0';
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return reject("decimal value overflows uint64");
        }
        parsed = parsed * 10 + digit;
    }
    if (parsed < EXPERT_ARENA_MIN || parsed > EXPERT_ARENA_MAX) {
        return reject("value is outside the safe range 16384..1048576 bytes");
    }
    arena_bytes = static_cast<size_t>(parsed);
    return true;
}

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

namespace {

// Mirror of the stock build_moe_ffn weight pipeline (llama-graph.cpp):
// soft_max over all router logits, gather the selected rows, normalize by the
// clamped row sum.  Computed in a ggml graph so every rounding matches the
// stock graph bit-for-bit; n_expert_used is small and fixed.
bool compute_moe_weights(ggml_backend_t backend, const float * logits, const int32_t * ids,
                         size_t n_used, float * weights /* [n_used] */) {
    ggml_init_params ip { 8 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    if (!c) return false;
    ggml_tensor * logits_t = ggml_new_tensor_2d(c, GGML_TYPE_F32, 512, 1);
    ggml_tensor * probs    = ggml_soft_max(c, logits_t);
    ggml_tensor * ids_t    = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t) n_used);
    ggml_tensor * w3 = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, 512, 1), ids_t); // [1, n_used, 1]
    ggml_tensor * w2 = ggml_reshape_2d(c, w3, (int64_t) n_used, 1);
    ggml_tensor * wsum = ggml_clamp(c, ggml_sum_rows(c, w2), 6.103515625e-5f, INFINITY);
    ggml_tensor * wn = ggml_div(c, w2, wsum); // [n_used, 1]
    ggml_cgraph * graph = ggml_new_graph(c);
    ggml_build_forward_expand(graph, wn);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, backend);
    if (!buf) { ggml_free(c); return false; }
    ggml_backend_tensor_set(logits_t, logits, 0, 512 * sizeof(float));
    ggml_backend_tensor_set(ids_t, ids, 0, n_used * sizeof(int32_t));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) ggml_backend_tensor_get(wn, weights, 0, n_used * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    return ok;
}

// Mirror of the stock ggml_swiglu_split gate*silu(up).. silu(gate)*up fused
// kernel, computed per expert in a ggml graph so the exp approximation and
// rounding match the stock graph bit-for-bit.
bool compute_swiglu(ggml_backend_t backend, const float * gate, const float * up,
                    float * act /* [F] */) {
    ggml_init_params ip { 4 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    if (!c) return false;
    ggml_tensor * g_t = ggml_new_tensor_2d(c, GGML_TYPE_F32, F, 1);
    ggml_tensor * u_t = ggml_new_tensor_2d(c, GGML_TYPE_F32, F, 1);
    ggml_tensor * act_t = ggml_swiglu_split(c, g_t, u_t);
    ggml_cgraph * graph = ggml_new_graph(c);
    ggml_build_forward_expand(graph, act_t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, backend);
    if (!buf) { ggml_free(c); return false; }
    ggml_backend_tensor_set(g_t, gate, 0, F * sizeof(float));
    ggml_backend_tensor_set(u_t, up, 0, F * sizeof(float));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) ggml_backend_tensor_get(act_t, act, 0, F * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    return ok;
}

}

bool flashnext_execute_selected_moe(
        const std::array<flashnext_selected_expert_slice, flashnext_layer_transaction::experts_used> & slices,
        flashnext_layer_transaction & tx, size_t arena_bytes) {
    if (!configure_expert_arena(arena_bytes)) return false;
    tx.selected_moe.fill(0.0f);
    tx.io.selected_expert_bytes_read = 0;
    tx.io.unselected_expert_reads = 0;
    tx.io.tensor_window_bytes = arena_bytes;
    tx.io.pread_calls = 0;

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return false;
    void * arena = nullptr;
    if (posix_memalign(&arena, 64, arena_bytes) != 0) { ggml_backend_free(backend); return false; }

    // stock build_moe_ffn order: normalized softmax weights for the selected
    // experts, in the graph's top-k order (tx.selected_expert_ids).  Callers
    // that never populated router_logits (legacy standalone drivers) keep the
    // previous unnormalized routing_weights behavior.
    std::array<float, flashnext_layer_transaction::experts_used> weights{};
    bool logits_populated = false;
    for (float v : tx.router_logits) { logits_populated = logits_populated || v != 0.0f; }
    if (!logits_populated) {
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(tx.routing_weights[i]);
    } else if (!compute_moe_weights(backend, tx.router_logits.data(), tx.selected_expert_ids.data(),
                             tx.selected_expert_ids.size(), weights.data())) {
        std::free(arena);
        ggml_backend_free(backend);
        return false;
    }

    std::array<float, F> gate{}, up{}, act{};
    std::array<float, K> down{};
    bool ok = true;
    bool first = true;
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
        // stock ggml_swiglu_split: silu(gate) * up through the same ggml kernel
        ok = compute_swiglu(backend, gate.data(), up.data(), act.data());
        if (!ok) break;
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
        // stock aggregation: ggml_mul by the per-expert weight, then sequential
        // ggml_add over experts in top-k order, all in float32
        const float w_i = weights[i];
        if (first) {
            for (size_t j = 0; j < K; ++j) { tx.selected_moe[j] = down[j] * w_i; }
            first = false;
        } else {
            for (size_t j = 0; j < K; ++j) { tx.selected_moe[j] = tx.selected_moe[j] + down[j] * w_i; }
        }
    }
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
