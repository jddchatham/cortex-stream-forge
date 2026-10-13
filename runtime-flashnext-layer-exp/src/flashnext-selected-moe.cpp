#include "flashnext-selected-moe.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <vector>
#include <atomic>
#include <limits>
#include <numeric>

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

bool tiled_matvec(ggml_backend_t backend, int projection, int fd, ggml_type type, int64_t rows,
                  uint64_t offset, const float * input, float * output,
                  void * arena, size_t arena_bytes, uint64_t & bytes,
                  uint64_t & calls, int layer = -1, int token = -1, int expert = -1, int row_base = 0) {
    const size_t row_bytes = ggml_row_size(type, K == rows ? F : K);
    const int64_t width = type == DOWN_TYPE ? F : K;
    const size_t legal_row_bytes = ggml_row_size(type, width);
    const int64_t rows_per_tile = static_cast<int64_t>(arena_bytes / legal_row_bytes);
    if (rows_per_tile < 1) return false;
    const bool trace = std::getenv("CORTEX_FLASHNEXT_CPU_DISPATCH_TRACE") != nullptr;
    if (trace) std::fprintf(stderr, "CORTEX_DISPATCH provider_compute type=%d rows=%lld width=%lld op=ggml_mul_mat\n",
                            (int) type, (long long) rows, (long long) width);

    static std::atomic<uint64_t> invocation_id{0};
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
        char x_name[128];
        char y_name[128];
        std::snprintf(x_name, sizeof(x_name), "cortex_moe_l%d_p%d_t%d_e%d_r%d_x",
                      layer, projection, token, expert, row_base + (int) row);
        std::snprintf(y_name, sizeof(y_name), "cortex_moe_l%d_p%d_t%d_e%d_r%d_y",
                      layer, projection, token, expert, row_base + (int) row);
        ggml_set_name(x, x_name);
        ggml_set_name(y, y_name);
        if (trace) std::fprintf(stderr, "CORTEX_DISPATCH graph_construct op=%d name=%s ne=[%lld,%lld,%lld,%lld]\n",
                                (int) y->op, ggml_get_name(y), (long long) y->ne[0], (long long) y->ne[1],
                                (long long) y->ne[2], (long long) y->ne[3]);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) { ggml_free(ctx); return false; }
        w->data = arena;
        const bool transfer_trace = std::getenv("CORTEX_FLASHNEXT_TRANSFER_TRACE") != nullptr;
        auto report_transfer = [&](const char * phase, const float * values, size_t count) {
            if (!transfer_trace) return;
            uint32_t nonzero = 0, finite = 0;
            for (size_t i = 0; i < count; ++i) {
                if (std::isfinite(values[i])) ++finite;
                if (values[i] != 0.0f) ++nonzero;
            }
            std::fprintf(stderr,
                         "CORTEX_PROVIDER_TRANSFER id=l%d-p%d-t%d-e%d-r%d phase=%s tensor=%s data=%p buffer=%p ne0=%lld ne1=%lld nb0=%zu nb1=%zu bytes=%zu nonzero=%u finite=%u first=%g last=%g\n",
                         layer, projection, token, expert, row_base + (int) row, phase,
                         ggml_get_name(x), x->data, (void *) x->buffer,
                         (long long) x->ne[0], (long long) x->ne[1], x->nb[0], x->nb[1],
                         count * sizeof(float), nonzero, finite, count ? values[0] : 0.0f,
                         count ? values[count - 1] : 0.0f);
        };
        report_transfer("staging", input, (size_t) width);
        ggml_backend_tensor_set(x, input, 0, static_cast<size_t>(width) * sizeof(float));
        if (transfer_trace) {
            std::vector<float> readback((size_t) width, 0.0f);
            ggml_backend_tensor_get(x, readback.data(), 0, readback.size() * sizeof(float));
            report_transfer("after_tensor_set", readback.data(), readback.size());
        }
        ggml_cpu_diagnostic_target diag{};
        diag.requested_row = -1;
        std::vector<uint8_t> diag_weight(136), diag_activation(type == GATE_UP_TYPE ? ggml_row_size(GGML_TYPE_Q8_K, 256) : ggml_row_size(GGML_TYPE_Q8_0, 32));
        std::vector<float> diag_f32(type == GATE_UP_TYPE ? 256 : 32);
        const size_t row_weight_bytes = ggml_row_size(type, width);
        const ggml_type row_activation_type = type == GATE_UP_TYPE ? GGML_TYPE_Q8_K : GGML_TYPE_Q8_0;
        const size_t row_activation_bytes = ggml_row_size(row_activation_type, width);
        std::vector<uint8_t> diag_weight_row(row_weight_bytes), diag_activation_row(row_activation_bytes);
        std::vector<float> diag_activation_row_f32(static_cast<size_t>(width));
        float diag_dot = 0.0f;
        std::vector<int> requested_rows;
        if (std::getenv("CORTEX_FLASHNEXT_REQUESTED_ROW_CAPTURE") &&
            layer == 4 && projection == 2 && expert == 252) {
            if (token == 4) requested_rows = {1026, 2208, 1298, 2146, 2060};
            if (token == 7) requested_rows = {675, 537};
        }
        if (std::getenv("CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE") != nullptr) {
            diag.enabled = true;
            diag.target = nullptr;
            diag.provider_local_match = true;
            diag.weight_block = diag_weight.data();
            diag.weight_block_size = diag_weight.size();
            diag.activation_block = diag_activation.data();
            diag.activation_block_size = diag_activation.size();
            diag.activation_f32 = diag_f32.data();
            diag.activation_f32_count = diag_f32.size();
            diag.dot_result = &diag_dot;
            diag.weight_row = diag_weight_row.data();
            diag.weight_row_size = diag_weight_row.size();
            diag.activation_row = diag_activation_row.data();
            diag.activation_row_size = diag_activation_row.size();
            diag.activation_row_f32 = diag_activation_row_f32.data();
            diag.activation_row_f32_count = diag_activation_row_f32.size();
            ggml_backend_cpu_set_diagnostic_target(backend, &diag);
        }
        if (transfer_trace) {
            std::vector<float> readback((size_t) width, 0.0f);
            ggml_backend_tensor_get(x, readback.data(), 0, readback.size() * sizeof(float));
            report_transfer("before_graph", readback.data(), readback.size());
        }
        bool ok = true;
        if (!requested_rows.empty()) {
            for (int requested : requested_rows) {
                diag.requested_row = requested;
                diag.row_capture_once = 0;
                diag.dot_capture_once = 0;
                ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS && ok;
                if (diag.row_capture_once) {
                    std::fprintf(stderr, "CORTEX_REQUESTED_ROW_CAPTURE layer=%d token=%d expert=%d row=%d dot=%.17g dest_offset=%llu stored=%.17g status=PASS\n",
                                 layer, token, expert, requested, double(diag_dot),
                                 (unsigned long long) diag.captured_dest_offset, double(diag.captured_output));
                } else {
                    std::fprintf(stderr, "CORTEX_REQUESTED_ROW_CAPTURE layer=%d token=%d expert=%d row=%d status=MISSING\n",
                                 layer, token, expert, requested);
                }
            }
        } else {
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        if (trace) std::fprintf(stderr, "CORTEX_DISPATCH graph_complete type=%d status=%s\n", (int) type, ok ? "PASS" : "FAIL");
        if (diag.enabled) {
            static std::atomic<bool> accepted[3]{};
            bool expected_accept = false;
            const bool accept_record = projection >= 0 && projection < 3 && diag.row_capture_once &&
                                       accepted[projection].compare_exchange_strong(expected_accept, true);
            struct record_header {
                uint32_t magic, version, projection, weight_type;
                uint64_t width, weight_bytes, activation_bytes, f32_count;
                float dot;
                uint64_t invocation;
                int32_t layer, token, expert, row, tile;
            } header{0x43585231u, 2u, (uint32_t) projection, (uint32_t) type,
                     (uint64_t) width, (uint64_t) row_weight_bytes, (uint64_t) row_activation_bytes,
                     (uint64_t) diag_activation_row_f32.size(), diag_dot,
                     invocation_id.fetch_add(1, std::memory_order_relaxed), layer, token, expert,
                     (int32_t) (row_base + row), (int32_t) (row / rows_per_tile)};
            if (std::getenv("CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE") != nullptr &&
                    layer == 4 && projection == 2 && token == 1 && expert == 257 &&
                    row_base + row == 0 && diag.row_capture_once) {
                const char * path = "/tmp/cortex-iter170-l4-p2-t1-e257-r0.bin";
                if (access(path, F_OK) != 0) {
                    FILE * f = std::fopen(path, "wb");
                    if (f) {
                        std::fwrite(&header, sizeof(header), 1, f);
                        std::fwrite(diag_weight_row.data(), 1, diag_weight_row.size(), f);
                        std::fwrite(diag_activation_row.data(), 1, diag_activation_row.size(), f);
                        std::fwrite(diag_activation_row_f32.data(), sizeof(float), diag_activation_row_f32.size(), f);
                        std::fclose(f);
                        std::fprintf(stderr, "CORTEX_EXACT_DOT_CAPTURE path=%s projection=%d layer=%d token=%d expert=%d row=%d weight_bytes=%zu activation_bytes=%zu dot=%g status=PASS\n",
                                     path, projection, layer, token, expert, row_base + row,
                                     diag_weight_row.size(), diag_activation_row.size(), diag_dot);
                    }
                }
            }
            if (accept_record) {
                char path[128];
                std::snprintf(path, sizeof(path), "/tmp/cortex-iter140-projection-%d.bin", projection);
                FILE * f = std::fopen(path, "wb");
                if (f) {
                    std::fwrite(&header, sizeof(header), 1, f);
                    std::fwrite(diag_weight_row.data(), 1, diag_weight_row.size(), f);
                    std::fwrite(diag_activation_row.data(), 1, diag_activation_row.size(), f);
                    std::fwrite(diag_activation_row_f32.data(), sizeof(float), diag_activation_row_f32.size(), f);
                    std::fclose(f);
                    std::fprintf(stderr, "CORTEX_ATOMIC_ROW_CAPTURE projection=%d width=%lld weight_bytes=%zu activation_bytes=%zu dot=%g status=PASS\n",
                                 projection, (long long) width, row_weight_bytes, row_activation_bytes, diag_dot);
                }
            }
            if (diag.capture_once) {
                FILE * f = std::fopen(type == GATE_UP_TYPE ? "/tmp/cortex-iter132-iq4xs-weight.bin" : "/tmp/cortex-iter132-iq4nl-weight.bin", "wb");
                if (f) { std::fwrite(diag_weight.data(), 1, diag_weight.size(), f); std::fclose(f); }
                f = std::fopen(type == GATE_UP_TYPE ? "/tmp/cortex-iter132-q8k.bin" : "/tmp/cortex-iter132-q80.bin", "wb");
                if (f) { std::fwrite(diag_activation.data(), 1, diag_activation.size(), f); std::fclose(f); }
                f = std::fopen(type == GATE_UP_TYPE ? "/tmp/cortex-iter132-f32-xs.bin" : "/tmp/cortex-iter132-f32-nl.bin", "wb");
                if (f) { std::fwrite(diag_f32.data(), sizeof(float), diag_f32.size(), f); std::fclose(f); }
            }
            if (diag.row_capture_once) {
                FILE * f = std::fopen(type == GATE_UP_TYPE ? "/tmp/cortex-iter139-iq4xs-weight-row.bin" : "/tmp/cortex-iter139-iq4nl-weight-row.bin", "wb");
                if (f) { std::fwrite(diag_weight_row.data(), 1, diag_weight_row.size(), f); std::fclose(f); }
                f = std::fopen(type == GATE_UP_TYPE ? "/tmp/cortex-iter139-q8k-row.bin" : "/tmp/cortex-iter139-q80-row.bin", "wb");
                if (f) { std::fwrite(diag_activation_row.data(), 1, diag_activation_row.size(), f); std::fclose(f); }
                f = std::fopen(type == GATE_UP_TYPE ? "/tmp/cortex-iter139-f32-xs-row.bin" : "/tmp/cortex-iter139-f32-nl-row.bin", "wb");
                if (f) { std::fwrite(diag_activation_row_f32.data(), sizeof(float), diag_activation_row_f32.size(), f); std::fclose(f); }
            }
            if (std::getenv("CORTEX_FLASHNEXT_TOKEN_Q8_CAPTURE") != nullptr &&
                layer == 4 && token >= 0 && token < 2 && expert >= 0 && diag.row_capture_once) {
                char path[160];
                std::snprintf(path, sizeof(path), "/tmp/cortex-iter162-p%d-l%d-t%d-e%d-q.bin",
                              projection, layer, token, expert);
                FILE * f = access(path, F_OK) == 0 ? nullptr : std::fopen(path, "wb");
                if (f) { std::fwrite(diag_activation_row.data(), 1, diag_activation_row.size(), f); std::fclose(f); }
                std::snprintf(path, sizeof(path), "/tmp/cortex-iter162-p%d-l%d-t%d-e%d-f32.bin",
                              projection, layer, token, expert);
                f = access(path, F_OK) == 0 ? nullptr : std::fopen(path, "wb");
                if (f) { std::fwrite(diag_activation_row_f32.data(), sizeof(float), diag_activation_row_f32.size(), f); std::fclose(f); }
                std::fprintf(stderr, "CORTEX_TOKEN_Q8_CAPTURE projection=%d layer=%d token=%d expert=%d qbytes=%zu f32=%zu status=PASS\n",
                             projection, layer, token, expert, diag_activation_row.size(), diag_activation_row_f32.size());
            }
            std::fprintf(stderr, "CORTEX_CPU_OPERAND_CAPTURE type=%d count=%u row_count=%u weight_bytes=%zu activation_bytes=%zu row_weight_bytes=%zu row_activation_bytes=%zu dot=%g plan=%u mul_mat=%u tiled=%u one_chunk=%u vec_dot=%u last_type=%d src_data=%u wt_data=%u src_ne=[%lld,%lld] src_nonzero=%u\n",
                         (int) type, (unsigned) diag.capture_once, (unsigned) diag.row_capture_once,
                         diag_weight.size(), diag_activation.size(), row_weight_bytes, row_activation_bytes, diag_dot,
                         (unsigned) diag.cpu_plan_count, (unsigned) diag.mul_mat_count,
                         (unsigned) diag.tiled_count, (unsigned) diag.one_chunk_count,
                         (unsigned) diag.vec_dot_count, (int) diag.last_weight_type,
                         (unsigned) diag.source_data_present, (unsigned) diag.weight_data_present,
                         (long long) diag.source_ne0, (long long) diag.source_ne1,
                         (unsigned) diag.source_nonzero_count);
        }
        if (ok) ggml_backend_tensor_get(y, output + row, 0, static_cast<size_t>(nr) * sizeof(float));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        if (!ok) return false;
    }
    return true;
}
}

bool flashnext_route_softmax_topk(
        const float * logits, size_t n_expert, size_t n_tokens, size_t k,
        float weight_scale, std::vector<int32_t> & selected_ids,
        std::vector<double> & normalized_weights) {
    if (!logits || n_expert == 0 || n_tokens == 0 || k == 0 || k > n_expert ||
            n_tokens > SIZE_MAX / k) return false;
    selected_ids.resize(n_tokens * k);
    normalized_weights.resize(n_tokens * k);
    for (size_t t = 0; t < n_tokens; ++t) {
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t e = 0; e < n_expert; ++e) mx = std::max(mx, logits[e + t*n_expert]);
        std::vector<double> probs(n_expert);
        double denom = 0.0;
        for (size_t e = 0; e < n_expert; ++e) {
            probs[e] = std::exp(double(logits[e + t*n_expert] - mx));
            denom += probs[e];
        }
        if (!(denom > 0.0) || !std::isfinite(denom)) return false;
        for (double & p : probs) p /= denom;
        std::vector<size_t> order(n_expert);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return probs[a] > probs[b];
        });
        double top_sum = 0.0;
        for (size_t j = 0; j < k; ++j) top_sum += probs[order[j]];
        if (!(top_sum > 0.0) || !std::isfinite(top_sum)) return false;
        for (size_t j = 0; j < k; ++j) {
            selected_ids[t*k + j] = static_cast<int32_t>(order[j]);
            normalized_weights[t*k + j] = double(weight_scale) * probs[order[j]] / top_sum;
        }
    }
    return true;
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
    uint64_t bytes_read = 0, pread_calls = 0;

    std::array<float, F> gate{}, up{}, act{};
    std::array<float, K> down{};
    bool ok = true;
    const size_t gate_bytes = static_cast<size_t>(F) * ggml_row_size(GATE_UP_TYPE, K);
    const size_t down_bytes = static_cast<size_t>(K) * ggml_row_size(DOWN_TYPE, F);
    for (size_t i = 0; i < tx.experts_used && ok; ++i) {
        const auto & slice = slices[i];
        if (slice.fds[0] < 0 || slice.fds[1] < 0 || slice.fds[2] < 0) { ok = false; break; }
        ok = tiled_matvec(backend, 0, slice.fds[0], GATE_UP_TYPE, F, slice.offsets[0],
                          tx.pre_moe_hidden.data(), gate.data(), arena, arena_bytes,
                          tx.io.selected_expert_bytes_read, tx.io.pread_calls, tx.layer, 0,
                          tx.selected_expert_ids[i], 0);
        ok = ok && tiled_matvec(backend, 1, slice.fds[1], GATE_UP_TYPE, F, slice.offsets[1],
                                tx.pre_moe_hidden.data(), up.data(), arena, arena_bytes,
                                tx.io.selected_expert_bytes_read, tx.io.pread_calls, tx.layer, 0,
                                tx.selected_expert_ids[i], 0);
        if (!ok) break;
        for (size_t j = 0; j < F; ++j) act[j] = (gate[j] / (1.0f + std::exp(-gate[j]))) * up[j];
        ok = tiled_matvec(backend, 2, slice.fds[2], DOWN_TYPE, K, slice.offsets[2],
                          act.data(), down.data(), arena, arena_bytes,
                          tx.io.selected_expert_bytes_read, tx.io.pread_calls, tx.layer, 0,
                          tx.selected_expert_ids[i], 0);
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
        ok = tiled_matvec(backend, 0, slices[i].fds[0], GATE_UP_TYPE, 640, slices[i].offsets[0], tx.pre_moe_hidden.data(), gate.data(), arena, arena_bytes, ignored_bytes, ignored_calls);
        ok = ok && tiled_matvec(backend, 1, slices[i].fds[1], GATE_UP_TYPE, 640, slices[i].offsets[1], tx.pre_moe_hidden.data(), up.data(), arena, arena_bytes, ignored_bytes, ignored_calls);
        if (!ok) break;
        for (size_t j = 0; j < act.size(); ++j) act[j] = gate[j] / (1.0f + std::exp(-gate[j])) * up[j];
        ok = tiled_matvec(backend, 2, slices[i].fds[2], DOWN_TYPE, 2560, slices[i].offsets[2], act.data(), down.data(), arena, arena_bytes, ignored_bytes, ignored_calls);
        if (!ok) break;
        for (size_t j = 0; j < sum.size(); ++j) sum[j] += tx.routing_weights[i] * down[j];
    }
    tx.selected_moe = sum;
    std::free(arena); ggml_backend_free(backend); for (int fd : fds) if (fd >= 0) ::close(fd);
    return ok;
}

bool flashnext_execute_selected_moe_rows(
        const flashnext_tensor_index & index,
        const std::string & index_path,
        const float * pre_moe,
        const int32_t * selected_expert_ids,
        const double * routing_weights,
        size_t n_tokens,
        float * selected_moe,
        size_t arena_bytes) {
    return flashnext_execute_selected_moe_rows_layer(index, 0, pre_moe, selected_expert_ids,
        routing_weights, n_tokens, selected_moe, arena_bytes);
}

bool flashnext_execute_selected_moe_rows_layer(
        const flashnext_tensor_index & index, int layer,
        const float * pre_moe, const int32_t * selected_expert_ids,
        const double * routing_weights, size_t n_tokens, float * selected_moe,
        size_t arena_bytes) {
    if (n_tokens == 0 || pre_moe == nullptr || selected_expert_ids == nullptr ||
            routing_weights == nullptr || selected_moe == nullptr) {
        return false;
    }
    if (n_tokens > SIZE_MAX / flashnext_layer_transaction::hidden ||
            n_tokens > SIZE_MAX / flashnext_layer_transaction::experts_used) {
        return false;
    }
    for (size_t it = 0; it < n_tokens; ++it) {
        flashnext_layer_transaction tx{};
        std::copy_n(pre_moe + it * flashnext_layer_transaction::hidden,
                flashnext_layer_transaction::hidden, tx.pre_moe_hidden.begin());
        std::copy_n(selected_expert_ids + it * flashnext_layer_transaction::experts_used,
                flashnext_layer_transaction::experts_used, tx.selected_expert_ids.begin());
        std::copy_n(routing_weights + it * flashnext_layer_transaction::experts_used,
                flashnext_layer_transaction::experts_used, tx.routing_weights.begin());
        tx.layer = layer;
        std::array<flashnext_selected_expert_slice, flashnext_layer_transaction::experts_used> slices{};
        std::array<int, 3> fds{{-1, -1, -1}};
        bool ok = true;
        for (size_t i = 0; i < slices.size() && ok; ++i) {
            flashnext_tensor_descriptor g, u, d;
            ok = index.resolve_expert(layer, tx.selected_expert_ids[i], flashnext_projection::gate, g) &&
                 index.resolve_expert(layer, tx.selected_expert_ids[i], flashnext_projection::up, u) &&
                 index.resolve_expert(layer, tx.selected_expert_ids[i], flashnext_projection::down, d);
            const std::string paths[3] = { g.path, u.path, d.path };
            for (int p = 0; p < 3 && ok; ++p) {
                if (fds[p] < 0) fds[p] = ::open(paths[p].c_str(), O_RDONLY|O_CLOEXEC);
                ok = fds[p] >= 0;
            }
            slices[i].fds = fds;
            slices[i].offsets = { g.offset, u.offset, d.offset };
        }
        ok = ok && flashnext_execute_selected_moe(slices, tx, arena_bytes);
        for (int fd : fds) if (fd >= 0) ::close(fd);
        if (!ok) {
            return false;
        }
        std::copy(tx.selected_moe.begin(), tx.selected_moe.end(),
                selected_moe + it * flashnext_layer_transaction::hidden);
    }
    return true;
}

bool flashnext_execute_selected_moe_synthetic_rows(
        const float * pre_moe, const int32_t * selected_expert_ids,
        const double * routing_weights, size_t n_tokens, float * selected_moe,
        size_t arena_bytes) {
    constexpr size_t H = 2560, F_SYN = 640, K_SYN = flashnext_layer_transaction::experts_used;
    if (!pre_moe || !selected_expert_ids || !routing_weights || !selected_moe ||
            n_tokens == 0 || n_tokens > SIZE_MAX / H || arena_bytes < 64) return false;
    for (size_t t = 0; t < n_tokens; ++t) {
        std::fill(selected_moe + t*H, selected_moe + (t+1)*H, 0.0f);
        std::vector<float> gate(F_SYN), up(F_SYN), act(F_SYN);
        for (size_t s = 0; s < K_SYN; ++s) {
            const int32_t expert = selected_expert_ids[t*K_SYN+s];
            if (expert < 0 || expert >= 512 || !std::isfinite(routing_weights[t*K_SYN+s])) return false;
            for (size_t j = 0; j < F_SYN; ++j) {
                float g = 0.0f, u = 0.0f;
                for (size_t i = 0; i < H; ++i) {
                    const float x = pre_moe[t*H+i];
                    g += x * (0.00001f * float((expert + j + i) % 17 - 8));
                    u += x * (0.00001f * float((expert + 3*j + i) % 19 - 9));
                }
                gate[j] = g; up[j] = u; act[j] = g / (1.0f + std::exp(-g)) * u;
            }
            for (size_t i = 0; i < H; ++i) {
                float y = 0.0f;
                for (size_t j = 0; j < F_SYN; ++j) y += act[j] * (0.00001f * float((expert + i + 5*j) % 23 - 11));
                selected_moe[t*H+i] += float(routing_weights[t*K_SYN+s]) * y;
            }
        }
    }
    return true;
}

bool flashnext_execute_selected_gate_rows_layer(
        const flashnext_tensor_index & index, int layer,
        const float * activation, const int32_t * selected_expert_ids,
        size_t n_tokens, float * gate_output, size_t arena_bytes) {
    return flashnext_execute_selected_projection_rows_layer(index, layer, flashnext_projection::gate,
        activation, selected_expert_ids, n_tokens, gate_output, 640, 2560, arena_bytes);
}

bool flashnext_execute_selected_projection_rows_layer(
        const flashnext_tensor_index & index, int layer, flashnext_projection projection,
        const float * activation, const int32_t * selected_expert_ids,
        size_t n_tokens, float * output, size_t n_rows, size_t input_width,
        size_t arena_bytes) {
    constexpr size_t KSEL = flashnext_layer_transaction::experts_used;
    if (!activation || !selected_expert_ids || !output || n_tokens == 0 || n_rows == 0 || input_width == 0) return false;
    const size_t expected_rows = projection == flashnext_projection::down ? 2560 : 640;
    const size_t expected_width = projection == flashnext_projection::down ? 640 : 2560;
    if (n_rows != expected_rows || input_width != expected_width) return false;
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return false;
    void * arena = nullptr;
    if (posix_memalign(&arena, 64, arena_bytes) != 0) { ggml_backend_free(backend); return false; }
    uint64_t bytes_read = 0, pread_calls = 0;
    bool ok = true;
    for (size_t t = 0; t < n_tokens && ok; ++t) {
        for (size_t s = 0; s < KSEL && ok; ++s) {
            flashnext_tensor_descriptor d;
            ok = index.resolve_expert(layer, selected_expert_ids[t*KSEL+s], projection, d);
            int fd = ok ? ::open(d.path.c_str(), O_RDONLY|O_CLOEXEC) : -1;
            if (fd < 0) ok = false;
            if (ok) {
                const ggml_type type = projection == flashnext_projection::down ? DOWN_TYPE : GATE_UP_TYPE;
                const size_t activation_offset = projection == flashnext_projection::down
                    ? (t*KSEL+s)*input_width : t*input_width;
                // Gate/up inputs are stored as token-major blocks with one
                // hidden vector per selected-expert slot.  The callback
                // populates slot 0 for each token; selecting t*input_width
                // here points at token 0's unused slot tail for t > 0.
                // Keep the provider bound to the verified token block.
                const size_t corrected_activation_offset = projection == flashnext_projection::down
                    ? activation_offset : t*KSEL*input_width;
                const int projection_id = projection == flashnext_projection::gate ? 0 :
                                          projection == flashnext_projection::up ? 1 : 2;
                ok = tiled_matvec(backend, projection_id, fd, type, (int64_t)n_rows, d.offset,
                    activation + corrected_activation_offset, output + (t*KSEL+s)*n_rows,
                    arena, arena_bytes, bytes_read, pread_calls, layer, (int) t,
                    selected_expert_ids[t*KSEL+s], 0);
            }
            if (fd >= 0) ::close(fd);
        }
    }
    std::free(arena); ggml_backend_free(backend); return ok;
}
