#include "llama.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <algorithm>

using bench_clock = std::chrono::high_resolution_clock;

struct TimingBreakdown {
    double io_ms = 0.0;
    double gctx_alloc_ms = 0.0;
    double graph_build_ms = 0.0;
    double buf_alloc_ms = 0.0;
    double tensor_set_ms = 0.0;
    double compute_ms = 0.0;
    double tensor_get_ms = 0.0;
    double cleanup_ms = 0.0;
    size_t total_bytes_read = 0;
    int num_tiles = 0;
};

static void apply_rmsnorm(const float * x, const float * w, int n_embd, float * dst) {
    double sum2 = 0.0;
    for (int i = 0; i < n_embd; ++i) sum2 += (double)x[i] * x[i];
    float mean = (float)(sum2 / n_embd);
    float scale = 1.0f / std::sqrt(mean + 1e-6f);
    for (int i = 0; i < n_embd; ++i) dst[i] = x[i] * scale * w[i];
}

static TimingBreakdown audit_tiled_projection(int fd, uint64_t base_offset, int n_embd, int n_vocab,
                                             const float * norm_h, float * logits,
                                             uint8_t * tile_arena, size_t arena_bytes, ggml_backend_t backend) {
    TimingBreakdown tb;
    const size_t row_bytes = 1088;
    const int tile_rows = arena_bytes / row_bytes;

    for (int start_row = 0; start_row < n_vocab; start_row += tile_rows) {
        tb.num_tiles++;
        int cur_rows = std::min(tile_rows, n_vocab - start_row);
        size_t bytes_to_read = (size_t)cur_rows * row_bytes;
        uint64_t file_off = base_offset + (uint64_t)start_row * row_bytes;

        // 1. File read (pread)
        auto t0 = bench_clock::now();
        size_t done = 0;
        while (done < bytes_to_read) {
            ssize_t n = pread(fd, tile_arena + done, bytes_to_read - done, (off_t)(file_off + done));
            if (n <= 0) break;
            done += (size_t)n;
        }
        auto t1 = bench_clock::now();
        tb.io_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        tb.total_bytes_read += done;

        // 2. GGML context initialization
        auto t2 = bench_clock::now();
        ggml_init_params ip { 16 * 1024 * 1024, nullptr, true };
        ggml_context * gctx = ggml_init(ip);
        auto t3 = bench_clock::now();
        tb.gctx_alloc_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // 3. Graph construction
        auto t4 = bench_clock::now();
        ggml_tensor * w = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, n_embd, cur_rows);
        ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, 1);
        ggml_tensor * y = ggml_mul_mat(gctx, w, x);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, y);
        auto t5 = bench_clock::now();
        tb.graph_build_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

        // 4. Backend buffer allocation
        auto t6 = bench_clock::now();
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
        w->data = tile_arena;
        auto t7 = bench_clock::now();
        tb.buf_alloc_ms += std::chrono::duration<double, std::milli>(t7 - t6).count();

        // 5. Input tensor copy
        auto t8 = bench_clock::now();
        ggml_backend_tensor_set(x, norm_h, 0, (size_t)n_embd * sizeof(float));
        auto t9 = bench_clock::now();
        tb.tensor_set_ms += std::chrono::duration<double, std::milli>(t9 - t8).count();

        // 6. Kernel compute
        auto t10 = bench_clock::now();
        ggml_backend_graph_compute(backend, gf);
        auto t11 = bench_clock::now();
        tb.compute_ms += std::chrono::duration<double, std::milli>(t11 - t10).count();

        // 7. Output tensor copy
        auto t12 = bench_clock::now();
        ggml_backend_tensor_get(y, logits + start_row, 0, (size_t)cur_rows * sizeof(float));
        auto t13 = bench_clock::now();
        tb.tensor_get_ms += std::chrono::duration<double, std::milli>(t13 - t12).count();

        // 8. Cleanup
        auto t14 = bench_clock::now();
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
        auto t15 = bench_clock::now();
        tb.cleanup_ms += std::chrono::duration<double, std::milli>(t15 - t14).count();
    }
    return tb;
}

int main(int argc, char ** argv) {
    if (argc < 2) return 1;
    const char * model_path = argv[1];

    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_EXTERNAL_GLOBAL_TENSORS", "1", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * norm_w = (const float *) model->output_norm->data;

    int fd = open(model_path, O_RDONLY);
    uint64_t tok_embd_off = 564424320;

    // Synthetic normalized hidden state vector
    std::vector<float> dummy_h(n_embd, 0.05f);
    std::vector<float> norm_h(n_embd);
    apply_rmsnorm(dummy_h.data(), norm_w, n_embd, norm_h.data());

    std::vector<float> logits(n_vocab);
    size_t arena_bytes = 16 * 1024 * 1024;
    std::vector<uint8_t> tile_arena(arena_bytes);

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    // Warm-up run
    audit_tiled_projection(fd, tok_embd_off, n_embd, n_vocab, norm_h.data(), logits.data(), tile_arena.data(), arena_bytes, backend);

    // Benchmark 5 repeated iterations
    const int n_runs = 5;
    TimingBreakdown total;
    for (int r = 0; r < n_runs; ++r) {
        TimingBreakdown tb = audit_tiled_projection(fd, tok_embd_off, n_embd, n_vocab, norm_h.data(), logits.data(), tile_arena.data(), arena_bytes, backend);
        total.io_ms += tb.io_ms;
        total.gctx_alloc_ms += tb.gctx_alloc_ms;
        total.graph_build_ms += tb.graph_build_ms;
        total.buf_alloc_ms += tb.buf_alloc_ms;
        total.tensor_set_ms += tb.tensor_set_ms;
        total.compute_ms += tb.compute_ms;
        total.tensor_get_ms += tb.tensor_get_ms;
        total.cleanup_ms += tb.cleanup_ms;
        total.total_bytes_read = tb.total_bytes_read;
        total.num_tiles = tb.num_tiles;
    }

    double avg_io = total.io_ms / n_runs;
    double avg_gctx = total.gctx_alloc_ms / n_runs;
    double avg_graph = total.graph_build_ms / n_runs;
    double avg_buf = total.buf_alloc_ms / n_runs;
    double avg_set = total.tensor_set_ms / n_runs;
    double avg_comp = total.compute_ms / n_runs;
    double avg_get = total.tensor_get_ms / n_runs;
    double avg_clean = total.cleanup_ms / n_runs;
    double avg_total = avg_io + avg_gctx + avg_graph + avg_buf + avg_set + avg_comp + avg_get + avg_clean;

    std::printf("\n=========================================================================\n");
    std::printf("PHASE 2: DETAILED AUDIT OF OUTPUT PROJECTION (LM HEAD)\n");
    std::printf("Vocabulary Rows:        %d\n", n_vocab);
    std::printf("Total Bytes Read:       %zu bytes (%.2f MiB)\n", total.total_bytes_read, total.total_bytes_read / (1024.0 * 1024.0));
    std::printf("Number of Tiles:        %d tiles (%.2f MiB nominal per tile)\n", total.num_tiles, arena_bytes / (1024.0 * 1024.0));
    std::printf("-------------------------------------------------------------------------\n");
    std::printf("Sub-Component Latency Breakdown (Mean over %d runs):\n", n_runs);
    std::printf("  1. Positional pread I/O:        %6.2f ms (%5.1f%%)\n", avg_io, (avg_io / avg_total) * 100.0);
    std::printf("  2. GGML Context init (gctx):    %6.2f ms (%5.1f%%)\n", avg_gctx, (avg_gctx / avg_total) * 100.0);
    std::printf("  3. Graph build (mul_mat):       %6.2f ms (%5.1f%%)\n", avg_graph, (avg_graph / avg_total) * 100.0);
    std::printf("  4. Backend buffer alloc (buf):  %6.2f ms (%5.1f%%)\n", avg_buf, (avg_buf / avg_total) * 100.0);
    std::printf("  5. Input tensor copy (set):     %6.2f ms (%5.1f%%)\n", avg_set, (avg_set / avg_total) * 100.0);
    std::printf("  6. Quantized GEMV compute:      %6.2f ms (%5.1f%%)\n", avg_comp, (avg_comp / avg_total) * 100.0);
    std::printf("  7. Output logit copy (get):     %6.2f ms (%5.1f%%)\n", avg_get, (avg_get / avg_total) * 100.0);
    std::printf("  8. Backend & gctx cleanup:      %6.2f ms (%5.1f%%)\n", avg_clean, (avg_clean / avg_total) * 100.0);
    std::printf("-------------------------------------------------------------------------\n");
    std::printf("Total LM Head Latency:            %6.2f ms\n", avg_total);
    std::printf("Total Pure Overhead (2..5 + 7..8):%6.2f ms (%5.1f%% of total)\n",
        avg_total - avg_io - avg_comp, ((avg_total - avg_io - avg_comp) / avg_total) * 100.0);
    std::printf("=========================================================================\n\n");

    close(fd);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
