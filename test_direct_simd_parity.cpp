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
#include <omp.h>

using bench_clock = std::chrono::high_resolution_clock;

static void apply_rmsnorm(const float * x, const float * w, int n_embd, float * dst) {
    double sum2 = 0.0;
    for (int i = 0; i < n_embd; ++i) sum2 += (double)x[i] * x[i];
    float mean = (float)(sum2 / n_embd);
    float scale = 1.0f / std::sqrt(mean + 1e-6f);
    for (int i = 0; i < n_embd; ++i) dst[i] = x[i] * scale * w[i];
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

    // Load actual prefill reference hidden state from layer 23
    // We run layer 0..23 single-context prefill for "What is 2 + 2?..."
    std::string prompt = "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n";
    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_tok, true, true);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = n_tok;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);

    const size_t row_bytes = 1088;
    std::vector<uint8_t> row_buf(row_bytes);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);

    std::vector<float> input_embeddings((size_t)n_embd * n_tok);
    for (int i = 0; i < n_tok; ++i) {
        uint64_t file_off = tok_embd_off + (uint64_t)tokens[i] * row_bytes;
        pread(fd, row_buf.data(), row_bytes, (off_t)file_off);
        traits->to_float(row_buf.data(), input_embeddings.data() + (size_t)i * n_embd, n_embd);
    }

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    const int n_layer = llama_model_n_layer(model);
    for (int il = 0; il < n_layer; ++il) {
        char s_buf[16], e_buf[16];
        std::snprintf(s_buf, sizeof(s_buf), "%d", il);
        std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
        setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);
        llama_set_embeddings_layer_inp(ctx, (il == 0) ? n_layer : il, false);
        llama_set_embeddings_layer_inp(ctx, il + 1, true);
        if (il > 0) llama_memory_clear(mem, false);
        llama_cortex_rebind_layer(model, il);

        llama_batch batch = llama_batch_init(n_tok, n_embd, 1);
        const float * src_embd = (il == 0) ? input_embeddings.data() : current_hidden.data();
        for (int i = 0; i < n_tok; ++i) {
            for (int d = 0; d < n_embd; ++d) batch.embd[i * n_embd + d] = src_embd[i * n_embd + d];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false;
        }
        batch.n_tokens = n_tok;
        llama_decode(ctx, batch);
        float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
        std::copy(out_h, out_h + (size_t)n_embd * n_tok, current_hidden.begin());
        llama_batch_free(batch);
    }

    float * final_h = current_hidden.data() + (size_t)(n_tok - 1) * n_embd;
    std::vector<float> norm_h(n_embd);
    apply_rmsnorm(final_h, norm_w, n_embd, norm_h.data());

    // 1. Compute via Baseline Method (ggml_backend_graph_compute with per-tile context)
    size_t arena_bytes = 16 * 1024 * 1024;
    std::vector<uint8_t> tile_arena(arena_bytes);
    std::vector<float> baseline_logits(n_vocab);
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    const int tile_rows = arena_bytes / row_bytes;
    auto t_base0 = bench_clock::now();
    for (int start_row = 0; start_row < n_vocab; start_row += tile_rows) {
        int cur_rows = std::min(tile_rows, n_vocab - start_row);
        size_t bytes_to_read = (size_t)cur_rows * row_bytes;
        uint64_t file_off = tok_embd_off + (uint64_t)start_row * row_bytes;
        pread(fd, tile_arena.data(), bytes_to_read, (off_t)file_off);

        ggml_init_params ip { 16 * 1024 * 1024, nullptr, true };
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, n_embd, cur_rows);
        ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, 1);
        ggml_tensor * y = ggml_mul_mat(gctx, w, x);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
        w->data = tile_arena.data();
        ggml_backend_tensor_set(x, norm_h.data(), 0, (size_t)n_embd * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(y, baseline_logits.data() + start_row, 0, (size_t)cur_rows * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
    }
    auto t_base1 = bench_clock::now();
    double base_ms = std::chrono::duration<double, std::milli>(t_base1 - t_base0).count();

    // 2. Compute via Direct SIMD Method (quantize norm_h ONCE + direct AVX2 vec_dot)
    std::vector<float> simd_logits(n_vocab);
    std::vector<uint8_t> q8_x(row_bytes); // quantized Q8_0 representation of norm_h
    const auto * cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);

    auto t_simd0 = bench_clock::now();
    cpu_traits->from_float(norm_h.data(), q8_x.data(), n_embd);

    for (int start_row = 0; start_row < n_vocab; start_row += tile_rows) {
        int cur_rows = std::min(tile_rows, n_vocab - start_row);
        size_t bytes_to_read = (size_t)cur_rows * row_bytes;
        uint64_t file_off = tok_embd_off + (uint64_t)start_row * row_bytes;
        pread(fd, tile_arena.data(), bytes_to_read, (off_t)file_off);

        #pragma omp parallel for num_threads(2) schedule(static)
        for (int r = 0; r < cur_rows; ++r) {
            const void * row_w = tile_arena.data() + (size_t)r * row_bytes;
            cpu_traits->vec_dot(n_embd, &simd_logits[start_row + r], 0, row_w, 0, q8_x.data(), 0, 1);
        }
    }
    auto t_simd1 = bench_clock::now();
    double simd_ms = std::chrono::duration<double, std::milli>(t_simd1 - t_simd0).count();

    // Compare with Golden Stock Reference
    const char * ref_path = "/home/user/bitnet-cortex/quant-refs/q8_0_q35_test_arithmetic_6_prefill_logits.bin";
    std::vector<float> ref_logits(n_vocab);
    FILE * rf = std::fopen(ref_path, "rb");
    std::fread(ref_logits.data(), sizeof(float), n_vocab, rf);
    std::fclose(rf);

    float max_err_base = 0.0f;
    float max_err_simd = 0.0f;
    float max_diff_base_simd = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        float eb = std::fabs(baseline_logits[i] - ref_logits[i]);
        if (eb > max_err_base) max_err_base = eb;

        float es = std::fabs(simd_logits[i] - ref_logits[i]);
        if (es > max_err_simd) max_err_simd = es;

        float diff = std::fabs(baseline_logits[i] - simd_logits[i]);
        if (diff > max_diff_base_simd) max_diff_base_simd = diff;
    }

    int top_ref  = std::max_element(ref_logits.begin(), ref_logits.end()) - ref_logits.begin();
    int top_base = std::max_element(baseline_logits.begin(), baseline_logits.end()) - baseline_logits.begin();
    int top_simd = std::max_element(simd_logits.begin(), simd_logits.end()) - simd_logits.begin();

    std::printf("\n=========================================================================\n");
    std::printf("DIRECT SIMD VEC_DOT vs GGML_MUL_MAT PARITY CHECK:\n");
    std::printf("  Baseline (GGML graph) Max Error vs Ref: %f\n", max_err_base);
    std::printf("  Direct SIMD (vec_dot) Max Error vs Ref: %f\n", max_err_simd);
    std::printf("  Max Absolute Difference (Base vs SIMD): %f\n", max_diff_base_simd);
    std::printf("  Top Token Match: ref=%d (%.4f) | base=%d (%.4f) | simd=%d (%.4f)\n",
        top_ref, ref_logits[top_ref], top_base, baseline_logits[top_base], top_simd, simd_logits[top_simd]);
    std::printf("-------------------------------------------------------------------------\n");
    std::printf("Execution Latency:\n");
    std::printf("  Baseline GGML graph loop:  %.2f ms\n", base_ms);
    std::printf("  Direct AVX2 SIMD loop:     %.2f ms (Speedup: %.2fx, Saved: %.2f ms)\n",
        simd_ms, base_ms / simd_ms, base_ms - simd_ms);
    std::printf("=========================================================================\n\n");

    close(fd);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
