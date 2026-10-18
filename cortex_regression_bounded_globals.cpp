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

struct MemoryStats {
    long vm_rss_kb = 0;
    long rss_anon_kb = 0;
    long rss_file_kb = 0;
};

static MemoryStats get_memory_stats() {
    MemoryStats stats;
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::sscanf(line.c_str() + 6, "%ld", &stats.vm_rss_kb);
        } else if (line.compare(0, 8, "RssAnon:") == 0) {
            std::sscanf(line.c_str() + 8, "%ld", &stats.rss_anon_kb);
        } else if (line.compare(0, 8, "RssFile:") == 0) {
            std::sscanf(line.c_str() + 8, "%ld", &stats.rss_file_kb);
        }
    }
    return stats;
}

static bool read_token_embeddings(int fd, uint64_t base_offset, const llama_token * tokens, int n_tokens, int n_embd, float * dst) {
    const size_t row_bytes = 1088;
    std::vector<uint8_t> row_buf(row_bytes);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);

    for (int i = 0; i < n_tokens; ++i) {
        uint64_t file_off = base_offset + (uint64_t)tokens[i] * row_bytes;
        size_t done = 0;
        while (done < row_bytes) {
            ssize_t n = pread(fd, row_buf.data() + done, row_bytes - done, (off_t)(file_off + done));
            if (n <= 0) return false;
            done += (size_t)n;
        }
        traits->to_float(row_buf.data(), dst + (size_t)i * n_embd, n_embd);
    }
    return true;
}

static bool compute_tiled_output_projection(int fd, uint64_t base_offset, int n_embd, int n_vocab,
                                            const float * norm_h, float * logits,
                                            uint8_t * tile_arena, size_t arena_bytes, ggml_backend_t backend) {
    const size_t row_bytes = 1088;
    const int tile_rows = arena_bytes / row_bytes;
    if (tile_rows < 1) return false;

    // Check runtime flag CORTEX_EXACT_LM_HEAD_SIMD (default: 1)
    const char * simd_env = std::getenv("CORTEX_EXACT_LM_HEAD_SIMD");
    bool use_direct_simd = (simd_env == nullptr || std::strcmp(simd_env, "0") != 0);

    if (use_direct_simd) {
        // Direct AVX2 SIMD path using ggml_get_type_traits_cpu
        const auto * cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        std::vector<uint8_t> q8_x(row_bytes);
        cpu_traits->from_float(norm_h, q8_x.data(), n_embd);

        for (int start_row = 0; start_row < n_vocab; start_row += tile_rows) {
            int cur_rows = std::min(tile_rows, n_vocab - start_row);
            size_t bytes_to_read = (size_t)cur_rows * row_bytes;
            uint64_t file_off = base_offset + (uint64_t)start_row * row_bytes;

            size_t done = 0;
            while (done < bytes_to_read) {
                ssize_t n = pread(fd, tile_arena + done, bytes_to_read - done, (off_t)(file_off + done));
                if (n <= 0) return false;
                done += (size_t)n;
            }

            #pragma omp parallel for num_threads(2) schedule(static)
            for (int r = 0; r < cur_rows; ++r) {
                const void * row_w = tile_arena + (size_t)r * row_bytes;
                cpu_traits->vec_dot(n_embd, &logits[start_row + r], 0, row_w, 0, q8_x.data(), 0, 1);
            }
        }
        return true;
    }

    // Baseline fallback path using GGML graph evaluation
    for (int start_row = 0; start_row < n_vocab; start_row += tile_rows) {
        int cur_rows = std::min(tile_rows, n_vocab - start_row);
        size_t bytes_to_read = (size_t)cur_rows * row_bytes;
        uint64_t file_off = base_offset + (uint64_t)start_row * row_bytes;

        size_t done = 0;
        while (done < bytes_to_read) {
            ssize_t n = pread(fd, tile_arena + done, bytes_to_read - done, (off_t)(file_off + done));
            if (n <= 0) return false;
            done += (size_t)n;
        }

        ggml_init_params ip { 16 * 1024 * 1024, nullptr, true };
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, n_embd, cur_rows);
        ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, 1);
        ggml_tensor * y = ggml_mul_mat(gctx, w, x);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
        w->data = tile_arena;
        ggml_backend_tensor_set(x, norm_h, 0, (size_t)n_embd * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(y, logits + start_row, 0, (size_t)cur_rows * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
    }
    return true;
}

static void apply_rmsnorm(const float * x, const float * w, int n_embd, float * dst) {
    double sum2 = 0.0;
    for (int i = 0; i < n_embd; ++i) sum2 += (double)x[i] * x[i];
    float mean = (float)(sum2 / n_embd);
    float scale = 1.0f / std::sqrt(mean + 1e-6f);
    for (int i = 0; i < n_embd; ++i) dst[i] = x[i] * scale * w[i];
}

struct TestCase {
    std::string name;
    std::string prompt;
    int max_tokens;
};

bool run_test_case(llama_model * model, int fd, uint64_t tok_embd_off,
                   const TestCase & tc, const std::string & ref_prefix,
                   ggml_backend_t backend, uint8_t * tile_arena, size_t arena_bytes,
                   double & out_decode_tok_s, double & out_head_ms, MemoryStats & out_mem) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tok = -llama_tokenize(vocab, tc.prompt.c_str(), tc.prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, tc.prompt.c_str(), tc.prompt.size(), tokens.data(), n_tok, true, true);

    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * norm_w = (const float *) model->output_norm->data;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = std::max(n_tok, 64);
    cp.n_threads = 2;
    cp.n_threads_batch = 2;

    std::vector<llama_context *> layer_ctxs(n_layer, nullptr);
    auto get_or_create_ctx = [&](int il) -> llama_context * {
        if (!layer_ctxs[il]) {
            char s_buf[16], e_buf[16];
            std::snprintf(s_buf, sizeof(s_buf), "%d", il);
            std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);
            layer_ctxs[il] = llama_init_from_model(model, cp);
            if (il > 0) llama_set_embeddings_layer_inp(layer_ctxs[il], il, false);
            llama_set_embeddings_layer_inp(layer_ctxs[il], il + 1, true);
        }
        return layer_ctxs[il];
    };

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    std::vector<float> input_embeddings((size_t)n_embd * n_tok);
    std::vector<float> last_logits(n_vocab);
    std::vector<float> norm_h(n_embd);

    // Read initial prompt embeddings externally via pread
    read_token_embeddings(fd, tok_embd_off, tokens.data(), n_tok, n_embd, input_embeddings.data());

    // Prefill across all 24 layers
    for (int il = 0; il < n_layer; ++il) {
        char s_buf[16], e_buf[16];
        std::snprintf(s_buf, sizeof(s_buf), "%d", il);
        std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
        setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

        llama_cortex_rebind_layer(model, il);
        llama_context * ctx = get_or_create_ctx(il);

        llama_batch batch = llama_batch_init(n_tok, n_embd, 1);
        const float * src_embd = (il == 0) ? input_embeddings.data() : current_hidden.data();
        for (int i = 0; i < n_tok; ++i) {
            for (int d = 0; d < n_embd; ++d) {
                batch.embd[i * n_embd + d] = src_embd[i * n_embd + d];
            }
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

    // Apply RMSNorm and tiled output projection for the final prompt token
    const float * final_prompt_h = current_hidden.data() + (size_t)(n_tok - 1) * n_embd;
    apply_rmsnorm(final_prompt_h, norm_w, n_embd, norm_h.data());

    auto t_head0 = bench_clock::now();
    compute_tiled_output_projection(fd, tok_embd_off, n_embd, n_vocab, norm_h.data(), last_logits.data(), tile_arena, arena_bytes, backend);
    auto t_head1 = bench_clock::now();
    double head_ms_accum = std::chrono::duration<double, std::milli>(t_head1 - t_head0).count();
    int head_calls = 1;

    // Verify prefill logits against golden reference
    std::string ref_log_path = ref_prefix + "_" + tc.name + "_prefill_logits.bin";
    std::ifstream f_log(ref_log_path, std::ios::binary);
    std::vector<float> ref_log(n_vocab);
    f_log.read(reinterpret_cast<char*>(ref_log.data()), n_vocab * sizeof(float));
    f_log.close();

    int logit_diffs = 0;
    float max_abs_diff = 0.0f;
    for (int v = 0; v < n_vocab; ++v) {
        float d = std::abs(last_logits[v] - ref_log[v]);
        if (d > max_abs_diff) max_abs_diff = d;
        if (d > 0.0f) logit_diffs++;
    }

    // Load golden tokens
    std::string ref_tok_path = ref_prefix + "_" + tc.name + "_tokens.bin";
    std::ifstream f_tok(ref_tok_path, std::ios::binary);
    int ref_count = 0;
    f_tok.read(reinterpret_cast<char*>(&ref_count), sizeof(int));
    std::vector<llama_token> ref_tokens(ref_count);
    f_tok.read(reinterpret_cast<char*>(ref_tokens.data()), ref_count * sizeof(llama_token));
    f_tok.close();

    // Greedy pick Step 1
    int cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (last_logits[v] > last_logits[cur_token]) cur_token = v;
    }

    std::vector<llama_token> gen_tokens = { (llama_token)cur_token };
    int cur_pos = n_tok;

    auto t_decode_start = bench_clock::now();
    int decode_steps = 0;

    std::vector<float> step_embd(n_embd);
    std::vector<float> step_hidden(n_embd);

    for (int step = 2; step <= tc.max_tokens; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) break;

        // External row-level embedding lookup
        read_token_embeddings(fd, tok_embd_off, (const llama_token*)&cur_token, 1, n_embd, step_embd.data());
        std::copy(step_embd.begin(), step_embd.end(), step_hidden.begin());

        for (int il = 0; il < n_layer; ++il) {
            char s_buf[16], e_buf[16];
            std::snprintf(s_buf, sizeof(s_buf), "%d", il);
            std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

            llama_cortex_rebind_layer(model, il);
            llama_context * ctx = get_or_create_ctx(il);

            llama_batch batch = llama_batch_init(1, n_embd, 1);
            for (int d = 0; d < n_embd; ++d) batch.embd[d] = step_hidden[d];
            batch.pos[0] = cur_pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = false;
            batch.n_tokens = 1;

            llama_decode(ctx, batch);
            float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
            std::copy(out_h, out_h + n_embd, step_hidden.begin());
            llama_batch_free(batch);
        }

        // Output projection via bounded tiles
        apply_rmsnorm(step_hidden.data(), norm_w, n_embd, norm_h.data());

        auto th0 = bench_clock::now();
        compute_tiled_output_projection(fd, tok_embd_off, n_embd, n_vocab, norm_h.data(), last_logits.data(), tile_arena, arena_bytes, backend);
        auto th1 = bench_clock::now();
        head_ms_accum += std::chrono::duration<double, std::milli>(th1 - th0).count();
        head_calls++;

        cur_token = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (last_logits[v] > last_logits[cur_token]) cur_token = v;
        }
        gen_tokens.push_back((llama_token)cur_token);
        cur_pos++;
        decode_steps++;
    }

    auto t_decode_end = bench_clock::now();
    double total_decode_sec = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
    out_decode_tok_s = decode_steps > 0 ? (decode_steps / total_decode_sec) : 0.0;
    out_head_ms = head_ms_accum / head_calls;
    out_mem = get_memory_stats();

    // Compare tokens against golden reference
    bool tokens_match = (gen_tokens.size() == ref_tokens.size());
    int match_count = 0;
    for (size_t i = 0; i < std::min(gen_tokens.size(), ref_tokens.size()); ++i) {
        if (gen_tokens[i] == ref_tokens[i]) match_count++;
        else {
            tokens_match = false;
            std::printf("MISMATCH at token %zu: gen=%d ref=%d\n", i, gen_tokens[i], ref_tokens[i]);
            break;
        }
    }

    std::printf("[%s] Prefill Max Diff: %.6f | Tokens: %d / %d matched (%s) | Decode: %.2f tok/s | LM-Head: %.1f ms | RSS: %.1f MiB\n",
        tc.name.c_str(), max_abs_diff, match_count, (int)ref_tokens.size(),
        tokens_match ? "PASS" : "FAIL", out_decode_tok_s, out_head_ms, out_mem.vm_rss_kb / 1024.0);

    for (int il = 0; il < n_layer; ++il) if (layer_ctxs[il]) llama_free(layer_ctxs[il]);
    return tokens_match && (max_abs_diff == 0.0f);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <model.gguf> <ref_prefix>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string ref_prefix = argv[2];

    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_EXTERNAL_GLOBAL_TENSORS", "1", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 2;

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) return 3;

    // GGUF v3 data_offset + relative offset = 11966848 + 552457472 = 564424320
    uint64_t tok_embd_off = 564424320;

    // Allow tile size selection via CORTEX_LM_HEAD_TILE_MIB (default: 4 MiB)
    size_t arena_bytes = 4 * 1024 * 1024;
    const char * tile_env = std::getenv("CORTEX_LM_HEAD_TILE_MIB");
    if (tile_env) {
        int mib = std::atoi(tile_env);
        if (mib > 0) arena_bytes = (size_t)mib * 1024 * 1024;
    }
    std::vector<uint8_t> tile_arena(arena_bytes);

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    std::vector<TestCase> cases = {
        { "q35_test_arithmetic_6", "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n", 6 },
        { "q35_test_reasoning_32", "<|im_start|>user\nSolve: If x + 5 = 12, what is x? Explain step by step.<|im_end|>\n<|im_start|>assistant\n", 32 },
        { "q35_test_code_64", "<|im_start|>user\nWrite a Python function to compute the factorial of n recursively.<|im_end|>\n<|im_start|>assistant\n", 64 }
    };

    std::printf("\n=========================================================================================\n");
    std::printf("CORTEX MILESTONE 7: OPERATION GLOBAL TENSOR RESIDENCY REGRESSION SUITE\n");
    std::printf("Bounded Pread Embedding + 16 MiB Tiled LM Head Projection + External Global Tensors\n");
    std::printf("=========================================================================================\n\n");

    bool all_pass = true;
    for (const auto & tc : cases) {
        double tok_s = 0.0, head_ms = 0.0;
        MemoryStats mem;
        if (!run_test_case(model, fd, tok_embd_off, tc, ref_prefix, backend, tile_arena.data(), arena_bytes, tok_s, head_ms, mem)) {
            all_pass = false;
        }
    }

    MemoryStats final_mem = get_memory_stats();
    std::printf("\n=========================================================================================\n");
    std::printf("FINAL REGRESSION RESULT: %s\n", all_pass ? "ALL PASSED (100% BITWISE PARITY)" : "FAILED");
    std::printf("Peak Process Residency: VmRSS=%.2f MiB | RssAnon=%.2f MiB | RssFile=%.2f MiB\n",
        final_mem.vm_rss_kb / 1024.0, final_mem.rss_anon_kb / 1024.0, final_mem.rss_file_kb / 1024.0);
    std::printf("=========================================================================================\n\n");

    close(fd);
    llama_model_free(model);
    llama_backend_free();
    return all_pass ? 0 : 4;
}
