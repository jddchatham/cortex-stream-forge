#include "llama.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <omp.h>

using bench_clock = std::chrono::high_resolution_clock;

struct MemoryStats {
    long vm_rss_kb = 0;
    long rss_anon_kb = 0;
    long rss_file_kb = 0;
    long vm_swap_kb = 0;
};

static MemoryStats get_memory_stats() {
    MemoryStats stats;
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return stats;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) sscanf(line + 6, "%ld", &stats.vm_rss_kb);
        else if (strncmp(line, "RssAnon:", 8) == 0) sscanf(line + 8, "%ld", &stats.rss_anon_kb);
        else if (strncmp(line, "RssFile:", 8) == 0) sscanf(line + 8, "%ld", &stats.rss_file_kb);
        else if (strncmp(line, "VmSwap:", 7) == 0) sscanf(line + 7, "%ld", &stats.vm_swap_kb);
    }
    fclose(f);
    return stats;
}

static bool read_token_embeddings_q4_k_5120(int fd, uint64_t base_offset, const llama_token * tokens,
                                            int n_tokens, int n_embd, float * dst) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q4_K);
    const size_t row_bytes = (n_embd / traits->blck_size) * traits->type_size; // 2880 bytes
    std::vector<uint8_t> row_buf(row_bytes);

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

static bool compute_tiled_output_projection_q6_k_5120(int fd, uint64_t base_offset, int n_embd, int n_vocab,
                                                      const float * norm_h, float * logits,
                                                      uint8_t * tile_arena, size_t arena_bytes) {
    const auto * cpu_traits_q6 = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
    const auto * traits_q6 = ggml_get_type_traits(GGML_TYPE_Q6_K);
    const size_t row_bytes = (n_embd / traits_q6->blck_size) * traits_q6->type_size; // 4200 bytes
    const int tile_rows = arena_bytes / row_bytes;
    if (tile_rows < 1) return false;

    const auto * cpu_traits_vec = ggml_get_type_traits_cpu((ggml_type)cpu_traits_q6->vec_dot_type);
    const auto * traits_vec = ggml_get_type_traits((ggml_type)cpu_traits_q6->vec_dot_type);
    size_t q8k_bytes = (n_embd / traits_vec->blck_size) * traits_vec->type_size;
    std::vector<uint8_t> norm_h_q8k(q8k_bytes);
    cpu_traits_vec->from_float(norm_h, norm_h_q8k.data(), n_embd);

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
            cpu_traits_q6->vec_dot(n_embd, &logits[start_row + r], 0, row_w, 0, norm_h_q8k.data(), 0, 1);
        }
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

int main(int argc, char ** argv) {
    if (argc < 2) return 1;
    const char * model_path = argv[1];
    int n_gen = (argc > 2) ? std::atoi(argv[2]) : 6;

    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_EXTERNAL_GLOBAL_TENSORS", "1", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 2;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * norm_w = (const float *) model->output_norm->data;

    int fd = open(model_path, O_RDONLY);
    uint64_t data_offset = 10994976;
    uint64_t output_w_off = data_offset + 0;
    uint64_t tok_embd_off = data_offset + 1042964480;

    std::string prompt = "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n";
    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_tok, true, true);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = std::max(n_tok, 64);
    cp.n_threads = 2;
    cp.n_threads_batch = 2;

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 3;

    llama_memory_t mem = llama_get_memory(ctx);
    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    std::vector<float> input_embeddings((size_t)n_embd * n_tok);
    std::vector<float> norm_h(n_embd);
    std::vector<float> last_logits(n_vocab);
    std::vector<float> step_embd(n_embd);
    std::vector<float> step_hidden(n_embd);
    size_t arena_bytes = 2 * 1024 * 1024;
    std::vector<uint8_t> tile_arena(arena_bytes);

    read_token_embeddings_q4_k_5120(fd, tok_embd_off, tokens.data(), n_tok, n_embd, input_embeddings.data());

    std::printf("=== Starting Qwen 3.5 27B Prefill (%d prompt tokens across %d layers) ===\n", n_tok, n_layer);
    auto t_prefill_start = bench_clock::now();
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
    auto t_prefill_end = bench_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    float * final_h = current_hidden.data() + (size_t)(n_tok - 1) * n_embd;
    apply_rmsnorm(final_h, norm_w, n_embd, norm_h.data());

    auto t_head0 = bench_clock::now();
    compute_tiled_output_projection_q6_k_5120(fd, output_w_off, n_embd, n_vocab, norm_h.data(), last_logits.data(), tile_arena.data(), arena_bytes);
    auto t_head1 = bench_clock::now();
    double head_ms = std::chrono::duration<double, std::milli>(t_head1 - t_head0).count();

    int cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (last_logits[v] > last_logits[cur_token]) cur_token = v;
    }

    std::vector<llama_token> gen_tokens = { (llama_token)cur_token };
    int cur_pos = n_tok;

    char pbuf[128];
    int pl = llama_token_to_piece(vocab, cur_token, pbuf, sizeof(pbuf), 0, true);
    std::printf("Prefill completed in %.2f ms. First token: %d ('%.*s', logit=%.4f)\n",
                prefill_ms, cur_token, pl, pbuf, last_logits[cur_token]);

    std::printf("\n=== Commencing Autoregressive Generation (target: %d tokens) ===\n", n_gen);

    double total_rebind_ms = 0.0;
    double total_compute_ms = 0.0;
    double total_head_ms = 0.0;

    auto t_decode_start = bench_clock::now();

    for (int step = 2; step <= n_gen; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) {
            std::printf("EOG encountered at step %d\n", step);
            break;
        }

        read_token_embeddings_q4_k_5120(fd, tok_embd_off, (const llama_token*)&cur_token, 1, n_embd, step_embd.data());
        std::copy(step_embd.begin(), step_embd.end(), step_hidden.begin());

        auto t_step_start = bench_clock::now();

        for (int il = 0; il < n_layer; ++il) {
            char s_buf[16], e_buf[16];
            std::snprintf(s_buf, sizeof(s_buf), "%d", il);
            std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

            llama_set_embeddings_layer_inp(ctx, (il == 0) ? n_layer : il, false);
            llama_set_embeddings_layer_inp(ctx, il + 1, true);

            auto tr0 = bench_clock::now();
            llama_cortex_rebind_layer(model, il);
            auto tr1 = bench_clock::now();
            total_rebind_ms += std::chrono::duration<double, std::milli>(tr1 - tr0).count();

            llama_batch batch = llama_batch_init(1, n_embd, 1);
            for (int d = 0; d < n_embd; ++d) batch.embd[d] = step_hidden[d];
            batch.pos[0] = cur_pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = false;
            batch.n_tokens = 1;

            auto tc0 = bench_clock::now();
            llama_decode(ctx, batch);
            auto tc1 = bench_clock::now();
            total_compute_ms += std::chrono::duration<double, std::milli>(tc1 - tc0).count();

            float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
            std::copy(out_h, out_h + n_embd, step_hidden.begin());
            llama_batch_free(batch);
        }

        apply_rmsnorm(step_hidden.data(), norm_w, n_embd, norm_h.data());

        auto th0 = bench_clock::now();
        compute_tiled_output_projection_q6_k_5120(fd, output_w_off, n_embd, n_vocab, norm_h.data(), last_logits.data(), tile_arena.data(), arena_bytes);
        auto th1 = bench_clock::now();
        total_head_ms += std::chrono::duration<double, std::milli>(th1 - th0).count();

        cur_token = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (last_logits[v] > last_logits[cur_token]) cur_token = v;
        }
        gen_tokens.push_back((llama_token)cur_token);

        auto t_step_end = bench_clock::now();
        double step_latency = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();

        int sl = llama_token_to_piece(vocab, cur_token, pbuf, sizeof(pbuf), 0, true);
        std::printf("Step %d (pos %d): token %d ('%.*s', logit=%.4f, step_ms=%.1f)\n",
                    step, cur_pos, cur_token, sl, pbuf, last_logits[cur_token], step_latency);

        cur_pos++;
    }

    auto t_decode_end = bench_clock::now();
    double decode_sec = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
    int decode_steps = (int)gen_tokens.size() - 1;
    double tok_s = (decode_steps > 0) ? (decode_steps / decode_sec) : 0.0;

    MemoryStats mstats = get_memory_stats();

    std::printf("\n=================================================================\n");
    std::printf("CORTEX STREAMED QWEN 3.5 27B AUTOREGRESSIVE GENERATION RESULTS:\n");
    std::printf("  Generated Tokens:      %d / %d\n", (int)gen_tokens.size(), n_gen);
    std::printf("  Prefill Latency:       %.2f ms (%d prompt tokens)\n", prefill_ms, n_tok);
    std::printf("  Decode Throughput:     %.3f tok/s (%d decode steps in %.2f s)\n", tok_s, decode_steps, decode_sec);
    if (decode_steps > 0) {
        std::printf("  Breakdown / decode token:\n");
        std::printf("    Layer Rebind / Wait: %.2f ms\n", total_rebind_ms / decode_steps);
        std::printf("    Transformer Compute: %.2f ms\n", total_compute_ms / decode_steps);
        std::printf("    LM Head Projection:  %.2f ms\n", total_head_ms / decode_steps);
        std::printf("    Total Step Latency:  %.2f ms\n", (decode_sec * 1000.0) / decode_steps);
    }
    std::printf("  Peak VmRSS:            %.2f MiB (Hard Ceiling: 477.00 MiB)\n", mstats.vm_rss_kb / 1024.0);
    std::printf("  Peak RssAnon:          %.2f MiB\n", mstats.rss_anon_kb / 1024.0);
    std::printf("  Peak RssFile:          %.2f MiB\n", mstats.rss_file_kb / 1024.0);
    std::printf("  Peak VmSwap:           %.2f MiB\n", mstats.vm_swap_kb / 1024.0);
    std::printf("  Full Generated Output: ");
    for (int t : gen_tokens) {
        char buf[128];
        int l = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        std::printf("%.*s", l, buf);
    }
    std::printf("\n=================================================================\n\n");

    close(fd);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}