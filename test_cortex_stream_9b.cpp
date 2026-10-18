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

// Token embeddings for Qwen 3.5 9B: type is Q4_K (type 12)!
static bool read_token_embeddings_q4_k(int fd, uint64_t base_offset, const llama_token * tokens,
                                       int n_tokens, int n_embd, float * dst) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q4_K);
    const size_t row_bytes = 2304; // For 4096 dim
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

// Output projection for Qwen 3.5 9B: type is Q6_K (type 14) at separate output.weight!
// Direct SIMD implementation: quantizes norm_h once into vec_dot_type (Q8_K) and computes dot products directly.
static bool compute_tiled_output_projection_q6_k(int fd, uint64_t base_offset, int n_embd, int n_vocab,
                                                 const float * norm_h, float * logits,
                                                 uint8_t * tile_arena, size_t arena_bytes) {
    const size_t row_bytes = 3360; // For 4096 dim Q6_K
    const int tile_rows = arena_bytes / row_bytes;
    if (tile_rows < 1) return false;

    const auto * cpu_traits_q6 = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
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
    const int n_gen = (argc > 2) ? std::atoi(argv[2]) : 6;
    std::string ptype = (argc > 3) ? argv[3] : "arithmetic";

    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_EXTERNAL_GLOBAL_TENSORS", "1", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;
    MemoryStats ms_load = get_memory_stats();
    std::printf("[MEM] After model load: RSS=%.2f MiB, Anon=%.2f MiB\n", ms_load.vm_rss_kb/1024.0, ms_load.rss_anon_kb/1024.0);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * norm_w = (const float *) model->output_norm->data;

    int fd = open(model_path, O_RDONLY);
    uint64_t tok_embd_off = 5109495968;
    uint64_t output_w_off = 4275124384;

    std::string prompt;
    std::string ref_prefix;
    if (ptype == "reasoning") {
        prompt = "<|im_start|>user\nIf all bloops are razzies and all razzies are fizzies, are all bloops fizzies? Explain briefly.<|im_end|>\n<|im_start|>assistant\n";
        ref_prefix = "/home/user/bitnet-cortex/quant-refs/qwen35_9b_stock_reasoning_64";
    } else if (ptype == "code") {
        prompt = "<|im_start|>user\nWrite a C++ function to reverse a string in-place.<|im_end|>\n<|im_start|>assistant\n";
        ref_prefix = "/home/user/bitnet-cortex/quant-refs/qwen35_9b_stock_code_64";
    } else { // arithmetic
        prompt = "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n";
        ref_prefix = "/home/user/bitnet-cortex/quant-refs/qwen35_9b_stock_arith_64";
    }

    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_tok, true, true);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = std::max(n_tok, 64);
    cp.n_threads = 2;
    cp.n_threads_batch = 2;

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    MemoryStats ms_ctx = get_memory_stats();
    std::printf("[MEM] After ctx init: RSS=%.2f MiB, Anon=%.2f MiB\n", ms_ctx.vm_rss_kb/1024.0, ms_ctx.rss_anon_kb/1024.0);
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    std::vector<float> input_embeddings((size_t)n_embd * n_tok);
    std::vector<float> norm_h(n_embd);
    std::vector<float> last_logits(n_vocab);

    read_token_embeddings_q4_k(fd, tok_embd_off, tokens.data(), n_tok, n_embd, input_embeddings.data());

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

    size_t arena_bytes = 4 * 1024 * 1024;
    std::vector<uint8_t> tile_arena(arena_bytes);

    auto t_head0 = bench_clock::now();
    compute_tiled_output_projection_q6_k(fd, output_w_off, n_embd, n_vocab, norm_h.data(), last_logits.data(), tile_arena.data(), arena_bytes);
    auto t_head1 = bench_clock::now();
    double head_ms = std::chrono::duration<double, std::milli>(t_head1 - t_head0).count();

    // Load golden prefill reference if exists
    std::string ref_path = ref_prefix + "_prefill_logits.bin";
    std::ifstream f_ref(ref_path, std::ios::binary);
    std::vector<float> ref_logits(n_vocab);
    float max_diff = -1.0f;
    if (f_ref.is_open()) {
        f_ref.read(reinterpret_cast<char*>(ref_logits.data()), n_vocab * sizeof(float));
        f_ref.close();
        max_diff = 0.0f;
        for (int v = 0; v < n_vocab; ++v) {
            float d = std::fabs(last_logits[v] - ref_logits[v]);
            if (d > max_diff) max_diff = d;
        }
    }

    int cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (last_logits[v] > last_logits[cur_token]) cur_token = v;
    }

    char buf[128];
    int l = llama_token_to_piece(vocab, cur_token, buf, sizeof(buf), 0, true);

    std::vector<llama_token> gen_tokens = { (llama_token)cur_token };
    int cur_pos = n_tok;

    std::vector<float> step_embd(n_embd);
    std::vector<float> step_hidden(n_embd);

    auto t_decode_start = bench_clock::now();

    for (int step = 2; step <= n_gen; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) break;

        read_token_embeddings_q4_k(fd, tok_embd_off, (const llama_token*)&cur_token, 1, n_embd, step_embd.data());
        std::copy(step_embd.begin(), step_embd.end(), step_hidden.begin());

        for (int il = 0; il < n_layer; ++il) {
            char s_buf[16], e_buf[16];
            std::snprintf(s_buf, sizeof(s_buf), "%d", il);
            std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

            llama_set_embeddings_layer_inp(ctx, (il == 0) ? n_layer : il, false);
            llama_set_embeddings_layer_inp(ctx, il + 1, true);

            llama_cortex_rebind_layer(model, il);

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

        apply_rmsnorm(step_hidden.data(), norm_w, n_embd, norm_h.data());
        compute_tiled_output_projection_q6_k(fd, output_w_off, n_embd, n_vocab, norm_h.data(), last_logits.data(), tile_arena.data(), arena_bytes);

        cur_token = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (last_logits[v] > last_logits[cur_token]) cur_token = v;
        }
        gen_tokens.push_back((llama_token)cur_token);
        cur_pos++;
    }

    auto t_decode_end = bench_clock::now();
    double decode_sec = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
    int decode_steps = (int)gen_tokens.size() - 1;
    double tok_s = (decode_steps > 0) ? (decode_steps / decode_sec) : 0.0;

    // Load golden tokens
    std::string ref_tok_path = ref_prefix + "_tokens.bin";
    std::ifstream f_tok(ref_tok_path, std::ios::binary);
    int ref_count = 0;
    std::vector<llama_token> ref_tokens;
    if (f_tok.is_open()) {
        f_tok.read(reinterpret_cast<char*>(&ref_count), sizeof(int));
        ref_tokens.resize(ref_count);
        f_tok.read(reinterpret_cast<char*>(ref_tokens.data()), ref_count * sizeof(llama_token));
        f_tok.close();
    }

    int match_count = 0;
    int check_len = std::min(gen_tokens.size(), ref_tokens.size());
    for (int i = 0; i < check_len; ++i) {
        if (gen_tokens[i] == ref_tokens[i]) match_count++;
    }
    bool matched = (check_len > 0 && match_count == check_len);

    MemoryStats mstats = get_memory_stats();

    std::printf("\n=================================================================\n");
    std::printf("CORTEX STREAMED QWEN 3.5 9B RESULTS & TELEMETRY [%s, n_gen=%d]:\n", ptype.c_str(), n_gen);
    std::printf("  Tokens Matched:        %d / %d (%s)\n", match_count, check_len, matched ? "PASS" : "FAIL");
    std::printf("  Prefill Latency:       %.2f ms (%d tokens)\n", prefill_ms, n_tok);
    std::printf("  LM Head Tiled Time:    %.2f ms (all %d vocab rows)\n", head_ms, n_vocab);
    std::printf("  Decode Throughput:     %.2f tok/s (%d decode steps)\n", tok_s, decode_steps);
    std::printf("  Max Logit Difference: %.6f\n", max_diff);
    std::printf("  Prefill Greedy Token:  %d ('%.*s')\n", cur_token, l, buf);
    std::printf("  Peak VmRSS:            %.2f MiB\n", mstats.vm_rss_kb / 1024.0);
    std::printf("  Peak RssAnon:          %.2f MiB\n", mstats.rss_anon_kb / 1024.0);
    std::printf("  Peak RssFile:          %.2f MiB\n", mstats.rss_file_kb / 1024.0);
    std::printf("  Peak VmSwap:           %.2f MiB\n", mstats.vm_swap_kb / 1024.0);
    std::printf("  Generated Text:        ");
    for (int t : gen_tokens) {
        char pbuf[128];
        int pl = llama_token_to_piece(vocab, t, pbuf, sizeof(pbuf), 0, true);
        std::printf("%.*s", pl, pbuf);
    }
    std::printf("\n=================================================================\n\n");

    close(fd);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return matched ? 0 : 1;
}
