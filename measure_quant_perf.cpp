#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <unistd.h>

using bench_clock = std::chrono::high_resolution_clock;
static double ms_since(bench_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(bench_clock::now() - t).count();
}

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

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model_path> [n_steps]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    int n_steps = argc > 2 ? std::atoi(argv[2]) : 8;

    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 2;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string prompt = "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n";
    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_tok, true, true);
    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = n_tok;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    std::vector<float> last_logits(n_vocab);

    auto t0_prefill = bench_clock::now();
    // Prefill
    for (int il = 0; il < n_layer; ++il) {
        char s_buf[16], e_buf[16];
        std::snprintf(s_buf, sizeof(s_buf), "%d", il);
        std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
        setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

        if (il > 0) llama_set_embeddings_layer_inp(ctx, il, false);
        if (il + 1 < n_layer) llama_set_embeddings_layer_inp(ctx, il + 1, true);

        if (il > 0) {
            llama_memory_clear(mem, false);
        }

        llama_cortex_rebind_layer(model, il);

        llama_batch batch;
        if (il == 0) {
            batch = llama_batch_init(n_tok, 0, 1);
            for (int i = 0; i < n_tok; ++i) {
                batch.token[i] = tokens[i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = false;
            }
            batch.n_tokens = n_tok;
        } else {
            batch = llama_batch_init(n_tok, n_embd, 1);
            for (int i = 0; i < n_tok; ++i) {
                for (int d = 0; d < n_embd; ++d) {
                    batch.embd[i * n_embd + d] = current_hidden[i * n_embd + d];
                }
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (il == n_layer - 1 && i == n_tok - 1);
            }
            batch.n_tokens = n_tok;
        }

        llama_decode(ctx, batch);

        if (il + 1 < n_layer) {
            float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
            std::copy(out_h, out_h + (size_t)n_embd * n_tok, current_hidden.begin());
        } else {
            float * step_logits = llama_get_logits_ith(ctx, n_tok - 1);
            std::copy(step_logits, step_logits + n_vocab, last_logits.begin());
        }
        llama_batch_free(batch);
    }
    double prefill_ms = ms_since(t0_prefill);

    int cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (last_logits[v] > last_logits[cur_token]) cur_token = v;
    }

    int cur_pos = n_tok;
    std::vector<double> pread_ms_per_step;
    std::vector<double> compute_ms_per_step;
    std::vector<double> total_ms_per_step;

    for (int step = 1; step <= n_steps; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) break;
        std::vector<float> step_hidden(n_embd);

        double step_pread = 0.0;
        double step_compute = 0.0;
        auto t0_step = bench_clock::now();

        for (int il = 0; il < n_layer; ++il) {
            auto t_read0 = bench_clock::now();
            llama_cortex_rebind_layer(model, il);
            step_pread += ms_since(t_read0);

            char s_buf[16], e_buf[16];
            std::snprintf(s_buf, sizeof(s_buf), "%d", il);
            std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

            if (il > 0) llama_set_embeddings_layer_inp(ctx, il, false);
            if (il + 1 < n_layer) llama_set_embeddings_layer_inp(ctx, il + 1, true);

            llama_batch batch;
            if (il == 0) {
                batch = llama_batch_init(1, 0, 1);
                batch.token[0] = cur_token;
                batch.pos[0] = cur_pos;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0] = false;
                batch.n_tokens = 1;
            } else {
                batch = llama_batch_init(1, n_embd, 1);
                for (int d = 0; d < n_embd; ++d) batch.embd[d] = step_hidden[d];
                batch.pos[0] = cur_pos;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0] = (il == n_layer - 1);
                batch.n_tokens = 1;
            }

            auto t_comp0 = bench_clock::now();
            llama_decode(ctx, batch);
            step_compute += ms_since(t_comp0);

            if (il + 1 < n_layer) {
                float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
                std::copy(out_h, out_h + n_embd, step_hidden.begin());
            } else {
                float * step_logits = llama_get_logits_ith(ctx, 0);
                std::copy(step_logits, step_logits + n_vocab, last_logits.begin());
            }
            llama_batch_free(batch);
        }

        double step_total = ms_since(t0_step);
        pread_ms_per_step.push_back(step_pread);
        compute_ms_per_step.push_back(step_compute);
        total_ms_per_step.push_back(step_total);

        cur_token = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (last_logits[v] > last_logits[cur_token]) cur_token = v;
        }
        cur_pos++;
    }

    MemoryStats mem_stats = get_memory_stats();

    std::sort(total_ms_per_step.begin(), total_ms_per_step.end());
    double median_total = total_ms_per_step[total_ms_per_step.size() / 2];
    double mean_pread = std::accumulate(pread_ms_per_step.begin(), pread_ms_per_step.end(), 0.0) / pread_ms_per_step.size();
    double mean_compute = std::accumulate(compute_ms_per_step.begin(), compute_ms_per_step.end(), 0.0) / compute_ms_per_step.size();

    std::printf("BENCHMARK_RESULT: model=%s\n", model_path);
    std::printf("  Prefill:      %6.2f ms (%d tokens, %.2f tok/s)\n", prefill_ms, n_tok, (n_tok * 1000.0) / prefill_ms);
    std::printf("  Median Step:  %6.2f ms (%.2f tok/s)\n", median_total, 1000.0 / median_total);
    std::printf("  Mean Pread:   %6.2f ms\n", mean_pread);
    std::printf("  Mean Compute: %6.2f ms\n", mean_compute);
    std::printf("  VmRSS:        %6.2f MiB\n", mem_stats.vm_rss_kb / 1024.0);
    std::printf("  RssAnon:      %6.2f MiB\n", mem_stats.rss_anon_kb / 1024.0);
    std::printf("  RssFile:      %6.2f MiB\n", mem_stats.rss_file_kb / 1024.0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
