#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>

using bench_clock = std::chrono::high_resolution_clock;
static double ms_since(bench_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(bench_clock::now() - t).count();
}

struct BenchResult {
    double total_ms;
    double pread_or_wait_ms;
    double compute_ms;
    double tok_per_sec;
};

BenchResult run_benchmark(const char * model_path, bool use_prefetch, int n_decode_steps) {
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);

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

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    std::vector<float> last_logits(n_vocab);

    // Prefill
    if (use_prefetch && n_layer > 1) llama_cortex_prefetch_layer(model, 0);
    for (int il = 0; il < n_layer; ++il) {
        if (use_prefetch && il + 1 < n_layer) llama_cortex_prefetch_layer(model, il + 1);

        char s_buf[16], e_buf[16];
        std::snprintf(s_buf, sizeof(s_buf), "%d", il);
        std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
        setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

        if (il > 0) llama_set_embeddings_layer_inp(ctx, il, false);
        if (il + 1 < n_layer) llama_set_embeddings_layer_inp(ctx, il + 1, true);

        if (il > 0) llama_memory_clear(mem, false);

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

    int cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (last_logits[v] > last_logits[cur_token]) cur_token = v;
    }

    int cur_pos = n_tok;
    std::vector<double> step_total_times;
    std::vector<double> step_rebind_times;
    std::vector<double> step_compute_times;

    for (int step = 2; step <= 1 + n_decode_steps; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) break;
        std::vector<float> step_hidden(n_embd);

        double rebind_accum = 0.0;
        double compute_accum = 0.0;
        auto t0_step = bench_clock::now();

        if (use_prefetch && n_layer > 1) {
            llama_cortex_prefetch_layer(model, 0);
        }

        for (int il = 0; il < n_layer; ++il) {
            if (use_prefetch && il + 1 < n_layer) {
                llama_cortex_prefetch_layer(model, il + 1);
            }

            auto t_rb0 = bench_clock::now();
            llama_cortex_rebind_layer(model, il);
            rebind_accum += ms_since(t_rb0);

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
            compute_accum += ms_since(t_comp0);

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
        step_total_times.push_back(step_total);
        step_rebind_times.push_back(rebind_accum);
        step_compute_times.push_back(compute_accum);

        cur_token = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (last_logits[v] > last_logits[cur_token]) cur_token = v;
        }
        cur_pos++;
    }

    double mean_total = std::accumulate(step_total_times.begin(), step_total_times.end(), 0.0) / step_total_times.size();
    double mean_rebind = std::accumulate(step_rebind_times.begin(), step_rebind_times.end(), 0.0) / step_rebind_times.size();
    double mean_compute = std::accumulate(step_compute_times.begin(), step_compute_times.end(), 0.0) / step_compute_times.size();

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return { mean_total, mean_rebind, mean_compute, 1000.0 / mean_total };
}

int main(int argc, char ** argv) {
    if (argc < 2) return 1;
    const char * model_path = argv[1];
    const int repeats = 3;
    const int steps = 5;

    std::printf("===================================================================\n");
    std::printf("CORTEX BENCHMARK: SYNCHRONOUS PREAD VS ASYNCHRONOUS DOUBLE BUFFERING\n");
    std::printf("===================================================================\n\n");

    std::vector<BenchResult> sync_results;
    for (int r = 0; r < repeats; ++r) {
        std::printf("[Run %d/3] Measuring Synchronous Baseline...\n", r + 1);
        auto res = run_benchmark(model_path, false, steps);
        sync_results.push_back(res);
        std::printf("  -> Total: %6.2f ms | Pread: %6.2f ms | Compute: %6.2f ms | Rate: %.2f tok/s\n",
            res.total_ms, res.pread_or_wait_ms, res.compute_ms, res.tok_per_sec);
    }

    std::vector<BenchResult> prefetch_results;
    for (int r = 0; r < repeats; ++r) {
        std::printf("[Run %d/3] Measuring Asynchronous Double-Buffered Prefetch...\n", r + 1);
        auto res = run_benchmark(model_path, true, steps);
        prefetch_results.push_back(res);
        std::printf("  -> Total: %6.2f ms | Wait/Rebind: %6.2f ms | Compute: %6.2f ms | Rate: %.2f tok/s\n",
            res.total_ms, res.pread_or_wait_ms, res.compute_ms, res.tok_per_sec);
    }

    auto get_median = [](std::vector<BenchResult> v) -> BenchResult {
        std::sort(v.begin(), v.end(), [](const BenchResult & a, const BenchResult & b) {
            return a.total_ms < b.total_ms;
        });
        return v[v.size() / 2];
    };

    auto med_sync = get_median(sync_results);
    auto med_pref = get_median(prefetch_results);

    std::printf("\n===================================================================\n");
    std::printf("FINAL COMPARISON (MEDIAN OVER %d RUNS)\n", repeats);
    std::printf("===================================================================\n");
    std::printf("Metric                     | Synchronous Pread | Asynchronous Prefetch | Delta\n");
    std::printf("---------------------------+-------------------+-----------------------+----------------\n");
    std::printf("Decode Step Latency (ms)   | %17.2f | %21.2f | %+6.2f ms (%+.1f%%)\n",
        med_sync.total_ms, med_pref.total_ms, med_pref.total_ms - med_sync.total_ms,
        ((med_pref.total_ms - med_sync.total_ms) / med_sync.total_ms) * 100.0);
    std::printf("Pread / Wait Latency (ms)  | %17.2f | %21.2f | %+6.2f ms (%+.1f%%)\n",
        med_sync.pread_or_wait_ms, med_pref.pread_or_wait_ms, med_pref.pread_or_wait_ms - med_sync.pread_or_wait_ms,
        ((med_pref.pread_or_wait_ms - med_sync.pread_or_wait_ms) / med_sync.pread_or_wait_ms) * 100.0);
    std::printf("Compute Execution (ms)     | %17.2f | %21.2f | %+6.2f ms (%+.1f%%)\n",
        med_sync.compute_ms, med_pref.compute_ms, med_pref.compute_ms - med_sync.compute_ms,
        ((med_pref.compute_ms - med_sync.compute_ms) / med_sync.compute_ms) * 100.0);
    std::printf("Decode Throughput (tok/s)  | %17.2f | %21.2f | %+6.2f tok/s (%+.1f%%)\n",
        med_sync.tok_per_sec, med_pref.tok_per_sec, med_pref.tok_per_sec - med_sync.tok_per_sec,
        ((med_pref.tok_per_sec - med_sync.tok_per_sec) / med_sync.tok_per_sec) * 100.0);

    return 0;
}
