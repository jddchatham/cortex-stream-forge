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

int main(int argc, char ** argv) {
    if (argc < 2) return 1;
    const char * model_path = argv[1];

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

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    std::vector<float> last_logits(n_vocab);

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

    int cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (last_logits[v] > last_logits[cur_token]) cur_token = v;
    }

    int cur_pos = n_tok;
    const int n_steps = 6; // steps 2..7

    std::vector<double> pread_ms_per_step;
    std::vector<double> compute_ms_per_step;
    std::vector<double> total_ms_per_step;

    for (int step = 2; step <= 1 + n_steps; ++step) {
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

    std::printf("\n=== PROFILE BREAKDOWN OVER %zu DECODE STEPS ===\n", total_ms_per_step.size());
    for (size_t s = 0; s < total_ms_per_step.size(); ++s) {
        std::printf("Step %zu: Total = %6.2f ms | Pread = %6.2f ms (%4.1f%%) | Compute = %6.2f ms (%4.1f%%)\n",
            s + 1, total_ms_per_step[s],
            pread_ms_per_step[s], (pread_ms_per_step[s] / total_ms_per_step[s]) * 100.0,
            compute_ms_per_step[s], (compute_ms_per_step[s] / total_ms_per_step[s]) * 100.0);
    }

    double avg_pread = std::accumulate(pread_ms_per_step.begin(), pread_ms_per_step.end(), 0.0) / pread_ms_per_step.size();
    double avg_comp = std::accumulate(compute_ms_per_step.begin(), compute_ms_per_step.end(), 0.0) / compute_ms_per_step.size();
    double avg_tot = std::accumulate(total_ms_per_step.begin(), total_ms_per_step.end(), 0.0) / total_ms_per_step.size();

    std::printf("\nAVERAGES:\n");
    std::printf("Mean Step Total:   %6.2f ms (%.2f tok/s)\n", avg_tot, 1000.0 / avg_tot);
    std::printf("Mean Pread:        %6.2f ms (%4.1f%%)\n", avg_pread, (avg_pread / avg_tot) * 100.0);
    std::printf("Mean Compute:      %6.2f ms (%4.1f%%)\n", avg_comp, (avg_comp / avg_tot) * 100.0);
    std::printf("Overhead/Routing:  %6.2f ms (%4.1f%%)\n", avg_tot - avg_pread - avg_comp, ((avg_tot - avg_pread - avg_comp) / avg_tot) * 100.0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
