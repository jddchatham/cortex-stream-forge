#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <algorithm>

using bench_clock = std::chrono::high_resolution_clock;

struct OpStats {
    uint64_t count = 0;
    double total_us = 0.0;
};

static std::unordered_map<int, OpStats> g_op_stats;
static std::unordered_map<std::string, OpStats> g_named_stats;
static bench_clock::time_point g_op_t0;
static bool g_profile_active = false;

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;
    if (!g_profile_active) return true;

    if (ask) {
        // Before execution
        g_op_t0 = bench_clock::now();
        return true;
    } else {
        // After execution
        auto t1 = bench_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - g_op_t0).count();
        int op = (int)t->op;
        g_op_stats[op].count++;
        g_op_stats[op].total_us += us;

        const char * name = ggml_get_name(t);
        if (name && name[0] != '\0') {
            g_named_stats[name].count++;
            g_named_stats[name].total_us += us;
        }
        return true;
    }
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
    cp.cb_eval = eval_callback;
    cp.cb_eval_user_data = nullptr;

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    std::vector<float> last_logits(n_vocab);

    // Warmup prefill without profiling
    for (int il = 0; il < n_layer; ++il) {
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

    // Now profile 5 decode tokens
    g_profile_active = true;
    const int n_profile_steps = 5;

    for (int step = 2; step <= 1 + n_profile_steps; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) break;
        std::vector<float> step_hidden(n_embd);

        for (int il = 0; il < n_layer; ++il) {
            llama_cortex_rebind_layer(model, il);
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

            llama_decode(ctx, batch);

            if (il + 1 < n_layer) {
                float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
                std::copy(out_h, out_h + n_embd, step_hidden.begin());
            } else {
                float * step_logits = llama_get_logits_ith(ctx, 0);
                std::copy(step_logits, step_logits + n_vocab, last_logits.begin());
            }
            llama_batch_free(batch);
        }

        cur_token = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (last_logits[v] > last_logits[cur_token]) cur_token = v;
        }
        cur_pos++;
    }
    g_profile_active = false;

    // Report operator stats
    struct OpReport {
        int op;
        std::string name;
        uint64_t count;
        double total_ms;
        double avg_us;
        double pct;
    };

    double grand_total_us = 0.0;
    for (const auto & kv : g_op_stats) grand_total_us += kv.second.total_us;

    std::vector<OpReport> reports;
    for (const auto & kv : g_op_stats) {
        int op = kv.first;
        const char * name = ggml_op_name((enum ggml_op)op);
        double ms = kv.second.total_us / 1000.0;
        double avg = kv.second.count > 0 ? (kv.second.total_us / kv.second.count) : 0.0;
        double pct = (kv.second.total_us / grand_total_us) * 100.0;
        reports.push_back({ op, name ? name : "UNKNOWN", kv.second.count, ms, avg, pct });
    }

    std::sort(reports.begin(), reports.end(), [](const OpReport & a, const OpReport & b) {
        return a.total_ms > b.total_ms;
    });

    std::printf("\n===================================================================\n");
    std::printf("GGML COMPUTE OPERATOR BREAKDOWN (OVER %d DECODE TOKENS)\n", n_profile_steps);
    std::printf("===================================================================\n");
    std::printf("%-20s | %10s | %12s | %10s | %8s\n", "Operator", "Calls", "Total (ms)", "Avg (us)", "Share %");
    std::printf("---------------------+------------+--------------+------------+---------\n");
    for (const auto & r : reports) {
        std::printf("%-20s | %10lu | %12.2f | %10.2f | %7.2f%%\n",
            r.name.c_str(), r.count, r.total_ms, r.avg_us, r.pct);
    }
    std::printf("---------------------+------------+--------------+------------+---------\n");
    std::printf("%-20s | %10s | %12.2f | %10s | 100.00%%\n",
        "TOTAL", "", grand_total_us / 1000.0, "");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
