#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <chrono>

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
    cp.n_ctx = 256; cp.n_batch = n_tok;

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
            if (il + 1 < n_layer) llama_set_embeddings_layer_inp(layer_ctxs[il], il + 1, true);
        }
        return layer_ctxs[il];
    };

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    std::vector<float> last_logits(n_vocab);

    // Prefill
    for (int il = 0; il < n_layer; ++il) {
        llama_cortex_rebind_layer(model, il);
        llama_context * ctx = get_or_create_ctx(il);
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

        if (llama_decode(ctx, batch) != 0) return 4;

        if (il + 1 < n_layer) {
            float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
            std::copy(out_h, out_h + (size_t)n_embd * n_tok, current_hidden.begin());
        } else {
            float * step_logits = llama_get_logits_ith(ctx, n_tok - 1);
            std::copy(step_logits, step_logits + n_vocab, last_logits.begin());
        }
        llama_batch_free(batch);
    }

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(n_vocab);
    for (int v = 0; v < n_vocab; ++v) ranked.push_back({ last_logits[v], v });
    std::sort(ranked.rbegin(), ranked.rend());
    llama_token cur_token = ranked[0].second;

    char piece[128];
    int pn = llama_token_to_piece(vocab, cur_token, piece, sizeof(piece), 0, false);
    std::printf("Step 1 (First Token): ID=%-6d piece='%.*s'\n", cur_token, pn > 0 ? pn : 0, pn > 0 ? piece : "");

    int cur_pos = n_tok;
    std::vector<llama_token> generated_tokens = { cur_token };

    for (int step = 2; step <= 8; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) {
            std::printf("EOG encountered at step %d\n", step);
            break;
        }
        auto t_step = bench_clock::now();
        std::vector<float> step_hidden(n_embd);

        for (int il = 0; il < n_layer; ++il) {
            llama_cortex_rebind_layer(model, il);
            char s_buf[16], e_buf[16]; std::snprintf(s_buf, sizeof(s_buf), "%d", il); std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1); setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1); setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);
            llama_context * ctx = get_or_create_ctx(il);
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

            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "Decode failed in step %d layer %d\n", step, il);
                return 6;
            }

            if (il + 1 < n_layer) {
                float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
                std::copy(out_h, out_h + n_embd, step_hidden.begin());
            } else {
                float * step_logits = llama_get_logits_ith(ctx, 0);
                std::copy(step_logits, step_logits + n_vocab, last_logits.begin());
            }
            llama_batch_free(batch);
        }

        ranked.clear();
        for (int v = 0; v < n_vocab; ++v) ranked.push_back({ last_logits[v], v });
        std::sort(ranked.rbegin(), ranked.rend());
        cur_token = ranked[0].second;
        generated_tokens.push_back(cur_token);
        cur_pos++;

        pn = llama_token_to_piece(vocab, cur_token, piece, sizeof(piece), 0, false);
        double step_ms = ms_since(t_step);
        std::printf("Step %d: ID=%-6d piece='%.*s' (latency: %.2f ms, %.2f tok/s)\n",
            step, cur_token, pn > 0 ? pn : 0, pn > 0 ? piece : "", step_ms, 1000.0 / step_ms);
    }

    std::printf("\n=== COMPLETE GENERATED TEXT ===\n");
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        pn = llama_token_to_piece(vocab, generated_tokens[i], piece, sizeof(piece), 0, false);
        std::printf("%.*s", pn > 0 ? pn : 0, pn > 0 ? piece : "");
    }
    std::printf("\n");

    for (int il = 0; il < n_layer; ++il) if (layer_ctxs[il]) llama_free(layer_ctxs[il]);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
