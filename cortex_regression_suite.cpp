#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

struct TestCase {
    std::string name;
    std::string prompt;
    int max_tokens;
};

bool verify_test(llama_model * model, const TestCase & tc, const std::string & ref_prefix) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tok = -llama_tokenize(vocab, tc.prompt.c_str(), tc.prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, tc.prompt.c_str(), tc.prompt.size(), tokens.data(), n_tok, true, true);
    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = std::max(n_tok, 64);

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

    // Verify prefill logits
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

    for (int step = 2; step <= tc.max_tokens; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) break;

        std::vector<float> step_hidden(n_embd);
        for (int il = 0; il < n_layer; ++il) {
            llama_cortex_rebind_layer(model, il);
            char s_buf[16], e_buf[16];
            std::snprintf(s_buf, sizeof(s_buf), "%d", il);
            std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

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
        gen_tokens.push_back((llama_token)cur_token);
        cur_pos++;
    }

    // Compare tokens
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

    std::printf("[%s] Prefill Logits Diff: %d / %d (max diff: %e) | Tokens: %d / %d matched (%s)\n",
        tc.name.c_str(), logit_diffs, n_vocab, max_abs_diff, match_count, (int)ref_tokens.size(),
        tokens_match ? "PASS" : "FAIL");

    for (int il = 0; il < n_layer; ++il) if (layer_ctxs[il]) llama_free(layer_ctxs[il]);
    return tokens_match && (logit_diffs == 0);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <model.gguf> <ref_prefix>\n", argv[0]);
        return 1;
    }
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;

    std::vector<TestCase> cases = {
        { "q35_test_arithmetic_6", "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n", 6 },
        { "q35_test_reasoning_32", "<|im_start|>user\nSolve: If x + 5 = 12, what is x? Explain step by step.<|im_end|>\n<|im_start|>assistant\n", 32 },
        { "q35_test_code_64", "<|im_start|>user\nWrite a Python function to compute the factorial of n recursively.<|im_end|>\n<|im_start|>assistant\n", 64 }
    };

    bool all_pass = true;
    for (const auto & tc : cases) {
        if (!verify_test(model, tc, argv[2])) all_pass = false;
    }

    std::printf("\nOVERALL TEST SUITE RESULT: %s\n", all_pass ? "ALL PASSED" : "FAILED");
    llama_model_free(model);
    llama_backend_free();
    return all_pass ? 0 : 3;
}
