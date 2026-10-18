#include "llama.h"
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

void run_stock_oracle(llama_model * model, const TestCase & tc, const std::string & prefix) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tok = -llama_tokenize(vocab, tc.prompt.c_str(), tc.prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_tok);
    llama_tokenize(vocab, tc.prompt.c_str(), tc.prompt.size(), prompt_tokens.data(), n_tok, true, true);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = std::max(n_tok, 64);
    cp.n_threads = 2;
    cp.n_threads_batch = 2;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "Failed to init stock context for %s\n", tc.name.c_str());
        return;
    }

    // Prefill prompt
    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tok - 1);
    }
    batch.n_tokens = n_tok;
    llama_decode(ctx, batch);
    llama_batch_free(batch);

    // Save prefill logits
    float * logits = llama_get_logits_ith(ctx, n_tok - 1);
    std::string logits_path = prefix + "_" + tc.name + "_prefill_logits.bin";
    std::ofstream f_log(logits_path, std::ios::binary);
    f_log.write(reinterpret_cast<const char*>(logits), n_vocab * sizeof(float));
    f_log.close();

    // Pick Step 1
    int cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > logits[cur_token]) cur_token = v;
    }

    std::vector<llama_token> gen_tokens = { (llama_token)cur_token };
    int cur_pos = n_tok;

    for (int step = 2; step <= tc.max_tokens; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) break;

        llama_batch step_batch = llama_batch_init(1, 0, 1);
        step_batch.token[0] = cur_token;
        step_batch.pos[0] = cur_pos;
        step_batch.n_seq_id[0] = 1;
        step_batch.seq_id[0][0] = 0;
        step_batch.logits[0] = true;
        step_batch.n_tokens = 1;

        llama_decode(ctx, step_batch);
        llama_batch_free(step_batch);

        float * step_logits = llama_get_logits_ith(ctx, 0);
        cur_token = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (step_logits[v] > step_logits[cur_token]) cur_token = v;
        }
        gen_tokens.push_back((llama_token)cur_token);
        cur_pos++;
    }

    // Save tokens
    std::string tok_path = prefix + "_" + tc.name + "_tokens.bin";
    std::ofstream f_tok(tok_path, std::ios::binary);
    int tok_count = (int)gen_tokens.size();
    f_tok.write(reinterpret_cast<const char*>(&tok_count), sizeof(int));
    f_tok.write(reinterpret_cast<const char*>(gen_tokens.data()), tok_count * sizeof(llama_token));
    f_tok.close();

    std::printf("[%s] Generated %d reference tokens. Saved to %s and %s\n",
        tc.name.c_str(), tok_count, logits_path.c_str(), tok_path.c_str());

    llama_free(ctx);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <model.gguf> <output_prefix>\n", argv[0]);
        return 1;
    }

    // Force stock mode (disable cortex streaming environment overrides)
    unsetenv("CORTEX_PREAD_ARENA");
    unsetenv("CORTEX_LOW_RESIDENCY_LAYER");
    unsetenv("CORTEX_DIRECT_MMAP");
    unsetenv("CORTEX_QWEN35_SEGMENT_START");
    unsetenv("CORTEX_QWEN35_SEGMENT_END");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "Failed to load model %s\n", argv[1]);
        return 2;
    }

    std::vector<TestCase> cases = {
        { "q35_test_arithmetic_6", "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n", 6 },
        { "q35_test_reasoning_32", "<|im_start|>user\nSolve: If x + 5 = 12, what is x? Explain step by step.<|im_end|>\n<|im_start|>assistant\n", 32 },
        { "q35_test_code_64", "<|im_start|>user\nWrite a Python function to compute the factorial of n recursively.<|im_end|>\n<|im_start|>assistant\n", 64 }
    };

    for (const auto & tc : cases) {
        run_stock_oracle(model, tc, argv[2]);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
