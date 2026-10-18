#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 2) return 1;
    const char * model_path = argv[1];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::printf("Failed to load model %s\n", model_path);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string prompt = "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n";
    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_tok, true, true);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = n_tok;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::printf("Failed to init context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tok - 1);
    }
    batch.n_tokens = n_tok;

    auto t0 = std::chrono::high_resolution_clock::now();
    if (llama_decode(ctx, batch) != 0) {
        std::printf("Prefill decode failed\n");
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, n_tok - 1);

    std::printf("Prefill successful! Time: %.2f ms\n", prefill_ms);
    std::printf("Generating 6 tokens greedy:\n");

    llama_token cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > logits[cur_token]) cur_token = v;
    }

    char buf[128];
    int l = llama_token_to_piece(vocab, cur_token, buf, sizeof(buf), 0, true);
    std::printf("Token 1: %d ('%.*s')\n", cur_token, l, buf);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
