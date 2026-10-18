#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::printf("Usage: %s <model_path> <out_prefix> <max_tokens>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const std::string out_prefix = argv[2];
    int max_tokens = std::atoi(argv[3]);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string prompt = "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n";
    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_tok, true, true);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = std::max(n_tok, 64);
    cp.n_threads = 2;
    cp.n_threads_batch = 2;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    llama_batch batch = llama_batch_init(std::max(n_tok, 64), 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tok - 1);
    }
    batch.n_tokens = n_tok;

    llama_decode(ctx, batch);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(ctx, n_tok - 1);

    // Save prefill logits
    std::string log_path = out_prefix + "_prefill_logits.bin";
    std::ofstream f_log(log_path, std::ios::binary);
    f_log.write(reinterpret_cast<const char*>(logits), n_vocab * sizeof(float));
    f_log.close();

    // Greedy pick Step 1
    llama_token cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > logits[cur_token]) cur_token = v;
    }

    std::vector<llama_token> gen_tokens = { cur_token };
    int cur_pos = n_tok;

    for (int step = 2; step <= max_tokens; ++step) {
        if (llama_vocab_is_eog(vocab, cur_token)) break;

        batch.n_tokens = 1;
        batch.token[0] = cur_token;
        batch.pos[0] = cur_pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        llama_decode(ctx, batch);
        const float * step_logits = llama_get_logits_ith(ctx, 0);

        cur_token = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (step_logits[v] > step_logits[cur_token]) cur_token = v;
        }
        gen_tokens.push_back(cur_token);
        cur_pos++;
    }

    std::string tok_path = out_prefix + "_tokens.bin";
    std::ofstream f_tok(tok_path, std::ios::binary);
    int n_gen = (int)gen_tokens.size();
    f_tok.write(reinterpret_cast<const char*>(&n_gen), sizeof(int));
    f_tok.write(reinterpret_cast<const char*>(gen_tokens.data()), n_gen * sizeof(llama_token));
    f_tok.close();

    std::printf("[%s] Generated %d tokens: ", out_prefix.c_str(), n_gen);
    for (int t : gen_tokens) {
        char buf[128];
        int l = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        std::printf("%.*s", l, buf);
    }
    std::printf("\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
