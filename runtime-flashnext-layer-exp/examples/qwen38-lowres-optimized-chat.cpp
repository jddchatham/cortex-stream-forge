#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static void one_pos(llama_batch & b, llama_pos pos) {
    b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.n_tokens = 1;
}

static llama_context * make_context(llama_model * model, int il) {
    char start[32], end[32];
    std::snprintf(start, sizeof(start), "%d", il);
    std::snprintf(end, sizeof(end), "%d", il + 1);
    setenv("CORTEX_QWEN35_SEGMENT_START", start, 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", end, 1);
    if (!llama_cortex_rebind_layer(model, il)) return nullptr;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2; cp.n_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx) llama_set_embeddings_layer_inp(ctx, il + 1, true);
    return ctx;
}

static llama_token run_token(llama_model * model, std::vector<llama_context *> & ctxs,
                             llama_token token, llama_pos pos, std::vector<float> & hidden) {
    const int n_layer = llama_model_n_layer(model), n_embd = llama_model_n_embd(model);
    for (int il = 0; il < n_layer; ++il) {
        llama_context * ctx = ctxs[il];
        if (!ctx) return -1;
        llama_batch batch;
        if (il == 0) {
            batch = llama_batch_init(1, 0, 1); batch.token[0] = token;
        } else {
            batch = llama_batch_init(1, n_embd, 1);
            std::copy(hidden.begin(), hidden.end(), batch.embd);
        }
        one_pos(batch, pos);
        if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); return -1; }
        if (il + 1 < n_layer) {
            float * out = llama_get_embeddings_layer_inp(ctx, il + 1);
            if (!out) { llama_batch_free(batch); return -1; }
            std::copy(out, out + n_embd, hidden.begin());
        } else {
            float * logits = llama_get_logits(ctx);
            if (!logits) { llama_batch_free(batch); return -1; }
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            token = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
        }
        llama_batch_free(batch);
    }
    return token;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) return 2;
    const int max_generate = argc == 3 ? std::atoi(argv[2]) : 8;
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    const int n_layer = llama_model_n_layer(model), n_embd = llama_model_n_embd(model);
    std::vector<llama_context *> ctxs(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        ctxs[il] = make_context(model, il);
        if (!ctxs[il]) return 4;
    }
    std::vector<float> hidden(n_embd);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string line; llama_pos pos = 0;
    std::cout << "Qwen3.8 Forge optimized low-residency chat. Type /exit to quit.\n> " << std::flush;
    while (std::getline(std::cin, line)) {
        if (line == "/exit") break;
        int n = -llama_tokenize(vocab, line.c_str(), (int32_t) line.size(), nullptr, 0, pos == 0, true);
        std::vector<llama_token> prompt(n);
        if (llama_tokenize(vocab, line.c_str(), (int32_t) line.size(), prompt.data(), n, pos == 0, true) < 0) return 5;
        llama_token next = -1;
        for (llama_token tok : prompt) {
            next = run_token(model, ctxs, tok, pos++, hidden);
            if (next < 0) return 6;
        }
        for (int i = 0; i < max_generate; ++i) {
            char piece[256]; int len = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, false);
            if (len > 0) std::cout.write(piece, len) << std::flush;
            next = run_token(model, ctxs, next, pos++, hidden);
            if (next < 0) return 7;
        }
        std::cout << "\n> " << std::flush;
    }
    for (llama_context * ctx : ctxs) llama_free(ctx);
    llama_model_free(model); llama_backend_free();
    return 0;
}
