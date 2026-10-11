#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

static void set_pos(llama_batch & b, llama_pos pos) {
    b.pos[0] = pos;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.n_tokens = 1;
}
using forge_clock = std::chrono::steady_clock;
static double elapsed_ms(forge_clock::time_point a, forge_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) return 2;
    const char * path = argv[1];
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) return 3;
    const int n_layer = std::min(llama_model_n_layer(model), argc == 3 ? std::atoi(argv[2]) : llama_model_n_layer(model));
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> hidden(n_embd);
    std::vector<std::vector<uint8_t>> states(n_layer);
    llama_token next = 9707; // "Hello" in the Qwen vocabulary.
    for (int step = 0; step < 2; ++step) {
        const auto step_begin = forge_clock::now();
        double rebind_ms = 0, context_ms = 0, restore_ms = 0, decode_ms = 0, save_ms = 0;
        for (int il = 0; il < n_layer; ++il) {
            char start[32], end[32];
            std::snprintf(start, sizeof(start), "%d", il);
            std::snprintf(end, sizeof(end), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", start, 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", end, 1);
            auto t = forge_clock::now();
            if (!llama_cortex_rebind_layer(model, il)) return 6;
            auto u = forge_clock::now(); rebind_ms += elapsed_ms(t, u);
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = 2;
            cp.n_batch = 1;
            const int threads = std::getenv("CORTEX_THREADS") ? std::atoi(std::getenv("CORTEX_THREADS")) : 0;
            if (threads > 0) { cp.n_threads = threads; cp.n_threads_batch = threads; }
            t = forge_clock::now(); llama_context * ctx = llama_init_from_model(model, cp); u = forge_clock::now();
            context_ms += elapsed_ms(t, u);
            if (!ctx) return 7;
            llama_set_embeddings_layer_inp(ctx, il + 1, true);
            if (!states[il].empty()) { t = forge_clock::now(); if (llama_state_set_data(ctx, states[il].data(), states[il].size()) == 0) return 8; u = forge_clock::now(); restore_ms += elapsed_ms(t, u); }
            llama_batch batch;
            if (il == 0) {
                batch = llama_batch_init(1, 0, 1);
                batch.token[0] = next;
            } else {
                batch = llama_batch_init(1, n_embd, 1);
                std::copy(hidden.begin(), hidden.end(), batch.embd);
            }
            set_pos(batch, step);
            t = forge_clock::now(); if (llama_decode(ctx, batch) != 0) return 9; u = forge_clock::now(); decode_ms += elapsed_ms(t, u);
            if (il + 1 < n_layer) {
                float * out = llama_get_embeddings_layer_inp(ctx, il + 1);
                if (!out) return 10;
                std::copy(out, out + n_embd, hidden.begin());
            } else {
                float * logits = llama_get_logits(ctx);
                if (!logits) return 11;
                const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
                next = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
            }
            llama_batch_free(batch);
            const size_t state_size = llama_state_get_size(ctx);
            states[il].resize(state_size);
            t = forge_clock::now(); if (llama_state_get_data(ctx, states[il].data(), states[il].size()) != state_size) return 12; u = forge_clock::now(); save_ms += elapsed_ms(t, u);
            llama_free(ctx);
        }
        std::printf("qwen38 direct-mmap step=%d token=%d total_ms=%.1f rebind_ms=%.1f context_ms=%.1f restore_ms=%.1f decode_ms=%.1f save_ms=%.1f\n", step, next, elapsed_ms(step_begin, forge_clock::now()), rebind_ms, context_ms, restore_ms, decode_ms, save_ms);
    }

    llama_model_free(model);
    llama_backend_free();
    puts("qwen38 lowres sequential: PASS");
    return 0;
}
