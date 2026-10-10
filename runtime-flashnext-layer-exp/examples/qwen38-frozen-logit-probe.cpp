#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const char * model_path = argv[1];
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 3;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_chat_message> messages = {{"user", "Hi"}};
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<char> formatted(8192);
    int ntext = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    if (ntext < 0) return 4;
    formatted.resize((size_t) ntext + 1);
    ntext = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
    int ntok = -llama_tokenize(vocab, formatted.data(), ntext, nullptr, 0, true, true);
    std::vector<llama_token> prompt(ntok);
    if (llama_tokenize(vocab, formatted.data(), ntext, prompt.data(), ntok, true, true) < 0) return 5;
    std::printf("REFERENCE_PROMPT_TEXT_BYTES %d\nREFERENCE_PROMPT_TOKENS %d:", ntext, ntok);
    for (llama_token t : prompt) std::printf(" %d", t);
    std::printf("\n");
    std::vector<std::vector<uint8_t>> states(llama_model_n_layer(model));
    const int n_embd = llama_model_n_embd(model), n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<float> hidden(n_embd), final_logits(n_vocab);
    llama_pos pos = 0;
    for (llama_token token : prompt) {
        for (int il = 0; il < llama_model_n_layer(model); ++il) {
            char start[32], end[32]; std::snprintf(start, sizeof(start), "%d", il); std::snprintf(end, sizeof(end), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", start, 1); setenv("CORTEX_QWEN35_SEGMENT_END", end, 1);
            if (!llama_cortex_rebind_layer(model, il)) return 6;
            llama_context_params cp = llama_context_default_params(); cp.n_ctx = 256; cp.n_batch = 1;
            llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return 7;
            llama_set_embeddings_layer_inp(ctx, il + 1, true);
            if (!states[il].empty() && llama_state_set_data(ctx, states[il].data(), states[il].size()) == 0) return 8;
            llama_batch b = llama_batch_init(1, il == 0 ? 0 : n_embd, 1);
            if (il == 0) b.token[0] = token; else std::copy(hidden.begin(), hidden.end(), b.embd);
            b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
            if (llama_decode(ctx, b) != 0) return 9;
            if (il + 1 < llama_model_n_layer(model)) {
                float * out = llama_get_embeddings_layer_inp(ctx, il + 1); if (!out) return 10;
                std::copy(out, out + n_embd, hidden.begin());
            } else {
                float * logits = llama_get_logits(ctx); if (!logits) return 11;
                std::copy(logits, logits + n_vocab, final_logits.begin());
            }
            const size_t sz = llama_state_get_size(ctx); states[il].resize(sz);
            if (llama_state_get_data(ctx, states[il].data(), sz) != sz) return 12;
            llama_batch_free(b); llama_free(ctx);
        }
        ++pos;
    }
    std::vector<int> ids(n_vocab); std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + 8, ids.end(), [&final_logits](int a, int b) { return final_logits[a] > final_logits[b]; });
    std::printf("REFERENCE_FINAL_POS %lld VOCAB %d TOP:", (long long) (pos - 1), n_vocab);
    for (int i = 0; i < 8; ++i) std::printf(" %d=%.8g", ids[i], final_logits[ids[i]]);
    std::printf("\nREFERENCE_ARGMAX %d EOS246=%g EOS244=%g\n", ids[0], final_logits[248046], final_logits[248044]);
    llama_model_free(model); llama_backend_free();
    return 0;
}
