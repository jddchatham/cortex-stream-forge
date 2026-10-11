#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static llama_context * make_ctx(llama_model * model) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx = 8;
    p.n_batch = 1;
    return llama_init_from_model(model, p);
}

static void set_token(llama_batch & b, llama_token token, llama_pos pos) {
    b.token[0] = token;
    b.pos[0] = pos;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.n_tokens = 1;
}

static void set_embd_pos(llama_batch & b, llama_pos pos) {
    b.pos[0] = pos;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.n_tokens = 1;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    const int n_embd = llama_model_n_embd(model);
    llama_batch b = llama_batch_init(1, 0, 1);

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "2", 1);
    setenv("CORTEX_QWEN35_EXTERNAL_LAYER", "1", 1);
    llama_context * oracle = make_ctx(model);
    if (!oracle) return 4;
    llama_set_embeddings_layer_inp(oracle, 1, true);
    llama_set_embeddings_layer_inp(oracle, 2, true);
    set_token(b, 9707, 0);
    if (llama_decode(oracle, b) != 0) return 5;
    float * h0 = llama_get_embeddings_layer_inp(oracle, 1);
    if (!h0) return 6;
    std::vector<float> hidden(h0, h0 + n_embd);
    set_token(b, 1879, 1);
    if (llama_decode(oracle, b) != 0) return 5;
    float * h1 = llama_get_embeddings_layer_inp(oracle, 1);
    if (!h1) return 6;
    hidden.insert(hidden.end(), h1, h1 + n_embd);

    setenv("CORTEX_QWEN35_SEGMENT_START", "1", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "2", 1);
    llama_context * window = make_ctx(model);
    if (!window) return 7;
    llama_set_embeddings_layer_inp(window, 0, false);
    llama_set_embeddings_layer_inp(window, 1, false);
    llama_set_embeddings_layer_inp(window, 2, true);
    llama_batch_free(b);
    b = llama_batch_init(1, n_embd, 1);
    std::copy(hidden.begin(), hidden.begin() + n_embd, b.embd);
    set_embd_pos(b, 0);
    if (llama_decode(window, b) != 0) return 8;
    std::copy(hidden.begin() + n_embd, hidden.end(), b.embd);
    set_embd_pos(b, 1);
    if (llama_decode(window, b) != 0) return 8;

    float * got = llama_get_embeddings_layer_inp(window, 2);
    float * ref = llama_get_embeddings_layer_inp(oracle, 2);
    if (!got || !ref) return 9;
    float max_err = 0.0f;
    for (int i = 0; i < n_embd; ++i) max_err = std::max(max_err, std::fabs(got[i] - ref[i]));
    llama_batch_free(b);
    llama_free(window);
    llama_free(oracle);
    llama_free_model(model);
    llama_backend_free();
    if (max_err > 1e-5f) {
        fprintf(stderr, "recurrent hidden-state mismatch: max_err=%g\n", max_err);
        return 10;
    }
    puts("q35 recurrent window handoff: PASS (max_err <= 1e-5)");
    return 0;
}
