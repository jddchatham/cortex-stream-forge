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

static llama_batch one_token(llama_token token, llama_pos pos) {
    llama_batch b = llama_batch_init(1, 0, 1);
    b.token[0] = token;
    b.pos[0] = pos;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.n_tokens = 1;
    return b;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    const int n_embd = llama_model_n_embd(model);

    setenv("CORTEX_STREAM_SEGMENT_START", "0", 1);
    setenv("CORTEX_STREAM_SEGMENT_END", "2", 1);
    setenv("CORTEX_STREAM_EXTERNAL_LAYER", "1", 1);
    llama_context * oracle = make_ctx(model);
    if (!oracle) return 4;
    llama_set_embeddings_layer_inp(oracle, 1, true);
    llama_set_embeddings_layer_inp(oracle, 2, true);
    llama_batch bo = one_token(1, 0);
    if (llama_decode(oracle, bo) != 0) return 5;
    float * oracle_hidden = llama_get_embeddings_layer_inp(oracle, 1);
    if (!oracle_hidden) return 6;
    std::vector<float> hidden(oracle_hidden, oracle_hidden + n_embd);
    llama_batch_free(bo);

    setenv("CORTEX_STREAM_SEGMENT_START", "1", 1);
    setenv("CORTEX_STREAM_SEGMENT_END", "2", 1);
    setenv("CORTEX_STREAM_EXTERNAL_LAYER", "1", 1);
    llama_context * candidate = make_ctx(model);
    if (!candidate) return 7;
    llama_set_embeddings_layer_inp(candidate, 0, false);
    llama_set_embeddings_layer_inp(candidate, 1, false);
    llama_set_embeddings_layer_inp(candidate, 2, true);
    llama_batch bc = llama_batch_init(1, n_embd, 1);
    for (int i = 0; i < n_embd; ++i) bc.embd[i] = hidden[i];
    bc.pos[0] = 0;
    bc.n_seq_id[0] = 1;
    bc.seq_id[0][0] = 0;
    bc.n_tokens = 1;
    if (llama_decode(candidate, bc) != 0) return 8;

    float * a = llama_get_embeddings_layer_inp(oracle, 2);
    float * b = llama_get_embeddings_layer_inp(candidate, 2);
    if (!a || !b) return 9;
    float max_err = 0.0f;
    for (int i = 0; i < n_embd; ++i) max_err = std::max(max_err, std::fabs(a[i] - b[i]));

    llama_free(candidate);
    llama_batch_free(bc);
    llama_free(oracle);
    llama_free_model(model);
    llama_backend_free();
    if (max_err > 1e-5f) {
        fprintf(stderr, "deepseek hidden-state mismatch: max_err=%g\n", max_err);
        return 10;
    }
    printf("deepseek llama window handoff: PASS (max_err <= 1e-5)\n");
    return 0;
}
