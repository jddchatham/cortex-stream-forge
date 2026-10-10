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

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "4", 1);
    setenv("CORTEX_QWEN35_EXTERNAL_LAYER", "3", 1);
    llama_context * a = make_ctx(model);
    if (!a) return 4;
    llama_set_embeddings_layer_inp(a, 3, true);
    llama_set_embeddings_layer_inp(a, 4, true);
    llama_batch ba = llama_batch_init(1, 0, 1);
    ba.token[0] = 9707; // Hello in the Qwen vocabulary
    ba.pos[0] = 0;
    ba.n_seq_id[0] = 1;
    ba.seq_id[0][0] = 0;
    ba.n_tokens = 1;
    if (llama_decode(a, ba) != 0) return 5;
    float * hidden = llama_get_embeddings_layer_inp(a, 3);
    if (!hidden) return 6;
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> h(hidden, hidden + n_embd);
    ba.token[0] = 1879;
    ba.pos[0] = 1;
    if (llama_decode(a, ba) != 0) return 5;
    hidden = llama_get_embeddings_layer_inp(a, 3);
    if (!hidden) return 6;
    h.insert(h.end(), hidden, hidden + n_embd);
    llama_batch_free(ba);

    setenv("CORTEX_QWEN35_SEGMENT_START", "3", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "4", 1);
    setenv("CORTEX_QWEN35_EXTERNAL_LAYER", "3", 1);
    llama_context * b = make_ctx(model);
    if (!b) return 7;
    llama_set_embeddings_layer_inp(b, 0, false);
    llama_set_embeddings_layer_inp(b, 1, false);
    llama_set_embeddings_layer_inp(b, 3, false);
    llama_set_embeddings_layer_inp(b, 4, true);
    llama_batch bb = llama_batch_init(1, n_embd, 1);
    for (int i = 0; i < n_embd; ++i) bb.embd[i] = h[i];
    bb.pos[0] = 0;
    bb.n_seq_id[0] = 1;
    bb.seq_id[0][0] = 0;
    bb.n_tokens = 1;
    if (llama_decode(b, bb) != 0) return 8;
    for (int i = 0; i < n_embd; ++i) bb.embd[i] = h[n_embd + i];
    bb.pos[0] = 1;
    if (llama_decode(b, bb) != 0) return 8;

    float * window_out = llama_get_embeddings_layer_inp(b, 4);
    float * oracle_out = llama_get_embeddings_layer_inp(a, 4);
    if (!window_out || !oracle_out) return 9;
    float max_err = 0.0f;
    for (int i = 0; i < n_embd; ++i) {
        max_err = std::max(max_err, std::fabs(window_out[i] - oracle_out[i]));
    }
    if (max_err > 1e-5f) {
        fprintf(stderr, "hidden-state mismatch: max_err=%g\n", max_err);
        return 10;
    }

    llama_free(a);
    llama_batch_free(bb);
    llama_free(b);
    llama_free_model(model);
    llama_backend_free();
    puts("q35 full-attention window handoff: PASS (max_err <= 1e-5)");
    return 0;
}
