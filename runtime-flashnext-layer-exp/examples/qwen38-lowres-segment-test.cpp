#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    const char * model_path = argv[1];
    const int layer = std::atoi(argv[2]);
    const int end = std::atoi(argv[3]);
    char layer_text[32], end_text[32];
    std::snprintf(layer_text, sizeof(layer_text), "%d", layer);
    std::snprintf(end_text, sizeof(end_text), "%d", end);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", layer_text, 1);
    setenv("CORTEX_QWEN35_SEGMENT_START", layer_text, 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", end_text, 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 3;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8;
    cp.n_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    llama_set_embeddings_layer_inp(ctx, layer, false);
    llama_set_embeddings_layer_inp(ctx, end, true);

    const int n_embd = llama_model_n_embd(model);
    llama_batch batch;
    if (layer == 0) {
        batch = llama_batch_init(1, 0, 1);
        batch.token[0] = 9707;
    } else {
        batch = llama_batch_init(1, n_embd, 1);
        for (int i = 0; i < n_embd; ++i) batch.embd[i] = 0.001f * float((i % 17) - 8);
    }
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.n_tokens = 1;
    if (llama_decode(ctx, batch) != 0) return 5;
    float * output = llama_get_embeddings_layer_inp(ctx, end);
    if (!output) return 6;

    double checksum = 0.0;
    for (int i = 0; i < n_embd; ++i) checksum += output[i];
    std::printf("qwen38 lowres segment: PASS layer=%d end=%d hidden_checksum=%.9g\n", layer, end, checksum);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
