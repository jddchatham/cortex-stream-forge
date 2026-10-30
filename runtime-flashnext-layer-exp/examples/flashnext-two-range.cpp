#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUTPUT.bin\n", argv[0]);
        return 2;
    }
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_TWO_RANGE", "1", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { llama_backend_free(); return 3; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4;
    cp.n_batch = 1;
    cp.embeddings = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); llama_backend_free(); return 4; }

    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0] = 248045;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    batch.n_tokens = 1;
    const int rc = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    bool ok = rc == 0;
    const float * output = ok ? llama_get_embeddings_ith(ctx, 0) : nullptr;
    if (!output) ok = false;
    if (ok) {
        std::ofstream file(argv[2], std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char *>(output), 10240 * sizeof(float));
        ok = file.good();
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 5;
}
