// Stock two-layer reference runner (diagnostic).
//
// Runs the unmodified stock FlashNext graph for a single token decode:
// normal model load (no low-residency probe, no pread arena, no external
// global tensors, no segment cuts), in-graph token embedding lookup, and the
// integrity-protected diagnostic captures. Used as the stock reference for
// streamed two-layer equivalence checks.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }

    static const char * cortex_envs[] = {
        "CORTEX_LOW_RESIDENCY_LAYER",
        "CORTEX_EXTERNAL_GLOBAL_TENSORS",
        "CORTEX_FLASHNEXT_ROUTER_ONLY",
        "CORTEX_FLASHNEXT_SEGMENT_A",
        "CORTEX_FLASHNEXT_SEGMENT_A_ROUTER",
        "CORTEX_FLASHNEXT_PRE_MOE",
        "CORTEX_FLASHNEXT_LAYER_START",
        "CORTEX_FLASHNEXT_LAYER_END",
        "CORTEX_FLASHNEXT_LAYER_RANGE",
        "CORTEX_QWEN35_SEGMENT_END",
        "CORTEX_FLASHNEXT_FINAL_HEAD",
        "CORTEX_FLASHNEXT_SEGMENT_B",
        "CORTEX_PREAD_ARENA",
        "CORTEX_DIRECT_MMAP",
        "CORTEX_FLASHNEXT_CHAIN_1_2",
        "CORTEX_FLASHNEXT_CHAIN_1_2_3",
        "CORTEX_FLASHNEXT_CHAIN_0_48",
        "CORTEX_FLASHNEXT_CHAIN_0_N",
        "CORTEX_FLASHNEXT_INJECT_PRE_COMMIT_FAILURE",
    };
    for (const char * name : cortex_envs) {
        unsetenv(name);
    }
    setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "STOCK_REF model load failed\n");
        llama_backend_free();
        return 3;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx  = 256;
    cp.n_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "STOCK_REF context init failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0]    = 248045;
    batch.pos[0]      = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]   = 1;
    batch.n_tokens    = 1;

    const int status = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    std::printf("STOCK_REF_DECODE status=%d\n", status);

    // stock logits reference for the 48-layer chain: full-graph output
    // projection of the final HC-mixed activation, token 248045 pos 0.
    const float * logits = llama_get_logits_ith(ctx, 0);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (logits && n_vocab > 0) {
        const char * out = std::getenv("STOCK_LOGITS_OUT");
        FILE * f = std::fopen(out ? out : "stock_logits_p0.bin", "wb");
        if (f) { std::fwrite(logits, sizeof(float), (size_t) n_vocab, f); std::fclose(f); }
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
        std::printf("STOCK_REF_LOGITS n=%d argmax=%d logit=%.9g\n", n_vocab, best, logits[best]);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return status == 0 ? 0 : 5;
}
