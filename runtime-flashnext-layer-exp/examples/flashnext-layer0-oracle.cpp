#include "llama.h"
#include "../src/llama-context.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool write_f32(const char * path, const float * data, size_t count) {
    FILE * f = std::fopen(path, "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data, sizeof(float), count, f) == count;
    std::fclose(f);
    return ok;
}

int main(int argc, char ** argv) {
    const bool sequential = std::strcmp(argv[1], "sequential") == 0 || std::strcmp(argv[1], "sequential-staged") == 0;
    const bool sequential_staged = std::strcmp(argv[1], "sequential-staged") == 0;
    if ((argc != 5 && argc != 7) || (!sequential && std::strcmp(argv[1], "conventional") != 0 && std::strcmp(argv[1], "staged") != 0)) {
        std::fprintf(stderr, "usage: %s conventional|staged MODEL BOUNDARY.f32 ROUTER.f32 [TOKEN POSITION]\n", argv[0]);
        std::fprintf(stderr, "       %s sequential MODEL BOUNDARY1.f32 ROUTER1.f32 BOUNDARY2.f32 ROUTER2.f32\n", argv[0]);
        return 2;
    }
    const bool conventional = !sequential_staged && (sequential || std::strcmp(argv[1], "conventional") == 0);
    unsetenv("CORTEX_FLASHNEXT_SEGMENT_A");
    unsetenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER");
    unsetenv("CORTEX_FLASHNEXT_STAGED_MOE");
    unsetenv("CORTEX_FLASHNEXT_PRE_MOE");
    unsetenv("CORTEX_FLASHNEXT_CONVENTIONAL_STAGE_A_CAPTURE");
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_TWO_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER0_ORACLE", "1", 1);
    setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
    if (conventional) {
        setenv("CORTEX_FLASHNEXT_CONVENTIONAL_STAGE_A_CAPTURE", "1", 1);
        setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
    } else {
        setenv("CORTEX_FLASHNEXT_SEGMENT_A", "1", 1);
        setenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER", "1", 1);
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) return 3;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4;
    cp.n_batch = 1;
    // Capture through the internal named-tensor API below; public embedding
    // extraction assumes the model's normal output width.
    cp.embeddings = false;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;

    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0] = 248045;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    // The pre-MoE reference exposes a 512-wide router tensor, not a
    // vocabulary-sized logits tensor; do not ask the public logits extractor
    // to read it as final vocabulary output.
    batch.logits[0] = 0;
    batch.n_tokens = 1;
    llama_context * internal = ctx;
    const char * boundary_name = conventional
        ? "flashnext_conventional_stage_a_boundary-0"
        : "flashnext_segment_a_boundary-0";
    const char * router_name = conventional
        ? "flashnext_conventional_stage_a_router-0"
        : "diag_router_logits-0";
    auto run_capture = [&](int token, int position, const char * boundary_path, const char * router_path) -> bool {
        batch.token[0] = token;
        batch.pos[0] = position;
        const int status = llama_decode(ctx, batch);
        if (status != 0) return false;
        llama_synchronize(ctx);
        std::vector<float> boundary(12804), router(512);
        if (internal->debug_copy_flashnext_tensor(boundary_name, boundary.data(), boundary.size() * sizeof(float)) != boundary.size() * sizeof(float) ||
            internal->debug_copy_flashnext_tensor(router_name, router.data(), router.size() * sizeof(float)) != router.size() * sizeof(float)) return false;
        return write_f32(boundary_path, boundary.data(), boundary.size()) && write_f32(router_path, router.data(), router.size());
    };
    const bool ok = sequential
        ? run_capture(248045, 0, argv[3], argv[4]) && run_capture(248046, 1, argv[5], argv[6])
        : run_capture(argc == 7 ? std::atoi(argv[5]) : 248045, argc == 7 ? std::atoi(argv[6]) : 0, argv[3], argv[4]);
    if (!ok) return 5;
    std::printf("FLASHNEXT_ORACLE mode=%s boundary_elements=12804 router_elements=512\n", argv[1]);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
