#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 256; cp.n_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return 4;
    llama_batch b = llama_batch_init(1, 0, 1);
    // Deterministic first formatted-chat token from the verified Qwen family.
    b.token[0] = 248045; b.pos[0] = 0; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
    if (llama_decode(ctx, b) != 0) return 5;
    const float * logits = llama_get_logits_ith(ctx, 0); if (!logits) return 6;
    constexpr int n_expert = 512, top_k = 10;
    std::vector<int> ids(n_expert); std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin()+top_k, ids.end(), [logits](int a, int b) { return logits[a] > logits[b]; });
    float max_logit = logits[0];
    for (int i = 1; i < n_expert; ++i) max_logit = std::max(max_logit, logits[i]);
    double sum_exp = 0.0;
    for (int i = 0; i < n_expert; ++i) sum_exp += std::exp((double) logits[i] - max_logit);
    uint64_t hash = 1469598103934665603ull;
    for (int i=0;i<n_expert;++i) { uint32_t u; std::memcpy(&u, logits+i, 4); hash ^= u; hash *= 1099511628211ull; }
    std::printf("FLASHNEXT_ROUTER layer=0 token=248045 experts=%d top_k=%d hash=%016llx\n", n_expert, top_k, (unsigned long long)hash);
    for (int i=0;i<top_k;++i) {
        const double weight = std::exp((double) logits[ids[i]] - max_logit) / sum_exp;
        std::printf("ROUTE %d id=%d logit=%.9g weight=%.12g\n", i, ids[i], logits[ids[i]], weight);
    }
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
}
