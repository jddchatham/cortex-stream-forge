#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

static uint64_t fnv1a(const void * p, size_t n) {
    const auto * b = static_cast<const unsigned char *>(p);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const char * path = argv[1];
    // The probe intentionally starts in the existing low-residency provider;
    // the selected initial segment is only a bootstrap allocation and every
    // subsequent layer, including block 64, is explicitly rebound.
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) return 3;
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt = {248045,846,198,12675,248046,198,248045,74455,198};
    std::vector<std::vector<uint8_t>> states(llama_model_n_layer(model));
    std::vector<float> hidden(n_embd);
    llama_pos pos = 0;
    llama_context * target_last = nullptr;
    std::vector<float> h_nextn;

    for (llama_token token : prompt) {
        for (int il = 0; il < llama_model_n_layer(model); ++il) {
            char start[32], end[32];
            std::snprintf(start, sizeof(start), "%d", il);
            std::snprintf(end, sizeof(end), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", start, 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", end, 1);
            if (!llama_cortex_rebind_layer(model, il)) return 10;
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = 256; cp.n_batch = 1;
            llama_context * ctx = llama_init_from_model(model, cp);
            if (!ctx) return 11;
            if (il + 1 < llama_model_n_layer(model)) {
                llama_set_embeddings_layer_inp(ctx, il + 1, true);
            }
            llama_set_embeddings_nextn(ctx, il == llama_model_n_layer(model) - 1, false);
            if (!states[il].empty() && llama_state_set_data(ctx, states[il].data(), states[il].size()) == 0) return 12;
            llama_batch b = llama_batch_init(1, il == 0 ? 0 : n_embd, 1);
            if (il == 0) b.token[0] = token; else std::copy(hidden.begin(), hidden.end(), b.embd);
            b.pos[0] = pos; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
            if (llama_decode(ctx, b) != 0) return 13;
            if (il + 1 < llama_model_n_layer(model)) {
                float * out = llama_get_embeddings_layer_inp(ctx, il + 1);
                if (!out) return 14;
                std::copy(out, out + n_embd, hidden.begin());
            } else {
                float * h = llama_get_embeddings_nextn_ith(ctx, 0);
                if (!h) return 15;
                h_nextn.assign(h, h + n_embd);
                target_last = ctx;
                ctx = nullptr;
            }
            if (ctx) {
                const size_t sz = llama_state_get_size(ctx);
                states[il].resize(sz);
                if (llama_state_get_data(ctx, states[il].data(), sz) != sz) return 16;
            }
            llama_batch_free(b); if (ctx) llama_free(ctx);
        }
        ++pos;
    }

    const size_t bytes = h_nextn.size() * sizeof(float);
    FILE * f = std::fopen("qwen38-mtp-hnextn.f32", "wb");
    if (!f || std::fwrite(h_nextn.data(), 1, bytes, f) != bytes) return 20;
    std::fclose(f);
    double sum = 0, norm = 0; float mn = h_nextn[0], mx = h_nextn[0];
    for (float x : h_nextn) { mn = std::min(mn,x); mx = std::max(mx,x); sum += x; norm += double(x)*x; }
    std::printf("HNEXTN shape=[%d] dtype=f32 bytes=%zu pos=%lld fnv=%016llx min=%g max=%g mean=%g norm=%g\n",
        n_embd, bytes, (long long)(pos-1), (unsigned long long)fnv1a(h_nextn.data(), bytes), mn, mx, sum/h_nextn.size(), std::sqrt(norm));

    // MTP receives the hidden row from the previous target position and the
    // token that occupies the next position.  The target context remains
    // alive as ctx_other for the separate MTP context.
    llama_context_params mcp = llama_context_default_params();
    mcp.ctx_type = LLAMA_CONTEXT_TYPE_MTP; mcp.ctx_other = target_last;
    mcp.n_ctx = 256; mcp.n_batch = 1;
    llama_context * mtp = llama_init_from_model(model, mcp);
    if (!mtp) return 30;
    llama_set_embeddings_nextn(mtp, true, true);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "64", 1);
    if (!llama_cortex_rebind_layer(model, 64)) return 31;
    llama_batch mb = llama_batch_init(1, n_embd, 1);
    mb.token[0] = 248068; std::copy(h_nextn.begin(), h_nextn.end(), mb.embd);
    mb.pos[0] = pos; mb.n_seq_id[0] = 1; mb.seq_id[0][0] = 0; mb.logits[0] = 1; mb.n_tokens = 1;
    if (llama_decode(mtp, mb) != 0) return 32;
    float * logits = llama_get_logits(mtp); if (!logits) return 33;
    std::vector<int> ids(n_vocab); std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin()+10, ids.end(), [logits](int a,int b){return logits[a]>logits[b];});
    std::printf("MTP_TOP"); for (int i=0;i<10;++i) std::printf(" %d=%.8g", ids[i], logits[ids[i]]); std::printf("\n");
    std::printf("MTP_LOGIT_FNV %016llx EOS246=%g\n", (unsigned long long)fnv1a(logits,(size_t)n_vocab*sizeof(float)), logits[248046]);
    llama_batch_free(mb); llama_free(mtp); llama_free(target_last); llama_model_free(model); llama_backend_free();
    return 0;
}
