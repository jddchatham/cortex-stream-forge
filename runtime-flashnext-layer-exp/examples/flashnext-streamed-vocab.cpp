#include "llama-model.h"
#include "llama-context.h"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "flashnext-vocab-projector.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) return 2;
    const size_t tile = argc == 3 ? std::strtoull(argv[2], nullptr, 10) : 1024;
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_FLASHNEXT_EXTERNAL_EMBEDDING", "1", 1);
    setenv("CORTEX_FLASHNEXT_SEGMENT_B", "1", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD", "1", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY", "1", 1);
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "STREAMED_VOCAB model=FAIL\n"); return 3; }
    auto * base = reinterpret_cast<llama_model_base *>(model);
    cortex_output_weight_descriptor d;
    if (!base->get_cortex_output_weight_descriptor(d)) {
        std::fprintf(stderr, "STREAMED_VOCAB descriptor=FAIL\n"); llama_model_free(model); return 4;
    }
    const size_t hc = (size_t) base->hparams.dsv4_hc_mult;
    if (hc == 0 || d.ne[0] > SIZE_MAX / hc) { std::fprintf(stderr, "STREAMED_VOCAB final_head=invalid_hc\n"); llama_model_free(model); return 5; }
    std::vector<float> hidden((size_t) d.ne[0] * hc);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = 0.0001f * float(1 + (i % 31));
    cortex_vocab_projection_result projection;
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 1; cp.n_batch = 1;
    llama_context * context = llama_init_from_model(model, cp);
    cortex_post_moe_result normalized;
    bool ok = context != nullptr && reinterpret_cast<llama_context *>(context)->cortex_execute_final_head_only(hidden.data(), hidden.size(), normalized);
    std::vector<float> projected;
    if (ok) {
        std::fprintf(stderr, "CORTEX_FINAL_HEAD_CAPTURE input_shape=[%lld,%zu,1] input_bytes=%zu output_floats=%zu\n",
            (long long) d.ne[0], hc, hidden.size() * sizeof(float), normalized.hc_output.size());
        hidden = normalized.hc_output;
        ok = cortex_stream_vocab_project(d, hidden, tile, projection);
    }
    std::printf("STREAMED_VOCAB descriptor=%s shard=%u fd=%d offset=%llu type=%s dims=[%lld,%lld] row_bytes=%zu bytes=%zu tile_rows=%zu bytes_read=%zu scratch_bytes=%zu argmax=%d logit=%g status=%s\n",
        d.name.c_str(), d.shard, d.file_descriptor, (unsigned long long) d.absolute_offset,
        ggml_type_name(d.type), (long long) d.ne[0], (long long) d.ne[1], d.row_bytes, d.bytes,
        tile, projection.bytes_read, projection.scratch_bytes, projection.token, projection.logit, ok ? "PASS" : "FAIL");
    if (ok) for (size_t i = 0; i < projection.top_count; ++i) {
        const llama_vocab * vocab = llama_model_get_vocab(model); char piece[128];
        int n = llama_token_to_piece(vocab, projection.top_tokens[i], piece, sizeof(piece), 0, false);
        std::printf("STREAMED_VOCAB_TOP rank=%zu token=%d logit=%g eog=%d control=%d piece=%.*s\n",
            i + 1, projection.top_tokens[i], projection.top_logits[i],
            llama_vocab_is_eog(vocab, projection.top_tokens[i]),
            llama_vocab_is_control(vocab, projection.top_tokens[i]), std::max(0, n), n > 0 ? piece : "");
    }
    if (context) llama_free(context);
    llama_model_free(model);
    return ok ? 0 : 5;
}
