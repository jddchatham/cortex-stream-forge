#include "flashnext-vocab-projector.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <array>

static bool read_exact(int fd, void * dst, size_t bytes, off_t off) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, (uint8_t *) dst + done, bytes - done, off + (off_t) done);
        if (n <= 0) return false;
        done += (size_t) n;
    }
    return true;
}

bool cortex_stream_vocab_project(const cortex_output_weight_descriptor & d,
                                 const std::vector<float> & hidden,
                                 size_t tile_rows,
                                 cortex_vocab_projection_result & result) {
    result = {};
    if (d.file_descriptor < 0 || d.ne[0] != (int64_t) hidden.size() || d.ne[1] <= 0 ||
        tile_rows == 0 || d.row_bytes == 0 || tile_rows > SIZE_MAX / d.row_bytes ||
        tile_rows * d.row_bytes > 16u * 1024u * 1024u) return false;
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return false;
    std::vector<uint8_t> scratch(tile_rows * d.row_bytes);
    result.scratch_bytes = scratch.size();
    result.logit = -INFINITY;
    for (int64_t row = 0; row < d.ne[1]; row += (int64_t) tile_rows) {
        int64_t nr = std::min<int64_t>((int64_t) tile_rows, d.ne[1] - row);
        size_t bytes = (size_t) nr * d.row_bytes;
        if (!read_exact(d.file_descriptor, scratch.data(), bytes,
                        (off_t) d.absolute_offset + (off_t) row * (off_t) d.row_bytes)) {
            ggml_backend_free(backend); return false;
        }
        result.bytes_read += bytes;
        ggml_context * ctx = ggml_init({ 64u * 1024u * 1024u, nullptr, true });
        if (!ctx) { ggml_backend_free(backend); return false; }
        ggml_tensor * w = ggml_new_tensor_2d(ctx, d.type, d.ne[0], nr);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ne[0], 1);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        bool ok = buffer != nullptr;
        if (ok) {
            ggml_backend_tensor_set(w, scratch.data(), 0, bytes);
            ggml_backend_tensor_set(x, hidden.data(), 0, hidden.size() * sizeof(float));
            ok = ok && ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(backend);
        }
        std::vector<float> logits((size_t) nr);
        if (ok) ggml_backend_tensor_get(y, logits.data(), 0, logits.size() * sizeof(float));
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        if (!ok) { ggml_backend_free(backend); return false; }
        for (int64_t i = 0; i < nr; ++i) if (logits[(size_t) i] > result.logit) {
            result.logit = logits[(size_t) i]; result.token = (int) (row + i);
        }
        for (int64_t i = 0; i < nr; ++i) {
            const int token = (int) (row + i); const float value = logits[(size_t) i];
            size_t at = result.top_count;
            if (at < 10) result.top_count++;
            else if (value <= result.top_logits[9]) continue;
            while (at > 0 && value > result.top_logits[at - 1]) {
                if (at < 10) { result.top_logits[at] = result.top_logits[at - 1]; result.top_tokens[at] = result.top_tokens[at - 1]; }
                --at;
            }
            if (at < 10) { result.top_logits[at] = value; result.top_tokens[at] = token; }
        }
    }
    ggml_backend_free(backend);
    return result.token >= 0;
}
