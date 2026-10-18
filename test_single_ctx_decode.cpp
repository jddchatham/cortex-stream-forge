#include "llama.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <algorithm>

using bench_clock = std::chrono::high_resolution_clock;

struct MemoryStats {
    long vm_rss_kb = 0;
    long rss_anon_kb = 0;
    long rss_file_kb = 0;
};

static MemoryStats get_memory_stats() {
    MemoryStats stats;
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::sscanf(line.c_str() + 6, "%ld", &stats.vm_rss_kb);
        } else if (line.compare(0, 8, "RssAnon:") == 0) {
            std::sscanf(line.c_str() + 8, "%ld", &stats.rss_anon_kb);
        } else if (line.compare(0, 8, "RssFile:") == 0) {
            std::sscanf(line.c_str() + 8, "%ld", &stats.rss_file_kb);
        }
    }
    return stats;
}

static bool read_token_embeddings(int fd, uint64_t base_offset, const llama_token * tokens, int n_tokens, int n_embd, float * dst) {
    const size_t row_bytes = 1088;
    std::vector<uint8_t> row_buf(row_bytes);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);

    for (int i = 0; i < n_tokens; ++i) {
        uint64_t file_off = base_offset + (uint64_t)tokens[i] * row_bytes;
        size_t done = 0;
        while (done < row_bytes) {
            ssize_t n = pread(fd, row_buf.data() + done, row_bytes - done, (off_t)(file_off + done));
            if (n <= 0) return false;
            done += (size_t)n;
        }
        traits->to_float(row_buf.data(), dst + (size_t)i * n_embd, n_embd);
    }
    return true;
}

static bool compute_tiled_output_projection(int fd, uint64_t base_offset, int n_embd, int n_vocab,
                                            const float * norm_h, float * logits,
                                            uint8_t * tile_arena, size_t arena_bytes, ggml_backend_t backend) {
    const size_t row_bytes = 1088;
    const int tile_rows = arena_bytes / row_bytes;
    if (tile_rows < 1) return false;

    for (int start_row = 0; start_row < n_vocab; start_row += tile_rows) {
        int cur_rows = std::min(tile_rows, n_vocab - start_row);
        size_t bytes_to_read = (size_t)cur_rows * row_bytes;
        uint64_t file_off = base_offset + (uint64_t)start_row * row_bytes;

        size_t done = 0;
        while (done < bytes_to_read) {
            ssize_t n = pread(fd, tile_arena + done, bytes_to_read - done, (off_t)(file_off + done));
            if (n <= 0) return false;
            done += (size_t)n;
        }

        ggml_init_params ip { 16 * 1024 * 1024, nullptr, true };
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(gctx, GGML_TYPE_Q8_0, n_embd, cur_rows);
        ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_embd, 1);
        ggml_tensor * y = ggml_mul_mat(gctx, w, x);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, backend);
        w->data = tile_arena;
        ggml_backend_tensor_set(x, norm_h, 0, (size_t)n_embd * sizeof(float));
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_tensor_get(y, logits + start_row, 0, (size_t)cur_rows * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);
    }
    return true;
}

static void apply_rmsnorm(const float * x, const float * w, int n_embd, float * dst) {
    double sum2 = 0.0;
    for (int i = 0; i < n_embd; ++i) sum2 += (double)x[i] * x[i];
    float mean = (float)(sum2 / n_embd);
    float scale = 1.0f / std::sqrt(mean + 1e-6f);
    for (int i = 0; i < n_embd; ++i) dst[i] = x[i] * scale * w[i];
}

int main(int argc, char ** argv) {
    if (argc < 2) return 1;
    const char * model_path = argv[1];

    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_EXTERNAL_GLOBAL_TENSORS", "1", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 1;

    std::string prompt = "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n";
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_tok, true, true);

    const int n_embd = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * norm_w = (const float *) model->output_norm->data;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = n_tok;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;

    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);

    int fd = open(model_path, O_RDONLY);
    uint64_t tok_embd_off = 564424320;

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    std::vector<float> input_embeddings((size_t)n_embd * n_tok);
    std::vector<float> norm_h(n_embd);
    std::vector<float> last_logits(n_vocab);

    read_token_embeddings(fd, tok_embd_off, tokens.data(), n_tok, n_embd, input_embeddings.data());

    // Single-context prefill across all 24 layers
    for (int il = 0; il < n_layer; ++il) {
        char s_buf[16], e_buf[16];
        std::snprintf(s_buf, sizeof(s_buf), "%d", il);
        std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
        setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

        llama_set_embeddings_layer_inp(ctx, (il == 0) ? n_layer : il, false);
        llama_set_embeddings_layer_inp(ctx, il + 1, true);

        if (il > 0) {
            llama_memory_clear(mem, false);
        }

        llama_cortex_rebind_layer(model, il);

        llama_batch batch = llama_batch_init(n_tok, n_embd, 1);
        const float * src_embd = (il == 0) ? input_embeddings.data() : current_hidden.data();
        for (int i = 0; i < n_tok; ++i) {
            for (int d = 0; d < n_embd; ++d) {
                batch.embd[i * n_embd + d] = src_embd[i * n_embd + d];
            }
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false;
        }
        batch.n_tokens = n_tok;

        llama_decode(ctx, batch);
        float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
        std::copy(out_h, out_h + (size_t)n_embd * n_tok, current_hidden.begin());
        llama_batch_free(batch);
    }

    float * final_h = current_hidden.data() + (size_t)(n_tok - 1) * n_embd;
    apply_rmsnorm(final_h, norm_w, n_embd, norm_h.data());

    size_t arena_bytes = 16 * 1024 * 1024;
    std::vector<uint8_t> tile_arena(arena_bytes);

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    compute_tiled_output_projection(fd, tok_embd_off, n_embd, n_vocab, norm_h.data(), last_logits.data(), tile_arena.data(), arena_bytes, backend);

    int cur_token = 0;
    for (int v = 1; v < n_vocab; ++v) {
        if (last_logits[v] > last_logits[cur_token]) cur_token = v;
    }

    std::printf("First token = %d\n", cur_token);
    std::vector<llama_token> gen_tokens = { (llama_token)cur_token };
    int cur_pos = n_tok;

    std::vector<float> step_embd(n_embd);
    std::vector<float> step_hidden(n_embd);

    for (int step = 2; step <= 6; ++step) {
        read_token_embeddings(fd, tok_embd_off, (const llama_token*)&cur_token, 1, n_embd, step_embd.data());
        std::copy(step_embd.begin(), step_embd.end(), step_hidden.begin());

        for (int il = 0; il < n_layer; ++il) {
            char s_buf[16], e_buf[16];
            std::snprintf(s_buf, sizeof(s_buf), "%d", il);
            std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
            setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);

            llama_set_embeddings_layer_inp(ctx, (il == 0) ? n_layer : il, false);
            llama_set_embeddings_layer_inp(ctx, il + 1, true);

            llama_cortex_rebind_layer(model, il);

            llama_batch batch = llama_batch_init(1, n_embd, 1);
            for (int d = 0; d < n_embd; ++d) batch.embd[d] = step_hidden[d];
            batch.pos[0] = cur_pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = false;
            batch.n_tokens = 1;

            llama_decode(ctx, batch);
            float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
            std::copy(out_h, out_h + n_embd, step_hidden.begin());
            llama_batch_free(batch);
        }

        apply_rmsnorm(step_hidden.data(), norm_w, n_embd, norm_h.data());
        compute_tiled_output_projection(fd, tok_embd_off, n_embd, n_vocab, norm_h.data(), last_logits.data(), tile_arena.data(), arena_bytes, backend);

        cur_token = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (last_logits[v] > last_logits[cur_token]) cur_token = v;
        }
        gen_tokens.push_back((llama_token)cur_token);
        cur_pos++;
        std::printf("Step %d: token = %d\n", step, cur_token);
    }

    MemoryStats m = get_memory_stats();
    std::printf("Peak RSS: %.2f MiB | Anon: %.2f MiB | File: %.2f MiB\n",
        m.vm_rss_kb / 1024.0, m.rss_anon_kb / 1024.0, m.rss_file_kb / 1024.0);

    close(fd);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
