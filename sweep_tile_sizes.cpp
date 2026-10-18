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
#include <omp.h>

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

static void apply_rmsnorm(const float * x, const float * w, int n_embd, float * dst) {
    double sum2 = 0.0;
    for (int i = 0; i < n_embd; ++i) sum2 += (double)x[i] * x[i];
    float mean = (float)(sum2 / n_embd);
    float scale = 1.0f / std::sqrt(mean + 1e-6f);
    for (int i = 0; i < n_embd; ++i) dst[i] = x[i] * scale * w[i];
}

struct SweepResult {
    int tile_mib = 0;
    int num_tiles = 0;
    double io_ms = 0.0;
    double compute_ms = 0.0;
    double total_ms = 0.0;
    double vm_rss_mib = 0.0;
    float max_err = 0.0f;
};

static SweepResult run_tile_sweep(int tile_mib, int fd, uint64_t tok_embd_off,
                                  int n_embd, int n_vocab, const float * norm_h,
                                  const std::vector<float> & ref_logits,
                                  const struct ggml_type_traits_cpu * cpu_traits) {
    SweepResult res;
    res.tile_mib = tile_mib;
    const size_t row_bytes = 1088;
    const size_t arena_bytes = (size_t)tile_mib * 1024 * 1024;
    const int tile_rows = arena_bytes / row_bytes;
    std::vector<uint8_t> tile_arena(arena_bytes);
    std::vector<float> logits(n_vocab);
    std::vector<uint8_t> q8_x(row_bytes);

    cpu_traits->from_float(norm_h, q8_x.data(), n_embd);

    const int n_runs = 5;
    double total_io = 0.0;
    double total_comp = 0.0;

    for (int run = 0; run < n_runs; ++run) {
        int tiles = 0;
        double r_io = 0.0;
        double r_comp = 0.0;

        for (int start_row = 0; start_row < n_vocab; start_row += tile_rows) {
            tiles++;
            int cur_rows = std::min(tile_rows, n_vocab - start_row);
            size_t bytes_to_read = (size_t)cur_rows * row_bytes;
            uint64_t file_off = tok_embd_off + (uint64_t)start_row * row_bytes;

            auto t0 = bench_clock::now();
            size_t done = 0;
            while (done < bytes_to_read) {
                ssize_t n = pread(fd, tile_arena.data() + done, bytes_to_read - done, (off_t)(file_off + done));
                if (n <= 0) break;
                done += (size_t)n;
            }
            auto t1 = bench_clock::now();
            r_io += std::chrono::duration<double, std::milli>(t1 - t0).count();

            auto t2 = bench_clock::now();
            #pragma omp parallel for num_threads(2) schedule(static)
            for (int r = 0; r < cur_rows; ++r) {
                const void * row_w = tile_arena.data() + (size_t)r * row_bytes;
                cpu_traits->vec_dot(n_embd, &logits[start_row + r], 0, row_w, 0, q8_x.data(), 0, 1);
            }
            auto t3 = bench_clock::now();
            r_comp += std::chrono::duration<double, std::milli>(t3 - t2).count();
        }
        res.num_tiles = tiles;
        total_io += r_io;
        total_comp += r_comp;
    }

    res.io_ms = total_io / n_runs;
    res.compute_ms = total_comp / n_runs;
    res.total_ms = res.io_ms + res.compute_ms;

    MemoryStats m = get_memory_stats();
    res.vm_rss_mib = m.vm_rss_kb / 1024.0;

    float max_err = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        float d = std::fabs(logits[i] - ref_logits[i]);
        if (d > max_err) max_err = d;
    }
    res.max_err = max_err;
    return res;
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

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * norm_w = (const float *) model->output_norm->data;

    int fd = open(model_path, O_RDONLY);
    uint64_t tok_embd_off = 564424320;

    const auto * cpu_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);

    const char * ref_path = "/home/user/bitnet-cortex/quant-refs/q8_0_q35_test_arithmetic_6_prefill_logits.bin";
    std::vector<float> ref_logits(n_vocab);
    FILE * rf = std::fopen(ref_path, "rb");
    std::fread(ref_logits.data(), sizeof(float), n_vocab, rf);
    std::fclose(rf);

    // Run single-context prefill to get real normalized hidden state
    std::string prompt = "<|im_start|>user\nWhat is 2 + 2? Answer with only the number.<|im_end|>\n<|im_start|>assistant\n";
    int n_tok = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n_tok, true, true);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = n_tok; cp.n_threads = 2; cp.n_threads_batch = 2;
    setenv("CORTEX_QWEN35_SEGMENT_START", "0", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);

    const size_t row_bytes = 1088;
    std::vector<uint8_t> row_buf(row_bytes);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<float> input_embeddings((size_t)n_embd * n_tok);
    for (int i = 0; i < n_tok; ++i) {
        pread(fd, row_buf.data(), row_bytes, (off_t)(tok_embd_off + (uint64_t)tokens[i] * row_bytes));
        traits->to_float(row_buf.data(), input_embeddings.data() + (size_t)i * n_embd, n_embd);
    }

    std::vector<float> current_hidden((size_t)n_embd * n_tok);
    const int n_layer = llama_model_n_layer(model);
    for (int il = 0; il < n_layer; ++il) {
        char s_buf[16], e_buf[16];
        std::snprintf(s_buf, sizeof(s_buf), "%d", il);
        std::snprintf(e_buf, sizeof(e_buf), "%d", il + 1);
        setenv("CORTEX_QWEN35_SEGMENT_START", s_buf, 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", e_buf, 1);
        llama_set_embeddings_layer_inp(ctx, (il == 0) ? n_layer : il, false);
        llama_set_embeddings_layer_inp(ctx, il + 1, true);
        if (il > 0) llama_memory_clear(mem, false);
        llama_cortex_rebind_layer(model, il);

        llama_batch batch = llama_batch_init(n_tok, n_embd, 1);
        const float * src = (il == 0) ? input_embeddings.data() : current_hidden.data();
        for (int i = 0; i < n_tok; ++i) {
            for (int d = 0; d < n_embd; ++d) batch.embd[i * n_embd + d] = src[i * n_embd + d];
            batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = false;
        }
        batch.n_tokens = n_tok;
        llama_decode(ctx, batch);
        float * out_h = llama_get_embeddings_layer_inp(ctx, il + 1);
        std::copy(out_h, out_h + (size_t)n_embd * n_tok, current_hidden.begin());
        llama_batch_free(batch);
    }

    float * final_h = current_hidden.data() + (size_t)(n_tok - 1) * n_embd;
    std::vector<float> norm_h(n_embd);
    apply_rmsnorm(final_h, norm_w, n_embd, norm_h.data());

    std::vector<int> sweep_sizes = { 4, 8, 16, 32 };
    std::printf("\n=========================================================================================\n");
    std::printf("PHASE 3: EXACT LM HEAD TILE SIZE SWEEP (DIRECT AVX2 SIMD)\n");
    std::printf("=========================================================================================\n");
    std::printf("%-10s %-8s %-12s %-12s %-12s %-12s %-10s\n",
        "Tile Size", "Tiles", "Pread I/O", "Compute", "Total LM-Head", "Peak VmRSS", "Max Error");
    std::printf("-----------------------------------------------------------------------------------------\n");

    for (int sz : sweep_sizes) {
        SweepResult sr = run_tile_sweep(sz, fd, tok_embd_off, n_embd, n_vocab, norm_h.data(), ref_logits, cpu_traits);
        std::printf("%2d MiB      %-8d %8.2f ms   %8.2f ms   %8.2f ms   %8.2f MiB   %.6f\n",
            sr.tile_mib, sr.num_tiles, sr.io_ms, sr.compute_ms, sr.total_ms, sr.vm_rss_mib, sr.max_err);
    }
    std::printf("=========================================================================================\n\n");

    close(fd);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
