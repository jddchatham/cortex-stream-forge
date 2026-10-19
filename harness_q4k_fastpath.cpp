#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <omp.h>

#include "llama.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

using bench_clock = std::chrono::high_resolution_clock;

extern "C" {
    void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
}

int main(int argc, char ** argv) {
    llama_backend_init();

    std::cout << "==========================================================" << std::endl;
    std::cout << "CORTEX Q4_K FAST PATH BITWISE REFERENCE HARNESS" << std::endl;
    std::cout << "==========================================================" << std::endl;

    const char * model_path = "/home/user/.ollama/models/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c";
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open model: " << model_path << std::endl;
        return 1;
    }

    const int n_embd = 4096;
    const size_t row_bytes = 2304;
    const int num_test_rows = 128;
    const uint64_t tensor_abs_off = 81602720; // blk.0.ffn_gate.weight

    std::vector<uint8_t> captured_q4k_rows(num_test_rows * row_bytes);
    ssize_t n_read = pread(fd, captured_q4k_rows.data(), captured_q4k_rows.size(), (off_t)tensor_abs_off);
    close(fd);
    if (n_read != (ssize_t)captured_q4k_rows.size()) {
        std::cerr << "Failed to read weight rows, read=" << n_read << std::endl;
        return 1;
    }

    // Realistic activation vector
    std::vector<float> act(n_embd);
    for (int i = 0; i < n_embd; ++i) {
        act[i] = std::sin((float)i * 0.05f) * 1.5f + std::cos((float)i * 0.01f) * 0.5f;
    }
    const auto * traits_vec = ggml_get_type_traits(GGML_TYPE_Q8_K);
    const auto * cpu_traits_vec = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    size_t q8k_bytes = (n_embd / traits_vec->blck_size) * traits_vec->type_size;
    std::vector<uint8_t> q8k_act(q8k_bytes);
    cpu_traits_vec->from_float(act.data(), q8k_act.data(), n_embd);

    // Compute reference dot products
    std::vector<float> ref_results(num_test_rows);
    for (int r = 0; r < num_test_rows; ++r) {
        const void * vx = captured_q4k_rows.data() + r * row_bytes;
        ggml_vec_dot_q4_K_q8_K(n_embd, &ref_results[r], 0, vx, 0, q8k_act.data(), 0, 1);
    }

    std::cout << "Successfully captured " << num_test_rows << " real Q4_K weight rows (4096 cols)." << std::endl;
    std::cout << "Sample reference dot products:" << std::endl;
    for (int r = 0; r < 8; ++r) {
        uint32_t bits;
        memcpy(&bits, &ref_results[r], sizeof(bits));
        std::printf("  Row %2d: val=%+14.6f (hex: 0x%08X)\n", r, ref_results[r], bits);
    }

    // Benchmark baseline throughput on 1 thread
    const int N_WARM = 200;
    const int N_BENCH = 2000;
    
    // Warmup
    for (int it = 0; it < N_WARM; ++it) {
        for (int r = 0; r < num_test_rows; ++r) {
            float dummy = 0.0f;
            const void * vx = captured_q4k_rows.data() + r * row_bytes;
            ggml_vec_dot_q4_K_q8_K(n_embd, &dummy, 0, vx, 0, q8k_act.data(), 0, 1);
        }
    }

    auto t0 = bench_clock::now();
    for (int it = 0; it < N_BENCH; ++it) {
        for (int r = 0; r < num_test_rows; ++r) {
            float dummy = 0.0f;
            const void * vx = captured_q4k_rows.data() + r * row_bytes;
            ggml_vec_dot_q4_K_q8_K(n_embd, &dummy, 0, vx, 0, q8k_act.data(), 0, 1);
        }
    }
    auto t1 = bench_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double us_per_dot = total_us / (N_BENCH * num_test_rows);
    std::cout << "\nBaseline single-core kernel latency: " << us_per_dot << " us / dot (4096 elements)" << std::endl;
    std::cout << "Throughput: " << (4096.0 / (us_per_dot * 1e-6) / 1e9) << " Gelem/s" << std::endl;

    llama_backend_free();
    return 0;
}
