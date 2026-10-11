#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

static bool read_exact(int fd, void * dst, size_t n, off_t off) {
    size_t done = 0;
    while (done < n) {
        ssize_t got = pread(fd, (uint8_t *) dst + done, n - done, off + (off_t) done);
        if (got <= 0) { std::fprintf(stderr, "pread: %s\n", std::strerror(errno)); return false; }
        done += (size_t) got;
    }
    return true;
}

static float max_abs(const float * a, const float * b, size_t n, float * rel) {
    float ma = 0.0f, mr = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        ma = std::max(ma, std::fabs(a[i] - b[i]));
        mr = std::max(mr, std::fabs(a[i] - b[i]) / std::max(1e-6f, std::fabs(a[i])));
    }
    *rel = mr;
    return ma;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) return 2;
    const char * path = argv[1];
    const size_t arena_bytes = argc == 3 ? std::strtoull(argv[2], nullptr, 0) : 8*1024*1024;
    const char * wanted = "blk.0.ffn_down.weight";

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { std::perror("open"); return 3; }
    ggml_context * meta_ctx = nullptr;
    gguf_init_params ip = { true, &meta_ctx };
    gguf_context * gguf = gguf_init_from_file(path, ip);
    if (!gguf) return 4;
    int64_t tid = gguf_find_tensor(gguf, wanted);
    if (tid < 0) { std::fprintf(stderr, "tensor not found: %s\n", wanted); return 5; }
    const ggml_type type = gguf_get_tensor_type(gguf, tid);
    const int64_t * ne = gguf_get_tensor_ne(gguf, tid);
    const int64_t K = ne[0], N = ne[1];
    const size_t row_bytes = ggml_row_size(type, K);
    const size_t tensor_bytes = gguf_get_tensor_size(gguf, tid);
    const off_t offset = (off_t) gguf_get_data_offset(gguf) + (off_t) gguf_get_tensor_offset(gguf, tid);
    if (tensor_bytes != row_bytes * (size_t) N || arena_bytes < row_bytes) {
        std::fprintf(stderr, "invalid geometry or arena: tensor=%zu row=%zu arena=%zu\n", tensor_bytes, row_bytes, arena_bytes);
        return 6;
    }
    void * arena = nullptr;
    if (posix_memalign(&arena, 64, arena_bytes) != 0) return 7;
    std::vector<uint8_t> full(tensor_bytes);
    if (!read_exact(fd, full.data(), full.size(), offset)) return 8;

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) return 9;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto set_threads = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) set_threads(backend, 2);

    std::vector<float> input((size_t) K, 0.0f);
    for (int64_t i = 0; i < K; ++i) input[(size_t) i] = 0.001f * float((i % 17) - 8);
    std::vector<float> oracle((size_t) N), tiled((size_t) N);

    auto run = [&](const void * weights, int64_t cols, float * out) -> bool {
        ggml_init_params p = { 64*1024*1024, nullptr, true };
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, type, K, cols);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        w->data = const_cast<void *>(weights);
        ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
        const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        if (ok) ggml_backend_tensor_get(y, out, 0, (size_t) cols*sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return ok;
    };

    if (!run(full.data(), N, oracle.data())) return 10;
    const int64_t rows_per_tile = std::max<int64_t>(1, (int64_t) (arena_bytes / row_bytes));
    const size_t tile_bytes = (size_t) rows_per_tile * row_bytes;
    std::printf("TENSOR_WINDOW tensor=%s type=%s dims=[%lld,%lld] offset=%lld bytes=%zu arena=%zu row_bytes=%zu tile_rows=%lld tile_bytes=%zu\n",
        wanted, ggml_type_name(type), (long long) K, (long long) N, (long long) offset,
        tensor_bytes, arena_bytes, row_bytes, (long long) rows_per_tile, tile_bytes);
    std::printf("TENSOR_WINDOW arena=%p\n", arena);
    for (int64_t row = 0, tile = 0; row < N; row += rows_per_tile, ++tile) {
        const int64_t cols = std::min<int64_t>(rows_per_tile, N - row);
        const size_t bytes = (size_t) cols * row_bytes;
        if (!read_exact(fd, arena, bytes, offset + (off_t) row * (off_t) row_bytes)) return 11;
        if (!run(arena, cols, tiled.data() + row)) return 12;
        std::printf("TENSOR_WINDOW tile=%lld rows=[%lld,%lld) bytes=%zu sync=after_compute arena=%p\n",
            (long long) tile, (long long) row, (long long) (row + cols), bytes, arena);
    }
    float rel = 0.0f;
    float abs = max_abs(oracle.data(), tiled.data(), (size_t) N, &rel);
    std::printf("TENSOR-WINDOW PROOF: %s max_abs=%g max_rel=%g\n", abs <= 1e-5f ? "PASS" : "FAIL", abs, rel);
    ggml_backend_free(backend);
    gguf_free(gguf);
    ggml_free(meta_ctx);
    free(arena);
    close(fd);
    return abs <= 1e-5f ? 0 : 13;
}
