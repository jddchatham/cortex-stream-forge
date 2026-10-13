#include "ggml.h"
#include "ggml-quants.h"
#include "gguf.h"
#include "flashnext-tensor-index.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static bool write_all(int fd, const void * p, size_t n, off_t off) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t done = 0; done < n;) {
        ssize_t w = pwrite(fd, b + done, n - done, off + (off_t) done);
        if (w <= 0) return false; done += (size_t) w;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s OUT.gguf OUT.json\n", argv[0]); return 2; }
    const char * gguf_path = argv[1]; const char * json_path = argv[2];
    constexpr int64_t W = 2560, F = 640, E = 512, Q = 256;
    ggml_quantize_init(GGML_TYPE_IQ4_XS);
    const size_t gate_row = ggml_row_size(GGML_TYPE_IQ4_XS, W);
    const size_t down_row = ggml_row_size(GGML_TYPE_IQ4_NL, F);
    if (!gate_row || !down_row || W % ggml_blck_size(GGML_TYPE_IQ4_XS) || F % ggml_blck_size(GGML_TYPE_IQ4_NL)) return 3;
    const size_t gate_bytes = gate_row * (size_t) F * E;
    const size_t down_bytes = down_row * (size_t) W * E;
    ggml_init_params gip{16*1024*1024, nullptr, true};
    ggml_context * gctx = ggml_init(gip); if (!gctx) return 4;
    ggml_tensor * gate = ggml_new_tensor_3d(gctx, GGML_TYPE_IQ4_XS, W, F, E);
    ggml_tensor * up   = ggml_new_tensor_3d(gctx, GGML_TYPE_IQ4_XS, W, F, E);
    ggml_tensor * down = ggml_new_tensor_3d(gctx, GGML_TYPE_IQ4_NL, F, W, E);
    ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(up,   "blk.0.ffn_up_exps.weight");
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    gguf_context * meta = gguf_init_empty();
    gguf_set_val_u32(meta, GGUF_KEY_GENERAL_ALIGNMENT, 64);
    gguf_set_val_str(meta, "general.architecture", "qwen4exp");
    gguf_add_tensor(meta, gate); gguf_add_tensor(meta, up); gguf_add_tensor(meta, down);
    if (!gguf_write_to_file(meta, gguf_path, true)) return 5;
    struct stat meta_stat{}; if (stat(gguf_path, &meta_stat) != 0) return 5;
    const size_t metadata_data_base = ((size_t) meta_stat.st_size + 63) & ~size_t(63);
    gguf_free(meta); ggml_free(gctx);
    { int mfd = open(gguf_path, O_RDWR|O_CLOEXEC); if (mfd < 0 || ftruncate(mfd, metadata_data_base) != 0) return 5; close(mfd); }
    gguf_init_params gip2{true, nullptr}; gguf_context * read = gguf_init_from_file(gguf_path, gip2);
    if (!read || gguf_get_n_tensors(read) != 3) return 6;
    const size_t data_base = gguf_get_data_offset(read);
    const int64_t tid_g = gguf_find_tensor(read, "blk.0.ffn_gate_exps.weight");
    const int64_t tid_u = gguf_find_tensor(read, "blk.0.ffn_up_exps.weight");
    const int64_t tid_d = gguf_find_tensor(read, "blk.0.ffn_down_exps.weight");
    if (tid_g < 0 || tid_u < 0 || tid_d < 0) return 7;
    const size_t go = data_base + gguf_get_tensor_offset(read, tid_g);
    const size_t uo = data_base + gguf_get_tensor_offset(read, tid_u);
    const size_t doff = data_base + gguf_get_tensor_offset(read, tid_d);
    const size_t total = doff + down_bytes;
    gguf_free(read);
    int fd = open(gguf_path, O_RDWR|O_CLOEXEC); if (fd < 0 || ftruncate(fd, total) != 0) return 8;
    std::vector<float> row(W), qrow(gate_row / sizeof(float) + 1);
    std::vector<uint8_t> encoded(std::max(gate_row, down_row));
    const int experts[] = {0, 3};
    for (int e : experts) for (int r = 0; r < F; ++r) {
        for (int i=0;i<W;++i) row[i] = 0.01f * float(1 + (e*7 + r*3 + i)%23);
        quantize_row_iq4_xs_ref(row.data(), (block_iq4_xs *) encoded.data(), W);
        if (!write_all(fd, encoded.data(), gate_row, go + (off_t)e*gate_row*F + (off_t)r*gate_row)) return 9;
        if (!write_all(fd, encoded.data(), gate_row, uo + (off_t)e*gate_row*F + (off_t)r*gate_row)) return 10;
    }
    for (int e : experts) for (int r = 0; r < W; ++r) {
        for (int i=0;i<F;++i) row[i] = 0.02f * float(1 + (e*5 + r*2 + i)%19);
        quantize_row_iq4_nl_ref(row.data(), (block_iq4_nl *) encoded.data(), F);
        if (!write_all(fd, encoded.data(), down_row, doff + (off_t)e*down_row*W + (off_t)r*down_row)) return 11;
    }
    close(fd);
    std::ofstream j(json_path);
    j << "{\"model_dir\":\"\",\"tensors\":[";
    auto entry = [&](const char * n, size_t off, size_t bytes, int a, int b) {
        j << "{\"name\":\"" << n << "\",\"shard\":\"" << gguf_path << "\",\"offset\":" << off << ",\"bytes\":" << bytes << ",\"shape\":[" << a << "," << b << ",512],\"type\":23}";
    };
    entry("blk.0.ffn_gate_exps.weight", go, gate_bytes, 2560, 640); j << ',';
    entry("blk.0.ffn_up_exps.weight", uo, gate_bytes, 2560, 640); j << ',';
    j << "{\"name\":\"blk.0.ffn_down_exps.weight\",\"shard\":\"" << gguf_path << "\",\"offset\":" << doff << ",\"bytes\":" << down_bytes << ",\"shape\":[640,2560,512],\"type\":20}" << "]}";
    j.close();
    flashnext_tensor_index idx; std::string err;
    if (!idx.load(json_path, &err)) { fprintf(stderr, "fixture index: %s\n", err.c_str()); return 12; }
    flashnext_tensor_descriptor d;
    if (!idx.resolve_expert(0, 3, flashnext_projection::gate, d, &err) || d.offset != go + 3*gate_row*F) return 13;
    struct stat st{}; if (stat(gguf_path, &st) != 0) return 14;
    printf("IQ4XS_FIXTURE PASS path=%s index=%s logical_bytes=%lld allocated_blocks=%lld gate_row=%zu down_row=%zu expert_stride=%zu\n", gguf_path, json_path, (long long)st.st_size, (long long)st.st_blocks*512LL, gate_row, down_row, gate_row*(size_t)F);
    return 0;
}
