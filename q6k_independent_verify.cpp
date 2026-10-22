#include "runtime-flashnext-layer-exp/ggml/src/ggml-quants.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    const char * shard = argv[1], * norm_path = argv[2], * cortex_path = argv[3];
    constexpr size_t rows = 248320, cols = 2560, offset = 10963104;
    constexpr size_t row_bytes = 2100, block_bytes = sizeof(block_q6_K), blocks = 10;
    static_assert(block_bytes == 210, "unexpected Q6_K block size");
    int fd = ::open(shard, O_RDONLY | O_CLOEXEC); if (fd < 0) return 3;
    off_t end = lseek(fd, 0, SEEK_END);
    if (end < 0 || offset + rows * row_bytes > (size_t) end || row_bytes != blocks * block_bytes) return 4;
    FILE * nf = std::fopen(norm_path, "rb"); FILE * cf = std::fopen(cortex_path, "rb");
    if (!nf || !cf) return 5;
    std::vector<float> x(cols), cortex(rows), row(cols);
    if (std::fread(x.data(), sizeof(float), cols, nf) != cols ||
        std::fread(cortex.data(), sizeof(float), rows, cf) != rows) return 6;
    std::vector<size_t> ids;
    for (size_t i = 0; i < 100; ++i) ids.push_back(i * (rows - 1) / 99);
    const size_t top[] = {846,8678,1785,2244,1156};
    for (size_t id : top) ids.push_back(id);
    float max_abs = 0, max_rel = 0; double ss = 0; size_t count = 0, first = rows;
    bool top_ok = true;
    for (size_t r : ids) {
        uint8_t packed[row_bytes];
        if (pread(fd, packed, row_bytes, (off_t) (offset + r * row_bytes)) != (ssize_t) row_bytes) return 7;
        dequantize_row_q6_K((const block_q6_K *) packed, row.data(), cols);
        float sum = 0; for (size_t i = 0; i < cols; ++i) sum += x[i] * row[i];
        const float ref = sum, d = std::fabs(ref - cortex[r]);
        max_abs = std::max(max_abs, d); max_rel = std::max(max_rel, d / std::max(std::fabs(ref), 1e-30f));
        ss += double(d) * double(d); ++count; if (d != 0 && first == rows) first = r;
        if (std::find(top, top + 5, r) != top + 5) std::printf("ROW id=%zu reference=%.9g cortex=%.9g abs=%.9g rel=%.9g\n", r, ref, cortex[r], d, d / std::max(std::fabs(ref), 1e-30f));
    }
    std::printf("Q6K_METADATA rows=%zu cols=%zu offset=%zu row_bytes=%zu blocks_per_row=%zu block_bytes=%zu file_size=%lld\n", rows, cols, offset, row_bytes, blocks, block_bytes, (long long) end);
    std::printf("Q6K_REFERENCE_SAMPLE count=%zu max_abs=%.9g max_rel=%.9g rms=%.9g first_diff=%s\n", count, max_abs, max_rel, std::sqrt(ss / count), first == rows ? "NONE" : std::to_string(first).c_str());
    for (size_t i = 0; i < 5; ++i) {
        uint8_t packed[row_bytes]; pread(fd, packed, row_bytes, (off_t) (offset + top[i] * row_bytes));
        dequantize_row_q6_K((const block_q6_K *) packed, row.data(), cols);
        float sum = 0; for (size_t j = 0; j < cols; ++j) sum += x[j] * row[j];
        std::printf("TOP_ROW rank=%zu id=%zu reference=%.9g cortex=%.9g abs=%.9g\n", i + 1, top[i], sum, cortex[top[i]], std::fabs(sum - cortex[top[i]]));
    }
    for (size_t i = 0; i < 5; ++i) if (top[i] >= rows) top_ok = false;
    std::printf("Q6K_INDEPENDENT_PASS=%s TOP_ORDER_AGREEMENT=%s\n", max_abs <= 1e-5f ? "YES" : "NO", top_ok ? "YES" : "NO");
    std::fclose(nf); std::fclose(cf); ::close(fd); return max_abs <= 1e-5f ? 0 : 8;
}
