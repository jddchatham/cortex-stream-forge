#include "flashnext-ple-provider.h"
#include "../ggml/src/ggml-quants.h"

#include <cerrno>
#include <limits>
#include <unistd.h>
#include <vector>

bool flashnext_read_quantized_rows(int fd, uint64_t tensor_offset,
        ggml_type type, int64_t row_elements, int64_t row_count,
        const int32_t * rows, size_t count, float * dst) {
    if (fd < 0 || !rows || !dst || row_elements <= 0 || row_count <= 0) return false;
    if ((uint64_t) row_elements > (uint64_t) std::numeric_limits<size_t>::max()) return false;
    const size_t row_bytes = ggml_row_size(type, row_elements);
    if (row_bytes == 0 || count > std::numeric_limits<size_t>::max() / row_bytes) return false;
    std::vector<uint8_t> packed(row_bytes);
    for (size_t i = 0; i < count; ++i) {
        if (rows[i] < 0 || (int64_t) rows[i] >= row_count) return false;
        const uint64_t r = (uint64_t) rows[i];
        if (r > (std::numeric_limits<uint64_t>::max() - tensor_offset) / row_bytes) return false;
        const uint64_t off = tensor_offset + r * row_bytes;
        size_t done = 0;
        while (done < row_bytes) {
            const ssize_t got = pread(fd, packed.data() + done, row_bytes - done, (off_t) (off + done));
            if (got <= 0) return false;
            done += (size_t) got;
        }
        if (type == GGML_TYPE_IQ4_XS) {
            dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(packed.data()),
                dst + i * (size_t) row_elements, row_elements);
        } else if (type == GGML_TYPE_IQ4_NL) {
            dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(packed.data()),
                dst + i * (size_t) row_elements, row_elements);
        } else {
            auto * traits = ggml_get_type_traits(type);
            if (!traits || !traits->to_float) return false;
            traits->to_float(packed.data(), dst + i * (size_t) row_elements, row_elements);
        }
    }
    return true;
}
