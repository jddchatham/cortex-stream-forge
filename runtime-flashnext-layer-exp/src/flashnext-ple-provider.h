#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// Metadata-only bounded row reader. It owns no model/context state and never
// allocates the source tensor; dst must contain count * row_elements floats.
bool flashnext_read_quantized_rows(int fd, uint64_t tensor_offset,
        ggml_type type, int64_t row_elements, int64_t row_count,
        const int32_t * rows, size_t count, float * dst);
