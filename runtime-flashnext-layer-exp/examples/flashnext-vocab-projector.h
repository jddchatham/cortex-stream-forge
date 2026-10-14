#pragma once

#include "llama-model.h"
#include <cstddef>
#include <vector>
#include <array>

struct cortex_vocab_projection_result {
    int token = -1;
    float logit = 0.0f;
    size_t bytes_read = 0;
    size_t scratch_bytes = 0;
    static constexpr size_t MAX_TOP = 64;
    std::array<int, MAX_TOP> top_tokens{};
    std::array<float, MAX_TOP> top_logits{};
    size_t top_count = 0;
};

bool cortex_stream_vocab_project(const cortex_output_weight_descriptor & descriptor,
                                 const std::vector<float> & hidden,
                                 size_t tile_rows,
                                 cortex_vocab_projection_result & result);
