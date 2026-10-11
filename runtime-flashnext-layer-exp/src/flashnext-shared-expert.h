#pragma once

#include "flashnext-layer-transaction.h"

#include <array>
#include <cstdint>

struct flashnext_shared_expert_slices {
    std::array<int, 4> fds{{-1, -1, -1, -1}};
    uint64_t gate_offset = 0;
    uint64_t up_offset = 0;
    uint64_t down_offset = 0;
    uint64_t sigmoid_offset = 0;
};

bool flashnext_execute_shared_expert(const flashnext_shared_expert_slices & slices,
                                     flashnext_layer_transaction & tx);

