#pragma once

#include "flashnext-layer-transaction.h"

#include <functional>
#include <string>

// Typed, bounded boundary between the Qwen graph and streamed MoE execution.
// Callers own the callback inputs and the transaction owns all boundary data.
struct flashnext_staged_moe {
    enum class phase { empty, pre_moe, streamed, post_moe, closed };
    using stage_a_fn = std::function<bool(flashnext_layer_transaction &, std::string &)>;
    using stage_b_fn = std::function<bool(flashnext_layer_transaction &, std::string &)>;
    using stage_c_fn = std::function<bool(const flashnext_layer_transaction &, std::string &)>;

    flashnext_layer_transaction tx;
    phase state = phase::empty;

    bool begin(int layer, int64_t position, std::string & error);
    bool run(stage_a_fn stage_a, stage_b_fn stage_b, stage_c_fn stage_c, std::string & error);
    bool close(std::string & error);
};
