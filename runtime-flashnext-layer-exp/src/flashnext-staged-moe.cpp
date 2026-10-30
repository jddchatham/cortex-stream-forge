#include "flashnext-staged-moe.h"
#include "flashnext-selected-moe.h"
#include "flashnext-shared-expert.h"

#include <cmath>
#include <fcntl.h>
#include <unistd.h>

namespace {
template <typename Range>
bool finite_range(const Range & values) {
    for (const auto value : values) if (!std::isfinite(value)) return false;
    return true;
}
}
bool flashnext_staged_moe::begin(int layer, int64_t position, std::string & error) {
    if (state != phase::empty) { error = "transaction already begun"; return false; }
    if (layer < 0 || position < 0) { error = "invalid layer or position"; return false; }
    tx = flashnext_layer_transaction{};
    tx.layer = layer;
    tx.position = position;
    state = phase::empty;
    return true;
}

bool flashnext_staged_moe::execute_streamed_moe(const flashnext_tensor_index & index,
                                                const std::string & index_path,
                                                std::string & error) {
    if (state != phase::pre_moe) { error = "Stage B requires completed Stage A"; return false; }

    if (!flashnext_execute_selected_moe(index, index_path, tx)) {
        error = "selected-expert streaming failed";
        return false;
    }

    flashnext_shared_expert_slices shared;
    std::array<flashnext_tensor_descriptor, 4> descriptors;
    const flashnext_projection projections[] = {
        flashnext_projection::gate, flashnext_projection::up,
        flashnext_projection::down, flashnext_projection::shared_control,
    };
    for (size_t i = 0; i < descriptors.size(); ++i) {
        if (!index.resolve_shared(tx.layer, projections[i], descriptors[i], &error)) return false;
        shared.fds[i] = ::open(descriptors[i].path.c_str(), O_RDONLY | O_CLOEXEC);
        if (shared.fds[i] < 0) { error = "cannot open shared-expert shard"; break; }
    }
    if (error.empty()) {
        shared.gate_offset = descriptors[0].offset;
        shared.up_offset = descriptors[1].offset;
        shared.down_offset = descriptors[2].offset;
        shared.sigmoid_offset = descriptors[3].offset;
        if (!flashnext_execute_shared_expert(shared, tx)) error = "shared-expert streaming failed";
    }
    for (int fd : shared.fds) if (fd >= 0) ::close(fd);
    if (!error.empty()) return false;

    for (size_t i = 0; i < tx.complete_moe.size(); ++i)
        tx.complete_moe[i] = tx.selected_moe[i] + tx.shared_moe[i];
    state = phase::streamed;
    return finite_range(tx.complete_moe);
}
bool flashnext_staged_moe::run(stage_a_fn stage_a, stage_b_fn stage_b,
                               stage_c_fn stage_c, std::string & error) {
    if (state != phase::empty) { error = "invalid begin state"; return false; }
    if (!stage_a || !stage_b || !stage_c) { error = "all three stages are required"; return false; }
    if (!stage_a(tx, error)) return false;
    if (!finite_range(tx.pre_moe_hidden) || !finite_range(tx.residual_hc) || !finite_range(tx.inject)) {
        error = "nonfinite Stage-A boundary"; return false;
    }
    state = phase::pre_moe;
    if (!stage_b(tx, error)) return false;
    if (!finite_range(tx.selected_moe) || !finite_range(tx.shared_moe) || !finite_range(tx.complete_moe)) {
        error = "nonfinite Stage-B output"; return false;
    }
    state = phase::streamed;
    if (!stage_c(tx, error)) return false;
    if (!finite_range(tx.layer_output)) { error = "nonfinite Stage-C output"; return false; }
    state = phase::post_moe;
    return true;
}
bool flashnext_staged_moe::close(std::string & error) {
    if (state == phase::closed) { error = "transaction already closed"; return false; }
    if (state != phase::post_moe) { error = "cannot close incomplete transaction"; return false; }
    state = phase::closed;
    return true;
}
