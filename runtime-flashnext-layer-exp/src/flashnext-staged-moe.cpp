#include "flashnext-staged-moe.h"

#include <cmath>

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
