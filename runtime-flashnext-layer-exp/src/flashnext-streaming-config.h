#pragma once
#include <cstdint>
#include <string>

enum class flashnext_cache_policy { bounded_reuse, opportunistic, resident };
enum class flashnext_compute_backend { cpu, auto_select };

struct flashnext_streaming_config {
    uint64_t memory_budget_bytes = 0;
    uint64_t swap_budget_bytes = 0;
    uint64_t mandatory_runtime_bytes = 0;
    uint64_t safety_headroom_bytes = 0;
    uint64_t layer_cache_bytes = 0;
    uint64_t expert_cache_bytes = 16 * 1024;
    uint64_t ple_cache_bytes = 0;
    flashnext_cache_policy cache = flashnext_cache_policy::bounded_reuse;
    flashnext_compute_backend backend = flashnext_compute_backend::cpu;
    std::string profile = "auto";
    bool require_hard_limits = false;
    bool hard_limits_verified = false;
};

flashnext_streaming_config flashnext_streaming_config_from_environment();
bool flashnext_streaming_config_validate(const flashnext_streaming_config &, std::string * error);
uint64_t flashnext_effective_expert_cache_bytes();
