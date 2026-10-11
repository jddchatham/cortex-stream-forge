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
    std::string resolved_cgroup_path;
    uint64_t effective_memory_limit_bytes = 0;
    uint64_t effective_swap_limit_bytes = 0;
};

// Resolve a cgroup-v2 membership using proc-style fixture files.  Empty paths
// use the live process.  This is also used by deterministic resolver tests.
bool flashnext_resolve_cgroup_limits(const std::string & cgroup_file,
        const std::string & mountinfo_file, uint64_t * memory_limit,
        uint64_t * swap_limit, std::string * resolved_path, std::string * error);

flashnext_streaming_config flashnext_streaming_config_from_environment();
bool flashnext_streaming_config_validate(const flashnext_streaming_config &, std::string * error);
uint64_t flashnext_effective_expert_cache_bytes();
