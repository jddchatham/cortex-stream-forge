#include "flashnext-streaming-config.h"
#include <cerrno>
#include <cstdlib>
#include <fstream>

static uint64_t env_u64(const char * name, uint64_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !*value) return fallback;
    char * end = nullptr; errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return errno || end == value || *end ? fallback : static_cast<uint64_t>(parsed);
}

static bool cgroup_limit(const char * name, uint64_t * result) {
    std::ifstream in(std::string("/sys/fs/cgroup/") + name);
    std::string value;
    if (!(in >> value) || value == "max") return false;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (end == value.c_str() || *end) return false;
    *result = static_cast<uint64_t>(parsed);
    return true;
}

flashnext_streaming_config flashnext_streaming_config_from_environment() {
    flashnext_streaming_config c;
    const char * profile = std::getenv("CORTEX_STREAMING_PROFILE");
    c.profile = profile && *profile ? profile : "auto";
    uint64_t cgroup_memory = 0, cgroup_swap = 0;
    const bool memory_verified = cgroup_limit("memory.max", &cgroup_memory);
    const bool swap_verified = cgroup_limit("memory.swap.max", &cgroup_swap);
    c.require_hard_limits = true;
    if (c.profile == "reference-2g") {
        c.memory_budget_bytes = 2ull * 1024 * 1024 * 1024;
        c.swap_budget_bytes = 0;
        c.mandatory_runtime_bytes = 256ull * 1024 * 1024;
        c.safety_headroom_bytes = 512ull * 1024 * 1024;
        c.hard_limits_verified = memory_verified && swap_verified &&
            cgroup_memory == c.memory_budget_bytes && cgroup_swap == c.swap_budget_bytes;
    } else {
        c.memory_budget_bytes = env_u64("CORTEX_STREAMING_MEMORY_BUDGET", memory_verified ? cgroup_memory : 0);
        c.swap_budget_bytes = env_u64("CORTEX_STREAMING_SWAP_BUDGET", swap_verified ? cgroup_swap : 0);
        c.mandatory_runtime_bytes = env_u64("CORTEX_STREAMING_MANDATORY_BYTES", 0);
        c.safety_headroom_bytes = env_u64("CORTEX_STREAMING_HEADROOM_BYTES", 0);
        c.layer_cache_bytes = env_u64("CORTEX_STREAMING_LAYER_CACHE_BYTES", 0);
        c.expert_cache_bytes = env_u64("CORTEX_STREAMING_EXPERT_CACHE_BYTES", 16 * 1024);
        c.ple_cache_bytes = env_u64("CORTEX_STREAMING_PLE_CACHE_BYTES", 0);
        c.cache = flashnext_cache_policy::opportunistic;
        c.hard_limits_verified = memory_verified && swap_verified &&
            c.memory_budget_bytes <= cgroup_memory && c.swap_budget_bytes <= cgroup_swap;
    }
    return c;
}

bool flashnext_streaming_config_validate(const flashnext_streaming_config & c, std::string * error) {
    const auto fail = [&](const char * message) { if (error) *error = message; return false; };
    if (c.profile != "auto" && c.profile != "reference-2g") return fail("unknown streaming profile");
    if (c.require_hard_limits && !c.hard_limits_verified) return fail("effective hard memory/swap limits are not verified");
    if (!c.memory_budget_bytes) return fail("memory budget is not established");
    if (c.profile == "reference-2g" && c.swap_budget_bytes != 0) return fail("reference-2g requires zero swap");
    if (c.mandatory_runtime_bytes > c.memory_budget_bytes || c.safety_headroom_bytes > c.memory_budget_bytes - c.mandatory_runtime_bytes) return fail("mandatory runtime/headroom exceeds budget");
    const uint64_t available = c.memory_budget_bytes - c.mandatory_runtime_bytes - c.safety_headroom_bytes;
    if (c.layer_cache_bytes > available) return fail("layer cache exceeds available budget");
    if (c.expert_cache_bytes > available - c.layer_cache_bytes) return fail("expert cache exceeds available budget");
    if (c.ple_cache_bytes > available - c.layer_cache_bytes - c.expert_cache_bytes) return fail("PLE cache exceeds available budget");
    if (c.profile == "reference-2g" && c.cache == flashnext_cache_policy::resident) return fail("resident caching is not permitted by reference-2g");
    if (c.profile == "reference-2g" && c.backend != flashnext_compute_backend::cpu) return fail("reference-2g requires CPU backend");
    return true;
}

uint64_t flashnext_effective_expert_cache_bytes() {
    const flashnext_streaming_config c = flashnext_streaming_config_from_environment();
    std::string error;
    return flashnext_streaming_config_validate(c, &error) ? c.expert_cache_bytes : 0;
}
