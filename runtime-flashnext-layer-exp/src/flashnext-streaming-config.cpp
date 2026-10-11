#include "flashnext-streaming-config.h"
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <vector>

static bool read_limit(const std::string & path, uint64_t * out) {
    std::ifstream in(path); std::string value;
    if (!(in >> value)) return false;
    if (value == "max") { *out = UINT64_MAX; return true; }
    char * end = nullptr; errno = 0;
    const auto parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno || end == value.c_str() || *end) return false;
    *out = (uint64_t) parsed; return true;
}

bool flashnext_resolve_cgroup_limits(const std::string & cgroup_file,
        const std::string & mountinfo_file, uint64_t * memory_limit,
        uint64_t * swap_limit, std::string * resolved_path, std::string * error) {
    const std::string cg_file = cgroup_file.empty() ? "/proc/self/cgroup" : cgroup_file;
    const std::string mi_file = mountinfo_file.empty() ? "/proc/self/mountinfo" : mountinfo_file;
    auto fail = [&](const char * msg) { if (error) *error = msg; return false; };
    std::ifstream cg(cg_file); std::string line, membership;
    while (std::getline(cg, line)) if (line.rfind("0::", 0) == 0) membership = line.substr(3);
    if (membership.empty()) return fail("cgroup v2 membership unavailable");
    std::ifstream mi(mi_file); std::string root, mountpoint;
    while (std::getline(mi, line)) {
        const size_t sep = line.find(" - cgroup2 ");
        if (sep == std::string::npos) continue;
        std::istringstream left(line.substr(0, sep));
        std::vector<std::string> fields; std::string field;
        while (left >> field) fields.push_back(field);
        if (fields.size() >= 5) { root = fields[3]; mountpoint = fields[4]; break; }
    }
    if (root.empty() || mountpoint.empty()) return fail("cgroup2 mount unavailable");
    if (root != "/") {
        if (membership.rfind(root, 0) != 0) return fail("cgroup namespace root does not contain membership");
        membership = membership.substr(root.size());
        if (membership.empty()) membership = "/";
    }
    if (membership.front() != '/') return fail("malformed cgroup membership");
    const std::string base = mountpoint + membership;
    uint64_t memory = UINT64_MAX, swap = UINT64_MAX;
    std::string cur = base;
    for (;;) {
        uint64_t m = 0, s = 0;
        if (!read_limit(cur + "/memory.max", &m) || !read_limit(cur + "/memory.swap.max", &s)) {
            // Some delegated mounts expose controller files only below the
            // mount root.  The root is the inheritance boundary in that case.
            if (cur == mountpoint) break;
            return fail("cgroup limit missing or malformed");
        }
        memory = std::min(memory, m); swap = std::min(swap, s);
        if (cur == mountpoint) break;
        const size_t slash = cur.find_last_of('/');
        cur = slash == 0 ? mountpoint : cur.substr(0, slash);
        if (cur.size() < mountpoint.size()) return fail("cgroup ancestor escaped mount");
    }
    if (memory == UINT64_MAX || swap == UINT64_MAX) return fail("effective cgroup limit is unlimited");
    if (memory_limit) *memory_limit = memory;
    if (swap_limit) *swap_limit = swap;
    if (resolved_path) *resolved_path = base;
    return true;
}

static uint64_t env_u64(const char * name, uint64_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !*value) return fallback;
    char * end = nullptr; errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return errno || end == value || *end ? fallback : static_cast<uint64_t>(parsed);
}

flashnext_streaming_config flashnext_streaming_config_from_environment() {
    flashnext_streaming_config c;
    const char * profile = std::getenv("CORTEX_STREAMING_PROFILE");
    c.profile = profile && *profile ? profile : "auto";
    uint64_t cgroup_memory = 0, cgroup_swap = 0;
    std::string cgroup_error;
    const bool limits = flashnext_resolve_cgroup_limits("", "", &cgroup_memory, &cgroup_swap, &c.resolved_cgroup_path, &cgroup_error);
    const bool memory_verified = limits, swap_verified = limits;
    c.effective_memory_limit_bytes = cgroup_memory;
    c.effective_swap_limit_bytes = cgroup_swap;
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
