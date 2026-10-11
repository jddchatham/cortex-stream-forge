#include "../src/flashnext-streaming-config.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <filesystem>
#include <unistd.h>

static bool expect(bool value, const char * label) {
    std::printf("STREAMING_CONFIG_TEST %s=%s\n", label, value ? "PASS" : "FAIL");
    return value;
}

int main() {
    setenv("CORTEX_STREAMING_PROFILE", "reference-2g", 1);
    const auto reference = flashnext_streaming_config_from_environment();
    std::string error;
    bool ok = true;
    if (std::getenv("CORTEX_STREAMING_SCOPE_PROBE")) {
        const bool valid = flashnext_streaming_config_validate(reference, &error);
        std::printf("STREAMING_SCOPE_PROBE valid=%s cgroup=%s effective_memory=%llu effective_swap=%llu error=%s\n",
            valid ? "YES" : "NO", reference.resolved_cgroup_path.c_str(),
            (unsigned long long) reference.effective_memory_limit_bytes,
            (unsigned long long) reference.effective_swap_limit_bytes, error.c_str());
        return valid ? 0 : 1;
    }
    ok &= expect(reference.memory_budget_bytes == 2ull * 1024 * 1024 * 1024, "reference_memory");
    ok &= expect(reference.swap_budget_bytes == 0, "reference_swap");
    // This host has no readable cgroup files; environment-derived execution
    // must fail closed rather than claim a hard 2 GiB gate.
    ok &= expect(!flashnext_streaming_config_validate(reference, &error), "reference_unverified_rejected");
    auto reference_contract = reference;
    reference_contract.require_hard_limits = false;
    ok &= expect(flashnext_streaming_config_validate(reference_contract, &error), "reference_contract_valid");
    auto resident = reference_contract;
    resident.cache = flashnext_cache_policy::resident;
    ok &= expect(!flashnext_streaming_config_validate(resident, &error), "reference_resident_rejected");
    setenv("CORTEX_STREAMING_PROFILE", "auto", 1);
    setenv("CORTEX_STREAMING_MEMORY_BUDGET", "1073741824", 1);
    setenv("CORTEX_STREAMING_SWAP_BUDGET", "0", 1);
    setenv("CORTEX_STREAMING_MANDATORY_BYTES", "268435456", 1);
    setenv("CORTEX_STREAMING_HEADROOM_BYTES", "268435456", 1);
    setenv("CORTEX_STREAMING_EXPERT_CACHE_BYTES", "16384", 1);
    const auto automatic = flashnext_streaming_config_from_environment();
    ok &= expect(!flashnext_streaming_config_validate(automatic, &error), "auto_unverified_rejected");
    auto automatic_contract = automatic;
    automatic_contract.require_hard_limits = false;
    ok &= expect(flashnext_streaming_config_validate(automatic_contract, &error), "auto_contract_valid");
    auto oversized = automatic_contract;
    oversized.layer_cache_bytes = 800ull * 1024 * 1024;
    ok &= expect(!flashnext_streaming_config_validate(oversized, &error), "cache_budget_rejected");

    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("flashnext-cgroup-test-" + std::to_string((long long)getpid()));
    fs::create_directories(root / "cg" / "a" / "b");
    auto write = [](const fs::path & p, const std::string & value) {
        std::ofstream f(p); f << value << "\n";
    };
    write(root / "proc", "0::/a/b\n");
    write(root / "mountinfo", "1 0 0:1 / " + (root / "cg").string() + " rw - cgroup2 cgroup rw\n");
    write(root / "cg" / "memory.max", "2147483648"); write(root / "cg" / "memory.swap.max", "0");
    write(root / "cg" / "a" / "memory.max", "2147483648"); write(root / "cg" / "a" / "memory.swap.max", "max");
    write(root / "cg" / "a" / "b" / "memory.max", "max"); write(root / "cg" / "a" / "b" / "memory.swap.max", "0");
    uint64_t mem = 0, swp = 0; std::string path, why;
    ok &= expect(flashnext_resolve_cgroup_limits((root / "proc").string(), (root / "mountinfo").string(), &mem, &swp, &path, &why) && mem == 2147483648ull && swp == 0, "cgroup_nested_inherited");
    write(root / "cg" / "a" / "memory.max", "bad");
    ok &= expect(!flashnext_resolve_cgroup_limits((root / "proc").string(), (root / "mountinfo").string(), &mem, &swp, &path, &why), "cgroup_malformed_rejected");
    write(root / "mountinfo", "1 0 0:1 / /sys/fs/cgroup rw - tmpfs tmpfs rw\n");
    ok &= expect(!flashnext_resolve_cgroup_limits((root / "proc").string(), (root / "mountinfo").string(), &mem, &swp, &path, &why), "cgroup_v1_rejected");
    fs::create_directories(root / "cg2" / "b");
    write(root / "proc", "0::/a/b\n");
    write(root / "mountinfo", "1 0 0:1 /a " + (root / "cg2").string() + " rw - cgroup2 cgroup rw\n");
    write(root / "cg2" / "memory.max", "2147483648"); write(root / "cg2" / "memory.swap.max", "0");
    write(root / "cg2" / "b" / "memory.max", "max"); write(root / "cg2" / "b" / "memory.swap.max", "0");
    const bool translated = flashnext_resolve_cgroup_limits((root / "proc").string(), (root / "mountinfo").string(), &mem, &swp, &path, &why);
    if (!translated) std::printf("STREAMING_SCOPE_TRANSLATION_ERROR %s\n", why.c_str());
    ok &= expect(translated && mem == 2147483648ull && swp == 0, "cgroup_namespace_root_translation");
    fs::remove_all(root);
    return ok ? 0 : 1;
}
