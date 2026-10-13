#include "../src/flashnext-streaming-config.h"
#include <cstdio>
#include <cstdlib>
#include <string>

static bool expect(bool value, const char * label) {
    std::printf("STREAMING_CONFIG_TEST %s=%s\n", label, value ? "PASS" : "FAIL");
    return value;
}

int main() {
    setenv("CORTEX_STREAMING_PROFILE", "reference-2g", 1);
    const auto reference = flashnext_streaming_config_from_environment();
    std::string error;
    bool ok = true;
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
    return ok ? 0 : 1;
}
