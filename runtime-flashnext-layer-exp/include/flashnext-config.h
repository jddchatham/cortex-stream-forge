#pragma once

#include <cstdlib>
#include <string>

// Runtime configuration is supplied by the trusted launcher.  Do not use a
// machine-specific fallback when the variable is absent.
inline std::string flashnext_tensor_index_from_environment() {
    const char * value = std::getenv("FLASHNEXT_TENSOR_INDEX");
    return value != nullptr ? std::string(value) : std::string();
}
