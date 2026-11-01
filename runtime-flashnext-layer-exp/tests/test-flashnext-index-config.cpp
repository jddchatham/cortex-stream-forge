#include "../include/flashnext-config.h"

#include <cassert>
#include <cstdlib>

int main() {
    unsetenv("FLASHNEXT_TENSOR_INDEX");
    assert(flashnext_tensor_index_from_environment().empty());
    setenv("FLASHNEXT_TENSOR_INDEX", "/validated/index.json", 1);
    assert(flashnext_tensor_index_from_environment() == "/validated/index.json");
    return 0;
}
