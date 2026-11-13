#pragma once
// Deterministic input contract ONLY. No model filenames, readers, mappings or
// serialized model state. Both paths generate these dyadic values independently.
#include "ggml.h"
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdint>
namespace synthetic106 {
struct Case { const char*name;int type;int columns;int rows; };
inline float activation(int i) { return float((i*13+7)%257-128)/128.0f; }
inline float weight(int row,int col) {
 uint32_t v=uint32_t(row)*2654435761U+uint32_t(col)*2246822519U+3266489917U;
 v^=v>>13;return float(int(v%513)-256)/8192.0f;
}
inline void require(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
}
