#pragma once
// Shared INPUT definition only: fixed full dimensions, deterministic dyadic
// F32 values, and bounded block templates defining dense quantized test weights.
// Neither this input factory nor either worker accepts a model/file input.
#include "ggml.h"
#include <cstdint>
namespace fixture107 {
constexpr int H=5120,F=40960,BLOCK=256,TEMPLATES=257,TILE=128;
constexpr float EPS=1e-6f,CLAMP=20.f;
inline uint32_t mix(uint32_t v){v^=v>>16;v*=2246822519U;v^=v>>13;v*=3266489917U;return v^(v>>16);}
inline float input(int v,int i){if(v==2)return 0.f;return v==0?float((i*13+7)%257-128)/128.f:float((i*29+31)%509-254)/256.f;}
inline float gamma(int kind,int i){return 1.f+float((i*(kind?7:3)+kind)%17-8)/64.f;}
inline float block_value(int projection,int code,int i){return float(int(mix(uint32_t(projection+1)*2654435761U+uint32_t(code)*1597334677U+uint32_t(i)*3812015801U)%513)-256)/(projection==2?8192.f:128.f);}
inline int code(int projection,int row,int block){return int(mix(uint32_t(row)*2654435761U+uint32_t(block)*2246822519U+uint32_t(projection+1)*3266489917U)%TEMPLATES);}
}
