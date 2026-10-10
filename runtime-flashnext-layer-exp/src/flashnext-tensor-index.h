#pragma once

#include <cstdint>
#include <string>

enum class flashnext_projection { gate, up, down, shared_control, hc_norm, hc_down, hc_up, hc_inject };

struct flashnext_tensor_descriptor {
    std::string name;
    std::string shard;
    std::string path;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    int type = -1;
    int64_t expert_stride = 0;
    int64_t shape0 = 0, shape1 = 0, shape2 = 0;
    uint32_t block_size = 0;
    uint64_t row_bytes = 0;
};

class flashnext_tensor_index {
public:
    bool load(const std::string & path, std::string * error = nullptr);
    bool resolve_expert(int layer, int expert, flashnext_projection projection,
                        flashnext_tensor_descriptor & out, std::string * error = nullptr) const;
    bool resolve_shared(int layer, flashnext_projection projection,
                        flashnext_tensor_descriptor & out, std::string * error = nullptr) const;
    bool resolve_hc_attention(int layer, flashnext_projection projection,
                              flashnext_tensor_descriptor & out, std::string * error = nullptr) const;
    bool resolve_named(const std::string & name, flashnext_tensor_descriptor & out, std::string * error = nullptr) const;
};
