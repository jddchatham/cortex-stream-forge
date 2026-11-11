#pragma once
#include "ggml.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace r99 {
struct Config {
    int hidden = 5120, ffn = 40960, heads = 80, kv_heads = 8, head = 128;
    int context = 256, window = 1024, layer = 0;
    bool fixture = false;
    float epsilon = 1e-6f, rope_base = 10000.0f, clamp = 20.0f;
    bool sliding() const { return layer % 5 != 4; }
    int slots() const { return (context + 255) / 256 * 256; }
    void validate() const;
};
struct Spec {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    int64_t columns = 0, rows = 0;
    uint64_t offset = 0, bytes = 0;
};
std::vector<Spec> schema(const Config & cfg);
void validate_specs(const Config &, const std::vector<Spec> &);
struct Source {
    virtual ~Source() = default;
    virtual bool is_model() const { return false; }
    virtual void read(const Spec &, size_t offset, void * destination, size_t bytes) = 0;
};
struct ModelFile final : Source {
    int fd = -1;
    explicit ModelFile(const std::string & path);
    ~ModelFile() override;
    bool is_model() const override { return true; }
    ModelFile(const ModelFile &) = delete;
    ModelFile & operator=(const ModelFile &) = delete;
    void read(const Spec &, size_t, void *, size_t) override;
};
struct KV {
    uint64_t owner = 0;
    int position = 0;
    std::vector<ggml_fp16_t> key, value;
};
struct Result {
    std::vector<float> output;
    KV next;
    std::vector<std::string> sequence;
    std::map<std::string, std::vector<float>> captures;
    size_t weight_bytes = 0, graph_bytes = 0, kv_bytes = 0;
};
// Exactly one token; x0 is the original scaled embedding, not the prior layer output.
Result execute(const Config &, const std::vector<Spec> &, Source &,
               const std::vector<float> & input, const std::vector<float> & x0,
               const KV &, bool capture = false, const std::string & fail_at = "");
std::vector<float> mask(const Config &, int position);
uint64_t mem_available();
int live_resources();
void save_kv(const std::string &, const Config &, const KV &);
KV load_kv(const std::string &, const Config &);
struct Schedule {
    int layer = 0, position = 0;
    void require(int requested_layer, int requested_position) const;
    void commit(int completed_layer, int completed_position);
};
}
