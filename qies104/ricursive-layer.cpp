#include "admission.h"
#include "ricursive-layer.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace r99 {
namespace {
std::atomic<int> resources{0};
constexpr size_t weight_limit = 768ULL * 1024 * 1024;
constexpr size_t graph_limit = 128ULL * 1024 * 1024;
constexpr size_t io_limit = 1024 * 1024;
void require(bool condition, const char * text) { if (!condition) throw std::runtime_error(text); }
bool allowed(ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_IQ4_XS; }
size_t bytes_for(ggml_type type, int64_t columns, int64_t rows) {
    require(allowed(type) && columns > 0 && rows > 0 && columns <= 40960 && rows <= 40960, "invalid tensor type/dimensions");
    require(columns % ggml_blck_size(type) == 0, "unaligned quantized row");
    return ggml_row_size(type, columns) * (size_t) rows;
}
struct Context {
    ggml_context * p = nullptr;
    explicit Context(size_t bytes) { p = ggml_init({bytes, nullptr, true}); require(p, "context allocation failed"); ++resources; }
    ~Context() { if (p) { ggml_free(p); --resources; } }
};
struct Backend {
    ggml_backend_t p = nullptr;
    Backend() { p = ggml_backend_cpu_init(); require(p, "CPU backend allocation failed"); ++resources; ggml_backend_cpu_set_n_threads(p, 1); }
    ~Backend() { if (p) { ggml_backend_free(p); --resources; } }
};
struct Buffer {
    ggml_backend_buffer_t p = nullptr;
    Buffer(ggml_context * c, ggml_backend_t b, size_t limit) {
        size_t estimated = 0;
        for (auto * t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t)) {
            if (!t->view_src) { require(ggml_nbytes(t) <= limit - estimated, "buffer budget exceeded"); estimated += ggml_nbytes(t); }
        }
        p = ggml_backend_alloc_ctx_tensors(c, b); require(p, "backend buffer allocation failed"); ++resources;
        if (ggml_backend_buffer_get_size(p) > limit) { ggml_backend_buffer_free(p); p = nullptr; --resources; throw std::runtime_error("aligned buffer budget exceeded"); }
    }
    ~Buffer() { if (p) { ggml_backend_buffer_free(p); --resources; } }
};
bool finite(const std::vector<float> & values) { return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }); }
void fail(const std::string & requested, const char * point) { if (requested == point) throw std::runtime_error(point); }
void exact_read(int fd, void * destination, size_t bytes, uint64_t offset) {
    auto * out = static_cast<uint8_t *>(destination);
    for (size_t done = 0; done < bytes;) {
        ssize_t n = ::pread(fd, out + done, bytes - done, (off_t) (offset + done));
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "short/failed tensor read"); done += (size_t) n;
    }
}
void write_all(int fd, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t done = 0; done < size;) {
        ssize_t n = ::write(fd, bytes + done, size - done);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "state write failed"); done += (size_t) n;
    }
}
uint64_t hash(const void * data, size_t n) {
    uint64_t value = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { value ^= static_cast<const uint8_t *>(data)[i]; value *= 1099511628211ULL; }
    return value;
}
uint32_t bits(float value) { uint32_t out; std::memcpy(&out, &value, 4); return out; }
std::array<uint64_t, 24> state_header(const Config & c, const KV & s) {
    return {{0x3130564b393951ULL, 1, c.fixture ? 1ULL : 0ULL,
             (uint64_t) c.layer, (uint64_t) c.hidden, (uint64_t) c.ffn, (uint64_t) c.heads,
             (uint64_t) c.kv_heads, (uint64_t) c.head, (uint64_t) c.context, (uint64_t) c.window,
             (uint64_t) s.position, (uint64_t) s.key.size(), bits(c.epsilon), bits(c.rope_base), bits(c.clamp),
             hash(s.key.data(), s.key.size() * 2), hash(s.value.data(), s.value.size() * 2),
             c.fixture ? 0ULL : 0x9016e260f93992f0ULL, c.fixture ? 0ULL : 0xacc7cb590b012fe2ULL,
             c.fixture ? 0ULL : 0x1fffddf767b53774ULL, c.fixture ? 0ULL : 0x387fac1b30ca7bfcULL,
             0, 0x04030201}};
}
uint64_t owner_of(const Config & c) {
    const std::array<uint64_t, 12> fields{{(uint64_t) c.fixture, (uint64_t) c.layer,
        (uint64_t) c.hidden, (uint64_t) c.ffn, (uint64_t) c.heads, (uint64_t) c.kv_heads,
        (uint64_t) c.head, (uint64_t) c.context, (uint64_t) c.window,
        bits(c.epsilon), bits(c.rope_base), bits(c.clamp)}};
    return hash(fields.data(), sizeof(fields));
}
void validate_kv(const Config & c, const KV & s) {
    require((s.owner == 0 && s.position == 0 && s.key.empty() && s.value.empty()) || s.owner == owner_of(c), "KV owner/config mismatch");
    require(s.position >= 0 && s.position <= c.context, "invalid KV position");
    size_t count = (size_t) c.head * c.kv_heads * c.slots();
    require((s.key.empty() && s.value.empty() && s.position == 0) || (s.key.size() == count && s.value.size() == count), "KV shape mismatch");
    for (auto v : s.key) require(std::isfinite(ggml_fp16_to_fp32(v)), "nonfinite KV key");
    for (auto v : s.value) require(std::isfinite(ggml_fp16_to_fp32(v)), "nonfinite KV value");
}
}
void Config::validate() const {
    const uint16_t endian = 1; require(*(const uint8_t *) &endian == 1, "little-endian native state required");
    require(hidden > 0 && hidden <= 5120 && ffn > 0 && ffn <= 40960 && heads > 0 && heads <= 80 && kv_heads > 0 && kv_heads <= 8 && heads % kv_heads == 0, "invalid layer dimensions");
    require(head > 0 && head <= 128 && head % 2 == 0 && context > 0 && context <= 256 && window > 0 && window <= 1024 && layer >= 0 && layer < 80, "unsupported context/layer");
    if (!fixture) require(hidden == 5120 && ffn == 40960 && heads == 80 && kv_heads == 8 && head == 128 && window == 1024 && epsilon == 1e-6f && rope_base == 10000 && clamp == 20, "wrong real-model configuration");
    require(std::isfinite(epsilon) && epsilon > 0 && std::isfinite(rope_base) && rope_base > 0 && std::isfinite(clamp) && clamp >= 0, "invalid hyperparameters");
}
std::vector<Spec> schema(const Config & c) {
    c.validate(); std::vector<Spec> out;
    auto add = [&](const char * suffix, int64_t columns, int64_t rows) { out.push_back({"blk." + std::to_string(c.layer) + "." + suffix, GGML_TYPE_F32, columns, rows, 0, bytes_for(GGML_TYPE_F32, columns, rows)}); };
    add("attn_norm.weight", c.hidden, 1);
    add("attn_q.weight", c.hidden, c.head * c.heads);
    add("attn_k.weight", c.hidden, c.head * c.kv_heads);
    add("attn_v.weight", c.hidden, c.head * c.kv_heads);
    add("attn_output.weight", c.head * c.heads, c.hidden);
    add("attn_q_norm.weight", c.head, 1); add("attn_k_norm.weight", c.head, 1);
    add("attn_gate.weight", c.hidden, c.head * c.heads); add("attn_vres_lambda.weight", 1, 1);
    add("post_attention_norm.weight", c.hidden, 1); add("ffn_norm.weight", c.hidden, 1);
    add("ffn_gate.weight", c.hidden, c.ffn); add("ffn_up.weight", c.hidden, c.ffn);
    add("ffn_down.weight", c.ffn, c.hidden); add("post_ffw_norm.weight", c.hidden, 1);
    return out;
}
void validate_specs(const Config & c, const std::vector<Spec> & specs) {
    auto expected = schema(c); require(specs.size() == expected.size(), "layer requires exactly 15 tensors");
    size_t total = 0;
    for (const auto & e : expected) {
        auto it = std::find_if(specs.begin(), specs.end(), [&](const Spec & s) { return s.name == e.name; });
        require(it != specs.end() && std::count_if(specs.begin(), specs.end(), [&](const Spec & s) { return s.name == e.name; }) == 1, "missing/duplicate tensor");
        require(it->columns == e.columns && it->rows == e.rows, "tensor shape mismatch");
        require(it->bytes == bytes_for(it->type, it->columns, it->rows), "tensor byte count mismatch");
        if (it->rows == 1) require(it->type == GGML_TYPE_F32, "norm/lambda must be F32");
        require(it->offset <= std::numeric_limits<uint64_t>::max() - it->bytes, "offset overflow");
        require(it->bytes <= weight_limit - total, "active layer exceeds weight budget"); total += it->bytes;
    }
}
uint64_t mem_available() {
    std::ifstream f("/proc/meminfo"); std::string key, units; uint64_t n;
    while (f >> key >> n >> units) if (key == "MemAvailable:") return n * 1024;
    throw std::runtime_error("cannot read MemAvailable");
}
ModelFile::ModelFile(const std::string & path) {
    q104::real("ModelFile constructor");
    require(mem_available() >= 6ULL * 1024 * 1024 * 1024, "real GGUF access refused: unchanged 6 GiB MemAvailable gate");
    fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW); require(fd >= 0, "cannot open model");
    struct stat st{}; char magic[4]{};
    if (::fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size != 40426079712LL || ::pread(fd, magic, 4, 0) != 4 || std::memcmp(magic, "GGUF", 4)) {
        ::close(fd); fd = -1; throw std::runtime_error("wrong model file identity/size; SHA verification also required by caller");
    }
    ++resources;
}
ModelFile::~ModelFile() { if (fd >= 0) { ::close(fd); --resources; } }
void ModelFile::read(const Spec & s, size_t off, void * dst, size_t n) {
    require(off <= s.bytes && n <= s.bytes - off && s.offset + s.bytes <= 40426079712ULL, "read outside tensor/model");
    exact_read(fd, dst, n, s.offset + off);
}
std::vector<float> mask(const Config & c, int p) {
    c.validate(); require(p >= 0 && p < c.context, "position outside bounded context");
    std::vector<float> result(c.slots(), -std::numeric_limits<float>::infinity());
    for (int k = 0; k <= p; ++k) if (!c.sliding() || p - k < c.window) result[k] = 0;
    return result;
}
int live_resources() { return resources.load(); }
Result execute(const Config & c, const std::vector<Spec> & specs, Source & source,
               const std::vector<float> & input, const std::vector<float> & x0,
               const KV & previous, bool capture, const std::string & fail_at) {
    if (!c.fixture || source.is_model()) q104::real("direct r99::execute");
    q104::synthetic();
    require(c.hidden<=256 && c.ffn<=512 && c.heads<=4 && c.kv_heads<=2 && c.head<=64, "QIES104 synthetic geometry budget");
    validate_specs(c, specs); validate_kv(c, previous);
    require(c.fixture != source.is_model(), "fixture/model source namespace mismatch");
    require(previous.position < c.context && input.size() == (size_t) c.hidden && x0.size() == input.size() && finite(input) && finite(x0), "invalid inputs/budget");
    Result result; Backend backend; Context weights(2 * 1024 * 1024);
    std::map<std::string, ggml_tensor *> w;
    auto short_name = [&](const Spec & s) { return s.name.substr(s.name.find('.', 4) + 1); };
    for (const auto & s : specs) {
        auto * t = ggml_new_tensor_2d(weights.p, s.type, s.columns, s.rows); require(t, "weight tensor allocation");
        w.emplace(short_name(s), t); result.weight_bytes += s.bytes;
    }
    Buffer weight_buffer(weights.p, backend.p, weight_limit + 1024 * 1024);
    fail(fail_at, "after_weight_allocation");
    std::vector<uint8_t> chunk(io_limit);
    for (const auto & s : specs) {
        for (size_t done = 0; done < s.bytes;) {
            size_t count = std::min(chunk.size(), (size_t) s.bytes - done);
            source.read(s, done, chunk.data(), count); ggml_backend_tensor_set(w.at(short_name(s)), chunk.data(), done, count); done += count;
        }
    }
    chunk.clear(); chunk.shrink_to_fit();
    fail(fail_at, "after_weight_load");
    Context graph(16 * 1024 * 1024); auto * ctx = graph.p;
    auto * g = ggml_new_graph_custom(ctx, 2048, false); require(g, "graph allocation");
    std::vector<std::pair<std::string, ggml_tensor *>> taps;
    auto tap = [&](const char * name, ggml_tensor * t) { require(t, "operator construction failed"); result.sequence.emplace_back(name); taps.emplace_back(name, t); return t; };
    auto norm = [&](ggml_tensor * x, const char * name) { return ggml_mul(ctx, ggml_rms_norm(ctx, x, c.epsilon), w.at(name)); };
    auto mm = [&](const char * name, ggml_tensor * x) { return ggml_mul_mat(ctx, w.at(name), x); };
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden, 1);
    auto * original = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.hidden, 1);
    auto * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    auto * input_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.slots(), 1);
    auto * keys = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.head, c.kv_heads, c.slots());
    auto * values = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.head, c.kv_heads, c.slots());
    auto * a = tap("attn_norm", norm(x, "attn_norm.weight"));
    auto * q = ggml_reshape_3d(ctx, mm("attn_q.weight", a), c.head, c.heads, 1);
    auto * k = ggml_reshape_3d(ctx, mm("attn_k.weight", a), c.head, c.kv_heads, 1);
    auto * mix = ggml_sigmoid(ctx, w.at("attn_vres_lambda.weight"));
    auto * vi = tap("vres_in", ggml_add(ctx, original, ggml_mul(ctx, ggml_sub(ctx, a, original), mix)));
    auto * v = tap("Vcur", ggml_reshape_3d(ctx, mm("attn_v.weight", vi), c.head, c.kv_heads, 1));
    q = tap("Qcur_normed", norm(q, "attn_q_norm.weight"));
    k = tap("Kcur_normed", norm(k, "attn_k_norm.weight"));
    if (c.sliding()) {
        q = ggml_rope_ext(ctx, q, position, nullptr, c.head, GGML_ROPE_TYPE_NEOX, 2097152, c.rope_base, 1, 0, 1, 32, 1);
        k = ggml_rope_ext(ctx, k, position, nullptr, c.head, GGML_ROPE_TYPE_NEOX, 2097152, c.rope_base, 1, 0, 1, 32, 1);
    }
    q = tap("Qcur", q); k = tap("Kcur", k);
    auto * key_slot = ggml_view_3d(ctx, keys, c.head, c.kv_heads, 1, keys->nb[1], keys->nb[2], previous.position * keys->nb[2]);
    auto * value_slot = ggml_view_3d(ctx, values, c.head, c.kv_heads, 1, values->nb[1], values->nb[2], previous.position * values->nb[2]);
    ggml_build_forward_expand(g, ggml_cpy(ctx, k, key_slot)); ggml_build_forward_expand(g, ggml_cpy(ctx, v, value_slot));
    auto * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    auto * kp = ggml_permute(ctx, keys, 0, 2, 1, 3);
    auto * vp = ggml_permute(ctx, values, 0, 2, 1, 3);
    auto * kq = ggml_mul_mat(ctx, kp, qp); require(ggml_prec_set_acc(kq, GGML_PREC_F32), "attention precision");
    kq = tap("kq_soft_max", ggml_soft_max_ext(ctx, kq, input_mask, 1.0f / std::sqrt((float) c.head), 0));
    vp = ggml_cont(ctx, ggml_transpose(ctx, vp));
    auto * attention = ggml_mul_mat(ctx, vp, kq);
    attention = ggml_permute(ctx, attention, 0, 2, 1, 3);
    attention = tap("attn_out", ggml_cont_2d(ctx, attention, c.head * c.heads, 1));
    auto * gate = tap("attn_gate", ggml_sigmoid(ctx, mm("attn_gate.weight", a)));
    attention = tap("attn_gated", ggml_mul(ctx, attention, gate));
    auto * post = tap("attn_post_norm", norm(mm("attn_output.weight", attention), "post_attention_norm.weight"));
    auto * residual = tap("ffn_inp", ggml_add(ctx, post, x));
    auto * ffn_in = tap("ffn_norm", norm(residual, "ffn_norm.weight"));
    auto * fg = mm("ffn_gate.weight", ffn_in); auto * fu = mm("ffn_up.weight", ffn_in);
    auto * active = tap("ffn_swiglu", c.clamp > 0 ? ggml_swiglu_clamp(ctx, fg, fu, c.clamp) : ggml_swiglu_split(ctx, fg, fu));
    auto * ffout = tap("ffn_out", mm("ffn_down.weight", active));
    ffout = tap("ffn_post_norm", norm(ffout, "post_ffw_norm.weight"));
    auto * out = tap("l_out", ggml_add(ctx, ffout, residual)); ggml_build_forward_expand(g, out);
    Buffer graph_buffer(ctx, backend.p, graph_limit); result.graph_bytes = ggml_backend_buffer_get_size(graph_buffer.p);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * 4); ggml_backend_tensor_set(original, x0.data(), 0, x0.size() * 4);
    int pos = previous.position; ggml_backend_tensor_set(position, &pos, 0, sizeof(pos));
    auto masks = mask(c, pos); ggml_backend_tensor_set(input_mask, masks.data(), 0, masks.size() * 4);
    KV candidate; candidate.owner = owner_of(c); candidate.key.assign((size_t) c.head * c.kv_heads * c.slots(), 0); candidate.value = candidate.key;
    if (!previous.key.empty()) { candidate.key = previous.key; candidate.value = previous.value; }
    ggml_backend_tensor_set(keys, candidate.key.data(), 0, candidate.key.size() * 2);
    ggml_backend_tensor_set(values, candidate.value.data(), 0, candidate.value.size() * 2);
    fail(fail_at, "before_compute");
    require(ggml_backend_graph_compute(backend.p, g) == GGML_STATUS_SUCCESS, "GGML compute failed");
    fail(fail_at, "after_compute");
    result.output.resize(c.hidden); ggml_backend_tensor_get(out, result.output.data(), 0, result.output.size() * 4); require(finite(result.output), "nonfinite layer output");
    ggml_backend_tensor_get(keys, candidate.key.data(), 0, candidate.key.size() * 2);
    ggml_backend_tensor_get(values, candidate.value.data(), 0, candidate.value.size() * 2); candidate.position = pos + 1; validate_kv(c, candidate);
    if (capture) for (const auto & t : taps) {
        require(t.second->type == GGML_TYPE_F32 && ggml_is_contiguous(t.second), "capture must be contiguous F32");
        auto & values_out = result.captures[t.first]; values_out.resize(ggml_nelements(t.second));
        ggml_backend_tensor_get(t.second, values_out.data(), 0, values_out.size() * 4); require(finite(values_out), "nonfinite intermediate");
    }
    fail(fail_at, "before_publish"); result.next = std::move(candidate); result.kv_bytes = (result.next.key.size() + result.next.value.size()) * 2;
    return result;
}
void save_kv(const std::string & path, const Config & c, const KV & s) {
    c.validate(); validate_kv(c, s); auto header = state_header(c, s);
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600); require(fd >= 0, "exclusive KV create failed");
    try { write_all(fd, header.data(), sizeof(header)); write_all(fd, s.key.data(), s.key.size() * 2); write_all(fd, s.value.data(), s.value.size() * 2); require(::fsync(fd) == 0, "KV fsync failed"); }
    catch (...) { ::close(fd); throw; }
    require(::close(fd) == 0, "KV close failed");
}
KV load_kv(const std::string & path, const Config & c) {
    c.validate(); std::ifstream f(path, std::ios::binary); require((bool) f, "cannot open KV"); std::array<uint64_t, 24> h{};
    f.read((char *) h.data(), sizeof(h)); require((bool) f, "truncated KV header");
    size_t count = (size_t) c.head * c.kv_heads * c.slots(); require(h[11] <= (uint64_t) c.context && (h[12] == count || (h[12] == 0 && h[11] == 0)), "KV size bound");
    KV s; s.owner = owner_of(c); s.position = (int) h[11]; s.key.resize(h[12]); s.value.resize(h[12]);
    f.read((char *) s.key.data(), s.key.size() * 2); f.read((char *) s.value.data(), s.value.size() * 2);
    require((bool) f && f.peek() == std::char_traits<char>::eof(), "truncated/extra KV bytes");
    require(h == state_header(c, s), "KV identity/checksum mismatch"); return s;
}
void Schedule::require(int l, int p) const { r99::require(p >= 0 && p < 256 && l >= 0 && l < 80 && l == layer && p == position, "layer/token schedule mismatch"); }
void Schedule::commit(int l, int p) { require(l, p); if (++layer == 80) { layer = 0; ++position; } }
}
