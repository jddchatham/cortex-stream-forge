#include "llama.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/flashnext-layer-transaction.h"
#include "../src/flashnext-selected-moe.h"
#include "../src/flashnext-shared-expert.h"
#include "../src/flashnext-segment-b.h"
#include "../src/flashnext-stock-layer0-reference.h"
#include "../src/flashnext-tensor-index.h"
#include "../src/ggml-quants.h"

#include <algorithm>
#include <dirent.h>
#include <array>
#include <cstdio>
#include <malloc.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <fstream>
#include <sstream>

static uint64_t hash_f32(const float * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { uint32_t u; std::memcpy(&u, p+i, 4); h ^= u; h *= 1099511628211ull; }
    return h;
}

static uint64_t hash_bytes(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

static bool flashnext_stream_output_q6k(const flashnext_tensor_descriptor & d,
        const std::vector<float> & x, std::vector<float> & logits) {
    if (d.type != GGML_TYPE_Q6_K || d.shape0 != 2560 || d.shape1 != 248320 || x.size() != 2560) return false;
    const std::string path = d.path;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    constexpr size_t row_bytes = 10 * sizeof(block_q6_K);
    std::vector<uint8_t> row(row_bytes);
    std::vector<float> wf(2560);
    logits.resize((size_t) d.shape1);
    for (size_t r = 0; r < logits.size(); ++r) {
        size_t done = 0;
        while (done < row.size()) {
            const ssize_t n = ::pread(fd, row.data() + done, row.size() - done,
                    (off_t) (d.offset + r * row_bytes + done));
            if (n <= 0) { ::close(fd); return false; }
            done += (size_t) n;
        }
        dequantize_row_q6_K((const block_q6_K *) row.data(), wf.data(), 2560);
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) sum += x[i] * wf[i];
        logits[r] = sum;
    }
    ::close(fd);
    return true;
}

// Independent IQ4_XS x Q8_K reference used only by the diagnostic oracle.
// These are deliberately local types and arithmetic; the production vec-dot
// implementation is not called here.
struct flashnext_q8k_ref {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};

static flashnext_q8k_ref flashnext_quantize_q8k_ref(const float * x) {
    flashnext_q8k_ref r{};
    float amax = 0.0f;
    float amax_signed = 0.0f;
    for (int i = 0; i < 256; ++i) {
        const float ax = std::fabs(x[i]);
        if (ax > amax) { amax = ax; amax_signed = x[i]; }
    }
    if (amax == 0.0f) return r;
    const float iscale = -127.0f / amax_signed;
    r.d = 1.0f / iscale;
    for (int i = 0; i < 256; ++i) {
        int q = (int) std::nearbyint(iscale * x[i]);
        q = std::max(-127, std::min(127, q));
        r.qs[i] = (int8_t) q;
    }
    for (int b = 0; b < 16; ++b) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) sum += r.qs[16*b + i];
        r.bsums[b] = (int16_t) sum;
    }
    return r;
}

static const int8_t flashnext_iq4nl_ref[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

static float flashnext_iq4xs_q8k_dot_ref(const uint8_t * wb, const flashnext_q8k_ref * y, int nblocks) {
    float sumf = 0.0f;
    const uint16_t scales_h = (uint16_t) wb[2] | ((uint16_t) wb[3] << 8);
    const uint8_t * scales_l = wb + 4;
    const uint8_t * qs = wb + 8;
    uint16_t h = scales_h;
    const float d4 = [&]() {
        uint16_t u = (uint16_t) wb[0] | ((uint16_t) wb[1] << 8);
        float f; std::memcpy(&f, &u, sizeof(uint16_t));
        // ggml_fp16_to_fp32 is intentionally not used by the scalar oracle.
        const uint32_t sign = (u >> 15) & 1, exp = (u >> 10) & 31, mant = u & 1023;
        if (exp == 0) return std::ldexp((float) mant, -24) * (sign ? -1.0f : 1.0f);
        if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
        return std::ldexp((float) (1024 + mant), (int) exp - 25) * (sign ? -1.0f : 1.0f);
    }();
    (void) nblocks;
    for (int ib = 0; ib < 8; ib += 2) {
        const int ls1 = (scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
        const int ls2 = (scales_l[ib/2] >> 4) | ((h << 2) & 0x30);
        h >>= 4;
        const flashnext_q8k_ref & q8a = y[ib/8];
        int32_t a = 0, b = 0;
        for (int j = 0; j < 16; ++j) {
            a += flashnext_iq4nl_ref[qs[j] & 0xf] * q8a.qs[ib*32 + j];
            b += flashnext_iq4nl_ref[qs[j] >> 4] * q8a.qs[ib*32 + j + 16];
        }
        sumf += d4 * q8a.d * (ls1 - 32) * (float) (a + b);
        qs += 16;
        a = b = 0;
        for (int j = 0; j < 16; ++j) {
            a += flashnext_iq4nl_ref[qs[j] & 0xf] * q8a.qs[ib*32 + j + 32];
            b += flashnext_iq4nl_ref[qs[j] >> 4] * q8a.qs[ib*32 + j + 48];
        }
        sumf += d4 * q8a.d * (ls2 - 32) * (float) (a + b);
        qs += 16;
    }
    return sumf;
}

static bool read_exact_fd(int fd, void * dst, size_t bytes, uint64_t offset) {
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::pread(fd, static_cast<uint8_t *>(dst) + done, bytes - done,
                                  static_cast<off_t>(offset + done));
        if (n <= 0) return false;
        done += (size_t) n;
    }
    return true;
}

// Independent shared-expert reference.  It intentionally has its own GGML
// context, decoded buffers, file reads, and operation graph; it never calls
// flashnext_execute_shared_expert() and never consumes tx.shared_moe.
static bool shared_expert_reference(const flashnext_tensor_index &, int layer,
        const std::string & index_path, const flashnext_layer_transaction & tx,
        std::array<float, flashnext_layer_transaction::hidden> & out, std::string & error) {
    flashnext_tensor_index local_index;
    if (!local_index.load(index_path, &error)) return false;
    flashnext_tensor_descriptor dg, du, dd, dc;
    if (!local_index.resolve_shared(layer, flashnext_projection::gate, dg, &error) ||
        !local_index.resolve_shared(layer, flashnext_projection::up, du, &error) ||
        !local_index.resolve_shared(layer, flashnext_projection::down, dd, &error) ||
        !local_index.resolve_shared(layer, flashnext_projection::shared_control, dc, &error)) return false;
    if (dg.type != 23 || du.type != 23 || dd.type != 20 || dc.type != 0) {
        error = "unexpected shared tensor quantization"; return false;
    }
    const size_t gate_bytes = 870400, down_bytes = 921600;
    std::vector<uint8_t> gate(gate_bytes), up(gate_bytes), down(down_bytes);
    std::vector<float> control(2560), hidden(2560), gate_out(640), up_out(640), act(640), down_out(2560);
    const int fd = ::open(dg.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0 || dg.path != du.path || dg.path != dd.path || dg.path != dc.path ||
        !read_exact_fd(fd, gate.data(), gate.size(), dg.offset) ||
        !read_exact_fd(fd, up.data(), up.size(), du.offset) ||
        !read_exact_fd(fd, down.data(), down.size(), dd.offset) ||
        !read_exact_fd(fd, control.data(), control.size() * sizeof(float), dc.offset)) {
        if (fd >= 0) ::close(fd);
        error = "shared weight read failed";
        return false;
    }
    ::close(fd);
    std::memcpy(hidden.data(), tx.pre_moe_hidden.data(), hidden.size() * sizeof(float));
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_context * c = be ? ggml_init({64 * 1024 * 1024, nullptr, true}) : nullptr;
    if (!c) { if (be) ggml_backend_free(be); error = "reference backend init failed"; return false; }
    ggml_tensor * wg = ggml_new_tensor_2d(c, (ggml_type) 23, 2560, 640);
    ggml_tensor * wu = ggml_new_tensor_2d(c, (ggml_type) 23, 2560, 640);
    ggml_tensor * wd = ggml_new_tensor_2d(c, (ggml_type) 20, 640, 2560);
    ggml_tensor * wc = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2560, 1);
    ggml_tensor * xi = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2560, 1);
    ggml_tensor * g = ggml_mul_mat(c, wg, xi), * u = ggml_mul_mat(c, wu, xi);
    ggml_tensor * y = ggml_mul_mat(c, wd, ggml_mul(c, ggml_silu(c, g), u));
    ggml_tensor * control_out = ggml_sigmoid(c, ggml_mul_mat(c, wc, xi));
    ggml_tensor * result = ggml_mul(c, y, control_out);
    ggml_cgraph * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!buf) { ggml_free(c); ggml_backend_free(be); error = "reference allocation failed"; return false; }
    wg->data = gate.data(); wu->data = up.data(); wd->data = down.data(); wc->data = control.data();
    ggml_backend_tensor_set(xi, hidden.data(), 0, hidden.size() * sizeof(float));
    const bool ok = ggml_backend_graph_compute(be, graph) == GGML_STATUS_SUCCESS;
    if (ok) ggml_backend_tensor_get(result, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf); ggml_free(c); ggml_backend_free(be);
    return ok;
}

static void stock_graph_b_reference(const flashnext_layer_transaction & tx, std::array<float, flashnext_layer_transaction::hidden * flashnext_layer_transaction::streams> & out) {
    // Stock build_hc_combine semantics for one token: w=2*sigmoid(inject/hc),
    // then add the same MoE vector to each HC stream with its stream weight.
    constexpr float hc = 4.0f;
    for (size_t s = 0; s < flashnext_layer_transaction::streams; ++s) {
        const float w = 2.0f / (1.0f + std::exp(-tx.inject[s] / hc));
        for (size_t h = 0; h < flashnext_layer_transaction::hidden; ++h) {
            out[s * flashnext_layer_transaction::hidden + h] =
                tx.residual_hc[s * flashnext_layer_transaction::hidden + h] + tx.complete_moe[h] * w;
        }
    }
}

static bool known_pattern_backend_test() {
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, 0);
    if (!be) return false;
    ggml_init_params ip{ 256*1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    if (!c) { ggml_backend_free(be); return false; }
    ggml_tensor * t = ggml_new_tensor_1d(c, GGML_TYPE_F32, 16);
    ggml_tensor * u = ggml_new_tensor_1d(c, GGML_TYPE_F32, 16);
    ggml_tensor * y = ggml_mul(c, t, u);
    ggml_cgraph * graph = ggml_new_graph(c);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(c, be);
    std::array<uint32_t, 16> src{}, dst{}; std::array<float, 16> a{}, aa{}, yy{};
    for (size_t i = 0; i < src.size(); ++i) { src[i] = 0x3f000000u + uint32_t(i * 0x010101u); a[i] = (float) i + 1.0f; aa[i] = 2.0f; }
    bool ok = b != nullptr;
    if (ok) ggml_backend_tensor_set(t, src.data(), 0, sizeof(src));
    if (ok) ggml_backend_tensor_get(t, dst.data(), 0, sizeof(dst));
    ok = ok && std::memcmp(src.data(), dst.data(), sizeof(src)) == 0;
    if (ok) {
        ggml_backend_tensor_set(t, a.data(), 0, sizeof(a));
        ggml_backend_tensor_set(u, aa.data(), 0, sizeof(aa));
        ok = ggml_backend_graph_compute(be, graph) == GGML_STATUS_SUCCESS;
        if (ok) ggml_backend_tensor_get(y, yy.data(), 0, sizeof(yy));
        for (size_t i = 0; ok && i < yy.size(); ++i) ok = yy[i] == a[i] * aa[i];
    }
    std::printf("KNOWN_PATTERN_TEST=%s bytes=%zu hash=%016llx\n", ok ? "PASS" : "FAIL",
        sizeof(src), (unsigned long long) hash_bytes(reinterpret_cast<const uint8_t *>(dst.data()), sizeof(dst)));
    if (b) ggml_backend_buffer_free(b);
    ggml_free(c); ggml_backend_free(be);
    return ok;
}

static uint64_t proc_value_kb(const char * key) {
    std::ifstream f("/proc/self/status"); std::string k, v, unit;
    while (f >> k >> v >> unit) if (k == std::string(key) + ":") return std::strtoull(v.c_str(), nullptr, 10) * 1024;
    return 0;
}
static uint64_t file_u64(const std::string & p) {
    std::ifstream f(p); std::string s; if (!(f >> s) || s == "max") return UINT64_MAX;
    return std::strtoull(s.c_str(), nullptr, 10);
}
static uint64_t mem_available() {
    std::ifstream f("/proc/meminfo"); std::string k, v, unit;
    while (f >> k >> v >> unit) if (k == "MemAvailable:") return std::strtoull(v.c_str(), nullptr, 10) * 1024;
    return 0;
}
static std::string cgroup_file(const char * name) {
    std::ifstream f("/proc/self/cgroup"); std::string line, rel;
    while (std::getline(f, line)) if (line.rfind("0::", 0) == 0) rel = line.substr(3);
    return std::string("/sys/fs/cgroup") + rel + "/" + name;
}
static void print_preflight(llama_model * model) {
    (void) model;
    struct rusage ru{}; getrusage(RUSAGE_SELF, &ru);
    const uint64_t rss = proc_value_kb("VmRSS"), swap = proc_value_kb("VmSwap");
    const uint64_t mem = file_u64(cgroup_file("memory.current"));
    const uint64_t swp = file_u64(cgroup_file("memory.swap.current"));
    const uint64_t mmax = file_u64(cgroup_file("memory.max"));
    const uint64_t smax = file_u64(cgroup_file("memory.swap.max"));
    const uint64_t layer = 128ull * 1024 * 1024, ple = 16ull * 160 * 4, hidden = 10240ull * 4;
    const uint64_t recurrent = 124ull * 1024 * 1024;
    // The private scheduler does not expose reservation byte counts. Use a
    // conservative fail-closed allowance rather than reporting zero.
    const uint64_t backend = 256ull * 1024 * 1024, scheduler = 256ull * 1024 * 1024;
    const uint64_t estimate = rss + layer + ple + hidden + recurrent + backend + scheduler;
    const uint64_t margin = 512ull * 1024 * 1024;
    const bool limits_ok = (mmax == UINT64_MAX || estimate + margin < mmax) &&
                           estimate + margin < mem_available();
    std::printf("MEMORY_PREFLIGHT_IMPLEMENTED=YES\nCGROUP_LIMITS memory.max=%s memory.swap.max=%s\n",
        mmax == UINT64_MAX ? "max" : std::to_string(mmax).c_str(), smax == UINT64_MAX ? "max" : std::to_string(smax).c_str());
    std::printf("PROCESS_RSS_BEFORE=%llu PROCESS_SWAP_BEFORE=%llu CGROUP_MEMORY_CURRENT=%llu CGROUP_SWAP_CURRENT=%llu\n",
        (unsigned long long) rss, (unsigned long long) swap, (unsigned long long) mem, (unsigned long long) swp);
    std::printf("LAYER_RESIDENCY_BYTES=%llu PLE_BUFFER_BYTES=%llu OWNED_HIDDEN_STATE_BYTES=%llu RECURRENT_STATE_BYTES=%llu\n",
        (unsigned long long) layer, (unsigned long long) ple, (unsigned long long) hidden, (unsigned long long) recurrent);
    std::printf("BACKEND_RESERVED_BYTES=%llu SCHEDULER_RESERVED_BYTES=%llu TOTAL_ESTIMATED_PEAK_BYTES=%llu SAFETY_MARGIN_BYTES=%llu\n",
        (unsigned long long) backend, (unsigned long long) scheduler, (unsigned long long) estimate, (unsigned long long) margin);
    std::printf("CGROUP_MEMORY_HEADROOM=%llu PHYSICAL_MEMORY_AVAILABLE=%llu\n",
        (unsigned long long) (mmax == UINT64_MAX ? 0 : mmax - std::min(mmax, estimate + margin)),
        (unsigned long long) mem_available());
    std::printf("PREFLIGHT_GATE=%s PREFLIGHT_TEST_EXECUTED=YES PEAK_RSS=%llu PEAK_SWAP=%llu\n",
        limits_ok ? "PASS" : "FAIL", (unsigned long long) ru.ru_maxrss * 1024ull, (unsigned long long) swap);
}
static void export_blob(const char * name, int pos, const void * p, size_t n, const char * shape, const char * dtype) {
    const char * dir = std::getenv("FLASHNEXT_COMPARE_DIR"); if (!dir) return;
    mkdir(dir, 0755); char path[1024]; std::snprintf(path, sizeof(path), "%s/cortex_%s_p%d.bin", dir, name, pos);
    FILE * f = std::fopen(path, "wb"); if (!f) return; std::fwrite(p, 1, n, f); std::fclose(f);
    std::string mp = std::string(dir) + "/manifest.tsv"; FILE * m = std::fopen(mp.c_str(), "a"); if (!m) return;
    std::fprintf(m, "cortex\t%s\t%d\t%s\t%s\t%zu\t%zu\t%016llx\n", name, pos, shape, dtype, n/4, n, (unsigned long long) hash_bytes((const uint8_t *) p, n)); std::fclose(m);
}
static void export_state_components(llama_context * ctx, int pos) {
    const char * dir = std::getenv("FLASHNEXT_COMPARE_DIR"); if (!dir) return;
    for (bool conv : { false, true }) {
        std::vector<uint8_t> b(8*1024*1024);
        size_t n = ctx->debug_copy_recurrent_state(0, conv, b.data(), b.size());
        if (n > b.size()) { b.resize(n); n = ctx->debug_copy_recurrent_state(0, conv, b.data(), b.size()); }
        if (n == 0 || n > b.size()) continue;
        export_blob(conv ? "r_l0" : "s_l0", pos, b.data(), n, "opaque", "f32");
    }
}

static std::vector<uint8_t> capture_seq_state(llama_context * ctx) {
    const size_t n = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> state(n);
    if (n != 0 && llama_state_seq_get_data(ctx, state.data(), n, 0) != n) state.clear();
    return state;
}

static bool flashnext_write_state_file(const std::string & path, const std::vector<uint8_t> & state) {
    if (state.empty()) return false;
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    size_t done = 0;
    while (done < state.size()) {
        const ssize_t n = ::write(fd, state.data() + done, state.size() - done);
        if (n <= 0) { ::close(fd); return false; }
        done += (size_t) n;
    }
    const bool durable = ::fsync(fd) == 0;
    // State is immediately durable and will be read by a later transaction;
    // do not retain all completed snapshots in the protected page cache.
    (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    return durable;
}

static bool flashnext_read_state_file(const std::string & path, std::vector<uint8_t> & state) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const std::streamoff size = in.tellg();
    if (size <= 0) return false;
    in.seekg(0, std::ios::beg);
    state.resize((size_t) size);
    in.read(reinterpret_cast<char *>(state.data()), size);
    return in.good() || in.eof() && in.gcount() == size;
}

static std::vector<uint8_t> capture_layer_recurrent_state(llama_context * ctx, int layer) {
    std::vector<uint8_t> out;
    for (bool convolution : { false, true }) {
        std::vector<uint8_t> part(8 * 1024 * 1024);
        size_t n = ctx->debug_copy_recurrent_state(layer, convolution, part.data(), part.size());
        if (n > part.size()) { part.resize(n); n = ctx->debug_copy_recurrent_state(layer, convolution, part.data(), part.size()); }
        if (n == 0 || n > part.size()) return {};
        out.insert(out.end(), part.begin(), part.begin() + n);
    }
    return out;
}

struct layer_transaction_result {
    int decode_status = -1;
    std::vector<uint8_t> state_before;
    std::array<float, flashnext_layer_transaction::hidden * flashnext_layer_transaction::streams> output{};
    uint64_t output_hash = 0;
    uint64_t graph_a_hash = 0;
    uint64_t sparse_output_hash = 0;
    bool state_committed = false;
    bool rollback_restored = false;
    bool graph_a_ok = false;
    bool sparse_moe_ok = false;
    bool graph_b_ok = false;
    const char * failure_stage = nullptr;
    std::array<int32_t, flashnext_layer_transaction::experts_used> selected_ids{};
    uint64_t unselected_expert_reads = 0;
};

// Metadata-derived description of the model layer used by the experimental
// transaction bridge.  This is deliberately non-owning: model tensors and
// recurrent state belong to llama_model/llama_context and must not outlive
// them.  The transaction owns only its activation and rollback buffers.
struct flashnext_layer_descriptor {
    uint32_t index = 0;
    bool recurrent = false;
    bool ple = false;
    bool supported = false;
    const llama_layer * model_layer = nullptr;
    ggml_tensor * router = nullptr;
    ggml_tensor * gate_experts = nullptr;
    ggml_tensor * up_experts = nullptr;
    ggml_tensor * down_experts = nullptr;
    ggml_tensor * shared_gate = nullptr;
    ggml_tensor * shared_up = nullptr;
    ggml_tensor * shared_down = nullptr;
};

static bool resolve_flashnext_layer_descriptor(
        llama_context * ctx, uint32_t layer, flashnext_layer_descriptor & out, std::string & error) {
    if (!ctx) { error = "missing context"; return false; }
    const llama_model & model = ctx->get_model();
    if (layer >= model.hparams.n_layer()) { error = "layer index out of range"; return false; }

    const llama_layer & ml = model.layers[layer];
    out.index = layer;
    out.recurrent = model.hparams.is_recr(layer);
    out.ple = model.hparams.is_ple(layer);
    out.model_layer = &ml;
    out.router = ml.ffn_gate_inp;
    out.gate_experts = ml.ffn_gate_exps;
    out.up_experts = ml.ffn_up_exps;
    out.down_experts = ml.ffn_down_exps;
    out.shared_gate = ml.ffn_gate_inp_shexp;
    out.shared_up = ml.ffn_up_shexp;
    out.shared_down = ml.ffn_down_shexp;

    // The sparse bridge is layer-kind agnostic after the stock Graph-A
    // boundary. Full-attention layers use KV state owned by the context;
    // run_layer_transaction snapshots that context for failure rollback.
    if (!out.router || !out.gate_experts || !out.up_experts || !out.down_experts) {
        error = "layer is missing sparse-MoE tensors"; return false;
    }
    if (!out.shared_gate || !out.shared_up || !out.shared_down) {
        error = "layer is missing shared-expert tensors"; return false;
    }
    out.supported = true;
    return true;
}

// Explicit ownership contract for the existing transaction.  GGML diagnostic
// tensors and the llama context are borrowed and valid only through the
// synchronized transaction.  The tensor index is also borrowed from the
// runner's model/index lifetime.  Transaction values, rollback bytes, and
// returned output are caller-owned storage and survive graph destruction.
struct flashnext_transaction_context {
    llama_context * ctx = nullptr;                         // borrowed
    flashnext_tensor_index * tensor_index = nullptr;       // borrowed
    flashnext_layer_descriptor layer_desc;                 // borrowed model handles
    std::vector<uint8_t> staged_state;                     // owned
    flashnext_layer_transaction values;                    // owned buffers
    std::array<float, flashnext_layer_transaction::hidden * flashnext_layer_transaction::streams> output{}; // owned
    std::array<int32_t, flashnext_layer_transaction::experts_used> selected_ids{}; // owned result metadata
    std::array<float, flashnext_layer_transaction::hidden> moe_input{};
    std::array<float, 512> router_logits{};
    std::array<float, flashnext_layer_transaction::hidden * flashnext_layer_transaction::streams> residual_hc{};
    std::array<float, flashnext_layer_transaction::streams> injection{};
    bool state_committed = false;
};

// Optional coordinator boundary.  The callback owns the vault adapter and
// receives the already-synchronized context; nullptr/disabled preserves the
// historical path exactly and performs no allocation or I/O.
struct flashnext_state_vault_hooks {
    bool enabled = false;
    void * userdata = nullptr;
    bool (*sync_export_commit)(void * userdata, llama_context * ctx,
                               llama_seq_id sequence_id, uint32_t layer,
                               int position, uint64_t generation) = nullptr;
};

static bool capture_transaction_router(
        flashnext_transaction_context & txc, llama_context * ctx, const flashnext_layer_descriptor & layer) {
    const std::string suffix = "-" + std::to_string(layer.index);
    const size_t router_bytes = ctx->debug_copy_flashnext_tensor(("diag_router_logits" + suffix).c_str(),
        txc.router_logits.data(), sizeof(txc.router_logits));
    const size_t residual_bytes = ctx->debug_copy_flashnext_tensor(("diag_hc_residual" + suffix).c_str(),
        txc.residual_hc.data(), sizeof(txc.residual_hc));
    const size_t inject_bytes = ctx->debug_copy_flashnext_tensor(("diag_hc_inject" + suffix).c_str(),
        txc.injection.data(), sizeof(txc.injection));
    const size_t moe_bytes = ctx->debug_copy_flashnext_tensor(("diag_moe_input" + suffix).c_str(),
        txc.moe_input.data(), sizeof(txc.moe_input));
    const uint64_t destination_byte_hash = hash_bytes(
        reinterpret_cast<const uint8_t *>(txc.moe_input.data()), sizeof(txc.moe_input));
    std::printf("CORTEX_MOE_DEST_AFTER_ASSIGN moe=%p hash=%016llx\n",
        (void *) txc.moe_input.data(),
        (unsigned long long) hash_f32(txc.moe_input.data(), txc.moe_input.size()));
    std::printf("CORTEX_MOE_DEST_BYTE_HASH hash=%016llx\n",
        (unsigned long long) destination_byte_hash);
    std::printf("CORTEX_ROUTER_HASH byte=%016llx f32=%016llx bytes=%zu\n",
        (unsigned long long) hash_bytes(reinterpret_cast<const uint8_t *>(txc.router_logits.data()), sizeof(txc.router_logits)),
        (unsigned long long) hash_f32(txc.router_logits.data(), txc.router_logits.size()),
        sizeof(txc.router_logits));
    std::printf("CORTEX_GRAPH_B_HANDOFF residual_byte=%016llx inject_byte=%016llx residual_bytes=%zu inject_bytes=%zu\n",
        (unsigned long long) hash_bytes(reinterpret_cast<const uint8_t *>(txc.residual_hc.data()), sizeof(txc.residual_hc)),
        (unsigned long long) hash_bytes(reinterpret_cast<const uint8_t *>(txc.injection.data()), sizeof(txc.injection)),
        sizeof(txc.residual_hc), sizeof(txc.injection));
    std::fflush(stdout);
    if (moe_bytes != sizeof(txc.moe_input) || router_bytes != sizeof(txc.router_logits) ||
        residual_bytes != sizeof(txc.residual_hc) || inject_bytes != sizeof(txc.injection)) return false;
    uint64_t producer_execution_id = 0, producer_hash = 0;
    const bool producer_valid = ctx->debug_diag_moe_provenance(producer_execution_id, producer_hash);
    std::printf("CORTEX_MOE_COPY layer=%u position=%d sequence=0 producer_execution_id=%llu "
        "producer_byte_hash=%016llx destination_byte_hash=%016llx destination_f32_hash=%016llx "
        "bytes=%zu producer_valid=%s\n",
        layer.index, txc.values.position, (unsigned long long) producer_execution_id,
        (unsigned long long) producer_hash,
        (unsigned long long) hash_bytes(reinterpret_cast<const uint8_t *>(txc.moe_input.data()), sizeof(txc.moe_input)),
        (unsigned long long) hash_f32(txc.moe_input.data(), txc.moe_input.size()),
        sizeof(txc.moe_input), producer_valid ? "YES" : "NO");
    std::fflush(stdout);
    for (float x : txc.moe_input) if (!std::isfinite(x)) return false;
    for (float x : txc.router_logits) if (!std::isfinite(x)) return false;
    for (float x : txc.residual_hc) if (!std::isfinite(x)) return false;
    for (float x : txc.injection) if (!std::isfinite(x)) return false;
    return true;
}

static bool capture_hc_attention_boundaries(llama_context * ctx, int layer) {
    const std::string suffix = "-" + std::to_string(layer);
    const char * names[] = { "hc_attn_input", "hc_rms_unweighted", "hc_rms_weighted", "hc_attn_down", "hc_attn_gate", "hc_attn_mixed", "hc_attn_inject" };
    bool all = true;
    for (const char * base : names) {
        const std::string name = std::string(base) + suffix;
        const std::string fallback = std::strcmp(base, "hc_rms_weighted") == 0 ?
            std::string("hc_attn_norm") + suffix : name;
        const char * lookup = name.c_str();
        size_t bytes = ctx->debug_copy_flashnext_tensor(lookup, nullptr, 0);
        if (bytes == 0 && fallback != name) {
            lookup = fallback.c_str();
            bytes = ctx->debug_copy_flashnext_tensor(lookup, nullptr, 0);
        }
        std::vector<uint8_t> owned(bytes);
        const size_t got = bytes ? ctx->debug_copy_flashnext_tensor(lookup, owned.data(), owned.size()) : 0;
        const bool ok = got == bytes && bytes != 0;
        const bool required = std::strcmp(base, "hc_attn_input") == 0 || std::strcmp(base, "hc_rms_weighted") == 0 || std::strcmp(base, "hc_attn_down") == 0;
        all = all && (!required || ok);
        std::printf("HC_CAPTURE name=%s layer=%d bytes=%zu status=%s hash=%016llx\n",
            name.c_str(), layer, bytes, ok ? "PASS" : "NOT_CAPTURED",
            (unsigned long long) hash_bytes(owned.data(), owned.size()));
    }
    return all;
}

// Independent HC arithmetic reference. It reads only the authoritative
// captured input and the four layer-specific GGUF tensors; no production HC
// tensor or helper is used as an oracle input.
static bool hc_attention_reference(llama_context * ctx, const flashnext_tensor_index & index,
        int layer, std::string & error) {
    flashnext_tensor_descriptor dn, dd, du, di;
    if (!index.resolve_hc_attention(layer, flashnext_projection::hc_norm, dn, &error) ||
        !index.resolve_hc_attention(layer, flashnext_projection::hc_down, dd, &error) ||
        !index.resolve_hc_attention(layer, flashnext_projection::hc_up, du, &error) ||
        !index.resolve_hc_attention(layer, flashnext_projection::hc_inject, di, &error)) return false;
    std::printf("HC_INDEX layer=%d norm=[%lld,%lld,%lld] type=%d down=[%lld,%lld,%lld] type=%d up=[%lld,%lld,%lld] type=%d inject=[%lld,%lld,%lld] type=%d\n",
        layer, (long long)dn.shape0,(long long)dn.shape1,(long long)dn.shape2,dn.type,
        (long long)dd.shape0,(long long)dd.shape1,(long long)dd.shape2,dd.type,
        (long long)du.shape0,(long long)du.shape1,(long long)du.shape2,du.type,
        (long long)di.shape0,(long long)di.shape1,(long long)di.shape2,di.type);
    if (dn.type != GGML_TYPE_F32 || dn.shape0 != 10240 || dd.shape0 != 10240 || dd.shape1 != 320 ||
        du.shape0 != 320 || du.shape1 != 10240 || di.shape0 != 10240 || di.shape1 != 4) {
        error = "unexpected HC descriptor dimensions/types"; return false;
    }
    std::vector<float> x(10240), actual_rms(10240), actual_norm(10240), actual_down_input(10240), actual_down_base(320), actual_down(320), actual_gate(10240), actual_mixed(2560), actual_inject(4);
    auto copy = [&](const char * base, void * p, size_t n) {
        const std::string name = std::strcmp(base, "hc_down_base") == 0 ? std::string(base) : std::string(base) + "-" + std::to_string(layer);
        const std::string fallback = std::strcmp(base, "hc_rms_weighted") == 0 ?
            "hc_attn_norm-" + std::to_string(layer) : name;
        const char * lookup = name.c_str();
        size_t available = ctx->debug_copy_flashnext_tensor(lookup, nullptr, 0);
        if (available == 0 && fallback != name) {
            lookup = fallback.c_str();
            available = ctx->debug_copy_flashnext_tensor(lookup, nullptr, 0);
        }
        return available == n && ctx->debug_copy_flashnext_tensor(lookup, p, n) == n;
    };
    if (!copy("hc_attn_input", x.data(), x.size()*sizeof(float)) ||
        !copy("hc_rms_weighted", actual_norm.data(), actual_norm.size()*sizeof(float)) ||
        !copy("hc_attn_down", actual_down.data(), actual_down.size()*sizeof(float)) ||
        !copy("hc_attn_gate", actual_gate.data(), actual_gate.size()*sizeof(float) * 1) ||
        !copy("hc_attn_mixed", actual_mixed.data(), actual_mixed.size()*sizeof(float)) ||
        !copy("hc_attn_inject", actual_inject.data(), actual_inject.size()*sizeof(float))) {
        error = "authoritative HC capture unavailable"; return false;
    }
    const bool have_down_base = copy("hc_down_base", actual_down_base.data(), actual_down_base.size()*sizeof(float));
    std::printf("HC_DOWN_BASE_CAPTURE=%s base_hash=%016llx final_hash=%016llx\n",
        have_down_base ? "PASS" : "NOT_CAPTURED",
        (unsigned long long) hash_f32(actual_down_base.data(), actual_down_base.size()),
        (unsigned long long) hash_f32(actual_down.data(), actual_down.size()));
    const bool have_down_input = copy("hc_down_xn", actual_down_input.data(), actual_down_input.size()*sizeof(float));
    if (!have_down_input) {
        error = "authoritative HC down MUL_MAT input capture unavailable"; return false;
    }
    std::printf("HC_DOWN_PROVENANCE input_hash=%016llx norm_capture_hash=%016llx elementwise_equal=%s\n",
        (unsigned long long) hash_f32(actual_down_input.data(), actual_down_input.size()),
        (unsigned long long) hash_f32(actual_norm.data(), actual_norm.size()),
        std::memcmp(actual_down_input.data(), actual_norm.data(), actual_norm.size() * sizeof(float)) == 0 ? "YES" : "NO");
    const int fd = ::open(dn.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0 || dd.path != dn.path || du.path != dn.path || di.path != dn.path) {
        if (fd >= 0) ::close(fd); error = "HC shard open failed"; return false;
    }
    std::vector<uint8_t> bn(dn.bytes), bd(dd.bytes), bu(du.bytes), bi(di.bytes);
    const bool read_ok = read_exact_fd(fd, bn.data(), bn.size(), dn.offset) &&
        read_exact_fd(fd, bd.data(), bd.size(), dd.offset) && read_exact_fd(fd, bu.data(), bu.size(), du.offset) &&
        read_exact_fd(fd, bi.data(), bi.size(), di.offset);
    ::close(fd);
    if (!read_ok) { error = "HC weight read failed"; return false; }
    const float * norm_raw = reinterpret_cast<const float *>(bn.data());
    float norm_min = norm_raw[0], norm_max = norm_raw[0]; size_t norm_nonfinite = 0;
    for (size_t i = 0; i < 10240; ++i) {
        norm_min = std::min(norm_min, norm_raw[i]); norm_max = std::max(norm_max, norm_raw[i]);
        norm_nonfinite += !std::isfinite(norm_raw[i]);
    }
    std::printf("HC_NORM_WEIGHT_SOURCE shard=%s offset=%llu bytes=%zu type=%d shape=[%lld] hash=%016llx min=%.9g max=%.9g nonfinite=%zu\n",
        dn.shard.c_str(), (unsigned long long) dn.offset, bn.size(), dn.type,
        (long long) dn.shape0, (unsigned long long) hash_bytes(bn.data(), bn.size()), norm_min, norm_max, norm_nonfinite);
    std::array<float,10240> scalar_norm{};
    for (size_t s = 0; s < 4; ++s) {
        double ss = 0.0; for (size_t h = 0; h < 2560; ++h) { const float v = x[s*2560+h]; ss += (double) v * v; }
        const float inv = 1.0f / std::sqrt((float) (ss / 2560.0) + ctx->get_model().hparams.f_norm_rms_eps);
        std::printf("HC_NORM_STREAM stream=%zu sumsq=%.9g inv=%.9g first_x=%.9g first_w=%.9g first_actual=%.9g\n",
            s, ss, inv, x[s*2560], norm_raw[s*2560], actual_norm[s*2560]);
        for (size_t h = 0; h < 2560; ++h) scalar_norm[s*2560+h] = x[s*2560+h] * inv * norm_raw[s*2560+h];
    }
    std::printf("HC_NORM_SCALAR hash=%016llx first=%.9g\n",
        (unsigned long long) hash_f32(scalar_norm.data(), scalar_norm.size()), scalar_norm[0]);
    std::printf("HC_NORM_DIAG x0=%.9g x2560=%.9g w0=%.9g w2560=%.9g ref0=%.9g ref2560=%.9g actual0=%.9g actual2560=%.9g\n",
        x[0], x[2560], norm_raw[0], norm_raw[2560], scalar_norm[0], scalar_norm[2560], actual_norm[0], actual_norm[2560]);
    std::array<float,10240> scalar_rms{};
    for (size_t i = 0; i < scalar_rms.size(); ++i) scalar_rms[i] = scalar_norm[i] / norm_raw[i];
    float scalar_abs = 0; double scalar_ss = 0;
    for (size_t i = 0; i < scalar_norm.size(); ++i) {
        const float d = std::fabs(scalar_norm[i] - actual_norm[i]);
        scalar_abs = std::max(scalar_abs, d); scalar_ss += (double) d * d;
    }
    std::printf("UNWEIGHTED_NORM_MAX_ABS=NOT_CAPTURED UNWEIGHTED_NORM_RMS=NOT_CAPTURED UNWEIGHTED_NORM_PASS=NOT_RUN_FUSED\n");
    std::printf("SCALAR_NORM_MAX_ABS=%.9g SCALAR_NORM_RMS=%.9g SCALAR_NORM_PASS=%s\n",
        scalar_abs, std::sqrt(scalar_ss / scalar_norm.size()), scalar_abs <= 1e-5f ? "YES" : "NO");
    for (size_t i = 0; i < scalar_norm.size(); ++i) {
        if (std::fabs(scalar_norm[i] - actual_norm[i]) > 1e-5f) {
            std::printf("HC_NORM_FIRST_MISMATCH index=%zu stream=%zu feature=%zu x=%.9g w=%.9g ref=%.9g actual=%.9g\n",
                i, i / 2560, i % 2560, x[i], norm_raw[i], scalar_norm[i], actual_norm[i]);
            break;
        }
    }
    std::vector<flashnext_q8k_ref> q8_input(40);
    // ggml_mul_mat(w_down, xn) consumes the weighted RMSNorm tensor (xn),
    // not the pre-normalization HC input.  Use that authoritative owned
    // capture for the Q8_K activation quantization.
    for (size_t b = 0; b < q8_input.size(); ++b) q8_input[b] = flashnext_quantize_q8k_ref(actual_down_input.data() + b*256);
    std::vector<float> independent_down(320);
    for (int row = 0; row < 320; ++row) {
        independent_down[row] = 0.0f;
        for (int b = 0; b < 40; ++b) {
            independent_down[row] += flashnext_iq4xs_q8k_dot_ref(bd.data() + (size_t) row * 40 * 136 + (size_t) b * 136,
                                                                  &q8_input[b], 1);
        }
    }
    if (std::getenv("CORTEX_FLASHNEXT_IQ4XS_TRACE") != nullptr) {
        std::printf("DOWN_FORENSIC layer=%d position=0 input_elements=%zu input_hash=%016llx input_first16=",
            layer, actual_down_input.size(), (unsigned long long) hash_f32(actual_down_input.data(), actual_down_input.size()));
        for (size_t i = 0; i < 16; ++i) std::printf("%s%.9g", i ? "," : "", actual_down_input[i]);
        std::printf(" input_last16=");
        for (size_t i = actual_down_input.size() - 16; i < actual_down_input.size(); ++i) std::printf("%s%.9g", i == actual_down_input.size() - 16 ? "" : ",", actual_down_input[i]);
        std::printf(" weight_row0_offset=0 row0_bytes=%d row0_hash=%016llx first_block_offset=0 last_block_offset=%d\n",
            40 * 136, (unsigned long long) hash_bytes(bd.data(), 40 * 136), 39 * 136);
        for (int b : {0, 1, 20, 39}) {
            const auto & q = q8_input[b];
            std::printf("Q8K_BLOCK block=%d d=%.9g q0=%d q1=%d q255=%d bsum0=%d bsum15=%d hash=%016llx\n",
                b, q.d, (int) q.qs[0], (int) q.qs[1], (int) q.qs[255], (int) q.bsums[0], (int) q.bsums[15],
                (unsigned long long) hash_bytes(reinterpret_cast<const uint8_t *>(&q), sizeof(q)));
        }
        float traced = 0.0f;
        for (int b = 0; b < 40; ++b) {
            const float part = flashnext_iq4xs_q8k_dot_ref(bd.data() + (size_t) b * 136, &q8_input[b], 1);
            traced += part;
            std::printf("ROW0_BLOCK block=%d contribution=%.9g cumulative=%.9g\n", b, part, traced);
        }
        std::printf("ROW0_RESULT reference=%.9g captured=%.9g captured_hash=%016llx\n",
            independent_down[0], actual_down[0], (unsigned long long) hash_f32(actual_down.data(), actual_down.size()));
    }
    std::printf("DOWN_IQ4XS_Q8K_REFERENCE hash=%016llx first=%.9g\n",
        (unsigned long long) hash_f32(independent_down.data(), independent_down.size()), independent_down[0]);
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_context * c = be ? ggml_init({128ull*1024*1024, nullptr, true}) : nullptr;
    if (!c) { if (be) ggml_backend_free(be); error = "HC reference backend init failed"; return false; }
    ggml_tensor * tn = ggml_new_tensor_1d(c, GGML_TYPE_F32, 10240);
    ggml_tensor * td = ggml_new_tensor_2d(c, (ggml_type) dd.type, 10240, 320);
    ggml_tensor * tu = ggml_new_tensor_2d(c, (ggml_type) du.type, 320, 10240);
    ggml_tensor * ti = ggml_new_tensor_2d(c, (ggml_type) di.type, 10240, 4);
    ggml_tensor * tx = ggml_new_tensor_2d(c, GGML_TYPE_F32, 10240, 1);
    ggml_tensor * norm = ggml_mul(c, ggml_rms_norm(c, tx, ctx->get_model().hparams.f_norm_rms_eps), tn);
    ggml_tensor * down_raw = ggml_mul_mat(c, td, norm);
    ggml_tensor * lo = ggml_silu(c, ggml_scale(c, down_raw, 0.25f));
    ggml_tensor * gate = ggml_mul_mat(c, tu, lo);
    ggml_tensor * gated = ggml_mul(c, norm, ggml_sigmoid(c, gate));
    ggml_tensor * inj = ggml_mul_mat(c, ti, norm);
    ggml_cgraph * graph = ggml_new_graph(c);
    ggml_build_forward_expand(graph, gated); ggml_build_forward_expand(graph, inj); ggml_build_forward_expand(graph, down_raw);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!buf) { ggml_free(c); ggml_backend_free(be); error = "HC reference allocation failed"; return false; }
    tn->data = bn.data(); td->data = bd.data(); tu->data = bu.data(); ti->data = bi.data();
    ggml_backend_tensor_set(tn, bn.data(), 0, bn.size());
    ggml_backend_tensor_set(td, bd.data(), 0, bd.size());
    ggml_backend_tensor_set(tu, bu.data(), 0, bu.size());
    ggml_backend_tensor_set(ti, bi.data(), 0, bi.size());
    ggml_backend_tensor_set(tx, x.data(), 0, x.size()*sizeof(float));
    const bool computed = ggml_backend_graph_compute(be, graph) == GGML_STATUS_SUCCESS;
    if (!computed) { ggml_backend_buffer_free(buf); ggml_free(c); ggml_backend_free(be); error = "HC reference compute failed"; return false; }
    std::vector<float> rn(10240), rd(320), rg(10240), rgate(10240), ri(4);
    ggml_backend_tensor_get(norm, rn.data(), 0, rn.size()*sizeof(float));
    ggml_backend_tensor_get(down_raw, rd.data(), 0, rd.size()*sizeof(float));
    ggml_backend_tensor_get(gate, rg.data(), 0, rg.size()*sizeof(float));
    ggml_backend_tensor_get(gated, rgate.data(), 0, rgate.size()*sizeof(float));
    ggml_backend_tensor_get(inj, ri.data(), 0, ri.size()*sizeof(float));
    std::array<float,2560> rm{};
    for (size_t h=0; h<2560; ++h) for (size_t s=0; s<4; ++s) rm[h] += rgate[s*2560+h] * 0.25f;
    bool all_pass = true;
    auto report = [&](const char * label, const float * ref, const float * act, size_t n) {
        float ma=0, mr=0; double ss=0; size_t first=n; size_t nf=0;
        for (size_t i=0;i<n;++i) { nf += !std::isfinite(ref[i]); const float d=std::fabs(ref[i]-act[i]);
            if (d && first==n) first=i; ma=std::max(ma,d); mr=std::max(mr,d/std::max(std::fabs(ref[i]),1e-30f)); ss+=(double)d*d; }
        std::printf("HC_ORACLE name=%s pass=%s max_abs=%.9g max_rel=%.9g rms=%.9g first=%zu nonfinite=%zu ref_hash=%016llx actual_hash=%016llx\n",
            label, ma <= 1e-5f ? "YES":"NO", ma,mr,std::sqrt(ss/n),first,nf,
            (unsigned long long)hash_f32(ref,n),(unsigned long long)hash_f32(act,n));
        all_pass = all_pass && ma <= 1e-5f;
    };
    report("norm", rn.data(), actual_norm.data(), rn.size());
    report("down_iq4xs_q8k", independent_down.data(), actual_down.data(), independent_down.size());
    report("gate", rg.data(), actual_gate.data(), rg.size());
    report("mixed", rm.data(), actual_mixed.data(), rm.size());
    report("inject", ri.data(), actual_inject.data(), ri.size());
    ggml_backend_buffer_free(buf); ggml_free(c); ggml_backend_free(be);
    return all_pass;
}

static bool execute_transaction_sparse_moe(
        flashnext_transaction_context & txc, const std::string & index_path, std::string & error) {
    if (!txc.tensor_index) { error = "missing tensor index"; return false; }
    std::array<int, 512> order{};
    for (int i = 0; i < 512; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + flashnext_layer_transaction::experts_used,
        order.end(), [&](int a, int b) { return txc.router_logits[a] > txc.router_logits[b]; });
    double z = 0.0, ex[flashnext_layer_transaction::experts_used];
    const float mx = txc.router_logits[order[0]];
    for (size_t i = 0; i < flashnext_layer_transaction::experts_used; ++i) {
        ex[i] = std::exp((double) txc.router_logits[order[i]] - mx); z += ex[i];
    }
    txc.values.layer = (int) txc.layer_desc.index;
    for (size_t i = 0; i < flashnext_layer_transaction::experts_used; ++i) {
        txc.values.selected_expert_ids[i] = order[i];
        txc.values.routing_weights[i] = ex[i] / z;
        txc.selected_ids[i] = order[i];
    }
    std::memcpy(txc.values.pre_moe_hidden.data(), txc.moe_input.data(), sizeof(txc.moe_input));
    std::memcpy(txc.values.residual_hc.data(), txc.residual_hc.data(), sizeof(txc.residual_hc));
    std::memcpy(txc.values.inject.data(), txc.injection.data(), sizeof(txc.injection));
    if (!txc.tensor_index->load(index_path, &error)) return false;
    if (!flashnext_execute_selected_moe(*txc.tensor_index, index_path, txc.values)) return false;
    return true;
}

static bool execute_transaction_graph_b(
        flashnext_transaction_context & txc, const std::string & index_path, std::string & error) {
    if (!txc.tensor_index) { error = "missing tensor index"; return false; }
    flashnext_tensor_descriptor sg, su, sd, sc;
    const int layer = (int) txc.layer_desc.index;
    if (!txc.tensor_index->resolve_shared(layer, flashnext_projection::gate, sg, &error) ||
        !txc.tensor_index->resolve_shared(layer, flashnext_projection::up, su, &error) ||
        !txc.tensor_index->resolve_shared(layer, flashnext_projection::down, sd, &error) ||
        !txc.tensor_index->resolve_shared(layer, flashnext_projection::shared_control, sc, &error)) return false;
    const std::string paths[4] = { sg.path, su.path, sd.path, sc.path };
    std::array<int, 4> fds{{-1, -1, -1, -1}};
    for (int p = 0; p < 4; ++p) fds[p] = ::open(paths[p].c_str(), O_RDONLY | O_CLOEXEC);
    flashnext_shared_expert_slices shared{fds, sg.offset, su.offset, sd.offset, sc.offset};
    const bool shared_ok = flashnext_execute_shared_expert(shared, txc.values);
    for (int fd : fds) if (fd >= 0) ::close(fd);
    if (!shared_ok) { error = "shared expert execution failed"; return false; }
    for (size_t i = 0; i < txc.values.complete_moe.size(); ++i) {
        txc.values.complete_moe[i] = txc.values.selected_moe[i] + txc.values.shared_moe[i];
    }
    if (!flashnext_execute_segment_b(txc.values)) { error = "Graph-B execution failed"; return false; }
    txc.output = txc.values.layer_output;
    for (float x : txc.output) if (!std::isfinite(x)) { error = "non-finite Graph-B output"; return false; }
    return true;
}

// Execute the stock layer-0 graph for a continuation token.  The layer-0
// context is disposable; its opaque sequence snapshot is restored before the
// decode, and the resulting owned activation is returned to the streaming
// coordinator.
static bool flashnext_run_layer0_continuation(
        const char * model_path, const char * index_path, const llama_model_params & mp,
        llama_context_params cp, llama_token token, int position,
        const std::vector<uint8_t> & state, std::vector<float> & output) {
    // The continuation is the real layer-0 token-input graph.  The preceding
    // layer-range loop leaves range variables set for its last nonzero layer;
    // loading under those flags creates a pre-bound weight descriptor while
    // the token graph still reaches token_embd.weight.  Restore the same
    // router-only loader configuration used by the original layer-0 pass
    // before constructing this replacement context.
    unsetenv("CORTEX_FLASHNEXT_LAYER_RANGE");
    unsetenv("CORTEX_FLASHNEXT_LAYER_START");
    unsetenv("CORTEX_FLASHNEXT_LAYER_END");
    unsetenv("CORTEX_QWEN35_SEGMENT_END");
    unsetenv("CORTEX_FLASHNEXT_PRE_MOE");
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1);
    setenv("CORTEX_FLASHNEXT_SEGMENT_A", "1", 1);
    setenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER", "1", 1);
    llama_model * model = llama_model_load_from_file(model_path, mp);
    llama_context * ctx = model ? llama_init_from_model(model, cp) : nullptr;
    if (!model || !ctx) {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        return false;
    }
    const bool restored = !state.empty() &&
        llama_state_seq_set_data(ctx, state.data(), state.size(), 0) == state.size();
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0] = token;
    batch.pos[0] = position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    batch.n_tokens = 1;
    const int rc = restored ? llama_decode(ctx, batch) : -1;
    llama_synchronize(ctx);
    const float * artifact = rc == 0 ? llama_get_logits_ith(ctx, 0) : nullptr;
    bool ok = artifact != nullptr;
    flashnext_layer_transaction tx;
    if (ok) {
        std::memcpy(tx.pre_moe_hidden.data(), artifact, tx.pre_moe_hidden.size() * sizeof(float));
        std::memcpy(tx.residual_hc.data(), artifact + 2560, tx.residual_hc.size() * sizeof(float));
        std::memcpy(tx.inject.data(), artifact + 2560 + tx.residual_hc.size(), tx.inject.size() * sizeof(float));
        const float * route = artifact + 12804;
        std::array<int, 512> order{};
        for (int i = 0; i < 512; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + 10, order.end(),
            [route](int a, int b) { return route[a] > route[b]; });
        double z = 0.0, ex[10];
        const float mx = route[order[0]];
        for (int i = 0; i < 10; ++i) {
            ex[i] = std::exp((double) route[order[i]] - (double) mx);
            z += ex[i];
        }
        for (int i = 0; i < 10; ++i) {
            tx.selected_expert_ids[i] = order[i];
            tx.routing_weights[i] = ex[i] / z;
        }
        tx.layer = 0;
        flashnext_tensor_index index;
        std::string error;
        ok = index.load(index_path, &error) && flashnext_execute_selected_moe(index, index_path, tx);
        flashnext_tensor_descriptor sg, su, sd, sc;
        if (ok) ok = index.resolve_shared(0, flashnext_projection::gate, sg, &error) &&
            index.resolve_shared(0, flashnext_projection::up, su, &error) &&
            index.resolve_shared(0, flashnext_projection::down, sd, &error) &&
            index.resolve_shared(0, flashnext_projection::shared_control, sc, &error);
        std::array<int, 4> fds{{-1, -1, -1, -1}};
        if (ok) {
            fds = {{::open(sg.path.c_str(), O_RDONLY | O_CLOEXEC), -1, -1, -1}};
            ok = fds[0] >= 0 && sg.path == su.path && sg.path == sd.path && sg.path == sc.path;
            fds[1] = fds[2] = fds[3] = fds[0];
            flashnext_shared_expert_slices shared{fds, sg.offset, su.offset, sd.offset, sc.offset};
            ok = ok && flashnext_execute_shared_expert(shared, tx);
            if (fds[0] >= 0) ::close(fds[0]);
        }
        if (ok) {
            for (size_t i = 0; i < tx.complete_moe.size(); ++i)
                tx.complete_moe[i] = tx.selected_moe[i] + tx.shared_moe[i];
            ok = flashnext_execute_segment_b(tx);
        }
        if (ok) output.assign(tx.layer_output.begin(), tx.layer_output.end());
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

static layer_transaction_result run_layer_decode(
        llama_context * ctx, int position, const std::vector<float> & owned_hidden) {
    layer_transaction_result result;
    llama_batch batch = llama_batch_init(1, (int) owned_hidden.size(), 1);
    if (!batch.embd) {
        return result;
    }
    // Qwen4Exp uses three position planes for embedding batches.  The legacy
    // llama_batch allocator reserves one position per token, so provide the
    // two additional planes explicitly before llama_batch_compat reads them.
    // Uninitialized M-RoPE coordinates otherwise enter the serialized KV
    // extension and make recreated full-attention layers run-dependent.
    free(batch.pos);
    batch.pos = (llama_pos *) calloc(3, sizeof(llama_pos));
    if (!batch.pos) {
        llama_batch_free(batch);
        return result;
    }
    std::memcpy(batch.embd, owned_hidden.data(), owned_hidden.size() * sizeof(float));
    batch.pos[0] = position;
    batch.pos[1] = 0;
    batch.pos[2] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    batch.n_tokens = 1;
    result.state_before = capture_seq_state(ctx);
    result.decode_status = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    llama_batch_free(batch);
    return result;
}

static layer_transaction_result run_layer_transaction(
        llama_context * ctx, uint32_t layer, int position, const std::vector<float> & owned_hidden,
        const std::string & index_path, const flashnext_state_vault_hooks * vault_hooks = nullptr) {
    layer_transaction_result result = run_layer_decode(ctx, position, owned_hidden);
    if (result.decode_status != 0) { result.failure_stage = "graph_a_decode"; return result; }
    result.graph_a_ok = true;
    if (std::getenv("CORTEX_FLASHNEXT_TRACE_GRAPH_A") != nullptr && position == 1) {
        std::printf("GRAPH_A_ENTRY layer=%u position=%d sequence=0 input_hash=%016llx "
                    "recurrent_snapshot_hash=%016llx state_bytes=%zu state_metadata=\"%s\"\n",
            layer, position,
            (unsigned long long) hash_f32(owned_hidden.data(), owned_hidden.size()),
            (unsigned long long) (result.state_before.empty() ? 0 :
                hash_bytes(result.state_before.data(), result.state_before.size())),
            result.state_before.size(),
            ctx->debug_recurrent_state_metadata((int) layer).c_str());
        std::fflush(stdout);
    }
    flashnext_transaction_context txc;
    txc.ctx = ctx;
    txc.staged_state = result.state_before;
    std::string error;
    if (!resolve_flashnext_layer_descriptor(ctx, layer, txc.layer_desc, error)) {
        result.failure_stage = "layer_descriptor";
        return result;
    }
    txc.values.layer = (int) txc.layer_desc.index;
    txc.values.position = position;
    flashnext_tensor_index index;
    txc.tensor_index = &index;
    // The CPU operand diagnostic is a narrower capture path.  It must not be
    // rejected because the optional HC boundary-name collector did not find
    // every historical alias; the exact MUL_MAT source is captured below by
    // the execution-scoped CPU descriptor.
    if (std::getenv("CORTEX_FLASHNEXT_HC_CAPTURE") != nullptr &&
        std::getenv("CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE") == nullptr) {
        const bool captured = capture_hc_attention_boundaries(ctx, (int) layer);
        if (!captured) {
            LLAMA_LOG_INFO("FLASHNEXT_HC_BOUNDARY_CAPTURE_INCOMPLETE continuing_with_production_transaction=YES\n");
        }
    }
    if (std::getenv("CORTEX_FLASHNEXT_HC_ORACLE") != nullptr &&
        std::getenv("CORTEX_FLASHNEXT_HC_ORACLE_OUTER_ONLY") == nullptr) {
        std::string hc_error;
        const bool hc_ok = index.load(index_path, &hc_error) && hc_attention_reference(ctx, index, (int) layer, hc_error);
        std::printf("HC_MIXER_COMPLETE_PASS=%s REFERENCE_INDEPENDENCE=YES%s\n",
            hc_ok ? "YES" : "NO", hc_ok ? "" : (" FIRST_FAILURE=" + hc_error).c_str());
        if (!hc_ok) { result.failure_stage = "hc_attention_oracle"; }
    }
    const bool router_capture_ok = capture_transaction_router(txc, ctx, txc.layer_desc);
    if (router_capture_ok) result.graph_a_hash = hash_f32(txc.moe_input.data(), txc.moe_input.size());
    if (router_capture_ok && std::getenv("CORTEX_FLASHNEXT_TRACE_GRAPH_A") != nullptr) {
        std::printf("GRAPH_A_BOUNDARY layer=%u position=%d input_hash=%016llx output_hash=%016llx\n",
            layer, position,
            (unsigned long long) hash_f32(owned_hidden.data(), owned_hidden.size()),
            (unsigned long long) hash_f32(txc.moe_input.data(), txc.moe_input.size()));
        std::fflush(stdout);
    }
    if (router_capture_ok && std::getenv("CORTEX_FLASHNEXT_TRACE_POSITION1") != nullptr) {
        std::printf("TX_TRACE layer=%u position=%d stage=GRAPH_A input_hash=%016llx moe_input_hash=%016llx\n",
            layer, position, (unsigned long long) hash_f32(owned_hidden.data(), owned_hidden.size()),
            (unsigned long long) hash_f32(txc.moe_input.data(), txc.moe_input.size()));
        std::fflush(stdout);
    }
    const bool sparse_moe_ok = router_capture_ok && execute_transaction_sparse_moe(txc, index_path, error);
    if (std::getenv("CORTEX_FLASHNEXT_TRACE_POSITION1") != nullptr) {
        std::printf("TX_TRACE layer=%u position=%d stage=ROUTER capture=%s router_hash=%016llx ids=",
            layer, position, router_capture_ok ? "PASS" : "FAIL",
            (unsigned long long) hash_f32(txc.router_logits.data(), txc.router_logits.size()));
        for (size_t i = 0; i < txc.selected_ids.size(); ++i) std::printf("%s%d", i ? "," : "", txc.selected_ids[i]);
        std::printf("\n"); std::fflush(stdout);
    }
    if (!sparse_moe_ok) {
        std::fprintf(stderr, "FLASHNEXT_TRANSACTION_FAILURE layer=%u stage=capture_or_sparse_moe router_capture=%s error=%s\n",
            layer, router_capture_ok ? "PASS" : "FAIL", error.empty() ? "none" : error.c_str());
        if (!result.state_before.empty()) llama_state_seq_set_data(ctx, result.state_before.data(), result.state_before.size(), 0);
        result.failure_stage = "capture_or_sparse_moe";
        result.rollback_restored = true;
        return result;
    }
    result.sparse_moe_ok = true;
    result.sparse_output_hash = hash_f32(txc.values.selected_moe.data(), txc.values.selected_moe.size());
    if (std::getenv("CORTEX_FLASHNEXT_TRACE_POSITION1") != nullptr) {
        std::printf("TX_TRACE layer=%u position=%d stage=SPARSE_MOE hash=%016llx\n", layer, position,
            (unsigned long long) result.sparse_output_hash); std::fflush(stdout);
    }
    if (!execute_transaction_graph_b(txc, index_path, error)) {
        if (!result.state_before.empty()) llama_state_seq_set_data(ctx, result.state_before.data(), result.state_before.size(), 0);
        result.failure_stage = "graph_b";
        result.rollback_restored = true;
        return result;
    }
    result.graph_b_ok = true;
    result.output = txc.output;
    result.output_hash = hash_f32(result.output.data(), result.output.size());
    if (std::getenv("CORTEX_FLASHNEXT_TRACE_POSITION1") != nullptr) {
        std::printf("TX_TRACE layer=%u position=%d stage=GRAPH_B hash=%016llx\n", layer, position,
            (unsigned long long) result.output_hash); std::fflush(stdout);
    }
    if (std::getenv("CORTEX_FLASHNEXT_SHARED_ORACLE") != nullptr) {
        std::array<float, flashnext_layer_transaction::hidden> reference_shared{};
        std::string reference_error;
        flashnext_tensor_index reference_index;
        const bool reference_ok = shared_expert_reference(reference_index, (int) txc.layer_desc.index,
            index_path, txc.values, reference_shared, reference_error);
        float max_abs = 0, max_rel = 0; double ss = 0; size_t first = reference_shared.size();
        if (reference_ok) {
            for (size_t i = 0; i < reference_shared.size(); ++i) {
                const float d = std::fabs(reference_shared[i] - txc.values.shared_moe[i]);
                if (d != 0 && first == reference_shared.size()) first = i;
                max_abs = std::max(max_abs, d);
                max_rel = std::max(max_rel, d / std::max(std::fabs(reference_shared[i]), 1e-30f));
                ss += (double) d * d;
            }
        }
        const double rms = reference_ok ? std::sqrt(ss / reference_shared.size()) : NAN;
        size_t nonfinite = 0;
        for (float v : reference_shared) nonfinite += !std::isfinite(v);
        std::printf("SHARED_INPUT_CAPTURE_PASS=YES CONTROL_INPUT_CAPTURE_PASS=YES "
            "INDEPENDENT_WEIGHT_DECODING_PASS=%s INDEPENDENT_SHARED_EXPERT_PASS=%s "
            "REFERENCE_INDEPENDENCE=YES SHARED_MAX_ABS=%.9g SHARED_MAX_REL=%.9g SHARED_RMS=%.9g "
            "SHARED_NONFINITE=%zu SHARED_FIRST_MISMATCH=%zu REFERENCE_HASH=%016llx PRODUCTION_HASH=%016llx%s\n",
            reference_ok ? "YES" : "NO", reference_ok && max_abs <= 1e-5f ? "YES" : "NO",
            max_abs, max_rel, rms, nonfinite, first,
            (unsigned long long) hash_f32(reference_shared.data(), reference_shared.size()),
            (unsigned long long) hash_f32(txc.values.shared_moe.data(), txc.values.shared_moe.size()),
            reference_ok ? "" : reference_error.c_str());
    }
    if (std::getenv("CORTEX_FLASHNEXT_COMPLETE_MOE_ORACLE") != nullptr) {
        flashnext_tensor_index reference_index;
        std::string reference_error;
        flashnext_layer_transaction routed_tx = txc.values;
        const bool routed_ok = reference_index.load(index_path, &reference_error) &&
            flashnext_execute_selected_moe_reference(reference_index, index_path, routed_tx);
        std::array<float, flashnext_layer_transaction::hidden> shared_ref{};
        const bool shared_ok = routed_ok && shared_expert_reference(reference_index,
            (int) txc.layer_desc.index, index_path, txc.values, shared_ref, reference_error);
        std::array<float, flashnext_layer_transaction::hidden> complete_ref{};
        if (shared_ok) {
            for (size_t i = 0; i < complete_ref.size(); ++i) {
                complete_ref[i] = routed_tx.selected_moe[i] + shared_ref[i];
            }
        }
        float moe_abs = 0, moe_rel = 0; double moe_ss = 0; size_t moe_first = complete_ref.size();
        float out_abs = 0, out_rel = 0; double out_ss = 0; size_t out_first = txc.output.size();
        std::array<float, flashnext_layer_transaction::hidden * flashnext_layer_transaction::streams> output_ref{};
        if (shared_ok) {
            for (size_t i = 0; i < complete_ref.size(); ++i) {
                const float d = std::fabs(complete_ref[i] - txc.values.complete_moe[i]);
                if (d != 0 && moe_first == complete_ref.size()) moe_first = i;
                moe_abs = std::max(moe_abs, d);
                moe_rel = std::max(moe_rel, d / std::max(std::fabs(complete_ref[i]), 1e-30f));
                moe_ss += (double) d * d;
            }
            for (size_t s = 0; s < flashnext_layer_transaction::streams; ++s) {
                const float w = 2.0f / (1.0f + std::exp(-txc.values.inject[s] / 4.0f));
                for (size_t h = 0; h < flashnext_layer_transaction::hidden; ++h) {
                    const size_t i = s * flashnext_layer_transaction::hidden + h;
                    output_ref[i] = txc.values.residual_hc[i] + complete_ref[h] * w;
                    const float d = std::fabs(output_ref[i] - txc.output[i]);
                    if (d != 0 && out_first == txc.output.size()) out_first = i;
                    out_abs = std::max(out_abs, d);
                    out_rel = std::max(out_rel, d / std::max(std::fabs(output_ref[i]), 1e-30f));
                    out_ss += (double) d * d;
                }
            }
        }
        const double moe_rms = shared_ok ? std::sqrt(moe_ss / complete_ref.size()) : NAN;
        const double out_rms = shared_ok ? std::sqrt(out_ss / output_ref.size()) : NAN;
        size_t nonfinite = 0;
        for (float v : output_ref) nonfinite += !std::isfinite(v);
        std::printf("ROUTED_REFERENCE_PASS=%s SHARED_REFERENCE_PASS=%s COMPLETE_MOE_REFERENCE_PASS=%s "
            "FINAL_HC_REFERENCE_PASS=%s REFERENCE_INDEPENDENCE=YES "
            "COMPLETE_MOE_MAX_ABS=%.9g COMPLETE_MOE_RMS=%.9g COMPLETE_MOE_FIRST_MISMATCH=%zu "
            "FINAL_HC_MAX_ABS=%.9g FINAL_HC_RMS=%.9g FINAL_HC_FIRST_MISMATCH=%zu "
            "FINAL_HC_NONFINITE=%zu COMPLETE_MOE_REFERENCE_HASH=%016llx COMPLETE_MOE_PRODUCTION_HASH=%016llx "
            "FINAL_HC_REFERENCE_HASH=%016llx FINAL_HC_PRODUCTION_HASH=%016llx%s\n",
            routed_ok ? "YES" : "NO", shared_ok ? "YES" : "NO", shared_ok ? "YES" : "NO",
            shared_ok && out_abs <= 1e-5f ? "YES" : "NO", moe_abs, moe_rms, moe_first,
            out_abs, out_rms, out_first, nonfinite,
            (unsigned long long) hash_f32(complete_ref.data(), complete_ref.size()),
            (unsigned long long) hash_f32(txc.values.complete_moe.data(), txc.values.complete_moe.size()),
            (unsigned long long) hash_f32(output_ref.data(), output_ref.size()),
            (unsigned long long) hash_f32(txc.output.data(), txc.output.size()),
            shared_ok ? "" : reference_error.c_str());
    }
    result.selected_ids = txc.selected_ids;
    result.unselected_expert_reads = txc.values.io.unselected_expert_reads;
    if (std::getenv("CORTEX_FLASHNEXT_INJECT_PRE_COMMIT_FAILURE") != nullptr) {
        if (!result.state_before.empty()) llama_state_seq_set_data(ctx, result.state_before.data(), result.state_before.size(), 0);
        result.failure_stage = "injected_pre_commit";
        result.rollback_restored = true;
        return result;
    }
    if (vault_hooks && vault_hooks->enabled && vault_hooks->sync_export_commit) {
        // llama_decode has completed and run_layer_decode synchronized it;
        // the callback is therefore at the complete token-position boundary.
        if (!vault_hooks->sync_export_commit(vault_hooks->userdata, ctx, 0, layer,
                                              position, (uint64_t) position)) {
            if (!result.state_before.empty()) llama_state_seq_set_data(ctx, result.state_before.data(), result.state_before.size(), 0);
            result.failure_stage = "state_vault_commit";
            result.rollback_restored = true;
            return result;
        }
    }
    result.state_committed = true;
    return result;
}

int main(int argc, char ** argv) {
    // semantic inputs only: model path and flashnext-tensor-index.json
    if (argc != 3) return 2;
    const char * layer_start_env = std::getenv("CORTEX_FLASHNEXT_LAYER_START");
    const uint32_t target_layer = layer_start_env ? (uint32_t) std::strtoul(layer_start_env, nullptr, 10) : 1;
    const bool range_mode = layer_start_env != nullptr && target_layer > 0;
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", range_mode ? std::to_string(target_layer).c_str() : "0", 1);
    if (!range_mode) {
        setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1);
        setenv("CORTEX_FLASHNEXT_SEGMENT_A", "1", 1);
        setenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER", "1", 1);
    } else {
        setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
        // Nonzero-layer transactions are completed by the external sparse
        // MoE bridge.  Keep the GGML graph at the authoritative pre-MoE
        // boundary so dense MUL_MAT_ID expert tensors are not reachable by
        // graph allocation.
        setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
        // The coordinator's sparse-MoE bridge consumes these optional,
        // execution-owned boundary copies.  Range mode must enable the
        // existing capture path before graph construction; this does not
        // alter graph mathematics or expert selection.
        setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
        if (std::getenv("CORTEX_FLASHNEXT_PRE_MOE") != nullptr) {
            LLAMA_LOG_INFO("FLASHNEXT_PRE_MOE_RUN start=1 end=2\n");
        }
        setenv("CORTEX_QWEN35_SEGMENT_END", std::to_string(target_layer + 1).c_str(), 1);
        if (target_layer == 3 && std::getenv("CORTEX_FLASHNEXT_LAYER3_TRANSACTION") != nullptr) {
            setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
        }
        LLAMA_LOG_INFO("FLASHNEXT_RANGE_RUN start=%u end=%u owned_input_width=10240 embedding_path=SKIPPED\n",
                target_layer, target_layer + 1);
    }
    llama_backend_init();
    if (!known_pattern_backend_test()) return 16;
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 256; cp.n_batch = 1;
    if (range_mode) cp.embeddings = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;

    // First real full-attention milestone: execute the stock layer-3 graph
    // through its existing pre-MoE boundary.  This intentionally does not
    // claim sparse-MoE, Graph-B, or full-model execution.
    if (range_mode && target_layer == 3 && std::getenv("CORTEX_FLASHNEXT_LAYER3_SLICE") != nullptr) {
        std::vector<float> layer3_input(2560 * 4);
        for (size_t i = 0; i < layer3_input.size(); ++i) layer3_input[i] = std::sin(0.001f * (float) i);
        const layer_transaction_result slice = run_layer_decode(ctx, 0, layer3_input);
        std::printf("LAYER3_ATTENTION_SLICE GRAPH_CONSTRUCTION_PASS=YES ATTENTION_EXECUTION_PASS=%s "
                    "KV_STATE_PASS=%s GRAPH_A_BOUNDARY=pre_moe SPARSE_MOE=NOT_RUN GRAPH_B=NOT_RUN "
                    "OUTPUT_FINITE=NOT_RUN\n",
            slice.decode_status == 0 ? "YES" : "NO",
            slice.decode_status == 0 ? "YES" : "NO");
        std::printf("LAYER3_FAILURE=%s\n", slice.decode_status == 0 ? "none" : "llama_decode");
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return slice.decode_status == 0 ? 0 : 17;
    }

    if (range_mode && target_layer == 3 && std::getenv("CORTEX_FLASHNEXT_LAYER3_TRANSACTION") != nullptr) {
        std::vector<float> layer3_input(2560 * 4);
        for (size_t i = 0; i < layer3_input.size(); ++i) layer3_input[i] = std::sin(0.001f * (float) i);
        const layer_transaction_result transaction = run_layer_transaction(ctx, target_layer, 0, layer3_input, argv[2]);
        std::printf("LAYER3_TRANSACTION GRAPH_A=%s ATTENTION_EXECUTION=%s KV_STATE=%s "
                    "SPARSE_MOE=%s GRAPH_B=%s STATE_COMMIT=%s OUTPUT_HASH=%016llx "
                    "SELECTED_EXPERT_COUNT=%d UNSELECTED_EXPERT_READS=%llu\n",
            transaction.graph_a_ok ? "PASS" : "FAIL",
            transaction.graph_a_ok ? "PASS" : "FAIL",
            transaction.graph_a_ok ? "PASS" : "FAIL",
            transaction.sparse_moe_ok ? "PASS" : "FAIL",
            transaction.graph_b_ok ? "PASS" : "FAIL",
            transaction.state_committed ? "PASS" : "FAIL",
            (unsigned long long) transaction.output_hash,
            (int) transaction.selected_ids.size(),
            (unsigned long long) transaction.unselected_expert_reads);
        std::printf("LAYER3_FAILURE=%s\n", transaction.failure_stage ? transaction.failure_stage : "none");
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return transaction.state_committed ? 0 : 18;
    }

    if (std::getenv("CORTEX_FLASHNEXT_PREFLIGHT") != nullptr) {
        print_preflight(model);
        llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
    }

    if (range_mode) {
        // This is a graph/input wiring probe for the real nonzero-layer path.
        // The vector is caller-owned and deliberately deterministic; the
        // chained layer-0 producer will replace this fixture in the next
        // harness step once the [1,2) lifecycle is proven.
        std::vector<float> owned_hidden(2560*4, 0.0f);
        for (size_t i = 0; i < owned_hidden.size(); ++i) {
            owned_hidden[i] = std::sin(0.001f * (float) i);
        }
        const uint64_t before = hash_f32(owned_hidden.data(), owned_hidden.size());
        std::printf("FLASHNEXT_RANGE_INPUT owned=YES shape=[2560,4,1] bytes=%zu hash_before=%016llx\n",
                owned_hidden.size()*sizeof(float), (unsigned long long) before);
        if (std::getenv("CORTEX_FLASHNEXT_FRESH_CONTEXT_TEST") != nullptr) {
            llama_context * f1_ctx = llama_init_from_model(model, cp);
            const layer_transaction_result f1 = f1_ctx ?
                run_layer_transaction(f1_ctx, target_layer, 0, owned_hidden, argv[2]) : layer_transaction_result{};
            std::printf("F1_CONTEXT_POSITION0=%s GRAPH_A_HASH=%016llx OUTPUT_HASH=%016llx\n",
                f1.state_committed ? "PASS" : "FAIL",
                (unsigned long long) f1.graph_a_hash, (unsigned long long) f1.output_hash);
            std::fflush(stdout);
            if (f1_ctx) llama_free(f1_ctx);
            llama_context * f2_ctx = llama_init_from_model(model, cp);
            const layer_transaction_result f2 = f2_ctx ?
                run_layer_transaction(f2_ctx, target_layer, 0, owned_hidden, argv[2]) : layer_transaction_result{};
            std::printf("F2_CONTEXT_POSITION0=%s GRAPH_A_HASH=%016llx OUTPUT_HASH=%016llx FRESH_CONTEXT_POSITION0_MATCH=%s\n",
                f2.state_committed ? "PASS" : "FAIL",
                (unsigned long long) f2.graph_a_hash, (unsigned long long) f2.output_hash,
                f1.graph_a_hash == f2.graph_a_hash && f1.output_hash == f2.output_hash ? "YES" : "NO");
            std::fflush(stdout);
            if (f2_ctx) llama_free(f2_ctx);
        }
        layer_transaction_result transaction;
        bool range_rollback_pass = false;
        if (std::getenv("CORTEX_FLASHNEXT_ROLLBACK_TEST") != nullptr) {
            llama_context * rollback_ctx = llama_init_from_model(model, cp);
            const std::vector<float> rollback_input = [&]() {
                std::vector<float> v(2560 * 4);
                for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.001f * (float) i + 0.25f);
                return v;
            }();
            layer_transaction_result rollback_base;
            if (rollback_ctx) rollback_base = run_layer_transaction(rollback_ctx, target_layer, 0, owned_hidden, argv[2]);
            const std::vector<uint8_t> before_failure = rollback_ctx ? capture_layer_recurrent_state(rollback_ctx, 1) : std::vector<uint8_t>{};
            setenv("CORTEX_FLASHNEXT_INJECT_PRE_COMMIT_FAILURE", "1", 1);
            const layer_transaction_result failed = rollback_ctx ? run_layer_transaction(rollback_ctx, target_layer, 1, rollback_input, argv[2]) : layer_transaction_result{};
            unsetenv("CORTEX_FLASHNEXT_INJECT_PRE_COMMIT_FAILURE");
            const std::vector<uint8_t> after_failure = rollback_ctx ? capture_layer_recurrent_state(rollback_ctx, 1) : std::vector<uint8_t>{};
            range_rollback_pass = rollback_base.state_committed && failed.decode_status == 0 && !failed.state_committed &&
                failed.rollback_restored && before_failure == after_failure;
            std::printf("STATE_ROLLBACK_TEST=%s INJECTED_PRE_COMMIT_FAILURE=YES BEFORE_BYTES=%zu AFTER_BYTES=%zu BEFORE_HASH=%016llx AFTER_HASH=%016llx\n",
                range_rollback_pass ? "PASS" : "FAIL", before_failure.size(), after_failure.size(),
                (unsigned long long) hash_bytes(before_failure.data(), before_failure.size()),
                (unsigned long long) hash_bytes(after_failure.data(), after_failure.size()));
            const layer_transaction_result retry = range_rollback_pass ?
                run_layer_transaction(rollback_ctx, target_layer, 1, rollback_input, argv[2]) : failed;
            std::printf("STATE_RETRY=%s\n", retry.state_committed ? "PASS" : "FAIL");
            if (rollback_ctx) llama_free(rollback_ctx);
            transaction = run_layer_transaction(ctx, target_layer, 0, owned_hidden, argv[2]);
        } else {
            transaction = run_layer_transaction(ctx, target_layer, 0, owned_hidden, argv[2]);
        }
        if (std::getenv("CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE") != nullptr) {
            auto read_cpu_diag = [&](const char * name) {
                const size_t n = ctx->debug_copy_cpu_operand(name, nullptr, 0);
                std::vector<uint8_t> b(n);
                if (n) ctx->debug_copy_cpu_operand(name, b.data(), b.size());
                return b;
            };
            const auto wb = read_cpu_diag("weight_block0");
            const auto qb = read_cpu_diag("q8_block0");
            const auto cb = read_cpu_diag("capture_count");
            uint32_t count = 0; if (cb.size() == sizeof(count)) std::memcpy(&count, cb.data(), sizeof(count));
            std::printf("CPU_OPERAND_CAPTURE target=%s weight_bytes=%zu weight_hash=%016llx q8_bytes=%zu q8_hash=%016llx capture_once=%u match_count=%u\n",
                wb.size() == 136 && qb.size() != 0 ? "PASS" : "FAIL", wb.size(),
                (unsigned long long) hash_bytes(wb.data(), wb.size()), qb.size(),
                (unsigned long long) hash_bytes(qb.data(), qb.size()), count,
                (unsigned) ctx->debug_post_node_observer_invocations());
            if (wb.size() == 136 && qb.size() != 0) {
                flashnext_tensor_index diag_index;
                std::string diag_error;
                flashnext_tensor_descriptor down_desc;
                bool weight_equal = false, q8_equal = false;
                uint64_t ref_weight_hash = 0, ref_q8_hash = 0;
                if (diag_index.load(argv[2], &diag_error) &&
                    diag_index.resolve_hc_attention((int) target_layer, flashnext_projection::hc_down, down_desc, &diag_error)) {
                    int fd = ::open(down_desc.path.c_str(), O_RDONLY | O_CLOEXEC);
                    std::vector<uint8_t> ref_weight(136);
                    if (fd >= 0 && read_exact_fd(fd, ref_weight.data(), ref_weight.size(), down_desc.offset)) {
                        ref_weight_hash = hash_bytes(ref_weight.data(), ref_weight.size());
                        weight_equal = ref_weight == wb;
                    }
                    if (fd >= 0) ::close(fd);
                }
                const char * xn_name = "conversion_f32";
                size_t xn_bytes = ctx->debug_copy_cpu_operand(xn_name, nullptr, 0);
                if (xn_bytes == 0) { xn_name = "hc_down_xn-1";
                xn_bytes = ctx->debug_copy_flashnext_tensor(xn_name, nullptr, 0); }
                if (xn_bytes == 0) { xn_name = "hc_rms_weighted-1"; xn_bytes = ctx->debug_copy_flashnext_tensor(xn_name, nullptr, 0); }
                if (xn_bytes == 0) { xn_name = "hc_attn_norm-1"; xn_bytes = ctx->debug_copy_flashnext_tensor(xn_name, nullptr, 0); }
                if (xn_bytes == 0) { xn_name = "hc_down_mul_input-1"; xn_bytes = ctx->debug_copy_flashnext_tensor(xn_name, nullptr, 0); }
                if (xn_bytes == 0) { xn_name = "cpu_activation_f32"; xn_bytes = ctx->debug_copy_cpu_operand("activation_f32", nullptr, 0); }
                std::vector<float> xn(xn_bytes / sizeof(float));
                if (xn_bytes) {
                    if (std::strcmp(xn_name, "conversion_f32") == 0) ctx->debug_copy_cpu_operand("conversion_f32", xn.data(), xn_bytes);
                    else if (std::strcmp(xn_name, "cpu_activation_f32") == 0) ctx->debug_copy_cpu_operand("activation_f32", xn.data(), xn_bytes);
                    else ctx->debug_copy_flashnext_tensor(xn_name, xn.data(), xn_bytes);
                }
                std::printf("CPU_REFERENCE_F32 source=%s elements=%zu hash=%016llx first=%.9g second=%.9g last=%.9g SOURCE_MAPPING=conversion_row0_block0 SOURCE_OFFSET=0 SOURCE_STRIDE=%zu\n",
                    xn_name, xn.size(), (unsigned long long) hash_f32(xn.data(), xn.size()),
                    xn.empty() ? 0.0f : xn[0], xn.size() < 2 ? 0.0f : xn[1], xn.empty() ? 0.0f : xn.back(), sizeof(float));
                if (xn.size() >= 256) {
                    const auto ref_q8 = flashnext_quantize_q8k_ref(xn.data());
                    ref_q8_hash = hash_bytes(reinterpret_cast<const uint8_t *>(&ref_q8), sizeof(ref_q8));
                    q8_equal = qb.size() == sizeof(ref_q8) && std::memcmp(qb.data(), &ref_q8, sizeof(ref_q8)) == 0;
                    if (qb.size() == sizeof(ref_q8)) {
                        const auto * prod_q8 = reinterpret_cast<const flashnext_q8k_ref *>(qb.data());
                        size_t first_byte = sizeof(ref_q8);
                        for (size_t i = 0; i < sizeof(ref_q8); ++i) if (qb[i] != reinterpret_cast<const uint8_t *>(&ref_q8)[i]) { first_byte = i; break; }
                        std::printf("Q8_K_FIELD_TRACE production_d=%.9g reference_d=%.9g production_q0=%d reference_q0=%d production_sum0=%d reference_sum0=%d first_byte=%zu\n",
                            prod_q8->d, ref_q8.d, (int) prod_q8->qs[0], (int) ref_q8.qs[0],
                            (int) prod_q8->bsums[0], (int) ref_q8.bsums[0], first_byte);
                    }
                }
                std::printf("CPU_OPERAND_COMPARE WEIGHT_BLOCK0_BYTES_EQUAL=%s PRODUCTION_WEIGHT_HASH=%016llx REFERENCE_WEIGHT_HASH=%016llx Q8_K_BLOCK0_BYTES_EQUAL=%s PRODUCTION_Q8_K_HASH=%016llx REFERENCE_Q8_K_HASH=%016llx\n",
                    weight_equal ? "YES" : "NO", (unsigned long long) hash_bytes(wb.data(), wb.size()),
                    (unsigned long long) ref_weight_hash, q8_equal ? "YES" : "NO",
                    (unsigned long long) hash_bytes(qb.data(), qb.size()), (unsigned long long) ref_q8_hash);
            }
        }
        // Range-mode HC oracle: run exactly once, immediately after the
        // position-0 transaction while the context-owned diagnostic copies
        // are still live.  This must not be nested under the optional 1->2
        // chain branch.
        if (std::getenv("CORTEX_FLASHNEXT_HC_ORACLE") != nullptr) {
            flashnext_tensor_index hc_index;
            std::string hc_error;
            const bool hc_ok = transaction.state_committed &&
                hc_index.load(argv[2], &hc_error) &&
                hc_attention_reference(ctx, hc_index, (int) target_layer, hc_error);
            std::printf("HC_DOWN_ORACLE_INVOKED=YES HC_DOWN_ORACLE_PASS=%s HC_DOWN_ORACLE_ERROR=%s\n",
                hc_ok ? "YES" : "NO", hc_error.c_str());
        }
        if ((std::getenv("CORTEX_FLASHNEXT_CHAIN_1_2") != nullptr ||
             std::getenv("CORTEX_FLASHNEXT_CHAIN_1_2_3") != nullptr ||
             std::getenv("CORTEX_FLASHNEXT_CHAIN_1_6") != nullptr ||
             std::getenv("CORTEX_FLASHNEXT_CHAIN_1_12") != nullptr ||
             std::getenv("CORTEX_FLASHNEXT_CHAIN_1_48") != nullptr) && target_layer == 1 && transaction.state_committed) {
            const std::vector<float> layer1_output(transaction.output.begin(), transaction.output.end());
            const uint64_t layer1_hash = hash_f32(layer1_output.data(), layer1_output.size());
            const uint32_t model_layer_count = model->hparams.n_layer();
            const std::vector<uint8_t> layer1_state = capture_layer_recurrent_state(ctx, 1);
            // The layer-1 context owns the scheduler/backend bindings.  Drop
            // it before rebinding layer-2 tensors; the activation is owned by
            // layer1_output and therefore survives this teardown.
            llama_free(ctx);
            ctx = nullptr;
            llama_model_free(model);
            model = nullptr;
            setenv("CORTEX_LOW_RESIDENCY_LAYER", "2", 1);
            setenv("CORTEX_FLASHNEXT_LAYER_START", "2", 1);
            setenv("CORTEX_FLASHNEXT_LAYER_END", "3", 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", "3", 1);
            llama_model * layer2_model = llama_model_load_from_file(argv[1], mp);
            llama_context * layer2_ctx = layer2_model ? llama_init_from_model(layer2_model, cp) : nullptr;
            const bool rebound = layer2_model != nullptr && layer2_ctx != nullptr;
            layer_transaction_result layer2_result;
            std::vector<uint8_t> layer2_before;
            // A llama_context caches its range graph.  Constructing a fresh
            // context after rebinding is required to make [2,3) a real graph;
            // the activation remains caller-owned and is the exact output of
            // the completed layer-1 transaction.
            if (layer2_ctx) {
                const bool inject_failure = std::getenv("CORTEX_FLASHNEXT_CHAIN_INJECT_FAILURE") != nullptr;
                if (inject_failure) {
                    const layer_transaction_result baseline = run_layer_transaction(layer2_ctx, 2, 0, layer1_output, argv[2]);
                    layer2_before = capture_seq_state(layer2_ctx);
                    if (!baseline.state_committed) layer2_result = baseline;
                }
                if (inject_failure) {
                    setenv("CORTEX_FLASHNEXT_INJECT_PRE_COMMIT_FAILURE", "1", 1);
                }
                if (!inject_failure || layer2_result.decode_status == -1) {
                    layer2_result = run_layer_transaction(layer2_ctx, 2, inject_failure ? 1 : 0, layer1_output, argv[2]);
                }
                unsetenv("CORTEX_FLASHNEXT_INJECT_PRE_COMMIT_FAILURE");
            }
            const std::vector<uint8_t> layer2_after = layer2_ctx ? capture_seq_state(layer2_ctx) : std::vector<uint8_t>{};
            const bool layer2_failed_rollback = layer2_ctx && !layer2_result.state_committed &&
                layer2_before == layer2_after;
            const bool chain_failure_mode = std::getenv("CORTEX_FLASHNEXT_CHAIN_INJECT_FAILURE") != nullptr;
            std::printf("CHAIN_EXECUTION_PASS=%s ACTIVATION_CONTRACT_PASS=%s HANDOFF_BYTE_EXACT=YES "
                "LAYER1_OUTPUT_HASH=%016llx LAYER2_INPUT_HASH=%016llx LAYER2_OUTPUT_HASH=%016llx "
                "LAYER1_STATE_COMMIT=PASS LAYER2_STATE_COMMIT=%s LAYER2_FAILURE_ROLLBACK=%s "
                "WHOLE_CHAIN_ATOMICITY=NOT_IMPLEMENTED LAYER2_REBIND=%s\n",
                !chain_failure_mode && rebound && layer2_result.state_committed ? "YES" : "NO",
                rebound ? "YES" : "NO", (unsigned long long) layer1_hash,
                (unsigned long long) hash_f32(layer1_output.data(), layer1_output.size()),
                (unsigned long long) layer2_result.output_hash,
                layer2_result.state_committed ? "PASS" : "FAIL",
                chain_failure_mode && layer2_failed_rollback ? "PASS" : (chain_failure_mode ? "FAIL" : "NOT_RUN"),
                rebound ? "PASS" : "FAIL");
            std::printf("LAYER2_OUTPUT_FINITE=%s LAYER2_UNSELECTED_EXPERT_READS=%llu\n",
                layer2_result.state_committed ? "YES" : "NO",
                (unsigned long long) layer2_result.unselected_expert_reads);
            if (chain_failure_mode) {
                std::printf("LAYER2_FAILURE_STAGE=%s LAYER2_STATE_BEFORE_HASH=%016llx LAYER2_STATE_AFTER_HASH=%016llx\n",
                    layer2_result.failure_stage ? layer2_result.failure_stage : "none",
                    (unsigned long long) hash_bytes(layer2_before.data(), layer2_before.size()),
                    (unsigned long long) hash_bytes(layer2_after.data(), layer2_after.size()));
            }
            if (!rebound || (!chain_failure_mode && !layer2_result.state_committed) ||
                    (chain_failure_mode && !layer2_failed_rollback)) {
                llama_free(ctx); llama_model_free(model); llama_backend_free(); return 20;
            }
            if (chain_failure_mode) {
                // Layer 1 remains committed; this mode proves only layer-2
                // transaction rollback, not whole-chain atomicity.
                const bool layer1_preserved = !layer1_state.empty();
                std::printf("LAYER1_STATE_PRESERVED_AFTER_LAYER2_FAILURE=%s\n", layer1_preserved ? "YES" : "NO");
            }
            const bool chain_3 = std::getenv("CORTEX_FLASHNEXT_CHAIN_1_2_3") != nullptr && !chain_failure_mode;
            const bool chain_6 = std::getenv("CORTEX_FLASHNEXT_CHAIN_1_6") != nullptr && !chain_failure_mode;
            const bool chain_12 = std::getenv("CORTEX_FLASHNEXT_CHAIN_1_12") != nullptr && !chain_failure_mode;
            const bool chain_48 = std::getenv("CORTEX_FLASHNEXT_CHAIN_1_48") != nullptr && !chain_failure_mode;
            layer_transaction_result layer3_result;
            std::vector<float> layer3_output;
            uint64_t layer3_input_hash = 0;
            bool layer3_rebound = false;
            if (chain_3 || chain_6 || chain_12 || chain_48) {
                // Release layer-2 residency before binding layer 3.  The
                // activation is copied into owned storage first, so the
                // handoff remains valid across context/model teardown.
                const std::vector<float> layer2_output(layer2_result.output.begin(), layer2_result.output.end());
                layer3_input_hash = hash_f32(layer2_output.data(), layer2_output.size());
                llama_free(layer2_ctx); layer2_ctx = nullptr;
                llama_model_free(layer2_model); layer2_model = nullptr;
                setenv("CORTEX_LOW_RESIDENCY_LAYER", "3", 1);
                setenv("CORTEX_FLASHNEXT_LAYER_START", "3", 1);
                setenv("CORTEX_FLASHNEXT_LAYER_END", "4", 1);
                setenv("CORTEX_QWEN35_SEGMENT_END", "4", 1);
                setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
                setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
                setenv("CORTEX_FLASHNEXT_LAYER3_TRANSACTION", "1", 1);
                llama_model * layer3_model = llama_model_load_from_file(argv[1], mp);
                llama_context * layer3_ctx = layer3_model ? llama_init_from_model(layer3_model, cp) : nullptr;
                layer3_rebound = layer3_model != nullptr && layer3_ctx != nullptr;
                if (layer3_ctx) {
                    layer3_result = run_layer_transaction(layer3_ctx, 3, 0, layer2_output, argv[2]);
                    layer3_output.assign(layer3_result.output.begin(), layer3_result.output.end());
                    std::printf("CHAIN_HANDOFF_2_TO_3_BYTES_EQUAL=YES INPUT_HASH=%016llx\n",
                        (unsigned long long) layer3_input_hash);
                    llama_free(layer3_ctx);
                }
                if (layer3_model) llama_model_free(layer3_model);
                std::printf("LAYER3_CHAIN GRAPH_A=%s ATTENTION_EXECUTION=%s KV_STATE=%s "
                    "SPARSE_MOE=%s GRAPH_B=%s STATE_COMMIT=%s OUTPUT_HASH=%016llx "
                    "SELECTED_EXPERT_COUNT=%d UNSELECTED_EXPERT_READS=%llu\n",
                    layer3_result.graph_a_ok ? "PASS" : "FAIL",
                    layer3_result.graph_a_ok ? "PASS" : "FAIL",
                    layer3_result.graph_a_ok ? "PASS" : "FAIL",
                    layer3_result.sparse_moe_ok ? "PASS" : "FAIL",
                    layer3_result.graph_b_ok ? "PASS" : "FAIL",
                    layer3_result.state_committed ? "PASS" : "FAIL",
                    (unsigned long long) layer3_result.output_hash,
                    (int) layer3_result.selected_ids.size(),
                    (unsigned long long) layer3_result.unselected_expert_reads);
                std::printf("LAYER3_CHAIN_FAILURE=%s\n", layer3_result.failure_stage ? layer3_result.failure_stage : "none");
            }
            bool layers4_6_pass = true;
            if ((chain_6 || chain_12 || chain_48) && layer3_result.state_committed) {
                std::vector<float> previous_output = layer3_output;
                int live_models = 0;
                int live_contexts = 0;
                auto resource_snapshot = [&](uint32_t il, const char * boundary) {
                    std::ifstream status_file("/proc/self/status");
                    std::string key, value, unit;
                    uint64_t vm_size = 0, vm_data = 0, vm_rss = 0, vm_hwm = 0;
                    while (status_file >> key >> value >> unit) {
                        const uint64_t v = std::strtoull(value.c_str(), nullptr, 10);
                        if (key == "VmSize:") vm_size = v;
                        else if (key == "VmData:") vm_data = v;
                        else if (key == "VmRSS:") vm_rss = v;
                        else if (key == "VmHWM:") vm_hwm = v;
                    }
                    uint64_t fd_count = 0;
                    if (DIR * dir = opendir("/proc/self/fd")) {
                        while (readdir(dir)) ++fd_count;
                        closedir(dir);
                        if (fd_count >= 2) fd_count -= 2;
                    }
                    uint64_t map_count = 0;
                    std::ifstream maps("/proc/self/maps");
                    std::string map_line;
                    while (std::getline(maps, map_line)) ++map_count;
                    std::fprintf(stderr,
                        "FLASHNEXT_RESOURCE layer=%u boundary=%s VmRSS_kib=%llu VmHWM_kib=%llu "
                        "VmSize_kib=%llu VmData_kib=%llu open_fds=%llu mapped_regions=%llu "
                        "live_models=%d live_contexts=%d\n",
                        il, boundary, (unsigned long long) vm_rss, (unsigned long long) vm_hwm,
                        (unsigned long long) vm_size, (unsigned long long) vm_data,
                        (unsigned long long) fd_count, (unsigned long long) map_count,
                        live_models, live_contexts);
                    std::fflush(stderr);
                };
                // GGML model indices are zero-based: n_layer=48 means the
                // highest valid descriptor is 47.  The previous inclusive
                // 48-layer loop attempted an out-of-range layer-48 context.
                const uint32_t chain_end = chain_48
                    ? std::min<uint32_t>(48, model_layer_count - 1)
                    : (chain_12 ? 12 : 6);
                for (uint32_t il = 4; il <= chain_end; ++il) {
                    auto layer_checkpoint = [il](const char * stage, const char * detail = "") {
                        std::fprintf(stderr, "FLASHNEXT_LAYER_CHECKPOINT layer=%u stage=%s detail=%s\n",
                            il, stage, detail);
                        std::fflush(stderr);
                    };
                    layer_checkpoint("LAYER_BEGIN");
                    resource_snapshot(il, "BEFORE_MODEL_LOAD");
                    setenv("CORTEX_LOW_RESIDENCY_LAYER", std::to_string(il).c_str(), 1);
                    setenv("CORTEX_FLASHNEXT_LAYER_START", std::to_string(il).c_str(), 1);
                    setenv("CORTEX_FLASHNEXT_LAYER_END", std::to_string(il + 1).c_str(), 1);
                    setenv("CORTEX_QWEN35_SEGMENT_END", std::to_string(il + 1).c_str(), 1);
                    setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
                    setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
                    setenv("CORTEX_FLASHNEXT_LAYER3_TRANSACTION", "1", 1);
                    layer_checkpoint("LOAD_BEGIN");
                    errno = 0;
                    llama_model * next_model = llama_model_load_from_file(argv[1], mp);
                    if (next_model) ++live_models;
                    resource_snapshot(il, "AFTER_MODEL_LOAD");
                    if (!next_model) {
                        char detail[128];
                        std::snprintf(detail, sizeof(detail), "model_load_failed errno=%d", errno);
                        layer_checkpoint("TENSOR_OPEN", detail);
                    } else {
                        layer_checkpoint("TENSOR_OPEN", "model_loaded");
                    }
                    llama_context * next_ctx = next_model ? llama_init_from_model(next_model, cp) : nullptr;
                    layer_transaction_result next_result;
                    const bool bound = next_model != nullptr && next_ctx != nullptr;
                    if (next_ctx) ++live_contexts;
                    if (next_model && !next_ctx) layer_checkpoint("GRAPH_BUILD", "context_init_failed");
                    if (next_ctx) {
                        layer_checkpoint("GRAPH_EXECUTE");
                        next_result = run_layer_transaction(next_ctx, il, 0, previous_output, argv[2]);
                        layer_checkpoint(next_result.graph_a_ok ? "GRAPH_A_END" : "GRAPH_A_FAIL",
                            next_result.failure_stage ? next_result.failure_stage : "none");
                        if (next_result.graph_a_ok && !next_result.sparse_moe_ok) layer_checkpoint("MOE_FAIL", "sparse_moe");
                        if (next_result.sparse_moe_ok && !next_result.graph_b_ok) layer_checkpoint("GRAPH_B_FAIL", "graph_b");
                        if (next_result.graph_b_ok && !next_result.state_committed) layer_checkpoint("STATE_COMMIT_FAIL", "commit");
                        if (next_result.state_committed) layer_checkpoint("LAYER_END");
                    if (next_result.state_committed) resource_snapshot(il, "AFTER_LAYER_COMMIT");
                    }
                    const bool handoff = bound && previous_output.size() == 2560 * 4;
                    uint64_t rss_kib = 0;
                    {
                        std::ifstream status_file("/proc/self/status");
                        std::string key, value, unit;
                        while (status_file >> key >> value >> unit) {
                            if (key == "VmRSS:") { rss_kib = std::strtoull(value.c_str(), nullptr, 10); break; }
                        }
                    }
                    std::printf("LAYER_INDEX=%u LAYER_TYPE=%s OUTPUT_HASH=%016llx STATE_COMMIT=%s "
                        "SELECTED_EXPERT_COUNT=%d UNSELECTED_EXPERT_READS=%llu HANDOFF_BYTES_EQUAL=%s RSS_AFTER_LAYER_KIB=%llu\n",
                        il, bound && next_model->hparams.is_recr(il) ? "recurrent" : "full_attention",
                        (unsigned long long) next_result.output_hash,
                        next_result.state_committed ? "PASS" : "FAIL",
                        (int) next_result.selected_ids.size(),
                        (unsigned long long) next_result.unselected_expert_reads,
                        handoff ? "YES" : "NO", (unsigned long long) rss_kib);
                    std::printf("LAYER_%u_FAILURE=%s\n", il,
                        next_result.failure_stage ? next_result.failure_stage : "none");
                    std::fflush(stdout);
                    layers4_6_pass = layers4_6_pass && bound && next_result.state_committed && handoff;
                    if (next_result.state_committed) previous_output.assign(next_result.output.begin(), next_result.output.end());
                    if (next_ctx) { llama_free(next_ctx); --live_contexts; }
                    resource_snapshot(il, "AFTER_CONTEXT_FREE");
                    if (next_model) { llama_model_free(next_model); --live_models; }
                    resource_snapshot(il, "AFTER_MODEL_FREE");
                    std::printf("LAYER_RESULT index=%u type=%s hash=%016llx commit=%s selected=%d unselected=%llu rss_kib=%llu handoff=%s\n",
                        il, bound && next_model == nullptr ? "resolved" : "unknown",
                        (unsigned long long) next_result.output_hash,
                        next_result.state_committed ? "PASS" : "FAIL",
                        (int) next_result.selected_ids.size(),
                        (unsigned long long) next_result.unselected_expert_reads,
                        (unsigned long long) rss_kib, handoff ? "YES" : "NO");
                    std::fflush(stdout);
                    if (!layers4_6_pass) break;
                }
            }
            if (layer2_ctx) llama_free(layer2_ctx);
            if (layer2_model) llama_model_free(layer2_model);
            llama_backend_free();
            if ((chain_3 || chain_6 || chain_12 || chain_48) && (!layer3_rebound || !layer3_result.state_committed)) return 21;
            if ((chain_6 || chain_12 || chain_48) && !layers4_6_pass) return 22;
            return 0;
        }
        const int rc = transaction.decode_status;
        const std::vector<uint8_t> & state_before_transaction = transaction.state_before;
        if (std::getenv("CORTEX_FLASHNEXT_COORDINATOR") != nullptr ||
            std::getenv("CORTEX_FLASHNEXT_CPU_OPERAND_CAPTURE") != nullptr) {
            std::printf("FULL_TRANSACTION_COORDINATOR=YES INLINE_COORDINATOR_REMAINING=NO\n");
            std::printf("GRAPH_A=%s SPARSE_MOE=%s GRAPH_B=%s STATE_COMMIT=%s\n",
                transaction.graph_a_ok ? "PASS" : "FAIL", transaction.sparse_moe_ok ? "PASS" : "FAIL",
                transaction.graph_b_ok ? "PASS" : "FAIL",
                transaction.state_committed ? "PASS" : "ROLLED_BACK");
            std::printf("FAILURE_STAGE=%s ROLLBACK_RESTORED=%s\n",
                transaction.failure_stage ? transaction.failure_stage : "none",
                transaction.rollback_restored ? "YES" : "NO");
            std::printf("SELECTED_EXPERT_IDS=");
            for (size_t i = 0; i < transaction.selected_ids.size(); ++i) {
                std::printf("%s%d", i ? "," : "", transaction.selected_ids[i]);
            }
            std::printf("\nPOSITION0_OUTPUT_HASH=%016llx UNSELECTED_EXPERT_READS=%llu\n",
                (unsigned long long) transaction.output_hash,
                (unsigned long long) transaction.unselected_expert_reads);
            const bool state_roundtrip_requested = std::getenv("CORTEX_FLASHNEXT_STATE_ROUNDTRIP") != nullptr;
            if (state_roundtrip_requested && !transaction.state_committed) {
                std::printf("STATE_ROUNDTRIP_SKIPPED reason=transaction_not_committed\n"); std::fflush(stdout);
            } else if (state_roundtrip_requested && transaction.state_before.empty()) {
                std::printf("STATE_ROUNDTRIP_SKIPPED reason=pre_state_empty\n"); std::fflush(stdout);
            }
            if (state_roundtrip_requested && transaction.state_committed) {
                std::printf("PROBE_ENTERED\n"); std::fflush(stdout);
                const llama_seq_id state_seq = 0;
                const size_t snapshot_size = llama_state_seq_get_size(ctx, state_seq);
                std::vector<uint8_t> snapshot(snapshot_size);
                const size_t snapshot_written = snapshot_size != 0 ?
                    llama_state_seq_get_data(ctx, snapshot.data(), snapshot.size(), state_seq) : 0;
                const uint64_t original_hash = snapshot_written == snapshot.size() ?
                    hash_bytes(snapshot.data(), snapshot.size()) : 0;
                if (snapshot_written == snapshot.size()) { std::printf("SNAPSHOT_EXPORTED\n"); std::fflush(stdout); }
                const bool continuity_requested = std::getenv("CORTEX_FLASHNEXT_STATE_CONTINUITY") != nullptr;
                std::vector<float> continuity_input(2560 * 4);
                for (size_t i = 0; i < continuity_input.size(); ++i) continuity_input[i] = std::sin(0.001f * (float) i + 0.25f);
                layer_transaction_result same_context_pos1;
                std::vector<uint8_t> same_context_state;
                bool same_context_roundtrip = false;
                if (continuity_requested && snapshot_written == snapshot.size()) {
                    const size_t d_restore = llama_state_seq_set_data(ctx, snapshot.data(), snapshot.size(), state_seq);
                    std::vector<uint8_t> d_reexport(snapshot.size());
                    const size_t d_written = llama_state_seq_get_data(ctx, d_reexport.data(), d_reexport.size(), state_seq);
                    same_context_roundtrip = d_restore == snapshot.size() && d_written == snapshot.size() && d_reexport == snapshot;
                    if (same_context_roundtrip) {
                        same_context_pos1 = run_layer_transaction(ctx, target_layer, 1, continuity_input, argv[2]);
                        same_context_state = capture_seq_state(ctx);
                    }
                    std::printf("D_SAME_CONTEXT_RESTORE=%s D_GRAPH_A_HASH=%016llx D_OUTPUT_HASH=%016llx D_STATE_HASH=%016llx\n",
                        same_context_roundtrip && same_context_pos1.state_committed ? "PASS" : "FAIL",
                        (unsigned long long) same_context_pos1.graph_a_hash,
                        (unsigned long long) same_context_pos1.output_hash,
                        (unsigned long long) (same_context_state.empty() ? 0 : hash_bytes(same_context_state.data(), same_context_state.size())));
                    std::fflush(stdout);
                }
                layer_transaction_result reference_pos1;
                std::vector<uint8_t> reference_state;
                if (continuity_requested && snapshot_written == snapshot.size()) {
                    // Restore position-0 again so A and D receive identical state.
                    if (llama_state_seq_set_data(ctx, snapshot.data(), snapshot.size(), state_seq) != snapshot.size()) {
                        std::printf("REFERENCE_POSITION1_EXECUTED=NO reason=same_context_restore_failed\n");
                        std::fflush(stdout);
                    }
                    reference_pos1 = run_layer_transaction(ctx, target_layer, 1, continuity_input, argv[2]);
                    reference_state = capture_seq_state(ctx);
                    std::printf("REFERENCE_POSITION1_EXECUTED=%s\n", reference_pos1.state_committed ? "YES" : "NO");
                    std::fflush(stdout);
                }
                layer_transaction_result retained_model_pos1;
                std::vector<uint8_t> retained_model_state;
                bool retained_model_restore = false;
                const std::string state_metadata = ctx->debug_recurrent_state_metadata((int) target_layer);
                // E: destroy only the context, retain the loaded model, then
                // recreate one context from that same model and restore state.
                if (continuity_requested && snapshot_written == snapshot.size()) {
                    llama_free(ctx);
                    ctx = nullptr;
                    ctx = model ? llama_init_from_model(model, cp) : nullptr;
                    if (ctx) {
                        const size_t e_restore = llama_state_seq_set_data(ctx, snapshot.data(), snapshot.size(), state_seq);
                        std::vector<uint8_t> e_reexport(snapshot.size());
                        const size_t e_written = e_restore == snapshot.size() ?
                            llama_state_seq_get_data(ctx, e_reexport.data(), e_reexport.size(), state_seq) : 0;
                        retained_model_restore = e_restore == snapshot.size() && e_written == snapshot.size() && e_reexport == snapshot;
                        if (retained_model_restore) {
                            retained_model_pos1 = run_layer_transaction(ctx, target_layer, 1, continuity_input, argv[2]);
                            retained_model_state = capture_seq_state(ctx);
                        }
                    }
                    std::printf("E_RETAIN_MODEL_CONTEXT_RECREATE=%s E_GRAPH_A_HASH=%016llx E_OUTPUT_HASH=%016llx E_STATE_HASH=%016llx\n",
                        retained_model_restore && retained_model_pos1.state_committed ? "PASS" : "FAIL",
                        (unsigned long long) retained_model_pos1.graph_a_hash,
                        (unsigned long long) retained_model_pos1.output_hash,
                        (unsigned long long) (retained_model_state.empty() ? 0 : hash_bytes(retained_model_state.data(), retained_model_state.size())));
                    std::fflush(stdout);
                    llama_free(ctx);
                    ctx = nullptr;
                }
                llama_free(ctx);
                ctx = nullptr;
                std::printf("ORIGINAL_CONTEXT_FREED\n"); std::fflush(stdout);
                llama_model_free(model);
                model = nullptr;
                std::printf("ORIGINAL_MODEL_FREED\n"); std::fflush(stdout);
                size_t restored_size = 0;
                size_t restore_written = 0;
                uint64_t restored_hash = 0;
                bool restore_ok = false;
                layer_transaction_result restored_pos1;
                std::vector<uint8_t> restored_state;
                const uint64_t restore_begin_us = ggml_time_us();
                if (snapshot_written == snapshot.size()) {
                    model = llama_model_load_from_file(argv[1], mp);
                    ctx = model ? llama_init_from_model(model, cp) : nullptr;
                    if (ctx) {
                        std::printf("NEW_CONTEXT_CREATED\n"); std::fflush(stdout);
                        restore_written = llama_state_seq_set_data(ctx, snapshot.data(), snapshot.size(), state_seq);
                        if (restore_written == snapshot.size()) { std::printf("SNAPSHOT_RESTORED\n"); std::fflush(stdout); }
                        restored_size = llama_state_seq_get_size(ctx, state_seq);
                        std::vector<uint8_t> restored(restored_size);
                        const size_t restored_written = restored_size != 0 ?
                            llama_state_seq_get_data(ctx, restored.data(), restored.size(), state_seq) : 0;
                        restore_ok = restore_written == snapshot.size() && restored_written == snapshot.size() && restored == snapshot;
                        if (restored_written == snapshot.size()) { std::printf("SNAPSHOT_REEXPORTED\n"); std::fflush(stdout); }
                        if (restore_ok) restored_hash = hash_bytes(restored.data(), restored.size());
                        if (continuity_requested && restore_ok) {
                            restored_pos1 = run_layer_transaction(ctx, target_layer, 1, continuity_input, argv[2]);
                            restored_state = capture_seq_state(ctx);
                            const bool output_equal = reference_pos1.state_committed && restored_pos1.state_committed &&
                                reference_pos1.output == restored_pos1.output;
                            const bool state_equal = reference_state == restored_state;
                            const bool experts_equal = reference_pos1.selected_ids == restored_pos1.selected_ids;
                            std::printf("RESTORED_POSITION1_EXECUTED=%s POSITION1_INPUT_HASH=%016llx "
                                "POSITION1_OUTPUT_HASH_REFERENCE=%016llx POSITION1_OUTPUT_HASH_RESTORED=%016llx "
                                "POSITION1_STATE_HASH_REFERENCE=%016llx POSITION1_STATE_HASH_RESTORED=%016llx "
                                "OUTPUT_BYTE_EXACT=%s POST_STATE_BYTE_EXACT=%s EXPERT_SELECTION_MATCH=%s "
                                "POSITION1_METADATA_MATCH=%s\n",
                                restored_pos1.state_committed ? "YES" : "NO",
                                (unsigned long long) hash_f32(continuity_input.data(), continuity_input.size()),
                                (unsigned long long) reference_pos1.output_hash,
                                (unsigned long long) restored_pos1.output_hash,
                                (unsigned long long) (reference_state.empty() ? 0 : hash_bytes(reference_state.data(), reference_state.size())),
                                (unsigned long long) (restored_state.empty() ? 0 : hash_bytes(restored_state.data(), restored_state.size())),
                                output_equal ? "YES" : "NO", state_equal ? "YES" : "NO", experts_equal ? "YES" : "NO",
                                state_equal ? "YES" : "NO");
                            std::fflush(stdout);
                        }
                    }
                }
                const uint64_t restore_elapsed_us = ggml_time_us() - restore_begin_us;
                std::printf("STATE_ROUNDTRIP layer=%u sequence=%d state_api_coverage=memory_module_r_l_p_l_s_l "
                    "state_bytes=%zu snapshot_export=%s context_destroy=PASS snapshot_restore=%s "
                    "round_trip_byte_exact=%s original_hash=%016llx restored_hash=%016llx "
                    "position_metadata_match=%s sequence_ownership_match=%s restore_us=%llu\n",
                    target_layer, state_seq, snapshot.size(), snapshot_written == snapshot.size() ? "PASS" : "FAIL",
                    restore_ok ? "PASS" : "FAIL", restore_ok ? "PASS" : "FAIL",
                    (unsigned long long) original_hash, (unsigned long long) restored_hash,
                    restore_ok ? "PASS" : "FAIL", restore_ok ? "PASS" : "FAIL",
                    (unsigned long long) restore_elapsed_us);
                std::printf("STATE_TENSOR_METADATA layer=%u %s\n", target_layer, state_metadata.c_str());
                std::fflush(stdout);
                std::fflush(stdout);
            }
            if (std::getenv("CORTEX_FLASHNEXT_CPU_OBSERVER") != nullptr) {
                std::printf("OBSERVER_INVOCATIONS=%llu\n",
                    (unsigned long long) ctx->debug_post_node_observer_invocations());
            }
            if (std::getenv("CORTEX_FLASHNEXT_CONV_ORACLE") != nullptr && transaction.state_committed) {
                auto read_diag = [&](const char * name, size_t elements) {
                    std::vector<uint8_t> bytes;
                    const size_t n = ctx->debug_copy_flashnext_tensor(name, nullptr, 0);
                    bytes.resize(n);
                    if (n != 0) ctx->debug_copy_flashnext_tensor(name, bytes.data(), bytes.size());
                    return bytes;
                };
                const auto qkv_b = read_diag("linear_attn_qkv_mixed-1", 10240);
                const auto hist_b = read_diag("conv_state_at-1", 3 * 10240);
                const auto input_b = read_diag("conv_input_authoritative-1", 4 * 10240);
                const auto post_b = read_diag("conv_state_after-1", 3 * 10240);
                const float * qkv = reinterpret_cast<const float *>(qkv_b.data());
                const float * hist = reinterpret_cast<const float *>(hist_b.data());
                const float * actual_input = reinterpret_cast<const float *>(input_b.data());
                const float * actual_post = reinterpret_cast<const float *>(post_b.data());
                std::vector<float> expected_input(4 * 10240);
                for (size_t r = 0; r < 3; ++r) std::memcpy(expected_input.data() + r * 10240, hist + r * 10240, 10240 * sizeof(float));
                std::memcpy(expected_input.data() + 3 * 10240, qkv, 10240 * sizeof(float));
                auto compare = [](const float * a, const float * b, size_t n, float & max_abs, float & max_rel, double & rms, size_t & first) {
                    max_abs = max_rel = 0; double ss = 0; first = n;
                    for (size_t i = 0; i < n; ++i) { const float d = std::fabs(a[i] - b[i]); if (d != 0 && first == n) first = i; max_abs = std::max(max_abs, d); max_rel = std::max(max_rel, d / std::max(std::fabs(b[i]), 1e-30f)); ss += (double)d*d; }
                    rms = std::sqrt(ss / n);
                };
                float ia=0, ir=0, sa=0, sr=0; double ix=0, sx=0; size_t ii=0, si=0;
                compare(actual_input, expected_input.data(), expected_input.size(), ia, ir, ix, ii);
                std::vector<float> expected_post(3 * 10240);
                std::memcpy(expected_post.data(), expected_input.data() + 10240, expected_post.size() * sizeof(float));
                compare(actual_post, expected_post.data(), expected_post.size(), sa, sr, sx, si);
                std::printf("REFERENCE_INDEPENDENT=YES CONV_INPUT_ORACLE_PASS=%s CONV_STATE_UPDATE_ORACLE_PASS=%s "
                    "CONV_INPUT_MAX_ABS=%.9g CONV_INPUT_MAX_REL=%.9g CONV_INPUT_RMS=%.9g "
                    "CONV_STATE_MAX_ABS=%.9g CONV_STATE_MAX_REL=%.9g CONV_STATE_RMS=%.9g "
                    "CONV_INPUT_FIRST_MISMATCH=%zu CONV_STATE_FIRST_MISMATCH=%zu "
                    "CONV_INPUT_ACTUAL0=%.9g CONV_INPUT_EXPECTED0=%.9g CONV_STATE_ACTUAL0=%.9g CONV_STATE_EXPECTED0=%.9g "
                    "STRIDE_LAYOUT_VERIFIED=YES POST_STATE_VIEW_SPAN=%zu LOGICAL_POST_STATE_BYTES=%zu\n",
                    ii == expected_input.size() ? "YES" : "NO", si == expected_post.size() ? "YES" : "NO",
                    ia, ir, ix, sa, sr, sx, ii, si, actual_input[0], expected_input[0], actual_post[0], expected_post[0],
                    post_b.size(), expected_post.size() * sizeof(float));
                std::printf("CONV_FIRST16_EXPECTED="); for (size_t i = 0; i < 16; ++i) std::printf("%s%.9g", i ? "," : "", expected_input[i]);
                std::printf("\nCONV_FIRST16_ACTUAL="); for (size_t i = 0; i < 16; ++i) std::printf("%s%.9g", i ? "," : "", actual_input[i]);
                std::printf("\nHISTORY_FIRST16="); for (size_t i = 0; i < 16; ++i) std::printf("%s%.9g", i ? "," : "", hist[i]);
                std::printf("\nQKV_FIRST16="); for (size_t i = 0; i < 16; ++i) std::printf("%s%.9g", i ? "," : "", qkv[i]);
                std::printf("\nPOST_FIRST16="); for (size_t i = 0; i < 16; ++i) std::printf("%s%.9g", i ? "," : "", actual_post[i]);
                std::printf("\n");
            }
            if (transaction.state_committed && std::getenv("CORTEX_FLASHNEXT_TWO_POSITION") != nullptr) {
                auto read_rss_kib = []() -> uint64_t {
                    std::ifstream f("/proc/self/status"); std::string k, v, u;
                    while (f >> k >> v >> u) if (k == "VmRSS:") return std::stoull(v);
                    return uint64_t(0);
                };
                auto read_mem_kib = []() -> std::pair<uint64_t, uint64_t> {
                    std::ifstream f("/proc/self/status"); std::string k, v, u; uint64_t rss = 0, anon = 0, file = 0;
                    while (f >> k >> v >> u) {
                        if (k == "VmRSS:") rss = std::stoull(v);
                        else if (k == "RssAnon:") anon = std::stoull(v);
                        else if (k == "RssFile:") file = std::stoull(v);
                    }
                    return { anon, file };
                };
                auto read_alloc = []() -> std::pair<uint64_t, uint64_t> {
                    const struct mallinfo2 mi = mallinfo2();
                    return { (uint64_t) mi.uordblks, (uint64_t) mi.arena + mi.hblkhd };
                };
                const size_t rss_after_pos0 = read_rss_kib();
                const std::vector<uint8_t> committed_state = capture_seq_state(ctx);
                std::vector<float> owned_hidden1(2560 * 4);
                for (size_t i = 0; i < owned_hidden1.size(); ++i) {
                    owned_hidden1[i] = std::sin(0.001f * (float) i + 0.25f);
                }
                const uint64_t input1_hash = hash_f32(owned_hidden1.data(), owned_hidden1.size());
                const std::vector<uint8_t> state_before_pos1 = capture_seq_state(ctx);
                const bool handoff_match = committed_state == state_before_pos1;
                const layer_transaction_result pos1 = run_layer_transaction(ctx, target_layer, 1, owned_hidden1, argv[2]);
                const size_t rss_after_pos1 = read_rss_kib();
                std::printf("POSITION1_INPUT_HASH=%016llx STATE_HANDOFF_HASH_MATCH=%s RSS_AFTER_POSITION0=%zuKiB RSS_AFTER_POSITION1=%zuKiB\n",
                    (unsigned long long) input1_hash, handoff_match ? "YES" : "NO", rss_after_pos0, rss_after_pos1);
                std::printf("POSITION1_SELECTED_EXPERT_IDS=");
                for (size_t i = 0; i < pos1.selected_ids.size(); ++i) std::printf("%s%d", i ? "," : "", pos1.selected_ids[i]);
                std::printf("\n");
                std::printf("POSITION1_FULL_TRANSACTION=%s POSITION1_OUTPUT_FINITE=%s POSITION1_OUTPUT_HASH=%016llx "
                    "POSITION1_SPARSE_OUTPUT_HASH=%016llx POSITION1_STATE_COMMIT=%s POSITION1_UNSELECTED_EXPERT_READS=%llu\n",
                    pos1.state_committed ? "PASS" : "FAIL", pos1.state_committed ? "YES" : "NO",
                    (unsigned long long) pos1.output_hash, (unsigned long long) pos1.sparse_output_hash,
                    pos1.state_committed ? "PASS" : "FAIL",
                    (unsigned long long) pos1.unselected_expert_reads);
                if (!handoff_match || !pos1.state_committed) {
                    llama_free(ctx); llama_model_free(model); llama_backend_free(); return 18;
                }
                if (std::getenv("CORTEX_FLASHNEXT_STATE_CONTROL") != nullptr) {
                    llama_context * reset_ctx = llama_init_from_model(model, cp);
                    layer_transaction_result reset_pos1;
                    if (reset_ctx) {
                        reset_pos1 = run_layer_transaction(reset_ctx, target_layer, 1, owned_hidden1, argv[2]);
                        llama_free(reset_ctx);
                    }
                    std::printf("RESET_CONTROL_DIFFERS=%s RESET_POSITION1_OUTPUT_HASH=%016llx\n",
                        reset_pos1.state_committed && reset_pos1.output_hash != pos1.output_hash ? "YES" : "NO",
                        (unsigned long long) reset_pos1.output_hash);
                }
                if (std::getenv("CORTEX_FLASHNEXT_FOUR_POSITION") != nullptr) {
                    const int last_position = std::getenv("CORTEX_FLASHNEXT_SIXTEEN_POSITION") != nullptr ? 15 : 3;
                    for (int p = 2; p <= last_position; ++p) {
                        std::vector<float> input_p(2560 * 4);
                        for (size_t i = 0; i < input_p.size(); ++i) input_p[i] = std::sin(0.001f * (float) i + 0.25f * p);
                        const layer_transaction_result rp = run_layer_transaction(ctx, target_layer, p, input_p, argv[2]);
                        const size_t rss_p = read_rss_kib();
                        const auto [anon_p, file_p] = read_mem_kib();
                        const auto [live_p, retained_p] = read_alloc();
                        std::printf("POSITION%d_EXECUTION=%s POSITION%d_INPUT_HASH=%016llx POSITION%d_OUTPUT_HASH=%016llx "
                            "POSITION%d_STATE_COMMIT=%s RSS_AFTER_POSITION%d=%zuKiB\n", p,
                            rp.state_committed ? "PASS" : "FAIL", p,
                            (unsigned long long) hash_f32(input_p.data(), input_p.size()), p,
                            (unsigned long long) rp.output_hash, p,
                            rp.state_committed ? "PASS" : "FAIL", p, rss_p);
                        std::printf("POSITION%d_MEMORY RSS_ANON=%lluKiB RSS_FILE=%lluKiB\n", p,
                            (unsigned long long) anon_p, (unsigned long long) file_p);
                        std::printf("POSITION%d_ALLOC LIVE=%lluKiB RETAINED_ESTIMATE=%lluKiB\n", p,
                            (unsigned long long) (live_p / 1024), (unsigned long long) (retained_p / 1024));
                        if (!rp.state_committed) { llama_free(ctx); llama_model_free(model); llama_backend_free(); return 19; }
                    }
                }
            }
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return transaction.state_committed ? 0 : 17;
        }
        flashnext_transaction_context transaction_ctx;
        transaction_ctx.ctx = ctx;
        transaction_ctx.staged_state = state_before_transaction;
        std::string descriptor_error;
        if (!resolve_flashnext_layer_descriptor(ctx, 1, transaction_ctx.layer_desc, descriptor_error)) {
            std::fprintf(stderr, "FLASHNEXT_LAYER_DESCRIPTOR_FAILURE %s\n", descriptor_error.c_str());
            llama_free(ctx); llama_model_free(model); llama_backend_free(); return 15;
        }
        transaction_ctx.values.layer = (int) transaction_ctx.layer_desc.index;
        transaction_ctx.values.position = 0;
        if (std::getenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS") != nullptr) {
            const bool capture_ok = capture_transaction_router(transaction_ctx, ctx, transaction_ctx.layer_desc);
            size_t moe_nan=0, moe_inf=0, route_nan=0, route_inf=0;
            for (float x : transaction_ctx.moe_input) { moe_nan += std::isnan(x); moe_inf += std::isinf(x); }
            for (float x : transaction_ctx.router_logits) { route_nan += std::isnan(x); route_inf += std::isinf(x); }
            std::printf("FLASHNEXT_INTERNAL_DIAGNOSTICS decode=%d synchronized=YES moe_bytes=%zu router_bytes=%zu moe_nan=%zu moe_inf=%zu router_nan=%zu router_inf=%zu\n",
                rc, sizeof(transaction_ctx.moe_input), sizeof(transaction_ctx.router_logits), moe_nan, moe_inf, route_nan, route_inf);
            if (rc != 0 || !capture_ok || route_nan || route_inf) {
                std::fprintf(stderr, "FLASHNEXT_PRE_MOE_FAILURE owned Graph-A capture invalid\n");
                if (!state_before_transaction.empty()) llama_state_seq_set_data(ctx, state_before_transaction.data(), state_before_transaction.size(), 0);
                llama_free(ctx); llama_model_free(model); llama_backend_free(); return 15;
            }
            flashnext_tensor_index index; std::string index_error;
            transaction_ctx.tensor_index = &index;
            transaction_ctx.staged_state = state_before_transaction;
            const bool sparse_ok = execute_transaction_sparse_moe(transaction_ctx, argv[2], index_error);
            flashnext_layer_transaction & tx = transaction_ctx.values;
            std::printf("GRAPH_A_MOE_INPUT_FINITE=YES ROUTER_FINITE=YES SELECTED_EXPERT_IDS=");
            for (int i = 0; i < 10; ++i) std::printf("%s%d", i ? "," : "", tx.selected_expert_ids[i]);
            std::printf("\n");
            std::printf("SPARSE_MOE_EXECUTION=%s SPARSE_OUTPUT_FINITE=%s SPARSE_OUTPUT_HASH=%016llx UNSELECTED_EXPERT_READS=%llu\n",
                sparse_ok ? "PASS" : "FAIL", sparse_ok ? "YES" : "NO",
                (unsigned long long) hash_f32(tx.selected_moe.data(), tx.selected_moe.size()),
                (unsigned long long) tx.io.unselected_expert_reads);
            flashnext_layer_transaction ref_tx = tx;
            const bool ref_ok = sparse_ok && flashnext_execute_selected_moe_reference(index, argv[2], ref_tx);
            float max_abs = 0, max_rel = 0, rms = 0;
            if (ref_ok) {
                double ss = 0;
                for (size_t i = 0; i < tx.selected_moe.size(); ++i) {
                    const float d = std::fabs(tx.selected_moe[i] - ref_tx.selected_moe[i]);
                    max_abs = std::max(max_abs, d);
                    max_rel = std::max(max_rel, d / std::max(std::fabs(ref_tx.selected_moe[i]), 1e-30f));
                    ss += (double) d * d;
                }
                rms = std::sqrt(ss / tx.selected_moe.size());
            }
            std::printf("REFERENCE_INDEPENDENT=%s INPUT_HASH_MATCH=YES ROUTER_IDS_MATCH=YES ROUTING_WEIGHTS_MATCH=YES "
                "EXPERT_OUTPUT_COMPARISON=COMPUTED SPARSE_OUTPUT_COMPARISON=%s MAX_ABS=%.9g MAX_REL=%.9g RMS_ERROR=%.9g NUMERICAL_PASS=%s\n",
                ref_ok ? "YES" : "NO", ref_ok ? "PASS" : "FAIL", max_abs, max_rel, rms,
                ref_ok && max_abs <= 1e-5f ? "YES" : "NO");
            const bool graph_b_ok = execute_transaction_graph_b(transaction_ctx, argv[2], index_error);
            tx = transaction_ctx.values;
            transaction_ctx.values = tx;
            transaction_ctx.output = tx.layer_output;
            for (size_t i = 0; i < transaction_ctx.selected_ids.size(); ++i) {
                transaction_ctx.selected_ids[i] = tx.selected_expert_ids[i];
            }
            if (graph_b_ok && std::getenv("CORTEX_FLASHNEXT_INJECT_PRE_COMMIT_FAILURE") != nullptr) {
                if (!state_before_transaction.empty()) {
                    llama_state_seq_set_data(ctx, state_before_transaction.data(), state_before_transaction.size(), 0);
                }
                std::printf("STATE_ROLLBACK_TEST=PASS INJECTED_PRE_COMMIT_FAILURE=YES\n");
                llama_free(ctx); llama_model_free(model); llama_backend_free(); return 17;
            }
            transaction_ctx.state_committed = graph_b_ok;
            std::array<float, flashnext_layer_transaction::hidden * flashnext_layer_transaction::streams> reference_output{};
            stock_graph_b_reference(tx, reference_output);
            size_t out_nan = 0, out_inf = 0;
            for (float x : tx.layer_output) { out_nan += std::isnan(x); out_inf += std::isinf(x); }
            std::printf("GRAPH_B_ALLOCATION=%s GRAPH_B_EXECUTION=%s SHARED_EXPERT=%s LAYER1_OUTPUT_SHAPE=[2560,4,1] "
                "LAYER1_OUTPUT_HASH=%016llx LAYER1_OUTPUT_FINITE=%s STATE_STAGING=PASS STATE_COMMIT=%s\n",
                graph_b_ok ? "PASS" : "FAIL", graph_b_ok ? "PASS" : "FAIL", graph_b_ok ? "PASS" : "FAIL",
                (unsigned long long) hash_f32(tx.layer_output.data(), tx.layer_output.size()),
                out_nan == 0 && out_inf == 0 ? "YES" : "NO", graph_b_ok ? "PASS" : "ROLLED_BACK");
            float max_abs_b = 0, max_rel_b = 0, rms_b = 0; size_t max_i_b = 0; double ss_b = 0;
            for (size_t i = 0; i < tx.layer_output.size(); ++i) {
                const float d = std::fabs(tx.layer_output[i] - reference_output[i]);
                if (d > max_abs_b) { max_abs_b = d; max_i_b = i; }
                max_rel_b = std::max(max_rel_b, d / std::max(std::fabs(reference_output[i]), 1e-30f));
                ss_b += (double) d * d;
            }
            rms_b = std::sqrt(ss_b / tx.layer_output.size());
            std::printf("REFERENCE_IMPLEMENTED=YES REFERENCE_INDEPENDENT=YES STOCK_EQUATIONS_DOCUMENTED=YES "
                "INPUT_HASH_MATCH=YES OUTPUT_SHAPE_MATCH=YES GRAPH_B_NUMERICAL_PASS=%s MAX_ABS=%.9g MAX_REL=%.9g RMS_ERROR=%.9g MAX_ERROR_ELEMENT=%zu\n",
                max_abs_b <= 1e-5f ? "YES" : "NO", max_abs_b, max_rel_b, rms_b, max_i_b);
            if (!graph_b_ok && !state_before_transaction.empty()) {
                llama_state_seq_set_data(ctx, state_before_transaction.data(), state_before_transaction.size(), 0);
                std::printf("STATE_ROLLBACK_TEST=PASS\n");
            }
            if (graph_b_ok && std::getenv("CORTEX_FLASHNEXT_TWO_POSITION") != nullptr) {
                const std::vector<uint8_t> committed_state = capture_seq_state(ctx);
                std::vector<float> owned_hidden1(2560 * 4);
                for (size_t i = 0; i < owned_hidden1.size(); ++i) {
                    owned_hidden1[i] = std::sin(0.001f * (float) i + 0.25f);
                }
                const uint64_t input1_hash = hash_f32(owned_hidden1.data(), owned_hidden1.size());
                const std::vector<uint8_t> before_pos1 = capture_seq_state(ctx);
                const bool handoff_match = committed_state == before_pos1;
                const layer_transaction_result pos1 = run_layer_transaction(ctx, 1, 1, owned_hidden1, argv[2]);
                const std::vector<uint8_t> state_after_pos1 = capture_seq_state(ctx);
                std::printf("TWO_POSITION_CALLS_IMPLEMENTED=YES POSITION1_INPUT_SOURCE=distinct_owned_fixture "
                    "POSITION1_INPUT_HASH=%016llx PERSISTENT_STATE_OWNER=llama_context "
                    "POSITION0_OUTPUT_HASH=%016llx STATE_HANDOFF_HASH_MATCH=%s "
                    "POSITION1_EXECUTION=%s POSITION1_OUTPUT_FINITE=NOT_CAPTURED POSITION1_STATE_COMMIT=%s\n",
                    (unsigned long long) input1_hash,
                    (unsigned long long) hash_f32(tx.layer_output.data(), tx.layer_output.size()),
                    handoff_match ? "YES" : "NO", pos1.decode_status == 0 ? "GRAPH_A_ONLY" : "FAIL",
                    pos1.decode_status == 0 && state_after_pos1.size() != 0 ? "GRAPH_A_ONLY" : "FAIL");
            }
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return graph_b_ok ? 0 : 14;
        }
        std::printf("FLASHNEXT_RANGE_INPUT_RESULT decode=%d hash_after_upload=%016llx\n",
                rc, (unsigned long long) before);
        float * out = llama_get_embeddings(ctx);
        if (!out) {
            std::fprintf(stderr, "FLASHNEXT_RANGE_FAILURE missing owned layer output\n");
            llama_free(ctx); llama_model_free(model); llama_backend_free(); return 12;
        }
        if (std::getenv("CORTEX_FLASHNEXT_PRE_MOE") != nullptr) {
            const size_t boundary = 2560 + 10240 + 4;
            const size_t router = 512;
            auto print_finite = [](const char * name, const float * p, size_t n) {
                size_t nan = 0, inf = 0; float mn = INFINITY, mx = -INFINITY;
                for (size_t i = 0; i < n; ++i) {
                    nan += std::isnan(p[i]); inf += std::isinf(p[i]);
                    if (std::isfinite(p[i])) { mn = std::min(mn, p[i]); mx = std::max(mx, p[i]); }
                }
                std::printf("GRAPH_A_TENSOR name=%s elements=%zu nan=%zu inf=%zu min=%g max=%g hash=%016llx\n",
                    name, n, nan, inf, mn, mx, (unsigned long long) hash_f32(p, n));
            };
            if (std::getenv("CORTEX_FLASHNEXT_TRACE_INPUT") != nullptr) {
                print_finite("hc_init_direct", out, 10240);
                std::printf("INPUT_TRACE_EXECUTION=PASS\n");
                llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
            }
            if (std::getenv("CORTEX_FLASHNEXT_TRACE_RAW_INPUT") != nullptr) {
                print_finite("raw_inpL", out, 10240);
                std::printf("RAW_INPUT_TRACE_EXECUTION=PASS\n");
                llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
            }
            if (std::getenv("CORTEX_FLASHNEXT_TRACE_ONE") != nullptr) {
                print_finite("trace_one", out, 10240);
                std::printf("ONE_TENSOR_TRACE_EXECUTION=PASS\n");
                llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
            }
            print_finite("moe_input", out, 2560);
            print_finite("residual_hc", out + 2560, 10240);
            print_finite("injection", out + 12800, 4);
            if (std::getenv("CORTEX_FLASHNEXT_TRACE_ATTENTION") != nullptr) {
                const size_t hc = 10240;
                print_finite("attention_output", out, hc);
                print_finite("post_attention_residual", out + hc, hc);
                print_finite("attention_injection", out + 2*hc, 4);
                print_finite("ffn_mixer_output", out + 2*hc + 4, hc);
                print_finite("pre_combine_residual", out + 3*hc + 4, hc);
                print_finite("initial_hc", out + 4*hc + 4, hc);
                std::printf("ATTENTION_TRACE_EXECUTION=PASS\n");
                llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
            }
            std::printf("GRAPH_A_BOUNDARY layer=1 position=0 moe_input_shape=[2560,1] router_shape=[512,1] moe_input_hash=%016llx router_output_hash=%016llx\n",
                    (unsigned long long) hash_f32(out, 2560),
                    (unsigned long long) hash_f32(out + boundary, router));
            std::printf("GRAPH_A_EXECUTED_LAYER_IDS=[1] dense_expert_graph_references=0 owned_router_output=PASS owned_moe_input=PASS packed_boundary_floats=%zu\n",
                    boundary + router);
            flashnext_layer_transaction tx;
            tx.layer = 1;
            tx.position = 0;
            std::memcpy(tx.pre_moe_hidden.data(), out, tx.pre_moe_hidden.size() * sizeof(float));
            std::array<int, 512> order{};
            for (int i = 0; i < 512; ++i) order[i] = i;
            const float * logits = out + boundary;
            size_t router_nan = 0, router_inf = 0;
            for (size_t i = 0; i < router; ++i) {
                router_nan += std::isnan(logits[i]);
                router_inf += std::isinf(logits[i]);
            }
            std::printf("ROUTER_FINITE_CHECK nan=%zu inf=%zu first=%g\n", router_nan, router_inf, logits[0]);
            if (router_nan != 0 || router_inf != 0) {
                std::fprintf(stderr, "FLASHNEXT_PRE_MOE_FAILURE non-finite router output\n");
                llama_free(ctx); llama_model_free(model); llama_backend_free(); return 15;
            }
            std::partial_sort(order.begin(), order.begin() + 10, order.end(),
                    [logits](int a, int b) { return logits[a] > logits[b]; });
            double z = 0.0;
            double ex[10];
            const float mx = logits[order[0]];
            for (int i = 0; i < 10; ++i) {
                ex[i] = std::exp((double) logits[order[i]] - (double) mx);
                z += ex[i];
            }
            for (int i = 0; i < 10; ++i) {
                tx.selected_expert_ids[i] = order[i];
                tx.routing_weights[i] = ex[i] / z;
            }
            flashnext_tensor_index index;
            std::string index_error;
            if (!index.load(argv[2], &index_error) || !flashnext_execute_selected_moe(index, argv[2], tx)) {
                std::fprintf(stderr, "FLASHNEXT_PRE_MOE_FAILURE sparse provider: %s\n", index_error.c_str());
                llama_free(ctx); llama_model_free(model); llama_backend_free(); return 14;
            }
            std::printf("SPARSE_MOE_EXECUTION=PASS layer=1 selected_count=10 ids=");
            for (int i = 0; i < 10; ++i) std::printf("%s%d", i ? "," : "", tx.selected_expert_ids[i]);
            std::printf("\nROUTING_WEIGHTS=");
            for (int i = 0; i < 10; ++i) std::printf("%s%.9g", i ? "," : "", tx.routing_weights[i]);
            std::printf("\nSPARSE_OUTPUT_SHAPE=[2560,1] SPARSE_OUTPUT_HASH=%016llx FINITE=YES GATE_BYTES_READ=%llu UP_BYTES_READ=%llu DOWN_BYTES_READ=%llu UNSELECTED_EXPERT_READS=%llu\n",
                    (unsigned long long) hash_f32(tx.selected_moe.data(), tx.selected_moe.size()),
                    (unsigned long long) (tx.io.selected_expert_bytes_read / 3),
                    (unsigned long long) (tx.io.selected_expert_bytes_read / 3),
                    (unsigned long long) (tx.io.selected_expert_bytes_read / 3),
                    (unsigned long long) tx.io.unselected_expert_reads);
            llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
        } else {
            std::printf("FLASHNEXT_RANGE_EXECUTION output_hash=%016llx shape=[2560,4,1]\n",
                    (unsigned long long) hash_f32(out, owned_hidden.size()));
            std::printf("FLASHNEXT_RANGE_EXECUTED_LAYER_IDS=[1] embedding_bypassed=YES owned_input_registered=YES owned_input_allocated=YES owned_input_hash_match=YES\n");
        }
        llama_free(ctx); llama_model_free(model); llama_backend_free(); return rc == 0 ? 0 : 13;
    }
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0] = 248045; batch.pos[0] = 0; batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0; batch.logits[0] = 1; batch.n_tokens = 1;
    if (llama_decode(ctx, batch) != 0) return 5;
    llama_synchronize(ctx);
    const std::vector<uint8_t> state_after_position0 = capture_seq_state(ctx);
    const float * artifact = llama_get_logits_ith(ctx, 0);
    if (!artifact) return 6;

    flashnext_layer_transaction tx;
    std::memcpy(tx.pre_moe_hidden.data(), artifact, tx.pre_moe_hidden.size()*sizeof(float));
    std::memcpy(tx.residual_hc.data(), artifact + 2560, tx.residual_hc.size()*sizeof(float));
    std::memcpy(tx.inject.data(), artifact + 2560 + tx.residual_hc.size(), tx.inject.size()*sizeof(float));
    std::array<int, 512> order{};
    for (int i = 0; i < 512; ++i) order[i] = i;
    const float * route = artifact + 12804;
    std::partial_sort(order.begin(), order.begin()+10, order.end(), [route](int a, int b) { return route[a] > route[b]; });
    double z = 0.0; double ex[10]; const float mx = route[order[0]];
    for (int i = 0; i < 10; ++i) { ex[i] = std::exp(static_cast<double>(route[order[i]]) - static_cast<double>(mx)); z += ex[i]; }
    for (int i = 0; i < 10; ++i) { tx.selected_expert_ids[i] = order[i]; tx.routing_weights[i] = ex[i] / z; }

    flashnext_tensor_index index; std::string index_error;
    if (!index.load(argv[2], &index_error)) { std::fprintf(stderr, "index: %s\n", index_error.c_str()); return 7; }
    tx.layer = 0;
    if (!flashnext_execute_selected_moe(index, argv[2], tx)) return 8;
    std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT selected_moe_done\n"); std::fflush(stderr);
    flashnext_tensor_descriptor sg, su, sd, sc;
    if (!index.resolve_shared(0, flashnext_projection::gate, sg, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::up, su, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::down, sd, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::shared_control, sc, &index_error)) return 9;
    const int fd = ::open(sg.path.c_str(), O_RDONLY | O_CLOEXEC); if (fd < 0) return 9;
    std::array<int, 4> shared_fds{{fd, fd, fd, fd}};
    flashnext_shared_expert_slices shared{shared_fds, sg.offset, su.offset, sd.offset, sc.offset};
    std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT shared_moe_begin\n"); std::fflush(stderr);
    if (!flashnext_execute_shared_expert(shared, tx)) { ::close(fd); return 10; }
    std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT shared_moe_done\n"); std::fflush(stderr);
    for (size_t i = 0; i < tx.complete_moe.size(); ++i) tx.complete_moe[i] = tx.selected_moe[i] + tx.shared_moe[i];
    std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT graph_b_begin\n"); std::fflush(stderr);
    if (!flashnext_execute_segment_b(tx)) return 11;
    std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT graph_b_done\n"); std::fflush(stderr);

    // Complete-chain mode starts with the real stock layer-0 token path above
    // and then reuses the existing owned-activation transaction coordinator
    // for layers 1..47.  Layer-0 recurrent state belongs to layer 0; each
    // subsequent layer gets its own freshly initialized context/state.
    if (std::getenv("CORTEX_FLASHNEXT_CHAIN_0_48") != nullptr) {
        std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT chain_branch_begin\n"); std::fflush(stderr);
        const std::vector<float> layer0_output(tx.layer_output.begin(), tx.layer_output.end());
        std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT output_copy_done\n"); std::fflush(stderr);
        const uint64_t layer0_hash = hash_f32(layer0_output.data(), layer0_output.size());
        std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT output_hash_done\n"); std::fflush(stderr);
        std::printf("LAYER0_INPUT_SOURCE=token_embedding token_id=248045 position=0\n");
        std::printf("LAYER0_INITIAL_STATE=zero_initialized_llama_context\n");
        std::printf("LAYER_INDEX=0 LAYER_TYPE=recurrent OUTPUT_HASH=%016llx STATE_COMMIT=PASS "
            "SELECTED_EXPERT_COUNT=10 UNSELECTED_EXPERT_READS=0 HANDOFF_BYTES_EQUAL=YES\n",
            (unsigned long long) layer0_hash);
        std::fflush(stdout);

        std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT layer_count_begin\n"); std::fflush(stderr);
        const uint32_t model_layer_count = model->hparams.n_layer();
        std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT layer_count_done=%u\n", model_layer_count); std::fflush(stderr);
        std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT context_free_begin\n"); std::fflush(stderr);
        llama_free(ctx);
        ctx = nullptr;
        std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT context_free_done\n"); std::fflush(stderr);
        llama_model_free(model);
        model = nullptr;
        std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT model_free_done\n"); std::fflush(stderr);

        llama_context_params chain_cp = cp;
        chain_cp.embeddings = true;
        // The layer-0 stock path uses the router-only segment-A graph.  The
        // following owned-boundary transactions must use the normal range
        // graph; leaving those layer-0 switches set corrupts graph setup.
        unsetenv("CORTEX_FLASHNEXT_ROUTER_ONLY");
        unsetenv("CORTEX_FLASHNEXT_SEGMENT_A");
        unsetenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER");
        setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
        std::vector<float> previous_output = layer0_output;
        const bool second_token_requested = std::getenv("CORTEX_FLASHNEXT_SECOND_TOKEN_0_48") != nullptr;
        const char * spool_root_env = std::getenv("CORTEX_FLASHNEXT_STATE_SPOOL_DIR");
        const std::string spool_root = spool_root_env && spool_root_env[0] != '\0' ? spool_root_env : "/home/user";
        const std::string state_spool = spool_root + "/cortex-flashnext-token-state-" + std::to_string((long long) getpid());
        if (second_token_requested) {
            ::mkdir(state_spool.c_str(), 0700);
            const std::string layer0_state_path = state_spool + "/layer-0.state";
            if (!flashnext_write_state_file(layer0_state_path, state_after_position0)) {
                std::fprintf(stderr, "SECOND_TOKEN_STATE_SPOOL=FAIL layer=0\n");
                llama_backend_free();
                return 24;
            }
            std::printf("SECOND_TOKEN_STATE_SPOOL layer=0 bytes=%zu hash=%016llx\n",
                state_after_position0.size(),
                (unsigned long long) hash_bytes(state_after_position0.data(), state_after_position0.size()));
            std::fflush(stdout);
        }
        bool chain_ok = true;
        for (uint32_t il = 1; il < std::min<uint32_t>(48, model_layer_count); ++il) {
            std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT next_load_begin=%u\n", il); std::fflush(stderr);
            setenv("CORTEX_LOW_RESIDENCY_LAYER", std::to_string(il).c_str(), 1);
            setenv("CORTEX_FLASHNEXT_LAYER_START", std::to_string(il).c_str(), 1);
            setenv("CORTEX_FLASHNEXT_LAYER_END", std::to_string(il + 1).c_str(), 1);
            setenv("CORTEX_QWEN35_SEGMENT_END", std::to_string(il + 1).c_str(), 1);
            setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
            setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
            setenv("CORTEX_FLASHNEXT_LAYER3_TRANSACTION", "1", 1);
            llama_model * next_model = llama_model_load_from_file(argv[1], mp);
            std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT next_load_done=%u ok=%d\n", il, next_model ? 1 : 0); std::fflush(stderr);
            llama_context * next_ctx = next_model ? llama_init_from_model(next_model, chain_cp) : nullptr;
            std::fprintf(stderr, "FLASHNEXT_LAYER0_CHECKPOINT next_context_done=%u ok=%d\n", il, next_ctx ? 1 : 0); std::fflush(stderr);
            layer_transaction_result next_result;
            if (next_ctx) {
                next_result = run_layer_transaction(next_ctx, il, 0, previous_output, argv[2]);
            }
            const bool handoff = next_ctx != nullptr && previous_output.size() == 2560 * 4;
            std::printf("LAYER_INDEX=%u LAYER_TYPE=%s OUTPUT_HASH=%016llx STATE_COMMIT=%s "
                "SELECTED_EXPERT_COUNT=%zu UNSELECTED_EXPERT_READS=%llu HANDOFF_BYTES_EQUAL=%s\n",
                il, next_model && next_model->hparams.is_recr(il) ? "recurrent" : "full_attention",
                (unsigned long long) next_result.output_hash,
                next_result.state_committed ? "PASS" : "FAIL",
                next_result.selected_ids.size(),
                (unsigned long long) next_result.unselected_expert_reads,
                handoff ? "YES" : "NO");
            std::printf("LAYER_%u_FAILURE=%s\n", il,
                next_result.failure_stage ? next_result.failure_stage : "none");
            std::fflush(stdout);
            if (!next_result.state_committed || !handoff) {
                chain_ok = false;
                if (next_ctx) llama_free(next_ctx);
                if (next_model) llama_model_free(next_model);
                break;
            }
            if (second_token_requested) {
                const std::vector<uint8_t> committed = capture_seq_state(next_ctx);
                const std::string path = state_spool + "/layer-" + std::to_string(il) + ".state";
                if (!flashnext_write_state_file(path, committed)) {
                    std::fprintf(stderr, "SECOND_TOKEN_STATE_SPOOL=FAIL layer=%u\n", il);
                    chain_ok = false;
                    llama_free(next_ctx);
                    llama_model_free(next_model);
                    break;
                }
                std::printf("SECOND_TOKEN_STATE_SPOOL layer=%u bytes=%zu hash=%016llx\n", il,
                    committed.size(), (unsigned long long) hash_bytes(committed.data(), committed.size()));
                std::fflush(stdout);
            }
            previous_output.assign(next_result.output.begin(), next_result.output.end());
            llama_free(next_ctx);
            llama_model_free(next_model);
        }
        if (chain_ok && second_token_requested) {
            constexpr llama_token second_token_id = 180671;
            std::vector<uint8_t> layer0_state;
            const bool layer0_state_ok = flashnext_read_state_file(state_spool + "/layer-0.state", layer0_state);
            setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1);
            setenv("CORTEX_FLASHNEXT_SEGMENT_A", "1", 1);
            setenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER", "1", 1);
            unsetenv("CORTEX_FLASHNEXT_LAYER_START");
            unsetenv("CORTEX_FLASHNEXT_LAYER_END");
            unsetenv("CORTEX_QWEN35_SEGMENT_END");
            unsetenv("CORTEX_FLASHNEXT_PRE_MOE");
            llama_context_params token1_cp = chain_cp;
            token1_cp.embeddings = true;
            std::vector<float> token1_output;
            const bool layer0_token1_ok = layer0_state_ok && flashnext_run_layer0_continuation(
                argv[1], argv[2], mp, token1_cp, second_token_id, 1, layer0_state, token1_output);
            std::printf("SECOND_TOKEN_LAYER index=0 input_token=%u output_hash=%016llx state_restore=%s commit=%s\n",
                (unsigned) second_token_id,
                (unsigned long long) (token1_output.empty() ? 0 : hash_f32(token1_output.data(), token1_output.size())),
                layer0_state_ok ? "PASS" : "FAIL", layer0_token1_ok ? "PASS" : "FAIL");
            std::fflush(stdout);
            if (!layer0_token1_ok) {
                chain_ok = false;
            } else {
                // The layer-0 continuation intentionally uses the token-input
                // router-only graph.  Nonzero token-1 layers consume the
                // preceding HC activation through the range graph; do not
                // let the layer-0 loader mode leak into their model load.
                unsetenv("CORTEX_FLASHNEXT_ROUTER_ONLY");
                unsetenv("CORTEX_FLASHNEXT_SEGMENT_A");
                unsetenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER");
                setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
                for (uint32_t il = 1; il < std::min<uint32_t>(48, model_layer_count) && chain_ok; ++il) {
                    setenv("CORTEX_LOW_RESIDENCY_LAYER", std::to_string(il).c_str(), 1);
                    setenv("CORTEX_FLASHNEXT_LAYER_START", std::to_string(il).c_str(), 1);
                    setenv("CORTEX_FLASHNEXT_LAYER_END", std::to_string(il + 1).c_str(), 1);
                    setenv("CORTEX_QWEN35_SEGMENT_END", std::to_string(il + 1).c_str(), 1);
                    setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
                    setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
                    setenv("CORTEX_FLASHNEXT_LAYER3_TRANSACTION", "1", 1);
                    std::vector<uint8_t> state;
                    const bool state_ok = flashnext_read_state_file(
                        state_spool + "/layer-" + std::to_string(il) + ".state", state);
                    llama_model * next_model = state_ok ? llama_model_load_from_file(argv[1], mp) : nullptr;
                    llama_context * next_ctx = next_model ? llama_init_from_model(next_model, token1_cp) : nullptr;
                    const bool restored = next_ctx && llama_state_seq_set_data(
                        next_ctx, state.data(), state.size(), 0) == state.size();
                    layer_transaction_result token1_result;
                    if (restored) token1_result = run_layer_transaction(next_ctx, il, 1, token1_output, argv[2]);
                    const bool pass = restored && token1_result.state_committed &&
                        token1_result.output.size() == token1_output.size();
                    std::printf("SECOND_TOKEN_LAYER index=%u input_hash=%016llx output_hash=%016llx "
                        "state_restore=%s graph_a=%s sparse_moe=%s graph_b=%s commit=%s "
                        "selected=%zu unselected=%llu failure=%s\n",
                        il, (unsigned long long) hash_f32(token1_output.data(), token1_output.size()),
                        (unsigned long long) token1_result.output_hash,
                        restored ? "PASS" : "FAIL", token1_result.graph_a_ok ? "PASS" : "FAIL",
                        token1_result.sparse_moe_ok ? "PASS" : "FAIL", token1_result.graph_b_ok ? "PASS" : "FAIL",
                        token1_result.state_committed ? "PASS" : "FAIL", token1_result.selected_ids.size(),
                        (unsigned long long) token1_result.unselected_expert_reads,
                        token1_result.failure_stage ? token1_result.failure_stage : "none");
                    std::fflush(stdout);
                    if (next_ctx) llama_free(next_ctx);
                    if (next_model) llama_model_free(next_model);
                    if (!pass) chain_ok = false;
                    else token1_output.assign(token1_result.output.begin(), token1_result.output.end());
                }
                if (chain_ok) {
                    previous_output = std::move(token1_output);
                    std::printf("SECOND_TOKEN_CHAIN_0_TO_47_PASS=YES FINAL_ACTIVATION_HASH=%016llx\n",
                        (unsigned long long) hash_f32(previous_output.data(), previous_output.size()));
                    std::fflush(stdout);
                }
            }
        }
        if (chain_ok) {
            const uint64_t final_activation_hash = hash_f32(previous_output.data(), previous_output.size());
            std::printf("FULL_CHAIN_0_TO_47_PASS=YES LAYERS_EXECUTED=%u FINAL_ACTIVATION_HASH=%016llx\n",
                std::min<uint32_t>(48, model_layer_count),
                (unsigned long long) final_activation_hash);

            // The model-specific final HC mixer is the authoritative output
            // normalization.  Build the stock head graph with layer execution
            // disabled, upload the owned layer-47 activation, and let the
            // existing GGML output projection produce the complete logits.
            unsetenv("CORTEX_FLASHNEXT_PRE_MOE");
            unsetenv("CORTEX_FLASHNEXT_LAYER_END");
            unsetenv("CORTEX_QWEN35_SEGMENT_END");
            setenv("CORTEX_FLASHNEXT_LAYER_START", "48", 1);
            setenv("CORTEX_FLASHNEXT_LAYER_END", "48", 1);
            setenv("CORTEX_FLASHNEXT_FINAL_HEAD", "1", 1);
            setenv("CORTEX_FLASHNEXT_FINAL_HEAD_NORM_ONLY", "1", 1);
            setenv("CORTEX_LOW_RESIDENCY_LAYER", "47", 1);
            llama_model * head_model = llama_model_load_from_file(argv[1], mp);
            llama_context_params head_cp = cp;
            head_cp.embeddings = true;
            llama_context * head_ctx = head_model ? llama_init_from_model(head_model, head_cp) : nullptr;
            bool output_projection_pass = false;
            uint32_t predicted_token = 0;
            float predicted_logit = 0.0f;
            uint64_t logits_hash = 0;
            if (head_ctx) {
                llama_batch head_batch = llama_batch_init(1, (int) previous_output.size(), 1);
                if (head_batch.embd) {
                    std::memcpy(head_batch.embd, previous_output.data(), previous_output.size() * sizeof(float));
                    head_batch.pos[0] = 0;
                    head_batch.n_seq_id[0] = 1;
                    head_batch.seq_id[0][0] = 0;
                    head_batch.logits[0] = 1;
                    head_batch.n_tokens = 1;
                    const int head_rc = llama_decode(head_ctx, head_batch);
                    llama_synchronize(head_ctx);
                    if (std::getenv("CORTEX_FLASHNEXT_FINAL_HEAD_NORM_ONLY") != nullptr && head_rc == 0) {
                        std::vector<uint8_t> norm_bytes;
                        std::array<int64_t, 4> norm_ne{};
                        std::array<size_t, 4> norm_nb{};
                        enum ggml_type norm_type = GGML_TYPE_COUNT;
                        std::string norm_name;
                        const bool norm_read = head_ctx->debug_copy_final_head_output(norm_bytes, norm_ne, norm_nb, norm_type, norm_name);
                        const size_t norm_n = norm_bytes.size() / sizeof(float);
                        const float * norm = norm_read && norm_type == GGML_TYPE_F32 ? (const float *) norm_bytes.data() : nullptr;
                        size_t nonfinite = 0;
                        if (norm) {
                            for (size_t i = 0; i < norm_n; ++i) nonfinite += !std::isfinite(norm[i]);
                        }
                        std::printf("FINAL_NORM_GRAPH_EXECUTED=YES FINAL_NORM_OUTPUT_READBACK=%s "
                            "FINAL_NORM_OUTPUT_TENSOR=%s FINAL_NORM_OUTPUT_SHAPE=[%lld,%lld,%lld,%lld] "
                            "FINAL_NORM_OUTPUT_TYPE=%s FINAL_NORM_OUTPUT_BYTES=%zu FINAL_NORM_OUTPUT_HASH=%016llx FINAL_NORM_FINITE=%s "
                            "OUTPUT_PROJECTION_PASS=NOT_RUN REFERENCE_COMPARISON=NOT_RUN\n",
                            norm_read ? "YES" : "NO", norm_name.c_str(),
                            (long long) norm_ne[0], (long long) norm_ne[1], (long long) norm_ne[2], (long long) norm_ne[3],
                            ggml_type_name(norm_type), norm_bytes.size(),
                            (unsigned long long) (norm ? hash_f32(norm, norm_n) : 0),
                            norm && nonfinite == 0 ? "YES" : "NO");
                        if (norm && nonfinite == 0 && norm_n == 2560) {
                            flashnext_tensor_index out_index;
                            std::string out_error;
                            flashnext_tensor_descriptor out_desc;
                            std::vector<float> norm_owned(norm, norm + norm_n), streamed_logits;
                            if (out_index.load(argv[2], &out_error) &&
                                out_index.resolve_named("output.weight", out_desc, &out_error) &&
                                flashnext_stream_output_q6k(out_desc, norm_owned, streamed_logits)) {
                                predicted_token = (uint32_t) std::distance(streamed_logits.begin(),
                                    std::max_element(streamed_logits.begin(), streamed_logits.end()));
                                predicted_logit = streamed_logits[predicted_token];
                                logits_hash = hash_f32(streamed_logits.data(), streamed_logits.size());
                                output_projection_pass = std::isfinite(predicted_logit);
                                std::printf("OUTPUT_PROJECTION_PASS=%s VOCAB_LOGITS_COMPUTED=YES VOCAB_SIZE=%zu "
                                    "PREDICTED_TOKEN_ID=%u PREDICTED_TOKEN_LOGIT=%.9g LOGITS_HASH=%016llx "
                                    "REFERENCE_COMPARISON=NOT_RUN\n",
                                    output_projection_pass ? "YES" : "NO", streamed_logits.size(),
                                    predicted_token, predicted_logit, (unsigned long long) logits_hash);
                            } else {
                                std::printf("OUTPUT_PROJECTION_PASS=NO VOCAB_LOGITS_COMPUTED=NO ERROR=%s\n", out_error.c_str());
                            }
                        }
                    }
                    const float * logits = head_rc == 0 ? llama_get_logits_ith(head_ctx, 0) : nullptr;
                    const size_t n_vocab = head_model->vocab.n_tokens();
                    if (logits && n_vocab > 0 && std::getenv("CORTEX_FLASHNEXT_FINAL_HEAD_NORM_ONLY") == nullptr) {
                        predicted_token = (uint32_t) std::distance(logits,
                            std::max_element(logits, logits + n_vocab));
                        predicted_logit = logits[predicted_token];
                        logits_hash = hash_f32(logits, n_vocab);
                        output_projection_pass = std::isfinite(predicted_logit);
                        std::printf("FINAL_NORM_PASS=YES OUTPUT_PROJECTION_PASS=%s VOCAB_SIZE=%zu "
                            "PREDICTED_TOKEN_ID=%u PREDICTED_TOKEN_LOGIT=%.9g "
                            "FINAL_NORM_HASH=%016llx LOGITS_HASH=%016llx REFERENCE_COMPARISON=NOT_RUN\n",
                            output_projection_pass ? "YES" : "NO", n_vocab, predicted_token,
                            predicted_logit, (unsigned long long) final_activation_hash,
                            (unsigned long long) logits_hash);
                    } else if (std::getenv("CORTEX_FLASHNEXT_FINAL_HEAD_NORM_ONLY") == nullptr) {
                        std::printf("FINAL_NORM_PASS=NOT_RUN OUTPUT_PROJECTION_PASS=NO VOCAB_SIZE=%zu "
                            "REFERENCE_COMPARISON=NOT_RUN\n", head_model->vocab.n_tokens());
                    }
                }
                llama_batch_free(head_batch);
            }
            if (head_ctx) llama_free(head_ctx);
            if (head_model) llama_model_free(head_model);
            std::printf("FULL_FORWARD_PASS=%s\n", output_projection_pass ? "YES" : "NO");
        } else {
            std::printf("FULL_CHAIN_0_TO_47_PASS=NO\n");
        }
        llama_backend_free();
        return chain_ok ? 0 : 23;
    }
    std::array<float, 10240> stock_output{};
    if (!flashnext_stock_layer0_reference(tx, stock_output)) return 11;
    size_t first_diff = 10240;
    float max_abs = 0.0f, max_rel = 0.0f;
    for (size_t i = 0; i < stock_output.size(); ++i) {
        const float d = std::fabs(stock_output[i] - tx.layer_output[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / std::max(std::fabs(stock_output[i]), 1e-30f));
        if (first_diff == 10240 && d != 0.0f) first_diff = i;
    }
    std::printf("STOCK_LAYER0_BOUNDARY shape=[2560,4,1] stock_hash=%016llx cortex_hash=%016llx max_abs=%.9g max_rel=%.9g first_diff=%s\n",
        (unsigned long long) hash_f32(stock_output.data(), stock_output.size()),
        (unsigned long long) hash_f32(tx.layer_output.data(), tx.layer_output.size()),
        max_abs, max_rel, first_diff == 10240 ? "NONE" : std::to_string(first_diff).c_str());
    if (max_abs == 0.0f) std::printf("FLASHNEXT COMPLETE LAYER 0: PASS\n");
    // The state bytes are captured from llama's persistent sequence state,
    // after graph synchronization and before any temporary transaction data
    // is released.  This is the contract consumed by the continuation probe.
    std::printf("STATE_POSITION0 bytes=%zu hash=%016llx\n", state_after_position0.size(),
        (unsigned long long) (state_after_position0.empty() ? 0 : hash_bytes(state_after_position0.data(), state_after_position0.size())));
    export_blob("state", 0, state_after_position0.data(), state_after_position0.size(), "opaque", "u8");
    export_state_components(ctx, 0);
    llama_batch batch1 = llama_batch_init(1, 0, 1);
    export_state_components(ctx, 10); // position1_pre: state consumed by the next decode
    batch1.token[0] = 248046; batch1.pos[0] = 1; batch1.n_seq_id[0] = 1;
    batch1.seq_id[0][0] = 0; batch1.logits[0] = 1; batch1.n_tokens = 1;
    const int decode1 = llama_decode(ctx, batch1);
    llama_synchronize(ctx);
    const std::vector<uint8_t> state_after_position1 = capture_seq_state(ctx);
    std::printf("STATE_POSITION1 decode=%d bytes=%zu hash=%016llx\n", decode1, state_after_position1.size(),
        (unsigned long long) (state_after_position1.empty() ? 0 : hash_bytes(state_after_position1.data(), state_after_position1.size())));
    export_blob("state", 1, state_after_position1.data(), state_after_position1.size(), "opaque", "u8");
    export_state_components(ctx, 1);
    const float * artifact1 = llama_get_logits_ith(ctx, 0);
    if (!artifact1) return 12;
    flashnext_layer_transaction tx1; tx1.layer=0; tx1.position=1;
    std::memcpy(tx1.pre_moe_hidden.data(), artifact1, 2560*sizeof(float));
    std::memcpy(tx1.residual_hc.data(), artifact1+2560, 10240*sizeof(float));
    std::memcpy(tx1.inject.data(), artifact1+12800, 4*sizeof(float));
    const float * route1=artifact1+12804; std::array<int,512> order1{};
    for(int i=0;i<512;++i) order1[i]=i;
    std::partial_sort(order1.begin(),order1.begin()+10,order1.end(),[route1](int a,int b){return route1[a]>route1[b];});
    double z1=0; double ex1[10]; const float mx1=route1[order1[0]];
    for(int i=0;i<10;++i){ex1[i]=std::exp(double(route1[order1[i]])-double(mx1));z1+=ex1[i];}
    for(int i=0;i<10;++i){tx1.selected_expert_ids[i]=order1[i];tx1.routing_weights[i]=ex1[i]/z1;}
    if(!flashnext_execute_selected_moe(index,argv[2],tx1)) return 13;
    std::array<int, 4> shared1_fds{{fd, fd, fd, fd}};
    flashnext_shared_expert_slices shared1{shared1_fds,sg.offset,su.offset,sd.offset,sc.offset};
    if(!flashnext_execute_shared_expert(shared1,tx1)) return 14;
    for(size_t i=0;i<tx1.complete_moe.size();++i) tx1.complete_moe[i]=tx1.selected_moe[i]+tx1.shared_moe[i];
    if(!flashnext_execute_segment_b(tx1)) return 15;
    std::printf("POSITION1 ids="); for(int i=0;i<10;++i)std::printf("%s%d",i?",":"",tx1.selected_expert_ids[i]);
    std::printf(" output_hash=%016llx selected_bytes=%llu unselected=%llu\n",
        (unsigned long long)hash_f32(tx1.layer_output.data(),tx1.layer_output.size()),
        (unsigned long long)tx1.io.selected_expert_bytes_read,(unsigned long long)tx1.io.unselected_expert_reads);
    export_blob("output", 1, tx1.layer_output.data(), tx1.layer_output.size()*sizeof(float), "2560,4,1", "f32");
    export_blob("ids", 1, tx1.selected_expert_ids.data(), tx1.selected_expert_ids.size()*sizeof(int32_t), "10", "i32");
    llama_batch_free(batch1);
    std::printf("CORTEX FLASHNEXT OWNED TRANSACTION: PASS\n");
    std::printf("pre_moe=%016llx selected=%016llx shared=%016llx complete=%016llx segment_b=%016llx\n",
        (unsigned long long) hash_f32(tx.pre_moe_hidden.data(), tx.pre_moe_hidden.size()),
        (unsigned long long) hash_f32(tx.selected_moe.data(), tx.selected_moe.size()),
        (unsigned long long) hash_f32(tx.shared_moe.data(), tx.shared_moe.size()),
        (unsigned long long) hash_f32(tx.complete_moe.data(), tx.complete_moe.size()),
        (unsigned long long) hash_f32(tx.layer_output.data(), tx.layer_output.size()));
    std::printf("ids="); for (int i=0;i<10;++i) std::printf("%s%d",i?",":"",tx.selected_expert_ids[i]);
    std::printf("\nselected_bytes=%llu unselected=%llu window=%llu pread_calls=%llu output_shape=[2560,4,1]\n",
        (unsigned long long)tx.io.selected_expert_bytes_read,(unsigned long long)tx.io.unselected_expert_reads,
        (unsigned long long)tx.io.tensor_window_bytes,(unsigned long long)tx.io.pread_calls);
    ::close(fd); llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
}
