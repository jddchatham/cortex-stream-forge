#include "llama.h"
#include "llama-ext.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/flashnext-layer-transaction.h"
#include "../src/flashnext-selected-moe.h"
#include "../src/flashnext-shared-expert.h"
#include "../src/flashnext-segment-b.h"
#include "../src/flashnext-tensor-index.h"
#include "../src/ggml-quants.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static uint64_t hash_f32(const float * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { uint32_t u; std::memcpy(&u, p+i, 4); h ^= u; h *= 1099511628211ull; }
    return h;
}

static bool finite_f32(const float * p, size_t n) {
    return std::all_of(p, p + n, [](float value) { return std::isfinite(value); });
}

static void log_f32_summary(const char * name, const float * p, size_t n) {
    size_t nan = 0, inf = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        nan += std::isnan(p[i]); inf += std::isinf(p[i]);
        if (std::isfinite(p[i])) max_abs = std::max(max_abs, std::abs((double) p[i]));
    }
    std::fprintf(stderr, "QIES066_VALUE name=%s n=%zu nan=%zu inf=%zu max_abs=%.9g\n", name, n, nan, inf, max_abs);
}

static bool write_logits_file(const std::string & dir, size_t index, const std::vector<float> & logits) {
    if (dir.empty()) return true;
    const std::string path = dir + "/logits-select-" + std::to_string(index) + ".bin";
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) { std::perror(("open " + path).c_str()); return false; }
    const uint8_t * data = reinterpret_cast<const uint8_t *>(logits.data());
    size_t bytes = logits.size() * sizeof(float), done = 0;
    while (done < bytes) {
        const ssize_t n = ::write(fd, data + done, bytes - done);
        if (n <= 0) { ::close(fd); return false; }
        done += (size_t) n;
    }
    return ::close(fd) == 0;
}

static bool write_capture_blob(const std::string & root, int position, const char * name,
        const void * data, size_t bytes) {
    if (root.empty() || data == nullptr || bytes == 0) return false;
    const std::string path = root + "/p" + std::to_string(position) + "/" + name + ".bin";
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    size_t done = 0;
    while (done < bytes) {
        const ssize_t n = ::write(fd, (const uint8_t *) data + done, bytes - done);
        if (n <= 0) { ::close(fd); return false; }
        done += (size_t) n;
    }
    return ::close(fd) == 0;
}

static bool flashnext_stream_output_q6k(const flashnext_tensor_descriptor & d,
        const std::vector<float> & x, std::vector<float> & logits) {
    if (d.type != GGML_TYPE_Q6_K || d.shape0 != 2560 || d.shape1 != 248320 || x.size() != 2560) return false;
    const int fd = ::open(d.path.c_str(), O_RDONLY);
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

struct layer_transaction_result {
    int decode_status = -1;
    bool graph_a_ok = false;
    bool sparse_moe_ok = false;
    bool graph_b_ok = false;
    bool state_committed = false;
    bool rollback_restored = false;
    const char * failure_stage = nullptr;
    std::array<float, 2560 * 4> output{};
    uint64_t output_hash = 0;
    uint64_t sparse_output_hash = 0;
    std::array<int32_t, 10> selected_ids{};
    uint64_t unselected_expert_reads = 0;
    std::vector<uint8_t> state_before;
};

static std::vector<uint8_t> capture_seq_state(llama_context * ctx) {
    const size_t n = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> state(n);
    if (n != 0 && llama_state_seq_get_data(ctx, state.data(), n, 0) != n) state.clear();
    return state;
}

// Per-layer state is disk-backed so the 48 full hybrid-state blobs do not
// become 5.7 GiB of permanently resident anonymous memory. A new directory is
// required for each run, and O_EXCL prevents accidental artifact reuse.
// Final HC mixer mirror of build_hc_mix(res_hc, hc_head_*, nullptr, nullptr, -1)
// (unfused path, il=-1): grouped RMSNorm over the 4 streams, hc_head_down,
// silu(1/hc scale), hc_head_up, sigmoid gate, per-stream product, mean over
// streams.  All ggml ops, bit-exact against the stock graph.
static bool flashnext_final_head_mix(const flashnext_tensor_descriptor & dn,
        const flashnext_tensor_descriptor & dd, const flashnext_tensor_descriptor & du,
        const float * hc_in /* 10240 */, std::vector<float> & mixed /* 2560 out */) {
    if (dn.type != GGML_TYPE_F32 || dn.shape0 != 10240 ||
        dd.shape0 != 10240 || du.shape0 != 320 || du.shape1 != 10240) return false;
    const size_t dd_bytes = (size_t) ggml_row_size((ggml_type) dd.type, dd.shape0) * dd.shape1;
    const size_t du_bytes = (size_t) ggml_row_size((ggml_type) du.type, du.shape0) * du.shape1;
    std::vector<uint8_t> dn_b(dn.shape0 * sizeof(float)), dd_b(dd_bytes), du_b(du_bytes);
    auto slurp = [](const flashnext_tensor_descriptor & d, uint8_t * p, size_t n) {
        const int fd = ::open(d.path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        size_t done = 0;
        while (done < n) {
            const ssize_t r = ::pread(fd, p + done, n - done, (off_t) (d.offset + done));
            if (r <= 0) { ::close(fd); return false; }
            done += (size_t) r;
        }
        ::close(fd);
        return true;
    };
    if (!slurp(dn, dn_b.data(), dn_b.size()) || !slurp(dd, dd_b.data(), dd_b.size()) ||
        !slurp(du, du_b.data(), du_b.size())) return false;

    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) return false;
    ggml_init_params ip { 64 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    if (!c) { ggml_backend_free(be); return false; }

    constexpr int64_t hc = 4;
    ggml_tensor * x    = ggml_new_tensor_3d(c, GGML_TYPE_F32, 2560, hc, 1);
    ggml_tensor * wn   = ggml_new_tensor(c, GGML_TYPE_F32, 2, (int64_t[]) {2560, hc});
    ggml_tensor * wd   = ggml_new_tensor_2d(c, (ggml_type) dd.type, dd.shape0, dd.shape1);
    ggml_tensor * wu   = ggml_new_tensor_2d(c, (ggml_type) du.type, du.shape0, du.shape1);
    ggml_tensor * rms  = ggml_rms_norm(c, x, 1.0e-6f);
    ggml_tensor * xn   = ggml_reshape_2d(c, ggml_mul(c, rms, wn), 10240, 1);
    ggml_tensor * lo   = ggml_silu(c, ggml_scale(c, ggml_mul_mat(c, wd, xn), 1.0f / (float) hc));
    ggml_tensor * gate = ggml_mul_mat(c, wu, lo); // [10240, 1]
    ggml_tensor * gated = ggml_reshape_3d(c, ggml_mul(c, xn, ggml_sigmoid(c, gate)), 2560, hc, 1);
    ggml_tensor * m    = ggml_view_2d(c, gated, 2560, 1, ggml_row_size(GGML_TYPE_F32, 2560) * hc, 0);
    m = ggml_cont(c, m);
    for (int64_t s = 1; s < hc; ++s) {
        ggml_tensor * v = ggml_view_2d(c, gated, 2560, 1,
                ggml_row_size(GGML_TYPE_F32, 2560) * hc, ggml_row_size(GGML_TYPE_F32, 2560) * s);
        m = ggml_add(c, m, v);
    }
    m = ggml_scale(c, m, 1.0f / (float) hc);

    ggml_cgraph * graph = ggml_new_graph(c);
    ggml_build_forward_expand(graph, m);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!buf) { ggml_free(c); ggml_backend_free(be); return false; }
    wn->data = dn_b.data();
    wd->data = dd_b.data();
    wu->data = du_b.data();
    ggml_backend_tensor_set(x, hc_in, 0, 10240 * sizeof(float));
    const bool ok = ggml_backend_graph_compute(be, graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        mixed.resize(2560);
        ggml_backend_tensor_get(m, mixed.data(), 0, 2560 * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    ggml_backend_free(be);
    return ok;
}

// Tiled ggml output projection: out[r] = dot(output.weight row r, x), streamed
// with pread tiles.  Per-row vec_dot matches the stock in-graph mul_mat.
static bool flashnext_stream_output_matvec(const flashnext_tensor_descriptor & d,
        const float * x /* 2560 */, std::vector<float> & out) {
    if (d.shape0 != 2560 || d.shape1 <= 0) return false;
    const size_t row_bytes = ggml_row_size((ggml_type) d.type, 2560);
    const size_t arena_bytes = 32 * 1024 * 1024;
    const int64_t rows_per_tile = (int64_t) (arena_bytes / row_bytes);
    if (rows_per_tile < 1) return false;
    const int fd = ::open(d.path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    void * arena = nullptr;
    if (posix_memalign(&arena, 64, arena_bytes) != 0) { ::close(fd); return false; }
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) { std::free(arena); ::close(fd); return false; }
    out.assign((size_t) d.shape1, 0.0f);
    bool ok = true;
    for (int64_t row = 0; row < d.shape1 && ok; row += rows_per_tile) {
        const int64_t nr = std::min<int64_t>(rows_per_tile, d.shape1 - row);
        const size_t tile = (size_t) nr * row_bytes;
        size_t done = 0;
        while (done < tile) {
            const ssize_t r = ::pread(fd, (uint8_t *) arena + done, tile - done, (off_t) (d.offset + (uint64_t) row * row_bytes + done));
            if (r <= 0) { ok = false; break; }
            done += (size_t) r;
        }
        if (!ok) break;
        ggml_init_params ip { 32 * 1024 * 1024, nullptr, true };
        ggml_context * c = ggml_init(ip);
        if (!c) { ok = false; break; }
        ggml_tensor * w = ggml_new_tensor_2d(c, (ggml_type) d.type, 2560, nr);
        ggml_tensor * xi = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2560, 1);
        ggml_tensor * y  = ggml_mul_mat(c, w, xi);
        ggml_cgraph * g = ggml_new_graph(c);
        ggml_build_forward_expand(g, y);
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(c, be);
        if (!b) { ok = false; ggml_free(c); break; }
        w->data = arena;
        ggml_backend_tensor_set(xi, x, 0, 2560 * sizeof(float));
        ok = ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS;
        if (ok) ggml_backend_tensor_get(y, out.data() + row, 0, (size_t) nr * sizeof(float));
        ggml_backend_buffer_free(b);
        ggml_free(c);
    }
    std::free(arena);
    ggml_backend_free(be);
    ::close(fd);
    return ok;
}

struct flashnext_state_store {
    struct slot { int fd = -1; uint8_t * map = nullptr; size_t size = 0; };
    std::string dir;
    std::vector<slot> slots;

    explicit flashnext_state_store(size_t n, std::string path) : dir(std::move(path)), slots(n) {
        if (::mkdir(dir.c_str(), 0700) != 0) {
            throw std::runtime_error("state directory must be new and empty: " + dir);
        }
        for (size_t i = 0; i < n; ++i) {
            const std::string path_i = dir + "/state-layer-" + std::to_string(i) + ".bin";
            slots[i].fd = ::open(path_i.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (slots[i].fd < 0) throw std::runtime_error("state store open failed: " + path_i);
        }
    }
    ~flashnext_state_store() {
        for (auto & s : slots) {
            if (s.map) ::munmap(s.map, s.size);
            if (s.fd >= 0) ::close(s.fd);
        }
    }
    uint8_t * prepare(size_t i, size_t size) {
        if (i >= slots.size() || slots[i].fd < 0) return nullptr;
        auto & s = slots[i];
        if (s.size < size) {
            if (s.map) { ::munmap(s.map, s.size); s.map = nullptr; }
            if (::ftruncate(s.fd, (off_t) size) != 0) return nullptr;
            s.map = (uint8_t *) ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, s.fd, 0);
            if (s.map == MAP_FAILED) s.map = nullptr;
            s.size = s.map ? size : 0;
        }
        return s.map;
    }
    bool store(size_t i, const std::vector<uint8_t> & bytes) {
        uint8_t * dst = prepare(i, bytes.size());
        if (!dst || bytes.empty()) return false;
        std::memcpy(dst, bytes.data(), bytes.size());
        return true;
    }
    const uint8_t * data(size_t i) const { return i < slots.size() ? slots[i].map : nullptr; }
    size_t size(size_t i) const { return i < slots.size() ? slots[i].size : 0; }
    uint64_t total_bytes() const {
        uint64_t total = 0;
        for (const auto & s : slots) total += s.size;
        return total;
    }
};

static bool restore_layer_state(llama_context * ctx, const flashnext_state_store & store,
        size_t layer, int position) {
    const uint8_t * data = store.data(layer);
    const size_t size = store.size(layer);
    if (position == 0 && data == nullptr) return true;
    if (data == nullptr || size == 0) {
        std::fprintf(stderr, "missing hybrid state for layer %zu at position %d\n", layer, position);
        return false;
    }
    const size_t restored = llama_state_seq_set_data(ctx, data, size, 0);
    if (restored != size) {
        std::fprintf(stderr, "hybrid-state restore failed for layer %zu at position %d: %zu/%zu bytes\n",
                layer, position, restored, size);
        return false;
    }
    return true;
}

static bool save_layer_state(llama_context * ctx, flashnext_state_store & store, size_t layer) {
    const std::vector<uint8_t> state = capture_seq_state(ctx);
    if (state.empty() || !store.store(layer, state)) {
        std::fprintf(stderr, "hybrid-state save failed for layer %zu\n", layer);
        return false;
    }
    return true;
}

static layer_transaction_result run_layer_decode(
        llama_context * ctx, int position, llama_token token_id, const std::vector<float> & owned_hidden) {
    layer_transaction_result result;
    llama_batch batch = llama_batch_init(1, (int) owned_hidden.size(), 1);
    if (!batch.embd) { result.failure_stage = "batch_init"; return result; }
    // Carry the token ID alongside the externally supplied activation. The
    // graph still consumes embd; Qwen PLE uses token metadata to form its hash.
    batch.token = (llama_token *) std::malloc(sizeof(llama_token));
    if (!batch.token) { llama_batch_free(batch); result.failure_stage = "batch_token_alloc"; return result; }
    batch.token[0] = token_id;
    std::memcpy(batch.embd, owned_hidden.data(), owned_hidden.size() * sizeof(float));
    batch.pos[0] = position;
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

struct flashnext_transaction_context {
    llama_context * ctx = nullptr;
    flashnext_tensor_index * tensor_index = nullptr;
    flashnext_layer_descriptor layer_desc;
    std::vector<uint8_t> staged_state;
    flashnext_layer_transaction values;
    std::array<float, 2560 * 4> output{};
    std::array<int32_t, 10> selected_ids{};
    std::array<float, 2560> moe_input{};
    std::array<float, 512> router_logits{};
    std::array<float, 2560 * 4> residual_hc{};
    std::array<float, 4> injection{};
    bool state_committed = false;
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

    if (!out.router || !out.gate_experts || !out.up_experts || !out.down_experts) {
        error = "layer is missing sparse-MoE tensors"; return false;
    }
    if (!out.shared_gate || !out.shared_up || !out.shared_down) {
        error = "layer is missing shared-expert tensors"; return false;
    }
    out.supported = true;
    return true;
}

static bool capture_transaction_router(
        flashnext_transaction_context & txc, llama_context * ctx, const flashnext_layer_descriptor & layer) {
    const std::string suffix = "-" + std::to_string(layer.index);
    const size_t moe_bytes = ctx->debug_copy_flashnext_tensor(("diag_moe_input" + suffix).c_str(),
        txc.moe_input.data(), sizeof(txc.moe_input));
    const size_t router_bytes = ctx->debug_copy_flashnext_tensor(("diag_router_logits" + suffix).c_str(),
        txc.router_logits.data(), sizeof(txc.router_logits));
    std::string residual_name = "diag_tx_residual" + suffix;
    size_t residual_probe = ctx->debug_copy_flashnext_tensor(residual_name.c_str(), nullptr, 0);
    if (residual_probe != sizeof(txc.residual_hc)) residual_name = "diag_hc_residual" + suffix;
    const size_t residual_bytes = ctx->debug_copy_flashnext_tensor(residual_name.c_str(),
        txc.residual_hc.data(), sizeof(txc.residual_hc));
    std::string inject_name = "diag_tx_inject" + suffix;
    size_t inject_probe = ctx->debug_copy_flashnext_tensor(inject_name.c_str(), nullptr, 0);
    if (inject_probe != sizeof(txc.injection)) inject_name = "diag_hc_inject" + suffix;
    const size_t inject_bytes = ctx->debug_copy_flashnext_tensor(inject_name.c_str(),
        txc.injection.data(), sizeof(txc.injection));
    if (moe_bytes != sizeof(txc.moe_input) || router_bytes != sizeof(txc.router_logits) ||
        residual_bytes != sizeof(txc.residual_hc) || inject_bytes != sizeof(txc.injection)) return false;
    for (float x : txc.moe_input) if (!std::isfinite(x)) return false;
    for (float x : txc.router_logits) if (!std::isfinite(x)) return false;
    for (float x : txc.residual_hc) if (!std::isfinite(x)) return false;
    for (float x : txc.injection) if (!std::isfinite(x)) return false;
    return true;
}

static bool execute_transaction_sparse_moe(
        flashnext_transaction_context & txc, const std::string & index_path, std::string & error) {
    if (!txc.tensor_index) { error = "missing tensor index"; return false; }
    std::array<int, 512> order{};
    for (int i = 0; i < 512; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + 10,
        order.end(), [&](int a, int b) { return txc.router_logits[a] > txc.router_logits[b]; });
    double z = 0.0, ex[10];
    const float mx = txc.router_logits[order[0]];
    for (size_t i = 0; i < 10; ++i) {
        ex[i] = std::exp((double) txc.router_logits[order[i]] - mx); z += ex[i];
    }
    txc.values.layer = (int) txc.layer_desc.index;
    for (size_t i = 0; i < 10; ++i) {
        txc.values.selected_expert_ids[i] = order[i];
        txc.values.routing_weights[i] = ex[i] / z;
        txc.selected_ids[i] = order[i];
    }
    std::memcpy(txc.values.pre_moe_hidden.data(), txc.moe_input.data(), sizeof(txc.moe_input));
    std::memcpy(txc.values.router_logits.data(), txc.router_logits.data(), sizeof(txc.router_logits));
    std::memcpy(txc.values.residual_hc.data(), txc.residual_hc.data(), sizeof(txc.residual_hc));
    std::memcpy(txc.values.inject.data(), txc.injection.data(), sizeof(txc.injection));
    if (!txc.tensor_index->load(index_path, &error)) return false;
    if (!flashnext_execute_selected_moe(*txc.tensor_index, index_path, txc.values)) return false;
    if (txc.values.layer == 1) {
        if (const char * root = std::getenv("QIES067_CAPTURE_DIR")) {
            write_capture_blob(root, txc.values.position, "qies069_l1_router_logits", txc.router_logits.data(), sizeof(txc.router_logits));
            write_capture_blob(root, txc.values.position, "qies069_l1_selected_ids", txc.values.selected_expert_ids.data(), sizeof(txc.values.selected_expert_ids));
            write_capture_blob(root, txc.values.position, "qies069_l1_selected_moe", txc.values.selected_moe.data(), sizeof(txc.values.selected_moe));
        }
    }
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
    const std::array<std::string, 4> shared_paths{{sg.path, su.path, sd.path, sc.path}};
    std::array<int, 4> shared_fds{{-1, -1, -1, -1}};
    for (size_t i = 0; i < shared_fds.size(); ++i) shared_fds[i] = ::open(shared_paths[i].c_str(), O_RDONLY | O_CLOEXEC);
    if (std::any_of(shared_fds.begin(), shared_fds.end(), [](int fd) { return fd < 0; })) {
        for (int fd : shared_fds) if (fd >= 0) ::close(fd);
        error = "failed to open shared expert tensor shard"; return false;
    }
    flashnext_shared_expert_slices shared{shared_fds, sg.offset, su.offset, sd.offset, sc.offset};
    const bool shared_ok = flashnext_execute_shared_expert(shared, txc.values);
    for (int fd : shared_fds) ::close(fd);
    if (!shared_ok) { error = "shared expert execution failed"; return false; }
    if (layer == 1) {
        if (const char * root = std::getenv("QIES067_CAPTURE_DIR")) {
            write_capture_blob(root, txc.values.position, "qies069_l1_shared_moe", txc.values.shared_moe.data(), sizeof(txc.values.shared_moe));
        }
    }
    log_f32_summary("selected_moe", txc.values.selected_moe.data(), txc.values.selected_moe.size());
    log_f32_summary("shared_moe", txc.values.shared_moe.data(), txc.values.shared_moe.size());
    log_f32_summary("residual", txc.values.residual_hc.data(), txc.values.residual_hc.size());
    log_f32_summary("inject", txc.values.inject.data(), txc.values.inject.size());
    for (size_t i = 0; i < txc.values.complete_moe.size(); ++i) {
        txc.values.complete_moe[i] = txc.values.selected_moe[i] + txc.values.shared_moe[i];
    }
    if (layer == 1) {
        if (const char * root = std::getenv("QIES067_CAPTURE_DIR")) {
            write_capture_blob(root, txc.values.position, "qies069_l1_complete_moe", txc.values.complete_moe.data(), sizeof(txc.values.complete_moe));
        }
    }
    log_f32_summary("complete_moe", txc.values.complete_moe.data(), txc.values.complete_moe.size());
    if (!flashnext_execute_segment_b(txc.values)) { error = "segment B execution failed"; return false; }
    if (const char * root = std::getenv("QIES067_CAPTURE_DIR")) {
        const std::string output_name = "qies069_layer" + std::to_string(layer) + "_output";
        write_capture_blob(root, txc.values.position, output_name.c_str(), txc.values.layer_output.data(), sizeof(txc.values.layer_output));
    }
    if (!finite_f32(txc.values.layer_output.data(), txc.values.layer_output.size())) {
        log_f32_summary("segment_b_output", txc.values.layer_output.data(), txc.values.layer_output.size());
        error = "segment B produced non-finite layer output"; return false;
    }
    txc.output = txc.values.layer_output;
    txc.state_committed = true;
    return true;
}

static layer_transaction_result run_layer_transaction(
        llama_context * ctx, uint32_t layer, int position, const std::vector<float> & owned_hidden,
        const std::string & index_path, llama_token token_id) {
    layer_transaction_result result = run_layer_decode(ctx, position, token_id, owned_hidden);
    if (result.decode_status != 0) { result.failure_stage = "graph_a_decode"; return result; }
    result.graph_a_ok = true;

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

    const bool router_capture_ok = capture_transaction_router(txc, ctx, txc.layer_desc);
    const bool sparse_moe_ok = router_capture_ok && execute_transaction_sparse_moe(txc, index_path, error);
    if (!sparse_moe_ok) {
        if (!result.state_before.empty()) llama_state_seq_set_data(ctx, result.state_before.data(), result.state_before.size(), 0);
        result.failure_stage = "capture_or_sparse_moe";
        result.rollback_restored = true;
        return result;
    }
    result.sparse_moe_ok = true;
    result.sparse_output_hash = hash_f32(txc.values.selected_moe.data(), txc.values.selected_moe.size());
    if (!execute_transaction_graph_b(txc, index_path, error)) {
        if (!result.state_before.empty()) llama_state_seq_set_data(ctx, result.state_before.data(), result.state_before.size(), 0);
        result.failure_stage = "graph_b";
        std::fprintf(stderr, "layer %u graph B failed: %s\n", layer, error.c_str());
        result.rollback_restored = true;
        return result;
    }
    result.graph_b_ok = true;
    result.output = txc.output;
    result.output_hash = hash_f32(result.output.data(), result.output.size());
    result.selected_ids = txc.selected_ids;
    result.unselected_expert_reads = txc.values.io.unselected_expert_reads;
    result.state_committed = true;
    return result;
}

// Full 48-layer forward step for a given position
// If is_token_input == true, layer 0 processes token_id
// If is_token_input == false, layer 0 processes activation
static bool forward_step(
        const std::string & model_path,
        const std::string & index_path,
        int position,
        llama_token token_id,
        const std::vector<llama_token> & token_history,
        flashnext_state_store & layer_states,
        std::vector<float> & out_logits,
        bool compute_logits) {

    if (const char * root = std::getenv("QIES067_CAPTURE_DIR")) {
        const std::string dir = std::string(root) + "/p" + std::to_string(position);
        if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) return false;
        setenv("CORTEX_FLASHNEXT_DIAG_DUMP", dir.c_str(), 1);
    }

    if (position != (int) token_history.size()) {
        std::fprintf(stderr, "token history length %zu does not match position %d\n", token_history.size(), position);
        return false;
    }
    const llama_token prev2 = position >= 2 ? token_history[position - 2] : (llama_token) -1;
    const llama_token prev1 = position >= 1 ? token_history[position - 1] : (llama_token) -1;
    const std::string ple_prev = std::to_string(prev2) + "," + std::to_string(prev1);
    setenv("CORTEX_FLASHNEXT_PLE_PREV", ple_prev.c_str(), 1);

    // 1. Layer 0 execution
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1);
    setenv("CORTEX_FLASHNEXT_SEGMENT_A", "1", 1);
    setenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER", "1", 1);
    setenv("CORTEX_EXTERNAL_GLOBAL_TENSORS", "1", 1);
    // The initial layer-0 router-only graph still includes PLE input wiring.
    // Keep layer-range mode enabled from graph construction so the streamed
    // PLE embedding rows are materialized by the layer-range input binder.
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    if (std::getenv("QIES068_MATCH_CHAIN_LAYER0_GRAPH") != nullptr) {
        unsetenv("CORTEX_FLASHNEXT_LAYER_START");
        unsetenv("CORTEX_FLASHNEXT_LAYER_END");
        unsetenv("CORTEX_QWEN35_SEGMENT_END");
    } else {
        setenv("CORTEX_FLASHNEXT_LAYER_START", "0", 1);
        setenv("CORTEX_FLASHNEXT_LAYER_END", "1", 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);
    }
    unsetenv("CORTEX_FLASHNEXT_PRE_MOE");

    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model0 = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model0) return false;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 1;
    llama_context * ctx0 = llama_init_from_model(model0, cp);
    if (!ctx0) { llama_model_free(model0); return false; }

    if (!restore_layer_state(ctx0, layer_states, 0, position)) {
        llama_free(ctx0); llama_model_free(model0); return false;
    }

    const int n_embd_inp = llama_model_n_embd_inp(model0);
    std::vector<float> token_embedding((size_t) n_embd_inp);
    if (n_embd_inp <= 0 || !llama_cortex_token_embd_rows(model0, &token_id, 1, token_embedding.data())) {
        llama_free(ctx0); llama_model_free(model0); return false;
    }
    const char * capture_root_env = std::getenv("QIES067_CAPTURE_DIR");
    const std::string capture_root = capture_root_env ? capture_root_env : "";
    if (!capture_root.empty()) {
        write_capture_blob(capture_root, position, "qies068_raw_token_embedding", token_embedding.data(), token_embedding.size() * sizeof(float));
        const uint32_t token_meta[3] = {(uint32_t) token_id, (uint32_t) position, 0};
        write_capture_blob(capture_root, position, "qies068_token_pos_seq", token_meta, sizeof(token_meta));
    }
    llama_batch batch0 = llama_batch_init(1, n_embd_inp, 1);
    if (!batch0.embd) {
        llama_batch_free(batch0); llama_free(ctx0); llama_model_free(model0); return false;
    }
    batch0.token = (llama_token *) std::malloc(sizeof(llama_token));
    if (!batch0.token) { llama_batch_free(batch0); llama_free(ctx0); llama_model_free(model0); return false; }
    batch0.token[0] = token_id;
    std::memcpy(batch0.embd, token_embedding.data(), token_embedding.size() * sizeof(float));
    batch0.pos[0] = position; batch0.n_seq_id[0] = 1;
    batch0.seq_id[0][0] = 0; batch0.logits[0] = 1; batch0.n_tokens = 1;
    if (llama_decode(ctx0, batch0) != 0) {
        llama_batch_free(batch0); llama_free(ctx0); llama_model_free(model0); return false;
    }
    llama_synchronize(ctx0);
    if (!save_layer_state(ctx0, layer_states, 0)) {
        llama_batch_free(batch0); llama_free(ctx0); llama_model_free(model0); return false;
    }
    llama_batch_free(batch0);

    const float * artifact = llama_get_logits_ith(ctx0, 0);
    if (!artifact) { llama_free(ctx0); llama_model_free(model0); return false; }
    if (!capture_root.empty()) {
        write_capture_blob(capture_root, position, "qies068_segment_a_artifact", artifact, 12804 * sizeof(float));
        write_capture_blob(capture_root, position, "qies068_router_logits", artifact + 12804, 512 * sizeof(float));
        write_capture_blob(capture_root, position, "qies068_pre_moe_hidden", artifact, 2560 * sizeof(float));
        write_capture_blob(capture_root, position, "qies068_residual_hc", artifact + 2560, 10240 * sizeof(float));
        write_capture_blob(capture_root, position, "qies068_injection", artifact + 2560 + 10240, 4 * sizeof(float));
        std::vector<float> ple(2560);
        const size_t ple_bytes = ctx0->debug_copy_flashnext_tensor("diag_ple_embd-0", ple.data(), ple.size() * sizeof(float));
        if (ple_bytes == ple.size() * sizeof(float)) write_capture_blob(capture_root, position, "qies068_ple_embedding", ple.data(), ple_bytes);
    }

    flashnext_layer_transaction tx0;
    tx0.layer = 0;
    tx0.position = position;
    std::memcpy(tx0.pre_moe_hidden.data(), artifact, tx0.pre_moe_hidden.size()*sizeof(float));
    std::memcpy(tx0.residual_hc.data(), artifact + 2560, tx0.residual_hc.size()*sizeof(float));
    std::memcpy(tx0.inject.data(), artifact + 2560 + tx0.residual_hc.size(), tx0.inject.size()*sizeof(float));
    std::array<int, 512> order0{};
    for (int i = 0; i < 512; ++i) order0[i] = i;
    const float * route0 = artifact + 12804;
    std::memcpy(tx0.router_logits.data(), route0, sizeof(tx0.router_logits));
    std::partial_sort(order0.begin(), order0.begin()+10, order0.end(), [route0](int a, int b) { return route0[a] > route0[b]; });
    double z0 = 0.0; double ex0[10]; const float mx0 = route0[order0[0]];
    for (int i = 0; i < 10; ++i) { ex0[i] = std::exp(static_cast<double>(route0[order0[i]]) - static_cast<double>(mx0)); z0 += ex0[i]; }
    for (int i = 0; i < 10; ++i) { tx0.selected_expert_ids[i] = order0[i]; tx0.routing_weights[i] = ex0[i] / z0; }

    flashnext_tensor_index index; std::string index_error;
    if (!index.load(index_path, &index_error)) { llama_free(ctx0); llama_model_free(model0); return false; }
    if (!flashnext_execute_selected_moe(index, index_path, tx0)) { llama_free(ctx0); llama_model_free(model0); return false; }
    if (!capture_root.empty()) {
        write_capture_blob(capture_root, position, "qies068_selected_moe", tx0.selected_moe.data(), tx0.selected_moe.size() * sizeof(float));
        write_capture_blob(capture_root, position, "qies068_selected_expert_ids", tx0.selected_expert_ids.data(), tx0.selected_expert_ids.size() * sizeof(int32_t));
    }

    flashnext_tensor_descriptor sg, su, sd, sc;
    if (!index.resolve_shared(0, flashnext_projection::gate, sg, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::up, su, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::down, sd, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::shared_control, sc, &index_error)) {
        llama_free(ctx0); llama_model_free(model0); return false;
    }
    const std::array<std::string, 4> shared_paths{{sg.path, su.path, sd.path, sc.path}};
    std::array<int, 4> shared_fds{{-1, -1, -1, -1}};
    for (size_t i = 0; i < shared_fds.size(); ++i) shared_fds[i] = ::open(shared_paths[i].c_str(), O_RDONLY | O_CLOEXEC);
    if (std::any_of(shared_fds.begin(), shared_fds.end(), [](int fd) { return fd < 0; })) {
        for (int fd : shared_fds) if (fd >= 0) ::close(fd);
        llama_free(ctx0); llama_model_free(model0); return false;
    }
    flashnext_shared_expert_slices shared{shared_fds, sg.offset, su.offset, sd.offset, sc.offset};
    const bool shared_ok = flashnext_execute_shared_expert(shared, tx0);
    for (int fd : shared_fds) ::close(fd);
    if (!shared_ok) { llama_free(ctx0); llama_model_free(model0); return false; }

    if (!capture_root.empty()) write_capture_blob(capture_root, position, "qies068_shared_moe", tx0.shared_moe.data(), tx0.shared_moe.size() * sizeof(float));

    for (size_t i = 0; i < tx0.complete_moe.size(); ++i) tx0.complete_moe[i] = tx0.selected_moe[i] + tx0.shared_moe[i];
    if (!capture_root.empty()) write_capture_blob(capture_root, position, "qies068_complete_moe", tx0.complete_moe.data(), tx0.complete_moe.size() * sizeof(float));
    if (!flashnext_execute_segment_b(tx0)) { llama_free(ctx0); llama_model_free(model0); return false; }
    if (!finite_f32(tx0.layer_output.data(), tx0.layer_output.size())) {
        std::fprintf(stderr, "layer 0 segment B produced non-finite output at position %d\n", position);
        llama_free(ctx0); llama_model_free(model0); return false;
    }
    if (!capture_root.empty()) write_capture_blob(capture_root, position, "qies068_layer0_output", tx0.layer_output.data(), tx0.layer_output.size() * sizeof(float));

    std::vector<float> current_output(tx0.layer_output.begin(), tx0.layer_output.end());
    llama_free(ctx0);
    llama_model_free(model0);

    // 2. Layers 1..47
    unsetenv("CORTEX_FLASHNEXT_ROUTER_ONLY");
    unsetenv("CORTEX_FLASHNEXT_SEGMENT_A");
    unsetenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER");
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
    setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER3_TRANSACTION", "1", 1);

    for (uint32_t il = 1; il < 48; ++il) {
        if (!finite_f32(current_output.data(), current_output.size())) {
            std::fprintf(stderr, "non-finite layer input before layer %u at position %d\n", il, position);
            return false;
        }
        setenv("CORTEX_LOW_RESIDENCY_LAYER", std::to_string(il).c_str(), 1);
        setenv("CORTEX_FLASHNEXT_LAYER_START", std::to_string(il).c_str(), 1);
        setenv("CORTEX_FLASHNEXT_LAYER_END", std::to_string(il + 1).c_str(), 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", std::to_string(il + 1).c_str(), 1);

        llama_model * next_model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!next_model) return false;
        llama_context_params next_cp = cp;
        next_cp.embeddings = true;
        llama_context * next_ctx = llama_init_from_model(next_model, next_cp);
        if (!next_ctx) { llama_model_free(next_model); return false; }

        if (!restore_layer_state(next_ctx, layer_states, il, position)) {
            llama_free(next_ctx); llama_model_free(next_model); return false;
        }

        layer_transaction_result res = run_layer_transaction(next_ctx, il, position, current_output, index_path, token_id);
        if (!res.state_committed) {
            std::fprintf(stderr, "layer %u transaction failed at position %d: stage=%s decode=%d graph_a=%s sparse_moe=%s graph_b=%s\n",
                    il, position, res.failure_stage ? res.failure_stage : "unknown", res.decode_status,
                    res.graph_a_ok ? "yes" : "no", res.sparse_moe_ok ? "yes" : "no", res.graph_b_ok ? "yes" : "no");
            llama_free(next_ctx); llama_model_free(next_model); return false;
        }
        if (!save_layer_state(next_ctx, layer_states, il)) {
            llama_free(next_ctx); llama_model_free(next_model); return false;
        }
        if (!finite_f32(res.output.data(), res.output.size())) {
            std::fprintf(stderr, "layer %u produced non-finite output at position %d\n", il, position);
            llama_free(next_ctx); llama_model_free(next_model); return false;
        }
        current_output.assign(res.output.begin(), res.output.end());
        llama_free(next_ctx);
        llama_model_free(next_model);
    }

    if (!compute_logits) return true;

    // Reuse the validated host-side final HC mixer and tiled ggml projection.
    // The full-model final-head graph tries to reserve streamed layers that are
    // intentionally not resident in this executor.
    flashnext_tensor_index head_index;
    flashnext_tensor_descriptor dn, dd, du, dout;
    std::string e0, e1, e2, e3;
    std::vector<float> mixed;
    if (!head_index.load(index_path, &e0) ||
        !head_index.resolve_named("output_hc_norm.weight", dn, &e1) ||
        !head_index.resolve_named("output_hc_down.weight", dd, &e2) ||
        !head_index.resolve_named("output_hc_up.weight", du, &e3) ||
        !head_index.resolve_named("output.weight", dout, &e3) ||
        !flashnext_final_head_mix(dn, dd, du, current_output.data(), mixed) ||
        !flashnext_stream_output_matvec(dout, mixed.data(), out_logits)) return false;
    return !out_logits.empty() && finite_f32(out_logits.data(), out_logits.size());
    return true;
}

int main(int argc, char ** argv) {
    std::string prompt, model_path, index_path, state_dir, logits_dir, capture_dir;
    int max_generate = 16;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s --prompt TEXT [--max-tokens N] [--model FILE] [--tensor-index FILE] [--state-dir NEW_DIR] [--logits-dir NEW_DIR]\n", argv[0]);
            return 0;
        }
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", arg.c_str()); return 2; }
        const std::string value = argv[++i];
        if (arg == "--prompt") prompt = value;
        else if (arg == "--model") model_path = value;
        else if (arg == "--tensor-index") index_path = value;
        else if (arg == "--state-dir") state_dir = value;
        else if (arg == "--logits-dir") logits_dir = value;
        else if (arg == "--capture-dir") capture_dir = value;
        else if (arg == "--max-tokens") {
            char * end = nullptr;
            const long parsed = std::strtol(value.c_str(), &end, 10);
            if (!end || *end != '\0' || parsed < 1 || parsed > 1024) {
                std::fprintf(stderr, "--max-tokens must be an integer from 1 to 1024\n"); return 2;
            }
            max_generate = (int) parsed;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str()); return 2;
        }
    }
    if (model_path.empty()) { const char * p = std::getenv("QUORTEX_MODEL"); if (p) model_path = p; }
    if (index_path.empty()) { const char * p = std::getenv("QUORTEX_TENSOR_INDEX"); if (p) index_path = p; }
    if (state_dir.empty()) { const char * p = std::getenv("QUORTEX_STATE_DIR"); if (p) state_dir = p; }
    if (logits_dir.empty()) { const char * p = std::getenv("QUORTEX_LOGITS_DIR"); if (p) logits_dir = p; }
    if (capture_dir.empty()) { const char * p = std::getenv("QIES067_CAPTURE_DIR"); if (p) capture_dir = p; }
    if (!capture_dir.empty()) {
        if (::mkdir(capture_dir.c_str(), 0700) != 0) { std::fprintf(stderr, "capture directory must be new: %s\n", capture_dir.c_str()); return 2; }
        setenv("QIES067_CAPTURE_DIR", capture_dir.c_str(), 1);
    }
    if (index_path.empty()) index_path = "flashnext-tensor-index.json";
    if (model_path.empty() || prompt.empty()) {
        std::fprintf(stderr, "--model (or QUORTEX_MODEL) and a non-empty --prompt are required\n"); return 2;
    }
    if (state_dir.empty()) state_dir = "/tmp/quortex-qies066-" + std::to_string((long long) getpid());
    if (!logits_dir.empty() && ::mkdir(logits_dir.c_str(), 0700) != 0) {
        std::fprintf(stderr, "logits directory must be new: %s\n", logits_dir.c_str()); return 2;
    }
    if (std::getenv("CORTEX_FLASHNEXT_LOADER_CACHE") == nullptr) setenv("CORTEX_FLASHNEXT_LOADER_CACHE", "1", 1);
    if (std::getenv("CORTEX_FLASHNEXT_EXPERT_ARENA_BYTES") == nullptr) setenv("CORTEX_FLASHNEXT_EXPERT_ARENA_BYTES", "1048576", 1);
    const auto run_start = std::chrono::steady_clock::now();
    auto first_piece_time = run_start;
    bool emitted_piece = false;

    llama_backend_init();

    // Load vocabulary only for prompt tokenization and text streaming
    llama_model_params v_mp = llama_model_default_params();
    v_mp.vocab_only = true;
    llama_model * v_model = llama_model_load_from_file(model_path.c_str(), v_mp);
    if (!v_model) {
        std::fprintf(stderr, "Failed to load vocabulary\n");
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(v_model);
    const auto vocab_ready = std::chrono::steady_clock::now();

    flashnext_state_store layer_states(48, state_dir);
    int current_pos = 0;
    const auto tokenize_start = std::chrono::steady_clock::now();
    // Render this model's embedded Qwen3.8 template with enable_thinking=false.
    // We use the audited GGUF branch verbatim because the llama C helper only
    // exposes the older fixed-template API and cannot pass Jinja variables.
    const char * chat_template = llama_model_chat_template(v_model, nullptr);
    if (chat_template == nullptr || std::strstr(chat_template, "enable_thinking") == nullptr ||
            std::strstr(chat_template, "<think>\\n\\n</think>\\n\\n") == nullptr) {
        std::fprintf(stderr, "model template lacks verified non-thinking support\n");
        llama_model_free(v_model); llama_backend_free(); return 4;
    }
    const std::string turn_prompt = "<|im_start|>user\n" + prompt +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    std::fprintf(stderr, "QIES072 nonthinking_template=GGUF_native enable_thinking=false\n");
    std::fprintf(stderr, "QIES072 formatted_prompt_bytes=%zu\n", turn_prompt.size());
    std::fprintf(stderr, "QIES072 formatted_prompt=%s\n", turn_prompt.c_str());
    int n_toks = -llama_tokenize(vocab, turn_prompt.c_str(), (int32_t) turn_prompt.size(), nullptr, 0, false, true);
    if (n_toks <= 0 || n_toks + max_generate > 256) {
        std::fprintf(stderr, "prompt tokenization failed or prompt+generation exceeds n_ctx=256\n");
        llama_model_free(v_model); llama_backend_free(); return 4;
    }
    std::vector<llama_token> prompt_toks((size_t) n_toks);
    const int tokenized = llama_tokenize(vocab, turn_prompt.c_str(), (int32_t) turn_prompt.size(),
            prompt_toks.data(), n_toks, false, true);
    if (tokenized != n_toks) { std::fprintf(stderr, "prompt tokenization failed (%d/%d)\n", tokenized, n_toks); return 4; }

    const auto tokenize_done = std::chrono::steady_clock::now();
    std::vector<llama_token> token_history;
    std::vector<float> logits;
    for (size_t i = 0; i < prompt_toks.size(); ++i) {
        std::fprintf(stderr, "QIES066 prefill_position=%d token_id=%d/%zu\n", current_pos, (int) prompt_toks[i], prompt_toks.size());
        std::fflush(stderr);
        const bool capture_prefixes = !capture_dir.empty();
        if (!forward_step(model_path, index_path, current_pos, prompt_toks[i], token_history,
                    layer_states, logits, capture_prefixes || i + 1 == prompt_toks.size())) {
            std::fprintf(stderr, "prompt prefill failed at position %d\n", current_pos);
            llama_model_free(v_model); llama_backend_free(); return 5;
        }
        if (capture_prefixes) {
            const std::string dir = capture_dir + "/p" + std::to_string(current_pos);
            if (!write_logits_file(dir, 0, logits)) return 7;
            std::fprintf(stderr, "QIES067_PREFIX_COMPLETE=%d\n", current_pos); std::fflush(stderr);
        }
        token_history.push_back(prompt_toks[i]);
        current_pos++;
        if (i == 0 && std::getenv("QIES069_STOP_AFTER_FIRST_PREFIX") != nullptr) break;
    }
    const auto prefill_done = std::chrono::steady_clock::now();
    const double prefill_sec = std::chrono::duration<double>(prefill_done - tokenize_done).count();

    std::vector<llama_token> generated;
    for (int gen_step = 0; gen_step < max_generate && !logits.empty(); ++gen_step) {
        if (!write_logits_file(logits_dir, (size_t) gen_step, logits)) {
            std::fprintf(stderr, "failed to write selection logits at step %d\n", gen_step);
            llama_model_free(v_model); llama_backend_free(); return 7;
        }
        const llama_token next_token = (llama_token) std::distance(logits.begin(),
                std::max_element(logits.begin(), logits.end()));
        if (llama_vocab_is_eog(vocab, next_token)) break;
        generated.push_back(next_token);
        char piece[256];
        int piece_len = llama_token_to_piece(vocab, next_token, piece, sizeof(piece), 0, false);
        std::vector<char> larger_piece;
        if (piece_len < 0) {
            larger_piece.resize((size_t) -piece_len);
            piece_len = llama_token_to_piece(vocab, next_token, larger_piece.data(), (int32_t) larger_piece.size(), 0, false);
            if (piece_len > 0) std::cout.write(larger_piece.data(), piece_len);
        } else if (piece_len > 0) {
            std::cout.write(piece, piece_len);
        }
        std::cout.flush();
        if (!emitted_piece) { emitted_piece = true; first_piece_time = std::chrono::steady_clock::now(); }
        if (gen_step + 1 < max_generate) {
            if (!forward_step(model_path, index_path, current_pos, next_token, token_history, layer_states, logits, true)) {
                std::fprintf(stderr, "generation forward failed at position %d\n", current_pos);
                llama_model_free(v_model); llama_backend_free(); return 6;
            }
            token_history.push_back(next_token);
            current_pos++;
        }
    }
    std::cout << '\n' << std::flush;
    const auto run_done = std::chrono::steady_clock::now();
    const double total_sec = std::chrono::duration<double>(run_done - run_start).count();
    const double first_latency = emitted_piece ? std::chrono::duration<double>(first_piece_time - run_start).count() : 0.0;
    const double gen_elapsed = std::chrono::duration<double>(run_done - prefill_done).count();
    std::fprintf(stderr, "QIES066 prompt_tokens=%zu generated_tokens=%zu max_tokens=%d\n", prompt_toks.size(), generated.size(), max_generate);
    std::fprintf(stderr, "QIES066 prompt_token_ids=");
    for (size_t i = 0; i < prompt_toks.size(); ++i) std::fprintf(stderr, "%s%d", i ? "," : "", (int) prompt_toks[i]);
    std::fprintf(stderr, "\nQIES066 generated_token_ids=");
    for (size_t i = 0; i < generated.size(); ++i) std::fprintf(stderr, "%s%d", i ? "," : "", (int) generated[i]);
    std::fprintf(stderr, "\nQIES066 timing vocab_load_s=%.3f tokenize_s=%.3f prefill_s=%.3f first_token_latency_s=%.3f generation_s=%.3f total_s=%.3f subsequent_tokens_per_s=%.6f state_store_bytes=%llu\n",
            std::chrono::duration<double>(vocab_ready - run_start).count(),
            std::chrono::duration<double>(tokenize_done - tokenize_start).count(), prefill_sec, first_latency, gen_elapsed, total_sec,
            generated.size() > 1 && gen_elapsed > 0 ? (double) (generated.size() - 1) / gen_elapsed : 0.0,
            (unsigned long long) layer_states.total_bytes());

    llama_model_free(v_model);
    llama_backend_free();
    return 0;
}
