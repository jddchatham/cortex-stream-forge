#include "llama.h"
#include "llama-ext.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/flashnext-layer-transaction.h"
#include "../src/flashnext-selected-moe.h"
#include "../src/flashnext-shared-expert.h"
#include "../src/flashnext-segment-b.h"
#include "../src/flashnext-tensor-index.h"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <unordered_map>
#if defined(__linux__)
#include <malloc.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>

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

struct cortex_profile_stats {
    bool enabled = false;
    double decode_sec = 0.0;
    double routing_sec = 0.0;
    double sparse_moe_sec = 0.0;
    double graph_b_sec = 0.0;
    double output_projection_sec = 0.0;
    double checkpoint_sec = 0.0;
    double model_load_sec = 0.0;
    double context_create_sec = 0.0;
    double state_materialize_sec = 0.0;
    double state_restore_sec = 0.0;
    double state_capture_sec = 0.0;
    double state_compare_sec = 0.0;
    double teardown_sec = 0.0;
    uint64_t forward_steps = 0;
};

static cortex_profile_stats g_profile;

static bool cortex_profile_enabled() {
    return g_profile.enabled;
}

static double cortex_now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void cortex_profile_report(const char * label, double wall_sec, double cpu_sec) {
    if (!g_profile.enabled) return;
    std::printf("CORTEX_PROFILE label=%s wall_sec=%.6f cpu_sec=%.6f model_load_sec=%.6f context_create_sec=%.6f state_capture_sec=%.6f state_materialize_sec=%.6f state_restore_sec=%.6f state_compare_sec=%.6f teardown_sec=%.6f decode_sec=%.6f routing_sec=%.6f sparse_moe_sec=%.6f graph_b_sec=%.6f output_projection_sec=%.6f checkpoint_sec=%.6f steps=%llu pread_calls=%llu pread_bytes=%llu\n",
        label, wall_sec, cpu_sec, g_profile.model_load_sec, g_profile.context_create_sec,
        g_profile.state_capture_sec, g_profile.state_materialize_sec,
        g_profile.state_restore_sec, g_profile.state_compare_sec, g_profile.teardown_sec,
        g_profile.decode_sec,
        g_profile.routing_sec, g_profile.sparse_moe_sec, g_profile.graph_b_sec,
        g_profile.output_projection_sec,
        g_profile.checkpoint_sec, (unsigned long long) g_profile.forward_steps,
        (unsigned long long) flashnext_profile::pread_calls(),
        (unsigned long long) flashnext_profile::pread_bytes());
}

struct chat_state_blob {
    static constexpr size_t chunk_size = 64 * 1024;
    size_t size = 0;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> chunks;

    bool empty() const { return size == 0; }
    std::vector<uint8_t> materialize() const {
        std::vector<uint8_t> out(size);
        size_t off = 0;
        for (const auto & c : chunks) { std::memcpy(out.data() + off, c->data(), c->size()); off += c->size(); }
        return out;
    }
};

static uint64_t hash_update(uint64_t h, const void * data, size_t n) {
    const auto * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

template<typename T> static bool ck_write(std::ofstream & f, uint64_t & h, const T & v) {
    f.write(reinterpret_cast<const char *>(&v), sizeof(v));
    if (!f) return false;
    h = hash_update(h, &v, sizeof(v));
    return true;
}

template<typename T> static bool ck_read(std::ifstream & f, uint64_t & h, T & v) {
    f.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!f) return false;
    h = hash_update(h, &v, sizeof(v));
    return true;
}

static bool ck_write_bytes(std::ofstream & f, uint64_t & h, const void * p, size_t n) {
    f.write(static_cast<const char *>(p), n);
    if (!f) return false;
    h = hash_update(h, p, n);
    return true;
}

static bool ck_read_bytes(std::ifstream & f, uint64_t & h, void * p, size_t n) {
    f.read(static_cast<char *>(p), n);
    if (!f) return false;
    h = hash_update(h, p, n);
    return true;
}

static constexpr uint32_t cortex_checkpoint_version = 2;
static constexpr char cortex_checkpoint_magic[8] = {'C','S','F','C','K','P','T','1'};

static std::string checkpoint_identity(const std::string & model_path,
        const std::string & index_path, const std::string & prompt) {
    struct stat st{};
    uint64_t size = 0, mtime = 0;
    if (stat(model_path.c_str(), &st) == 0) {
        size = static_cast<uint64_t>(st.st_size);
        mtime = static_cast<uint64_t>(st.st_mtime);
    }
    const uint64_t prompt_hash = hash_bytes(reinterpret_cast<const uint8_t *>(prompt.data()), prompt.size());
    return model_path + "\n" + std::to_string(size) + "\n" + std::to_string(mtime) +
        "\n" + index_path + "\n" + std::to_string(prompt_hash);
}

static bool write_checkpoint(const std::string & path, const std::string & model_path,
        const std::string & index_path, const std::string & prompt, int current_pos,
        const std::vector<float> & logits, const std::vector<chat_state_blob> & states,
        const std::vector<std::shared_ptr<std::vector<uint8_t>>> & pool, std::string & error) {
    const double profile_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    const std::string tmp = path + ".tmp." + std::to_string(static_cast<unsigned long long>(getpid()));
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) { error = "cannot open temporary checkpoint"; return false; }
    const uint64_t placeholder = 0;
    f.write(cortex_checkpoint_magic, sizeof(cortex_checkpoint_magic));
    f.write(reinterpret_cast<const char *>(&cortex_checkpoint_version), sizeof(cortex_checkpoint_version));
    f.write(reinterpret_cast<const char *>(&placeholder), sizeof(placeholder));
    f.write(reinterpret_cast<const char *>(&placeholder), sizeof(placeholder));
    uint64_t h = 1469598103934665603ull;
    const std::string identity = checkpoint_identity(model_path, index_path, prompt);
    const uint64_t identity_len = identity.size();
    if (!ck_write(f, h, identity_len) || !ck_write_bytes(f, h, identity.data(), identity.size()) ||
        !ck_write(f, h, current_pos)) { error = "checkpoint header write failed"; return false; }
    const uint64_t n_logits = logits.size();
    if (!ck_write(f, h, n_logits) || (n_logits && !ck_write_bytes(f, h, logits.data(), n_logits*sizeof(float)))) {
        error = "checkpoint logits write failed"; return false;
    }
    std::unordered_map<const std::vector<uint8_t> *, uint64_t> ids;
    for (size_t i = 0; i < pool.size(); ++i) ids.emplace(pool[i].get(), static_cast<uint64_t>(i));
    const uint64_t n_pool = pool.size();
    if (!ck_write(f, h, n_pool)) { error = "checkpoint pool header failed"; return false; }
    for (const auto & chunk : pool) {
        const uint64_t n = chunk ? chunk->size() : 0;
        if (!chunk || !ck_write(f, h, n) || (n && !ck_write_bytes(f, h, chunk->data(), n))) {
            error = "checkpoint pool write failed"; return false;
        }
    }
    const uint64_t n_states = states.size();
    if (!ck_write(f, h, n_states)) { error = "checkpoint state header failed"; return false; }
    for (const auto & state : states) {
        const uint64_t size = state.size, n_chunks = state.chunks.size();
        if (!ck_write(f, h, size) || !ck_write(f, h, n_chunks)) { error = "checkpoint state write failed"; return false; }
        for (const auto & chunk : state.chunks) {
            const auto it = ids.find(chunk.get());
            if (it == ids.end() || !ck_write(f, h, it->second)) { error = "checkpoint dangling chunk"; return false; }
        }
    }
    const uint64_t body_size = static_cast<uint64_t>(f.tellp()) - 28;
    f.seekp(12);
    f.write(reinterpret_cast<const char *>(&body_size), sizeof(body_size));
    f.write(reinterpret_cast<const char *>(&h), sizeof(h));
    f.close();
    if (!f) { error = "checkpoint flush failed"; std::remove(tmp.c_str()); return false; }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { error = "checkpoint atomic rename failed"; std::remove(tmp.c_str()); return false; }
    if (cortex_profile_enabled()) g_profile.checkpoint_sec += cortex_now_sec() - profile_start;
    return true;
}

static bool read_checkpoint(const std::string & path, const std::string & model_path,
        const std::string & index_path, const std::string & prompt, int & current_pos,
        std::vector<float> & logits, std::vector<chat_state_blob> & states,
        std::vector<std::shared_ptr<std::vector<uint8_t>>> & pool, std::string & error) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { error = "checkpoint not found"; return false; }
    const std::streamoff file_size = f.tellg(); f.seekg(0);
    if (file_size < 28) { error = "checkpoint truncated"; return false; }
    char magic[8]; uint32_t version = 0; uint64_t body_size = 0, expected_hash = 0;
    f.read(magic, sizeof(magic)); f.read(reinterpret_cast<char *>(&version), sizeof(version));
    f.read(reinterpret_cast<char *>(&body_size), sizeof(body_size)); f.read(reinterpret_cast<char *>(&expected_hash), sizeof(expected_hash));
    if (!f || std::memcmp(magic, cortex_checkpoint_magic, sizeof(magic)) != 0 || version != cortex_checkpoint_version ||
        body_size != static_cast<uint64_t>(file_size - 28)) { error = "checkpoint header incompatible"; return false; }
    uint64_t h = 1469598103934665603ull, identity_len = 0;
    if (!ck_read(f, h, identity_len) || identity_len > 4096) { error = "checkpoint identity invalid"; return false; }
    std::string identity(identity_len, '\0');
    if (!ck_read_bytes(f, h, identity.data(), identity.size()) || identity != checkpoint_identity(model_path, index_path, prompt) ||
        !ck_read(f, h, current_pos)) { error = "checkpoint identity mismatch"; return false; }
    uint64_t n_logits = 0;
    if (!ck_read(f, h, n_logits) || n_logits > 248320) { error = "checkpoint logits invalid"; return false; }
    logits.resize(n_logits);
    if (n_logits && !ck_read_bytes(f, h, logits.data(), n_logits*sizeof(float))) { error = "checkpoint logits truncated"; return false; }
    uint64_t n_pool = 0;
    if (!ck_read(f, h, n_pool) || n_pool > 1000000) { error = "checkpoint pool invalid"; return false; }
    pool.clear(); pool.reserve(n_pool);
    for (uint64_t i = 0; i < n_pool; ++i) {
        uint64_t n = 0; if (!ck_read(f, h, n) || n > (1ull<<20)) { error = "checkpoint chunk invalid"; return false; }
        auto chunk = std::make_shared<std::vector<uint8_t>>(n);
        if (n && !ck_read_bytes(f, h, chunk->data(), n)) { error = "checkpoint chunk truncated"; return false; }
        pool.push_back(std::move(chunk));
    }
    uint64_t n_states = 0;
    if (!ck_read(f, h, n_states) || n_states != 48) { error = "checkpoint layer count invalid"; return false; }
    states.assign(n_states, {});
    for (auto & state : states) {
        uint64_t size = 0, n_chunks = 0;
        if (!ck_read(f, h, size) || !ck_read(f, h, n_chunks) || n_chunks > 100000) { error = "checkpoint state invalid"; return false; }
        state.size = size; state.chunks.reserve(n_chunks);
        size_t total = 0;
        for (uint64_t j = 0; j < n_chunks; ++j) {
            uint64_t id = 0; if (!ck_read(f, h, id) || id >= pool.size()) { error = "checkpoint chunk reference invalid"; return false; }
            state.chunks.push_back(pool[id]); total += pool[id]->size();
        }
        if (total != state.size) { error = "checkpoint state length mismatch"; return false; }
    }
    if (h != expected_hash || f.peek() != std::char_traits<char>::eof()) { error = "checkpoint integrity failure"; return false; }
    return true;
}

static std::vector<uint8_t> capture_seq_state(llama_context * ctx);

static chat_state_blob intern_state(const std::vector<uint8_t> & raw,
        std::vector<std::shared_ptr<std::vector<uint8_t>>> & pool) {
    chat_state_blob out; out.size = raw.size();
    for (size_t off = 0; off < raw.size(); off += chat_state_blob::chunk_size) {
        const size_t len = std::min(chat_state_blob::chunk_size, raw.size() - off);
        std::shared_ptr<std::vector<uint8_t>> found;
        if (std::getenv("CORTEX_CHAT_DISABLE_INTERNING") == nullptr) {
            for (const auto & candidate : pool) {
                if (candidate->size() == len && std::memcmp(candidate->data(), raw.data() + off, len) == 0) { found = candidate; break; }
            }
        }
        if (!found) {
            found = std::make_shared<std::vector<uint8_t>>(raw.begin() + off, raw.begin() + off + len);
            pool.push_back(found);
        }
        out.chunks.push_back(std::move(found));
    }
    return out;
}

static chat_state_blob capture_interned_state(llama_context * ctx,
        std::vector<std::shared_ptr<std::vector<uint8_t>>> & pool) {
    const double capture_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    const std::vector<uint8_t> raw = capture_seq_state(ctx);
    chat_state_blob result = intern_state(raw, pool);
    if (cortex_profile_enabled()) g_profile.state_capture_sec += cortex_now_sec() - capture_start;
    return result;
}

// state_pool is an interning index, not a historical snapshot archive.  Once
// a layer snapshot is replaced, chunks reachable only from the old snapshot
// are no longer needed for future restore and must be released.
static void compact_state_pool(const std::vector<chat_state_blob> & states,
        std::vector<std::shared_ptr<std::vector<uint8_t>>> & pool) {
    std::unordered_set<const std::vector<uint8_t> *> live_ids;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> live;
    for (const auto & state : states) {
        for (const auto & chunk : state.chunks) {
            if (chunk && live_ids.insert(chunk.get()).second) live.push_back(chunk);
        }
    }
    pool.swap(live);
}

static uint64_t checkpoint_state_digest(const std::vector<chat_state_blob> & states,
        const std::vector<float> & logits) {
    uint64_t h = 1469598103934665603ull;
    for (const auto & state : states) {
        h = hash_update(h, &state.size, sizeof(state.size));
        for (const auto & chunk : state.chunks) h = hash_update(h, chunk->data(), chunk->size());
    }
    h = hash_update(h, logits.data(), logits.size()*sizeof(float));
    return h;
}

static void chat_state_compare(const std::vector<chat_state_blob> & states, uint32_t layer, int token) {
    if (layer == 0 || states[layer].empty() || states[layer - 1].empty()) return;
    const double compare_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    const std::vector<uint8_t> a = states[layer - 1].materialize();
    const std::vector<uint8_t> b = states[layer].materialize();
    const size_t n = std::min(a.size(), b.size());
    size_t equal = 0;
    constexpr size_t chunk = 64 * 1024;
    size_t equal_chunks = 0;
    for (size_t off = 0; off < n; off += chunk) {
        const size_t len = std::min(chunk, n - off);
        if (std::memcmp(a.data() + off, b.data() + off, len) == 0) ++equal_chunks;
        for (size_t i = 0; i < len; ++i) equal += a[off + i] == b[off + i];
    }
    std::printf("CHAT_STATE_COMPARE token=%d layer=%u bytes_a=%zu bytes_b=%zu equal_bytes=%zu equal_64k_chunks=%zu hash_a=%016llx hash_b=%016llx\n",
        token, layer, a.size(), b.size(), equal, equal_chunks,
        (unsigned long long) hash_bytes(a.data(), a.size()), (unsigned long long) hash_bytes(b.data(), b.size()));
    std::fflush(stdout);
    if (cortex_profile_enabled()) g_profile.state_compare_sec += cortex_now_sec() - compare_start;
}

struct chat_memory_snapshot {
    uint64_t rss_kib = 0;
    uint64_t hwm_kib = 0;
    uint64_t swap_kib = 0;
    uint64_t available_kib = 0;
};

static chat_memory_snapshot chat_memory_read() {
    chat_memory_snapshot out;
#if defined(__linux__)
    {
        std::ifstream f("/proc/self/status"); std::string k, v, u;
        while (f >> k >> v >> u) {
            const uint64_t n = std::strtoull(v.c_str(), nullptr, 10);
            if (k == "VmRSS:") out.rss_kib = n;
            else if (k == "VmHWM:") out.hwm_kib = n;
            else if (k == "VmSwap:") out.swap_kib = n;
        }
    }
    {
        std::ifstream f("/proc/meminfo"); std::string k, v, u;
        while (f >> k >> v >> u) {
            if (k == "MemAvailable:") { out.available_kib = std::strtoull(v.c_str(), nullptr, 10); break; }
        }
    }
#elif defined(__APPLE__)
    task_vm_info_data_t vm_info{};
    mach_msg_type_number_t vm_count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info), &vm_count) == KERN_SUCCESS) {
        out.rss_kib = vm_info.resident_size / 1024;
        out.hwm_kib = vm_info.resident_size_max / 1024;
    }
    vm_statistics64_data_t vm_stat{};
    mach_msg_type_number_t vm_stat_count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm_stat), &vm_stat_count) == KERN_SUCCESS) {
        uint64_t page = 0; size_t page_len = sizeof(page);
        sysctlbyname("hw.pagesize", &page, &page_len, nullptr, 0);
        out.available_kib = (vm_stat.free_count + vm_stat.inactive_count + vm_stat.speculative_count) * page / 1024;
    }
    xsw_usage swap_usage{}; size_t swap_len = sizeof(swap_usage);
    if (sysctlbyname("vm.swapusage", &swap_usage, &swap_len, nullptr, 0) == 0) {
        out.swap_kib = swap_usage.xsu_used / 1024;
    }
#else
    // Unsupported platforms retain zero values and cannot pass a nonzero guard.
#endif
    return out;
}

static void chat_mem(const char * event, int token, int layer, size_t retained) {
    const chat_memory_snapshot m = chat_memory_read();
    std::printf("CHAT_LIFECYCLE event=%s token=%d layer=%d VmRSS_KiB=%llu VmHWM_KiB=%llu VmSwap_KiB=%llu retained_state_bytes=%zu\n",
        event, token, layer, (unsigned long long) m.rss_kib, (unsigned long long) m.hwm_kib,
        (unsigned long long) m.swap_kib, retained);
    std::fflush(stdout);
    if (std::getenv("CORTEX_CHAT_DUMP_MAPS") != nullptr) {
        char path[256];
        std::snprintf(path, sizeof(path), "/tmp/chat_%s_t%d_l%d.smaps_rollup", event, token, layer);
        std::ifstream in("/proc/self/smaps_rollup"); std::ofstream out(path); out << in.rdbuf();
        std::snprintf(path, sizeof(path), "/tmp/chat_%s_t%d_l%d.maps", event, token, layer);
        std::ifstream maps_in("/proc/self/maps"); std::ofstream mout(path); mout << maps_in.rdbuf();
    }
}

static bool chat_memory_guard(const char * event, int token, int layer) {
    const chat_memory_snapshot m = chat_memory_read();
    const char * rss_env = std::getenv("CORTEX_CHAT_MAX_RSS_KIB");
    const char * avail_env = std::getenv("CORTEX_CHAT_MIN_AVAILABLE_KIB");
    const uint64_t max_rss_kib = rss_env ? std::strtoull(rss_env, nullptr, 10) : 5ULL * 1024ULL * 1024ULL;
    const uint64_t min_available_kib = avail_env ? std::strtoull(avail_env, nullptr, 10) : 2ULL * 1024ULL * 1024ULL;
    const bool ok = m.rss_kib <= max_rss_kib && m.available_kib >= min_available_kib;
    std::printf("CHAT_MEMORY_GUARD event=%s token=%d layer=%d rss_kib=%llu available_kib=%llu max_rss_kib=%llu min_available_kib=%llu result=%s\n",
        event, token, layer, (unsigned long long) m.rss_kib, (unsigned long long) m.available_kib,
        (unsigned long long) max_rss_kib, (unsigned long long) min_available_kib, ok ? "PASS" : "STOP");
    std::fflush(stdout);
    return ok;
}
static void chat_optional_trim(const char * event, int token, int layer, size_t retained) {
#if defined(__linux__)
    if (std::getenv("CORTEX_CHAT_TRIM") != nullptr) {
        malloc_trim(0);
        chat_mem(event, token, layer, retained);
    }
#else
    (void) event; (void) token; (void) layer; (void) retained;
#endif
}

static std::string chat_decode_piece(const llama_vocab * vocab, llama_token token) {
    char piece[256] = {};
    const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
    return n > 0 ? std::string(piece, piece + n) : std::string();
}

static void chat_print_top_logits(const llama_vocab * vocab, const std::vector<float> & logits) {
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const size_t k = std::min<size_t>(5, order.size());
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&logits](int32_t a, int32_t b) {
        return logits[a] > logits[b];
    });
    for (size_t i = 0; i < k; ++i) {
        const int32_t id = order[i];
        const std::string text = chat_decode_piece(vocab, id);
        std::printf("CHAT_TOP_LOGIT rank=%zu token_id=%d logit=%.9g text=%s\n", i + 1, id, logits[id], text.c_str());
    }
}

static bool flashnext_stream_output_q6k(const flashnext_tensor_descriptor & d,
        const std::vector<float> & x, std::vector<float> & logits) {
    if (d.type != GGML_TYPE_Q6_K || d.shape0 != 2560 || d.shape1 != 248320 || x.size() != 2560) return false;
    const double profile_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
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
            flashnext_profile::record_pread(static_cast<size_t>(n));
            done += (size_t) n;
        }
        dequantize_row_q6_K((const block_q6_K *) row.data(), wf.data(), 2560);
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) sum += x[i] * wf[i];
        logits[r] = sum;
    }
    ::close(fd);
    if (cortex_profile_enabled()) g_profile.output_projection_sec += cortex_now_sec() - profile_start;
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

static layer_transaction_result run_layer_decode(
        llama_context * ctx, int position, const std::vector<float> & owned_hidden) {
    layer_transaction_result result;
    llama_batch batch = llama_batch_init(1, (int) owned_hidden.size(), 1);
    if (!batch.embd) { result.failure_stage = "batch_init"; return result; }
    std::memcpy(batch.embd, owned_hidden.data(), owned_hidden.size() * sizeof(float));
    // Embedding batches use one position per M-RoPE section.  llama_batch_init
    // allocates only n_tokens positions, so replace that allocation and set all
    // sections explicitly; leaving the extra sections uninitialized makes the
    // serialized hybrid state process-dependent.
    std::free(batch.pos);
    batch.pos = static_cast<llama_pos *>(std::calloc(GGML_MROPE_SECTIONS, sizeof(llama_pos)));
    for (size_t i = 0; i < GGML_MROPE_SECTIONS; ++i) batch.pos[i] = position;
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
    (void) layer;
    // The graph's Segment-A output is the authoritative transaction boundary.
    // Do not use the diagnostic-owned tensors here: they are optional scratch
    // captures and can be stale or absent independently of the packed artifact.
    // Layout is [MoE input: 2560] [HC residual: 10240] [FFN injection: 4]
    // [router logits: 512], all F32, with router offset 12804.
    constexpr size_t moe_floats = 2560;
    constexpr size_t residual_floats = 2560 * 4;
    constexpr size_t inject_floats = 4;
    constexpr size_t router_floats = 512;
    constexpr size_t router_offset = moe_floats + residual_floats + inject_floats;
    const float * artifact = llama_get_logits_ith(ctx, 0);
    if (!artifact) return false;
    std::memcpy(txc.moe_input.data(), artifact, moe_floats * sizeof(float));
    std::memcpy(txc.residual_hc.data(), artifact + moe_floats, residual_floats * sizeof(float));
    std::memcpy(txc.injection.data(), artifact + moe_floats + residual_floats, inject_floats * sizeof(float));
    std::memcpy(txc.router_logits.data(), artifact + router_offset, router_floats * sizeof(float));
    for (float x : txc.moe_input) if (!std::isfinite(x)) return false;
    for (float x : txc.router_logits) if (!std::isfinite(x)) return false;
    for (float x : txc.residual_hc) if (!std::isfinite(x)) return false;
    for (float x : txc.injection) if (!std::isfinite(x)) return false;
    return true;
}

static bool execute_transaction_sparse_moe(
        flashnext_transaction_context & txc, const std::string & index_path, std::string & error) {
    if (!txc.tensor_index) { error = "missing tensor index"; return false; }
    const double route_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
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
    std::memcpy(txc.values.residual_hc.data(), txc.residual_hc.data(), sizeof(txc.residual_hc));
    std::memcpy(txc.values.inject.data(), txc.injection.data(), sizeof(txc.injection));
    if (cortex_profile_enabled()) g_profile.routing_sec += cortex_now_sec() - route_start;
    const double expert_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    if (!txc.tensor_index->load(index_path, &error)) return false;
    if (!flashnext_execute_selected_moe(*txc.tensor_index, index_path, txc.values)) return false;
    if (cortex_profile_enabled()) g_profile.sparse_moe_sec += cortex_now_sec() - expert_start;
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

    const std::array<flashnext_tensor_descriptor, 4> descs{{sg, su, sd, sc}};
    std::array<int, 4> fds{{-1, -1, -1, -1}};
    for (size_t p = 0; p < 4; ++p) {
        fds[p] = ::open(descs[p].path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fds[p] < 0) {
            for (size_t c = 0; c < p; ++c) if (fds[c] >= 0) ::close(fds[c]);
            error = "failed to open shared expert shard";
            return false;
        }
    }
    flashnext_shared_expert_slices shared{fds, sg.offset, su.offset, sd.offset, sc.offset};
    const bool shared_ok = flashnext_execute_shared_expert(shared, txc.values);
    for (int & fd : fds) { if (fd >= 0) { ::close(fd); fd = -1; } }
    if (!shared_ok) { error = "shared expert execution failed"; return false; }

    for (size_t i = 0; i < txc.values.complete_moe.size(); ++i) {
        txc.values.complete_moe[i] = txc.values.selected_moe[i] + txc.values.shared_moe[i];
    }
    if (!flashnext_execute_segment_b(txc.values)) { error = "segment B execution failed"; return false; }
    txc.output = txc.values.layer_output;
    txc.state_committed = true;
    return true;
}

static layer_transaction_result run_layer_transaction(
        llama_context * ctx, uint32_t layer, int position, const std::vector<float> & owned_hidden,
        const std::string & index_path) {
    const double decode_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    layer_transaction_result result = run_layer_decode(ctx, position, owned_hidden);
    if (cortex_profile_enabled()) g_profile.decode_sec += cortex_now_sec() - decode_start;
    if (result.decode_status != 0) {
        std::fprintf(stderr, "CORTEX_PROFILE_DECODE_FAIL layer=%u position=%d status=%d\n",
            layer, position, result.decode_status);
        result.failure_stage = "graph_a_decode"; return result;
    }
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
    const double graph_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    if (!execute_transaction_graph_b(txc, index_path, error)) {
        if (!result.state_before.empty()) llama_state_seq_set_data(ctx, result.state_before.data(), result.state_before.size(), 0);
        result.failure_stage = "graph_b";
        result.rollback_restored = true;
        return result;
    }
    if (cortex_profile_enabled()) g_profile.graph_b_sec += cortex_now_sec() - graph_start;
    result.graph_b_ok = true;
    result.output = txc.output;
    result.output_hash = hash_f32(result.output.data(), result.output.size());
    result.selected_ids = txc.selected_ids;
    result.unselected_expert_reads = txc.values.io.unselected_expert_reads;
    result.state_committed = true;
    return result;
}

static bool forward_step(
        const std::string & model_path,
        const std::string & index_path,
        int position,
        llama_token token_id,
        std::vector<chat_state_blob> & layer_states,
        std::vector<std::shared_ptr<std::vector<uint8_t>>> & state_pool,
        std::vector<float> & out_logits,
        bool compute_logits) {
    if (cortex_profile_enabled()) ++g_profile.forward_steps;
    auto retained = [&]() { size_t n = 0; for (const auto & s : layer_states) n += s.size; return n; };

    // Clean any residual environment variables
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD");
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY");
    unsetenv("CORTEX_FLASHNEXT_FINAL_HEAD_NORM_ONLY");
    unsetenv("CORTEX_FLASHNEXT_LAYER_RANGE");
    unsetenv("CORTEX_FLASHNEXT_PRE_MOE");
    unsetenv("CORTEX_FLASHNEXT_LAYER_START");
    unsetenv("CORTEX_FLASHNEXT_LAYER_END");
    unsetenv("CORTEX_QWEN35_SEGMENT_END");

    // 1. Layer 0 execution
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "0", 1);
    setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1);
    setenv("CORTEX_FLASHNEXT_SEGMENT_A", "1", 1);
    setenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER", "1", 1);
    setenv("CORTEX_FLASHNEXT_EXTERNAL_EMBEDDING", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_START", "0", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_END", "1", 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", "1", 1);

    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    chat_mem("before_layer0_model_load", position, 0, retained());
    const double model0_load_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    llama_model * model0 = llama_model_load_from_file(model_path.c_str(), mp);
    if (cortex_profile_enabled()) g_profile.model_load_sec += cortex_now_sec() - model0_load_start;
    if (!model0) return false;
    chat_mem("after_layer0_model_load", position, 0, retained());
    if (!chat_memory_guard("after_layer0_model_load", position, 0)) { llama_model_free(model0); return false; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 1;
    const double ctx0_create_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    llama_context * ctx0 = llama_init_from_model(model0, cp);
    if (cortex_profile_enabled()) g_profile.context_create_sec += cortex_now_sec() - ctx0_create_start;
    if (!ctx0) { llama_model_free(model0); return false; }
    chat_mem("after_layer0_context_load", position, 0, retained());
    if (!chat_memory_guard("after_layer0_context_load", position, 0)) { llama_free(ctx0); llama_model_free(model0); return false; }

    // Restore layer 0 state if non-empty
    if (!layer_states[0].empty()) {
        const double state_materialize_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
        const std::vector<uint8_t> state = layer_states[0].materialize();
        if (cortex_profile_enabled()) g_profile.state_materialize_sec += cortex_now_sec() - state_materialize_start;
        const double state_restore_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
        llama_state_seq_set_data(ctx0, state.data(), state.size(), 0);
        if (cortex_profile_enabled()) g_profile.state_restore_sec += cortex_now_sec() - state_restore_start;
    }

    std::array<float, 2560> token_embedding{};
    if (!llama_cortex_external_token_rows(model0, &token_id, 1, token_embedding.data())) {
        llama_free(ctx0); llama_model_free(model0); return false;
    }
    llama_batch batch0 = llama_batch_init(1, 2560, 1);
    std::memcpy(batch0.embd, token_embedding.data(), token_embedding.size() * sizeof(float));
    std::free(batch0.pos);
    batch0.pos = static_cast<llama_pos *>(std::calloc(GGML_MROPE_SECTIONS, sizeof(llama_pos)));
    for (size_t i = 0; i < GGML_MROPE_SECTIONS; ++i) batch0.pos[i] = position;
    batch0.n_seq_id[0] = 1;
    batch0.seq_id[0][0] = 0; batch0.logits[0] = 1; batch0.n_tokens = 1;
    const int layer0_decode_rc = llama_decode(ctx0, batch0);
    if (layer0_decode_rc != 0) {
        std::fprintf(stderr, "Layer 0 decode failed at pos %d status=%d\n", position, layer0_decode_rc);
        llama_batch_free(batch0); llama_free(ctx0); llama_model_free(model0); return false;
    }
    llama_synchronize(ctx0);
    layer_states[0] = capture_interned_state(ctx0, state_pool);
    llama_batch_free(batch0);

    const float * artifact = llama_get_logits_ith(ctx0, 0);
    if (!artifact) { llama_free(ctx0); llama_model_free(model0); return false; }

    flashnext_layer_transaction tx0;
    tx0.layer = 0;
    tx0.position = position;
    std::memcpy(tx0.pre_moe_hidden.data(), artifact, tx0.pre_moe_hidden.size()*sizeof(float));
    std::memcpy(tx0.residual_hc.data(), artifact + 2560, tx0.residual_hc.size()*sizeof(float));
    std::memcpy(tx0.inject.data(), artifact + 2560 + tx0.residual_hc.size(), tx0.inject.size()*sizeof(float));
    std::array<int, 512> order0{};
    for (int i = 0; i < 512; ++i) order0[i] = i;
    const float * route0 = artifact + 12804;
    std::partial_sort(order0.begin(), order0.begin()+10, order0.end(), [route0](int a, int b) { return route0[a] > route0[b]; });
    double z0 = 0.0; double ex0[10]; const float mx0 = route0[order0[0]];
    for (int i = 0; i < 10; ++i) { ex0[i] = std::exp(static_cast<double>(route0[order0[i]]) - static_cast<double>(mx0)); z0 += ex0[i]; }
    for (int i = 0; i < 10; ++i) { tx0.selected_expert_ids[i] = order0[i]; tx0.routing_weights[i] = ex0[i] / z0; }

    flashnext_tensor_index index; std::string index_error;
    if (!index.load(index_path, &index_error)) { llama_free(ctx0); llama_model_free(model0); return false; }
    if (!flashnext_execute_selected_moe(index, index_path, tx0)) { llama_free(ctx0); llama_model_free(model0); return false; }

    flashnext_tensor_descriptor sg, su, sd, sc;
    if (!index.resolve_shared(0, flashnext_projection::gate, sg, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::up, su, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::down, sd, &index_error) ||
        !index.resolve_shared(0, flashnext_projection::shared_control, sc, &index_error)) {
        llama_free(ctx0); llama_model_free(model0); return false;
    }
    const std::array<flashnext_tensor_descriptor, 4> descs0{{sg, su, sd, sc}};
    std::array<int, 4> fds0{{-1, -1, -1, -1}};
    for (size_t p = 0; p < 4; ++p) {
        fds0[p] = ::open(descs0[p].path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fds0[p] < 0) {
            for (size_t c = 0; c < p; ++c) if (fds0[c] >= 0) ::close(fds0[c]);
            llama_free(ctx0); llama_model_free(model0); return false;
        }
    }
    flashnext_shared_expert_slices shared0{fds0, sg.offset, su.offset, sd.offset, sc.offset};
    const bool shared0_ok = flashnext_execute_shared_expert(shared0, tx0);
    for (int & fd : fds0) { if (fd >= 0) { ::close(fd); fd = -1; } }
    if (!shared0_ok) { llama_free(ctx0); llama_model_free(model0); return false; }

    for (size_t i = 0; i < tx0.complete_moe.size(); ++i) tx0.complete_moe[i] = tx0.selected_moe[i] + tx0.shared_moe[i];
    if (!flashnext_execute_segment_b(tx0)) { llama_free(ctx0); llama_model_free(model0); return false; }

    std::vector<float> current_output(tx0.layer_output.begin(), tx0.layer_output.end());
    const double layer0_ctx_free_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    llama_free(ctx0);
    if (cortex_profile_enabled()) g_profile.teardown_sec += cortex_now_sec() - layer0_ctx_free_start;
    chat_mem("after_layer0_context_free", position, 0, retained());
    const double layer0_model_free_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    llama_model_free(model0);
    if (cortex_profile_enabled()) g_profile.teardown_sec += cortex_now_sec() - layer0_model_free_start;
    chat_mem("after_layer0_model_free", position, 0, retained());
    chat_optional_trim("after_layer0_trim", position, 0, retained());

    // 2. Layers 1..47
    unsetenv("CORTEX_FLASHNEXT_ROUTER_ONLY");
    unsetenv("CORTEX_FLASHNEXT_SEGMENT_A");
    unsetenv("CORTEX_FLASHNEXT_SEGMENT_A_ROUTER");
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
    setenv("CORTEX_FLASHNEXT_INTERNAL_TENSOR_DIAGNOSTICS", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER3_TRANSACTION", "1", 1);

    const uint32_t layer_limit = std::min<uint32_t>(48, std::max<uint32_t>(1,
        std::getenv("CORTEX_CHAT_LAYER_LIMIT") ? std::strtoul(std::getenv("CORTEX_CHAT_LAYER_LIMIT"), nullptr, 10) : 48));
    for (uint32_t il = 1; il < layer_limit; ++il) {
        setenv("CORTEX_LOW_RESIDENCY_LAYER", std::to_string(il).c_str(), 1);
        setenv("CORTEX_FLASHNEXT_LAYER_START", std::to_string(il).c_str(), 1);
        setenv("CORTEX_FLASHNEXT_LAYER_END", std::to_string(il + 1).c_str(), 1);
        setenv("CORTEX_QWEN35_SEGMENT_END", std::to_string(il + 1).c_str(), 1);

        chat_mem("before_layer_model_load", position, il, retained());
        const double next_model_load_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
        llama_model * next_model = llama_model_load_from_file(model_path.c_str(), mp);
        if (cortex_profile_enabled()) g_profile.model_load_sec += cortex_now_sec() - next_model_load_start;
        if (!next_model) return false;
        chat_mem("after_layer_model_load", position, il, retained());
        if (!chat_memory_guard("after_layer_model_load", position, il)) { llama_model_free(next_model); return false; }
        llama_context_params next_cp = cp;
        next_cp.embeddings = true;
        const double next_ctx_create_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
        llama_context * next_ctx = llama_init_from_model(next_model, next_cp);
        if (cortex_profile_enabled()) g_profile.context_create_sec += cortex_now_sec() - next_ctx_create_start;
        if (!next_ctx) { llama_model_free(next_model); return false; }
        if (!chat_memory_guard("after_layer_context_load", position, il)) { llama_free(next_ctx); llama_model_free(next_model); return false; }

        if (!layer_states[il].empty()) {
            const double state_materialize_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
            const std::vector<uint8_t> state = layer_states[il].materialize();
            if (cortex_profile_enabled()) g_profile.state_materialize_sec += cortex_now_sec() - state_materialize_start;
            const double state_restore_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
            llama_state_seq_set_data(next_ctx, state.data(), state.size(), 0);
            if (cortex_profile_enabled()) g_profile.state_restore_sec += cortex_now_sec() - state_restore_start;
        }

        layer_transaction_result res = run_layer_transaction(next_ctx, il, position, current_output, index_path);
        if (!res.state_committed) {
            std::fprintf(stderr, "Layer %u transaction failed at pos %d\n", il, position);
            llama_free(next_ctx); llama_model_free(next_model); return false;
        }
        layer_states[il] = capture_interned_state(next_ctx, state_pool);
        chat_state_compare(layer_states, il, position);
        current_output.assign(res.output.begin(), res.output.end());
        // The transaction's pre-decode snapshot is only needed until the
        // transaction commits.  The serialized state copied above is the
        // sole state required for the next token; retaining state_before
        // through context/model destruction duplicated ~118 MiB per layer.
        res.state_before.clear();
        res.state_before.shrink_to_fit();
        const double next_ctx_free_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
        llama_free(next_ctx);
        if (cortex_profile_enabled()) g_profile.teardown_sec += cortex_now_sec() - next_ctx_free_start;
        chat_mem("after_layer_context_free", position, il, retained());
        const double next_model_free_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
        llama_model_free(next_model);
        if (cortex_profile_enabled()) g_profile.teardown_sec += cortex_now_sec() - next_model_free_start;
        chat_mem("after_layer_model_free", position, il, retained());
        if (!chat_memory_guard("after_layer_model_free", position, il)) return false;
        chat_optional_trim("after_layer_trim", position, il, retained());
    }

    if (!compute_logits || layer_limit < 48) return true;

    // 3. Final Head Projection
    unsetenv("CORTEX_FLASHNEXT_PRE_MOE");
    unsetenv("CORTEX_FLASHNEXT_LAYER_END");
    unsetenv("CORTEX_QWEN35_SEGMENT_END");
    setenv("CORTEX_FLASHNEXT_LAYER_START", "48", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_END", "48", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD", "1", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD_NORM_ONLY", "1", 1);
    setenv("CORTEX_FLASHNEXT_FINAL_HEAD_ONLY", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", "47", 1);

    const double head_model_load_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    llama_model * head_model = llama_model_load_from_file(model_path.c_str(), mp);
    if (cortex_profile_enabled()) g_profile.model_load_sec += cortex_now_sec() - head_model_load_start;
    if (!head_model) return false;
    llama_context_params head_cp = cp;
    head_cp.embeddings = true;
    const double head_ctx_create_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    llama_context * head_ctx = llama_init_from_model(head_model, head_cp);
    if (cortex_profile_enabled()) g_profile.context_create_sec += cortex_now_sec() - head_ctx_create_start;
    if (!head_ctx) { llama_model_free(head_model); return false; }

    llama_batch head_batch = llama_batch_init(1, (int) current_output.size(), 1);
    std::memcpy(head_batch.embd, current_output.data(), current_output.size() * sizeof(float));
    std::free(head_batch.pos);
    head_batch.pos = static_cast<llama_pos *>(std::calloc(GGML_MROPE_SECTIONS, sizeof(llama_pos)));
    // head has no KV state, pos 0 is clean
    for (size_t i = 0; i < GGML_MROPE_SECTIONS; ++i) head_batch.pos[i] = 0;
    head_batch.n_seq_id[0] = 1;
    head_batch.seq_id[0][0] = 0;
    head_batch.logits[0] = 1;
    head_batch.n_tokens = 1;
    const int head_rc = llama_decode(head_ctx, head_batch);
    llama_synchronize(head_ctx);
    llama_batch_free(head_batch);

    if (head_rc != 0) {
        std::fprintf(stderr, "Head decode failed at pos %d\n", position);
        llama_free(head_ctx); llama_model_free(head_model); return false;
    }

    std::vector<uint8_t> norm_bytes;
    std::array<int64_t, 4> norm_ne{};
    std::array<size_t, 4> norm_nb{};
    enum ggml_type norm_type = GGML_TYPE_COUNT;
    std::string norm_name;
    const bool norm_read = head_ctx->debug_copy_final_head_output(norm_bytes, norm_ne, norm_nb, norm_type, norm_name);
    const double head_ctx_free_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    llama_free(head_ctx);
    if (cortex_profile_enabled()) g_profile.teardown_sec += cortex_now_sec() - head_ctx_free_start;
    const double head_model_free_start = cortex_profile_enabled() ? cortex_now_sec() : 0.0;
    llama_model_free(head_model);
    if (cortex_profile_enabled()) g_profile.teardown_sec += cortex_now_sec() - head_model_free_start;

    if (!norm_read || norm_type != GGML_TYPE_F32 || norm_bytes.size() != 2560 * sizeof(float)) {
        std::fprintf(stderr, "Head norm copy failed\n");
        return false;
    }

    const float * norm_p = (const float *) norm_bytes.data();
    std::vector<float> norm_owned(norm_p, norm_p + 2560);

    flashnext_tensor_descriptor out_desc;
    if (!index.resolve_named("output.weight", out_desc, &index_error)) return false;
    if (!flashnext_stream_output_q6k(out_desc, norm_owned, out_logits)) return false;

    return true;
}


int main(int argc, char ** argv) {
    std::string model_path = "/home/user/models/qwen38-flashnext-abliterated/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf";
    if (argc >= 2) {
        model_path = argv[1];
    }
    const char * index_env = std::getenv("CORTEX_TENSOR_INDEX");
    const std::string index_path = index_env ? index_env : "flashnext-tensor-index.json";
    int max_tokens = 32;
    if (argc >= 3) {
        max_tokens = std::atoi(argv[2]);
    }
    g_profile.enabled = std::getenv("CORTEX_PROFILE_RESUME") != nullptr;
    const double process_wall_start = cortex_now_sec();
    const std::clock_t process_cpu_start = std::clock();

    llama_backend_init();

    llama_model_params v_mp = llama_model_default_params();
    v_mp.vocab_only = true;
    llama_model * v_model = llama_model_load_from_file(model_path.c_str(), v_mp);
    if (!v_model) {
        std::fprintf(stderr, "Failed to load vocabulary\n");
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(v_model);

    if (std::getenv("CORTEX_CHAT_TOKENIZER_PROBE") != nullptr) {
        const std::array<llama_token, 22> ids = {
            248045, 846, 198, 3710, 369, 220, 17, 478, 220, 17,
            30, 21134, 440, 1132, 279, 1324, 13, 248046, 198, 248045,
            74455, 198,
        };
        std::printf("TOKENIZER_PROBE_IDS count=%zu\n", ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            char piece[256] = {};
            const int n = llama_token_to_piece(vocab, ids[i], piece, sizeof(piece), 0, true);
            std::printf("TOKENIZER_PROBE id=%d piece_len=%d piece=", ids[i], n);
            for (int j = 0; j < n; ++j) std::printf("%02x", (unsigned char) piece[j]);
            std::printf("\n");
        }
        const std::string text = "What is 2 + 2? Answer with only the number.";
        const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, false, true);
        std::vector<llama_token> tokenized(n > 0 ? n : 0);
        const int got = n > 0 ? llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), tokenized.data(), n, false, true) : n;
        std::printf("TOKENIZER_PROBE_TEXT count=%d ids=", got);
        for (int i = 0; i < got; ++i) std::printf("%s%d", i ? "," : "", tokenized[i]);
        std::printf("\n");
        llama_model_free(v_model); llama_backend_free();
        return 0;
    }

    std::vector<chat_state_blob> layer_states(48);
    std::vector<std::shared_ptr<std::vector<uint8_t>>> state_pool;
    int current_pos = 0;

    if (std::getenv("CORTEX_CHAT_TWO_TOKEN_DIAGNOSTIC") != nullptr) {
        const llama_token diagnostic_tokens[3] = { 248045, 846, 198 };
        int diagnostic_count = 2;
        if (const char * count_env = std::getenv("CORTEX_CHAT_DIAGNOSTIC_COUNT")) {
            diagnostic_count = std::max(1, std::min(3, std::atoi(count_env)));
        }
        std::vector<float> ignored_logits;
        bool ok = true;
        llama_model_free(v_model); v_model = nullptr;
        for (int i = 0; i < diagnostic_count; ++i) {
            ok = forward_step(model_path, index_path, i, diagnostic_tokens[i], layer_states, state_pool, ignored_logits, false) && ok;
            compact_state_pool(layer_states, state_pool);
            size_t state_bytes = 0; for (const auto & s : layer_states) state_bytes += s.size;
            size_t stored_bytes = 0; for (const auto & c : state_pool) stored_bytes += c->size();
            std::printf("CHAT_TOKEN_RESULT token_index=%d token_id=%d ok=%s retained_state_bytes=%zu\n",
                i, diagnostic_tokens[i], ok ? "YES" : "NO", state_bytes);
            std::printf("CHAT_STATE_STORAGE token_index=%d logical_bytes=%zu unique_chunk_bytes=%zu chunks=%zu\n",
                i, state_bytes, stored_bytes, state_pool.size());
            if (std::getenv("CORTEX_CHAT_REINIT_BACKEND") != nullptr) {
                llama_backend_free(); llama_backend_init();
                std::printf("CHAT_BACKEND_REINIT token_index=%d\n", i); std::fflush(stdout);
            }
            if (!ok) break;
        }
        llama_backend_free();
        return ok ? 0 : 9;
    }

    if (std::getenv("CORTEX_CHAT_ACCEPTANCE_22") != nullptr) {
        const std::array<llama_token, 22> expected = {
            248045, 846, 198, 3710, 369, 220, 17, 478, 220, 17,
            30, 21134, 440, 1132, 279, 1324, 13, 248046, 198, 248045,
            74455, 198,
        };
        static_assert(expected.size() == 22);
        const std::string acceptance_prompt =
            "<|im_start|>user\nWhat is 2 + 2? Answer with only the number."
            "<|im_end|>\n<|im_start|>assistant\n";
        const int32_t token_count = -llama_tokenize(vocab, acceptance_prompt.data(),
            (int32_t) acceptance_prompt.size(), nullptr, 0, false, true);
        std::vector<llama_token> tokenized(token_count > 0 ? (size_t) token_count : 0);
        const int32_t tokenized_count = token_count > 0 ? llama_tokenize(vocab,
            acceptance_prompt.data(), (int32_t) acceptance_prompt.size(), tokenized.data(),
            token_count, false, true) : token_count;
        if (tokenized_count != (int32_t) expected.size() ||
            !std::equal(expected.begin(), expected.end(), tokenized.begin())) {
            std::fprintf(stderr, "ACCEPTANCE_TOKEN_ASSERT_FAIL tokenizer sequence mismatch count=%d\n", tokenized_count);
            llama_model_free(v_model); llama_backend_free(); return 10;
        }
        if (std::getenv("CORTEX_CHAT_DISABLE_INTERNING") != nullptr) {
            std::fprintf(stderr, "ACCEPTANCE_REQUIRES_INTERNING_FAIL\n");
            llama_model_free(v_model); llama_backend_free(); return 10;
        }
        if (const char * layer_env = std::getenv("CORTEX_CHAT_LAYER_LIMIT");
            layer_env != nullptr && std::atoi(layer_env) != 48) {
            std::fprintf(stderr, "ACCEPTANCE_REQUIRES_48_LAYERS_FAIL configured=%s\n", layer_env);
            llama_model_free(v_model); llama_backend_free(); return 10;
        }
        std::printf("ACCEPTANCE_TOKEN_ASSERT PASS count=22\n");
        const auto acceptance_start = std::chrono::steady_clock::now();
        std::vector<float> logits;
        bool prefill_ok = true;
        for (int i = 0; i < 22; ++i) {
            const bool last = i == 21;
            prefill_ok = forward_step(model_path, index_path, i, expected[i], layer_states, state_pool, logits, last) && prefill_ok;
            compact_state_pool(layer_states, state_pool);
            size_t logical = 0; for (const auto & s : layer_states) logical += s.size;
            size_t unique = 0; for (const auto & c : state_pool) unique += c->size();
            std::printf("ACCEPTANCE_PREFILL token_index=%d token_id=%d layers=48 transactions=%d ok=%s logical_state_bytes=%zu unique_state_bytes=%zu\n",
                i, expected[i], (i + 1) * 48, prefill_ok ? "YES" : "NO", logical, unique);
            if (!prefill_ok) break;
        }
        std::printf("ACCEPTANCE_PREFILL_RESULT=%s\n", prefill_ok ? "PASS" : "FAIL");
        if (!prefill_ok || logits.empty()) { llama_backend_free(); return 11; }
        if (const char * checkpoint_path = std::getenv("CORTEX_CHAT_SAVE_CHECKPOINT")) {
            std::string checkpoint_error;
            if (!write_checkpoint(checkpoint_path, model_path, index_path, acceptance_prompt, 22,
                    logits, layer_states, state_pool, checkpoint_error)) {
                std::fprintf(stderr, "CHECKPOINT_SAVE_FAIL path=%s error=%s\n", checkpoint_path, checkpoint_error.c_str());
                llama_backend_free(); return 13;
            }
            std::printf("CHECKPOINT_SAVE PASS path=%s position=22 logits=%zu digest=%016llx\n", checkpoint_path, logits.size(),
                (unsigned long long) checkpoint_state_digest(layer_states, logits));
        }
        bool finite = true;
        for (float x : logits) finite = finite && std::isfinite(x);
        std::printf("ACCEPTANCE_FINITE_LOGITS=%s count=%zu\n", finite ? "PASS" : "FAIL", logits.size());
        chat_print_top_logits(vocab, logits);
        const llama_token next = (llama_token) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        const std::string piece = chat_decode_piece(vocab, next);
        std::printf("ACCEPTANCE_TOKEN token_id=%d text=%s decode=%s\n", next, piece.c_str(), piece.empty() ? "FAIL" : "PASS");
        const bool arithmetic = piece == "4" || piece == " 4";
        std::printf("ACCEPTANCE_ARITHMETIC expected=4 result=%s\n", arithmetic ? "PASS" : "FAIL");
        const auto mem = chat_memory_read();
        size_t unique = 0; for (const auto & c : state_pool) unique += c->size();
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - acceptance_start).count();
        std::printf("ACCEPTANCE_FINAL_MEMORY rss_kib=%zu hwm_kib=%zu swap_kib=%zu available_kib=%zu unique_state_bytes=%zu elapsed_sec=%.3f\n",
            mem.rss_kib, mem.hwm_kib, mem.swap_kib, mem.available_kib, unique, elapsed);
        llama_model_free(v_model); v_model = nullptr;
        llama_backend_free();
        return prefill_ok && finite && !piece.empty() && arithmetic ? 0 : 12;
    }

    if (const char * checkpoint_path = std::getenv("CORTEX_CHAT_RESUME_CHECKPOINT")) {
        const std::string acceptance_prompt =
            "<|im_start|>user\nWhat is 2 + 2? Answer with only the number."
            "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<chat_state_blob> resumed_states;
        std::vector<std::shared_ptr<std::vector<uint8_t>>> resumed_pool;
        std::vector<float> logits;
        int resumed_pos = 0;
        std::string checkpoint_error;
        if (!read_checkpoint(checkpoint_path, model_path, index_path, acceptance_prompt, resumed_pos,
                logits, resumed_states, resumed_pool, checkpoint_error)) {
            std::fprintf(stderr, "CHECKPOINT_LOAD_FAIL path=%s error=%s\n", checkpoint_path, checkpoint_error.c_str());
            llama_model_free(v_model); llama_backend_free(); return 14;
        }
        std::printf("CHECKPOINT_LOAD PASS path=%s position=%d logits=%zu unique_chunks=%zu\n",
            checkpoint_path, resumed_pos, logits.size(), resumed_pool.size());
        const std::string progress_path = std::getenv("CORTEX_CHAT_PROGRESS_CHECKPOINT") != nullptr
            ? std::getenv("CORTEX_CHAT_PROGRESS_CHECKPOINT") : checkpoint_path;
        const auto resume_mem_start = chat_memory_read();
        for (int gen_step = 0; gen_step < max_tokens && !logits.empty(); ++gen_step) {
            const llama_token next_token = (llama_token) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
            const std::string piece = chat_decode_piece(vocab, next_token);
            std::printf("RESUME_TOKEN step=%d position=%d token_id=%d text=", gen_step, resumed_pos, next_token);
            std::fwrite(piece.data(), 1, piece.size(), stdout); std::printf("\n");
            if (!forward_step(model_path, index_path, resumed_pos, next_token, resumed_states, resumed_pool, logits, true)) {
                std::fprintf(stderr, "CHECKPOINT_RESUME_FORWARD_FAIL position=%d\n", resumed_pos); break;
            }
            compact_state_pool(resumed_states, resumed_pool);
            ++resumed_pos;
            std::string save_error;
            if (!write_checkpoint(progress_path, model_path, index_path, acceptance_prompt, resumed_pos,
                    logits, resumed_states, resumed_pool, save_error)) {
                std::fprintf(stderr, "CHECKPOINT_PROGRESS_SAVE_FAIL position=%d error=%s\n", resumed_pos, save_error.c_str()); break;
            }
            std::printf("CHECKPOINT_PROGRESS position=%d path=%s digest=%016llx\n", resumed_pos, progress_path.c_str(),
                (unsigned long long) checkpoint_state_digest(resumed_states, logits));
        }
        if (g_profile.enabled) {
            const auto mem = chat_memory_read();
            const double cpu_sec = double(std::clock() - process_cpu_start) / double(CLOCKS_PER_SEC);
            cortex_profile_report("resume", cortex_now_sec() - process_wall_start, cpu_sec);
            std::printf("CORTEX_PROFILE_MEMORY start_rss_kib=%zu end_rss_kib=%zu end_hwm_kib=%zu end_swap_kib=%zu end_available_kib=%zu\n",
                resume_mem_start.rss_kib, mem.rss_kib, mem.hwm_kib, mem.swap_kib, mem.available_kib);
        }
        llama_model_free(v_model); llama_backend_free();
        return 0;
    }

    std::cout << "\n======================================================\n";
    std::cout << "  Cortex FlashNext 90.78 GiB Chat Runtime (48 Layers)  \n";
    std::cout << "  Model: Qwen 3.8 Flash Next Abliterated (IQ4_XS)     \n";
    std::cout << "  Hardware: AMD Athlon Silver 3050U (2 Cores, AVX2)   \n";
    std::cout << "======================================================\n";
    std::cout << "Type /exit to quit.\n\n" << std::flush;

    std::string user_input;
    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, user_input)) break;
        if (user_input == "/exit") break;
        if (user_input.empty()) continue;

        std::string turn_prompt = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";
        int n_toks = -llama_tokenize(vocab, turn_prompt.c_str(), (int32_t) turn_prompt.size(), nullptr, 0, false, true);
        std::vector<llama_token> prompt_toks(n_toks);
        llama_tokenize(vocab, turn_prompt.c_str(), (int32_t) turn_prompt.size(), prompt_toks.data(), n_toks, false, true);

        std::cout << "Cortex FlashNext: " << std::flush;

        // Prefill prompt tokens
        std::vector<float> logits;
        for (size_t i = 0; i < prompt_toks.size(); ++i) {
            const bool is_last = (i + 1 == prompt_toks.size());
            if (!forward_step(model_path, index_path, current_pos, prompt_toks[i], layer_states, state_pool, logits, is_last)) {
                std::fprintf(stderr, "\n[ERROR] Prefill step failed at pos %d\n", current_pos);
                break;
            }
            compact_state_pool(layer_states, state_pool);
            current_pos++;
        }

        // Autoregressive generation
        for (int gen_step = 0; gen_step < max_tokens; ++gen_step) {
            if (logits.empty()) break;

            llama_token next_token = (llama_token) std::distance(logits.begin(),
                std::max_element(logits.begin(), logits.end()));

            // Stop tokens
            if (next_token == 248046 || next_token == 248044) {
                break;
            }

            char piece[128] = {0};
            int piece_len = llama_token_to_piece(vocab, next_token, piece, sizeof(piece), 0, false);
            if (piece_len > 0) {
                std::cout.write(piece, piece_len);
                std::cout.flush();
            }

            if (!forward_step(model_path, index_path, current_pos, next_token, layer_states, state_pool, logits, true)) {
                std::fprintf(stderr, "\n[ERROR] Generation step failed at pos %d\n", current_pos);
                break;
            }
            compact_state_pool(layer_states, state_pool);
            current_pos++;
        }
        std::cout << "\n\n" << std::flush;
    }

    llama_model_free(v_model);
    llama_backend_free();
    return 0;
}
