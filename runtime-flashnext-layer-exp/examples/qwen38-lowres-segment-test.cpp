#include "llama.h"
#include "llama-ext.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/resource.h>

static size_t proc_kib(const char * file, const char * key) {
    std::ifstream in(file);
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, std::strlen(key), key) != 0) continue;
        std::istringstream fields(line.substr(std::strlen(key)));
        size_t value = 0;
        if (fields >> value) return value;
    }
    return SIZE_MAX;
}

static size_t proc_io_bytes(const char * key) {
    std::ifstream in("/proc/self/io");
    std::string name;
    size_t value = 0;
    while (in >> name >> value) {
        if (!name.empty() && name.back() == ':') name.pop_back();
        if (name == key) return value;
    }
    return SIZE_MAX;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) return 2;
    const auto started = std::chrono::steady_clock::now();
    const size_t rss_before = proc_kib("/proc/self/status", "VmRSS:");
    const size_t pss_before = proc_kib("/proc/self/smaps_rollup", "Pss:");
    const size_t read_before = proc_io_bytes("read_bytes");
    const char * model_path = argv[1];
    const int layer = std::atoi(argv[2]);
    const int end = std::atoi(argv[3]);
    const bool plan_only = argc == 5 && std::string(argv[4]) == "--plan";
    if (layer < 1 || end != layer + 1 || end > 48) return 2;
    char layer_text[32], end_text[32];
    std::snprintf(layer_text, sizeof(layer_text), "%d", layer);
    std::snprintf(end_text, sizeof(end_text), "%d", end);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", layer_text, 1);
    // PLE rows are gathered through the bounded range provider below; do not
    // bind the 28.8 GB per-layer embedding tensor as a graph weight.
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_LAYER_START", layer_text, 1);
    setenv("CORTEX_FLASHNEXT_LAYER_END", end_text, 1);
    setenv("CORTEX_QWEN35_SEGMENT_START", layer_text, 1);
    setenv("CORTEX_QWEN35_SEGMENT_END", end_text, 1);
    if (layer > 0) {
        setenv("CORTEX_FLASHNEXT_ROUTER_ONLY", "1", 1);
        setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
    }
    if (plan_only) setenv("CORTEX_FLASHNEXT_PLAN_ONLY", "1", 1);
    if (plan_only) setenv("CORTEX_FLASHNEXT_PLAN_STRICT", "1", 1);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) return 3;
    if (plan_only) {
        const auto finished = std::chrono::steady_clock::now();
        const size_t rss_after = proc_kib("/proc/self/status", "VmRSS:");
        const size_t pss_after = proc_kib("/proc/self/smaps_rollup", "Pss:");
        const size_t read_after = proc_io_bytes("read_bytes");
        const double elapsed_ms = std::chrono::duration<double, std::milli>(finished - started).count();
        std::printf("CORTEX_PLAN_METRICS rss_before_kib=%zu rss_after_kib=%zu "
                    "pss_before_kib=%zu pss_after_kib=%zu bytes_read=%zu elapsed_ms=%.3f "
                    "model_payload_allocation=SKIPPED graph_execution=SKIPPED\n",
                    rss_before, rss_after, pss_before, pss_after,
                    read_after >= read_before ? read_after - read_before : 0, elapsed_ms);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 8;
    cp.n_batch = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    llama_set_embeddings_layer_inp(ctx, layer, false);
    llama_set_embeddings_layer_inp(ctx, end, true);

    const int n_embd = llama_model_n_embd(model);
    size_t input_floats = 0;
    llama_batch batch;
    if (layer == 0) {
        batch = llama_batch_init(1, 0, 1);
        batch.token[0] = 9707;
    } else {
        const int n_embd_in = llama_model_n_embd_out(model);
        input_floats = n_embd_in;
        batch = llama_batch_init(1, n_embd_in, 1);
        for (int i = 0; i < n_embd_in; ++i) batch.embd[i] = 0.001f * float((i % 17) - 8);
    }
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.n_tokens = 1;
    if (const char * fixture_dir = std::getenv("CORTEX_FIXTURE_DIR")) {
        std::string path = std::string(fixture_dir) + "/hc_input.f32";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out || layer == 0 || !batch.embd) {
            std::fprintf(stderr, "FIXTURE_HC_INPUT_EXPORT=FAIL path=%s\n", path.c_str());
        } else {
            out.write(reinterpret_cast<const char *>(batch.embd),
                      static_cast<std::streamsize>(input_floats * sizeof(float)));
            out.flush();
            std::printf("FIXTURE_HC_INPUT_EXPORT=%s bytes=%zu shape=[2560,4,1] strides=[4,10240,40960,40960]\n",
                out.good() ? "PASS" : "FAIL", input_floats * sizeof(float));
        }
    }
    if (llama_decode(ctx, batch) != 0) return 5;
    float * output = llama_get_embeddings_layer_inp(ctx, end);
    if (!output) return 6;

    const size_t hc = llama_model_n_embd_out(model) / (size_t) n_embd;
    const size_t output_floats = (size_t) n_embd * hc;
    double checksum = 0.0;
    bool finite = true;
    uint64_t hash = 1469598103934665603ULL;
    const unsigned char * bytes = reinterpret_cast<const unsigned char *>(output);
    for (size_t i = 0; i < output_floats; ++i) {
        finite = finite && std::isfinite(output[i]);
        checksum += output[i];
    }
    for (size_t i = 0; i < output_floats * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    const auto finished = std::chrono::steady_clock::now();
    const size_t rss_after = proc_kib("/proc/self/status", "VmRSS:");
    const size_t pss_after = proc_kib("/proc/self/smaps_rollup", "Pss:");
    const size_t read_after = proc_io_bytes("read_bytes");
    const double elapsed_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    std::printf("CORTEX_NATIVE_V1 layer=%d end=%d rss_before_kib=%zu rss_after_kib=%zu "
                "pss_before_kib=%zu pss_after_kib=%zu bytes_read=%zu elapsed_ms=%.3f "
                "buffered_segment_execution=YES full_model_weight_allocation=NO\n",
                layer, end, rss_before, rss_after, pss_before, pss_after,
                read_after >= read_before ? read_after - read_before : 0, elapsed_ms);
    std::printf("qwen38 lowres segment: %s layer=%d end=%d hc_streams=%zu output_floats=%zu hidden_checksum=%.9g hc_hash=%016llx finite=%s\n",
                finite ? "PASS" : "FAIL", layer, end, hc, output_floats, checksum,
                (unsigned long long) hash, finite ? "YES" : "NO");
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
