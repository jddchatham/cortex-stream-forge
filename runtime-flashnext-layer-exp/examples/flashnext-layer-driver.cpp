#define CORTEX_LAYER_DRIVER_LIBRARY
#include "flashnext-owned-layer.cpp"
#include "flashnext-layer-driver.h"

struct flashnext_layer_driver {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    std::string index_path;
    uint32_t layer_start = 0;
    uint32_t layer_end = 0;
};

extern "C" flashnext_layer_driver * flashnext_layer_driver_open(
        const char * model_path, const char * tensor_index_path,
        uint32_t layer_start, uint32_t layer_end) {
    if (!model_path || !tensor_index_path || layer_start >= layer_end || layer_end > 48) return nullptr;
    llama_backend_init();
    setenv("CORTEX_PREAD_ARENA", "1", 1);
    setenv("CORTEX_LOW_RESIDENCY_LAYER", std::to_string(layer_start).c_str(), 1);
    setenv("CORTEX_FLASHNEXT_LAYER_START", std::to_string(layer_start).c_str(), 1);
    setenv("CORTEX_FLASHNEXT_LAYER_END", std::to_string(layer_end).c_str(), 1);
    setenv("CORTEX_FLASHNEXT_LAYER_RANGE", "1", 1);
    setenv("CORTEX_FLASHNEXT_PRE_MOE", "1", 1);
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { llama_backend_free(); return nullptr; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4; cp.n_batch = 1; cp.embeddings = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); llama_backend_free(); return nullptr; }
    auto * d = new flashnext_layer_driver;
    d->model = model; d->ctx = ctx; d->index_path = tensor_index_path;
    d->layer_start = layer_start; d->layer_end = layer_end;
    return d;
}

extern "C" int flashnext_layer_driver_run(flashnext_layer_driver * d, int32_t position,
        const float * input, size_t input_count, float * output, size_t output_capacity) {
    if (!d || !input || !output || input_count != 2560 || output_capacity < 10240) return -1;
    std::vector<float> owned(input, input + input_count);
    const auto result = run_layer_transaction(d->ctx, d->layer_start, position, owned, d->index_path);
    if (!result.state_committed || result.output.size() > output_capacity) return -2;
    std::copy(result.output.begin(), result.output.end(), output);
    return (int) result.output.size();
}

extern "C" size_t flashnext_layer_driver_state_size(const flashnext_layer_driver * d) {
    return d && d->ctx ? llama_state_seq_get_size(d->ctx, 0) : 0;
}

extern "C" int flashnext_layer_driver_copy_state(const flashnext_layer_driver * d, void * dst, size_t capacity) {
    if (!d || !d->ctx || !dst || capacity < flashnext_layer_driver_state_size(d)) return -1;
    return llama_state_seq_get_data(d->ctx, (uint8_t *) dst, capacity, 0) == capacity ? 0 : -2;
}

extern "C" void flashnext_layer_driver_close(flashnext_layer_driver * d) {
    if (!d) return;
    if (d->ctx) llama_free(d->ctx);
    if (d->model) llama_model_free(d->model);
    llama_backend_free();
    delete d;
}
