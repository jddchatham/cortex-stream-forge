#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct flashnext_layer_driver flashnext_layer_driver;

// Creates a driver for one bounded layer range. The model and tensor index
// remain owned by the driver; returned output is caller-owned.
flashnext_layer_driver * flashnext_layer_driver_open(const char * model_path,
        const char * tensor_index_path, uint32_t layer_start, uint32_t layer_end);
// Executes every layer in [layer_start, layer_end), handing each layer's
// 10,240-element output to the next layer without reinitializing the handle.
int flashnext_layer_driver_run(flashnext_layer_driver * driver, int32_t position,
        const float * input, size_t input_count, float * output, size_t output_capacity);
size_t flashnext_layer_driver_state_size(const flashnext_layer_driver * driver);
int flashnext_layer_driver_copy_state(const flashnext_layer_driver * driver, void * dst, size_t capacity);
void flashnext_layer_driver_close(flashnext_layer_driver * driver);

#ifdef __cplusplus
}
#endif
