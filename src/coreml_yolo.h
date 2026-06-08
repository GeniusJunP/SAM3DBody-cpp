#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context pointer
typedef void* CoreMLYoloContext;

// Initialize the CoreML model
CoreMLYoloContext init_coreml_yolo(const char* mlpackage_path);

// Free the context
void free_coreml_yolo(CoreMLYoloContext ctx);

// Run inference
// input_bchw: float array of size 1 * 3 * 640 * 640
// output: float array of size 1 * 56 * 8400
bool run_coreml_yolo(CoreMLYoloContext ctx, const float* input_bchw, float* output);

#ifdef __cplusplus
}
#endif
