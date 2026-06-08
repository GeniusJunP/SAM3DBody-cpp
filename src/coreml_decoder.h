#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context pointer
typedef void* CoreMLDecoderContext;

// Initialize the CoreML model
CoreMLDecoderContext init_coreml_decoder(const char* mlpackage_path);

// Free the context
void free_coreml_decoder(CoreMLDecoderContext ctx);

// Run inference
// features: float array [1, 1280, 32, 32]
// condition_info: float array [1, 3]
// ray_cond: float array [1, 2, 32, 32]
// pose_token_out: float array [1, 1024] (output)
bool run_coreml_decoder(CoreMLDecoderContext ctx,
                        const float* features,
                        const float* condition_info,
                        const float* ray_cond,
                        float* pose_token_out
#ifdef __cplusplus
                        , void* opaque_in = nullptr
#endif
                        );

#ifdef __cplusplus
}
#endif
