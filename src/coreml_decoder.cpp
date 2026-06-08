#include "coreml_decoder.h"
#include <cstdio>

extern "C" {

CoreMLDecoderContext init_coreml_decoder(const char* mlpackage_path) {
    (void)mlpackage_path;
    std::fprintf(stderr, "[FSB] CoreML Decoder is only available on macOS builds.\n");
    return nullptr;
}

bool run_coreml_decoder(CoreMLDecoderContext ctx,
                        const float* features,
                        const float* cond_info,
                        const float* ray_cond,
                        float* pose_token,
                        void* opaque_in) {
    (void)ctx; (void)features; (void)cond_info; (void)ray_cond; (void)pose_token; (void)opaque_in;
    return false;
}

void free_coreml_decoder(CoreMLDecoderContext ctx) {
    (void)ctx;
}

}
