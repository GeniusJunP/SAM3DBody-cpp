#include "coreml_yolo.h"
#include <cstdio>

extern "C" {

CoreMLYoloContext init_coreml_yolo(const char* mlpackage_path) {
    (void)mlpackage_path;
    std::fprintf(stderr, "[FSB] CoreML YOLO is only available on macOS builds.\n");
    return nullptr;
}

bool run_coreml_yolo(CoreMLYoloContext ctx, const float* input_nchw, float* output_nchw) {
    (void)ctx; (void)input_nchw; (void)output_nchw;
    return false;
}

void free_coreml_yolo(CoreMLYoloContext ctx) {
    (void)ctx;
}

}
