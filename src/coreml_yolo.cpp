#include "coreml_yolo.h"
#include <cstdio>

namespace fsb {

struct CoreMLYolo::Impl {
    bool loaded = false;
};

CoreMLYolo::CoreMLYolo() : impl_(new Impl()) {}
CoreMLYolo::~CoreMLYolo() = default;

bool CoreMLYolo::load(const std::string& mlpackage_path, ComputeUnit compute_units)
{
    (void)mlpackage_path; (void)compute_units;
    std::fprintf(stderr, "[FSB] CoreML YOLO is only available on macOS builds.\n");
    return false;
}

bool CoreMLYolo::run(const float* input_bchw, float* output)
{
    (void)input_bchw; (void)output;
    return false;
}

void CoreMLYolo::free() { impl_->loaded = false; }
bool CoreMLYolo::loaded() const { return impl_->loaded; }

} // namespace fsb
