#include "coreml_backbone.h"

#include <cstdio>

namespace fsb {

struct CoreMLBackbone::Impl {
    bool loaded = false;
};

CoreMLBackbone::CoreMLBackbone() : impl_(new Impl()) {}
CoreMLBackbone::~CoreMLBackbone() = default;

bool CoreMLBackbone::load(const std::string& mlpackage_path)
{
    (void)mlpackage_path;
    std::fprintf(stderr, "[FSB] CoreML backbone is only available on macOS builds.\n");
    return false;
}

bool CoreMLBackbone::run(const float* input_nchw, int batch, float* output_nchw, void** opaque_out)
{
    (void)input_nchw;
    (void)batch;
    (void)output_nchw;
    return false;
}

void CoreMLBackbone::free()
{
    impl_->loaded = false;
}

bool CoreMLBackbone::loaded() const
{
    return impl_->loaded;
}

} // namespace fsb
