#include "coreml_backbone.h"
#include <cstdio>

namespace fsb {

struct CoreMLBackbone::Impl {
    bool loaded = false;
};

CoreMLBackbone::CoreMLBackbone() : impl_(new Impl()) {}
CoreMLBackbone::~CoreMLBackbone() = default;

bool CoreMLBackbone::load(const std::string& mlpackage_path, ComputeUnit compute_units)
{
    (void)mlpackage_path; (void)compute_units;
    std::fprintf(stderr, "[FSB] CoreML backbone is only available on macOS builds.\n");
    return false;
}

bool CoreMLBackbone::run(const float* input_nchw, int batch, float* output_nchw,
                         std::shared_ptr<void>* retained_features_out)
{
    (void)input_nchw; (void)batch; (void)output_nchw; (void)retained_features_out;
    return false;
}

void CoreMLBackbone::free() { impl_->loaded = false; }
bool CoreMLBackbone::loaded() const { return impl_->loaded; }

} // namespace fsb
