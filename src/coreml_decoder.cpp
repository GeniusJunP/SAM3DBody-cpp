#include "coreml_decoder.h"
#include <cstdio>

namespace fsb {

struct CoreMLDecoder::Impl {
    bool loaded = false;
};

CoreMLDecoder::CoreMLDecoder() : impl_(new Impl()) {}
CoreMLDecoder::~CoreMLDecoder() = default;

bool CoreMLDecoder::load(const std::string& mlpackage_path, ComputeUnit compute_units)
{
    (void)mlpackage_path; (void)compute_units;
    std::fprintf(stderr, "[FSB] CoreML Decoder is only available on macOS builds.\n");
    return false;
}

bool CoreMLDecoder::run(int batch,
                        const float* features,
                        const float* condition_info,
                        const float* ray_cond,
                        float* pose_token_out,
                        int pose_token_dim,
                        std::shared_ptr<void> opaque_in)
{
    (void)batch; (void)features; (void)condition_info; (void)ray_cond;
    (void)pose_token_out; (void)pose_token_dim; (void)opaque_in;
    return false;
}

void CoreMLDecoder::free() { impl_->loaded = false; }
bool CoreMLDecoder::loaded() const { return impl_->loaded; }

} // namespace fsb
