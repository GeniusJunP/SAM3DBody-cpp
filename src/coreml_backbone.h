#pragma once

#include <memory>
#include <string>
#include "coreml_utils.h"

namespace fsb {

class CoreMLBackbone {
public:
    CoreMLBackbone();
    ~CoreMLBackbone();

    CoreMLBackbone(const CoreMLBackbone&) = delete;
    CoreMLBackbone& operator=(const CoreMLBackbone&) = delete;

    bool load(const std::string& mlpackage_path, ComputeUnit compute_units = ComputeUnit::All);

    // Runs SAM3D backbone on [B,3,S,S].
    // If retained_features_out is set, it receives a retained CoreML batch output
    // for zero-copy handoff to the CoreML decoder and output_nchw is left untouched.
    bool run(const float* input_nchw, int batch, float* output_nchw,
             std::shared_ptr<void>* retained_features_out = nullptr);

    void free();
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fsb
