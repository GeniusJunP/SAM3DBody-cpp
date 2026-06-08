#pragma once

#include <memory>
#include <string>

namespace fsb {

class CoreMLBackbone {
public:
    CoreMLBackbone();
    ~CoreMLBackbone();

    CoreMLBackbone(const CoreMLBackbone&) = delete;
    CoreMLBackbone& operator=(const CoreMLBackbone&) = delete;

    bool load(const std::string& mlpackage_path);

    // Runs a fixed-shape SAM3D backbone exported as [1,3,512,512] -> [1,1280,32,32].
    // Batch inputs are handled by invoking the CoreML model once per person.
    bool run(const float* input_nchw, int batch, float* output_nchw, void** opaque_out = nullptr);

    void free();
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fsb
