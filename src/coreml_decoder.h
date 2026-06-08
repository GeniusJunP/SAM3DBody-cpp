#pragma once

#include <memory>
#include <string>
#include "coreml_utils.h"

namespace fsb {

class CoreMLDecoder {
public:
    CoreMLDecoder();
    ~CoreMLDecoder();

    CoreMLDecoder(const CoreMLDecoder&) = delete;
    CoreMLDecoder& operator=(const CoreMLDecoder&) = delete;

    bool load(const std::string& mlpackage_path, ComputeUnit compute_units = ComputeUnit::All);

    // Run inference
    // features: float array [B, 1280, H, W]
    // condition_info: float array [B, 3]
    // ray_cond: float array [B, 2, H, W]
    // pose_token_out: float array [B, pose_token_dim] (output)
    bool run(int batch,
             const float* features,
             const float* condition_info,
             const float* ray_cond,
             float* pose_token_out,
             int pose_token_dim,
             std::shared_ptr<void> opaque_in = nullptr);

    void free();
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fsb
