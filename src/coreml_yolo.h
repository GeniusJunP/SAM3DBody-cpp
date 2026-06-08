#pragma once

#include <memory>
#include <string>
#include "coreml_utils.h"

namespace fsb {

class CoreMLYolo {
public:
    CoreMLYolo();
    ~CoreMLYolo();

    CoreMLYolo(const CoreMLYolo&) = delete;
    CoreMLYolo& operator=(const CoreMLYolo&) = delete;

    bool load(const std::string& mlpackage_path, ComputeUnit compute_units = ComputeUnit::All);

    // Run inference
    // input_bchw: float array of size 1 * 3 * 640 * 640
    // output: float array of size 1 * 56 * 8400
    bool run(const float* input_bchw, float* output);

    void free();
    bool loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fsb
