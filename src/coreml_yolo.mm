#include "coreml_yolo.h"

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "coreml_utils.h"
#include <iostream>
#include <vector>

#if !__has_feature(objc_arc)
#error "This file must be compiled with ARC."
#endif

namespace fsb {

struct CoreMLYolo::Impl {
    MLModel* model = nil;
    NSURL* compiled_url = nil;
    MLMultiArray* cached_input = nil;
    MLDictionaryFeatureProvider* cached_provider = nil;
};

CoreMLYolo::CoreMLYolo() : impl_(new Impl()) {}

CoreMLYolo::~CoreMLYolo() {
    free();
}

bool CoreMLYolo::load(const std::string& mlpackage_path, ComputeUnit compute_units) {
    @autoreleasepool {
        free();
        
        NSString* path = [NSString stringWithUTF8String:mlpackage_path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];
        if (!url) {
            std::cerr << "[CoreML YOLO] Invalid path." << std::endl;
            return false;
        }

        NSError* error = nil;
        NSURL* compiledUrl = url;
        if (![path hasSuffix:@".mlmodelc"]) {
            compiledUrl = [MLModel compileModelAtURL:url error:&error];
            if (error) {
                std::cerr << "[CoreML YOLO] Compile error: " << error.localizedDescription.UTF8String << std::endl;
                return false;
            }
            impl_->compiled_url = compiledUrl;
        }

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = (MLComputeUnits)compute_units;

        MLModel* model = [MLModel modelWithContentsOfURL:compiledUrl configuration:config error:&error];
        if (error || !model) {
            std::cerr << "[CoreML YOLO] Load error: " << (error ? error.localizedDescription.UTF8String : "Unknown") << std::endl;
            return false;
        }

        impl_->model = model;

        impl_->cached_input = [[MLMultiArray alloc] initWithShape:@[@1, @3, @640, @640] 
                                                       dataType:MLMultiArrayDataTypeFloat32 
                                                          error:&error];

        impl_->cached_provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"images": impl_->cached_input} error:&error];

        return true;
    }
}

void CoreMLYolo::free() {
    @autoreleasepool {
        if (impl_->compiled_url) {
            [[NSFileManager defaultManager] removeItemAtURL:impl_->compiled_url error:nil];
            impl_->compiled_url = nil;
        }
        impl_->model = nil;
        impl_->cached_input = nil;
        impl_->cached_provider = nil;
    }
}

bool CoreMLYolo::loaded() const {
    return impl_->model != nil;
}

bool CoreMLYolo::run(const float* input_bchw, float* output) {
    @autoreleasepool {
        if (!impl_->model) return false;

        NSError* error = nil;

        // Copy data to cached input
        float* inPtr = (float*)impl_->cached_input.dataPointer;
        std::memcpy(inPtr, input_bchw, 1 * 3 * 640 * 640 * sizeof(float));

        id<MLFeatureProvider> outProvider = [impl_->model predictionFromFeatures:impl_->cached_provider error:&error];
        if (error || !outProvider) {
            std::cerr << "[CoreML YOLO] Prediction error: " << (error ? error.localizedDescription.UTF8String : "Unknown") << std::endl;
            return false;
        }

        MLFeatureValue* outVal = [outProvider featureValueForName:@"output"];
        if (!outVal) {
            std::cerr << "[CoreML YOLO] Output feature 'output' not found." << std::endl;
            return false;
        }

        MLMultiArray* mlOutput = outVal.multiArrayValue;
        if (!mlOutput) {
            std::cerr << "[CoreML YOLO] Output is not a multi-array." << std::endl;
            return false;
        }

        NSArray<NSNumber*>* shape = mlOutput.shape;
        NSArray<NSNumber*>* strides = mlOutput.strides;
        int S0 = shape.count > 0 ? shape[0].intValue : 1;
        int S1 = shape.count > 1 ? shape[1].intValue : 1;
        int S2 = shape.count > 2 ? shape[2].intValue : 1;
        int St0 = strides.count > 0 ? strides[0].intValue : 0;
        int St1 = strides.count > 1 ? strides[1].intValue : 0;
        int St2 = strides.count > 2 ? strides[2].intValue : 0;

        if (mlOutput.dataType == MLMultiArrayDataTypeFloat32) {
            float* outPtr = (float*)mlOutput.dataPointer;
            for (int i = 0; i < S0; ++i) {
                for (int j = 0; j < S1; ++j) {
                    for (int k = 0; k < S2; ++k) {
                        int linear_idx = i * St0 + j * St1 + k * St2;
                        // Transpose from (1, 56, 8400) to (8400, 56) directly into the row_major output
                        int out_idx = k * S1 + j;
                        output[out_idx] = outPtr[linear_idx];
                    }
                }
            }
        } else if (mlOutput.dataType == MLMultiArrayDataTypeFloat16) {
            uint16_t* outPtr = (uint16_t*)mlOutput.dataPointer;
            for (int i = 0; i < S0; ++i) {
                for (int j = 0; j < S1; ++j) {
                    for (int k = 0; k < S2; ++k) {
                        int linear_idx = i * St0 + j * St1 + k * St2;
                        // Transpose from (1, 56, 8400) to (8400, 56) directly into the row_major output
                        // S0=1, S1=56, S2=8400. We want output to be indexed by k * S1 + j
                        int out_idx = k * S1 + j;
                        uint16_t h = outPtr[linear_idx];
                        
                        output[out_idx] = fsb_half_to_float(h);
                    }
                }
            }
        } else {
            std::cerr << "[CoreML YOLO] Unsupported output data type." << std::endl;
            return false;
        }

        return true;
    }
}

} // namespace fsb
