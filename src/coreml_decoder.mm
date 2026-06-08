#include "coreml_decoder.h"

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <iostream>
#include <vector>
#include "coreml_utils.h"

#if !__has_feature(objc_arc)
#error "This file must be compiled with ARC."
#endif

#ifndef FSB_BACKBONE_SIZE
#define FSB_BACKBONE_SIZE 512
#endif

namespace fsb {

struct CoreMLDecoder::Impl {
    MLModel* model = nil;
    NSURL* compiled_url = nil;
    MLMultiArray* cached_features = nil;
    MLMultiArray* cached_cond = nil;
    MLMultiArray* cached_ray = nil;
    MLDictionaryFeatureProvider* cached_provider = nil;
};

CoreMLDecoder::CoreMLDecoder() : impl_(new Impl()) {}

CoreMLDecoder::~CoreMLDecoder() {
    free();
}

bool CoreMLDecoder::load(const std::string& mlpackage_path, ComputeUnit compute_units) {
    @autoreleasepool {
        free();
        
        NSString* path = [NSString stringWithUTF8String:mlpackage_path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];
        if (!url) {
            std::cerr << "[CoreML Decoder] Invalid path." << std::endl;
            return false;
        }

        NSError* error = nil;
        NSURL* compiledUrl = url;
        if (![path hasSuffix:@".mlmodelc"]) {
            compiledUrl = [MLModel compileModelAtURL:url error:&error];
            if (error) {
                std::cerr << "[CoreML Decoder] Compile error: " << error.localizedDescription.UTF8String << std::endl;
                return false;
            }
            impl_->compiled_url = compiledUrl;
        }

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = (MLComputeUnits)compute_units;

        MLModel* model = [MLModel modelWithContentsOfURL:compiledUrl configuration:config error:&error];
        if (error || !model) {
            std::cerr << "[CoreML Decoder] Load error: " << (error ? error.localizedDescription.UTF8String : "Unknown") << std::endl;
            return false;
        }

        impl_->model = model;

        int feat_hw = FSB_BACKBONE_SIZE / 16;
        impl_->cached_features = [[MLMultiArray alloc] initWithShape:@[@1, @1280, @(feat_hw), @(feat_hw)]
                                                          dataType:MLMultiArrayDataTypeFloat32
                                                             error:&error];

        impl_->cached_cond = [[MLMultiArray alloc] initWithShape:@[@1, @3]
                                                      dataType:MLMultiArrayDataTypeFloat32
                                                         error:&error];

        impl_->cached_ray = [[MLMultiArray alloc] initWithShape:@[@1, @2, @(feat_hw), @(feat_hw)]
                                                     dataType:MLMultiArrayDataTypeFloat32
                                                        error:&error];

        impl_->cached_provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
            @"features": impl_->cached_features,
            @"condition_info": impl_->cached_cond,
            @"ray_cond": impl_->cached_ray
        } error:&error];

        return true;
    }
}

void CoreMLDecoder::free() {
    @autoreleasepool {
        if (impl_->compiled_url) {
            [[NSFileManager defaultManager] removeItemAtURL:impl_->compiled_url error:nil];
            impl_->compiled_url = nil;
        }
        impl_->model = nil;
        impl_->cached_features = nil;
        impl_->cached_cond = nil;
        impl_->cached_ray = nil;
        impl_->cached_provider = nil;
    }
}

bool CoreMLDecoder::loaded() const {
    return impl_->model != nil;
}

bool CoreMLDecoder::run(int batch,
                        const float* features,
                        const float* condition_info,
                        const float* ray_cond,
                        float* pose_token_out,
                        int pose_token_dim,
                        std::shared_ptr<void> opaque_in) {
    @autoreleasepool {
        if (!impl_->model || batch <= 0 || !condition_info || !ray_cond || !pose_token_out || pose_token_dim <= 0) {
            return false;
        }
        if (!opaque_in && !features) {
            return false;
        }

        NSError* error = nil;
        int feat_hw = FSB_BACKBONE_SIZE / 16;

        if (!impl_->cached_cond || [impl_->cached_cond.shape[0] intValue] != batch) {
            impl_->cached_cond = [[MLMultiArray alloc] initWithShape:@[@(batch), @3]
                                                          dataType:MLMultiArrayDataTypeFloat32
                                                             error:&error];
            impl_->cached_ray = [[MLMultiArray alloc] initWithShape:@[@(batch), @2, @(feat_hw), @(feat_hw)]
                                                         dataType:MLMultiArrayDataTypeFloat32
                                                            error:&error];
            impl_->cached_features = [[MLMultiArray alloc] initWithShape:@[@(batch), @1280, @(feat_hw), @(feat_hw)]
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:&error];
            impl_->cached_provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"features": impl_->cached_features,
                @"condition_info": impl_->cached_cond,
                @"ray_cond": impl_->cached_ray
            } error:&error];
        }

        std::memcpy((float*)impl_->cached_cond.dataPointer, condition_info, batch * 3 * sizeof(float));
        std::memcpy((float*)impl_->cached_ray.dataPointer, ray_cond, batch * 2 * feat_hw * feat_hw * sizeof(float));

        id<MLFeatureProvider> provider = impl_->cached_provider;
        if (opaque_in) {
            MLMultiArray* directFeatures = (__bridge MLMultiArray*)opaque_in.get();
            provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"features": directFeatures,
                @"condition_info": impl_->cached_cond,
                @"ray_cond": impl_->cached_ray
            } error:&error];
        } else {
            std::memcpy((float*)impl_->cached_features.dataPointer, features, batch * 1280 * feat_hw * feat_hw * sizeof(float));
        }

        id<MLFeatureProvider> outProvider = [impl_->model predictionFromFeatures:provider error:&error];
        if (error || !outProvider) {
            std::cerr << "[CoreML Decoder] Prediction error: " << (error ? error.localizedDescription.UTF8String : "Unknown") << std::endl;
            return false;
        }

        MLFeatureValue* outVal = [outProvider featureValueForName:@"pose_token"];
        if (!outVal) {
            std::cerr << "[CoreML Decoder] Output feature 'pose_token' not found." << std::endl;
            return false;
        }

        MLMultiArray* mlOutput = outVal.multiArrayValue;
        if (!mlOutput) {
            std::cerr << "[CoreML Decoder] Output is not a multi-array." << std::endl;
            return false;
        }

        NSArray<NSNumber*>* shape = mlOutput.shape;
        NSArray<NSNumber*>* strides = mlOutput.strides;
        int S0 = shape.count > 0 ? shape[0].intValue : 1;
        int S1 = shape.count > 1 ? shape[1].intValue : 1;
        int St0 = strides.count > 0 ? strides[0].intValue : 0;
        int St1 = strides.count > 1 ? strides[1].intValue : 0;

        bool is_contiguous = (St1 == 1 && St0 == S1);
        int output_elems = S0 * S1;
        if (S0 != batch || S1 != pose_token_dim) {
            std::cerr << "[CoreML Decoder] Unexpected output shape: ["
                      << S0 << ", " << S1 << "], expected ["
                      << batch << ", " << pose_token_dim << "]." << std::endl;
            return false;
        }

        if (is_contiguous) {
            if (mlOutput.dataType == MLMultiArrayDataTypeFloat32) {
                std::memcpy(pose_token_out, mlOutput.dataPointer, output_elems * sizeof(float));
            } else if (mlOutput.dataType == MLMultiArrayDataTypeFloat16) {
                const uint16_t* src = static_cast<const uint16_t*>(mlOutput.dataPointer);
                for (size_t i = 0; i < output_elems; ++i) {
                    pose_token_out[i] = fsb_half_to_float(src[i]);
                }
            } else {
                std::cerr << "[CoreML Decoder] Unsupported output data type." << std::endl;
                return false;
            }
        } else {
            if (mlOutput.dataType == MLMultiArrayDataTypeFloat32) {
                float* outPtr = (float*)mlOutput.dataPointer;
                for (int i = 0; i < S0; ++i) {
                    for (int j = 0; j < S1; ++j) {
                        int linear_idx = i * St0 + j * St1;
                        int out_idx = i * S1 + j;
                        pose_token_out[out_idx] = outPtr[linear_idx];
                    }
                }
            } else if (mlOutput.dataType == MLMultiArrayDataTypeFloat16) {
                uint16_t* outPtr = (uint16_t*)mlOutput.dataPointer;
                for (int i = 0; i < S0; ++i) {
                    for (int j = 0; j < S1; ++j) {
                        int linear_idx = i * St0 + j * St1;
                        int out_idx = i * S1 + j;
                        pose_token_out[out_idx] = fsb_half_to_float(outPtr[linear_idx]);
                    }
                }
            } else {
                std::cerr << "[CoreML Decoder] Unsupported output data type." << std::endl;
                return false;
            }
        }

        return true;
    }
}

} // namespace fsb
