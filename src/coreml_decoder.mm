#include "coreml_decoder.h"

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <iostream>
#include <vector>
#include "coreml_utils.h"
#include <Accelerate/Accelerate.h>

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
        int expected_batch = 1;

        if (!impl_->cached_cond) {
            impl_->cached_cond = [[MLMultiArray alloc] initWithShape:@[@(expected_batch), @3]
                                                          dataType:MLMultiArrayDataTypeFloat32
                                                             error:&error];
            impl_->cached_ray = [[MLMultiArray alloc] initWithShape:@[@(expected_batch), @2, @(feat_hw), @(feat_hw)]
                                                         dataType:MLMultiArrayDataTypeFloat32
                                                            error:&error];
            impl_->cached_features = [[MLMultiArray alloc] initWithShape:@[@(expected_batch), @1280, @(feat_hw), @(feat_hw)]
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                                 error:&error];
            impl_->cached_provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"features": impl_->cached_features,
                @"condition_info": impl_->cached_cond,
                @"ray_cond": impl_->cached_ray
            } error:&error];
        }

        NSMutableArray<id<MLFeatureProvider>>* providers = [NSMutableArray arrayWithCapacity:batch];

        NSArray* cond_shape = @[@(expected_batch), @3];
        NSArray* cond_strides = @[@3, @1];
        NSArray* ray_shape = @[@(expected_batch), @2, @(feat_hw), @(feat_hw)];
        NSArray* ray_strides = @[@(2 * feat_hw * feat_hw), @(feat_hw * feat_hw), @(feat_hw), @1];
        NSArray* feat_shape = @[@(expected_batch), @1280, @(feat_hw), @(feat_hw)];
        NSArray* feat_strides = @[@(1280 * feat_hw * feat_hw), @(feat_hw * feat_hw), @(feat_hw), @1];
        
        for (int b = 0; b < batch; ++b) {
            MLMultiArray* b_cond = [[MLMultiArray alloc] initWithDataPointer:(void*)(condition_info + b * 3)
                                                                       shape:cond_shape
                                                                    dataType:MLMultiArrayDataTypeFloat32
                                                                     strides:cond_strides
                                                                 deallocator:nil
                                                                       error:&error];
            MLMultiArray* b_ray = [[MLMultiArray alloc] initWithDataPointer:(void*)(ray_cond + b * 2 * feat_hw * feat_hw)
                                                                      shape:ray_shape
                                                                   dataType:MLMultiArrayDataTypeFloat32
                                                                    strides:ray_strides
                                                                deallocator:nil
                                                                      error:&error];
            
            id<MLFeatureProvider> provider = nil;
            MLMultiArray* b_feat = nil;
            if (opaque_in) {
                // opaque_in points to the full B-sized backbone feature multiarray.
                // We extract/slice the batch element for this step.
                MLMultiArray* batched_features = (__bridge MLMultiArray*)opaque_in.get();
                size_t slice_bytes = 1280 * feat_hw * feat_hw * (batched_features.dataType == MLMultiArrayDataTypeFloat32 ? sizeof(float) : sizeof(uint16_t));
                if (batched_features.dataType == MLMultiArrayDataTypeFloat32) {
                    b_feat = [[MLMultiArray alloc] initWithDataPointer:(uint8_t*)batched_features.dataPointer + b * slice_bytes
                                                                 shape:feat_shape
                                                              dataType:MLMultiArrayDataTypeFloat32
                                                               strides:feat_strides
                                                           deallocator:nil
                                                                 error:&error];
                } else {
                    b_feat = [[MLMultiArray alloc] initWithShape:feat_shape dataType:MLMultiArrayDataTypeFloat32 error:&error];
                    const uint16_t* src = (const uint16_t*)((const uint8_t*)batched_features.dataPointer + b * slice_bytes);
                    float* dst = (float*)b_feat.dataPointer;
                    size_t count = 1280 * feat_hw * feat_hw;
                    vImage_Buffer src_buf = { (void*)src, 1, count, count * sizeof(uint16_t) };
                    vImage_Buffer dst_buf = { (void*)dst, 1, count, count * sizeof(float) };
                    vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, 0);
                }
            } else {
                b_feat = [[MLMultiArray alloc] initWithDataPointer:(void*)(features + b * 1280 * feat_hw * feat_hw)
                                                             shape:feat_shape
                                                          dataType:MLMultiArrayDataTypeFloat32
                                                           strides:feat_strides
                                                       deallocator:nil
                                                             error:&error];
            }
            provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"features": b_feat,
                @"condition_info": b_cond,
                @"ray_cond": b_ray
            } error:&error];
            [providers addObject:provider];
        }

        MLArrayBatchProvider* batchProvider = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:providers];
        id<MLBatchProvider> outBatchProvider = [impl_->model predictionsFromBatch:batchProvider error:&error];

        if (error || !outBatchProvider) {
            std::cerr << "[CoreML Decoder] Prediction error: " << (error ? error.localizedDescription.UTF8String : "Unknown") << std::endl;
            return false;
        }

        for (int b = 0; b < batch; ++b) {
            id<MLFeatureProvider> outProvider = [outBatchProvider featuresAtIndex:b];
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
            if (S0 != expected_batch || S1 != pose_token_dim) {
                std::cerr << "[CoreML Decoder] Unexpected output shape: ["
                          << S0 << ", " << S1 << "], expected ["
                          << expected_batch << ", " << pose_token_dim << "]." << std::endl;
                return false;
            }

            float* out_ptr_for_b = pose_token_out + (b * pose_token_dim);

            if (is_contiguous) {
                if (mlOutput.dataType == MLMultiArrayDataTypeFloat32) {
                    std::memcpy(out_ptr_for_b, mlOutput.dataPointer, output_elems * sizeof(float));
                } else if (mlOutput.dataType == MLMultiArrayDataTypeFloat16) {
                    const uint16_t* src = static_cast<const uint16_t*>(mlOutput.dataPointer);
                    vImage_Buffer src_buf = { (void*)src, 1, (vImagePixelCount)output_elems, output_elems * sizeof(uint16_t) };
                    vImage_Buffer dst_buf = { (void*)out_ptr_for_b, 1, (vImagePixelCount)output_elems, output_elems * sizeof(float) };
                    vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, 0);
                } else {
                    std::cerr << "[CoreML Decoder] Unsupported output data type." << std::endl;
                    return false;
                }
            } else {
                for (int i = 0; i < S0; ++i) {
                    for (int j = 0; j < S1; ++j) {
                        int linear_idx = i * St0 + j * St1;
                        int out_idx = i * S1 + j;
                        if (mlOutput.dataType == MLMultiArrayDataTypeFloat32) {
                            out_ptr_for_b[out_idx] = ((float*)mlOutput.dataPointer)[linear_idx];
                        } else if (mlOutput.dataType == MLMultiArrayDataTypeFloat16) {
                            out_ptr_for_b[out_idx] = fsb_half_to_float(((uint16_t*)mlOutput.dataPointer)[linear_idx]);
                        } else {
                            std::cerr << "[CoreML Decoder] Unsupported output data type." << std::endl;
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }
}

} // namespace fsb
