#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "coreml_decoder.h"
#include <iostream>
#include <vector>
#include "coreml_utils.h"

#if !__has_feature(objc_arc)
#error "This file must be compiled with ARC."
#endif

struct CoreMLDecoderCtx {
    MLModel* model;
    MLMultiArray* cached_features;
    MLMultiArray* cached_cond;
    MLMultiArray* cached_ray;
    MLDictionaryFeatureProvider* cached_provider;
};

extern "C" CoreMLDecoderContext init_coreml_decoder(const char* mlpackage_path) {
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:mlpackage_path];
        NSURL* url = [NSURL fileURLWithPath:path];
        if (!url) {
            std::cerr << "[CoreML Decoder] Invalid path." << std::endl;
            return nullptr;
        }

        NSError* error = nil;
        NSURL* compiledUrl = url;
        if (![path hasSuffix:@".mlmodelc"]) {
            compiledUrl = [MLModel compileModelAtURL:url error:&error];
            if (error) {
                std::cerr << "[CoreML Decoder] Compile error: " << error.localizedDescription.UTF8String << std::endl;
                return nullptr;
            }
        }

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndGPU;

        MLModel* model = [MLModel modelWithContentsOfURL:compiledUrl configuration:config error:&error];
        if (error || !model) {
            std::cerr << "[CoreML Decoder] Load error: " << (error ? error.localizedDescription.UTF8String : "Unknown") << std::endl;
            return nullptr;
        }

        CoreMLDecoderCtx* ctx = new CoreMLDecoderCtx();
        ctx->model = model;

        int feat_hw = FSB_BACKBONE_SIZE / 16;
        ctx->cached_features = [[MLMultiArray alloc] initWithShape:@[@1, @1280, @(feat_hw), @(feat_hw)]
                                                          dataType:MLMultiArrayDataTypeFloat32
                                                             error:&error];

        ctx->cached_cond = [[MLMultiArray alloc] initWithShape:@[@1, @3]
                                                      dataType:MLMultiArrayDataTypeFloat32
                                                         error:&error];

        ctx->cached_ray = [[MLMultiArray alloc] initWithShape:@[@1, @2, @(feat_hw), @(feat_hw)]
                                                     dataType:MLMultiArrayDataTypeFloat32
                                                        error:&error];

        ctx->cached_provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
            @"features": ctx->cached_features,
            @"condition_info": ctx->cached_cond,
            @"ray_cond": ctx->cached_ray
        } error:&error];

        return (CoreMLDecoderContext)ctx;
    }
}

extern "C" void free_coreml_decoder(CoreMLDecoderContext ctx_ptr) {
    @autoreleasepool {
        if (!ctx_ptr) return;
        CoreMLDecoderCtx* ctx = (CoreMLDecoderCtx*)ctx_ptr;
        ctx->model = nil;
        ctx->cached_features = nil;
        ctx->cached_cond = nil;
        ctx->cached_ray = nil;
        ctx->cached_provider = nil;
        delete ctx;
    }
}

extern "C" bool run_coreml_decoder(CoreMLDecoderContext ctx_ptr,
                                   const float* features,
                                   const float* condition_info,
                                   const float* ray_cond,
                                   float* pose_token_out
#ifdef __cplusplus
                                   , void* opaque_in
#endif
                                   ) {
    @autoreleasepool {
        if (!ctx_ptr) return false;
        CoreMLDecoderCtx* ctx = (CoreMLDecoderCtx*)ctx_ptr;

        NSError* error = nil;

        int feat_hw = FSB_BACKBONE_SIZE / 16;
        std::memcpy((float*)ctx->cached_cond.dataPointer, condition_info, 1 * 3 * sizeof(float));
        std::memcpy((float*)ctx->cached_ray.dataPointer, ray_cond, 1 * 2 * feat_hw * feat_hw * sizeof(float));

        id<MLFeatureProvider> provider = ctx->cached_provider;
#ifdef __cplusplus
        if (opaque_in) {
            MLMultiArray* directFeatures = (__bridge MLMultiArray*)opaque_in;

            // Extract the raw data pointer and wrap it in a clean MLMultiArray
            // to bypass the MPSGraph shape/stride assertion bug when chaining outputs.
            // We MUST preserve the exact original shape, dataType (e.g. Float16),
            // and strides to prevent memory corruption and type mismatches.
            MLMultiArray* wrappedFeatures = [[MLMultiArray alloc]
                initWithDataPointer:directFeatures.dataPointer
                              shape:directFeatures.shape
                           dataType:directFeatures.dataType
                            strides:directFeatures.strides
                        deallocator:nil
                              error:&error];

            provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"features": wrappedFeatures,
                @"condition_info": ctx->cached_cond,
                @"ray_cond": ctx->cached_ray
            } error:&error];
        } else
#endif
        {
            std::memcpy((float*)ctx->cached_features.dataPointer, features, 1 * 1280 * feat_hw * feat_hw * sizeof(float));
        }

        id<MLFeatureProvider> outProvider = [ctx->model predictionFromFeatures:provider error:&error];
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
