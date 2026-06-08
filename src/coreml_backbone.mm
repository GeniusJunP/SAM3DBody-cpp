#include "coreml_backbone.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef FSB_BACKBONE_SIZE
#define FSB_BACKBONE_SIZE 512
#endif
#include "coreml_utils.h"

@interface FSBBackboneInput : NSObject <MLFeatureProvider>
- (instancetype)initWithArray:(MLMultiArray*)array;
@end

@implementation FSBBackboneInput {
    MLMultiArray* _array;
    NSSet<NSString*>* _names;
}

- (instancetype)initWithArray:(MLMultiArray*)array
{
    self = [super init];
    if (self) {
        _array = array;
        _names = [NSSet setWithObject:@"image"];
    }
    return self;
}

- (NSSet<NSString*>*)featureNames
{
    return _names;
}

- (MLFeatureValue*)featureValueForName:(NSString*)featureName
{
    if ([featureName isEqualToString:@"image"]) {
        return [MLFeatureValue featureValueWithMultiArray:_array];
    }
    return nil;
}

@end

namespace fsb {

struct CoreMLBackbone::Impl {
    MLModel* model = nil;
    NSURL* compiled_url = nil;
    MLMultiArray* cached_input = nil;
    FSBBackboneInput* cached_provider = nil;
};

CoreMLBackbone::CoreMLBackbone() : impl_(new Impl()) {}

CoreMLBackbone::~CoreMLBackbone()
{
    free();
}

bool CoreMLBackbone::load(const std::string& mlpackage_path, ComputeUnit compute_units)
{
    @autoreleasepool {
        free();

        NSString* path = [NSString stringWithUTF8String:mlpackage_path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        // Note: The ViT backbone was previously hardcoded to CPU+GPU because it
        // exceeded ANE limits causing fallback loops. The caller can now specify it.
        config.computeUnits = (MLComputeUnits)compute_units;

        NSError* error = nil;
        NSURL* model_url = url;
        if ([[path pathExtension] isEqualToString:@"mlpackage"] ||
            [[path pathExtension] isEqualToString:@"mlmodel"]) {
            model_url = [MLModel compileModelAtURL:url error:&error];
            if (!model_url) {
                std::fprintf(stderr, "[FSB] CoreML backbone compile failed: %s\n",
                             error ? [[error localizedDescription] UTF8String] : "unknown error");
                return false;
            }
            impl_->compiled_url = model_url;
        }

        MLModel* model = [MLModel modelWithContentsOfURL:model_url configuration:config error:&error];
        impl_->model = model;
        if (!impl_->model) {
            std::fprintf(stderr, "[FSB] CoreML backbone load failed: %s\n",
                         error ? [[error localizedDescription] UTF8String] : "unknown error");
            return false;
        }

        return true;
    }
}

bool CoreMLBackbone::run(const float* input_nchw, int batch, float* output_nchw,
                         std::shared_ptr<void>* retained_features_out)
{
    if (!impl_->model || !input_nchw || batch <= 0) {
        return false;
    }
    if (!retained_features_out && !output_nchw) {
        return false;
    }
    if (retained_features_out) {
        *retained_features_out = nullptr;
    }

    constexpr size_t input_elems = 3u * FSB_BACKBONE_SIZE * FSB_BACKBONE_SIZE;
    constexpr size_t grid = FSB_BACKBONE_SIZE / 16u;
    constexpr size_t output_elems = 1280u * grid * grid;

    @autoreleasepool {
        __block bool all_success = true;
        __block MLMultiArray* batched_features = nil;
        __block NSError* alloc_error = nil;

        // Lambda helper for strides-aware copying
        auto copy_features = [grid, output_elems](MLMultiArray* features, float* dst_slice, uint8_t* dst_array_slice) {
            NSArray<NSNumber*>* shape = features.shape;
            NSArray<NSNumber*>* strides = features.strides;
            int S1 = shape.count > 1 ? shape[1].intValue : 1;
            int S2 = shape.count > 2 ? shape[2].intValue : 1;
            int S3 = shape.count > 3 ? shape[3].intValue : 1;
            int St1 = strides.count > 1 ? strides[1].intValue : 0;
            int St2 = strides.count > 2 ? strides[2].intValue : 0;
            int St3 = strides.count > 3 ? strides[3].intValue : 0;

            bool is_contiguous = (St3 == 1 && St2 == S3 && St1 == S2 * S3);

            std::fprintf(stderr, "[DEBUG] shape: [1, %d, %d, %d] strides: [%d, %d, %d, %d] contiguous: %d dataType: %ld\n",
                         S1, S2, S3,
                         (strides.count > 0 ? strides[0].intValue : 0), St1, St2, St3,
                         is_contiguous ? 1 : 0, (long)features.dataType);

            if (dst_slice) {
                if (is_contiguous) {
                    if (features.dataType == MLMultiArrayDataTypeFloat32) {
                        std::memcpy(dst_slice, features.dataPointer, output_elems * sizeof(float));
                    } else if (features.dataType == MLMultiArrayDataTypeFloat16) {
                        const uint16_t* src = static_cast<const uint16_t*>(features.dataPointer);
                        for (size_t k = 0; k < output_elems; ++k) {
                            dst_slice[k] = fsb_half_to_float(src[k]);
                        }
                    }
                } else {
                    if (features.dataType == MLMultiArrayDataTypeFloat32) {
                        const float* src = static_cast<const float*>(features.dataPointer);
                        for (int i1 = 0; i1 < S1; ++i1) {
                            for (int i2 = 0; i2 < S2; ++i2) {
                                for (int i3 = 0; i3 < S3; ++i3) {
                                    int linear_idx = i1 * St1 + i2 * St2 + i3 * St3;
                                    int out_idx = i1 * (S2*S3) + i2 * S3 + i3;
                                    dst_slice[out_idx] = src[linear_idx];
                                }
                            }
                        }
                    } else if (features.dataType == MLMultiArrayDataTypeFloat16) {
                        const uint16_t* src = static_cast<const uint16_t*>(features.dataPointer);
                        for (int i1 = 0; i1 < S1; ++i1) {
                            for (int i2 = 0; i2 < S2; ++i2) {
                                for (int i3 = 0; i3 < S3; ++i3) {
                                    int linear_idx = i1 * St1 + i2 * St2 + i3 * St3;
                                    int out_idx = i1 * (S2*S3) + i2 * S3 + i3;
                                    dst_slice[out_idx] = fsb_half_to_float(src[linear_idx]);
                                }
                            }
                        }
                    }
                }
            }

            if (dst_array_slice) {
                if (is_contiguous) {
                    size_t bytes = output_elems * (features.dataType == MLMultiArrayDataTypeFloat32 ? sizeof(float) : sizeof(uint16_t));
                    std::memcpy(dst_array_slice, features.dataPointer, bytes);
                } else {
                    if (features.dataType == MLMultiArrayDataTypeFloat32) {
                        const float* src = static_cast<const float*>(features.dataPointer);
                        float* dst = (float*)dst_array_slice;
                        for (int i1 = 0; i1 < S1; ++i1) {
                            for (int i2 = 0; i2 < S2; ++i2) {
                                for (int i3 = 0; i3 < S3; ++i3) {
                                    int linear_idx = i1 * St1 + i2 * St2 + i3 * St3;
                                    int out_idx = i1 * (S2*S3) + i2 * S3 + i3;
                                    dst[out_idx] = src[linear_idx];
                                }
                            }
                        }
                    } else if (features.dataType == MLMultiArrayDataTypeFloat16) {
                        const uint16_t* src = static_cast<const uint16_t*>(features.dataPointer);
                        uint16_t* dst = (uint16_t*)dst_array_slice;
                        for (int i1 = 0; i1 < S1; ++i1) {
                            for (int i2 = 0; i2 < S2; ++i2) {
                                for (int i3 = 0; i3 < S3; ++i3) {
                                    int linear_idx = i1 * St1 + i2 * St2 + i3 * St3;
                                    int out_idx = i1 * (S2*S3) + i2 * S3 + i3;
                                    dst[out_idx] = src[linear_idx];
                                }
                            }
                        }
                    }
                }
            }
        };

        // Create an array of MLFeatureProviders for batch execution
        NSMutableArray<id<MLFeatureProvider>>* providers = [NSMutableArray arrayWithCapacity:batch];
        for (int i = 0; i < batch; ++i) {
            NSError* error = nil;
            MLMultiArray* in_arr = [[MLMultiArray alloc]
                initWithShape:@[@1, @3, @(FSB_BACKBONE_SIZE), @(FSB_BACKBONE_SIZE)]
                     dataType:MLMultiArrayDataTypeFloat32
                        error:&error];
            if (!in_arr) {
                std::fprintf(stderr, "[FSB] Input MLMultiArray alloc failed: %s\n",
                             error ? [[error localizedDescription] UTF8String] : "unknown");
                return false;
            }
            std::memcpy(in_arr.dataPointer, input_nchw + i * input_elems, input_elems * sizeof(float));
            [providers addObject:[[FSBBackboneInput alloc] initWithArray:in_arr]];
        }

        MLArrayBatchProvider* batchProvider = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:providers];
        
        NSError* error = nil;
        id<MLBatchProvider> resultBatch = [impl_->model predictionsFromBatch:batchProvider error:&error];
        if (!resultBatch) {
            std::fprintf(stderr, "[FSB] CoreML batch prediction failed: %s\n",
                         error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        for (int i = 0; i < batch; ++i) {
            id<MLFeatureProvider> result = [resultBatch featuresAtIndex:i];
            MLFeatureValue* val = [result featureValueForName:@"features"];
            MLMultiArray* features = val.multiArrayValue;
            if (!features) {
                std::fprintf(stderr, "[FSB] CoreML batch prediction missing features for index %d\n", i);
                return false;
            }

            if (retained_features_out) {
                if (!batched_features) {
                    batched_features = [[MLMultiArray alloc]
                        initWithShape:@[@(batch), @1280, @(grid), @(grid)]
                             dataType:features.dataType
                                error:&alloc_error];
                }
                if (batched_features) {
                    size_t slice_bytes = output_elems * (features.dataType == MLMultiArrayDataTypeFloat32 ? sizeof(float) : sizeof(uint16_t));
                    copy_features(features, nullptr, (uint8_t*)batched_features.dataPointer + i * slice_bytes);
                }
            }

            if (output_nchw) {
                copy_features(features, output_nchw + i * output_elems, nullptr);
            }
        }

        if (retained_features_out && batched_features && all_success) {
            *retained_features_out = fsb::make_shared_objc(batched_features);
        }

        return all_success;
    }
}

void CoreMLBackbone::free()
{
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

bool CoreMLBackbone::loaded() const
{
    return impl_->model != nil;
}

} // namespace fsb
