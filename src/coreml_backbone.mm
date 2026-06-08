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

bool CoreMLBackbone::load(const std::string& mlpackage_path)
{
    @autoreleasepool {
        free();

        NSString* path = [NSString stringWithUTF8String:mlpackage_path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        // The ViT backbone is known to exceed ANE limits (e.g. 1024x1024 global attention),
        // causing fallback loops that severely degrade performance. Hardcoding CPU+GPU.
        config.computeUnits = MLComputeUnitsCPUAndGPU;

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

        impl_->cached_input = [[MLMultiArray alloc]
            initWithShape:@[@1, @3, @(FSB_BACKBONE_SIZE), @(FSB_BACKBONE_SIZE)]
                  dataType:MLMultiArrayDataTypeFloat32
                     error:&error];
        if (!impl_->cached_input) {
            std::fprintf(stderr, "[FSB] CoreML backbone input alloc failed: %s\n",
                         error ? [[error localizedDescription] UTF8String] : "unknown error");
            return false;
        }

        impl_->cached_provider = [[FSBBackboneInput alloc] initWithArray:impl_->cached_input];

        return true;
    }
}

bool CoreMLBackbone::run(const float* input_nchw, int batch, float* output_nchw, void** opaque_out)
{
    if (!impl_->model || !input_nchw || !output_nchw || batch <= 0) {
        return false;
    }

    constexpr size_t input_elems = 3u * FSB_BACKBONE_SIZE * FSB_BACKBONE_SIZE;
    constexpr size_t grid = FSB_BACKBONE_SIZE / 16u;
    constexpr size_t output_elems = 1280u * grid * grid;

    @autoreleasepool {
        for (int b = 0; b < batch; ++b) {
            NSError* error = nil;
            std::memcpy(impl_->cached_input.dataPointer,
                        input_nchw + (size_t)b * input_elems,
                        input_elems * sizeof(float));

            id<MLFeatureProvider> result = [impl_->model predictionFromFeatures:impl_->cached_provider error:&error];
            if (!result) {
                std::fprintf(stderr, "[FSB] CoreML backbone prediction failed: %s\n",
                             error ? [[error localizedDescription] UTF8String] : "unknown error");
                return false;
            }

            MLFeatureValue* value = [result featureValueForName:@"features"];
            MLMultiArray* features = value.multiArrayValue;
            if (!features) {
                std::fprintf(stderr, "[FSB] CoreML backbone returned no output.\n");
                return false;
            }

            if (opaque_out) {
                opaque_out[b] = (void*)CFBridgingRetain(features);
                continue; // Skip the copy and type conversion entirely!
            }

            float* dst = output_nchw + (size_t)b * output_elems;
            NSArray<NSNumber*>* shape = features.shape;
            NSArray<NSNumber*>* strides = features.strides;
            int S0 = shape.count > 0 ? shape[0].intValue : 1;
            int S1 = shape.count > 1 ? shape[1].intValue : 1;
            int S2 = shape.count > 2 ? shape[2].intValue : 1;
            int S3 = shape.count > 3 ? shape[3].intValue : 1;
            int St0 = strides.count > 0 ? strides[0].intValue : 0;
            int St1 = strides.count > 1 ? strides[1].intValue : 0;
            int St2 = strides.count > 2 ? strides[2].intValue : 0;
            int St3 = strides.count > 3 ? strides[3].intValue : 0;

            bool is_contiguous = (St3 == 1 && St2 == S3 && St1 == S2 * S3 && St0 == S1 * S2 * S3);

            if (is_contiguous) {
                if (features.dataType == MLMultiArrayDataTypeFloat32) {
                    std::memcpy(dst, features.dataPointer, output_elems * sizeof(float));
                } else if (features.dataType == MLMultiArrayDataTypeFloat16) {
                    const uint16_t* src = static_cast<const uint16_t*>(features.dataPointer);
                    for (size_t i = 0; i < output_elems; ++i) {
                        dst[i] = fsb_half_to_float(src[i]);
                    }
                } else {
                    std::fprintf(stderr, "[FSB] CoreML backbone returned unsupported dtype %ld.\n",
                                 (long)features.dataType);
                    return false;
                }
            } else {
                if (features.dataType == MLMultiArrayDataTypeFloat32) {
                    const float* src = static_cast<const float*>(features.dataPointer);
                    for (int i0 = 0; i0 < S0; ++i0) {
                        for (int i1 = 0; i1 < S1; ++i1) {
                            for (int i2 = 0; i2 < S2; ++i2) {
                                for (int i3 = 0; i3 < S3; ++i3) {
                                    int linear_idx = i0 * St0 + i1 * St1 + i2 * St2 + i3 * St3;
                                    int out_idx = i0 * (S1*S2*S3) + i1 * (S2*S3) + i2 * S3 + i3;
                                    dst[out_idx] = src[linear_idx];
                                }
                            }
                        }
                    }
                } else if (features.dataType == MLMultiArrayDataTypeFloat16) {
                    const uint16_t* src = static_cast<const uint16_t*>(features.dataPointer);
                    for (int i0 = 0; i0 < S0; ++i0) {
                        for (int i1 = 0; i1 < S1; ++i1) {
                            for (int i2 = 0; i2 < S2; ++i2) {
                                for (int i3 = 0; i3 < S3; ++i3) {
                                    int linear_idx = i0 * St0 + i1 * St1 + i2 * St2 + i3 * St3;
                                    int out_idx = i0 * (S1*S2*S3) + i1 * (S2*S3) + i2 * S3 + i3;
                                    dst[out_idx] = fsb_half_to_float(src[linear_idx]);
                                }
                            }
                        }
                    }
                } else {
                    std::fprintf(stderr, "[FSB] CoreML backbone returned unsupported dtype %ld.\n",
                                 (long)features.dataType);
                    return false;
                }
            }
        }
    }

    return true;
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
