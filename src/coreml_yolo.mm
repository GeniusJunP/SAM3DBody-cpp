#include "coreml_yolo.h"

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
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
    CVPixelBufferRef cached_input_pb = NULL;
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

        NSDictionary *options = @{
            (NSString*)kCVPixelBufferCGImageCompatibilityKey: @YES,
            (NSString*)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES
        };
        CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, 640, 640, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)options, &impl_->cached_input_pb);
        if (status != kCVReturnSuccess || !impl_->cached_input_pb) {
            std::cerr << "[CoreML YOLO] Failed to allocate CVPixelBuffer" << std::endl;
            return false;
        }

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
        if (impl_->cached_input_pb) {
            CVPixelBufferRelease(impl_->cached_input_pb);
            impl_->cached_input_pb = NULL;
        }
    }
}

bool CoreMLYolo::loaded() const {
    return impl_->model != nil;
}

static bool extract_yolo_output(id<MLFeatureProvider> outProvider, float* output) {
    NSSet<NSString*>* outNames = outProvider.featureNames;
    if (outNames.count == 0) {
        std::cerr << "[CoreML YOLO] No output features returned." << std::endl;
        return false;
    }
    
    // Find the feature that is a MultiArray and has the largest number of elements (to avoid fetching metadata outputs)
    MLMultiArray* mlOutput = nil;
    NSInteger max_elements = 0;
    for (NSString* name in outNames) {
        MLFeatureValue* val = [outProvider featureValueForName:name];
        if (val && val.type == MLFeatureTypeMultiArray && val.multiArrayValue) {
            NSInteger elements = val.multiArrayValue.count;
            if (elements > max_elements) {
                max_elements = elements;
                mlOutput = val.multiArrayValue;
            }
        }
    }

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

    // YOLOv8/11 outputs can be [1, 56, 8400] or [1, 8400, 56]. We want to output [8400, 56] (row major)
    bool needs_transpose = (S1 < S2); 
    int out_cols = needs_transpose ? S1 : S2;

    if (mlOutput.dataType == MLMultiArrayDataTypeFloat32) {
        float* outPtr = (float*)mlOutput.dataPointer;
        for (int i = 0; i < S0; ++i) {
            for (int j = 0; j < S1; ++j) {
                for (int k = 0; k < S2; ++k) {
                    int linear_idx = i * St0 + j * St1 + k * St2;
                    int out_idx = needs_transpose ? (k * out_cols + j) : (j * out_cols + k);
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
                    int out_idx = needs_transpose ? (k * out_cols + j) : (j * out_cols + k);
                    output[out_idx] = fsb_half_to_float(outPtr[linear_idx]);
                }
            }
        }
    } else {
        std::cerr << "[CoreML YOLO] Unsupported output data type." << std::endl;
        return false;
    }

    return true;
}

bool CoreMLYolo::run(const float* input_bchw, float* output) {
    @autoreleasepool {
        if (!impl_->model || !impl_->cached_input_pb) return false;

        NSError* error = nil;

        // YOLO CoreML export uses an ImageType input. We must feed it a CVPixelBuffer.
        // The input_bchw is float in [0, 1], shape [1, 3, 640, 640], RGB order.
        CVPixelBufferLockBaseAddress(impl_->cached_input_pb, 0);
        uint8_t* base = (uint8_t*)CVPixelBufferGetBaseAddress(impl_->cached_input_pb);
        size_t stride = CVPixelBufferGetBytesPerRow(impl_->cached_input_pb);

        for (int y = 0; y < 640; ++y) {
            for (int x = 0; x < 640; ++x) {
                float r = input_bchw[0 * 640 * 640 + y * 640 + x];
                float g = input_bchw[1 * 640 * 640 + y * 640 + x];
                float b = input_bchw[2 * 640 * 640 + y * 640 + x];
                
                // Clamp and scale to [0, 255]
                r = std::max(0.0f, std::min(255.0f, r * 255.0f));
                g = std::max(0.0f, std::min(255.0f, g * 255.0f));
                b = std::max(0.0f, std::min(255.0f, b * 255.0f));

                base[y * stride + x * 4 + 0] = (uint8_t)b; // B
                base[y * stride + x * 4 + 1] = (uint8_t)g; // G
                base[y * stride + x * 4 + 2] = (uint8_t)r; // R
                base[y * stride + x * 4 + 3] = 255;        // A
            }
        }
        CVPixelBufferUnlockBaseAddress(impl_->cached_input_pb, 0);

        MLFeatureValue* imageFeature = [MLFeatureValue featureValueWithPixelBuffer:impl_->cached_input_pb];
        id<MLFeatureProvider> provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"image": imageFeature} error:&error];
        
        id<MLFeatureProvider> outProvider = [impl_->model predictionFromFeatures:provider error:&error];
        if (error || !outProvider) {
            std::cerr << "[CoreML YOLO] Prediction error: " << (error ? error.localizedDescription.UTF8String : "Unknown") << std::endl;
            return false;
        }

        return extract_yolo_output(outProvider, output);
    }
}

bool CoreMLYolo::run_image(const uint8_t* bgra_data, int width, int height, int stride, float* output) {
    @autoreleasepool {
        if (!impl_->model) return false;

        NSError* error = nil;
        CVPixelBufferRef pixelBuffer = NULL;
        NSDictionary *options = @{
            (NSString*)kCVPixelBufferCGImageCompatibilityKey: @YES,
            (NSString*)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES
        };
        
        CVReturn status = CVPixelBufferCreateWithBytes(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, 
                                                       (void*)bgra_data, stride, NULL, NULL, 
                                                       (__bridge CFDictionaryRef)options, &pixelBuffer);
        
        if (status != kCVReturnSuccess || !pixelBuffer) {
            std::cerr << "[CoreML YOLO] Failed to wrap CVPixelBuffer" << std::endl;
            return false;
        }

        MLFeatureValue* imageFeature = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
        id<MLFeatureProvider> provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"image": imageFeature} error:&error];
        
        id<MLFeatureProvider> outProvider = [impl_->model predictionFromFeatures:provider error:&error];
        CVPixelBufferRelease(pixelBuffer);

        if (error || !outProvider) {
            std::cerr << "[CoreML YOLO] Prediction error: " << (error ? error.localizedDescription.UTF8String : "Unknown") << std::endl;
            return false;
        }

        return extract_yolo_output(outProvider, output);
    }
}

} // namespace fsb

