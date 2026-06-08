#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "coreml_yolo.h"
#include "coreml_utils.h"
#include <iostream>
#include <vector>

#if !__has_feature(objc_arc)
#error "This file must be compiled with ARC."
#endif

struct CoreMLYoloCtx {
    MLModel* model;
    MLMultiArray* cached_input;
    MLDictionaryFeatureProvider* cached_provider;
};

extern "C" CoreMLYoloContext init_coreml_yolo(const char* mlpackage_path) {
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:mlpackage_path];
        NSURL* url = [NSURL fileURLWithPath:path];
        if (!url) {
            std::cerr << "[CoreML YOLO] Invalid path." << std::endl;
            return nullptr;
        }

        NSError* error = nil;
        NSURL* compiledUrl = url;
        if (![path hasSuffix:@".mlmodelc"]) {
            compiledUrl = [MLModel compileModelAtURL:url error:&error];
            if (error) {
                std::cerr << "[CoreML YOLO] Compile error: " << error.localizedDescription.UTF8String << std::endl;
                return nullptr;
            }
        }

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;

        MLModel* model = [MLModel modelWithContentsOfURL:compiledUrl configuration:config error:&error];
        if (error || !model) {
            std::cerr << "[CoreML YOLO] Load error: " << (error ? error.localizedDescription.UTF8String : "Unknown") << std::endl;
            return nullptr;
        }

        CoreMLYoloCtx* ctx = new CoreMLYoloCtx();
        ctx->model = model;

        ctx->cached_input = [[MLMultiArray alloc] initWithShape:@[@1, @3, @640, @640] 
                                                       dataType:MLMultiArrayDataTypeFloat32 
                                                          error:&error];

        ctx->cached_provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"images": ctx->cached_input} error:&error];

        return (CoreMLYoloContext)ctx;
    }
}

extern "C" void free_coreml_yolo(CoreMLYoloContext ctx_ptr) {
    @autoreleasepool {
        if (!ctx_ptr) return;
        CoreMLYoloCtx* ctx = (CoreMLYoloCtx*)ctx_ptr;
        ctx->model = nil;
        ctx->cached_input = nil;
        ctx->cached_provider = nil;
        delete ctx;
    }
}

extern "C" bool run_coreml_yolo(CoreMLYoloContext ctx_ptr, const float* input_bchw, float* output) {
    @autoreleasepool {
        if (!ctx_ptr) return false;
        CoreMLYoloCtx* ctx = (CoreMLYoloCtx*)ctx_ptr;

        NSError* error = nil;

        // Copy data to cached input
        float* inPtr = (float*)ctx->cached_input.dataPointer;
        std::memcpy(inPtr, input_bchw, 1 * 3 * 640 * 640 * sizeof(float));

        id<MLFeatureProvider> outProvider = [ctx->model predictionFromFeatures:ctx->cached_provider error:&error];
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
