# SAM-3D-Body CoreML Export/Build Guide

This is a guide for exporting and building native CoreML models to leverage Apple Silicon acceleration.

All commands below should be executed from the repository root directory (`SAM3DBody-cpp/`).

## 1. Environment Setup

* Xcode Command Line Tools must be installed beforehand for model compilation (`xcrun`). Use `xcode-select --install`.
* `coremltools 9.0` has been tested and verified up to `torch==2.7.0`.

```bash
conda create -n coreml_export python=3.11
conda activate coreml_export
conda install opencv=4.6.0
uv pip install -r coreml_export/requirements.txt
```

The conversion/export scripts depend on the original Python repository (`AmmarkoV/Fast-SAM-3D-Body`) and the PyTorch checkpoints.

```bash
# Clone the original Python repository
git clone --depth 1 https://github.com/AmmarkoV/Fast-SAM-3D-Body.git

# Download the PyTorch checkpoints
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam-3d-body-dinov3', local_dir='coreml_export/checkpoints/sam-3d-body-dinov3')"
```

## 2. CoreML Model Export

```bash
make models
```

After exporting, the models are automatically compiled into `.mlmodelc` using Xcode's `coremlcompiler`.

## 3. C++ Build

The C++ engine strictly requires the legacy OpenCV 4.6 API to maintain upstream compatibility. By default, the `Makefile` will use the `coreml_export` conda environment to resolve these dependencies.

```bash
make clean
make build
```

## 4. Inference Test

```bash
./build/fast_sam_3dbody_run \
  --from ../person.jpg \
  --coreml-backbone coreml_export/checkpoints/backbone_coreml.mlmodelc \
  --coreml-decoder coreml_export/checkpoints/decoder_coreml.mlmodelc \
  --coreml-yolo coreml_export/checkpoints/yolo11m-pose.mlmodelc \
  --headless
```

> **Note:** The `--coreml-*` argument family can be inherited by any other execution commands.
> For example, in the lightweight C++ webcam frontend or the offline BVH export pipeline, you can simply pass these CoreML paths in place of `--onnx-dir` to enable CoreML inference seamlessly.
>
> ```bash
> # Example: Running C++ webcam inference
> ./build/fast_sam_3dbody_run \
>   --onnx-dir ./onnx \
>   --gguf ./onnx/pipeline.gguf \
>   --coreml-backbone coreml_export/checkpoints/backbone_coreml.mlmodelc \
>   --coreml-decoder coreml_export/checkpoints/decoder_coreml.mlmodelc \
>   --coreml-yolo coreml_export/checkpoints/yolo11m-pose.mlmodelc \
>   --from 0
>
> # Example: Offline multi-person BVH generation
> ./build/offline_sam_3dbody_render \
>   --from video.mp4 \
>   --bvh output.bvh \
>   --coreml-backbone coreml_export/checkpoints/backbone_coreml.mlmodelc \
>   --coreml-decoder coreml_export/checkpoints/decoder_coreml.mlmodelc \
>   --coreml-yolo coreml_export/checkpoints/yolo11m-pose.mlmodelc
> ```

## 5. Performance and Known Limitations

### Performance
(Tested on: Apple M1 Max, 10-core CPU, 32GB RAM)

The first frame takes around 700 ms due to ANE/GPU warmup, but in the steady state from the second frame onwards, you can expect an approximate latency of **400 ms / frame (2.5 fps)**.

| Module | CoreML Measured (Steady State) | ONNX Runtime (CUDA Baseline) |
| :--- | :--- | :--- |
| **YOLOv11m** | ~20 ms | ~15 ms |
| **Backbone** | ~360 ms | ~191 ms |
| **Decoder** | ~13 ms | ~8.6 ms |

### Known Limitations & Architecture Notes
- **Decoder Batch Size Constraint**: Apple's GPU (Metal) compiler (E5RT) does not permit complex tensor reshaping (e.g. `view` for Attention `Q, K, V`) containing dynamic dimensions. If dynamic batch sizes (`RangeDim`) are used, shape inference fails at compile time, causing a silent fallback to CPU execution which severely degrades performance. Therefore, the Decoder is strictly compiled with a static batch size of **`B=1`**. At runtime, the C++ inference engine wraps the `B=1` model in an `MLArrayBatchProvider` to efficiently schedule and parallelize inference for multiple detections.
- **Backbone Computation Speed**: The CoreML execution of the Backbone currently takes significantly longer compared to ONNX Runtime (CUDA). This may improve in the future if model graph optimizations or full ANE mapping are introduced.
- **Export Script Options**: The export process utilizes `torch.jit.trace` with `FLOAT32` precision to preserve the ATen computational graph, enabling the `coremltools` compiler to successfully apply Scaled Dot Product Attention (SDPA) fusion. We intentionally omit `FLOAT16` casting during PyTorch export to bypass a known compiler hang in `coremltools` when exporting complex Vision Transformer models.
