# SAM-3D-Body CoreML Export/Build Guide (日本語)

Apple Siliconを活用し、推論を行うためのCoreMLネイティブモデルエクスポートおよびC++ビルドガイドです。

すべてのコマンドはリポジトリルート（`SAM3DBody-cpp/`）から実行します。

## 1. 環境構築

※ モデルのコンパイル（`xcrun`）に必要となるため、あらかじめ Xcode Command Line Tools をインストールしておいてください。 `xcode-select --install`  
※ `coremltools 9.0` は `torch==2.7.0` までテストされています。

```bash
conda create -n coreml_export python=3.11
conda activate coreml_export
conda install opencv=4.6.0
uv pip install -r coreml_export/requirements.txt
```

変換スクリプトはオリジナルのPythonコード（`AmmarkoV/Fast-SAM-3D-Body`）とPyTorchのチェックポイントに依存しています。

```bash
# オリジナルのPythonリポジトリをクローン
git clone --depth 1 https://github.com/AmmarkoV/Fast-SAM-3D-Body.git

# PyTorchのチェックポイントをダウンロード
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam-3d-body-dinov3', local_dir='coreml_export/checkpoints/sam-3d-body-dinov3')"
```

## 2. CoreMLモデルのエクスポート

```bash
make models
```

エクスポート後、Xcodeの `coremlcompiler` によって `.mlmodelc` にコンパイルされます。

## 3. C++ ビルド

```bash
make clean
make build
```

## 4. 推論テスト

```bash
./build/fast_sam_3dbody_run \
  --from ../person.jpg \
  --coreml-backbone coreml_export/checkpoints/backbone_coreml.mlmodelc \
  --coreml-decoder coreml_export/checkpoints/decoder_coreml.mlmodelc \
  --coreml-yolo coreml_export/checkpoints/yolo11m-pose.mlmodelc \
  --headless
```

> **Note:** 上記の `--coreml-*` 引数群は、他の実行コマンドにもそのまま引き継いで使用できます。
> 例えば、C++のウェブカメラ推論や、オフラインのBVH書き出しスクリプトを実行する際にも、元の `--onnx-dir` 等の代わりにこれらの引数を付与するだけで、すべてCoreMLによる推論が有効になります。
>
> ```bash
> # C++ウェブカメラ推論での実行例
> ./build/fast_sam_3dbody_run \
>   --onnx-dir ./onnx \
>   --gguf ./onnx/pipeline.gguf \
>   --coreml-backbone ./coreml_export/checkpoints/backbone_coreml.mlmodelc \
>   --coreml-decoder ./coreml_export/checkpoints/decoder_coreml.mlmodelc \
>   --coreml-yolo ./coreml_export/checkpoints/yolo11m-pose.mlmodelc \
>   --from 0
>
> # オフラインのマルチパスBVH生成での実行例
> ./build/offline_sam_3dbody_render \
>   --from video.mp4 \
>   --bvh output.bvh \
>   --coreml-backbone coreml_export/checkpoints/backbone_coreml.mlmodelc \
>   --coreml-decoder coreml_export/checkpoints/decoder_coreml.mlmodelc \
>   --coreml-yolo coreml_export/checkpoints/yolo11m-pose.mlmodelc
> ```
