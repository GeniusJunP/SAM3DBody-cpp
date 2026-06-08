#!/bin/bash
# Download pre-converted CoreML models from HuggingFace
# (設定が完了したら、URLを実際のリポジトリに書き換えてください)

HF_REPO_URL="https://huggingface.co/YOUR_USERNAME/SAM3DBody-CoreML/resolve/main"

echo "Downloading CoreML Backbone..."
mkdir -p coreml_models
curl -L -o coreml_models/backbone_coreml.mlpackage.zip "$HF_REPO_URL/backbone_coreml.mlpackage.zip"
curl -L -o coreml_models/backbone_coreml_compiled.zip "$HF_REPO_URL/backbone_coreml_compiled.zip"

echo "Extracting models..."
unzip -q coreml_models/backbone_coreml.mlpackage.zip -d coreml_models/
unzip -q coreml_models/backbone_coreml_compiled.zip -d coreml_models/

echo "Done! You can now run the pipeline with --coreml-backbone ./coreml_models/backbone_coreml_compiled/backbone_coreml.mlmodelc"
