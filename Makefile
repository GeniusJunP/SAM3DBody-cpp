
.PHONY: all models build clean

all: models build

models:
	@echo "========================================"
	@echo " Exporting Backbone"
	@echo "========================================"
	mkdir -p coreml_export/checkpoints
	cd coreml_export/checkpoints && python ../export_coreml_backbone.py --out ./backbone_coreml.mlpackage
	cd coreml_export/checkpoints && python ../export_coreml_decoder.py --out ./decoder_coreml.mlpackage
	cd coreml_export/checkpoints && python ../export_coreml_yolo.py --model ./yolo11m-pose.pt --out ./yolo11m-pose.mlpackage
	@echo "Compiling CoreML packages to mlmodelc..."
	xcrun coremlcompiler compile ./coreml_export/checkpoints/backbone_coreml.mlpackage ./coreml_export/checkpoints/
	xcrun coremlcompiler compile ./coreml_export/checkpoints/decoder_coreml.mlpackage ./coreml_export/checkpoints/
	xcrun coremlcompiler compile ./coreml_export/checkpoints/yolo11m-pose.mlpackage ./coreml_export/checkpoints/
	@echo "Models ready."



build:
	@echo "========================================"
	@echo " Building C++ Engine"
	@echo "========================================"
	CMAKE_PREFIX_PATH=$(CONDA_PREFIX) cmake -S . -B build -DFSB_COREML=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j

clean:
	rm -rf build
	rm -rf CMakeCache.txt CMakeFiles
	rm -rf coreml_export/backbone_coreml.mlmodelc
	rm -rf coreml_export/decoder_coreml.mlmodelc
	rm -rf coreml_export/yolo11m-pose.mlmodelc
