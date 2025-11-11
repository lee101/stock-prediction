.PHONY: all build clean test benchmark help export-models

BUILD_DIR = build
LIBTORCH_DIR = ../external/libtorch/libtorch

help:
	@echo "Fast Market Sim with Toto/Kronos - Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make export-models  - Export Toto/Kronos to TorchScript"
	@echo "  make build          - Build C++ library and examples"
	@echo "  make benchmark      - Run forecaster benchmarks"
	@echo "  make test           - Run integration tests"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make all            - Export models, build, and test"

export-models:
	@echo "📦 Exporting models to TorchScript..."
	cd .. && python export_models_to_torchscript.py --model both --device cuda
	@echo "✅ Models exported to compiled_models_torchscript/"

build:
	@echo "🔨 Building fast_market_sim..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -DCMAKE_BUILD_TYPE=Release ..
	cd $(BUILD_DIR) && make -j$(shell nproc)
	@echo "✅ Build complete!"

build-debug:
	@echo "🔨 Building fast_market_sim (Debug)..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR) -DCMAKE_BUILD_TYPE=Debug ..
	cd $(BUILD_DIR) && make -j$(shell nproc)
	@echo "✅ Debug build complete!"

benchmark: build
	@echo "⚡ Running forecaster benchmarks..."
	cd $(BUILD_DIR) && ./benchmark_forecaster ../compiled_models_torchscript/toto_fp32.pt

test: build
	@echo "🧪 Running integration tests..."
	cd $(BUILD_DIR) && ./test_integration

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	find . -name "*.o" -delete
	find . -name "*.so" -delete
	@echo "✅ Clean complete!"

all: export-models build benchmark
	@echo "✅ Full build and test complete!"

# Check prerequisites
check:
	@echo "Checking prerequisites..."
	@echo "LibTorch: $$(test -d $(LIBTORCH_DIR) && echo 'Found' || echo 'NOT FOUND')"
	@echo "CUDA: $$(which nvcc > /dev/null && nvcc --version | grep release || echo 'Not found')"
	@echo "CMake: $$(cmake --version | head -1)"
	@echo "Python: $$(python --version)"
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Quick start for new users
quickstart: export-models build
	@echo ""
	@echo "✅ Quick start complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run benchmarks: make benchmark"
	@echo "  2. Run tests: make test"
	@echo "  3. Check examples in $(BUILD_DIR)/"
