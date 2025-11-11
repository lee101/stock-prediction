.PHONY: all build clean test install benchmark help

# Paths
BUILD_DIR = build
LIBTORCH_DIR = ../external/libtorch/libtorch
PYTHON = python3

# Build configuration
CMAKE_FLAGS = -DCMAKE_PREFIX_PATH=$(LIBTORCH_DIR)
CMAKE_BUILD_TYPE = Release

help:
	@echo "Market Simulation C/CUDA Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make build          - Build the C/CUDA libraries"
	@echo "  make build-debug    - Build with debug symbols and sanitizers"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make test           - Run tests"
	@echo "  make benchmark      - Run performance benchmarks"
	@echo "  make install        - Install Python bindings"
	@echo "  make all            - Build and test everything"

all: build test

build:
	@echo "Building market simulation (Release mode)..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) ..
	cd $(BUILD_DIR) && make -j$(shell nproc)
	@echo "Build complete!"

build-debug:
	@echo "Building market simulation (Debug mode with sanitizers)..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=Debug ..
	cd $(BUILD_DIR) && make -j$(shell nproc)
	@echo "Debug build complete!"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf python/__pycache__
	rm -rf examples/__pycache__
	find . -name "*.so" -delete
	find . -name "*.pyc" -delete
	@echo "Clean complete!"

test: build
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v
	@echo "Tests complete!"

benchmark: build
	@echo "Running performance benchmarks..."
	$(PYTHON) examples/benchmark.py
	@echo "Benchmark complete!"

install: build
	@echo "Installing Python bindings..."
	cd $(BUILD_DIR) && make install
	$(PYTHON) -m pip install -e .
	@echo "Installation complete!"

# Quick sanity check
check:
	@echo "Checking environment..."
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "PyTorch: $(shell $(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
	@echo "CUDA: $(shell $(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not available')"
	@echo "LibTorch: $(shell test -d $(LIBTORCH_DIR) && echo 'Found' || echo 'Not found')"
	@echo "nvcc: $(shell which nvcc > /dev/null && nvcc --version | grep release || echo 'Not found')"
