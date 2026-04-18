# Build the .cu example against XGBoost's C API.
#
# Requires a *source* build of XGBoost with -DUSE_CUDA=ON, because the
# PyPI wheel ships only libxgboost.so (no headers). Set XGBOOST_HOME to
# the source tree root before running `make`.
#
#   git clone --recursive https://github.com/dmlc/xgboost.git $HOME/src/xgboost
#   cmake -S $HOME/src/xgboost -B $HOME/src/xgboost/build -DUSE_CUDA=ON -GNinja
#   cmake --build $HOME/src/xgboost/build -j
#   export XGBOOST_HOME=$HOME/src/xgboost
#   make
#
# CUDA_HOME defaults to /usr/local/cuda; override with `make CUDA_HOME=...`.

XGBOOST_HOME ?= $(HOME)/src/xgboost
CUDA_HOME    ?= /usr/local/cuda
NVCC         ?= $(CUDA_HOME)/bin/nvcc
ARCH         ?= sm_120   # Blackwell (RTX 5090). Use sm_89 for Ada, sm_86 for Ampere.

INC := -I$(XGBOOST_HOME)/include -I$(CUDA_HOME)/include
LIB := -L$(XGBOOST_HOME)/lib -L$(XGBOOST_HOME)/build -L$(CUDA_HOME)/lib64
LDL := -lxgboost -lcudart

NVCC_FLAGS := -O3 -std=c++17 -arch=$(ARCH)

BIN := xgb_cuda_market_sim_example

all: $(BIN)

$(BIN): xgb_cuda_market_sim_example.cu
	@if [ ! -f "$(XGBOOST_HOME)/include/xgboost/c_api.h" ]; then \
		echo "Error: $(XGBOOST_HOME)/include/xgboost/c_api.h not found."; \
		echo "       Set XGBOOST_HOME to an XGBoost source tree built with -DUSE_CUDA=ON."; \
		exit 2; \
	fi
	$(NVCC) $(NVCC_FLAGS) $(INC) $(LIB) $< $(LDL) -o $@
	@echo "Built $@. To run with the correct libxgboost.so:"
	@echo "    LD_LIBRARY_PATH=\$$XGBOOST_HOME/lib:\$$LD_LIBRARY_PATH ./$@"

run: $(BIN)
	LD_LIBRARY_PATH=$(XGBOOST_HOME)/lib:$(XGBOOST_HOME)/build:$$LD_LIBRARY_PATH ./$(BIN)

clean:
	rm -f $(BIN) xgb_cuda_market_sim.json

.PHONY: all run clean
