CC = gcc
CXX = g++
CFLAGS = -O3 -march=native -Wall -Wextra -Wpedantic -std=c11
CXXFLAGS = -O3 -march=native -Wall -Wextra -std=c++17
LDFLAGS = -lm

TORCH_DIR ?=
LIBTORCH_URL = https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip

C_SRCS = main.c trade_loop.c market_sim.c binance_rest.c

ifdef TORCH_DIR
  TORCH_CXXFLAGS = -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
  TORCH_LDFLAGS = -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_DIR)/lib -lstdc++ -lpthread
  POLICY_SRC = policy_infer.cpp
  POLICY_OBJ = policy_infer_cpp.o
  LINK = $(CXX)
else
  TORCH_CXXFLAGS =
  TORCH_LDFLAGS =
  POLICY_SRC = policy_infer.c
  POLICY_OBJ = policy_infer.o
  LINK = $(CC)
endif

C_OBJS = $(C_SRCS:.c=.o)
ALL_OBJS = $(C_OBJS) $(POLICY_OBJ)

# libcurl (optional, for real binance_rest)
ifdef USE_CURL
  LDFLAGS += -lcurl -lssl -lcrypto
endif

.PHONY: all clean test test_valgrind test_policy download_libtorch

all: ctrader

ctrader: $(ALL_OBJS)
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(TORCH_LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

policy_infer_cpp.o: policy_infer.cpp policy_infer.h
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -c -o $@ $<

test: test_market_sim test_binance_rest
	./test_market_sim
	./test_binance_rest

test_market_sim: tests/test_market_sim.c market_sim.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_binance_rest: tests/test_binance_rest.c binance_rest.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

ifdef TORCH_DIR
test_policy: tests/test_policy_infer.cpp policy_infer.cpp policy_infer.h
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -o $@ tests/test_policy_infer.cpp policy_infer.cpp $(LDFLAGS) $(TORCH_LDFLAGS)
	./test_policy
else
test_policy:
	@echo "TORCH_DIR not set, skipping policy_infer tests"
endif

test_valgrind: test_market_sim test_binance_rest
	valgrind --leak-check=full --error-exitcode=1 ./test_market_sim
	valgrind --leak-check=full --error-exitcode=1 ./test_binance_rest

download_libtorch:
	@if [ -d "libtorch" ]; then echo "libtorch already downloaded"; \
	else wget -q $(LIBTORCH_URL) -O /tmp/libtorch.zip && \
	     unzip -qo /tmp/libtorch.zip -d . && \
	     rm -f /tmp/libtorch.zip && \
	     echo "libtorch downloaded to ctrader/libtorch/"; fi

clean:
	rm -f ctrader test_market_sim test_binance_rest test_policy $(ALL_OBJS) policy_infer.o policy_infer_cpp.o
