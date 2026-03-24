CXX = g++
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -std=c11 -Wall
CXXFLAGS = -O3 -march=native -ffast-math -std=c++17 -Wall

# libtorch (CUDA)
TORCH_DIR ?= /usr/local/lib/libtorch
TORCH_CXXFLAGS = -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include \
                 -D_GLIBCXX_USE_CXX11_ABI=1
TORCH_LDFLAGS = -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 \
                -Wl,-rpath,$(TORCH_DIR)/lib

# detect CUDA support
CUDA_LIBS =
ifneq ($(wildcard $(TORCH_DIR)/lib/libtorch_cuda.so),)
    TORCH_LDFLAGS += -ltorch_cuda -lc10_cuda
    CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart
endif

C_SRCS = trading_env.c vec_env.c vendor/cJSON.c
CXX_SRCS = policy.cpp ppo.cpp checkpoint.cpp evaluate.cpp autoresearch.cpp

C_OBJS = $(C_SRCS:.c=.o)
CXX_OBJS = $(CXX_SRCS:.cpp=.o)
ALL_OBJS = $(C_OBJS) $(CXX_OBJS)

LDFLAGS = $(TORCH_LDFLAGS) $(CUDA_LIBS) -lpthread -lm

.PHONY: all clean test

all: crl_train crl_autoresearch crl_evaluate crl_benchmark

crl_train: main_train.cpp $(ALL_OBJS)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -o $@ $^ $(LDFLAGS)

crl_autoresearch: main_autoresearch.cpp $(ALL_OBJS)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -o $@ $^ $(LDFLAGS)

crl_evaluate: main_evaluate.cpp $(ALL_OBJS)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -o $@ $^ $(LDFLAGS)

crl_benchmark: main_benchmark.cpp $(ALL_OBJS)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -c -o $@ $<

# test targets (C only, no libtorch needed)
test_vec_env: tests/test_vec_env.c trading_env.c vec_env.c
	$(CC) $(CFLAGS) -o $@ $^ -lm
	./$@

test: test_vec_env
	@echo "All tests passed"

clean:
	rm -f *.o vendor/*.o crl_train crl_autoresearch crl_evaluate crl_benchmark test_vec_env
