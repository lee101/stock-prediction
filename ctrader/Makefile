CC = gcc
CXX = g++
CFLAGS = -O3 -march=native -Wall -Wextra -Wpedantic -std=c11
CXXFLAGS = -O3 -march=native -Wall -Wextra -std=c++17
LDFLAGS = -lm
TMPDIR ?= $(abspath .tmp)
export TMPDIR

TORCH_DIR ?=
LIBTORCH_URL = https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip

C_SRCS = main.c trade_loop.c market_sim.c binance_rest.c vendor/cJSON.c mktd_reader.c

ifdef TORCH_DIR
  TORCH_CXXFLAGS = -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=1
  TORCH_LDFLAGS = -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_DIR)/lib -lstdc++ -lpthread
  POLICY_SRC = policy_infer.cpp
  POLICY_OBJ = policy_infer_cpp.o
  LINK = $(CXX)
else
  POLICY_SRC = policy_infer.c
  POLICY_OBJ = policy_infer.o
  LINK = $(CC)
endif

C_OBJS = $(C_SRCS:.c=.o)
ALL_OBJS = $(C_OBJS) $(POLICY_OBJ)

# libcurl + openssl for binance REST API
LDFLAGS += -lcurl -lssl -lcrypto

.PHONY: all clean test test_parity test_valgrind test_asan download_libtorch

all: ctrader

$(TMPDIR):
	mkdir -p $@

libmarket_sim.so: market_sim.c market_sim.h | $(TMPDIR)
	$(CC) $(CFLAGS) -shared -fPIC -o $@ market_sim.c -lm

ctrader: $(ALL_OBJS) | $(TMPDIR)
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(TORCH_LDFLAGS)

%.o: %.c | $(TMPDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

policy_infer_cpp.o: policy_infer.cpp policy_infer.h | $(TMPDIR)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -c -o $@ $<

test: test_market_sim test_binance_rest test_mktd_reader
	./test_market_sim
	./test_binance_rest
	./test_mktd_reader

test_market_sim: tests/test_market_sim.c market_sim.c | $(TMPDIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_binance_rest: tests/test_binance_rest.c binance_rest.c vendor/cJSON.c | $(TMPDIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lcurl -lssl -lcrypto

test_sim_parity: tests/test_sim_parity.c market_sim.c | $(TMPDIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_parity: test_sim_parity
	./test_sim_parity

test_mktd_reader: tests/test_mktd_reader.c mktd_reader.c market_sim.c | $(TMPDIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm

ifdef TORCH_DIR
test_policy: tests/test_policy_infer.cpp policy_infer.cpp policy_infer.h | $(TMPDIR)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -o $@ tests/test_policy_infer.cpp policy_infer.cpp $(LDFLAGS) $(TORCH_LDFLAGS)
	./test_policy
else
test_policy:
	@echo "TORCH_DIR not set, skipping policy_infer tests"
endif

test_asan: | $(TMPDIR)
	$(CC) -O1 -g -fsanitize=address,undefined -Wall -Wextra -Wpedantic -std=c11 -o test_market_sim_asan tests/test_market_sim.c market_sim.c -lm
	./test_market_sim_asan
	rm -f test_market_sim_asan

test_valgrind: test_market_sim test_binance_rest test_mktd_reader
	valgrind --leak-check=full --error-exitcode=1 ./test_market_sim
	valgrind --leak-check=full --error-exitcode=1 ./test_binance_rest
	valgrind --leak-check=full --error-exitcode=1 ./test_mktd_reader

download_libtorch:
	curl -L $(LIBTORCH_URL) -o /tmp/libtorch.zip
	unzip -q /tmp/libtorch.zip -d /tmp/
	mv /tmp/libtorch .

clean:
	rm -rf $(TMPDIR)
	rm -f ctrader libmarket_sim.so test_market_sim test_binance_rest test_sim_parity test_policy $(C_OBJS) $(POLICY_OBJ)
