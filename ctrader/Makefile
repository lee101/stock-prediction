CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -Wpedantic -std=c11
LDFLAGS = -lm

SRCS = main.c trade_loop.c market_sim.c binance_rest.c policy_infer.c mktd_reader.c
OBJS = $(SRCS:.c=.o)

# libtorch (optional)
ifdef TORCH_DIR
  CFLAGS += -I$(TORCH_DIR)/include -I$(TORCH_DIR)/include/torch/csrc/api/include
  LDFLAGS += -L$(TORCH_DIR)/lib -ltorch -lc10 -Wl,-rpath,$(TORCH_DIR)/lib
endif

# libcurl (optional, for real binance_rest)
ifdef USE_CURL
  LDFLAGS += -lcurl -lssl -lcrypto
endif

.PHONY: all clean test test_valgrind

all: ctrader

ctrader: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: test_market_sim test_binance_rest test_mktd_reader
	./test_market_sim
	./test_binance_rest
	./test_mktd_reader

test_market_sim: tests/test_market_sim.c market_sim.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_binance_rest: tests/test_binance_rest.c binance_rest.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_mktd_reader: tests/test_mktd_reader.c mktd_reader.c market_sim.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_valgrind: test_market_sim test_binance_rest test_mktd_reader
	valgrind --leak-check=full --error-exitcode=1 ./test_market_sim
	valgrind --leak-check=full --error-exitcode=1 ./test_binance_rest
	valgrind --leak-check=full --error-exitcode=1 ./test_mktd_reader

clean:
	rm -f ctrader test_market_sim test_binance_rest test_mktd_reader $(OBJS)
