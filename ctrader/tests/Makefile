CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -Wpedantic -std=c11

.PHONY: all clean test valgrind

all: test_market_sim test_binance_rest

test_market_sim: test_market_sim.c ../market_sim.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_binance_rest: test_binance_rest.c ../binance_rest.c ../vendor/cJSON.c
	$(CC) $(CFLAGS) -o $@ $^ -lm -lcurl -lssl -lcrypto

test: all
	./test_market_sim
	./test_binance_rest

valgrind: all
	valgrind --leak-check=full --error-exitcode=1 ./test_market_sim
	valgrind --leak-check=full --error-exitcode=1 ./test_binance_rest

clean:
	rm -f test_market_sim test_binance_rest
