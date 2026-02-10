CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -lm

.PHONY: test clean

test: test_trading_sim
	./test_trading_sim

test_trading_sim: test_trading_sim.c trading_sim.h
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f test_trading_sim
