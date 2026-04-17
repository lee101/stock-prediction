CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -Wpedantic -std=c11
BINANCE_REST_LIBS = -lcurl -lssl -lcrypto

# Keep the native test entrypoint usable on lean dev boxes where libcurl
# headers are not installed. The REST-specific test binary is built only
# when both curl and OpenSSL headers are available.
CURL_HEADERS_OK := $(shell printf '#include <curl/curl.h>\n' | $(CC) -E -xc - >/dev/null 2>&1 && echo 1 || echo 0)
OPENSSL_HEADERS_OK := $(shell printf '#include <openssl/hmac.h>\n#include <openssl/evp.h>\n' | $(CC) -E -xc - >/dev/null 2>&1 && echo 1 || echo 0)
BINANCE_REST_TESTS_ENABLED := $(shell [ "$(CURL_HEADERS_OK)" = "1" ] && [ "$(OPENSSL_HEADERS_OK)" = "1" ] && echo 1 || echo 0)

.PHONY: all clean test valgrind

all: test_market_sim $(if $(filter 1,$(BINANCE_REST_TESTS_ENABLED)),test_binance_rest)

test_market_sim: test_market_sim.c ../market_sim.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

ifeq ($(BINANCE_REST_TESTS_ENABLED),1)
test_binance_rest: test_binance_rest.c ../binance_rest.c ../vendor/cJSON.c
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BINANCE_REST_LIBS)
endif

test: all
	./test_market_sim
ifeq ($(BINANCE_REST_TESTS_ENABLED),1)
	./test_binance_rest
else
	@echo "Skipping test_binance_rest: missing curl/OpenSSL development headers"
endif

valgrind: all
	valgrind --leak-check=full --error-exitcode=1 ./test_market_sim
ifeq ($(BINANCE_REST_TESTS_ENABLED),1)
	valgrind --leak-check=full --error-exitcode=1 ./test_binance_rest
else
	@echo "Skipping valgrind for test_binance_rest: missing curl/OpenSSL development headers"
endif

clean:
	rm -f test_market_sim test_binance_rest
