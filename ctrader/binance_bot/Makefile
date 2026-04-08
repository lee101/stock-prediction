# ctrader/binance_bot/Makefile
#
# Builds the pure-C binance_bot pieces: policy_mlp loader+forward,
# obs_builder, end-to-end backtest, and two parity tests.  Deliberately
# self-contained -- no libtorch, no libcurl, no BLAS.  Live trading
# (binance_rest.c) is NOT built here; that needs libcurl + libssl and is
# handled by ../Makefile.
#
# Usage:
#   cd ctrader/binance_bot
#   make                  # build all binaries into .tmp/
#   make test             # build and run both parity tests + backtest smoke
#   make fixtures         # regenerate parity fixtures from Python
#
# TMPDIR is pinned to ./.tmp to dodge the global /tmp gcc/triton race
# described in alpacaprod.md.

CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

REPO_ROOT = ../..
DATA_VAL  = $(REPO_ROOT)/pufferlib_market/data/stocks12_daily_v5_rsi_val.bin
CKPT_S42  = $(REPO_ROOT)/pufferlib_market/checkpoints/stocks12_v5_rsi/tp05_s42/best.pt
CTRDPOL   = ../models/stocks12_v5_rsi_s42.ctrdpol

BUILD_DIR = .tmp
export TMPDIR = $(abspath $(BUILD_DIR))

.PHONY: all test fixtures clean

all: $(BUILD_DIR)/backtest_main $(BUILD_DIR)/test_policy_mlp_parity $(BUILD_DIR)/test_obs_builder_parity

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/backtest_main: backtest_main.c policy_mlp.c obs_builder.c ../mktd_reader.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ backtest_main.c policy_mlp.c obs_builder.c ../mktd_reader.c $(LDFLAGS)

$(BUILD_DIR)/test_policy_mlp_parity: tests/test_policy_mlp_parity.c policy_mlp.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ tests/test_policy_mlp_parity.c policy_mlp.c $(LDFLAGS)

$(BUILD_DIR)/test_obs_builder_parity: tests/test_obs_builder_parity.c obs_builder.c ../mktd_reader.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ tests/test_obs_builder_parity.c obs_builder.c ../mktd_reader.c $(LDFLAGS)

fixtures:
	cd $(REPO_ROOT) && . .venv313/bin/activate && PYTHONPATH=. python scripts/export_policy_to_ctrader.py \
		--checkpoint $(CKPT_S42:$(REPO_ROOT)/%=%) \
		--out ctrader/models/stocks12_v5_rsi_s42.ctrdpol \
		--num-symbols 12
	cd $(REPO_ROOT) && . .venv313/bin/activate && PYTHONPATH=. python scripts/gen_policy_mlp_parity_fixture.py \
		--checkpoint $(CKPT_S42:$(REPO_ROOT)/%=%) \
		--out ctrader/binance_bot/tests/policy_mlp_parity_fixture.bin \
		--seed 0
	cd $(REPO_ROOT) && . .venv313/bin/activate && PYTHONPATH=. python scripts/gen_obs_parity_fixture.py \
		--data $(DATA_VAL:$(REPO_ROOT)/%=%) \
		--t 30 \
		--out ctrader/binance_bot/tests/obs_parity_fixture.bin

test: all
	./$(BUILD_DIR)/test_policy_mlp_parity $(CTRDPOL) tests/policy_mlp_parity_fixture.bin
	./$(BUILD_DIR)/test_obs_builder_parity $(DATA_VAL) tests/obs_parity_fixture.bin
	./$(BUILD_DIR)/backtest_main \
	    --data $(DATA_VAL) \
	    --policy $(CTRDPOL) \
	    --window-start 0 --window-steps 90 \
	    --fee-rate 0.001

clean:
	rm -rf $(BUILD_DIR)
