.PHONY: build clean test export train fast-env-benchmark sim-report sim-trend

PYTHON ?= python

SIM_TIMESTAMP := $(shell date -u +%Y%m%d-%H%M%S)
CI_SIM_PREFIX ?= $(SIM_TIMESTAMP)

FAST_ENV_BENCH_SYMBOLS ?= AAPL MSFT NVDA
FAST_ENV_BENCH_DATA_ROOT ?= trainingdata
FAST_ENV_BENCH_STEPS ?= 256
FAST_ENV_BENCH_CONTEXT ?= 64
FAST_ENV_BENCH_HORIZON ?= 1
FAST_ENV_BENCH_SEED ?= 1337
FAST_ENV_BENCH_DEVICE ?= cpu
FAST_ENV_BENCH_OUTPUT ?= results/bench_fast_vs_python

SIM_SYMBOLS ?= AAPL MSFT NVDA AMZN GOOG XLK SOXX
SIM_STEPS ?= 36
SIM_STEP_SIZE ?= 1
SIM_TOP_K ?= 5
SIM_INITIAL_CASH ?= 100000
SIM_FAST_SIM ?= 1
SIM_KRONOS_ONLY ?= 1
SIM_FLATTEN_END ?= 1
SIM_MAX_FEE_BPS ?= 25
SIM_MAX_AVG_SLIP ?= 100
SIM_MAX_DRAWDOWN_PCT ?= 2.0
SIM_MIN_FINAL_PNL ?= -6000
SIM_MAX_WORST_CASH ?= -8000
SIM_MIN_SYMBOL_PNL ?= -4000
SIM_MAX_TRADES ?=
SIM_MAX_TRADES_MAP ?= NVDA@ci_guard:4,MSFT@ci_guard:12,AAPL@ci_guard:20,AMZN@ci_guard:12,GOOG@ci_guard:12,XLK@ci_guard:12,SOXX@ci_guard:12
SIM_USE_STUB ?= 0
SIM_TREND_JSON ?= marketsimulator/run_logs/trend_summary.json
SIM_TREND_HISTORY ?= marketsimulator/run_logs/trend_history.csv
SIM_TREND_WINDOW ?= 12
SIM_TREND_TOP ?= 20
SIM_TREND_SYMBOLS ?= AAPL,MSFT,NVDA,AMZN,GOOG,XLK,SOXX
SIM_TREND_TIMESTAMP ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
SIM_TRADE_GLOB ?= marketsimulator/run_logs/*_trades_summary.json
SIM_BOOL_TRUE := 1 true TRUE yes YES on ON
SIM_FAST_SIM_OPT := $(if $(filter $(SIM_FAST_SIM),$(SIM_BOOL_TRUE)),--fast-sim,)
SIM_KRONOS_ONLY_OPT := $(if $(filter $(SIM_KRONOS_ONLY),$(SIM_BOOL_TRUE)),--kronos-only,)
SIM_FLATTEN_END_OPT := $(if $(filter $(SIM_FLATTEN_END),$(SIM_BOOL_TRUE)),--flatten-end,)
SIM_STUB_OPT := $(if $(filter $(SIM_USE_STUB),$(SIM_BOOL_TRUE)),--stub-config,)

export PYTHONPATH := .:$(PYTHONPATH)

build:
	cd .. && $(PYTHON) pufferlib_market/setup.py build_ext --inplace

clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "*.so" -delete
	find . -name "*.o" -delete

export:
	cd .. && $(PYTHON) pufferlib_market/export_data.py \
		--symbols SOLUSD,LINKUSD,UNIUSD,BTCUSD,ETHUSD,NVDA,NFLX,AAPL,AMZN,META,TSLA,JPM,V,WMT \
		--forecast-cache-root alpacanewccrosslearning/forecast_cache/mixed14_robust_20260205_2301_lb4000 \
		--data-root trainingdatahourly \
		--output pufferlib_market/data/market_data.bin

train:
	cd .. && $(PYTHON) pufferlib_market/train.py

test:
	cd .. && $(PYTHON) -c "import pufferlib_market.binding; print('binding OK')"

fast-env-benchmark:
	PYTHONPATH=. $(PYTHON) scripts/fast_env_benchmark.py \
		--symbols $(FAST_ENV_BENCH_SYMBOLS) \
		--data-root $(FAST_ENV_BENCH_DATA_ROOT) \
		--context-len $(FAST_ENV_BENCH_CONTEXT) \
		--horizon $(FAST_ENV_BENCH_HORIZON) \
		--steps $(FAST_ENV_BENCH_STEPS) \
		--seed $(FAST_ENV_BENCH_SEED) \
		--device $(FAST_ENV_BENCH_DEVICE) \
		--output-prefix $(FAST_ENV_BENCH_OUTPUT)

sim-report:
	$(PYTHON) scripts/run_sim_with_report.py \
		--prefix $(CI_SIM_PREFIX) \
		--max-fee-bps $(SIM_MAX_FEE_BPS) \
		--max-avg-slip $(SIM_MAX_AVG_SLIP) \
		--max-drawdown-pct $(SIM_MAX_DRAWDOWN_PCT) \
		--min-final-pnl $(SIM_MIN_FINAL_PNL) \
		--max-worst-cash $(SIM_MAX_WORST_CASH) \
		--min-symbol-pnl $(SIM_MIN_SYMBOL_PNL) \
		$(if $(SIM_MAX_TRADES),--max-trades $(SIM_MAX_TRADES),) \
		$(if $(SIM_MAX_TRADES_MAP),--max-trades-map "$(SIM_MAX_TRADES_MAP)",) \
		--fail-on-alert \
		-- \
		$(PYTHON) marketsimulator/run_trade_loop.py \
			--symbols $(SIM_SYMBOLS) \
			--steps $(SIM_STEPS) \
			--step-size $(SIM_STEP_SIZE) \
			--initial-cash $(SIM_INITIAL_CASH) \
			--top-k $(SIM_TOP_K) \
			$(SIM_KRONOS_ONLY_OPT) \
			$(SIM_FLATTEN_END_OPT) \
			$(SIM_FAST_SIM_OPT) \
			--compact-logs \
			--kronos-sharpe-cutoff -1.0 \
			--sharpe-cutoff -1.0 \
			$(SIM_STUB_OPT)

sim-trend:
	$(PYTHON) scripts/trend_analyze_trade_summaries.py \
		"$(SIM_TRADE_GLOB)" \
		--window $(SIM_TREND_WINDOW) \
		--top $(SIM_TREND_TOP) \
		--json-out $(SIM_TREND_JSON)
	$(PYTHON) scripts/append_trend_history.py \
		$(SIM_TREND_JSON) \
		$(SIM_TREND_HISTORY) \
		--timestamp "$(SIM_TREND_TIMESTAMP)" \
		--symbols "$(SIM_TREND_SYMBOLS)"
