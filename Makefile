.PHONY: build clean test export train

PYTHON ?= python

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
