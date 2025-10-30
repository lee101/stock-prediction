RUN_DIR ?= runs
SUMMARY_GLOB ?= $(RUN_DIR)/*_summary.json
LOG_GLOB ?= $(RUN_DIR)/*.log
CI_SIM_PREFIX ?= $(shell date -u +%Y%m%d-%H%M%S)
TREND_HISTORY ?= marketsimulator/run_logs/trend_history.csv
TREND_STATUS_HISTORY ?= marketsimulator/run_logs/trend_status_history.json
TREND_PAUSED_LOG ?= marketsimulator/run_logs/trend_paused_escalations.csv
ROTATION_STREAK_THRESHOLD ?= 8
ROTATION_CANDIDATE_SMA ?= 500
MARKETSIM_TREND_SUMMARY_PATH ?= marketsimulator/run_logs/trend_summary.json
MARKETSIM_TREND_PNL_SUSPEND_MAP ?= AAPL:-5000,AMZN:-400,SOXX:-150,NVDA:-1500
MARKETSIM_TREND_PNL_RESUME_MAP ?= AAPL:-3000,AMZN:-200,SOXX:-75,NVDA:-750

.PHONY: stub-run summarize metrics-csv metrics-check smoke

stub-run:
	@mkdir -p $(RUN_DIR)
	python tools/mock_stub_run.py --log $(RUN_DIR)/stub.log --summary $(RUN_DIR)/stub_summary.json

summarize:
	python tools/summarize_results.py --log-glob "$(LOG_GLOB)" --output marketsimulatorresults.md

metrics-csv:
	python tools/metrics_to_csv.py --input-glob "$(SUMMARY_GLOB)" --output $(RUN_DIR)/metrics.csv

metrics-check:
	python tools/check_metrics.py --glob "$(SUMMARY_GLOB)"

smoke:
	./scripts/metrics_smoke.sh $(RUN_DIR)/smoke

.PHONY: sim-report
sim-report:
	MARKETSIM_KELLY_DRAWDOWN_CAP=0.02 \
	MARKETSIM_KELLY_DRAWDOWN_CAP_MAP=NVDA@ci_guard:0.01 \
MARKETSIM_DRAWDOWN_SUSPEND_MAP=ci_guard:0.013,NVDA@ci_guard:0.003,MSFT@ci_guard:0.007,AAPL@ci_guard:0.0085 \
MARKETSIM_DRAWDOWN_RESUME_MAP=ci_guard:0.005,NVDA@ci_guard:0.00015,MSFT@ci_guard:0.002,AAPL@ci_guard:0.002 \
MARKETSIM_SYMBOL_SIDE_MAP=NVDA:sell \
MARKETSIM_SYMBOL_KELLY_SCALE_MAP=AAPL:0.2,MSFT:0.25,NVDA:0.01,AMZN:0.15,SOXX:0.15 \
MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP=AAPL:10800,MSFT:10800,NVDA:7200,AMZN:10800,SOXX:10800 \
MARKETSIM_SYMBOL_MIN_COOLDOWN_MAP=NVDA:360 \
MARKETSIM_SYMBOL_FORCE_PROBE_MAP=AAPL:true \
MARKETSIM_SYMBOL_MIN_MOVE_MAP=AAPL:0.08,AMZN:0.06,SOXX:0.04 \
MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP=AAPL:-0.03,AMZN:-0.02,SOXX:0.015 \
MARKETSIM_SYMBOL_MAX_ENTRIES_MAP=NVDA:1,MSFT:10,AAPL:10,AMZN:8,SOXX:6 \
MARKETSIM_TREND_SUMMARY_PATH=marketsimulator/run_logs/trend_summary.json \
MARKETSIM_TREND_PNL_SUSPEND_MAP=AAPL:-5000,AMZN:-400,SOXX:-150,NVDA:-1500 \
MARKETSIM_TREND_PNL_RESUME_MAP=AAPL:-3000,AMZN:-200,SOXX:-75,NVDA:-750 \
	python scripts/run_sim_with_report.py \
		--prefix $(CI_SIM_PREFIX) \
		--max-fee-bps 25 \
		--max-avg-slip 100 \
		--max-drawdown-pct 5 \
		--min-final-pnl -2000 \
		--max-worst-cash -40000 \
		--max-trades-map NVDA@ci_guard:2,MSFT@ci_guard:16,AAPL@ci_guard:20 \
		--fail-on-alert -- \
		python marketsimulator/run_trade_loop.py \
		--symbols AAPL MSFT NVDA AMZN SOXX \
		--steps 20 --step-size 1 \
		--initial-cash 100000 --kronos-only \
		--flatten-end --kronos-sharpe-cutoff -1.0
	python scripts/report_trend_gating.py --alert --summary --history "$(TREND_STATUS_HISTORY)" --paused-log "$(TREND_PAUSED_LOG)" --suspend-map "$(MARKETSIM_TREND_PNL_SUSPEND_MAP)" --resume-map "$(MARKETSIM_TREND_PNL_RESUME_MAP)"
	python scripts/trend_candidate_report.py --auto-threshold --sma-threshold $${SMA_THRESHOLD:-0}

.PHONY: trend-status
trend-status:
	python scripts/report_trend_gating.py --alert --summary --history "$(TREND_STATUS_HISTORY)" --paused-log "$(TREND_PAUSED_LOG)" --suspend-map "$(MARKETSIM_TREND_PNL_SUSPEND_MAP)" --resume-map "$(MARKETSIM_TREND_PNL_RESUME_MAP)"
	python scripts/trend_candidate_report.py --auto-threshold --sma-threshold $${SMA_THRESHOLD:-0}
	python scripts/trend_candidate_report.py --sma-threshold $${SMA_THRESHOLD:-300}
	python scripts/rotation_recommendations.py --paused-log "$(TREND_PAUSED_LOG)" --trend-summary "$(MARKETSIM_TREND_SUMMARY_PATH)" --streak-threshold $(ROTATION_STREAK_THRESHOLD) --candidate-sma $(ROTATION_CANDIDATE_SMA) --log-output marketsimulator/run_logs/rotation_recommendations.log
	python scripts/generate_rotation_markdown.py --input marketsimulator/run_logs/rotation_recommendations.log --output marketsimulator/run_logs/rotation_summary.md --streak-threshold $(ROTATION_STREAK_THRESHOLD) --latency-json marketsimulator/run_logs/provider_latency_rolling.json --latency-png marketsimulator/run_logs/provider_latency_history.png --latency-digest marketsimulator/run_logs/provider_latency_alert_digest.md --latency-leaderboard marketsimulator/run_logs/provider_latency_leaderboard.md

.PHONY: trend-pipeline
trend-pipeline:
	python scripts/run_daily_trend_pipeline.py

.PHONY: sim-trend
sim-trend:
	python scripts/trend_analyze_trade_summaries.py \
		marketsimulator/run_logs/*_trades_summary.json \
		--json-out marketsimulator/run_logs/trend_summary.json
	python scripts/append_trend_history.py \
		marketsimulator/run_logs/trend_summary.json \
		$(TREND_HISTORY) \
		--symbols AAPL,MSFT,NVDA \
		--timestamp $$(date -u +%Y-%m-%dT%H:%M:%SZ)
LATENCY_SNAPSHOT ?= marketsimulator/run_logs/provider_latency_rolling.json

.PHONY: latency-status
latency-status:
	python scripts/provider_latency_status.py --snapshot $(LATENCY_SNAPSHOT)
.PHONY: fast-env-benchmark
fast-env-benchmark:
	. .venv/bin/activate && \
	python analysis/fast_env_benchmark.py \
		--symbol AAPL \
		--data-root trainingdata \
		--context-len 128 \
		--steps 2048 \
		--trials 3 \
		--output-json results/bench_fast_vs_python.json \
		--output-csv results/bench_fast_vs_python.csv
