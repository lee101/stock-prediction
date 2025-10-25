RUN_DIR ?= runs
SUMMARY_GLOB ?= $(RUN_DIR)/*_summary.json
LOG_GLOB ?= $(RUN_DIR)/*.log

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
