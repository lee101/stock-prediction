#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import os

from hyperparamopt.storage import RunLog, RunRecord
from hyperparamopt.optimizer import StructuredOpenAIOptimizer, SuggestionRequest


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    log = RunLog()

    # Seed a couple of example runs if log is empty
    if len(log.list()) == 0:
        log.append(RunRecord.new(
            params={"max_positions": 2, "max_position_size": 0.47, "rebalance_frequency": 1, "min_expected_return": 0.00, "position_sizing_method": "equal_weight"},
            metrics={"sharpe": 0.9, "return": 0.15},
            score=0.9,
            objective="maximize_sharpe",
            source="seed",
        ))
        log.append(RunRecord.new(
            params={"max_positions": 3, "max_position_size": 0.32, "rebalance_frequency": 3, "min_expected_return": 0.02, "position_sizing_method": "equal_weight"},
            metrics={"sharpe": 1.1, "return": 0.18},
            score=1.1,
            objective="maximize_sharpe",
            source="seed",
        ))

    schema_path = Path(__file__).parent / "schema_trading.json"
    schema = json.loads(schema_path.read_text())

    opt = StructuredOpenAIOptimizer(run_log=log)
    req = SuggestionRequest(
        hyperparam_schema=schema,
        objective="maximize_sharpe",
        guidance="Respect typical portfolio constraints; avoid too frequent rebalances.",
        n=3,
        history_limit=100,
        model="gpt5-mini",
    )

    res = opt.suggest(req)
    print(json.dumps(res.suggestions, indent=2))


if __name__ == "__main__":
    main()

