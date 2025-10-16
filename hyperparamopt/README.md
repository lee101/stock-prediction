Hyperparamopt — LLM-Suggested Hyperparameters with Structured Outputs

What it does
- Logs each run’s hyperparameters and outcomes to `hyperparamopt/logs/runs.jsonl`.
- Uses OpenAI `gpt-5-mini` with JSON schema structured outputs to propose the next hyperparameters.
- CLI for logging runs, fetching best run, and requesting suggestions.

Quick start
1) Export your API key
   - `export OPENAI_API_KEY=...`

2) Create a JSON schema describing one suggestion (example `schema.json`):
```
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "max_positions": {"type": "integer", "minimum": 1, "maximum": 10},
    "rebalance_frequency": {"type": "integer", "enum": [1, 3, 5, 7]},
    "min_expected_return": {"type": "number", "minimum": 0.0, "maximum": 0.2},
    "position_sizing_method": {"type": "string", "enum": ["equal_weight", "return_weighted"]}
  },
  "required": ["max_positions", "rebalance_frequency", "min_expected_return", "position_sizing_method"]
}
```

3) Log runs as you evaluate them:
```
python -m hyperparamopt.runner log-run \
  --params '{"max_positions":3, "rebalance_frequency":3, "min_expected_return":0.02, "position_sizing_method":"equal_weight"}' \
  --metrics '{"sharpe": 1.12, "return": 0.18}' \
  --score 1.12 \
  --objective "maximize_sharpe"
```

4) Ask for the next suggestions (n=3):
```
python -m hyperparamopt.runner suggest \
  --schema schema.json \
  --objective "maximize_sharpe" \
  --guidance "Respect capital constraints and prefer low turnover." \
  -n 3
```

5) Inspect the current best run:
```
python -m hyperparamopt.runner best --objective maximize_sharpe
```

Notes
- Requires `openai>=1.0.0`. Add to `requirements.txt` and install.
- All runs for a given objective are kept together in the JSONL log; you can keep multiple objectives (e.g., `maximize_sharpe`, `maximize_return`, `minimize_volatility`).
- The suggestions response is strictly validated against your JSON schema by the OpenAI structured outputs API.

