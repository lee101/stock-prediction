#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import argparse
from typing import Any, Dict

from .storage import RunLog, RunRecord
from .optimizer import StructuredOpenAIOptimizer, SuggestionRequest


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cmd_log(args: argparse.Namespace) -> None:
    log = RunLog(args.log)
    params = json.loads(args.params)
    metrics = json.loads(args.metrics) if args.metrics else {}
    rec = RunRecord.new(
        params=params,
        metrics=metrics,
        score=float(args.score),
        objective=args.objective,
        source=args.source,
    )
    log.append(rec)
    print(f"Logged run id={rec.id} score={rec.score}")


def cmd_suggest(args: argparse.Namespace) -> None:
    log = RunLog(args.log)
    schema = load_json(Path(args.schema))
    opt = StructuredOpenAIOptimizer(run_log=log)
    req = SuggestionRequest(
        hyperparam_schema=schema,
        objective=args.objective,
        guidance=args.guidance,
        n=args.n,
        history_limit=args.history_limit,
        model=args.model,
    )
    res = opt.suggest(req)
    if args.out:
        Path(args.out).write_text(json.dumps(res.suggestions, indent=2))
        print(f"Wrote {len(res.suggestions)} suggestions to {args.out}")
    else:
        print(json.dumps(res.suggestions, indent=2))


def cmd_best(args: argparse.Namespace) -> None:
    log = RunLog(args.log)
    best = log.best(args.objective, maximize=(not args.minimize))
    if best is None:
        print("No runs found for the objective.")
        return
    print(json.dumps(best.__dict__, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hyperparam LLM optimizer")
    sub = p.add_subparsers(dest="cmd", required=True)

    # log
    p_log = sub.add_parser("log-run", help="Append a run to the log")
    p_log.add_argument("--params", required=True, help="JSON string of hyperparams")
    p_log.add_argument("--score", required=True, help="Objective score (float)")
    p_log.add_argument("--objective", required=True, help="Objective tag")
    p_log.add_argument("--metrics", help="JSON string of metrics")
    p_log.add_argument("--source", default="manual")
    p_log.add_argument("--log", default=str(Path("hyperparamopt/logs/runs.jsonl")))
    p_log.set_defaults(func=cmd_log)

    # suggest
    p_sug = sub.add_parser("suggest", help="Request next hyperparam suggestions via OpenAI")
    p_sug.add_argument("--schema", required=True, help="Path to JSON schema file for one suggestion")
    p_sug.add_argument("--objective", required=True)
    p_sug.add_argument("--guidance", default=None)
    p_sug.add_argument("-n", type=int, default=1)
    p_sug.add_argument("--history-limit", type=int, default=100)
    p_sug.add_argument("--model", default="gpt5-mini")
    p_sug.add_argument("--log", default=str(Path("hyperparamopt/logs/runs.jsonl")))
    p_sug.add_argument("--out", help="Optional output path for suggestions JSON")
    p_sug.set_defaults(func=cmd_suggest)

    # best
    p_best = sub.add_parser("best", help="Show best run for an objective")
    p_best.add_argument("--objective", required=True)
    p_best.add_argument("--minimize", action="store_true", help="Minimize instead of maximize")
    p_best.add_argument("--log", default=str(Path("hyperparamopt/logs/runs.jsonl")))
    p_best.set_defaults(func=cmd_best)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
