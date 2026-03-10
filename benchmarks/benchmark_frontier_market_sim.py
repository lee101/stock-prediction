from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.frontiermarketsim import FrontierSimConfig, build_frontier_simulator_from_data


def _parse_symbols(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [part.strip().upper() for part in raw.split(",") if part.strip()]
    return values or None


def _load_symbols_file(path: str | None) -> list[str] | None:
    if path is None:
        return None
    lines = Path(path).read_text().splitlines()
    symbols = [line.strip().upper() for line in lines if line.strip() and not line.strip().startswith("#")]
    return symbols or None


def _merge_symbols(explicit: list[str] | None, from_file: list[str] | None) -> list[str] | None:
    if explicit is None and from_file is None:
        return None
    merged: list[str] = []
    seen = set()
    for source in (explicit or []) + (from_file or []):
        if source in seen:
            continue
        seen.add(source)
        merged.append(source)
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the frontier multi-symbol market simulator.")
    parser.add_argument("--data-root", type=str, default="trainingdata", help="Directory containing OHLCV CSVs.")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (e.g. AAPL,MSFT,NVDA).")
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        help="Path to newline-separated symbol list. Merged with --symbols.",
    )
    parser.add_argument("--max-symbols", type=int, default=64, help="Maximum symbols to load from data root.")
    parser.add_argument("--min-rows", type=int, default=1024, help="Minimum rows required per symbol.")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel env slots.")
    parser.add_argument("--steps", type=int, default=2000, help="Benchmark steps to execute.")
    parser.add_argument(
        "--backend",
        choices=["auto", "fast", "torch"],
        default="auto",
        help="Simulation backend preference.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device for simulator tensors.")
    parser.add_argument("--context-len", type=int, default=128, help="Observation context length.")
    parser.add_argument("--horizon", type=int, default=1, help="Execution horizon in bars.")
    parser.add_argument("--seed", type=int, default=1337, help="PRNG seed.")
    parser.add_argument("--json", type=str, default=None, help="Optional output JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    symbols = _merge_symbols(_parse_symbols(args.symbols), _load_symbols_file(args.symbols_file))
    cfg = FrontierSimConfig(
        context_len=args.context_len,
        horizon=args.horizon,
        seed=args.seed,
        device=args.device,
    )
    use_fast = args.backend != "torch"

    simulator = build_frontier_simulator_from_data(
        args.data_root,
        symbols=symbols,
        max_symbols=args.max_symbols,
        min_rows=args.min_rows,
        num_envs=args.num_envs,
        cfg=cfg,
        use_fast_backend=use_fast,
    )
    result = simulator.run_benchmark(num_steps=args.steps)
    if args.backend == "fast" and result["backend"] != "fast":
        reason = result.get("fast_backend_error") or "unknown initialization failure"
        raise RuntimeError(f"Requested fast backend but fell back to torch backend: {reason}")

    payload = {
        "config": {
            "data_root": args.data_root,
            "symbols": symbols,
            "max_symbols": args.max_symbols,
            "min_rows": args.min_rows,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "backend": args.backend,
            "device": args.device,
            "context_len": args.context_len,
            "horizon": args.horizon,
            "seed": args.seed,
        },
        "result": result,
    }
    formatted = json.dumps(payload, indent=2, sort_keys=True)
    print(formatted)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(formatted + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
