from __future__ import annotations

import argparse
import math
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAUNCH_SCRIPT = REPO_ROOT / "deployments" / "binance-hybrid-spot" / "launch.sh"


@dataclass(frozen=True)
class BinanceHybridLaunchConfig:
    launch_script: str
    python_bin: str
    trade_script: str
    model: str
    symbols: list[str]
    execution_mode: str
    leverage: float
    interval: int | None
    fallback_mode: str
    rl_checkpoint: str | None


@dataclass(frozen=True)
class BinanceHybridEvalConstraints:
    max_leverage: float
    tradable_symbols_csv: str
    disable_shorts: bool


def parse_symbols_override(symbols_override: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if symbols_override is None:
        return None
    if isinstance(symbols_override, str):
        parts = re.split(r"[\s,]+", symbols_override.strip())
        normalized = [part.upper() for part in parts if part]
        return normalized or None
    normalized = [str(symbol).strip().upper() for symbol in symbols_override if str(symbol).strip()]
    return normalized or None


def parse_leverage_override(leverage_override: str | float | int | None) -> float | None:
    if leverage_override is None:
        return None
    if isinstance(leverage_override, str):
        raw = leverage_override.strip()
        if not raw:
            return None
        value = float(raw)
    else:
        value = float(leverage_override)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("leverage override must be a finite value greater than 0")
    return value


def _extract_exec_tokens(text: str) -> list[str]:
    parts: list[str] = []
    collecting = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not collecting:
            if not line.startswith("exec "):
                continue
            collecting = True
            line = line[len("exec ") :].strip()
        line = re.sub(r"\s+#.*$", "", line).rstrip()
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        if line:
            parts.append(line)
        if collecting and not continued:
            break
    if not parts:
        raise ValueError("Unable to find exec command in launch script")
    tokens = shlex.split(" ".join(parts))
    return [token for token in tokens if token != "$@"]


def parse_launch_script(
    path: str | Path,
    *,
    require_rl_checkpoint: bool = True,
) -> BinanceHybridLaunchConfig:
    launch_path = Path(path)
    tokens = _extract_exec_tokens(launch_path.read_text())
    script_index = next((i for i, token in enumerate(tokens) if token.endswith("trade_binance_live.py")), None)
    if script_index is None:
        raise ValueError(f"Unable to find trade_binance_live.py in {launch_path}")

    python_bin = tokens[0]
    trade_script = tokens[script_index]
    cli_tokens = tokens[script_index + 1 :]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--execution-mode", default="spot")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--fallback-mode", default="")
    parser.add_argument("--rl-checkpoint", default="")
    args, _unknown = parser.parse_known_args(cli_tokens)

    if not args.live:
        raise ValueError(f"{launch_path} is not configured with --live")
    if not args.symbols:
        raise ValueError(f"{launch_path} does not define --symbols")
    if not args.rl_checkpoint and require_rl_checkpoint:
        raise ValueError(f"{launch_path} does not define --rl-checkpoint")

    checkpoint_path: Path | None = None
    if args.rl_checkpoint:
        checkpoint_path = Path(args.rl_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (launch_path.parent / checkpoint_path).resolve()

    return BinanceHybridLaunchConfig(
        launch_script=str(launch_path.resolve()),
        python_bin=python_bin,
        trade_script=trade_script,
        model=str(args.model),
        symbols=[str(symbol).upper() for symbol in args.symbols],
        execution_mode=str(args.execution_mode),
        leverage=float(args.leverage),
        interval=args.interval,
        fallback_mode=str(args.fallback_mode),
        rl_checkpoint=str(checkpoint_path) if checkpoint_path is not None else None,
    )


def resolve_target_launch_config(
    launch_script: str | Path,
    *,
    symbols_override: str | list[str] | tuple[str, ...] | None = None,
    leverage_override: str | float | int | None = None,
) -> BinanceHybridLaunchConfig:
    launch_cfg = parse_launch_script(launch_script, require_rl_checkpoint=False)
    override_symbols = parse_symbols_override(symbols_override)
    override_leverage = parse_leverage_override(leverage_override)
    if not override_symbols and override_leverage is None:
        return launch_cfg
    return replace(
        launch_cfg,
        symbols=override_symbols if override_symbols else launch_cfg.symbols,
        leverage=override_leverage if override_leverage is not None else launch_cfg.leverage,
    )


def resolve_launch_eval_constraints(
    *,
    prod_launch_script: str | Path,
    eval_max_leverage: float | None = None,
    eval_tradable_symbols: str = "",
    eval_disable_shorts: bool | None = None,
) -> BinanceHybridEvalConstraints:
    launch_script_str = str(prod_launch_script).strip()
    launch_cfg = (
        parse_launch_script(prod_launch_script, require_rl_checkpoint=False)
        if launch_script_str
        else None
    )
    launch_matches_default_prod = (
        launch_cfg is not None
        and Path(launch_script_str).resolve() == Path(DEFAULT_LAUNCH_SCRIPT).resolve()
    )
    resolved_max_leverage = (
        float(eval_max_leverage)
        if eval_max_leverage is not None
        else (float(launch_cfg.leverage) if launch_cfg is not None else 1.0)
    )
    resolved_tradable_symbols = str(eval_tradable_symbols).strip()
    if not resolved_tradable_symbols and launch_cfg is not None:
        resolved_tradable_symbols = ",".join(launch_cfg.symbols)
    resolved_disable_shorts = (
        bool(eval_disable_shorts)
        if eval_disable_shorts is not None
        else bool(launch_matches_default_prod)
    )
    return BinanceHybridEvalConstraints(
        max_leverage=resolved_max_leverage,
        tradable_symbols_csv=resolved_tradable_symbols,
        disable_shorts=resolved_disable_shorts,
    )
