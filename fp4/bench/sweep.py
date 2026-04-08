"""Multi-seed sweep harness for the fp4 trainer bake-off (Phase 4, Unit P4-6).

Runs the cartesian product

    algos        x  constrained  x  seeds
    {ppo,sac,qr_ppo} x {on,off}  x  {0..4}

against ``fp4/bench/bench_trading.py`` as a subprocess, one cell at a time
(no nested parallelism -- other Phase 4 agents are doing the parallel work).
Aggregates per-cell metrics from the JSON written by ``bench_trading.py`` and
produces ``summary.csv`` + ``leaderboard.md`` under
``fp4/bench/results/sweep_<date>/``.

Trainer name mapping (since P4-4/P4-5 wire the new trainers into
``bench_trading.py`` as ``sac``/``qr_ppo`` choices): ``ppo`` dispatches to the
existing ``fp4`` trainer (the PPO built in Unit A); ``sac`` and ``qr_ppo``
dispatch to whatever ``bench_trading.py`` knows about at run-time. If the
trainer isn't wired up yet, the subprocess exits non-zero or returns a
``status=skip`` record and we note the skip reason in the summary rather than
aborting the whole sweep.

Constrained mode is enabled by setting ``FP4_CONSTRAINED=1`` in the subprocess
environment. Current trainers ignore the flag; future versions (P4-3 loss
module + trainers rewired) can read it. This keeps backwards compat while
letting the sweep already log both columns.

Usage:

    python fp4/bench/sweep.py                        # full sweep
    python fp4/bench/sweep.py --smoke                # 1 seed x 50k steps/cell
    python fp4/bench/sweep.py --algos ppo,sac        # subset
    python fp4/bench/sweep.py --seeds 0,1 --steps 1000000
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent / "bench_trading.py"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"

ALGOS = ("ppo", "sac", "qr_ppo")
ALGO_TO_TRAINER = {"ppo": "fp4", "sac": "sac", "qr_ppo": "qr_ppo"}
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_STEPS = 2_000_000
SMOKE_STEPS = 50_000
TARGET_BPS = "5"  # aggregate at 5bps per spec


@dataclass
class CellResult:
    algo: str
    constrained: bool
    seed: int
    steps: int
    status: str
    reason: str
    wall_sec: float
    sps: float
    median_5bps: float
    p10_5bps: float
    sortino_5bps: float
    max_dd_5bps: float
    n_neg_5bps: float
    result_path: str

    def to_row(self) -> dict[str, Any]:
        return {
            "algo": self.algo,
            "constrained": "on" if self.constrained else "off",
            "seed": self.seed,
            "steps": self.steps,
            "status": self.status,
            "reason": self.reason,
            "wall_sec": f"{self.wall_sec:.3f}",
            "sps": f"{self.sps:.2f}",
            "median_5bps": _fmt(self.median_5bps),
            "p10_5bps": _fmt(self.p10_5bps),
            "sortino_5bps": _fmt(self.sortino_5bps),
            "max_dd_5bps": _fmt(self.max_dd_5bps),
            "n_neg_5bps": _fmt(self.n_neg_5bps),
            "result_path": self.result_path,
        }


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.6g}"


def _nan() -> float:
    return float("nan")


def _parse_list(s: str, cast=str) -> list:
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def _extract_metrics(rec: dict[str, Any]) -> dict[str, float]:
    """Pull median/p10/sortino/max_dd/n_neg at 5bps from a bench_trading JSON.

    Tries the nested ``eval.by_slippage["5"]`` path first (full eval present),
    then falls back to ``train.trainer_output`` fields (training-time summary,
    which at least has sortino + p10 for the fp4 trainer).
    """
    out = {
        "median_5bps": _nan(),
        "p10_5bps": _nan(),
        "sortino_5bps": _nan(),
        "max_dd_5bps": _nan(),
        "n_neg_5bps": _nan(),
    }
    ev = rec.get("eval") or {}
    by = (ev.get("by_slippage") or {}) if isinstance(ev, dict) else {}
    row = by.get(TARGET_BPS) or by.get(int(TARGET_BPS))
    if isinstance(row, dict):
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else row
        for k_json, k_out in (
            ("median_return", "median_5bps"),
            ("median_total_return", "median_5bps"),
            ("p10_return", "p10_5bps"),
            ("p10_total_return", "p10_5bps"),
            ("sortino", "sortino_5bps"),
            ("median_sortino", "sortino_5bps"),
            ("max_drawdown", "max_dd_5bps"),
            ("median_max_drawdown", "max_dd_5bps"),
            ("n_neg", "n_neg_5bps"),
        ):
            v = summary.get(k_json) if isinstance(summary, dict) else None
            if v is None:
                v = row.get(k_json)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[k_out] = float(v)
    # Fallback: trainer_output training-time metrics.
    trainer_out = (((rec.get("train") or {}).get("trainer_output")) or {})
    if math.isnan(out["sortino_5bps"]):
        v = trainer_out.get("final_sortino")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out["sortino_5bps"] = float(v)
    if math.isnan(out["p10_5bps"]):
        v = trainer_out.get("final_p10")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out["p10_5bps"] = float(v)
    if math.isnan(out["median_5bps"]):
        v = trainer_out.get("mean_return")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out["median_5bps"] = float(v)
    return out


def _query_free_gpu_mb() -> Optional[int]:
    """Return min free MB across visible GPUs via nvidia-smi, or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=4,
        )
    except Exception:
        return None
    vals: list[int] = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(int(line))
        except ValueError:
            continue
    if not vals:
        return None
    return min(vals)


def _wait_for_gpu_mem(min_free_mb: int, timeout_sec: float = 600.0,
                      poll_sec: float = 5.0) -> bool:
    """Block until ≥min_free_mb is free on every visible GPU. Returns False on timeout."""
    if min_free_mb <= 0:
        return True
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        free = _query_free_gpu_mb()
        if free is None:
            # nvidia-smi unavailable → don't block.
            return True
        if free >= min_free_mb:
            return True
        time.sleep(poll_sec)
    return False


def _run_cell(
    algo: str,
    constrained: bool,
    seed: int,
    steps: int,
    sweep_dir: Path,
    min_free_gpu_mb: int = 0,
    oom_retries: int = 1,
) -> CellResult:
    trainer = ALGO_TO_TRAINER[algo]
    env = os.environ.copy()
    env["FP4_CONSTRAINED"] = "1" if constrained else "0"
    env["PYTHONPATH"] = str(REPO) + ":" + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        str(BENCH),
        "--trainer", trainer,
        "--steps", str(int(steps)),
        "--seed", str(int(seed)),
    ]

    # Block until enough GPU memory is free so we don't immediately OOM
    # against a sibling process. No-op if min_free_gpu_mb <= 0.
    if min_free_gpu_mb > 0:
        ok = _wait_for_gpu_mem(min_free_gpu_mb)
        if not ok:
            return CellResult(
                algo=algo, constrained=constrained, seed=seed, steps=steps,
                status="error",
                reason=f"timed out waiting for {min_free_gpu_mb}MB free GPU mem",
                wall_sec=0.0, sps=0.0,
                median_5bps=_nan(), p10_5bps=_nan(), sortino_5bps=_nan(),
                max_dd_5bps=_nan(), n_neg_5bps=_nan(), result_path="",
            )

    proc = None
    last_err = ""
    attempts = max(1, int(oom_retries) + 1)
    for attempt in range(attempts):
        try:
            proc = subprocess.run(
                cmd, cwd=str(REPO), env=env, capture_output=True, text=True
            )
        except Exception as exc:  # pragma: no cover - defensive
            return CellResult(
                algo=algo, constrained=constrained, seed=seed, steps=steps,
                status="error", reason=f"subprocess raised: {type(exc).__name__}: {exc}",
                wall_sec=0.0, sps=0.0,
                median_5bps=_nan(), p10_5bps=_nan(), sortino_5bps=_nan(),
                max_dd_5bps=_nan(), n_neg_5bps=_nan(), result_path="",
            )
        # Detect "CUDA out of memory" and back off + retry once.
        stderr_tail = (proc.stderr or "")[-2000:]
        oom = ("out of memory" in stderr_tail) or ("CUDA error: out of memory" in stderr_tail)
        if not oom:
            break
        last_err = stderr_tail
        if attempt + 1 >= attempts:
            break
        # Wait for the contending process to release memory before retrying.
        backoff = 30.0 + 30.0 * attempt
        time.sleep(backoff)
        if min_free_gpu_mb > 0:
            _wait_for_gpu_mem(min_free_gpu_mb)
    assert proc is not None

    # bench_trading prints a single JSON doc as its last stdout blob. Locate it.
    rec: dict[str, Any] = {}
    stdout = proc.stdout or ""
    try:
        # Typical: the whole stdout is a JSON doc.
        rec = json.loads(stdout)
    except Exception:
        # Fallback: slice from the first '{' we find.
        i = stdout.find("{")
        if i >= 0:
            try:
                rec = json.loads(stdout[i:])
            except Exception:
                rec = {}

    result_path = rec.get("result_path", "")
    # If we have a path, re-read the full record (stdout may have dropped 'train').
    if result_path:
        try:
            rec_full = json.loads(Path(result_path).read_text())
            rec = rec_full
        except Exception:
            pass

    status = rec.get("status") or ("ok" if proc.returncode == 0 else "error")
    reason = ""
    if status != "ok":
        reason = (rec.get("reason")
                  or (rec.get("train") or {}).get("reason")
                  or (proc.stderr or "")[-500:]
                  or "unknown")
    wall = float((rec.get("train") or {}).get("wall_sec") or 0.0)
    sps = float(rec.get("steps_per_sec") or 0.0)
    metrics = _extract_metrics(rec)

    # Mirror the per-cell JSON into sweep_dir for reproducibility.
    cell_out = sweep_dir / f"{algo}_c{'on' if constrained else 'off'}_s{seed}.json"
    try:
        cell_out.write_text(json.dumps({
            "algo": algo, "constrained": constrained, "seed": seed, "steps": steps,
            "status": status, "reason": reason, "returncode": proc.returncode,
            "bench_record": rec,
        }, indent=2, default=str))
    except Exception:
        pass

    return CellResult(
        algo=algo, constrained=constrained, seed=seed, steps=steps,
        status=status, reason=str(reason)[:300],
        wall_sec=wall, sps=sps,
        result_path=str(result_path),
        **metrics,
    )


def _write_summary_csv(path: Path, cells: list[CellResult]) -> None:
    fields = [
        "algo", "constrained", "seed", "steps", "status", "reason",
        "wall_sec", "sps",
        "median_5bps", "p10_5bps", "sortino_5bps", "max_dd_5bps", "n_neg_5bps",
        "result_path",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in cells:
            w.writerow(c.to_row())


def _agg(values: Iterable[float]) -> tuple[float, float, int]:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not xs:
        return _nan(), _nan(), 0
    if len(xs) == 1:
        return xs[0], 0.0, 1
    return statistics.mean(xs), statistics.stdev(xs), len(xs)


def _write_leaderboard(path: Path, cells: list[CellResult], *, steps: int, smoke: bool) -> None:
    # Collapse over seeds per (algo, constrained) cell.
    groups: dict[tuple[str, bool], list[CellResult]] = {}
    for c in cells:
        groups.setdefault((c.algo, c.constrained), []).append(c)

    rows = []
    for (algo, constrained), cs in groups.items():
        n_ok = sum(1 for c in cs if c.status == "ok")
        med_mean, med_std, _ = _agg(c.median_5bps for c in cs)
        p10_mean, p10_std, _ = _agg(c.p10_5bps for c in cs)
        sor_mean, sor_std, _ = _agg(c.sortino_5bps for c in cs)
        dd_mean, dd_std, _ = _agg(c.max_dd_5bps for c in cs)
        sps_mean, _, _ = _agg(c.sps for c in cs)
        rows.append({
            "algo": algo,
            "constrained": "on" if constrained else "off",
            "n_seeds": len(cs),
            "n_ok": n_ok,
            "median_5bps": (med_mean, med_std),
            "p10_5bps": (p10_mean, p10_std),
            "sortino_5bps": (sor_mean, sor_std),
            "max_dd_5bps": (dd_mean, dd_std),
            "sps": sps_mean,
        })

    def _sort_key(r):
        p10 = r["p10_5bps"][0]
        sor = r["sortino_5bps"][0]
        dd = r["max_dd_5bps"][0]
        # Put NaNs last.
        return (
            -(p10 if math.isfinite(p10) else -1e18),
            -(sor if math.isfinite(sor) else -1e18),
            (dd if math.isfinite(dd) else 1e18),
        )

    rows.sort(key=_sort_key)

    def ms(pair):
        m, s = pair
        if not math.isfinite(m):
            return "—"
        return f"{m:+.4g} ± {s:.3g}" if s and math.isfinite(s) else f"{m:+.4g}"

    lines = []
    lines.append(f"# fp4 sweep leaderboard ({_dt.datetime.now().isoformat(timespec='seconds')})")
    lines.append("")
    lines.append(f"- mode: {'SMOKE' if smoke else 'full'}")
    lines.append(f"- steps/cell: {steps}")
    lines.append(f"- cells: {len(groups)}  total runs: {len(cells)}")
    lines.append(f"- sort key: (p10@5bps DESC, sortino@5bps DESC, max_dd ASC)")
    lines.append("")
    lines.append("| rank | algo | constrained | seeds | ok | p10@5bps | sortino@5bps | median@5bps | max_dd@5bps | sps |")
    lines.append("|---:|:---|:---:|:---:|:---:|:---|:---|:---|:---|---:|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r['algo']} | {r['constrained']} | {r['n_seeds']} | {r['n_ok']} | "
            f"{ms(r['p10_5bps'])} | {ms(r['sortino_5bps'])} | {ms(r['median_5bps'])} | "
            f"{ms(r['max_dd_5bps'])} | {r['sps']:.0f} |"
        )
    lines.append("")
    # Include a per-seed appendix for auditing.
    lines.append("## Per-seed runs")
    lines.append("")
    lines.append("| algo | constrained | seed | status | p10@5bps | sortino@5bps | median@5bps | max_dd@5bps | sps | reason |")
    lines.append("|:---|:---:|:---:|:---|:---|:---|:---|:---|---:|:---|")
    for c in cells:
        lines.append(
            f"| {c.algo} | {'on' if c.constrained else 'off'} | {c.seed} | {c.status} | "
            f"{_fmt(c.p10_5bps) or '—'} | {_fmt(c.sortino_5bps) or '—'} | "
            f"{_fmt(c.median_5bps) or '—'} | {_fmt(c.max_dd_5bps) or '—'} | "
            f"{c.sps:.0f} | {c.reason[:80]} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def run_sweep(
    algos: list[str],
    constrained_modes: list[bool],
    seeds: list[int],
    steps: int,
    smoke: bool,
    sweep_dir: Path,
    min_free_gpu_mb: int = 0,
    oom_retries: int = 1,
) -> tuple[Path, Path, list[CellResult]]:
    sweep_dir.mkdir(parents=True, exist_ok=True)
    cells: list[CellResult] = []
    total = len(algos) * len(constrained_modes) * len(seeds)
    idx = 0
    for algo in algos:
        for constrained in constrained_modes:
            for seed in seeds:
                idx += 1
                tag = f"[{idx}/{total}] algo={algo} constrained={'on' if constrained else 'off'} seed={seed}"
                print(f">>> {tag}", flush=True)
                c = _run_cell(
                    algo, constrained, seed, steps, sweep_dir,
                    min_free_gpu_mb=min_free_gpu_mb, oom_retries=oom_retries,
                )
                print(
                    f"    status={c.status} sps={c.sps:.0f} "
                    f"p10={_fmt(c.p10_5bps) or '—'} sortino={_fmt(c.sortino_5bps) or '—'}"
                    + (f"  reason={c.reason[:120]}" if c.status != "ok" else ""),
                    flush=True,
                )
                cells.append(c)

    csv_path = sweep_dir / "summary.csv"
    md_path = sweep_dir / "leaderboard.md"
    _write_summary_csv(csv_path, cells)
    _write_leaderboard(md_path, cells, steps=steps, smoke=smoke)
    return csv_path, md_path, cells


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="fp4 multi-seed sweep harness")
    p.add_argument("--algos", default=",".join(ALGOS),
                   help=f"comma list of {ALGOS}")
    p.add_argument("--constrained", default="both",
                   choices=("on", "off", "both"))
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--smoke", action="store_true",
                   help="1 seed x 50k steps per cell, quick sanity")
    p.add_argument("--out-dir", default="",
                   help="override sweep output dir (default sweep_<date>)")
    p.add_argument("--min-free-gpu-mb", type=int, default=4096,
                   help="Block before launching each cell until at least N MB "
                        "of GPU memory are free (default 4096). Set 0 to disable.")
    p.add_argument("--oom-retries", type=int, default=1,
                   help="Retry a cell once after a CUDA OOM with backoff "
                        "(default 1; set 0 to fail immediately).")
    args = p.parse_args(argv)

    algos = _parse_list(args.algos)
    for a in algos:
        if a not in ALGO_TO_TRAINER:
            p.error(f"unknown algo: {a!r} (choices: {list(ALGO_TO_TRAINER)})")

    if args.constrained == "on":
        cmodes = [True]
    elif args.constrained == "off":
        cmodes = [False]
    else:
        cmodes = [False, True]

    seeds = [int(s) for s in _parse_list(args.seeds)]
    steps = int(args.steps)

    if args.smoke:
        seeds = seeds[:1] or [0]
        steps = SMOKE_STEPS

    date = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / f"sweep_{date}"

    csv_path, md_path, cells = run_sweep(
        algos=algos, constrained_modes=cmodes, seeds=seeds,
        steps=steps, smoke=args.smoke, sweep_dir=sweep_dir,
        min_free_gpu_mb=int(args.min_free_gpu_mb),
        oom_retries=int(args.oom_retries),
    )
    n_ok = sum(1 for c in cells if c.status == "ok")
    print(f"\nsweep complete: {n_ok}/{len(cells)} ok")
    print(f"  summary : {csv_path}")
    print(f"  leaderboard: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
