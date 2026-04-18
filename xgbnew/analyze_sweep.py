#!/usr/bin/env python3
"""Post-hoc multi-testing correction for ``eval_multiwindow`` sweep JSONs.

Given an N-config sweep of a strategy over rolling OOS windows, a
naively-selected "best config" overstates its true skill. This script
applies Bailey & López de Prado's **Deflated Sharpe Ratio** (DSR)
correction to the best config's per-window return distribution.

The input JSON is what ``xgbnew/eval_multiwindow.py`` writes:
``analysis/.../multiwindow_*.json`` with ``sweep_results`` where each
entry has ``windows[*].monthly_return_pct``.

Reports, per sweep:
  - N trials, median / IQR across configs
  - best config's Sharpe (monthly), skew, kurtosis
  - implied SR threshold (SR*) given N trials
  - DSR = P(true SR > 0 | observed SR̂, N, sample moments)
  - pass/fail at DSR ≥ 0.95

Also reports per-seed dispersion if ``random_state`` is varied: median,
p10, p90, stdev of monthly PnL across seeds at the best hyperparam cell.

Usage:
    python xgbnew/analyze_sweep.py path/to/multiwindow_*.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

try:
    from scipy.stats import norm  # type: ignore
    _NORM_CDF = norm.cdf
    _NORM_PPF = norm.ppf
except Exception:  # pragma: no cover — fall back to math.erf ppf
    def _NORM_CDF(x: float) -> float:  # type: ignore[misc]
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))

    def _NORM_PPF(p: float) -> float:  # type: ignore[misc]
        # Beasley-Springer-Moro via Acklam rational approximation.
        a = [-3.969683028665376e01,  2.209460984245205e02,
             -2.759285104469687e02,  1.383577518672690e02,
             -3.066479806614716e01,  2.506628277459239e00]
        b = [-5.447609879822406e01,  1.615858368580409e02,
             -1.556989798598866e02,  6.680131188771972e01,
             -1.328068155288572e01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e00, -2.549732539343734e00,
              4.374664141464968e00,  2.938163982698783e00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,
              2.445134137142996e00,  3.754408661907416e00]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        if p <= phigh:
            q = p - 0.5
            r = q*q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)


EULER_MASCHERONI = 0.5772156649015329


def _moments(xs: list[float]) -> tuple[float, float, float, float]:
    """(mean, population stdev, skew, excess_kurt) of ``xs``. Returns 0s for n<2.

    All moments use population (ddof=0) normalisation so that skew and
    excess-kurtosis match the textbook standardised-moment definitions
    used in the Bailey & López de Prado DSR formula. The returned stdev
    is therefore population stdev (not sample).
    """
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(xs) / n
    m2 = sum((x - mean) ** 2 for x in xs) / n
    sd = math.sqrt(m2) if m2 > 0 else 0.0
    if sd <= 0.0:
        return mean, 0.0, 0.0, 0.0
    m3 = sum((x - mean) ** 3 for x in xs) / n
    m4 = sum((x - mean) ** 4 for x in xs) / n
    skew = m3 / (sd ** 3)
    kurt = m4 / (sd ** 4) - 3.0
    return mean, sd, skew, kurt


def sr_threshold(sr_variance_across_trials: float, n_trials: int) -> float:
    """Implied Sharpe threshold SR* that the best of N iid trials would cross
    by chance, per Bailey–López de Prado (2014). Uses Gumbel-max approximation.

    Args:
        sr_variance_across_trials: Var of Sharpe ratios across the N trials.
        n_trials: number of trials (configs / seeds / combinations).

    Returns:
        Expected maximum SR under the null (zero true skill), on the same
        periodicity as the per-trial Sharpe.
    """
    if n_trials <= 1 or sr_variance_across_trials <= 0.0:
        return 0.0
    sd = math.sqrt(sr_variance_across_trials)
    g = EULER_MASCHERONI
    # E[max of N iid normals] ≈ (1-γ)·Φ^-1(1 - 1/N) + γ·Φ^-1(1 - 1/(N·e))
    p1 = 1.0 - 1.0 / n_trials
    p2 = 1.0 - 1.0 / (n_trials * math.e)
    return sd * ((1.0 - g) * _NORM_PPF(p1) + g * _NORM_PPF(p2))


def deflated_sharpe(
    sr_hat: float,
    n_obs: int,
    skew: float,
    excess_kurt: float,
    sr_star: float,
) -> float:
    """Bailey–López de Prado (2014) deflated Sharpe ratio, returns P(true SR > SR*)."""
    if n_obs <= 1:
        return 0.5
    # Denominator variance of SR estimator, accounting for non-normal returns.
    denom_var = 1.0 - skew * sr_hat + (excess_kurt / 4.0) * sr_hat * sr_hat
    denom_var = max(denom_var, 1e-12)
    z = (sr_hat - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(denom_var)
    return _NORM_CDF(z)


def analyze_sweep(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sweep = raw.get("sweep_results") or []
    if not sweep:
        raise SystemExit(f"No sweep_results in {path}")

    trial_rows = []
    for entry in sweep:
        windows = entry.get("windows") or []
        rets = [float(w["monthly_return_pct"]) for w in windows
                if w.get("monthly_return_pct") is not None]
        if len(rets) < 3:
            continue
        mean, sd, skew, ekurt = _moments(rets)
        sharpe = (mean / sd) if sd > 0 else 0.0
        trial_rows.append({
            "config": entry.get("config", {}),
            "n_windows": len(rets),
            "median_monthly_pct": float(statistics.median(rets)),
            "p10_monthly_pct": float(_percentile(rets, 10.0)),
            "p90_monthly_pct": float(_percentile(rets, 90.0)),
            "mean_monthly_pct": mean,
            "std_monthly_pct": sd,
            "skew_monthly": skew,
            "excess_kurt_monthly": ekurt,
            "sharpe_monthly": sharpe,
            "sortino_monthly": _sortino(rets),
            "n_neg": int(sum(1 for r in rets if r < 0.0)),
        })

    n_trials = len(trial_rows)
    if n_trials == 0:
        raise SystemExit("No trial rows with ≥3 windows.")

    # Variance of Sharpes across trials (Bonferroni input)
    sharpes = [t["sharpe_monthly"] for t in trial_rows]
    _, sr_sd, _, _ = _moments(sharpes)
    sr_var = sr_sd ** 2
    sr_star = sr_threshold(sr_var, n_trials)

    # Best trial by median monthly
    best = max(trial_rows, key=lambda r: (r["median_monthly_pct"], r["sharpe_monthly"]))
    dsr = deflated_sharpe(
        sr_hat=best["sharpe_monthly"],
        n_obs=best["n_windows"],
        skew=best["skew_monthly"],
        excess_kurt=best["excess_kurt_monthly"],
        sr_star=sr_star,
    )

    # Seed-axis dispersion: if multiple configs share (n_est, depth, lr, top_n, xgb_weight, leverage),
    # they're seed replicas. Report median / p10 / p90 of monthly PnL across them.
    seed_groups: dict[tuple, list[dict]] = {}
    for row in trial_rows:
        cfg = row["config"]
        key = (
            cfg.get("n_estimators"), cfg.get("max_depth"), cfg.get("learning_rate"),
            cfg.get("top_n"), cfg.get("xgb_weight"), cfg.get("leverage"),
        )
        seed_groups.setdefault(key, []).append(row)

    seed_dispersions = []
    for key, members in seed_groups.items():
        if len(members) < 2:
            continue
        meds = [m["median_monthly_pct"] for m in members]
        p10s = [m["p10_monthly_pct"] for m in members]
        sortinos = [m["sortino_monthly"] for m in members]
        negs = [m["n_neg"] for m in members]
        seeds = [m["config"].get("random_state") for m in members]
        seed_dispersions.append({
            "hyperparam_cell": dict(zip(
                ["n_estimators", "max_depth", "learning_rate", "top_n", "xgb_weight", "leverage"],
                key,
            )),
            "n_seeds": len(members),
            "seeds": seeds,
            "median_monthly_median": float(statistics.median(meds)),
            "median_monthly_p10": float(_percentile(meds, 10.0)),
            "median_monthly_p90": float(_percentile(meds, 90.0)),
            "median_monthly_std": float(statistics.pstdev(meds)) if len(meds) > 1 else 0.0,
            "p10_monthly_median": float(statistics.median(p10s)),
            "sortino_median": float(statistics.median(sortinos)),
            "neg_median": float(statistics.median(negs)),
            "neg_max": max(negs),
        })

    return {
        "input_file": str(path),
        "n_trials": n_trials,
        "trial_sharpe_variance": sr_var,
        "sr_star_monthly": sr_star,
        "best": best,
        "best_deflated_sharpe": dsr,
        "best_dsr_passes_95": bool(dsr >= 0.95),
        "seed_dispersion_per_hyperparam_cell": seed_dispersions,
        "all_trials": trial_rows,
    }


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _sortino(xs: list[float]) -> float:
    if not xs:
        return 0.0
    mean = sum(xs) / len(xs)
    downside = [x for x in xs if x < 0.0]
    if not downside:
        # No negative windows — undefined Sortino; report mean as a positive
        # sentinel (sortino_∞ clipped to a large finite value for downstream
        # consumers that expect a float).
        return float("inf") if mean > 0 else 0.0
    if len(downside) == 1:
        ds = abs(downside[0])
    else:
        down_var = sum(d * d for d in downside) / len(downside)
        ds = math.sqrt(down_var) if down_var > 0 else 0.0
    return mean / ds if ds > 0 else 0.0


def print_report(summary: dict) -> None:
    best = summary["best"]
    cfg = best["config"]
    print("=" * 78)
    print(f"  Sweep Bonferroni / DSR report  — {summary['input_file']}")
    print("=" * 78)
    print(f"  Trials               : {summary['n_trials']}")
    print(f"  SR² across trials    : {summary['trial_sharpe_variance']:.6f}")
    print(f"  SR* (null max SR)    : {summary['sr_star_monthly']:+.3f}")
    print()
    print("  Best trial:")
    print(f"    cfg                : {cfg}")
    print(f"    n_windows          : {best['n_windows']}")
    print(f"    median monthly     : {best['median_monthly_pct']:+.2f}%")
    print(f"    p10 / p90 monthly  : {best['p10_monthly_pct']:+.2f}% / {best['p90_monthly_pct']:+.2f}%")
    print(f"    mean / std monthly : {best['mean_monthly_pct']:+.2f}% / {best['std_monthly_pct']:.2f}%")
    print(f"    skew / excess kurt : {best['skew_monthly']:+.3f} / {best['excess_kurt_monthly']:+.3f}")
    print(f"    Sharpe (monthly)   : {best['sharpe_monthly']:+.3f}")
    print(f"    Sortino (monthly)  : {best['sortino_monthly']:+.3f}")
    print(f"    Neg windows        : {best['n_neg']}/{best['n_windows']}")
    print()
    print(f"  Deflated Sharpe P   : {summary['best_deflated_sharpe']:.4f}")
    verdict = "PASS (real skill)" if summary['best_dsr_passes_95'] else "FAIL (Bonferroni-inflated)"
    print(f"  DSR verdict @ 0.95   : {verdict}")
    print()

    disp = summary.get("seed_dispersion_per_hyperparam_cell") or []
    if disp:
        print("  Seed-axis dispersion (identifies lucky-seed spikes):")
        print("  " + "-" * 74)
        for d in disp:
            cell = d["hyperparam_cell"]
            print(
                f"    cell={cell} n_seeds={d['n_seeds']}"
            )
            print(
                f"      median monthly across seeds: "
                f"p50={d['median_monthly_median']:+.2f}% "
                f"p10={d['median_monthly_p10']:+.2f}% "
                f"p90={d['median_monthly_p90']:+.2f}% "
                f"σ={d['median_monthly_std']:.2f}%"
            )
            print(
                f"      sortino across seeds       : "
                f"median={d['sortino_median']:.2f} "
                f"neg median/max={d['neg_median']:.1f}/{d['neg_max']}"
            )
    else:
        print("  (no replicated hyperparam cells — can't compute seed dispersion)")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="multiwindow_*.json to analyze")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON report to stdout instead of text.")
    ns = ap.parse_args(argv)
    if not ns.path.exists():
        print(f"ERROR: {ns.path} not found", file=sys.stderr)
        return 2
    summary = analyze_sweep(ns.path)
    if ns.json:
        json.dump(summary, sys.stdout, indent=2, default=str)
        print()
    else:
        print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
