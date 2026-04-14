#!/usr/bin/env python3
"""
Chronos2 data loading and online augmentation for stock/crypto training.

Key ideas:
1. Load all daily stocks + hourly crypto into a unified list of time series
2. From hourly data, create multiple sliding-window daily aggregations (6-7x more data)
3. AugmentedChronos2Dataset applies online augmentation per training batch:
   - Amplitude jitter: scale entire series by exp(N(0, log_std)) -> scale invariance
   - Relative noise: small multiplicative noise on context -> prevents memorization
   - Time dropout: randomly NaN a fraction of context bars -> robustness
4. Percent-return variants: add return-space copies of each series for better stationarity

Usage:
    from chronos2_stock_augmentation import (
        AugConfig, AugmentedChronos2Dataset,
        prepare_all_training_series, create_sliding_daily_from_hourly,
    )
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

OHLC_COLS: Tuple[str, ...] = ("open", "high", "low", "close")

# Training set time budget: keep last N bars per series as validation
DEFAULT_VAL_BARS = 60   # ~3 months of daily bars
DEFAULT_TEST_BARS = 60  # another 3 months

# Cache version — bump when changing the data pipeline to invalidate stale caches
_CACHE_VERSION = "v2"


# ---------------------------------------------------------------------------
# Augmentation config
# ---------------------------------------------------------------------------

@dataclass
class AugConfig:
    """Online augmentation hyperparameters for Chronos2 training."""

    # Amplitude jitter: each slice is scaled by exp(N(0, amplitude_log_std)).
    # This teaches scale invariance on top of the arcsinh normalization in Chronos2.
    amplitude_log_std: float = 0.30

    # Relative noise: context += context_abs * N(0, noise_std_frac).
    # Prevents exact memorisation of training windows.
    noise_std_frac: float = 0.002

    # Time dropout: randomly NaN this fraction of context timesteps.
    # Teaches robustness to missing/stale bars.
    time_dropout_rate: float = 0.02

    # Sliding-window offsets for hourly→daily aggregation.
    # Offset o means: first bar starts at hourly index o, then steps of hours_per_day.
    # E.g. [0,1,2,3,4,5,6] gives 7 differently-aligned daily series per hourly stream.
    sliding_daily_offsets: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])

    # Bars to aggregate into one "daily equivalent" bar when converting hourly.
    hours_per_day: int = 7

    # Whether to also add a percent-return version of each series.
    # Return series is stationary and scale-invariant.
    add_return_variants: bool = True

    # Minimum number of bars for a series to be included.
    min_length: int = 50

    # Frequency subsampling: with this probability, subsample a slice at stride 2
    # (creating "2-day" bars by averaging adjacent bars). Teaches the model that
    # longer-range patterns at lower resolution are similar to daily patterns.
    # Set to 0 to disable.
    freq_subsample_prob: float = 0.0

    # Trend detrending: subtract a linear least-squares fit before each training
    # slice (applied to context only). Makes context more stationary.
    # Set to False to disable.
    detrend_context: bool = False

    # Channel dropout: with this probability, zero-out one random OHLC channel
    # on the context (not target). Teaches robustness to missing channels and
    # forces the model to infer information from the remaining channels.
    # Set to 0 to disable.
    channel_dropout_prob: float = 0.0

    # Time-warp probability: with this probability, randomly stretch/compress the
    # context by resampling with a random speed curve. Teaches invariance to small
    # temporal rescaling (e.g., different market speeds across regimes).
    # Set to 0 to disable.
    time_warp_prob: float = 0.0

    # Outlier injection: with this probability, replace 1-3 random context bars
    # with extreme moves (outlier_magnitude * local_std). Teaches crash/spike
    # robustness and prevents the model from being overconfident.
    # Set to 0 to disable.
    outlier_inject_prob: float = 0.0

    # Magnitude for injected outliers (in units of local rolling std).
    outlier_magnitude: float = 5.0

    # Gap injection: with this probability, add a sudden price gap at a random
    # bar in context (all 4 OHLC channels shifted by +/- gap_magnitude * prev_close).
    # Simulates overnight gap-up/gap-down events (news, earnings, macro surprises).
    # The gap persists for all subsequent bars (level shift), so the model must learn
    # to track absolute levels through discontinuities.
    # Set to 0 to disable.
    gap_inject_prob: float = 0.0

    # Gap magnitude as a fraction of the local mean (e.g., 0.05 = ±5% gap).
    gap_magnitude_frac: float = 0.05

    # Trend injection: with this probability, add a random smooth linear trend
    # to all OHLC channels. Simulates trending markets; forces the model to extract
    # relative patterns regardless of overall drift direction.
    # E.g. 0.15 = 15% chance of a random trend per context window.
    # Set to 0 to disable.
    trend_inject_prob: float = 0.0

    # Max magnitude of injected trend as a fraction of mean context value.
    # E.g. 0.10 = trend can reach ±10% of price over the full context.
    trend_magnitude_frac: float = 0.10

    # Volatility regime shift: with this probability, split context at a random
    # mid-point and multiply the second half (more recent) by a random volatility
    # multiplier.  Simulates GARCH-like volatility clustering (calm period →
    # turbulent period or vice versa) and teaches the model to adapt its uncertainty
    # estimate to the recent volatility regime.
    # Set to 0 to disable.
    vol_regime_prob: float = 0.0

    # Max volatility multiplier for regime shift.  The multiplier is drawn uniformly
    # in [1/vol_regime_max_mult, vol_regime_max_mult] on a log scale, so both
    # compression (calmer) and expansion (more volatile) are equally likely.
    vol_regime_max_mult: float = 4.0

    # Mean-reversion injection: with this probability, overlay a damped sinusoidal
    # oscillation on the context (all channels scaled equally).  Simulates
    # range-bound / oscillating market regimes and teaches the model that short-term
    # moves are often partially reversed.
    # Set to 0 to disable.
    mean_reversion_prob: float = 0.0

    # Amplitude of injected oscillation as a fraction of the channel mean.
    mean_reversion_amplitude: float = 0.03

    # Earnings shock injection: with this probability, inject a sudden large price
    # move at a random position in context, followed by either momentum continuation
    # or partial mean-reversion. Simulates earnings announcements and news events.
    # Distinct from gap_inject (persistent level shift) and outlier_inject (single
    # bar reverting to baseline) — this creates a multi-bar directional event.
    # Set to 0 to disable.
    earnings_shock_prob: float = 0.0

    # Maximum shock magnitude as a fraction of the local mean price.
    # Actual shock is U(0.05, earnings_shock_magnitude) so always ≥5%.
    # E.g. 0.15 = up to ±15% shock.
    earnings_shock_magnitude: float = 0.15

    # Structural break injection: with this probability, at a random split point
    # simultaneously apply a level shift (gap) AND a volatility regime change.
    # Distinct from gap_inject (level-only) and vol_regime (vol-only) — this
    # simulates macro/sector regime changes that alter both price level and volatility
    # (e.g., rate shock, corporate restructuring, index inclusion/exclusion).
    # The split is chosen in [T//4, 3*T//4] to ensure both halves have context.
    # Set to 0 to disable.
    struct_break_prob: float = 0.0

    # Level shift magnitude for structural break: fraction of local mean price.
    # The actual shift is U(0.02, struct_break_level_frac) with random sign.
    struct_break_level_frac: float = 0.08

    # Maximum volatility multiplier for structural break.
    # The vol multiplier is log-uniform in [1/max_mult, max_mult].
    struct_break_vol_mult: float = 3.0

    # Return momentum injection: with this probability, apply a simple AR(1)-like
    # smoothing to the returns within the context window, creating artificial
    # autocorrelation (positive = momentum, negative = mean-reversion tendency).
    # Implemented as a blend of the raw context with a smoothed version.
    # Unlike mean_reversion (which adds a sinusoidal overlay), this directly
    # manipulates the serial correlation structure of the return series.
    # Set to 0 to disable.
    return_momentum_prob: float = 0.0

    # Blend ratio for momentum injection: how much of the smoothed/reversed
    # version is mixed in (0.0 = no change, 1.0 = fully smoothed/reversed).
    return_momentum_blend: float = 0.4

    # AR coefficient for momentum injection: > 0 = momentum, < 0 = mean reversion.
    # If None (default), randomly chosen from U(-0.5, 0.5) each application.
    return_momentum_ar: Optional[float] = None


# ---------------------------------------------------------------------------
# Dataset with online augmentation
# ---------------------------------------------------------------------------

try:
    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
    _CHRONOS2_AVAILABLE = True
except ImportError:
    _CHRONOS2_AVAILABLE = False
    Chronos2Dataset = object  # type: ignore
    DatasetMode = None  # type: ignore


class AugmentedChronos2Dataset(Chronos2Dataset):  # type: ignore[misc]
    """
    Subclass of Chronos2Dataset that applies online augmentation in TRAIN mode.

    Augmentations (applied per-slice):
    1. Amplitude jitter  – scale entire window by exp(N(0, log_std))
    2. Relative noise    – add small noise proportional to |context|
    3. Time dropout      – NaN a random fraction of context timesteps

    Val/test slices are returned unmodified.
    """

    def __init__(self, *args, aug_config: Optional[AugConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_config = aug_config or AugConfig()

    def _construct_slice(self, task_idx: int):
        task_context, task_future_target, task_future_covariates, task_n_targets = (
            super()._construct_slice(task_idx)
        )

        if self.mode != DatasetMode.TRAIN:
            return task_context, task_future_target, task_future_covariates, task_n_targets

        cfg = self.aug_config

        # --- 1. Amplitude jitter ---
        if cfg.amplitude_log_std > 0:
            log_scale = random.gauss(0.0, cfg.amplitude_log_std)
            scale = math.exp(log_scale)
            task_context = task_context * scale
            if task_future_target is not None:
                # Scale target channels only (first task_n_targets rows)
                task_future_target = task_future_target.clone()
                task_future_target[:task_n_targets] = task_future_target[:task_n_targets] * scale

        # --- 2. Relative noise on context only ---
        if cfg.noise_std_frac > 0:
            noise = torch.randn_like(task_context) * cfg.noise_std_frac
            task_context = task_context + task_context.abs() * noise

        # --- 3. Time dropout on context only ---
        if cfg.time_dropout_rate > 0:
            T = task_context.shape[-1]
            if T > 0:
                mask = torch.rand(T) < cfg.time_dropout_rate
                if mask.any():
                    task_context = task_context.clone()
                    task_context[:, mask] = float("nan")

        # --- 4. Frequency subsampling (stride-2 "2-day" bars) ---
        if cfg.freq_subsample_prob > 0 and random.random() < cfg.freq_subsample_prob:
            # Subsample context at stride 2 by averaging adjacent pairs
            T = task_context.shape[-1]
            if T >= 4:
                # pad to even length
                if T % 2 == 1:
                    task_context = task_context[:, :-1]
                    T -= 1
                task_context = (task_context[:, 0::2] + task_context[:, 1::2]) * 0.5

        # --- 5. Linear detrend on context (makes it more stationary) ---
        if cfg.detrend_context:
            T = task_context.shape[-1]
            if T > 2:
                t = torch.arange(T, dtype=task_context.dtype, device=task_context.device)
                # per-channel least-squares detrend
                n = T
                sx = t.sum()
                sx2 = (t * t).sum()
                denom = n * sx2 - sx * sx
                if denom.abs() > 1e-8:
                    for ch in range(task_context.shape[0]):
                        valid = task_context[ch]
                        sy = valid.nansum()
                        sxy = (t * valid).nansum()
                        slope = (n * sxy - sx * sy) / denom
                        intercept = (sy - slope * sx) / n
                        task_context[ch] = valid - (slope * t + intercept)

        # --- 6. Channel dropout: zero out one random OHLC channel in context ---
        if cfg.channel_dropout_prob > 0 and random.random() < cfg.channel_dropout_prob:
            n_ch = task_context.shape[0]
            if n_ch >= 2:
                ch = random.randrange(n_ch)
                task_context = task_context.clone()
                task_context[ch] = float("nan")

        # --- 7. Time warp: resample context at a random speed curve ---
        if cfg.time_warp_prob > 0 and random.random() < cfg.time_warp_prob:
            T = task_context.shape[-1]
            if T >= 8:
                # Random warp factor in [0.8, 1.2]: stretch or compress by ±20%
                warp = 0.8 + random.random() * 0.4
                new_T = max(4, round(T * warp))
                # Linear interpolation along time axis
                old_idx = torch.linspace(0, T - 1, new_T)
                lo = old_idx.long().clamp(0, T - 2)
                hi = (lo + 1).clamp(0, T - 1)
                frac = (old_idx - lo.float()).unsqueeze(0)  # (1, new_T)
                ctx_float = task_context.float()
                task_context = ctx_float[:, lo] * (1 - frac) + ctx_float[:, hi] * frac
                # Pad/crop back to original T by repeating last bar or truncating
                if new_T < T:
                    pad = task_context[:, -1:].expand(-1, T - new_T)
                    task_context = torch.cat([task_context, pad], dim=-1)
                elif new_T > T:
                    task_context = task_context[:, :T]
                task_context = task_context.to(task_context.dtype)

        # --- 8. Outlier injection: replace 1-3 random bars with extreme moves ---
        if cfg.outlier_inject_prob > 0 and random.random() < cfg.outlier_inject_prob:
            T = task_context.shape[-1]
            if T >= 10:
                # Compute per-channel local std from finite values
                ctx_np = task_context.float()
                # std across time per channel; fallback to 1e-4 for constant channels
                per_ch_std = ctx_np.std(dim=-1, keepdim=True).clamp(min=1e-4)
                n_outliers = random.randint(1, min(3, T // 5))
                positions = random.sample(range(T), n_outliers)
                task_context = task_context.clone().float()
                for pos in positions:
                    # Each outlier bar: multiply by a random sign and outlier_magnitude
                    sign = 1.0 if random.random() > 0.5 else -1.0
                    task_context[:, pos] = (task_context[:, pos]
                                            + sign * cfg.outlier_magnitude * per_ch_std.squeeze(-1))
                task_context = task_context.to(task_context.dtype)

        # --- 9. Gap injection: add a level-shift at a random bar (overnight gap sim) ---
        if cfg.gap_inject_prob > 0 and random.random() < cfg.gap_inject_prob:
            T = task_context.shape[-1]
            if T >= 4:
                # Random gap position (not at first bar so there's some pre-gap context)
                gap_pos = random.randint(1, T - 1)
                task_context = task_context.clone().float()
                # Gap size: fraction of channel mean (fallback to 1e-4 if near-zero)
                ch_mean = task_context[:, :gap_pos].float().abs().mean(dim=-1).clamp(min=1e-4)
                sign = 1.0 if random.random() > 0.5 else -1.0
                gap_shift = sign * cfg.gap_magnitude_frac * ch_mean  # (n_channels,)
                # Apply gap to all bars from gap_pos onwards (persistent level shift)
                task_context[:, gap_pos:] = task_context[:, gap_pos:] + gap_shift.unsqueeze(-1)
                task_context = task_context.to(task_context.dtype)

        # --- 10. Trend injection: add random smooth linear trend to all channels ---
        # Forces model to learn relative patterns regardless of overall market drift.
        if cfg.trend_inject_prob > 0 and random.random() < cfg.trend_inject_prob:
            T = task_context.shape[-1]
            if T >= 4:
                task_context = task_context.clone().float()
                # Trend magnitude: fraction of channel mean, random sign, tapers as linear ramp
                ch_mean = task_context.abs().mean(dim=-1).clamp(min=1e-4)  # (n_channels,)
                sign = 1.0 if random.random() > 0.5 else -1.0
                # Linear ramp from 0 to ±trend_magnitude_frac * ch_mean over full context
                ramp = torch.linspace(0.0, 1.0, T, device=task_context.device,
                                      dtype=task_context.dtype)  # (T,)
                trend = sign * cfg.trend_magnitude_frac * ch_mean.unsqueeze(-1) * ramp  # (n_ch, T)
                task_context = task_context + trend
                task_context = task_context.to(task_context.dtype)

        # --- 11. Volatility regime shift ---
        # Multiply the second half of context by a random vol multiplier.
        # Simulates GARCH-like volatility clustering (calmer → turbulent or reverse).
        if cfg.vol_regime_prob > 0 and random.random() < cfg.vol_regime_prob:
            T = task_context.shape[-1]
            if T >= 8:
                # Split point uniformly in [T//4, 3*T//4]
                split = random.randint(T // 4, 3 * T // 4)
                task_context = task_context.clone().float()
                # Log-uniform multiplier: exp(U(-log(max_mult), log(max_mult)))
                log_mult = (random.random() * 2 - 1.0) * math.log(cfg.vol_regime_max_mult)
                multiplier = math.exp(log_mult)
                # Shift the second half around its own mean, scale, then shift back.
                # This preserves the level (no sudden jump) while changing volatility.
                chunk = task_context[:, split:]
                ch_mean = chunk.mean(dim=-1, keepdim=True)
                task_context[:, split:] = (chunk - ch_mean) * multiplier + ch_mean
                task_context = task_context.to(task_context.dtype)

        # --- 12. Mean-reversion injection ---
        # Overlay a damped sinusoidal oscillation on all channels.
        # Simulates range-bound markets; teaches the model that short moves revert.
        if cfg.mean_reversion_prob > 0 and random.random() < cfg.mean_reversion_prob:
            T = task_context.shape[-1]
            if T >= 8:
                task_context = task_context.clone().float()
                ch_mean = task_context.abs().mean(dim=-1).clamp(min=1e-4)  # (n_channels,)
                # Random period: between T/8 and T/2 bars
                period = random.uniform(T / 8.0, T / 2.0)
                # Random phase
                phase = random.uniform(0, 2 * math.pi)
                t = torch.arange(T, dtype=torch.float32, device=task_context.device)
                oscillation = torch.sin(2 * math.pi * t / period + phase)  # (T,)
                # Damping: oscillation decays to zero at end of context (most recent)
                # — so the model sees the full amplitude only in older bars
                damping = torch.linspace(1.0, 0.0, T, device=task_context.device)
                osc = oscillation * damping * cfg.mean_reversion_amplitude  # (T,)
                task_context = task_context + ch_mean.unsqueeze(-1) * osc.unsqueeze(0)
                task_context = task_context.to(task_context.dtype)

        # --- 13. Earnings shock injection ---
        # Simulates earnings/news: sudden large price move followed by either
        # momentum continuation or partial mean-reversion.  More realistic than
        # outlier_inject (which snaps back) or gap_inject (persistent level shift).
        if cfg.earnings_shock_prob > 0 and random.random() < cfg.earnings_shock_prob:
            T = task_context.shape[-1]
            if T >= 10:
                task_context = task_context.clone().float()
                ch_mean = task_context.abs().mean(dim=-1).clamp(min=1e-4)  # (n_channels,)
                # Shock position: between T//4 and 3*T//4 to avoid edges
                shock_pos = random.randint(T // 4, 3 * T // 4)
                sign = 1.0 if random.random() > 0.5 else -1.0
                magnitude = random.uniform(0.05, max(0.05, cfg.earnings_shock_magnitude))
                shock_size = sign * magnitude
                # Apply shock at shock_pos (sudden bar-level move on all channels)
                task_context[:, shock_pos] = task_context[:, shock_pos] + ch_mean * shock_size
                # Multi-bar follow-through: momentum or partial reversion
                if T > shock_pos + 1:
                    n_follow = min(3, T - shock_pos - 1)
                    if random.random() > 0.5:
                        # Momentum: 50% continuation fading over n_follow bars
                        for k in range(1, n_follow + 1):
                            follow = shock_size * 0.5 * (1.0 - k / (n_follow + 1))
                            task_context[:, shock_pos + k] = (
                                task_context[:, shock_pos + k] + ch_mean * follow)
                    else:
                        # Reversion: 30% pullback fading over n_follow bars
                        for k in range(1, n_follow + 1):
                            pullback = -shock_size * 0.3 * (1.0 - k / (n_follow + 1))
                            task_context[:, shock_pos + k] = (
                                task_context[:, shock_pos + k] + ch_mean * pullback)
                task_context = task_context.to(task_context.dtype)

        # --- 14. Structural break injection ---
        # Simultaneously applies a level shift AND a volatility regime change at a
        # random split point.  Simulates macro/sector regime changes (rate shock,
        # earnings restatement, index inclusion) that alter both price and vol.
        if cfg.struct_break_prob > 0 and random.random() < cfg.struct_break_prob:
            T = task_context.shape[-1]
            if T >= 12:
                task_context = task_context.clone().float()
                split = random.randint(T // 4, 3 * T // 4)
                ch_mean = task_context[:, :split].abs().mean(dim=-1).clamp(min=1e-4)  # (n_ch,)
                # Level shift: persistent gap at split point
                sign_level = 1.0 if random.random() > 0.5 else -1.0
                level_mag = random.uniform(0.02, max(0.02, cfg.struct_break_level_frac))
                gap_shift = sign_level * level_mag * ch_mean  # (n_ch,)
                task_context[:, split:] = task_context[:, split:] + gap_shift.unsqueeze(-1)
                # Volatility change: scale second-half deviations around new mean
                chunk = task_context[:, split:]
                new_mean = chunk.mean(dim=-1, keepdim=True)
                log_mult = (random.random() * 2 - 1.0) * math.log(cfg.struct_break_vol_mult)
                vol_mult = math.exp(log_mult)
                task_context[:, split:] = (chunk - new_mean) * vol_mult + new_mean
                task_context = task_context.to(task_context.dtype)

        # --- 15. Return momentum injection ---
        # Blends context with a smoothed version (momentum) or differenced version
        # (mean-reversion), creating artificial serial autocorrelation in the return
        # series.  Distinct from mean_reversion (sinusoidal overlay) and trend_inject
        # (linear drift) — operates on the local return autocorrelation structure.
        if cfg.return_momentum_prob > 0 and random.random() < cfg.return_momentum_prob:
            T = task_context.shape[-1]
            if T >= 8:
                task_context = task_context.clone().float()
                ar = cfg.return_momentum_ar
                if ar is None:
                    ar = random.uniform(-0.5, 0.5)
                blend = cfg.return_momentum_blend
                if ar > 0:
                    # Momentum: smooth returns with a kernel of width ~1/(1-ar)*2
                    # Use a simple 3-bar symmetric moving average as momentum proxy
                    w = min(max(3, round(1.0 / (1.0 - ar) * 2)), T // 4)
                    w = w if w % 2 == 1 else w + 1  # ensure odd kernel
                    kernel_size = min(w, T // 2)
                    if kernel_size >= 3:
                        import torch.nn.functional as F
                        ctx_exp = task_context.unsqueeze(1)  # (n_ch, 1, T)
                        kernel = torch.ones(1, 1, kernel_size, dtype=torch.float32,
                                            device=task_context.device) / kernel_size
                        smoothed = F.conv1d(ctx_exp, kernel,
                                            padding=kernel_size // 2)[:, 0, :]  # (n_ch, T)
                        # Crop to original T (padding may add extra bar)
                        smoothed = smoothed[:, :T]
                        task_context = (1 - blend) * task_context + blend * smoothed
                else:
                    # Mean-reversion: blend with the negated returns (detrended signal)
                    if T > 2:
                        returns_raw = torch.diff(task_context, dim=-1)          # (n_ch, T-1)
                        neg_returns = -returns_raw * abs(ar)                     # damped negation
                        # Reconstruct: cumsum of negated returns from initial bar
                        reconstructed = torch.zeros_like(task_context)
                        reconstructed[:, 0] = task_context[:, 0]
                        reconstructed[:, 1:] = task_context[:, 0:1] + neg_returns.cumsum(dim=-1)
                        task_context = (1 - blend) * task_context + blend * reconstructed
                task_context = task_context.to(task_context.dtype)

        return task_context, task_future_target, task_future_covariates, task_n_targets

    @classmethod
    def from_series_list(
        cls,
        series_list: List[dict],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        output_patch_size: int,
        min_past: int = 1,
        mode: str = "train",
        aug_config: Optional[AugConfig] = None,
    ) -> "AugmentedChronos2Dataset":
        """Convenience constructor that handles convert_inputs internally."""
        if not _CHRONOS2_AVAILABLE:
            raise ImportError("chronos-forecasting is not installed")
        inputs = [{"target": s["target"]} for s in series_list]
        dataset = cls(
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            min_past=min_past,
            mode=mode,
            aug_config=aug_config,
        )
        return dataset


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_ohlc_csv(
    csv_path: Path,
    cols: Tuple[str, ...] = OHLC_COLS,
    min_length: int = 50,
) -> Optional[np.ndarray]:
    """
    Load OHLC columns from a CSV. Returns float32 array of shape (len(cols), T).
    Returns None if the file is missing, malformed, or too short.
    """
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    df.columns = [c.lower().strip() for c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None

    arr = df[list(cols)].to_numpy(dtype=np.float32).T  # (n_cols, T)

    # Remove rows where all columns are NaN
    valid = np.all(np.isfinite(arr), axis=0)
    if not valid.any():
        return None
    first = int(np.argmax(valid))
    last = int(len(valid) - np.argmax(valid[::-1]))
    arr = arr[:, first:last]

    if arr.shape[1] < min_length:
        return None

    return arr


def load_all_series(
    data_dir: Path,
    cols: Tuple[str, ...] = OHLC_COLS,
    min_length: int = 50,
    glob: str = "*.csv",
    num_workers: int = 8,
) -> List[dict]:
    """
    Load all CSV files from data_dir using parallel IO.
    Returns list of dicts with keys "target" (np.ndarray shape (4,T)) and "symbol" (str).
    Skips subdirectories and files that fail validation.
    """
    import concurrent.futures

    data_dir = Path(data_dir)
    paths = sorted(p for p in data_dir.glob(glob) if p.is_file())

    def _load(csv_path: Path) -> Optional[dict]:
        arr = load_ohlc_csv(csv_path, cols, min_length=min_length)
        if arr is not None:
            return {"target": arr, "symbol": csv_path.stem}
        return None

    series: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        for result in ex.map(_load, paths):
            if result is not None:
                series.append(result)

    return sorted(series, key=lambda s: s["symbol"])


# ---------------------------------------------------------------------------
# Sliding-window hourly → daily aggregation
# ---------------------------------------------------------------------------

def create_sliding_daily_from_hourly(
    arr: np.ndarray,
    offsets: Sequence[int],
    hours_per_day: int = 7,
) -> List[np.ndarray]:
    """
    Create multiple daily-OHLC series from an hourly OHLC array by varying the
    aggregation start offset.

    For offset o, the daily bars are:
        bar_k = aggregate(arr[:, o + k*hours_per_day : o + (k+1)*hours_per_day])

    OHLC aggregation:
        daily_open  = first hourly open
        daily_high  = max of all hourly highs
        daily_low   = min of all hourly lows
        daily_close = last hourly close

    Args:
        arr:           Shape (4, T), columns = [open, high, low, close]
        offsets:       Sequence of start offsets to generate; typically range(hours_per_day)
        hours_per_day: Number of hourly bars to collapse into one daily bar

    Returns:
        List of arrays, each shape (4, n_days); one per offset.
        Entries with fewer than 30 daily bars are excluded.
    """
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError(f"Expected shape (4, T), got {arr.shape}")
    _, T = arr.shape

    results: List[np.ndarray] = []
    for offset in offsets:
        daily_bars: List[List[float]] = []
        pos = int(offset)
        while pos + hours_per_day <= T:
            w = arr[:, pos : pos + hours_per_day]
            d_open  = float(w[0, 0])
            d_high  = float(np.nanmax(w[1]))
            d_low   = float(np.nanmin(w[2]))
            d_close = float(w[3, -1])
            if np.isfinite([d_open, d_high, d_low, d_close]).all():
                daily_bars.append([d_open, d_high, d_low, d_close])
            pos += hours_per_day

        if len(daily_bars) >= 30:
            results.append(np.array(daily_bars, dtype=np.float32).T)  # (4, n_days)

    return results


# ---------------------------------------------------------------------------
# Percent-return transform
# ---------------------------------------------------------------------------

def to_return_series(arr: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
    """
    Convert OHLC price array to percent-return array.

    Formula per column c at timestep t (t >= 1):
        ret[c, t] = (arr[c, t] - arr[c, t-1]) / (|arr[c, t-1]| + eps)

    The first bar is dropped (no previous bar for the return). Returns None if
    the resulting array has fewer than 30 timesteps.

    Returns:
        Float32 array of shape (4, T-1) representing OHLC returns, or None.
    """
    if arr.shape[1] < 2:
        return None
    prev = arr[:, :-1]
    curr = arr[:, 1:]
    ret = (curr - prev) / (np.abs(prev) + eps)
    if ret.shape[1] < 30:
        return None
    return ret.astype(np.float32)


# ---------------------------------------------------------------------------
# Combined training dataset builder
# ---------------------------------------------------------------------------

def save_series_cache(series_list: List[dict], cache_path: Path) -> None:
    """Persist a list of {"target": np.ndarray} dicts to a compressed .npz file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"arr_{i}": s["target"] for i, s in enumerate(series_list)}
    arrays["_version"] = np.array([_CACHE_VERSION], dtype=object)
    np.savez_compressed(cache_path, **arrays)


def load_series_cache(cache_path: Path) -> Optional[List[dict]]:
    """Load a previously saved series list. Returns None if file missing or version mismatch."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        version = str(data.get("_version", np.array([""]))[0])
        if version != _CACHE_VERSION:
            print(f"Cache version mismatch ({version!r} != {_CACHE_VERSION!r}), ignoring")
            return None
        series = [{"target": data[k]} for k in sorted(data.files) if k.startswith("arr_")]
        return series
    except Exception as e:
        print(f"Failed to load cache {cache_path}: {e}")
        return None


def prepare_all_training_series(
    daily_data_dir: Optional[Path] = None,
    hourly_data_dirs: Optional[List[Path]] = None,
    aug_config: Optional[AugConfig] = None,
    cols: Tuple[str, ...] = OHLC_COLS,
    cache_path: Optional[Path] = None,
    num_workers: int = 8,
) -> List[dict]:
    """
    Build a unified list of training series from all available data.

    Sources:
    - Daily stock CSVs from daily_data_dir
    - Hourly crypto/stock CSVs from hourly_data_dirs (each dir processed separately)

    Static augmentations applied once at load time:
    - Sliding-window daily aggregations of hourly data (creates 6-7x more daily series)
    - Percent-return variants of every series (if aug_config.add_return_variants)

    Online augmentations are applied per-batch by AugmentedChronos2Dataset.

    Returns:
        List of dicts with key "target" (np.ndarray shape (4, T)).
    """
    # --- Try cache first ---
    if cache_path is not None:
        cached = load_series_cache(Path(cache_path))
        if cached is not None:
            print(f"Loaded {len(cached)} series from cache: {cache_path}")
            return cached

    cfg = aug_config or AugConfig()
    all_series: List[dict] = []
    n_daily = n_hourly = n_sliding = n_return = 0

    # --- Daily stocks ---
    if daily_data_dir is not None:
        for s in load_all_series(Path(daily_data_dir), cols, cfg.min_length, num_workers=num_workers):
            all_series.append({"target": s["target"]})
            n_daily += 1
            if cfg.add_return_variants:
                ret = to_return_series(s["target"])
                if ret is not None:
                    all_series.append({"target": ret})
                    n_return += 1

    # --- Hourly data: parallel sliding aggregation ---
    import concurrent.futures

    def _process_hourly_series(s: dict) -> List[np.ndarray]:
        arr = s["target"]
        results: List[np.ndarray] = [arr]
        if cfg.add_return_variants:
            ret = to_return_series(arr)
            if ret is not None:
                results.append(ret)
        min_hourly = cfg.hours_per_day * 30
        if arr.shape[1] >= min_hourly:
            for daily_arr in create_sliding_daily_from_hourly(
                arr, cfg.sliding_daily_offsets, cfg.hours_per_day
            ):
                results.append(daily_arr)
                if cfg.add_return_variants:
                    ret = to_return_series(daily_arr)
                    if ret is not None:
                        results.append(ret)
        return results

    for h_dir in (hourly_data_dirs or []):
        hourly_raw = load_all_series(Path(h_dir), cols, cfg.min_length, num_workers=num_workers)
        n_hourly += len(hourly_raw)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            for sub_results in ex.map(_process_hourly_series, hourly_raw):
                for arr in sub_results:
                    all_series.append({"target": arr})

    # Recount for reporting (easier than threading counters)
    total = len(all_series)
    print(
        f"Dataset: {total} series total "
        f"(daily≈{n_daily}, hourly≈{n_hourly}, +sliding+return_variants)"
    )

    # --- Save cache ---
    if cache_path is not None:
        print(f"Saving cache → {cache_path}")
        save_series_cache(all_series, Path(cache_path))

    return all_series


def split_series_list(
    series_list: List[dict],
    val_bars: int = DEFAULT_VAL_BARS,
    test_bars: int = DEFAULT_TEST_BARS,
    min_train_bars: int = 60,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split each series into train/val/test by holding out the last bars.

    Series shorter than min_train_bars + val_bars + test_bars are kept in train
    only (with no val/test contribution — too short for evaluation).

    Returns:
        (train_series, val_series, test_series)
        Each element is a list of {"target": np.ndarray} dicts.
    """
    train_list, val_list, test_list = [], [], []
    required = min_train_bars + val_bars + test_bars

    for s in series_list:
        arr = s["target"]
        T = arr.shape[-1]
        if T < required:
            train_list.append({"target": arr})
            continue
        train_end = T - val_bars - test_bars
        val_end   = T - test_bars

        train_list.append({"target": arr[:, :train_end]})
        val_list.append({"target": arr[:, train_end:val_end]})
        test_list.append({"target": arr[:, val_end:]})

    return train_list, val_list, test_list
