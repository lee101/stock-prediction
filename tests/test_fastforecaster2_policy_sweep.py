from fastforecaster2.policy_sweep import PolicySweepSpec, _iter_specs, _summary_rank_key, _trial_name


def test_iter_specs_skips_invalid_sell_and_min_intensity_combos():
    specs = list(
        _iter_specs(
            buy_thresholds=(3e-5,),
            sell_thresholds=(1e-5, 4e-5),
            entry_score_thresholds=(0.0, -0.001),
            top_ks=(1,),
            ema_alphas=(0.55,),
            max_hold_hours_values=(6,),
            max_trade_intensities=(8.0,),
            min_trade_intensities=(4.0, 10.0),
            switch_score_gaps=(0.0,),
        )
    )

    assert len(specs) == 1
    assert specs[0] == PolicySweepSpec(
        buy_threshold=3e-5,
        sell_threshold=1e-5,
        entry_score_threshold=0.0,
        top_k=1,
        ema_alpha=0.55,
        max_hold_hours=6,
        max_trade_intensity=8.0,
        min_trade_intensity=4.0,
        switch_score_gap=0.0,
    )


def test_trial_name_is_stable_and_path_safe():
    spec = PolicySweepSpec(
        buy_threshold=3e-5,
        sell_threshold=1.5e-5,
        entry_score_threshold=0.0015,
        top_k=1,
        ema_alpha=0.55,
        max_hold_hours=None,
        max_trade_intensity=12.0,
        min_trade_intensity=4.0,
        switch_score_gap=0.0,
    )

    assert _trial_name(spec) == "b3e-05_s1p5e-05_es0p0015_k1_ema0p55_hnone_maxi12_mini4_sgap0"


def test_trial_name_distinguishes_close_threshold_values():
    left = PolicySweepSpec(
        buy_threshold=3e-5,
        sell_threshold=1.35e-5,
        entry_score_threshold=0.0014,
        top_k=1,
        ema_alpha=0.55,
        max_hold_hours=6,
        max_trade_intensity=13.0,
        min_trade_intensity=4.0,
        switch_score_gap=0.0,
    )
    right = PolicySweepSpec(
        buy_threshold=3e-5,
        sell_threshold=1.4e-5,
        entry_score_threshold=0.0015,
        top_k=1,
        ema_alpha=0.55,
        max_hold_hours=6,
        max_trade_intensity=13.0,
        min_trade_intensity=4.0,
        switch_score_gap=0.0,
    )

    assert _trial_name(left) != _trial_name(right)


def test_summary_rank_key_prefers_goodness_score() -> None:
    stronger = {
        "sim_goodness_score": 1.2,
        "sim_pnl": 10.0,
        "sim_max_drawdown": 0.30,
        "sim_smoothness": 0.4,
    }
    weaker = {
        "sim_goodness_score": 0.9,
        "sim_pnl": 50.0,
        "sim_max_drawdown": 0.10,
        "sim_smoothness": 0.8,
    }

    assert _summary_rank_key(stronger) > _summary_rank_key(weaker)
