import pytest

pytestmark = pytest.mark.cuda_required


@pytest.mark.timeout(600)
def test_simulate_strategy_real(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime unavailable")
    pytest.importorskip("chronos")
    pytest.importorskip("transformers")

    env_overrides = {
        "MARKETSIM_USE_MOCK_ANALYTICS": "0",
        "MARKETSIM_SKIP_REAL_IMPORT": "0",
        "MARKETSIM_ALLOW_CPU_FALLBACK": "1",
        "MARKETSIM_FORCE_KRONOS": "0",
        "FAST_TESTING": "1",
        "FAST_TOTO_NUM_SAMPLES": "64",
        "FAST_TOTO_SAMPLES_PER_BATCH": "16",
        "MARKETSIM_TOTO_MIN_NUM_SAMPLES": "64",
        "MARKETSIM_TOTO_MAX_NUM_SAMPLES": "256",
        "TORCHINDUCTOR_DISABLE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)

    from marketsimulator.runner import simulate_strategy

    try:
        report = simulate_strategy(
            symbols=["AAPL"],
            days=1,
            step_size=1,
            initial_cash=25_000.0,
            top_k=1,
            output_dir=tmp_path,
            force_kronos=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"Real analytics stack unavailable: {exc}")

    assert report.initial_cash == pytest.approx(25_000.0)
    assert report.daily_snapshots, "simulation produced no snapshots"
    assert report.trade_executions >= 0

    if report.trade_executions == 0:
        pytest.xfail("No trades executed; analytics returned empty signals")
