from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "wandb_run_scout.py"


def _load_module():
    spec = spec_from_file_location("wandb_run_scout", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(
    *,
    run_id: str,
    name: str,
    group: str,
    state: str,
    runtime_sec: float,
    steps: int,
    score: float,
    config: dict[str, object],
):
    return SimpleNamespace(
        id=run_id,
        name=name,
        group=group,
        state=state,
        created_at="2026-03-26T00:00:00Z",
        url=f"https://wandb.ai/test/{run_id}",
        summary={
            "_runtime": runtime_sec,
            "_step": steps,
            "val/score": score,
            "val/return": score / 10.0,
            "val/sortino": score / 20.0,
        },
        config=config,
    )


def test_fetch_runs_filters_out_trivial_or_excluded_runs() -> None:
    mod = _load_module()
    runs = [
        _run(
            run_id="best",
            name="alpaca_progress7_long",
            group="hourly_sweep",
            state="finished",
            runtime_sec=1800,
            steps=400,
            score=123.0,
            config={"learning_rate": 1e-4, "weight_decay": 0.01},
        ),
        _run(
            run_id="tiny",
            name="verify_local_torch_wandboard",
            group="wandboard_verify",
            state="finished",
            runtime_sec=3,
            steps=1,
            score=500.0,
            config={"learning_rate": 1e-4, "weight_decay": 0.0},
        ),
        _run(
            run_id="running",
            name="alpaca_progress7_running",
            group="hourly_sweep",
            state="running",
            runtime_sec=1200,
            steps=300,
            score=130.0,
            config={"learning_rate": 2e-4, "weight_decay": 0.02},
        ),
    ]
    wandb = SimpleNamespace(Api=lambda: SimpleNamespace(runs=lambda *args, **kwargs: runs))

    filtered = mod._fetch_runs(
        wandb=wandb,
        project="stock",
        entity="lee101p",
        group=None,
        last_n_runs=10,
        metric_keys=["val/score"],
        extra_metric_keys=["val/score"],
        states=["finished"],
        name_contains=["alpaca"],
        group_contains=["hourly"],
        exclude_name_contains=["verify"],
        exclude_group_contains=["wandboard"],
        min_runtime_sec=60.0,
        min_steps=10,
    )

    assert [row["id"] for row in filtered] == ["best"]
    assert filtered[0]["runtime_sec"] == 1800
    assert filtered[0]["steps"] == 400


def test_format_markdown_includes_runtime_and_steps() -> None:
    mod = _load_module()
    report = mod.build_scout_report(
        runs=[
            {
                "name": "alpaca_progress7_long",
                "group": "hourly_sweep",
                "state": "finished",
                "primary_metric": 123.0,
                "runtime_sec": 1800.0,
                "steps": 400,
                "summary": {"val/return": 12.3, "val/sortino": 6.15},
                "url": "https://wandb.ai/test/best",
                "config": {"learning_rate": 1e-4, "weight_decay": 0.01},
            }
        ],
        project="stock",
        entity="lee101p",
        metric_label="val/score",
        top_k=1,
        param_names=["learning_rate", "weight_decay"],
    )

    markdown = mod.format_markdown(report)

    assert "Runtime" in markdown
    assert "Steps" in markdown
    assert "1800s" in markdown
    assert "| alpaca_progress7_long | 123.0000 |" in markdown
