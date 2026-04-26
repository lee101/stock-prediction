from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "alpaca_deploy_preflight.py"


def _load_module():
    spec = spec_from_file_location("alpaca_deploy_preflight", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_flag_csv_values_handles_split_and_equals_forms() -> None:
    mod = _load_module()
    split = mod.extract_flag_csv_values("python bot.py --stock-symbols AAPL,MSFT,msft", "--stock-symbols")
    inline = mod.extract_flag_csv_values("python bot.py --stock-symbols=DBX,TRIP,dbx", "--stock-symbols")
    assert split == ["AAPL", "MSFT"]
    assert inline == ["DBX", "TRIP"]


def test_extract_flag_csv_values_handles_nargs_style_values() -> None:
    mod = _load_module()
    out = mod.extract_flag_csv_values(
        "python bot.py --stock-symbols YELP NET dbx --live --interval 3600",
        "--stock-symbols",
    )
    assert out == ["YELP", "NET", "DBX"]


def test_parse_git_status_porcelain_extracts_dirty_paths_and_ahead_behind() -> None:
    mod = _load_module()
    text = "\n".join(
        [
            "## main...origin/main [ahead 2, behind 4]",
            " M alpacaprod.md",
            "M  unified_orchestrator/orchestrator.py",
            "R  old.py -> new.py",
        ]
    )
    out = mod.parse_git_status_porcelain(text)
    assert out.branch_line == "main...origin/main [ahead 2, behind 4]"
    assert out.ahead == 2
    assert out.behind == 4
    assert out.dirty_paths == ["alpacaprod.md", "unified_orchestrator/orchestrator.py", "new.py"]


def test_repo_relative_dirty_paths_outside_watchlist_filters_expected_paths() -> None:
    mod = _load_module()
    dirty = ["alpacaprod.md", "unified_orchestrator/orchestrator.py", "README.md"]
    watched = ("unified_orchestrator/orchestrator.py",)
    assert mod.repo_relative_dirty_paths_outside_watchlist(dirty, watched) == ["README.md", "alpacaprod.md"]


def test_parse_supervisor_pid_extracts_running_pid() -> None:
    mod = _load_module()
    text = "unified-stock-trader             RUNNING   pid 153050, uptime 0:27:23\n"
    assert mod._parse_supervisor_pid(text) == 153050


def test_build_service_report_marks_restart_reasons_and_apply_blockers(monkeypatch) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="unit-test-svc",
        manager="supervisor",
        actual_name="unit-test-svc",
        config_path=Path("/tmp/unit-test-svc.conf"),
        watched_repo_files=("watched.py",),
        symbols_flag="--stock-symbols",
        ownership_service_name="owner",
    )
    git_status = mod.GitStatusSummary(
        branch_line="main...origin/main [ahead 2, behind 4]",
        dirty_paths=["README.md", "watched.py"],
        ahead=2,
        behind=4,
    )

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: 123)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: "python bot.py --stock-symbols AAPL,MSFT",
    )
    monkeypatch.setattr(
        mod,
        "_read_runtime_cmd",
        lambda _pid: "python bot.py --stock-symbols AAPL,MSFT,GOOG",
    )
    monkeypatch.setattr(
        mod,
        "_read_process_start_utc",
        lambda _pid: "2026-03-28T09:10:36+00:00",
    )
    monkeypatch.setattr(
        mod,
        "files_newer_than_process",
        lambda _pid, _paths: ["watched.py"],
    )
    monkeypatch.setattr(
        mod,
        "get_service_owned_symbols",
        lambda service_name: {
            "owner": ["AAPL", "MSFT"],
        }.get(service_name, []),
    )

    report = mod.build_service_report(spec, git_status)

    assert report.running is True
    assert report.runtime_symbols == ["AAPL", "MSFT", "GOOG"]
    assert report.configured_symbols == ["AAPL", "MSFT"]
    assert report.runtime_symbols_outside_ownership == ["GOOG"]
    assert report.configured_symbols_outside_ownership == []
    assert report.restart_reasons == [
        "watched_files_newer_than_process",
        "runtime_command_differs_from_config",
        "runtime_symbols_do_not_match_service_ownership",
    ]
    assert report.apply_blockers == [
        "dirty_repo_outside_watchlist:1",
        "branch_behind_origin:4",
    ]
    assert report.safe_to_apply is False


def test_build_service_report_marks_stopped_service_restart_needed(monkeypatch) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="unit-test-svc",
        manager="supervisor",
        actual_name="unit-test-svc",
        config_path=Path("/tmp/unit-test-svc.conf"),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(mod, "read_supervisor_command", lambda _path: "python bot.py")
    monkeypatch.setattr(mod, "read_repo_configured_command", lambda _spec: "python bot.py")
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)

    assert report.running is False
    assert report.restart_reasons == ["service_not_running"]
    assert report.apply_blockers == []
    assert report.safe_to_apply is True


def test_runtime_matches_configured_command_via_launch_script_wrapper(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u trade_daily_stock_prod.py --daemon --live --execution-backend trading_server\n"
    )

    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"
    runtime = "python -u trade_daily_stock_prod.py --daemon --live --execution-backend trading_server"

    assert mod.runtime_matches_configured_command(runtime, configured) is True


def test_runtime_matches_configured_command_expands_launch_assignments(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "MODEL_DIR=analysis/xgbnew_daily/alltrain_ensemble_gpu\n"
        "MODEL_PATHS=\"${MODEL_DIR}/alltrain_seed0.pkl,${MODEL_DIR}/alltrain_seed7.pkl\"\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths \"${MODEL_PATHS}\" \\\n"
        "  --top-n 1 \\\n"
        "  --live\n",
        encoding="utf-8",
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"
    runtime = (
        "python -u -m xgbnew.live_trader "
        "--model-paths "
        "analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed0.pkl,"
        "analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed7.pkl "
        "--top-n 1 --live"
    )

    assert mod.runtime_matches_configured_command(runtime, configured) is True


def test_unmodeled_live_sidecars_from_launch_script_wrapper(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --top-n 1 \\\n"
        "  --crypto-weekend \\\n"
        "  --eod-deleverage \\\n"
        "  --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.unmodeled_live_sidecars_from_command(
        configured,
        ("--crypto-weekend", "--eod-deleverage"),
    ) == ["--crypto-weekend", "--eod-deleverage"]


def test_live_launch_env_mismatches_from_launch_script_wrapper(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "export ALP_PAPER=1\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.live_launch_env_mismatches_from_command(
        configured,
        (("ALP_PAPER", "0"), ("ALLOW_ALPACA_LIVE_TRADING", "1")),
    ) == ["ALP_PAPER='1'!='0'", "ALLOW_ALPACA_LIVE_TRADING=missing"]


def test_live_launch_env_mismatches_empty_when_required_env_matches(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "export ALP_PAPER=0\n"
        "export ALLOW_ALPACA_LIVE_TRADING=1\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.live_launch_env_mismatches_from_command(
        configured,
        (("ALP_PAPER", "0"), ("ALLOW_ALPACA_LIVE_TRADING", "1")),
    ) == []


def test_live_launch_env_mismatches_respects_unset(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "export ALP_PAPER=0\n"
        "export ALLOW_ALPACA_LIVE_TRADING=1\n"
        "unset ALP_PAPER\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.live_launch_env_mismatches_from_command(
        configured,
        (("ALP_PAPER", "0"), ("ALLOW_ALPACA_LIVE_TRADING", "1")),
    ) == ["ALP_PAPER=missing"]


def test_live_launch_env_mismatches_respects_exec_env_unset(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "export ALP_PAPER=0\n"
        "export ALLOW_ALPACA_LIVE_TRADING=1\n"
        "exec env -u ALP_PAPER python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.live_launch_env_mismatches_from_command(
        configured,
        (("ALP_PAPER", "0"), ("ALLOW_ALPACA_LIVE_TRADING", "1")),
    ) == ["ALP_PAPER=missing"]


def test_live_launch_env_mismatches_accepts_exec_env_assignment(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec env ALP_PAPER=0 ALLOW_ALPACA_LIVE_TRADING=1 "
        "python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.live_launch_env_mismatches_from_command(
        configured,
        (("ALP_PAPER", "0"), ("ALLOW_ALPACA_LIVE_TRADING", "1")),
    ) == []


def test_forbidden_live_launch_env_from_launch_script_wrapper(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "export ALPACA_SINGLETON_OVERRIDE=1\n"
        "export ALPACA_DEATH_SPIRAL_OVERRIDE=0\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.forbidden_live_launch_env_from_command(
        configured,
        ("ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"),
    ) == ["ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"]


def test_forbidden_live_launch_env_respects_unset(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "export ALPACA_SINGLETON_OVERRIDE=1\n"
        "export ALPACA_DEATH_SPIRAL_OVERRIDE=1\n"
        "unset ALPACA_SINGLETON_OVERRIDE ALPACA_DEATH_SPIRAL_OVERRIDE\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.forbidden_live_launch_env_from_command(
        configured,
        ("ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"),
    ) == []


def test_forbidden_live_launch_env_detects_exec_env_assignment(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "source \"$HOME/.secretbashrc\"\n"
        "unset ALPACA_SINGLETON_OVERRIDE ALPACA_DEATH_SPIRAL_OVERRIDE\n"
        "exec env ALPACA_SINGLETON_OVERRIDE=1 "
        "python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.forbidden_live_launch_env_from_command(
        configured,
        ("ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"),
    ) == ["ALPACA_SINGLETON_OVERRIDE"]


def test_required_live_launch_unsets_missing_from_launch_script_wrapper(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "source \"$HOME/.secretbashrc\"\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.required_live_launch_unsets_missing_from_command(
        configured,
        ("ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"),
    ) == ["ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"]


def test_required_live_launch_unsets_must_follow_secret_source(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "unset ALPACA_SINGLETON_OVERRIDE ALPACA_DEATH_SPIRAL_OVERRIDE\n"
        "source \"$HOME/.secretbashrc\"\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.required_live_launch_unsets_missing_from_command(
        configured,
        ("ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"),
    ) == ["ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"]


def test_required_live_launch_unsets_accepts_unset_after_secret_source(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "source \"$HOME/.secretbashrc\"\n"
        "unset ALPACA_SINGLETON_OVERRIDE ALPACA_DEATH_SPIRAL_OVERRIDE\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    assert mod.required_live_launch_unsets_missing_from_command(
        configured,
        ("ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"),
    ) == []


def test_xgb_model_paths_extraction_respects_unset(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "MODEL_PATHS=\"analysis/xgbnew_daily/custom/alltrain_seed0.pkl\"\n"
        "unset MODEL_PATHS\n"
        "exec python -u -m xgbnew.live_trader --model-paths \"${MODEL_PATHS}\" --live\n"
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    model_paths, parse_error = mod.extract_xgb_model_paths_from_command(configured)

    assert parse_error is None
    assert model_paths == []


def test_runtime_live_env_helpers_detect_running_process_drift(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod,
        "_read_runtime_env",
        lambda _pid: {
            "ALP_PAPER": "1",
            "ALLOW_ALPACA_LIVE_TRADING": "1",
            "ALPACA_SINGLETON_OVERRIDE": "1",
        },
    )

    assert mod.runtime_live_env_mismatches_from_pid(
        123,
        (("ALP_PAPER", "0"), ("ALLOW_ALPACA_LIVE_TRADING", "1")),
    ) == ["ALP_PAPER='1'!='0'"]
    assert mod.forbidden_runtime_live_env_from_pid(
        123,
        ("ALPACA_SINGLETON_OVERRIDE", "ALPACA_DEATH_SPIRAL_OVERRIDE"),
    ) == ["ALPACA_SINGLETON_OVERRIDE"]


def test_scan_unmodeled_live_sidecars_reports_parse_errors() -> None:
    mod = _load_module()

    sidecars, parse_error = mod.scan_unmodeled_live_sidecars_from_command(
        "python -m xgbnew.live_trader --top-n 1 'unterminated --crypto-weekend",
        ("--crypto-weekend",),
    )

    assert sidecars == []
    assert parse_error is not None
    assert "No closing quotation" in parse_error


def test_symbols_outside_ownership_filters_non_owned_symbols() -> None:
    mod = _load_module()
    out = mod.symbols_outside_ownership(["YELP", "NET", "DBX"], ["NET", "DBX"])
    assert out == ["YELP"]


def test_extract_systemd_execstart_parses_unit_text() -> None:
    mod = _load_module()
    text = "\n".join(
        [
            "[Service]",
            "Environment=PYTHONUNBUFFERED=1",
            "ExecStart=/bin/bash -lc 'echo hi'",
        ]
    )
    assert mod._extract_systemd_execstart(text) == "/bin/bash -lc 'echo hi'"


def test_daily_rl_trader_watchlist_includes_inference_loader_files() -> None:
    mod = _load_module()
    spec = mod.SPECS["daily-rl-trader"]

    assert spec.manager == "supervisor"
    assert spec.actual_name == "daily-rl-trader"
    assert spec.watched_repo_files == (
        "trade_daily_stock_prod.py",
        "pufferlib_market/inference.py",
        "pufferlib_market/inference_daily.py",
        "pufferlib_market/checkpoint_loader.py",
        "src/daily_stock_defaults.py",
        "config/trading_server/accounts.json",
        "deployments/daily-rl-trader/launch.sh",
        "deployments/daily-rl-trader/supervisor.conf",
    )
    assert spec.repo_config_path == REPO / "deployments" / "daily-rl-trader" / "supervisor.conf"
    assert spec.ownership_service_name is None


def test_xgb_daily_trader_live_spec_tracks_launch_and_parity_files() -> None:
    mod = _load_module()
    spec = mod.SPECS["xgb-daily-trader-live"]

    assert spec.manager == "supervisor"
    assert spec.actual_name == "xgb-daily-trader-live"
    assert spec.repo_config_path == REPO / "deployments" / "xgb-daily-trader-live" / "supervisor.conf"
    assert "deployments/xgb-daily-trader-live/launch.sh" in spec.watched_repo_files
    assert "xgbnew/live_trader.py" in spec.watched_repo_files
    assert "xgbnew/backtest.py" in spec.watched_repo_files
    assert "src/alpaca_singleton.py" in spec.watched_repo_files
    assert spec.unmodeled_live_sidecar_flags == ("--crypto-weekend", "--eod-deleverage")
    assert spec.required_launch_env == (
        ("ALP_PAPER", "0"),
        ("ALLOW_ALPACA_LIVE_TRADING", "1"),
    )
    assert spec.forbidden_launch_env == (
        "ALPACA_SINGLETON_OVERRIDE",
        "ALPACA_DEATH_SPIRAL_OVERRIDE",
    )
    assert spec.required_launch_unsets == (
        "ALPACA_SINGLETON_OVERRIDE",
        "ALPACA_DEATH_SPIRAL_OVERRIDE",
    )
    assert spec.xgb_ensemble_dir == REPO / "analysis" / "xgbnew_daily" / "alltrain_ensemble_gpu"
    assert spec.xgb_ensemble_seeds == (0, 7, 42, 73, 197)


def test_validate_xgb_ensemble_for_spec_uses_repo_validator(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "ensemble"
    ensemble_dir.mkdir()
    calls = []
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        xgb_ensemble_dir=ensemble_dir,
        xgb_ensemble_seeds=(0, 7),
        xgb_ensemble_min_pkl_bytes=123,
    )

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({
                "train_end": "2026-04-01",
                "models": [
                    {
                        "path": "analysis/xgbnew_daily/alltrain_seed0.pkl",
                        "sha256": "a" * 64,
                    },
                    {
                        "path": "analysis/xgbnew_daily/alltrain_seed7.pkl",
                        "sha256": "b" * 64,
                    },
                ],
            }),
            stderr="",
        )

    monkeypatch.setattr(mod, "_repo_python", lambda: "/repo/.venv/bin/python")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    report = mod.validate_xgb_ensemble_for_spec(spec)
    assert report.validation_error is None
    assert report.model_paths == [
        str(REPO / "analysis" / "xgbnew_daily" / "alltrain_seed0.pkl"),
        str(REPO / "analysis" / "xgbnew_daily" / "alltrain_seed7.pkl"),
    ]
    assert report.model_sha256 == ["a" * 64, "b" * 64]
    assert report.train_end == "2026-04-01"
    assert calls[0][0] == [
        "/repo/.venv/bin/python",
        str(REPO / "scripts" / "validate_xgb_ensemble.py"),
        str(ensemble_dir),
        "--seeds",
        "0,7",
        "--min-pkl-bytes",
        "123",
        "--json",
    ]
    assert calls[0][1]["cwd"] == str(REPO)


def test_extract_xgb_model_paths_from_launch_resolves_shell_vars(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "MODEL_DIR=\"analysis/xgbnew_daily/custom\"\n"
        "MODEL_PATHS=\"${MODEL_DIR}/alltrain_seed0.pkl,${MODEL_DIR}/alltrain_seed7.pkl\"\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --model-paths \"${MODEL_PATHS}\" \\\n"
        "  --live\n",
        encoding="utf-8",
    )
    configured = f"/bin/bash -lc 'cd /repo && exec {launch}'"

    paths, parse_error = mod.extract_xgb_model_paths_from_command(configured)

    assert parse_error is None
    assert paths == [
        REPO / "analysis" / "xgbnew_daily" / "custom" / "alltrain_seed0.pkl",
        REPO / "analysis" / "xgbnew_daily" / "custom" / "alltrain_seed7.pkl",
    ]


def test_validate_xgb_ensemble_for_spec_prefers_launch_model_paths(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "MODEL_DIR=\"analysis/xgbnew_daily/custom\"\n"
        "MODEL_PATHS=\"${MODEL_DIR}/alltrain_seed0.pkl,${MODEL_DIR}/alltrain_seed7.pkl\"\n"
        "exec python -u -m xgbnew.live_trader --model-paths \"${MODEL_PATHS}\" --live\n",
        encoding="utf-8",
    )
    calls = []
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        xgb_ensemble_dir=tmp_path / "old-good-dir",
        xgb_ensemble_seeds=(0, 7),
        xgb_ensemble_min_pkl_bytes=123,
    )

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({
                "train_end": "2026-04-02",
                "models": [
                    {
                        "path": str(REPO / "analysis" / "xgbnew_daily" / "custom" / "alltrain_seed0.pkl"),
                        "sha256": "c" * 64,
                    },
                    {
                        "path": str(REPO / "analysis" / "xgbnew_daily" / "custom" / "alltrain_seed7.pkl"),
                        "sha256": "d" * 64,
                    },
                ],
            }),
            stderr="",
        )

    monkeypatch.setattr(mod, "_repo_python", lambda: "/repo/.venv/bin/python")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    report = mod.validate_xgb_ensemble_for_spec(
        spec,
        f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )

    assert report.validation_error is None
    assert report.model_paths_parse_error is None
    assert report.model_paths_missing is False
    assert report.model_paths == [
        str(REPO / "analysis" / "xgbnew_daily" / "custom" / "alltrain_seed0.pkl"),
        str(REPO / "analysis" / "xgbnew_daily" / "custom" / "alltrain_seed7.pkl"),
    ]
    assert report.model_sha256 == ["c" * 64, "d" * 64]
    assert report.train_end == "2026-04-02"
    assert str(tmp_path / "old-good-dir") not in calls[0]
    assert "--model-paths" in calls[0]
    assert "--require-manifest" in calls[0]
    assert "--json" in calls[0]


def test_validate_xgb_ensemble_for_spec_reports_validator_failure(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "ensemble"
    ensemble_dir.mkdir()
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        xgb_ensemble_dir=ensemble_dir,
        xgb_ensemble_seeds=(0,),
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            2,
            stdout="[xgb-ensemble-validate] FAIL bad model\n",
            stderr="details\n",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    report = mod.validate_xgb_ensemble_for_spec(spec)

    assert report.validation_error is not None
    assert "[xgb-ensemble-validate] FAIL bad model" in report.validation_error
    assert "details" in report.validation_error
    assert report.model_paths == []
    assert report.model_paths_parse_error is None
    assert report.model_paths_missing is False


def test_validate_xgb_ensemble_for_spec_rejects_invalid_validator_json(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    ensemble_dir = tmp_path / "ensemble"
    ensemble_dir.mkdir()
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        xgb_ensemble_dir=ensemble_dir,
        xgb_ensemble_seeds=(0,),
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="OK\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    report = mod.validate_xgb_ensemble_for_spec(spec)

    assert report.validation_error is not None
    assert "validator JSON invalid" in report.validation_error


def test_validate_xgb_ensemble_for_spec_rejects_launch_without_model_paths(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader --live\n",
        encoding="utf-8",
    )
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        xgb_ensemble_dir=tmp_path / "old-good-dir",
        xgb_ensemble_seeds=(0, 7),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("validator should not run when launch command omits --model-paths")

    monkeypatch.setattr(mod.subprocess, "run", fail_if_called)

    report = mod.validate_xgb_ensemble_for_spec(
        spec,
        f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )

    assert report.validation_error is None
    assert report.model_paths == []
    assert report.model_paths_parse_error is None
    assert report.model_paths_missing is True


def test_validate_xgb_spy_data_from_command_blocks_missing_spy_for_regime_gate(tmp_path) -> None:
    mod = _load_module()
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --data-root {data_root} "
        "--regime-gate-window 20 --live\n"
    )

    path, error = mod.validate_xgb_spy_data_from_command(
        f"/bin/bash -lc 'cd /repo && exec {launch}'"
    )

    assert path == str((data_root / "SPY.csv").resolve(strict=False))
    assert error == f"missing:{data_root / 'SPY.csv'}"


def test_validate_xgb_spy_data_from_command_accepts_spy_for_vol_target(tmp_path) -> None:
    mod = _load_module()
    expected_date = "2026-01-02"
    mod._expected_latest_completed_daily_bar_date = lambda now=None: mod.date.fromisoformat(expected_date)
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    spy_csv = data_root / "SPY.csv"
    spy_csv.write_text(
        "timestamp,close\n"
        f"{expected_date}T21:00:00Z,100\n",
        encoding="utf-8",
    )
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --data-root {data_root} "
        "--vol-target-ann 0.15 --live\n"
    )

    path, error = mod.validate_xgb_spy_data_from_command(
        f"/bin/bash -lc 'cd /repo && exec {launch}'"
    )

    assert path == str((data_root / "SPY.csv").resolve(strict=False))
    assert error is None

    report = mod.inspect_xgb_spy_data_from_command(
        f"/bin/bash -lc 'cd /repo && exec {launch}'"
    )
    assert report.path == str(spy_csv.resolve(strict=False))
    assert report.error is None
    assert report.sha256 == hashlib.sha256(spy_csv.read_bytes()).hexdigest()
    assert report.latest_date == expected_date
    assert report.usable_rows == 1


def test_validate_xgb_spy_data_from_command_honors_explicit_spy_csv(tmp_path) -> None:
    mod = _load_module()
    expected_date = "2026-01-02"
    mod._expected_latest_completed_daily_bar_date = lambda now=None: mod.date.fromisoformat(expected_date)
    data_root = tmp_path / "missing_trainingdata"
    custom_spy = tmp_path / "market_inputs" / "custom_spy.csv"
    custom_spy.parent.mkdir()
    custom_spy.write_text(
        "timestamp,close\n"
        f"{expected_date}T21:00:00Z,100\n",
        encoding="utf-8",
    )
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --data-root {data_root} "
        f"--spy-csv {custom_spy} --vol-target-ann 0.15 --live\n"
    )

    path, error = mod.validate_xgb_spy_data_from_command(
        f"/bin/bash -lc 'cd /repo && exec {launch}'"
    )

    assert path == str(custom_spy.resolve(strict=False))
    assert error is None


def test_validate_xgb_spy_data_rejects_empty_explicit_spy_csv(tmp_path) -> None:
    mod = _load_module()
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    (data_root / "SPY.csv").write_text(
        "timestamp,close\n"
        "2026-01-02T21:00:00Z,100\n",
        encoding="utf-8",
    )
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "SPY_CSV=\"\"\n"
        f"exec python -u -m xgbnew.live_trader --data-root {data_root} "
        "--spy-csv \"${SPY_CSV}\" --vol-target-ann 0.15 --live\n"
    )

    path, error = mod.validate_xgb_spy_data_from_command(
        f"/bin/bash -lc 'cd /repo && exec {launch}'"
    )

    assert path is None
    assert error == "--spy-csv must not be empty"


def test_validate_xgb_spy_data_rejects_empty_explicit_data_root(tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "DATA_ROOT=\"\"\n"
        "exec python -u -m xgbnew.live_trader --data-root \"${DATA_ROOT}\" "
        "--vol-target-ann 0.15 --live\n"
    )

    path, error = mod.validate_xgb_spy_data_from_command(
        f"/bin/bash -lc 'cd /repo && exec {launch}'"
    )

    assert path is None
    assert error == "--data-root must not be empty"


def test_validate_xgb_spy_data_rejects_stale_spy_csv(tmp_path) -> None:
    mod = _load_module()
    mod._expected_latest_completed_daily_bar_date = lambda now=None: mod.date.fromisoformat("2026-01-05")
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    (data_root / "SPY.csv").write_text(
        "timestamp,close\n"
        "2026-01-02T21:00:00Z,100\n",
        encoding="utf-8",
    )
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --data-root {data_root} "
        "--vol-target-ann 0.15 --live\n"
    )

    path, error = mod.validate_xgb_spy_data_from_command(
        f"/bin/bash -lc 'cd /repo && exec {launch}'"
    )

    assert path == str((data_root / "SPY.csv").resolve(strict=False))
    assert error == f"stale_latest_date:2026-01-02/2026-01-05:{data_root / 'SPY.csv'}"


def test_validate_xgb_spy_data_rejects_non_numeric_closes(tmp_path) -> None:
    mod = _load_module()
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    (data_root / "SPY.csv").write_text(
        "timestamp,close\n"
        "2026-01-02T21:00:00Z,not-a-number\n",
        encoding="utf-8",
    )
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --data-root {data_root} "
        "--vol-target-ann 0.15 --live\n"
    )

    path, error = mod.validate_xgb_spy_data_from_command(
        f"/bin/bash -lc 'cd /repo && exec {launch}'"
    )

    assert path == str((data_root / "SPY.csv").resolve(strict=False))
    assert error == f"insufficient_rows:0/1:{data_root / 'SPY.csv'}"


def test_build_service_report_blocks_unmodeled_live_sidecars(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "exec python -u -m xgbnew.live_trader \\\n"
        "  --top-n 1 \\\n"
        "  --crypto-weekend \\\n"
        "  --eod-deleverage \\\n"
        "  --live\n"
    )
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        repo_config_path=tmp_path / "supervisor.conf",
        unmodeled_live_sidecar_flags=("--crypto-weekend", "--eod-deleverage"),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(
        mod,
        "read_repo_configured_command",
        lambda _spec: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)

    assert report.unmodeled_live_sidecars == ["--crypto-weekend", "--eod-deleverage"]
    assert report.apply_blockers == [
        "unmodeled_live_sidecars:--crypto-weekend,--eod-deleverage"
    ]
    assert report.safe_to_apply is False


def test_build_service_report_blocks_invalid_xgb_spy_data(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --data-root {data_root} "
        "--regime-gate-window 20 --live\n"
    )
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        repo_config_path=tmp_path / "supervisor.conf",
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(
        mod,
        "read_repo_configured_command",
        lambda _spec: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)

    assert report.xgb_spy_data_path == str((data_root / "SPY.csv").resolve(strict=False))
    assert report.xgb_spy_data_error == f"missing:{data_root / 'SPY.csv'}"
    assert "xgb_spy_data_invalid" in report.apply_blockers
    assert report.safe_to_apply is False


def test_build_service_report_records_xgb_spy_data_provenance(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    expected_date = "2026-01-02"
    monkeypatch.setattr(
        mod,
        "_expected_latest_completed_daily_bar_date",
        lambda now=None: mod.date.fromisoformat(expected_date),
    )
    data_root = tmp_path / "trainingdata"
    data_root.mkdir()
    spy_csv = data_root / "SPY.csv"
    spy_csv.write_text(
        "timestamp,close\n"
        f"{expected_date}T21:00:00Z,100\n",
        encoding="utf-8",
    )
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python -u -m xgbnew.live_trader --data-root {data_root} "
        "--vol-target-ann 0.15 --live\n"
    )
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        repo_config_path=tmp_path / "supervisor.conf",
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(
        mod,
        "read_repo_configured_command",
        lambda _spec: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)
    text = mod.render_text(git_status, [report])
    expected_sha = hashlib.sha256(spy_csv.read_bytes()).hexdigest()

    assert report.xgb_spy_data_path == str(spy_csv.resolve(strict=False))
    assert report.xgb_spy_data_sha256 == expected_sha
    assert report.xgb_spy_data_latest_date == expected_date
    assert report.xgb_spy_data_usable_rows == 1
    assert report.xgb_spy_data_error is None
    assert "xgb_spy_data_sha256: " + expected_sha in text
    assert f"xgb_spy_data_latest_date: {expected_date}" in text
    assert "xgb_spy_data_usable_rows: 1" in text


def test_build_service_report_blocks_invalid_xgb_ensemble(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        xgb_ensemble_dir=tmp_path / "ensemble",
        xgb_ensemble_seeds=(0,),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(mod, "read_supervisor_command", lambda _path: None)
    monkeypatch.setattr(mod, "read_repo_configured_command", lambda _spec: None)
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])
    monkeypatch.setattr(
        mod,
        "validate_xgb_ensemble_for_spec",
        lambda _spec, _cmd=None: mod.XGBEnsembleValidationReport(validation_error="bad model"),
    )

    report = mod.build_service_report(spec, git_status)

    assert report.xgb_ensemble_validation_error == "bad model"
    assert report.apply_blockers == ["xgb_ensemble_validation_failed"]
    assert report.safe_to_apply is False


def test_build_service_report_blocks_xgb_model_path_parse_errors(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        xgb_ensemble_dir=tmp_path / "ensemble",
        xgb_ensemble_seeds=(0,),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(mod, "read_supervisor_command", lambda _path: None)
    monkeypatch.setattr(mod, "read_repo_configured_command", lambda _spec: None)
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])
    monkeypatch.setattr(
        mod,
        "validate_xgb_ensemble_for_spec",
        lambda _spec, _cmd=None: mod.XGBEnsembleValidationReport(
            model_paths_parse_error="No closing quotation",
        ),
    )

    report = mod.build_service_report(spec, git_status)

    assert report.xgb_model_paths_parse_error == "No closing quotation"
    assert report.apply_blockers == ["xgb_model_paths_parse_error"]
    assert report.safe_to_apply is False


def test_build_service_report_blocks_missing_xgb_model_paths(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        xgb_ensemble_dir=tmp_path / "ensemble",
        xgb_ensemble_seeds=(0,),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(mod, "read_supervisor_command", lambda _path: None)
    monkeypatch.setattr(mod, "read_repo_configured_command", lambda _spec: None)
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])
    monkeypatch.setattr(
        mod,
        "validate_xgb_ensemble_for_spec",
        lambda _spec, _cmd=None: mod.XGBEnsembleValidationReport(model_paths_missing=True),
    )

    report = mod.build_service_report(spec, git_status)

    assert report.xgb_model_paths_missing is True
    assert report.apply_blockers == ["xgb_model_paths_missing"]
    assert report.safe_to_apply is False


def test_build_service_report_records_xgb_validation_provenance(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        xgb_ensemble_dir=tmp_path / "ensemble",
        xgb_ensemble_seeds=(0,),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(mod, "read_supervisor_command", lambda _path: None)
    monkeypatch.setattr(mod, "read_repo_configured_command", lambda _spec: None)
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])
    monkeypatch.setattr(
        mod,
        "validate_xgb_ensemble_for_spec",
        lambda _spec, _cmd=None: mod.XGBEnsembleValidationReport(
            model_paths=["/repo/model0.pkl"],
            model_sha256=["e" * 64],
            train_end="2026-04-01",
        ),
    )

    report = mod.build_service_report(spec, git_status)
    text = mod.render_text(git_status, [report])

    assert report.xgb_model_paths == ["/repo/model0.pkl"]
    assert report.xgb_model_sha256 == ["e" * 64]
    assert report.xgb_ensemble_train_end == "2026-04-01"
    assert "xgb_model_sha256: " + ("e" * 64) in text
    assert "xgb_ensemble_train_end: 2026-04-01" in text


def test_build_service_report_blocks_live_sidecar_parse_errors(monkeypatch) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        unmodeled_live_sidecar_flags=("--crypto-weekend",),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: "python -m xgbnew.live_trader 'unterminated --crypto-weekend",
    )
    monkeypatch.setattr(mod, "read_repo_configured_command", lambda _spec: None)
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)

    assert report.unmodeled_live_sidecars == []
    assert report.live_sidecar_parse_error is not None
    assert report.apply_blockers == ["live_sidecar_parse_error"]
    assert report.safe_to_apply is False


def test_build_service_report_blocks_bad_live_launch_env(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "export ALP_PAPER=1\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        repo_config_path=tmp_path / "supervisor.conf",
        required_launch_env=(
            ("ALP_PAPER", "0"),
            ("ALLOW_ALPACA_LIVE_TRADING", "1"),
        ),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(
        mod,
        "read_repo_configured_command",
        lambda _spec: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)
    text = mod.render_text(git_status, [report])

    assert report.live_launch_env_mismatches == [
        "ALP_PAPER='1'!='0'",
        "ALLOW_ALPACA_LIVE_TRADING=missing",
    ]
    assert report.apply_blockers == [
        "live_launch_env_mismatch:ALP_PAPER='1'!='0',ALLOW_ALPACA_LIVE_TRADING=missing"
    ]
    assert "live_launch_env_mismatches: ALP_PAPER='1'!='0'" in text
    assert report.safe_to_apply is False


def test_build_service_report_blocks_forbidden_live_launch_env(
    monkeypatch,
    tmp_path,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "export ALP_PAPER=0\n"
        "export ALLOW_ALPACA_LIVE_TRADING=1\n"
        "export ALPACA_SINGLETON_OVERRIDE=1\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        repo_config_path=tmp_path / "supervisor.conf",
        required_launch_env=(
            ("ALP_PAPER", "0"),
            ("ALLOW_ALPACA_LIVE_TRADING", "1"),
        ),
        forbidden_launch_env=("ALPACA_SINGLETON_OVERRIDE",),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(
        mod,
        "read_repo_configured_command",
        lambda _spec: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)
    text = mod.render_text(git_status, [report])

    assert report.live_launch_env_mismatches == []
    assert report.forbidden_live_launch_env == ["ALPACA_SINGLETON_OVERRIDE"]
    assert report.apply_blockers == [
        "forbidden_live_launch_env:ALPACA_SINGLETON_OVERRIDE"
    ]
    assert "forbidden_live_launch_env: ALPACA_SINGLETON_OVERRIDE" in text
    assert report.safe_to_apply is False


def test_build_service_report_blocks_missing_required_live_launch_unsets(
    monkeypatch,
    tmp_path,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "source \"$HOME/.secretbashrc\"\n"
        "export ALP_PAPER=0\n"
        "export ALLOW_ALPACA_LIVE_TRADING=1\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        repo_config_path=tmp_path / "supervisor.conf",
        required_launch_env=(
            ("ALP_PAPER", "0"),
            ("ALLOW_ALPACA_LIVE_TRADING", "1"),
        ),
        required_launch_unsets=("ALPACA_SINGLETON_OVERRIDE",),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: None)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(
        mod,
        "read_repo_configured_command",
        lambda _spec: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)
    text = mod.render_text(git_status, [report])

    assert report.required_live_launch_unsets_missing == ["ALPACA_SINGLETON_OVERRIDE"]
    assert report.apply_blockers == [
        "required_live_launch_unsets_missing:ALPACA_SINGLETON_OVERRIDE"
    ]
    assert "required_live_launch_unsets_missing: ALPACA_SINGLETON_OVERRIDE" in text
    assert report.safe_to_apply is False


def test_build_service_report_flags_runtime_live_env_drift_without_blocking_apply(
    monkeypatch,
    tmp_path,
) -> None:
    mod = _load_module()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "#!/usr/bin/env bash\n"
        "export ALP_PAPER=0\n"
        "export ALLOW_ALPACA_LIVE_TRADING=1\n"
        "exec python -u -m xgbnew.live_trader --live\n"
    )
    spec = mod.ServiceSpec(
        name="xgb-unit",
        manager="supervisor",
        actual_name="xgb-unit",
        config_path=Path("/tmp/xgb-unit.conf"),
        repo_config_path=tmp_path / "supervisor.conf",
        required_launch_env=(
            ("ALP_PAPER", "0"),
            ("ALLOW_ALPACA_LIVE_TRADING", "1"),
        ),
        forbidden_launch_env=("ALPACA_SINGLETON_OVERRIDE",),
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_supervisor_pid", lambda _program: 123)
    monkeypatch.setattr(
        mod,
        "read_supervisor_command",
        lambda _path: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(
        mod,
        "read_repo_configured_command",
        lambda _spec: f"/bin/bash -lc 'cd /repo && exec {launch}'",
    )
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(
        mod,
        "_read_runtime_env",
        lambda _pid: {
            "ALP_PAPER": "1",
            "ALLOW_ALPACA_LIVE_TRADING": "1",
            "ALPACA_SINGLETON_OVERRIDE": "1",
        },
    )
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)
    text = mod.render_text(git_status, [report])

    assert report.live_launch_env_mismatches == []
    assert report.forbidden_live_launch_env == []
    assert report.runtime_live_env_mismatches == ["ALP_PAPER='1'!='0'"]
    assert report.forbidden_runtime_live_env == ["ALPACA_SINGLETON_OVERRIDE"]
    assert "runtime_live_env_mismatch" in report.restart_reasons
    assert "forbidden_runtime_live_env" in report.restart_reasons
    assert report.apply_blockers == []
    assert report.safe_to_apply is True
    assert "runtime_live_env_mismatches: ALP_PAPER='1'!='0'" in text
    assert "forbidden_runtime_live_env: ALPACA_SINGLETON_OVERRIDE" in text


def test_main_allow_unmodeled_live_sidecars_unblocks_apply(monkeypatch) -> None:
    mod = _load_module()
    report = mod.ServiceReport(
        service="xgb-daily-trader-live",
        manager="supervisor",
        pid=None,
        running=False,
        process_start_utc=None,
        runtime_cmd=None,
        configured_cmd=None,
        repo_configured_cmd=None,
        unmodeled_live_sidecars=["--crypto-weekend"],
        apply_blockers=["unmodeled_live_sidecars:--crypto-weekend"],
        safe_to_apply=False,
    )
    applied: list[str] = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca_deploy_preflight.py",
            "--service", "xgb-daily-trader-live",
            "--apply",
            "--allow-unmodeled-live-sidecars",
        ],
    )
    monkeypatch.setattr(
        mod,
        "run_text",
        lambda _cmd, *, cwd=None: "## main...origin/main\n",
    )
    monkeypatch.setattr(mod, "build_service_report", lambda _spec, _git_status: report)
    monkeypatch.setattr(mod, "apply_service", lambda spec: applied.append(spec.name))

    assert mod.main() == 0
    assert applied == ["xgb-daily-trader-live"]
    assert report.apply_blockers == []
    assert report.safe_to_apply is True


def test_main_fail_on_unsafe_returns_nonzero_without_apply(monkeypatch) -> None:
    mod = _load_module()
    report = mod.ServiceReport(
        service="xgb-daily-trader-live",
        manager="supervisor",
        pid=None,
        running=False,
        process_start_utc=None,
        runtime_cmd=None,
        configured_cmd=None,
        repo_configured_cmd=None,
        apply_blockers=["dirty_repo_outside_watchlist:1"],
        safe_to_apply=False,
    )
    applied: list[str] = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca_deploy_preflight.py",
            "--service",
            "xgb-daily-trader-live",
            "--fail-on-unsafe",
        ],
    )
    monkeypatch.setattr(mod, "run_text", lambda _cmd, *, cwd=None: "## main...origin/main\n")
    monkeypatch.setattr(mod, "build_service_report", lambda _spec, _git_status: report)
    monkeypatch.setattr(mod, "apply_service", lambda spec: applied.append(spec.name))

    assert mod.main() == 2
    assert applied == []


def test_main_fail_on_unsafe_respects_unmodeled_sidecar_override(monkeypatch) -> None:
    mod = _load_module()
    report = mod.ServiceReport(
        service="xgb-daily-trader-live",
        manager="supervisor",
        pid=None,
        running=False,
        process_start_utc=None,
        runtime_cmd=None,
        configured_cmd=None,
        repo_configured_cmd=None,
        unmodeled_live_sidecars=["--crypto-weekend"],
        apply_blockers=["unmodeled_live_sidecars:--crypto-weekend"],
        safe_to_apply=False,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca_deploy_preflight.py",
            "--service",
            "xgb-daily-trader-live",
            "--fail-on-unsafe",
            "--allow-unmodeled-live-sidecars",
        ],
    )
    monkeypatch.setattr(mod, "run_text", lambda _cmd, *, cwd=None: "## main...origin/main\n")
    monkeypatch.setattr(mod, "build_service_report", lambda _spec, _git_status: report)

    assert mod.main() == 0
    assert report.apply_blockers == []
    assert report.safe_to_apply is True


def test_main_fail_on_unsafe_respects_invalid_xgb_ensemble_override(monkeypatch) -> None:
    mod = _load_module()
    report = mod.ServiceReport(
        service="xgb-daily-trader-live",
        manager="supervisor",
        pid=None,
        running=False,
        process_start_utc=None,
        runtime_cmd=None,
        configured_cmd=None,
        repo_configured_cmd=None,
        xgb_ensemble_validation_error="bad model",
        apply_blockers=["xgb_ensemble_validation_failed"],
        safe_to_apply=False,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca_deploy_preflight.py",
            "--service",
            "xgb-daily-trader-live",
            "--fail-on-unsafe",
            "--allow-invalid-xgb-ensemble",
        ],
    )
    monkeypatch.setattr(mod, "run_text", lambda _cmd, *, cwd=None: "## main...origin/main\n")
    monkeypatch.setattr(mod, "build_service_report", lambda _spec, _git_status: report)

    assert mod.main() == 0
    assert report.apply_blockers == []
    assert report.safe_to_apply is True


def test_main_fail_on_unsafe_respects_missing_xgb_model_paths_override(monkeypatch) -> None:
    mod = _load_module()
    report = mod.ServiceReport(
        service="xgb-daily-trader-live",
        manager="supervisor",
        pid=None,
        running=False,
        process_start_utc=None,
        runtime_cmd=None,
        configured_cmd=None,
        repo_configured_cmd=None,
        xgb_model_paths_missing=True,
        apply_blockers=["xgb_model_paths_missing"],
        safe_to_apply=False,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alpaca_deploy_preflight.py",
            "--service",
            "xgb-daily-trader-live",
            "--fail-on-unsafe",
            "--allow-invalid-xgb-ensemble",
        ],
    )
    monkeypatch.setattr(mod, "run_text", lambda _cmd, *, cwd=None: "## main...origin/main\n")
    monkeypatch.setattr(mod, "build_service_report", lambda _spec, _git_status: report)

    assert mod.main() == 0
    assert report.apply_blockers == []
    assert report.safe_to_apply is True


def test_build_service_report_flags_installed_config_drift_from_repo(monkeypatch) -> None:
    mod = _load_module()
    spec = mod.ServiceSpec(
        name="unit-test-svc",
        manager="systemd",
        actual_name="unit-test-svc.service",
        config_path=Path("/etc/systemd/system/unit-test-svc.service"),
        repo_config_path=REPO / "systemd" / "daily-rl-trader.service",
    )
    git_status = mod.GitStatusSummary(branch_line="main...origin/main", dirty_paths=[])

    monkeypatch.setattr(mod, "get_systemd_pid", lambda _unit: None)
    monkeypatch.setattr(mod, "read_systemd_execstart", lambda _unit: "python live.py --live")
    monkeypatch.setattr(mod, "read_repo_configured_command", lambda _spec: "python live.py --paper")
    monkeypatch.setattr(mod, "_read_runtime_cmd", lambda _pid: None)
    monkeypatch.setattr(mod, "_read_process_start_utc", lambda _pid: None)
    monkeypatch.setattr(mod, "files_newer_than_process", lambda _pid, _paths: [])

    report = mod.build_service_report(spec, git_status)

    assert report.repo_configured_cmd == "python live.py --paper"
    assert report.restart_reasons == [
        "service_not_running",
        "installed_config_differs_from_repo",
    ]


def test_llm_stock_trader_spec_exists_and_tracks_symbol_ownership() -> None:
    mod = _load_module()
    spec = mod.SPECS["llm-stock-trader"]

    assert spec.manager == "supervisor"
    assert spec.symbols_flag == "--stock-symbols"
    assert spec.ownership_service_name == "llm-stock-trader"


def test_trading_server_spec_exists_and_tracks_repo_config() -> None:
    mod = _load_module()
    spec = mod.SPECS["trading-server"]

    assert spec.manager == "supervisor"
    assert spec.actual_name == "trading-server"
    assert spec.repo_config_path == REPO / "deployments" / "trading-server" / "supervisor.conf"
    assert "config/trading_server/accounts.json" in spec.watched_repo_files
