from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.config_io import (
    apply_worksteal_config_overrides,
    build_worksteal_config_override_mapping,
    build_worksteal_config_from_result,
    build_worksteal_config_from_args,
    build_worksteal_config_explanation,
    default_best_config_path,
    default_best_overrides_path,
    load_worksteal_config,
    load_worksteal_config_overrides,
    maybe_handle_worksteal_config_output,
    render_worksteal_config_overrides_yaml,
    render_worksteal_config_explanation_yaml,
    render_worksteal_config_yaml,
    write_worksteal_config_file_or_warn,
    write_worksteal_config_overrides_file_or_warn,
)
from binance_worksteal.strategy import WorkStealConfig


def test_load_worksteal_config_overrides_supports_nested_config(tmp_path):
    path = tmp_path / "audit_config.yaml"
    path.write_text(
        "config:\n"
        "  dip_pct: 0.18\n"
        "  trailing_stop_pct: 0.03\n",
        encoding="utf-8",
    )

    overrides = load_worksteal_config_overrides(path)

    assert overrides == {"dip_pct": 0.18, "trailing_stop_pct": 0.03}


def test_load_worksteal_config_applies_overrides_to_base_config(tmp_path):
    path = tmp_path / "audit_config.yaml"
    path.write_text(
        "dip_pct: 0.18\n"
        "sma_filter_period: 7\n",
        encoding="utf-8",
    )

    config = load_worksteal_config(
        path,
        base_config=WorkStealConfig(dip_pct=0.20, sma_filter_period=20, stop_loss_pct=0.10),
    )

    assert config.dip_pct == pytest.approx(0.18)
    assert config.sma_filter_period == 7
    assert config.stop_loss_pct == pytest.approx(0.10)


def test_load_worksteal_config_normalizes_tuple_fields(tmp_path):
    path = tmp_path / "audit_config.yaml"
    path.write_text(
        "dip_pct_fallback:\n"
        "  - 0.18\n"
        "  - 0.15\n",
        encoding="utf-8",
    )

    config = load_worksteal_config(path)

    assert config.dip_pct_fallback == (0.18, 0.15)


def test_load_worksteal_config_rejects_unknown_fields(tmp_path):
    path = tmp_path / "audit_config.yaml"
    path.write_text(
        "dip_pct: 0.18\n"
        "unknown_field: 123\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported WorkStealConfig fields"):
        load_worksteal_config_overrides(path)


def test_load_worksteal_config_rejects_invalid_yaml(tmp_path):
    path = tmp_path / "audit_config.yaml"
    path.write_text("config: [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid config file format"):
        load_worksteal_config_overrides(path)


def test_load_worksteal_config_rejects_missing_file(tmp_path):
    path = tmp_path / "missing_config.yaml"

    with pytest.raises(FileNotFoundError, match=f"Config file not found: {path}"):
        load_worksteal_config_overrides(path)


def test_apply_worksteal_config_overrides_returns_same_config_for_empty_overrides():
    config = WorkStealConfig(dip_pct=0.20)

    updated = apply_worksteal_config_overrides(config, {})

    assert updated is config


def test_build_worksteal_config_from_args_applies_config_file_then_cli_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "dip_pct: 0.18\n"
        "sma_filter_period: 7\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(dip_pct=0.25, config_file=str(path))

    config = build_worksteal_config_from_args(
        base_config=WorkStealConfig(dip_pct=0.20, sma_filter_period=20, stop_loss_pct=0.10),
        config_file=args.config_file,
        args=args,
        raw_argv=["--config-file", str(path), "--dip-pct", "0.25"],
        flag_to_field={"--dip-pct": "dip_pct"},
    )

    assert config.dip_pct == pytest.approx(0.25)
    assert config.sma_filter_period == 7
    assert config.stop_loss_pct == pytest.approx(0.10)


def test_render_worksteal_config_yaml_emits_reusable_config_mapping():
    rendered = render_worksteal_config_yaml(
        WorkStealConfig(
            dip_pct=0.18,
            sma_filter_period=7,
            dip_pct_fallback=(0.18, 0.15),
        )
    )

    payload = yaml.safe_load(rendered)

    assert payload["config"]["dip_pct"] == pytest.approx(0.18)
    assert payload["config"]["sma_filter_period"] == 7
    assert payload["config"]["dip_pct_fallback"] == [0.18, 0.15]


def test_build_worksteal_config_override_mapping_only_includes_changed_fields():
    overrides = build_worksteal_config_override_mapping(
        WorkStealConfig(
            dip_pct=0.18,
            sma_filter_period=7,
            dip_pct_fallback=(0.18, 0.15),
        )
    )

    assert overrides == {
        "dip_pct": 0.18,
        "dip_pct_fallback": [0.18, 0.15],
        "sma_filter_period": 7,
    }


def test_render_worksteal_config_overrides_yaml_emits_only_changed_fields():
    rendered = render_worksteal_config_overrides_yaml(
        WorkStealConfig(
            dip_pct=0.18,
            sma_filter_period=7,
        )
    )

    payload = yaml.safe_load(rendered)

    assert payload["config"]["dip_pct"] == pytest.approx(0.18)
    assert payload["config"]["sma_filter_period"] == 7
    assert "initial_cash" not in payload["config"]


def test_build_worksteal_config_from_result_applies_swept_fields_to_base_config():
    config = build_worksteal_config_from_result(
        base_config=WorkStealConfig(
            initial_cash=5000.0,
            base_asset_symbol="ETHUSD",
            sma_check_method="current",
        ),
        result_row={
            "dip_pct": 0.18,
            "profit_target_pct": 0.12,
            "ignored_metric": 4.2,
        },
        swept_fields=["dip_pct", "profit_target_pct"],
    )

    assert config.initial_cash == pytest.approx(5000.0)
    assert config.base_asset_symbol == "ETHUSD"
    assert config.sma_check_method == "current"
    assert config.dip_pct == pytest.approx(0.18)
    assert config.profit_target_pct == pytest.approx(0.12)


def test_default_best_config_path_uses_output_stem(tmp_path):
    path = default_best_config_path(tmp_path / "sweep_results.csv")

    assert path == tmp_path / "sweep_results.best_config.yaml"


def test_default_best_overrides_path_uses_output_stem(tmp_path):
    path = default_best_overrides_path(tmp_path / "sweep_results.csv")

    assert path == tmp_path / "sweep_results.best_overrides.yaml"


def test_write_worksteal_config_file_or_warn_writes_yaml(tmp_path, capsys):
    path = tmp_path / "recommended.yaml"

    written = write_worksteal_config_file_or_warn(
        path,
        WorkStealConfig(dip_pct=0.18, sma_filter_period=7),
    )

    assert written == path
    assert "Wrote recommended config YAML" in capsys.readouterr().out
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["config"]["dip_pct"] == pytest.approx(0.18)
    assert payload["config"]["sma_filter_period"] == 7


def test_write_worksteal_config_file_or_warn_emits_warning_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "binance_worksteal.config_io.write_worksteal_config_file",
        lambda path, config: (_ for _ in ()).throw(OSError("disk full")),
    )

    written = write_worksteal_config_file_or_warn("ignored.yaml", WorkStealConfig())

    assert written is None
    assert capsys.readouterr().out.strip() == (
        "WARN: failed to write recommended config YAML to ignored.yaml: disk full"
    )


def test_write_worksteal_config_overrides_file_or_warn_writes_yaml(tmp_path, capsys):
    path = tmp_path / "recommended_overrides.yaml"

    written = write_worksteal_config_overrides_file_or_warn(
        path,
        WorkStealConfig(dip_pct=0.18, sma_filter_period=7),
    )

    assert written == path
    assert "Wrote recommended overrides YAML" in capsys.readouterr().out
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["config"]["dip_pct"] == pytest.approx(0.18)
    assert payload["config"]["sma_filter_period"] == 7
    assert "initial_cash" not in payload["config"]


def test_write_worksteal_config_overrides_file_or_warn_emits_warning_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "binance_worksteal.config_io.write_worksteal_config_overrides_file",
        lambda path, config, base_config=None: (_ for _ in ()).throw(OSError("disk full")),
    )

    written = write_worksteal_config_overrides_file_or_warn("ignored.yaml", WorkStealConfig())

    assert written is None
    assert capsys.readouterr().out.strip() == (
        "WARN: failed to write recommended overrides YAML to ignored.yaml: disk full"
    )


def test_build_worksteal_config_explanation_tracks_sources(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "dip_pct: 0.18\n"
        "sma_filter_period: 7\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(dip_pct=0.25, config_file=str(path))

    explanation = build_worksteal_config_explanation(
        base_config=WorkStealConfig(dip_pct=0.20, sma_filter_period=20, stop_loss_pct=0.10),
        config_file=args.config_file,
        args=args,
        raw_argv=["--config-file", str(path), "--dip-pct", "0.25"],
        flag_to_field={"--dip-pct": "dip_pct"},
    )

    assert explanation["config"]["dip_pct"] == pytest.approx(0.25)
    assert explanation["sources"]["dip_pct"] == "cli"
    assert explanation["sources"]["sma_filter_period"] == "config_file"
    assert explanation["changed_fields"]["dip_pct"]["config_file_value"] == pytest.approx(0.18)
    assert explanation["changed_fields"]["dip_pct"]["cli_value"] == pytest.approx(0.25)
    assert explanation["changed_fields"]["sma_filter_period"]["value"] == 7


def test_render_worksteal_config_explanation_yaml_emits_human_readable_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dip_pct: 0.18\n", encoding="utf-8")
    args = argparse.Namespace(dip_pct=0.25, config_file=str(path))

    rendered = render_worksteal_config_explanation_yaml(
        build_worksteal_config_explanation(
            base_config=WorkStealConfig(dip_pct=0.20),
            config_file=args.config_file,
            args=args,
            raw_argv=["--config-file", str(path), "--dip-pct", "0.25"],
            flag_to_field={"--dip-pct": "dip_pct"},
        )
    )

    payload = yaml.safe_load(rendered)
    assert payload["changed_fields"]["dip_pct"]["source"] == "cli"
    assert payload["config_file_overrides"]["dip_pct"] == pytest.approx(0.18)


def test_maybe_handle_worksteal_config_output_skips_builder_without_flags():
    args = argparse.Namespace(print_config=False, explain_config=False, dip_pct=0.25, config_file=None)
    called = False

    def build_config():
        nonlocal called
        called = True
        return WorkStealConfig(dip_pct=0.25)

    result = maybe_handle_worksteal_config_output(
        args=args,
        build_config=build_config,
        base_config=WorkStealConfig(dip_pct=0.20),
        config_file=None,
        raw_argv=[],
        flag_to_field={"--dip-pct": "dip_pct"},
    )

    assert result is None
    assert called is False


def test_maybe_handle_worksteal_config_output_prints_explanation_with_adjuster(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("dip_pct: 0.18\n", encoding="utf-8")
    args = argparse.Namespace(print_config=False, explain_config=True, dip_pct=0.25, config_file=str(path))

    rc = maybe_handle_worksteal_config_output(
        args=args,
        build_config=lambda: WorkStealConfig(dip_pct=0.25),
        base_config=WorkStealConfig(dip_pct=0.20),
        config_file=args.config_file,
        raw_argv=["--config-file", str(path), "--dip-pct", "0.25"],
        flag_to_field={"--dip-pct": "dip_pct"},
        explain_adjuster=lambda explanation: explanation.update({"note": "adjusted"}),
    )

    assert rc == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["config"]["dip_pct"] == pytest.approx(0.25)
    assert payload["changed_fields"]["dip_pct"]["source"] == "cli"
    assert payload["note"] == "adjusted"


def test_maybe_handle_worksteal_config_output_reports_builder_errors(capsys):
    args = argparse.Namespace(print_config=True, explain_config=False, config_file=None)

    rc = maybe_handle_worksteal_config_output(
        args=args,
        build_config=lambda: (_ for _ in ()).throw(ValueError("bad config")),
        base_config=WorkStealConfig(),
        config_file=None,
        raw_argv=["--print-config"],
        flag_to_field={},
    )

    assert rc == 1
    assert capsys.readouterr().out.strip() == "ERROR: bad config"
