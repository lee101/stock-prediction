from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "deploy_live_trader.sh"
WEEKLY_RETRAIN = REPO / "scripts" / "xgb_weekly_retrain.sh"
XGB_LAUNCH = REPO / "deployments" / "xgb-daily-trader-live" / "launch.sh"


def test_deploy_live_trader_shell_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_xgb_weekly_retrain_shell_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(WEEKLY_RETRAIN)], check=True)


def test_xgb_live_launch_sanitizes_break_glass_env_after_secrets() -> None:
    text = XGB_LAUNCH.read_text(encoding="utf-8")

    source_idx = text.index('source "$HOME/.secretbashrc"')
    unset_singleton_idx = text.index("unset ALPACA_SINGLETON_OVERRIDE")
    unset_death_spiral_idx = text.index("unset ALPACA_DEATH_SPIRAL_OVERRIDE")
    live_enable_idx = text.index("export ALLOW_ALPACA_LIVE_TRADING=1")

    assert source_idx < unset_singleton_idx < live_enable_idx
    assert source_idx < unset_death_spiral_idx < live_enable_idx


def test_deploy_live_trader_executes_happy_path_with_fake_supervisor(tmp_path: Path) -> None:
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    lock_path = tmp_path / "state" / "alpaca_live_writer.lock"
    history_log = tmp_path / "history" / "live_trader_history.log"
    report_dir = tmp_path / "reports"
    preflight = tmp_path / "preflight.py"

    preflight.write_text(
        "import json\n"
        "print(json.dumps({'reports': ["
        "{'service': 'xgb-daily-trader-live', 'safe_to_apply': True}"
        "]}))\n",
        encoding="utf-8",
    )
    (fakebin / "sudo").write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = \"-n\" ]; then shift; fi\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    (fakebin / "supervisorctl").write_text(
        "#!/usr/bin/env bash\n"
        "cmd=\"${1:-}\"\n"
        "unit=\"${2:-}\"\n"
        "case \"$cmd\" in\n"
        "  status)\n"
        "    if [ \"$unit\" = \"xgb-daily-trader-live\" ]; then\n"
        "      echo \"$unit RUNNING pid 1234, uptime 0:01:00\"\n"
        "    else\n"
        "      echo \"$unit STOPPED Not started\"\n"
        "    fi\n"
        "    ;;\n"
        "  restart|start)\n"
        "    mkdir -p \"$(dirname \"$LOCK_PATH\")\"\n"
        "    printf '{\"pid\":\"1234\"}\\n' > \"$LOCK_PATH\"\n"
        "    echo \"$unit: started\"\n"
        "    ;;\n"
        "  stop)\n"
        "    echo \"$unit: stopped\"\n"
        "    ;;\n"
        "  *)\n"
        "    echo \"unexpected supervisorctl $*\" >&2\n"
        "    exit 9\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    (fakebin / "sudo").chmod(0o755)
    (fakebin / "supervisorctl").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fakebin}:{os.environ['PATH']}",
        "REPO": str(tmp_path),
        "LOCK_PATH": str(lock_path),
        "HISTORY_LOG": str(history_log),
        "PREFLIGHT": str(preflight),
        "PREFLIGHT_REPORT_DIR": str(report_dir),
        "USER": "tester",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT), "--allow-unmodeled-live-sidecars", "xgb-daily-trader-live"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    reports = sorted(report_dir.glob("*.json"))
    assert len(reports) == 2
    for report in reports:
        assert json.loads(report.read_text(encoding="utf-8")) == {
            "reports": [
                {"service": "xgb-daily-trader-live", "safe_to_apply": True}
            ],
        }
    history = history_log.read_text(encoding="utf-8")
    assert "unit=xgb-daily-trader-live" in history
    assert "status=ok" in history
    assert "supervisor_pid=1234" in history
    assert "lock_pid=1234" in history
    initial_report = [path for path in reports if not path.name.endswith("_post.json")][0]
    post_report = [path for path in reports if path.name.endswith("_post.json")][0]
    assert f"preflight_report={initial_report}" in history
    assert f"post_preflight_report={post_report}" in history
    assert f"preflight report: {initial_report}" in result.stdout
    assert f"post-restart preflight report: {post_report}" in result.stdout


def test_deploy_live_trader_fails_if_post_preflight_still_needs_restart(
    tmp_path: Path,
) -> None:
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    lock_path = tmp_path / "state" / "alpaca_live_writer.lock"
    history_log = tmp_path / "history" / "live_trader_history.log"
    report_dir = tmp_path / "reports"
    preflight = tmp_path / "preflight.py"
    count_path = tmp_path / "preflight_count.txt"

    preflight.write_text(
        "import json\n"
        f"p = {str(count_path)!r}\n"
        "try:\n"
        "    n = int(open(p).read()) + 1\n"
        "except FileNotFoundError:\n"
        "    n = 1\n"
        "open(p, 'w').write(str(n))\n"
        "report = {'service': 'xgb-daily-trader-live', 'safe_to_apply': True}\n"
        "if n >= 2:\n"
        "    report['restart_reasons'] = ['runtime_live_env_mismatch']\n"
        "print(json.dumps({'reports': [report]}))\n",
        encoding="utf-8",
    )
    (fakebin / "sudo").write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = \"-n\" ]; then shift; fi\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    (fakebin / "supervisorctl").write_text(
        "#!/usr/bin/env bash\n"
        "cmd=\"${1:-}\"\n"
        "unit=\"${2:-}\"\n"
        "case \"$cmd\" in\n"
        "  status)\n"
        "    if [ \"$unit\" = \"xgb-daily-trader-live\" ]; then\n"
        "      echo \"$unit RUNNING pid 1234, uptime 0:01:00\"\n"
        "    else\n"
        "      echo \"$unit STOPPED Not started\"\n"
        "    fi\n"
        "    ;;\n"
        "  restart|start)\n"
        "    mkdir -p \"$(dirname \"$LOCK_PATH\")\"\n"
        "    printf '{\"pid\":\"1234\"}\\n' > \"$LOCK_PATH\"\n"
        "    echo \"$unit: started\"\n"
        "    ;;\n"
        "  stop)\n"
        "    echo \"$unit: stopped\"\n"
        "    ;;\n"
        "  *)\n"
        "    echo \"unexpected supervisorctl $*\" >&2\n"
        "    exit 9\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    (fakebin / "sudo").chmod(0o755)
    (fakebin / "supervisorctl").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fakebin}:{os.environ['PATH']}",
        "REPO": str(tmp_path),
        "LOCK_PATH": str(lock_path),
        "HISTORY_LOG": str(history_log),
        "PREFLIGHT": str(preflight),
        "PREFLIGHT_REPORT_DIR": str(report_dir),
        "USER": "tester",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT), "--allow-unmodeled-live-sidecars", "xgb-daily-trader-live"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 7
    reports = sorted(report_dir.glob("*.json"))
    assert len(reports) == 2
    post_report = [path for path in reports if path.name.endswith("_post.json")][0]
    history = history_log.read_text(encoding="utf-8")
    assert "status=post_preflight_unresolved" in history
    assert f"post_preflight_report={post_report}" in history
    assert "status=ok" not in history
    assert "runtime_live_env_mismatch" in result.stdout


def test_deploy_live_trader_fails_if_post_preflight_has_no_reports(
    tmp_path: Path,
) -> None:
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    lock_path = tmp_path / "state" / "alpaca_live_writer.lock"
    history_log = tmp_path / "history" / "live_trader_history.log"
    report_dir = tmp_path / "reports"
    preflight = tmp_path / "preflight.py"
    count_path = tmp_path / "preflight_count.txt"

    preflight.write_text(
        "import json\n"
        f"p = {str(count_path)!r}\n"
        "try:\n"
        "    n = int(open(p).read()) + 1\n"
        "except FileNotFoundError:\n"
        "    n = 1\n"
        "open(p, 'w').write(str(n))\n"
        "payload = {'reports': ["
        "{'service': 'xgb-daily-trader-live', 'safe_to_apply': True}"
        "]}\n"
        "if n >= 2:\n"
        "    payload = {'reports': []}\n"
        "print(json.dumps(payload))\n",
        encoding="utf-8",
    )
    (fakebin / "sudo").write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = \"-n\" ]; then shift; fi\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    (fakebin / "supervisorctl").write_text(
        "#!/usr/bin/env bash\n"
        "cmd=\"${1:-}\"\n"
        "unit=\"${2:-}\"\n"
        "case \"$cmd\" in\n"
        "  status)\n"
        "    if [ \"$unit\" = \"xgb-daily-trader-live\" ]; then\n"
        "      echo \"$unit RUNNING pid 1234, uptime 0:01:00\"\n"
        "    else\n"
        "      echo \"$unit STOPPED Not started\"\n"
        "    fi\n"
        "    ;;\n"
        "  restart|start)\n"
        "    mkdir -p \"$(dirname \"$LOCK_PATH\")\"\n"
        "    printf '{\"pid\":\"1234\"}\\n' > \"$LOCK_PATH\"\n"
        "    echo \"$unit: started\"\n"
        "    ;;\n"
        "  stop)\n"
        "    echo \"$unit: stopped\"\n"
        "    ;;\n"
        "  *)\n"
        "    echo \"unexpected supervisorctl $*\" >&2\n"
        "    exit 9\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    (fakebin / "sudo").chmod(0o755)
    (fakebin / "supervisorctl").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fakebin}:{os.environ['PATH']}",
        "REPO": str(tmp_path),
        "LOCK_PATH": str(lock_path),
        "HISTORY_LOG": str(history_log),
        "PREFLIGHT": str(preflight),
        "PREFLIGHT_REPORT_DIR": str(report_dir),
        "USER": "tester",
    }

    result = subprocess.run(
        ["bash", str(SCRIPT), "--allow-unmodeled-live-sidecars", "xgb-daily-trader-live"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 7
    history = history_log.read_text(encoding="utf-8")
    assert "status=post_preflight_unresolved" in history
    assert "status=ok" not in history
    assert "preflight report missing non-empty reports list" in result.stdout


def test_deploy_live_trader_runs_preflight_before_supervisor_mutations() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    preflight_idx = text.index("running preflight:")
    report_dir_idx = text.index('mkdir -p "$PREFLIGHT_REPORT_DIR"')
    report_path_idx = text.index('PREFLIGHT_REPORT_PATH="${PREFLIGHT_REPORT_DIR}/')
    stop_idx = text.index("sudo -n supervisorctl stop")
    restart_idx = text.index("sudo -n supervisorctl restart")
    start_idx = text.index("sudo -n supervisorctl start")

    assert report_dir_idx < preflight_idx
    assert report_path_idx < preflight_idx
    assert preflight_idx < stop_idx
    assert preflight_idx < restart_idx
    assert preflight_idx < start_idx
    assert 'python3 "$PREFLIGHT" --service "$TARGET" --fail-on-unsafe --json' in text
    assert "--allow-unmodeled-live-sidecars" in text
    assert "--allow-invalid-xgb-ensemble" in text
    assert "[--allow-invalid-xgb-ensemble] <unit_name>" in text
    assert "exit 6" in text


def test_deploy_live_trader_persists_preflight_report_in_history() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    preflight_run_idx = text.index('> "$PREFLIGHT_REPORT_PATH"')
    post_preflight_idx = text.index('> "$POST_PREFLIGHT_REPORT_PATH"')
    history_idx = text.index('_append_history "ok" "$final_pid" "$final_lock_pid"')

    assert "PREFLIGHT_REPORT_DIR" in text
    assert '_show_report_file "$PREFLIGHT_REPORT_PATH"' in text
    assert "preflight report: $PREFLIGHT_REPORT_PATH" in text
    assert "post-restart preflight report: $POST_PREFLIGHT_REPORT_PATH" in text
    assert 'preflight_report=${PREFLIGHT_REPORT_PATH:-none}' in text
    assert 'post_preflight_report=${POST_PREFLIGHT_REPORT_PATH:-none}' in text
    assert preflight_run_idx < history_idx
    assert post_preflight_idx < history_idx


def test_deploy_live_trader_post_preflight_parser_is_fail_closed() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    parser_idx = text.index("_post_preflight_has_unresolved_findings()")

    assert "if safe_to_apply is not True:" in text[parser_idx:]
    assert "if service != expected_service:" in text[parser_idx:]
    assert "missing expected service in preflight report" in text[parser_idx:]
    assert '_post_preflight_has_unresolved_findings "$POST_PREFLIGHT_REPORT_PATH" "$TARGET"' in text


def test_deploy_live_trader_records_failed_mutation_attempts_in_history() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '_append_history "preflight_failed"' in text
    assert '_append_history "stop_failed:$unit"' in text
    assert '_append_history "restart_failed"' in text
    assert '_append_history "start_failed"' in text
    assert '_append_history "lock_mismatch"' in text
    assert '_append_history "post_preflight_failed"' in text
    assert '_append_history "post_preflight_unresolved"' in text
    assert '_append_history "stopped_all"' in text

    assert text.index('_append_history "stop_failed:$unit"') < text.index("exit 3")
    restart_history_idx = text.index('_append_history "restart_failed"')
    restart_exit_idx = text.index("exit 4", restart_history_idx)
    start_history_idx = text.index('_append_history "start_failed"')
    start_exit_idx = text.index("exit 4", start_history_idx)
    assert restart_history_idx < restart_exit_idx
    assert start_history_idx < start_exit_idx
    assert text.index('_append_history "lock_mismatch"') < text.index("exit 5")
    assert text.index('_append_history "preflight_failed"') < text.index("exit 6")
    post_failed_idx = text.index('_append_history "post_preflight_failed"')
    post_failed_exit_idx = text.index("exit 7", post_failed_idx)
    post_unresolved_idx = text.index('_append_history "post_preflight_unresolved"')
    post_unresolved_exit_idx = text.index("exit 7", post_unresolved_idx)
    assert post_failed_idx < post_failed_exit_idx
    assert post_unresolved_idx < post_unresolved_exit_idx


def test_xgb_weekly_retrain_uses_guarded_deploy_after_preflight_before_rotation() -> None:
    text = WEEKLY_RETRAIN.read_text(encoding="utf-8")

    load_validation_idx = text.index("scripts/validate_xgb_ensemble.py")
    preflight_idx = text.index("preflight before live-dir rotation")
    rotate_idx = text.index('mv "${STAGE_DIR}" "${LIVE_DIR}"')
    normalize_idx = text.index('normalize_xgb_ensemble_manifest.py "${LIVE_DIR}"')
    deploy_idx = text.index('deploy_live_trader.sh"')
    prune_idx = text.index('name "alltrain_ensemble_gpu_prev_*"')

    assert load_validation_idx < preflight_idx
    assert preflight_idx < rotate_idx
    assert rotate_idx < normalize_idx
    assert normalize_idx < deploy_idx
    assert deploy_idx < prune_idx
    assert "--fail-on-unsafe" in text
    assert "--allow-unmodeled-live-sidecars" in text
    assert "sudo -n supervisorctl restart xgb-daily-trader-live" not in text


def test_xgb_weekly_retrain_rolls_back_live_dir_if_guarded_deploy_fails() -> None:
    text = WEEKLY_RETRAIN.read_text(encoding="utf-8")

    rotate_idx = text.index('mv "${STAGE_DIR}" "${LIVE_DIR}"')
    rollback_idx = text.index("rollback_live_dir_after_failed_redeploy()")
    deploy_rc_idx = text.index("deploy_rc=${PIPESTATUS[0]}")
    deploy_failure_idx = text.index('if [ "${deploy_rc}" -ne 0 ]; then')
    missing_script_idx = text.index("rollback_live_dir_after_failed_redeploy 5")

    assert rotate_idx < rollback_idx
    assert rollback_idx < deploy_rc_idx
    assert deploy_rc_idx < deploy_failure_idx
    assert deploy_failure_idx < missing_script_idx
    assert 'mv "${LIVE_DIR}" "${failed_dir}"' in text
    assert 'mv "${BACKUP_DIR}" "${LIVE_DIR}"' in text
    assert 'exit "${rc}"' in text


def test_xgb_weekly_retrain_attempts_recovery_deploy_after_rollback() -> None:
    text = WEEKLY_RETRAIN.read_text(encoding="utf-8")

    restore_idx = text.index('mv "${BACKUP_DIR}" "${LIVE_DIR}"')
    normalize_idx = text.index(
        'normalize_xgb_ensemble_manifest.py "${LIVE_DIR}"',
        restore_idx,
    )
    recovery_msg_idx = text.index("attempting recovery redeploy of restored previous live dir")
    recovery_cmd_idx = text.index("recovery_rc=${PIPESTATUS[0]}")
    exit_idx = text.index('exit "${rc}"')

    assert restore_idx < normalize_idx
    assert normalize_idx < recovery_msg_idx
    assert recovery_msg_idx < recovery_cmd_idx
    assert recovery_cmd_idx < exit_idx
    assert "OK recovery redeploy restored previous live artifacts" in text
    assert "WARN recovery redeploy failed" in text
