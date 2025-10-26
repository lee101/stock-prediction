#!/usr/bin/env python3
"""Launch the fal market simulator app and trigger in-process trading simulation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'requests' package is required for run_and_train_fal_marketsimulator.py. "
        "Install it via `uv pip install requests` in the active environment."
    ) from exc


YELLOW = "\033[33m"
RESET = "\033[0m"
RULE = f"{YELLOW}{'━' * 94}{RESET}"


class FalOutputParser:
    def __init__(self) -> None:
        self.ready_event = threading.Event()
        self.endpoint_event = threading.Event()
        self.sync_url: Optional[str] = None
        self.expect_sync_url = False

    def feed(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if "Synchronous Endpoints:" in line:
            self.expect_sync_url = True
            return
        if self.expect_sync_url and line.startswith("https://"):
            self.sync_url = line.strip()
            self.endpoint_event.set()
            self.expect_sync_url = False
        readiness_markers = (
            "Application startup complete",
            "Uvicorn running on",
            "Started server process",
            "==> Running",
        )
        if any(marker in line for marker in readiness_markers):
            self.ready_event.set()


def _print_cmd(prefix: str, cmd: Sequence[str]) -> None:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"{RULE}\n{prefix}: $ {rendered}\n{RULE}", flush=True)


def _append_endpoint(sync_url: str, endpoint_path: str) -> str:
    if sync_url.rstrip("/").endswith(endpoint_path.lstrip("/")):
        return sync_url
    return f"{sync_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


def _load_payload(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "symbols": args.symbols,
        "steps": args.steps,
        "step_size": args.step_size,
        "initial_cash": args.initial_cash,
        "top_k": args.top_k,
        "kronos_only": args.kronos_only,
        "compact_logs": args.compact_logs,
    }


def _trigger(url: str, payload: Dict[str, object], headers: Iterable[str]) -> requests.Response:
    req_headers = {"Content-Type": "application/json"}
    for header in headers:
        if ":" not in header:
            raise SystemExit(f"Invalid header format: {header!r}")
        key, value = header.split(":", 1)
        req_headers[key.strip()] = value.strip()

    print(f"{RULE}\nTriggering simulation via {url}\nPayload:\n{json.dumps(payload, indent=2)}\n{RULE}")
    response = requests.post(url, json=payload, headers=req_headers)
    print(f"Response status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except Exception:
        print(response.text)
    return response


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch fal market simulator and run trade loop.")
    parser.add_argument("--fal-app", default="falmarket/app.py::MarketSimulatorApp")
    parser.add_argument("--fal-binary", default="fal")
    parser.add_argument("--endpoint-path", default="/api/simulate")
    parser.add_argument("--fal-arg", action="append", default=[])
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--keep-alive", action="store_true")

    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--kronos-only", action="store_true")
    parser.add_argument("--compact-logs", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = _load_payload(args)

    cmd: List[str] = [args.fal_binary, "run", args.fal_app]
    if args.fal_arg:
        cmd.extend(args.fal_arg)

    _print_cmd("Starting fal run", cmd)
    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
        env=env,
    )

    parser = FalOutputParser()
    stop_event = threading.Event()

    def _monitor() -> None:
        assert process.stdout is not None
        try:
            for raw in iter(process.stdout.readline, ""):
                if raw == "" and process.poll() is not None:
                    break
                if raw:
                    sys.stdout.write(raw)
                    sys.stdout.flush()
                    parser.feed(raw)
                if stop_event.is_set():
                    break
        finally:
            if process.stdout is not None:
                process.stdout.close()

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    try:
        if not parser.endpoint_event.wait(timeout=120):
            raise RuntimeError("fal run terminated before emitting a synchronous endpoint URL.")
        if parser.sync_url is None:
            raise RuntimeError("Failed to capture synchronous endpoint URL from fal run output.")

        if not parser.ready_event.wait(timeout=120):
            print(
                "Fal run did not emit an explicit readiness marker; proceeding after endpoint discovery.",
                flush=True,
            )
        url = _append_endpoint(parser.sync_url, args.endpoint_path)
        _trigger(url, payload, args.header)

        if args.keep_alive:
            print("Keeping fal process alive. Press Ctrl+C to stop.")
            try:
                while True:
                    if process.poll() is not None:
                        break
                    signal.pause()
            except (KeyboardInterrupt, SystemExit):
                pass
        else:
            if process.poll() is None:
                print("Waiting for fal run process to finish...")
                process.wait()
    except KeyboardInterrupt:
        print("Interrupted. Cleaning up...")
    except Exception as exc:
        print(f"Error: {exc}")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return 1
    finally:
        stop_event.set()
        if not args.keep_alive and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        monitor.join(timeout=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
