#!/usr/bin/env python3
"""Helper wrapper to boot the fal StockTrainer app, stream logs, and trigger training."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import requests
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise SystemExit(
        "The 'requests' package is required for run_and_train_fal.py. "
        "Install it in the active environment (e.g. `uv pip install requests`)."
    ) from exc


YELLOW = "\033[33m"
RESET = "\033[0m"
RULE = f"{YELLOW}{'━' * 94}{RESET}"


@dataclass
class FalOutputParser:
    """Streaming parser for `fal run` output."""

    ready_event: threading.Event = field(default_factory=threading.Event)
    endpoint_event: threading.Event = field(default_factory=threading.Event)
    sync_url: Optional[str] = None
    expect_sync_url: bool = False

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
        if any(
            marker in line
            for marker in (
                "Application startup complete",
                "Uvicorn running on",
                "Started server process",
            )
        ):
            self.ready_event.set()


def _print_cmd(prefix: str, cmd: Sequence[str]) -> None:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"{RULE}\n{prefix}: $ {rendered}\n{RULE}", flush=True)


def _load_payload(args: argparse.Namespace) -> Dict[str, object]:
    if args.payload_json and args.payload_file:
        raise SystemExit("Use either --payload-json or --payload-file, not both.")
    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if args.payload_json:
        return json.loads(args.payload_json)

    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return {
        "trainer": "hf",
        "run_name": f"faltrain_{now}",
        "do_sweeps": True,
        "sweeps": {"parallel_trials": getattr(args, "parallel_trials", 2)},
    }


def _append_endpoint_path(sync_url: str, endpoint_path: str) -> str:
    if not endpoint_path:
        return sync_url
    if sync_url.rstrip("/").endswith(endpoint_path.lstrip("/")):
        return sync_url
    return f"{sync_url.rstrip('/')}/{endpoint_path.lstrip('/')}"


def _monitor_process(
    process: subprocess.Popen[str],
    parser: FalOutputParser,
    stop_event: threading.Event,
) -> None:
    assert process.stdout is not None
    try:
        for raw_line in iter(process.stdout.readline, ""):
            if raw_line == "" and process.poll() is not None:
                break
            if raw_line:
                sys.stdout.write(raw_line)
                sys.stdout.flush()
                parser.feed(raw_line)
            if stop_event.is_set():
                break
    finally:
        try:
            process.stdout.close()
        except Exception:
            pass


def _cleanup(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _trigger_training(
    url: str,
    payload: Dict[str, object],
    auth_token: Optional[str],
    headers: Optional[Iterable[str]],
) -> requests.Response:
    req_headers: Dict[str, str] = {"Content-Type": "application/json"}
    if auth_token:
        req_headers["Authorization"] = auth_token
    if headers:
        for header in headers:
            if ":" not in header:
                raise SystemExit(f"Invalid header format: {header!r}")
            key, value = header.split(":", 1)
            req_headers[key.strip()] = value.strip()
    print(f"{RULE}\nTriggering training via {url}\nPayload:\n{json.dumps(payload, indent=2)}\n{RULE}")
    response = requests.post(url, json=payload, headers=req_headers)
    print(f"Response status: {response.status_code}")
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            print(json.dumps(response.json(), indent=2))
        except json.JSONDecodeError:
            print(response.text)
    else:
        print(response.text)
    return response


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch faltrain via fal run and trigger training.")
    parser.add_argument(
        "--fal-app",
        default="faltrain/app.py::StockTrainerApp",
        help="fal application to run (default: %(default)s)",
    )
    parser.add_argument(
        "--fal-binary",
        default="fal",
        help="fal CLI binary to invoke (default: %(default)s)",
    )
    parser.add_argument(
        "--payload-json",
        help="Inline JSON payload passed to /api/train.",
    )
    parser.add_argument(
        "--payload-file",
        help="Path to a JSON file with the request body.",
    )
    parser.add_argument(
        "--endpoint-path",
        default="/api/train",
        help="Endpoint path appended to the synchronous URL when absent (default: %(default)s).",
    )
    parser.add_argument(
        "--auth-token",
        help="Authorization header value (e.g. 'Key <token>' or 'Bearer <token>').",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Additional HTTP header in 'Name: value' form (can repeat).",
    )
    parser.add_argument(
        "--fal-arg",
        action="append",
        default=[],
        help="Extra argument to forward to `fal run` (can repeat).",
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=2,
        help="Default `sweeps.parallel_trials` when generating payload (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the fal run process alive after the training request finishes.",
    )
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
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    parser = FalOutputParser()
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_process,
        args=(process, parser, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    try:
        if not parser.endpoint_event.wait():
            raise RuntimeError("fal run terminated before emitting a synchronous endpoint URL.")
        sync_url = parser.sync_url
        if not sync_url:
            raise RuntimeError("Failed to capture synchronous endpoint URL from fal run output.")

        parser.ready_event.wait()
        train_url = _append_endpoint_path(sync_url, args.endpoint_path)
        _trigger_training(train_url, payload, args.auth_token, args.header)

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
        _cleanup(process)
        return 1
    finally:
        stop_event.set()
        if not args.keep_alive:
            _cleanup(process)
        monitor_thread.join(timeout=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

