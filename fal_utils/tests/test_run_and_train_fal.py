import types
from argparse import Namespace

import run_and_train_fal as runner


def test_parser_extracts_sync_and_ready():
    parser = runner.FalOutputParser()
    assert parser.sync_url is None
    parser.feed("Some log line")
    parser.feed("Synchronous Endpoints:")
    assert parser.sync_url is None
    parser.feed("https://fal.run/test-app")
    assert parser.sync_url == "https://fal.run/test-app"
    assert parser.endpoint_event.is_set()
    parser.feed("Application startup complete.")
    assert parser.ready_event.is_set()


def test_append_endpoint_path_appends_when_missing():
    url = "https://fal.run/app"
    result = runner._append_endpoint_path(url, "/api/train")
    assert result == "https://fal.run/app/api/train"


def test_append_endpoint_path_no_duplicate():
    url = "https://fal.run/app/api/train"
    result = runner._append_endpoint_path(url, "/api/train")
    assert result == url


def test_load_payload_defaults_include_parallel_trials(monkeypatch):
    fake_now = types.SimpleNamespace(strftime=lambda fmt: "20250101_000000")
    monkeypatch.setattr(runner, "datetime", types.SimpleNamespace(utcnow=lambda: fake_now))
    args = Namespace(payload_json=None, payload_file=None, parallel_trials=4)
    payload = runner._load_payload(args)
    assert payload["trainer"] == "hf"
    assert payload["do_sweeps"] is True
    sweeps = payload["sweeps"]
    assert sweeps["parallel_trials"] == 4
    assert payload["run_name"].startswith("faltrain_20250101_000000")
