from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from marketsimulator.provisioning.cli import app


runner = CliRunner()


class _FakeVastClient:
    def __init__(self, *_, **__):
        self.search_calls = []
        self.instances = {}

    def search_offers(self, filters):
        self.search_calls.append(filters)
        return [{"id": 1, "gpu_name": "RTX_3090"}]

    def create_instance(self, offer_id, **kwargs):
        self.instances[offer_id] = kwargs
        return 42

    def wait_for_status(self, instance_id):
        return {"id": instance_id, "actual_status": "running", "public_ipaddr": "1.2.3.4"}

    def get_instance(self, instance_id):
        return {"id": instance_id, "ssh_host": "ssh.example", "ssh_port": 2222}


class _FakeRunPodClient:
    def __init__(self, *_, **__):
        self.calls = MagicMock()

    def create_pod(self, request):
        self.calls.create_pod = request
        return {"id": "pod-1"}

    def get_pod(self, pod_id):
        return {"id": pod_id, "publicIp": "4.3.2.1", "portMappings": {"22": 10022}}

    def runsync(self, endpoint_id, payload):
        self.calls.runsync = (endpoint_id, payload)
        return {"status": "COMPLETED", "output": {"result": 123}}


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "vast-key")
    monkeypatch.setenv("RUNPOD_API_KEY", "runpod-key")
    monkeypatch.setenv("DOCKER_IMAGE", "repo/image:tag")


def test_vast_search_cli(monkeypatch):
    fake_client = _FakeVastClient()
    monkeypatch.setattr("marketsimulator.provisioning.cli.VastClient", lambda *_: fake_client)

    result = runner.invoke(app, ["vast", "search", "--gpu", "RTX_4090", "--limit", "1"])

    assert result.exit_code == 0
    assert '"gpu_name": "RTX_3090"' in result.stdout
    assert fake_client.search_calls  # ensure invoked


def test_vast_rent_cli(monkeypatch):
    fake_client = _FakeVastClient()
    monkeypatch.setattr("marketsimulator.provisioning.cli.VastClient", lambda *_: fake_client)

    result = runner.invoke(
        app,
        [
            "vast",
            "rent",
            "123",
            "--disk-gb",
            "30",
            "--volume-gb",
            "50",
            "--portal-external-port",
            "32000",
        ],
    )

    assert result.exit_code == 0
    assert "Created instance 42." in result.stdout
    assert fake_client.instances[123]["disk_gb"] == 30


def test_runpod_pod_create_cli(monkeypatch):
    fake_client = _FakeRunPodClient()
    monkeypatch.setattr("marketsimulator.provisioning.cli.RunPodClient", lambda *_: fake_client)

    result = runner.invoke(
        app,
        [
            "runpod",
            "pod-create",
            "--name",
            "marketsim",
            "--gpu-types",
            "NVIDIA GeForce RTX 3090",
            "--env",
            "PORT=80",
        ],
    )

    assert result.exit_code == 0
    assert '"id": "pod-1"' in result.stdout
    request = fake_client.calls.create_pod
    assert request.env == {"PORT": "80"}


def test_runpod_runsync_cli(monkeypatch):
    fake_client = _FakeRunPodClient()
    monkeypatch.setattr("marketsimulator.provisioning.cli.RunPodClient", lambda *_: fake_client)

    result = runner.invoke(
        app,
        ["runpod", "runsync", "endpoint-1", "--symbol", "QQQ", "--window", "512"],
    )

    assert result.exit_code == 0
    assert '"result": 123' in result.stdout
    assert fake_client.calls.runsync == ("endpoint-1", {"symbol": "QQQ", "window": 512})
