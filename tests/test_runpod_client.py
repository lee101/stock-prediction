"""Tests for src/runpod_client.py — all API calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.runpod_client import (
    GPU_ALIASES,
    HOURLY_RATES,
    TRAINING_DOCKER_IMAGE,
    TRAINING_GPU_TYPES,
    Pod,
    PodConfig,
    RunPodClient,
    get_hourly_rate,
    get_supported_gpu_types,
    resolve_gpu_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(api_key: str = "test-key") -> tuple[RunPodClient, MagicMock]:
    """Return a RunPodClient with a mock session."""
    mock_session = MagicMock(spec=requests.Session)
    client = RunPodClient(api_key=api_key, session=mock_session)
    return client, mock_session


def _json_response(data: dict) -> MagicMock:
    """Build a mock Response that returns *data* from .json()."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = data
    resp.content = b"..."
    return resp


# ---------------------------------------------------------------------------
# resolve_gpu_type
# ---------------------------------------------------------------------------

def test_resolve_gpu_type_known_alias():
    assert resolve_gpu_type("a100") == "NVIDIA A100 80GB PCIe"


def test_resolve_gpu_type_passthrough_unknown():
    full_name = "NVIDIA A100 80GB PCIe"
    assert resolve_gpu_type(full_name) == full_name


def test_resolve_gpu_type_all_aliases():
    for alias, expected in TRAINING_GPU_TYPES.items():
        assert resolve_gpu_type(alias) == expected


# ---------------------------------------------------------------------------
# RunPodClient construction
# ---------------------------------------------------------------------------

def test_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="RUNPOD_API_KEY not set"):
        RunPodClient()


def test_client_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "env-key-123")
    mock_session = MagicMock(spec=requests.Session)
    client = RunPodClient(session=mock_session)
    assert client.api_key == "env-key-123"


def test_client_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "env-key")
    mock_session = MagicMock(spec=requests.Session)
    client = RunPodClient(api_key="explicit-key", session=mock_session)
    assert client.api_key == "explicit-key"


# ---------------------------------------------------------------------------
# create_pod
# ---------------------------------------------------------------------------

def test_create_pod_sends_correct_graphql_payload():
    client, mock_session = _make_client()

    # First call: find_gpu_type_id → list_gpu_types
    gpu_list_resp = _json_response({
        "data": {"gpuTypes": [{"id": "NVIDIA_A100_PCIE_80GB", "displayName": "NVIDIA A100 80GB PCIe", "memoryInGb": 80}]}
    })
    # Second call: podFindAndDeployOnDemand mutation
    create_resp = _json_response({
        "data": {"podFindAndDeployOnDemand": {"id": "pod-abc", "name": "trainer-1", "desiredStatus": "CREATED"}}
    })
    mock_session.post.side_effect = [gpu_list_resp, create_resp]

    config = PodConfig(
        name="trainer-1",
        gpu_type="NVIDIA A100 80GB PCIe",
        gpu_count=1,
        volume_size=200,
        container_disk=50,
    )
    pod = client.create_pod(config)

    assert pod.id == "pod-abc"
    assert pod.name == "trainer-1"
    assert pod.status == "CREATED"

    # Inspect the mutation payload
    _, mutation_kwargs = mock_session.post.call_args_list[1]
    payload = mutation_kwargs["json"]
    variables = payload["variables"]
    inp = variables["input"]

    assert inp["gpuTypeId"] == "NVIDIA_A100_PCIE_80GB"
    assert inp["gpuCount"] == 1
    assert inp["volumeInGb"] == 200
    assert inp["containerDiskInGb"] == 50
    assert inp["startSsh"] is True
    # No template → image should be set
    assert inp.get("imageName") == TRAINING_DOCKER_IMAGE
    assert "templateId" not in inp


def test_create_pod_with_template_id_omits_image():
    client, mock_session = _make_client()

    gpu_list_resp = _json_response({
        "data": {"gpuTypes": [{"id": "NVIDIA_A100_PCIE_80GB", "displayName": "NVIDIA A100 80GB PCIe", "memoryInGb": 80}]}
    })
    create_resp = _json_response({
        "data": {"podFindAndDeployOnDemand": {"id": "pod-xyz", "name": "tmpl-pod", "desiredStatus": "CREATED"}}
    })
    mock_session.post.side_effect = [gpu_list_resp, create_resp]

    config = PodConfig(name="tmpl-pod", template_id="my-template-id")
    client.create_pod(config)

    _, mutation_kwargs = mock_session.post.call_args_list[1]
    payload = mutation_kwargs["json"]
    inp = payload["variables"]["input"]

    assert inp["templateId"] == "my-template-id"
    assert "imageName" not in inp


# ---------------------------------------------------------------------------
# get_pod
# ---------------------------------------------------------------------------

def test_get_pod_parses_ssh_host_and_port():
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {
            "pod": {
                "id": "pod-123",
                "name": "my-pod",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "gpus": [{"id": "NVIDIA_A100_PCIE_80GB"}],
                    "ports": [
                        {"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 22, "publicPort": 10022, "type": "tcp"},
                        {"ip": "1.2.3.4", "isIpPublic": True, "privatePort": 8888, "publicPort": 18888, "type": "http"},
                    ],
                },
            }
        }
    })
    mock_session.post.return_value = resp

    pod = client.get_pod("pod-123")

    assert pod.id == "pod-123"
    assert pod.status == "RUNNING"
    assert pod.ssh_host == "1.2.3.4"
    assert pod.ssh_port == 10022
    assert pod.public_ip == "1.2.3.4"


def test_get_pod_returns_terminated_on_http_error():
    client, mock_session = _make_client()

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
    mock_session.post.return_value = mock_resp

    pod = client.get_pod("dead-pod")

    assert pod.id == "dead-pod"
    assert pod.status == "TERMINATED"


def test_get_pod_no_ssh_port_when_port_not_public():
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {
            "pod": {
                "id": "pod-456",
                "name": "pod-456",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "gpus": [],
                    "ports": [
                        # Not public — should be skipped
                        {"ip": "10.0.0.1", "isIpPublic": False, "privatePort": 22, "publicPort": 10022, "type": "tcp"},
                    ],
                },
            }
        }
    })
    mock_session.post.return_value = resp

    pod = client.get_pod("pod-456")
    assert pod.ssh_host == ""
    assert pod.ssh_port == 0


# ---------------------------------------------------------------------------
# list_pods
# ---------------------------------------------------------------------------

def test_list_pods_returns_pod_objects():
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {
            "myself": {
                "pods": [
                    {"id": "p1", "name": "pod-one", "desiredStatus": "RUNNING"},
                    {"id": "p2", "name": "pod-two", "desiredStatus": "EXITED"},
                ]
            }
        }
    })
    mock_session.post.return_value = resp

    pods = client.list_pods()

    assert len(pods) == 2
    assert isinstance(pods[0], Pod)
    assert pods[0].id == "p1"
    assert pods[0].name == "pod-one"
    assert pods[0].status == "RUNNING"
    assert pods[1].id == "p2"
    assert pods[1].status == "EXITED"


def test_list_pods_returns_empty_when_no_pods():
    client, mock_session = _make_client()

    resp = _json_response({"data": {"myself": {"pods": []}}})
    mock_session.post.return_value = resp

    pods = client.list_pods()
    assert pods == []


# ---------------------------------------------------------------------------
# wait_for_pod
# ---------------------------------------------------------------------------

def test_wait_for_pod_raises_timeout_when_never_ready():
    client, mock_session = _make_client()

    # Pod always returns STARTING with no SSH info
    resp = _json_response({
        "data": {
            "pod": {
                "id": "pod-slow",
                "name": "pod-slow",
                "desiredStatus": "STARTING",
                "runtime": None,
            }
        }
    })
    mock_session.post.return_value = resp

    with patch("src.runpod_client.time.sleep"), \
         patch("src.runpod_client.time.monotonic", side_effect=[0, 0, 10, 20, 30, 901, 901]):
        with pytest.raises(TimeoutError, match="pod-slow"):
            client.wait_for_pod("pod-slow", timeout=900, poll_interval=10)


def test_wait_for_pod_returns_when_running():
    client, mock_session = _make_client()

    running_resp = _json_response({
        "data": {
            "pod": {
                "id": "pod-fast",
                "name": "pod-fast",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "gpus": [{"id": "GPU_ID"}],
                    "ports": [
                        {"ip": "5.6.7.8", "isIpPublic": True, "privatePort": 22, "publicPort": 22222, "type": "tcp"},
                    ],
                },
            }
        }
    })
    mock_session.post.return_value = running_resp

    with patch("src.runpod_client.time.sleep"), \
         patch("src.runpod_client.time.monotonic", side_effect=[0, 0, 10]):
        pod = client.wait_for_pod("pod-fast", timeout=900, poll_interval=10)

    assert pod.id == "pod-fast"
    assert pod.ssh_host == "5.6.7.8"
    assert pod.ssh_port == 22222


# ---------------------------------------------------------------------------
# PodConfig defaults
# ---------------------------------------------------------------------------

def test_pod_config_default_image():
    config = PodConfig(name="test")
    assert config.image == TRAINING_DOCKER_IMAGE


def test_pod_config_default_template_id_is_none():
    config = PodConfig(name="test")
    assert config.template_id is None


# ---------------------------------------------------------------------------
# RTX 5090 support
# ---------------------------------------------------------------------------

def test_5090_in_gpu_aliases():
    assert "5090" in GPU_ALIASES
    assert GPU_ALIASES["5090"] == "NVIDIA GeForce RTX 5090"


def test_training_gpu_types_is_same_as_gpu_aliases():
    """TRAINING_GPU_TYPES is kept as a backwards-compatible reference to GPU_ALIASES."""
    assert TRAINING_GPU_TYPES is GPU_ALIASES


def test_5090_in_hourly_rates():
    assert "NVIDIA GeForce RTX 5090" in HOURLY_RATES
    assert HOURLY_RATES["NVIDIA GeForce RTX 5090"] == 1.25


def test_resolve_5090_alias():
    assert resolve_gpu_type("5090") == "NVIDIA GeForce RTX 5090"


def test_get_supported_gpu_types_returns_list_with_5090():
    types = get_supported_gpu_types()
    assert isinstance(types, list)
    assert "5090" in types


def test_get_supported_gpu_types_contains_all_aliases():
    types = get_supported_gpu_types()
    for alias in GPU_ALIASES:
        assert alias in types


# ---------------------------------------------------------------------------
# get_hourly_rate — safe cost lookup
# ---------------------------------------------------------------------------

def test_get_hourly_rate_known_gpu():
    assert get_hourly_rate("NVIDIA GeForce RTX 4090") == 0.69


def test_get_hourly_rate_5090():
    assert get_hourly_rate("NVIDIA GeForce RTX 5090") == 1.25


def test_get_hourly_rate_unknown_gpu_returns_zero():
    """Unknown GPU names must not raise; they return 0.0."""
    assert get_hourly_rate("NVIDIA RTX 9999") == 0.0


# ---------------------------------------------------------------------------
# wait_for_pod — improved timeout error message
# ---------------------------------------------------------------------------

def test_wait_for_pod_timeout_message_contains_pod_id():
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {
            "pod": {
                "id": "pod-timeout-test",
                "name": "pod-timeout-test",
                "desiredStatus": "STARTING",
                "runtime": None,
            }
        }
    })
    mock_session.post.return_value = resp

    with patch("src.runpod_client.time.sleep"), \
         patch("src.runpod_client.time.monotonic", side_effect=[0, 0, 10, 20, 30, 901, 901]):
        with pytest.raises(TimeoutError) as exc_info:
            client.wait_for_pod("pod-timeout-test", timeout=900, poll_interval=10)

    msg = str(exc_info.value)
    assert "pod-timeout-test" in msg
    assert "900" in msg  # timeout value present


# ---------------------------------------------------------------------------
# create_pod — alias resolution
# ---------------------------------------------------------------------------

def test_create_pod_resolves_5090_alias():
    """Passing '5090' as gpu_type should resolve to the full name before the API call."""
    client, mock_session = _make_client()

    gpu_list_resp = _json_response({
        "data": {"gpuTypes": [
            {"id": "NVIDIA_GEFORCE_RTX_5090", "displayName": "NVIDIA GeForce RTX 5090", "memoryInGb": 32}
        ]}
    })
    create_resp = _json_response({
        "data": {"podFindAndDeployOnDemand": {"id": "pod-5090", "name": "train-5090", "desiredStatus": "CREATED"}}
    })
    mock_session.post.side_effect = [gpu_list_resp, create_resp]

    config = PodConfig(name="train-5090", gpu_type="5090")
    pod = client.create_pod(config)

    assert pod.id == "pod-5090"
    assert pod.status == "CREATED"

    # Confirm create mutation was called with the resolved GPU type ID
    _, mutation_kwargs = mock_session.post.call_args_list[1]
    inp = mutation_kwargs["json"]["variables"]["input"]
    assert inp["gpuTypeId"] == "NVIDIA_GEFORCE_RTX_5090"


# ---------------------------------------------------------------------------
# list_gpu_types — health-check / pricing query
# ---------------------------------------------------------------------------

def test_list_gpu_types_returns_list():
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {
            "gpuTypes": [
                {
                    "id": "NVIDIA_A100_PCIE_80GB",
                    "displayName": "NVIDIA A100 80GB PCIe",
                    "memoryInGb": 80,
                    "lowestPrice": {"minimumBidPrice": 1.20, "unInterruptablePrice": 1.64},
                },
                {
                    "id": "NVIDIA_H100_80GB_HBM3",
                    "displayName": "NVIDIA H100 80GB HBM3",
                    "memoryInGb": 80,
                    "lowestPrice": {"minimumBidPrice": 2.99, "unInterruptablePrice": 3.89},
                },
            ]
        }
    })
    mock_session.post.return_value = resp

    gpus = client.list_gpu_types()

    assert len(gpus) == 2
    assert gpus[0]["id"] == "NVIDIA_A100_PCIE_80GB"
    assert gpus[0]["displayName"] == "NVIDIA A100 80GB PCIe"
    assert gpus[0]["memoryInGb"] == 80
    assert gpus[0]["lowestPrice"]["unInterruptablePrice"] == 1.64


def test_list_gpu_types_without_pricing_omits_lowest_price_from_query():
    """include_pricing=False should not request lowestPrice field."""
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {"gpuTypes": [{"id": "NVIDIA_A100_PCIE_80GB", "displayName": "NVIDIA A100 80GB PCIe", "memoryInGb": 80}]}
    })
    mock_session.post.return_value = resp

    client.list_gpu_types(include_pricing=False)

    _, call_kwargs = mock_session.post.call_args
    query_str = call_kwargs["json"]["query"]
    assert "lowestPrice" not in query_str


def test_list_gpu_types_with_pricing_includes_lowest_price_in_query():
    """include_pricing=True (default) should include lowestPrice in the query."""
    client, mock_session = _make_client()

    resp = _json_response({"data": {"gpuTypes": []}})
    mock_session.post.return_value = resp

    client.list_gpu_types(include_pricing=True)

    _, call_kwargs = mock_session.post.call_args
    query_str = call_kwargs["json"]["query"]
    assert "lowestPrice" in query_str
    assert "minimumBidPrice" in query_str
    assert "unInterruptablePrice" in query_str


def test_list_gpu_types_returns_empty_on_empty_response():
    client, mock_session = _make_client()

    resp = _json_response({"data": {"gpuTypes": []}})
    mock_session.post.return_value = resp

    gpus = client.list_gpu_types()
    assert gpus == []
