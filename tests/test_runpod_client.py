"""Tests for src/runpod_client.py — all API calls are mocked."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.runpod_client import (
    DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
    DEFAULT_POD_READY_TIMEOUT_SECONDS,
    GPU_ALIASES,
    HOURLY_RATES,
    MAX_POD_READY_POLL_INTERVAL_SECONDS,
    TRAINING_DOCKER_IMAGE,
    TRAINING_GPU_TYPES,
    Pod,
    PodConfig,
    RunPodClient,
    get_hourly_rate,
    get_supported_gpu_types,
    resolve_gpu_preferences,
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


def test_resolve_gpu_preferences_uses_default_fallbacks():
    gpu_preferences = resolve_gpu_preferences("4090")

    assert gpu_preferences[0] == "NVIDIA GeForce RTX 4090"
    assert "NVIDIA RTX PRO 4500 Ada Generation" in gpu_preferences


def test_resolve_gpu_preferences_parses_cli_fallbacks():
    assert resolve_gpu_preferences("4090", "6000-ada,l4") == (
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA L4",
    )


def test_resolve_gpu_preferences_allows_disabling_fallbacks():
    assert resolve_gpu_preferences("4090", "none") == ("NVIDIA GeForce RTX 4090",)


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


def test_create_pod_with_fallback_retries_capacity_errors():
    client, _ = _make_client()

    attempted_gpu_types: list[str] = []

    def _fake_create_pod(config: PodConfig) -> Pod:
        attempted_gpu_types.append(config.gpu_type)
        if config.gpu_type == "4090":
            raise RuntimeError("RunPod does not have the resources to deploy your pod right now")
        return Pod(id="pod-l4", name=config.name, status="CREATED", gpu_type=config.gpu_type)

    client.create_pod = _fake_create_pod  # type: ignore[method-assign]

    pod = client.create_pod_with_fallback(
        PodConfig(name="trainer", gpu_type="4090"),
        ("4090", "L4"),
    )

    assert attempted_gpu_types == ["4090", "L4"]
    assert pod.id == "pod-l4"
    assert pod.gpu_type == "L4"


def test_create_pod_with_fallback_raises_non_capacity_error_without_retry():
    client, _ = _make_client()

    attempted_gpu_types: list[str] = []

    def _fake_create_pod(config: PodConfig) -> Pod:
        attempted_gpu_types.append(config.gpu_type)
        raise RuntimeError("invalid template id")

    client.create_pod = _fake_create_pod  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="invalid template id"):
        client.create_pod_with_fallback(
            PodConfig(name="trainer", gpu_type="4090"),
            ("4090", "L4"),
        )

    assert attempted_gpu_types == ["4090"]


def test_create_pod_with_fallback_raises_after_all_capacity_errors():
    client, _ = _make_client()

    attempted_gpu_types: list[str] = []

    def _fake_create_pod(config: PodConfig) -> Pod:
        attempted_gpu_types.append(config.gpu_type)
        raise RuntimeError(f"RunPod capacity exhausted for {config.gpu_type}")

    client.create_pod = _fake_create_pod  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Could not provision any requested RunPod GPU"):
        client.create_pod_with_fallback(
            PodConfig(name="trainer", gpu_type="4090"),
            ("4090", "L4"),
        )

    assert attempted_gpu_types == ["4090", "L4"]


def test_create_pod_with_fallback_rejects_empty_preferences():
    client, _ = _make_client()

    with pytest.raises(ValueError, match="gpu_preferences must contain at least one GPU type"):
        client.create_pod_with_fallback(
            PodConfig(name="trainer", gpu_type="4090"),
            (),
        )


def test_create_ready_pod_with_fallback_waits_for_ready_pod():
    client, _ = _make_client()
    created_with: list[object] = []

    def _fake_create_pod_with_fallback(config: PodConfig, gpu_preferences: tuple[str, ...]) -> Pod:
        created_with.append((config, gpu_preferences))
        return Pod(id="pod-ready", name=config.name, status="CREATED", gpu_type="L4")

    def _fake_wait_for_pod(pod_id: str, *, timeout: int = 0, poll_interval: int = 0) -> Pod:
        created_with.append((pod_id, timeout, poll_interval))
        return Pod(id=pod_id, name="trainer", status="RUNNING", gpu_type="L4")

    client.create_pod_with_fallback = _fake_create_pod_with_fallback  # type: ignore[method-assign]
    client.wait_for_pod = _fake_wait_for_pod  # type: ignore[method-assign]

    pod = client.create_ready_pod_with_fallback(
        PodConfig(name="trainer", gpu_type="4090"),
        ("4090", "L4"),
        timeout=45,
        poll_interval=3,
    )

    assert pod.id == "pod-ready"
    create_call, wait_call = created_with
    assert create_call[0].name == "trainer"
    assert create_call[1] == ("4090", "L4")
    assert wait_call == ("pod-ready", 45, 3)


def test_create_ready_pod_with_fallback_terminates_pod_when_wait_fails():
    client, _ = _make_client()
    terminated: list[str] = []

    def _fake_create_pod_with_fallback(config: PodConfig, gpu_preferences: tuple[str, ...]) -> Pod:
        return Pod(id="pod-ready", name=config.name, status="CREATED", gpu_type="L4")

    def _fake_wait_for_pod(pod_id: str, *, timeout: int = 0, poll_interval: int = 0) -> Pod:
        raise TimeoutError("ssh never appeared")

    def _fake_terminate_pod(pod_id: str) -> None:
        terminated.append(pod_id)

    client.create_pod_with_fallback = _fake_create_pod_with_fallback  # type: ignore[method-assign]
    client.wait_for_pod = _fake_wait_for_pod  # type: ignore[method-assign]
    client.terminate_pod = _fake_terminate_pod  # type: ignore[method-assign]

    with pytest.raises(TimeoutError, match="ssh never appeared"):
        client.create_ready_pod_with_fallback(
            PodConfig(name="trainer", gpu_type="4090"),
            ("4090", "L4"),
            timeout=45,
            poll_interval=3,
        )

    assert terminated == ["pod-ready"]


def test_create_ready_pod_with_fallback_does_not_attempt_cleanup_when_create_fails():
    client, _ = _make_client()
    terminated: list[str] = []

    def _fake_create_pod_with_fallback(config: PodConfig, gpu_preferences: tuple[str, ...]) -> Pod:
        raise RuntimeError("provisioning failed")

    def _fake_terminate_pod(pod_id: str) -> None:
        terminated.append(pod_id)

    client.create_pod_with_fallback = _fake_create_pod_with_fallback  # type: ignore[method-assign]
    client.terminate_pod = _fake_terminate_pod  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="provisioning failed"):
        client.create_ready_pod_with_fallback(
            PodConfig(name="trainer", gpu_type="4090"),
            ("4090", "L4"),
        )

    assert terminated == []


def test_create_ready_pod_with_fallback_adds_note_when_cleanup_fails():
    client, _ = _make_client()

    def _fake_create_pod_with_fallback(config: PodConfig, gpu_preferences: tuple[str, ...]) -> Pod:
        return Pod(id="pod-ready", name=config.name, status="CREATED", gpu_type="L4")

    def _fake_wait_for_pod(pod_id: str, *, timeout: int = 0, poll_interval: int = 0) -> Pod:
        raise TimeoutError("ssh never appeared")

    def _fake_terminate_pod(pod_id: str) -> None:
        raise RuntimeError("terminate failed")

    client.create_pod_with_fallback = _fake_create_pod_with_fallback  # type: ignore[method-assign]
    client.wait_for_pod = _fake_wait_for_pod  # type: ignore[method-assign]
    client.terminate_pod = _fake_terminate_pod  # type: ignore[method-assign]

    with pytest.raises(TimeoutError, match="ssh never appeared") as excinfo:
        client.create_ready_pod_with_fallback(
            PodConfig(name="trainer", gpu_type="4090"),
            ("4090", "L4"),
            timeout=45,
            poll_interval=3,
        )

    assert excinfo.value.__notes__ is not None
    assert any(
        "failed to terminate pod pod-ready after readiness wait failed" in note
        and "terminate failed" in note
        for note in excinfo.value.__notes__
    )


# ---------------------------------------------------------------------------
# get_pod
# ---------------------------------------------------------------------------

def test_get_pod_parses_ssh_host_and_port():
    client, mock_session = _make_client()
    client._gpu_types_cache[False] = [
        {
            "id": "NVIDIA_A100_PCIE_80GB",
            "displayName": "NVIDIA A100 80GB PCIe",
            "memoryInGb": 80,
        }
    ]
    client._gpu_display_name_cache["NVIDIA_A100_PCIE_80GB"] = "NVIDIA A100 80GB PCIe"

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
    assert pod.gpu_type == "NVIDIA A100 80GB PCIe"
    assert pod.ssh_host == "1.2.3.4"
    assert pod.ssh_port == 10022
    assert pod.public_ip == "1.2.3.4"


def test_get_pod_uses_cached_gpu_display_name_map():
    client, mock_session = _make_client()
    client._gpu_display_name_cache["NVIDIA_A100_PCIE_80GB"] = "NVIDIA A100 80GB PCIe"

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
                    ],
                },
            }
        }
    })
    mock_session.post.return_value = resp

    pod = client.get_pod("pod-123")

    assert pod.gpu_type == "NVIDIA A100 80GB PCIe"


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


def test_get_pod_keeps_gpu_id_when_display_name_not_cached():
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {
            "pod": {
                "id": "pod-789",
                "name": "pod-789",
                "desiredStatus": "RUNNING",
                "runtime": {
                    "gpus": [{"id": "GPU_UNCACHED"}],
                    "ports": [],
                },
            }
        }
    })
    mock_session.post.return_value = resp

    pod = client.get_pod("pod-789")

    assert pod.gpu_type == "GPU_UNCACHED"


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
         patch(
             "src.runpod_client.time.monotonic",
             side_effect=[
                 0,
                 0,
                 DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
                 20,
                 30,
                 DEFAULT_POD_READY_TIMEOUT_SECONDS + 1,
                 DEFAULT_POD_READY_TIMEOUT_SECONDS + 1,
             ],
         ):
        with pytest.raises(TimeoutError, match="pod-slow"):
            client.wait_for_pod(
                "pod-slow",
                timeout=DEFAULT_POD_READY_TIMEOUT_SECONDS,
                poll_interval=DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
            )


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
         patch("src.runpod_client.time.monotonic", side_effect=[0, 0, DEFAULT_POD_READY_POLL_INTERVAL_SECONDS]):
        pod = client.wait_for_pod(
            "pod-fast",
            timeout=DEFAULT_POD_READY_TIMEOUT_SECONDS,
            poll_interval=DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
        )

    assert pod.id == "pod-fast"
    assert pod.ssh_host == "5.6.7.8"
    assert pod.ssh_port == 22222


def test_wait_for_pod_logs_only_on_state_change():
    client, mock_session = _make_client()

    starting_resp = _json_response({
        "data": {
            "pod": {
                "id": "pod-progress",
                "name": "pod-progress",
                "desiredStatus": "STARTING",
                "runtime": None,
            }
        }
    })
    running_resp = _json_response({
        "data": {
            "pod": {
                "id": "pod-progress",
                "name": "pod-progress",
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
    mock_session.post.side_effect = [starting_resp, starting_resp, running_resp]

    with patch("src.runpod_client.time.sleep"), \
         patch("src.runpod_client.time.monotonic", side_effect=[0, 0, 1, 2]), \
         patch("builtins.print") as mock_print:
        pod = client.wait_for_pod(
            "pod-progress",
            timeout=DEFAULT_POD_READY_TIMEOUT_SECONDS,
            poll_interval=DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
        )

    assert pod.id == "pod-progress"
    assert mock_print.call_count == 2
    assert "status=STARTING" in mock_print.call_args_list[0].args[0]
    assert "status=RUNNING" in mock_print.call_args_list[1].args[0]


def test_wait_for_pod_backs_off_when_state_is_unchanged():
    client, _ = _make_client()
    client.get_pod = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            Pod(id="pod-stuck", name="pod-stuck", status="STARTING"),
            Pod(id="pod-stuck", name="pod-stuck", status="STARTING"),
            Pod(id="pod-stuck", name="pod-stuck", status="STARTING"),
        ]
    )

    with patch("src.runpod_client.time.sleep") as mock_sleep, patch(
        "src.runpod_client.time.monotonic",
        side_effect=[0, 0, 1, 2, 31, 31],
    ):
        with pytest.raises(TimeoutError, match="pod-stuck"):
            client.wait_for_pod(
                "pod-stuck",
                timeout=30,
                poll_interval=DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
            )

    assert [call.args[0] for call in mock_sleep.call_args_list] == [
        DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
        DEFAULT_POD_READY_POLL_INTERVAL_SECONDS * 2,
        MAX_POD_READY_POLL_INTERVAL_SECONDS,
    ]


def test_wait_for_pod_resets_backoff_after_state_change():
    client, _ = _make_client()
    client.get_pod = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            Pod(id="pod-progress", name="pod-progress", status="STARTING"),
            Pod(id="pod-progress", name="pod-progress", status="STARTING"),
            Pod(id="pod-progress", name="pod-progress", status="PROVISIONING"),
            Pod(
                id="pod-progress",
                name="pod-progress",
                status="RUNNING",
                ssh_host="5.6.7.8",
                ssh_port=22222,
            ),
        ]
    )

    with patch("src.runpod_client.time.sleep") as mock_sleep, patch(
        "src.runpod_client.time.monotonic",
        side_effect=[0, 0, 1, 2, 3],
    ):
        pod = client.wait_for_pod(
            "pod-progress",
            timeout=30,
            poll_interval=DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
        )

    assert pod.status == "RUNNING"
    assert [call.args[0] for call in mock_sleep.call_args_list] == [
        DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
        DEFAULT_POD_READY_POLL_INTERVAL_SECONDS * 2,
        DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
    ]


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
         patch(
             "src.runpod_client.time.monotonic",
             side_effect=[
                 0,
                 0,
                 DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
                 20,
                 30,
                 DEFAULT_POD_READY_TIMEOUT_SECONDS + 1,
                 DEFAULT_POD_READY_TIMEOUT_SECONDS + 1,
             ],
         ):
        with pytest.raises(TimeoutError) as exc_info:
            client.wait_for_pod(
                "pod-timeout-test",
                timeout=DEFAULT_POD_READY_TIMEOUT_SECONDS,
                poll_interval=DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
            )

    msg = str(exc_info.value)
    assert "pod-timeout-test" in msg
    assert str(DEFAULT_POD_READY_TIMEOUT_SECONDS) in msg


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


def test_find_gpu_type_id_normalizes_runpod_display_variants():
    client, mock_session = _make_client()

    gpu_list_resp = _json_response({
        "data": {"gpuTypes": [
            {"id": "NVIDIA RTX PRO 4500 Blackwell", "displayName": "RTX PRO 4500", "memoryInGb": 24},
            {"id": "NVIDIA H100 80GB HBM3", "displayName": "H100 SXM", "memoryInGb": 80},
        ]}
    })
    mock_session.post.return_value = gpu_list_resp

    assert client.find_gpu_type_id("NVIDIA RTX PRO 4500 Ada Generation") == "NVIDIA RTX PRO 4500 Blackwell"
    assert client.find_gpu_type_id("NVIDIA H100 SXM") == "NVIDIA H100 80GB HBM3"


def test_find_gpu_type_id_reuses_cached_match_without_rewalking_gpu_list():
    client, _ = _make_client()
    client._gpu_types_cache[False] = [
        {
            "id": "NVIDIA_GEFORCE_RTX_5090",
            "displayName": "NVIDIA GeForce RTX 5090",
            "memoryInGb": 32,
        }
    ]

    assert client.find_gpu_type_id("5090") == "NVIDIA_GEFORCE_RTX_5090"

    with patch.object(
        client,
        "_get_cached_gpu_types",
        side_effect=AssertionError("GPU listing should not be consulted after a cached match"),
    ):
        assert client.find_gpu_type_id("5090") == "NVIDIA_GEFORCE_RTX_5090"


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


def test_list_gpu_types_normalizes_raw_gpu_records():
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {
            "gpuTypes": [
                {
                    "id": 123,
                    "displayName": "NVIDIA GeForce RTX 4090",
                    "memoryInGb": "24",
                    "lowestPrice": {
                        "minimumBidPrice": "0.69",
                        "unInterruptablePrice": "1.05",
                    },
                },
                {
                    "id": "",
                    "displayName": "",
                    "memoryInGb": "bad",
                },
                {
                    "id": "NVIDIA_L4",
                    "displayName": "NVIDIA L4",
                    "memoryInGb": 24,
                    "lowestPrice": "unexpected",
                },
            ]
        }
    })
    mock_session.post.return_value = resp

    gpus = client.list_gpu_types()

    assert gpus == [
        {
            "id": "123",
            "displayName": "NVIDIA GeForce RTX 4090",
            "memoryInGb": 24,
            "lowestPrice": {
                "minimumBidPrice": 0.69,
                "unInterruptablePrice": 1.05,
            },
        },
        {
            "id": "NVIDIA_L4",
            "displayName": "NVIDIA L4",
            "memoryInGb": 24,
        },
    ]


def test_list_gpu_types_reuses_cached_response_for_same_pricing_mode():
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {
            "gpuTypes": [
                {"id": "NVIDIA_A100_PCIE_80GB", "displayName": "NVIDIA A100 80GB PCIe", "memoryInGb": 80}
            ]
        }
    })
    mock_session.post.return_value = resp

    first = client.list_gpu_types(include_pricing=False)
    second = client.list_gpu_types(include_pricing=False)

    assert first == second
    assert mock_session.post.call_count == 1


def test_list_gpu_types_returns_isolated_copies_from_cache():
    client, mock_session = _make_client()

    resp = _json_response({
        "data": {
            "gpuTypes": [
                {
                    "id": "NVIDIA_A100_PCIE_80GB",
                    "displayName": "NVIDIA A100 80GB PCIe",
                    "memoryInGb": 80,
                    "lowestPrice": {"minimumBidPrice": 1.20, "unInterruptablePrice": 1.64},
                }
            ]
        }
    })
    mock_session.post.return_value = resp

    first = client.list_gpu_types()
    first[0]["displayName"] = "mutated"
    first[0]["lowestPrice"]["minimumBidPrice"] = 99.0

    second = client.list_gpu_types()

    assert second == [
        {
            "id": "NVIDIA_A100_PCIE_80GB",
            "displayName": "NVIDIA A100 80GB PCIe",
            "memoryInGb": 80,
            "lowestPrice": {
                "minimumBidPrice": 1.2,
                "unInterruptablePrice": 1.64,
            },
        }
    ]
    assert mock_session.post.call_count == 1


def test_list_gpu_types_coalesces_concurrent_cache_miss():
    client, _ = _make_client()

    call_count = 0
    call_count_lock = threading.Lock()
    start = threading.Event()
    results: list[list[dict] | None] = [None, None]

    def _fake_graphql(query: str) -> dict:
        nonlocal call_count
        start.wait(timeout=1)
        with call_count_lock:
            call_count += 1
        time.sleep(0.05)
        return {
            "gpuTypes": [
                {"id": "NVIDIA_A100_PCIE_80GB", "displayName": "NVIDIA A100 80GB PCIe", "memoryInGb": 80}
            ]
        }

    client._graphql = _fake_graphql  # type: ignore[method-assign]

    def _worker(slot: int) -> None:
        start.wait(timeout=1)
        results[slot] = client.list_gpu_types(include_pricing=False)

    threads = [
        threading.Thread(target=_worker, args=(0,)),
        threading.Thread(target=_worker, args=(1,)),
    ]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()

    assert call_count == 1
    assert results[0] == results[1] == [
        {"id": "NVIDIA_A100_PCIE_80GB", "displayName": "NVIDIA A100 80GB PCIe", "memoryInGb": 80}
    ]


def test_graphql_serializes_shared_session_access_across_threads():
    client, mock_session = _make_client()

    start = threading.Event()
    state_lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    def _fake_post(*_args, **_kwargs):
        nonlocal active_calls, max_active_calls
        start.wait(timeout=1)
        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
        time.sleep(0.05)
        with state_lock:
            active_calls -= 1
        return _json_response({"data": {"gpuTypes": []}})

    mock_session.post.side_effect = _fake_post

    def _worker() -> None:
        client._graphql("query { gpuTypes { id } }")

    threads = [
        threading.Thread(target=_worker),
        threading.Thread(target=_worker),
    ]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()

    assert mock_session.post.call_count == 2
    assert max_active_calls == 1


def test_create_pod_reuses_cached_gpu_listing_across_multiple_calls():
    client, mock_session = _make_client()

    gpu_list_resp = _json_response({
        "data": {
            "gpuTypes": [
                {"id": "NVIDIA_GEFORCE_RTX_4090", "displayName": "NVIDIA GeForce RTX 4090", "memoryInGb": 24},
                {"id": "NVIDIA_GEFORCE_RTX_5090", "displayName": "NVIDIA GeForce RTX 5090", "memoryInGb": 32},
            ]
        }
    })
    create_resp_one = _json_response({
        "data": {"podFindAndDeployOnDemand": {"id": "pod-4090", "name": "trainer-4090", "desiredStatus": "CREATED"}}
    })
    create_resp_two = _json_response({
        "data": {"podFindAndDeployOnDemand": {"id": "pod-5090", "name": "trainer-5090", "desiredStatus": "CREATED"}}
    })
    mock_session.post.side_effect = [gpu_list_resp, create_resp_one, create_resp_two]

    first = client.create_pod(PodConfig(name="trainer-4090", gpu_type="4090"))
    second = client.create_pod(PodConfig(name="trainer-5090", gpu_type="5090"))

    assert first.id == "pod-4090"
    assert second.id == "pod-5090"
    assert mock_session.post.call_count == 3
