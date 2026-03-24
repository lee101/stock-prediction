"""Tests for RunPod GPU fallback selection."""

from scripts.binance_autoresearch_forever import PodManager
from src.runpod_client import Pod, build_gpu_fallback_types, parse_gpu_fallback_types, resolve_gpu_type


def test_build_gpu_fallback_types_resolves_and_dedupes():
    chain = build_gpu_fallback_types("4090", ["5090", "6000-ada", "4090", "h100-sxm"])
    assert chain == [
        resolve_gpu_type("4090"),
        resolve_gpu_type("5090"),
        resolve_gpu_type("6000-ada"),
        resolve_gpu_type("h100-sxm"),
    ]


def test_build_gpu_fallback_types_allows_explicit_disable():
    assert build_gpu_fallback_types("4090", []) == [resolve_gpu_type("4090")]


def test_parse_gpu_fallback_types_supports_default_and_disable():
    assert parse_gpu_fallback_types("") is None
    assert parse_gpu_fallback_types(None) is None
    assert parse_gpu_fallback_types("none") == []
    assert parse_gpu_fallback_types("5090, l4") == ["5090", "l4"]


def test_pod_manager_retries_capacity_errors_with_fallback():
    class FakeClient:
        def __init__(self):
            self.calls: list[str] = []
            self.terminated: list[str] = []

        def create_pod(self, config):
            self.calls.append(config.gpu_type)
            if len(self.calls) == 1:
                raise RuntimeError(
                    "RunPod GraphQL error: [{'message': 'This machine does not have the resources to deploy your pod.'}]"
                )
            return Pod(id="pod-2", name=config.name, status="CREATED", gpu_type=config.gpu_type)

        def wait_for_pod(self, pod_id):
            return Pod(
                id=pod_id,
                name="pod-2",
                status="RUNNING",
                gpu_type=resolve_gpu_type("5090"),
                ssh_host="example.host",
                ssh_port=2222,
            )

        def terminate_pod(self, pod_id):
            self.terminated.append(pod_id)

    fake_client = FakeClient()
    mgr = PodManager(gpu_type="4090", gpu_fallback_types=["5090"])
    mgr.client = fake_client

    host, port = mgr.ensure_ready()

    assert fake_client.calls == [resolve_gpu_type("4090"), resolve_gpu_type("5090")]
    assert mgr.active_gpu_type == resolve_gpu_type("5090")
    assert (host, port) == ("example.host", 2222)


def test_pod_manager_uses_runtime_gpu_name_when_available():
    class FakeClient:
        def create_pod(self, config):
            return Pod(id="pod-4500", name=config.name, status="CREATED", gpu_type=config.gpu_type)

        def wait_for_pod(self, pod_id):
            return Pod(
                id=pod_id,
                name="pod-4500",
                status="RUNNING",
                gpu_type="NVIDIA RTX PRO 4500 Blackwell",
                ssh_host="fallback.host",
                ssh_port=2200,
            )

        def terminate_pod(self, pod_id):
            raise AssertionError("terminate_pod should not be called")

    mgr = PodManager(gpu_type="4090", gpu_fallback_types=["rtx-pro-4500"])
    mgr.client = FakeClient()

    host, port = mgr.ensure_ready()

    assert mgr.active_gpu_type == "NVIDIA RTX PRO 4500 Blackwell"
    assert (host, port) == ("fallback.host", 2200)
