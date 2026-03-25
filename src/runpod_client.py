"""RunPod API client for RL trading system remote GPU training."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import requests


RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
RUNPOD_REST_URL = "https://rest.runpod.io/v1"

TRAINING_DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

GPU_ALIASES: dict[str, str] = {
    "a100": "NVIDIA A100 80GB PCIe",
    "a100-sxm": "NVIDIA A100-SXM4-80GB",
    "h100": "NVIDIA H100 80GB HBM3",
    "h100-sxm": "NVIDIA H100 SXM",
    "4090": "NVIDIA GeForce RTX 4090",
    "5090": "NVIDIA GeForce RTX 5090",
    "rtx-pro-4500": "NVIDIA RTX PRO 4500 Ada Generation",
    "4500-ada": "NVIDIA RTX PRO 4500 Ada Generation",
    "l40s": "NVIDIA L40S",
    "l40": "NVIDIA L40",
    "l4": "NVIDIA L4",
    "a40": "NVIDIA A40",
    "6000-ada": "NVIDIA RTX 6000 Ada Generation",
    "rtx6000": "NVIDIA RTX 6000 Ada Generation",
}

DEFAULT_GPU_FALLBACKS: tuple[str, ...] = (
    "rtx-pro-4500",
    "6000-ada",
    "5090",
    "h100-sxm",
    "l4",
)

# Backwards-compatible alias
TRAINING_GPU_TYPES = GPU_ALIASES

HOURLY_RATES: dict[str, float] = {
    "NVIDIA A100 80GB PCIe": 1.19,
    "NVIDIA A100-SXM4-80GB": 1.39,
    "NVIDIA H100 80GB HBM3": 1.99,
    "NVIDIA H100 SXM": 2.69,
    "NVIDIA GeForce RTX 4090": 0.34,
    "NVIDIA GeForce RTX 5090": 0.69,
    "NVIDIA L40S": 0.79,
    "NVIDIA L40": 0.69,
    "NVIDIA A40": 0.35,
    "NVIDIA RTX 6000 Ada Generation": 0.74,
}


def resolve_gpu_type(alias: str) -> str:
    """Resolve a short GPU alias to its full display name.

    If the alias is not found in GPU_ALIASES it is returned unchanged,
    allowing callers to pass the full name directly.
    """
    return GPU_ALIASES.get(alias, alias)


def get_supported_gpu_types() -> list[str]:
    """Return the list of supported GPU alias keys."""
    return list(GPU_ALIASES.keys())


def build_gpu_fallback_types(primary: str, fallbacks: Optional[Sequence[str]] = None) -> list[str]:
    """Return a de-duplicated, resolved GPU preference chain."""
    resolved: list[str] = []
    seen: set[str] = set()
    fallback_candidates = DEFAULT_GPU_FALLBACKS if fallbacks is None else fallbacks
    for candidate in [primary, *fallback_candidates]:
        if not candidate:
            continue
        name = resolve_gpu_type(str(candidate).strip())
        if not name or name in seen:
            continue
        seen.add(name)
        resolved.append(name)
    return resolved


def parse_gpu_fallback_types(value: Optional[str]) -> Optional[list[str]]:
    """Parse a comma-separated fallback GPU list.

    Returns ``None`` when the default fallback chain should be used and
    ``[]`` when fallbacks are explicitly disabled via ``none``.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.lower() == "none":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _gpu_match_key(text: str) -> str:
    normalized = text.lower()
    for token in (
        "nvidia",
        "geforce",
        "generation",
        "workstation",
        "server",
        "edition",
        "blackwell",
        "ada",
        "max-q",
        "maxq",
    ):
        normalized = normalized.replace(token, " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def is_capacity_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "does not have the resources to deploy your pod" in text
        or "insufficient resources" in text
        or "no longer any instances available" in text
        or "supply_constraint" in text
        or ("capacity" in text and "runpod" in text)
    )


def get_hourly_rate(gpu_full_name: str) -> float:
    """Return the hourly rate for a GPU by its full display name, or 0.0 if unknown."""
    return HOURLY_RATES.get(gpu_full_name, 0.0)


@dataclass(slots=True)
class PodConfig:
    name: str
    gpu_type: str = "NVIDIA A100 80GB PCIe"
    gpu_count: int = 1
    image: str = TRAINING_DOCKER_IMAGE
    volume_size: int = 120
    container_disk: int = 40
    cloud_type: str = "COMMUNITY"  # COMMUNITY for direct SSH; SECURE for H100/datacenter
    template_id: Optional[str] = None
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Pod:
    id: str
    name: str
    status: str
    gpu_type: str = ""
    ssh_host: str = ""
    ssh_port: int = 0
    public_ip: str = ""


class RunPodClient:
    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not set")
        self.session = session or requests.Session()

    def _graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        payload: dict = {"query": query}
        if variables:
            payload["variables"] = variables
        response = self.session.post(
            RUNPOD_GRAPHQL_URL,
            params={"api_key": self.api_key},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")
        return data.get("data", {})

    def _rest_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _rest_post(self, path: str, payload: Optional[dict] = None) -> dict:
        response = self.session.post(
            f"{RUNPOD_REST_URL}{path}",
            headers=self._rest_headers(),
            json=payload or {},
            timeout=60,
        )
        response.raise_for_status()
        if response.content:
            return response.json()
        return {}

    def list_gpu_types(self, include_pricing: bool = True) -> list[dict]:
        """Return available GPU types from RunPod.

        Each entry has at least: ``id``, ``displayName``, ``memoryInGb``.
        When *include_pricing* is True (default) the query also fetches
        ``lowestPrice { minimumBidPrice unInterruptablePrice }`` so callers
        can compare on-demand cost across GPU types.

        Example::

            from src.runpod_client import RunPodClient
            c = RunPodClient(api_key="...")
            print(c.list_gpu_types())
        """
        if include_pricing:
            query = (
                "query { gpuTypes { id displayName memoryInGb"
                " lowestPrice { minimumBidPrice unInterruptablePrice } } }"
            )
        else:
            query = "query { gpuTypes { id displayName memoryInGb } }"
        data = self._graphql(query)
        return data.get("gpuTypes", [])

    def find_gpu_type_id(self, name_substr: str) -> str:
        needle = name_substr.lower()
        needle_key = _gpu_match_key(name_substr)
        normalized_matches: list[str] = []
        for gpu in self.list_gpu_types(include_pricing=False):
            display_name = gpu.get("displayName", "")
            gpu_id = gpu.get("id", "")
            if needle in display_name.lower() or needle in gpu_id.lower():
                return gpu_id
            if not needle_key:
                continue
            candidate_keys = (_gpu_match_key(display_name), _gpu_match_key(gpu_id))
            if needle_key in candidate_keys:
                return gpu_id
            if any(needle_key and needle_key in key for key in candidate_keys):
                normalized_matches.append(gpu_id)
        if normalized_matches:
            return normalized_matches[0]
        raise ValueError(f"No GPU type matching {name_substr!r}")

    def create_pod(self, config: PodConfig) -> Pod:
        env_list = [
            {"key": key, "value": str(value)}
            for key, value in sorted(config.env_vars.items())
            if value not in ("", None)
        ]
        resolved_gpu_type = resolve_gpu_type(config.gpu_type)
        gpu_type_id = self.find_gpu_type_id(resolved_gpu_type)
        query = """
        mutation($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                name
                desiredStatus
            }
        }"""
        input_payload: dict = {
            "name": config.name,
            "gpuTypeId": gpu_type_id,
            "gpuCount": config.gpu_count,
            "cloudType": config.cloud_type,
            "volumeInGb": config.volume_size,
            "containerDiskInGb": config.container_disk,
            "startSsh": True,
            "env": env_list,
        }
        if config.template_id:
            input_payload["templateId"] = config.template_id
        else:
            input_payload["imageName"] = config.image
            input_payload["dockerArgs"] = ""
            input_payload["ports"] = "22/tcp,8888/http"
        pod_data = self._graphql(query, {"input": input_payload}).get(
            "podFindAndDeployOnDemand", {}
        )
        return Pod(
            id=pod_data["id"],
            name=pod_data.get("name", config.name),
            status=pod_data.get("desiredStatus", "CREATED"),
            gpu_type=config.gpu_type,
        )

    def get_pod(self, pod_id: str) -> Pod:
        query = """query($podId: String!) {
            pod(input: { podId: $podId }) {
                id
                name
                desiredStatus
                runtime {
                    gpus { id }
                    ports { ip isIpPublic privatePort publicPort type }
                }
            }
        }"""
        try:
            pod_data = self._graphql(query, {"podId": pod_id}).get("pod", {}) or {}
        except (requests.HTTPError, RuntimeError):
            return Pod(id=pod_id, name="", status="TERMINATED")
        runtime = pod_data.get("runtime") or {}
        gpus = runtime.get("gpus") or []
        ports = runtime.get("ports") or []

        ssh_host = ""
        ssh_port = 0
        public_ip = ""
        for port in ports:
            if port.get("privatePort") == 22:
                ssh_host = port.get("ip", "")
                ssh_port = int(port.get("publicPort", 0) or 0)
                if port.get("isIpPublic"):
                    public_ip = port.get("ip", "")
                break
        gpu_type = ""
        if gpus:
            gpu_type = str(gpus[0].get("id", "") or "")

        return Pod(
            id=pod_data.get("id", pod_id),
            name=pod_data.get("name", ""),
            status=pod_data.get("desiredStatus", "UNKNOWN"),
            gpu_type=gpu_type,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            public_ip=public_ip,
        )

    def list_pods(self) -> list[Pod]:
        data = self._graphql("query { myself { pods { id name desiredStatus } } }")
        pods_data = data.get("myself", {}).get("pods", [])
        return [
            Pod(
                id=pod["id"],
                name=pod.get("name", ""),
                status=pod.get("desiredStatus", ""),
            )
            for pod in pods_data
        ]

    def find_pod_by_name(self, name: str) -> Optional[Pod]:
        for pod in self.list_pods():
            if pod.name == name:
                return pod
        return None

    def wait_for_pod(self, pod_id: str, timeout: int = 900, poll_interval: int = 10) -> Pod:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            pod = self.get_pod(pod_id)
            print(f"  Pod {pod.id}: status={pod.status}, ssh={pod.ssh_host}:{pod.ssh_port}")
            if pod.status == "RUNNING" and pod.ssh_host and pod.ssh_port:
                return pod
            time.sleep(poll_interval)
        elapsed = int(time.monotonic() - start)
        raise TimeoutError(
            f"Pod {pod_id!r} did not become ready after {elapsed}s (timeout={timeout}s)"
        )

    def start_pod(self, pod_id: str) -> None:
        self._rest_post(f"/pods/{pod_id}/start")
        print(f"  Pod {pod_id} start requested")

    def stop_pod(self, pod_id: str) -> None:
        self._rest_post(f"/pods/{pod_id}/stop")
        print(f"  Pod {pod_id} stopped")

    def terminate_pod(self, pod_id: str) -> None:
        query = """mutation($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }"""
        self._graphql(query, {"input": {"podId": pod_id}})
        print(f"  Pod {pod_id} terminated")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        def load_dotenv(*_args, **_kwargs) -> bool:
            return False

    load_dotenv()
    client = RunPodClient()
    print("Available GPUs:")
    for gpu in client.list_gpu_types():
        print(f"  {gpu['displayName']} ({gpu['memoryInGb']}GB) - {gpu['id']}")
    print("\nActive pods:")
    for pod in client.list_pods():
        print(f"  {pod.name} ({pod.id}): {pod.status}")
