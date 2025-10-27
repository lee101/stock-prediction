"""
RunPod automation helpers used by the provisioning CLI.

The client intentionally keeps things explicit—callers provide GPU types and
environment overrides to avoid surprising implicit defaults that might not map
to production needs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, Optional
import time

import requests

from .config import RunPodSettings


JsonDict = Dict[str, Any]


@dataclass(slots=True)
class PodRequest:
    """Parameters required to create a RunPod Pod (VM-like instance)."""

    name: str
    gpu_type_ids: list[str]
    image: str
    interruptible: bool = False
    volume_gb: int = 20
    container_disk_gb: int = 50
    ports: list[str] = None  # type: ignore[assignment]
    env: JsonDict | None = None
    support_public_ip: bool = True

    def __post_init__(self) -> None:
        if self.ports is None:
            self.ports = ["22/tcp", "80/http"]


class RunPodClient:
    """Small wrapper around RunPod's REST and queue APIs."""

    def __init__(self, settings: RunPodSettings, session: Optional[requests.Session] = None):
        self._settings = settings
        self._session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _rest_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }

    def _queue_headers(self) -> Dict[str, str]:
        return {"authorization": self._settings.api_key, "content-type": "application/json"}

    def _rest_url(self, path: str) -> str:
        return f"{self._settings.rest_base_url}{path}"

    def _queue_url(self, path: str) -> str:
        return f"{self._settings.queue_base_url}/{path.lstrip('/')}"

    def _post_rest(self, path: str, payload: JsonDict) -> JsonDict:
        response = self._session.post(self._rest_url(path), headers=self._rest_headers(), data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

    def _post_queue(self, path: str, payload: JsonDict) -> JsonDict:
        response = self._session.post(self._queue_url(path), headers=self._queue_headers(), data=json.dumps(payload))
        response.raise_for_status()
        if response.content:
            return response.json()
        return {}

    def _get_rest(self, path: str) -> JsonDict:
        response = self._session.get(self._rest_url(path), headers=self._rest_headers())
        response.raise_for_status()
        return response.json()

    def _get_queue(self, path: str) -> JsonDict:
        response = self._session.get(self._queue_url(path), headers=self._queue_headers())
        response.raise_for_status()
        return response.json()

    def _delete_rest(self, path: str) -> JsonDict:
        response = self._session.delete(self._rest_url(path), headers=self._rest_headers())
        response.raise_for_status()
        if response.content:
            return response.json()
        return {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_gpu_types(self) -> JsonDict:
        payload = {"query": "query GPU { gpuTypes { id displayName memoryInGb } }"}
        response = self._session.post(
            self._settings.graphql_url,
            params={"api_key": self._settings.api_key},
            headers={"content-type": "application/json"},
            data=json.dumps(payload),
        )
        response.raise_for_status()
        return response.json()

    def create_pod(self, request: PodRequest) -> JsonDict:
        body: JsonDict = {
            "name": request.name,
            "computeType": "GPU",
            "gpuTypeIds": request.gpu_type_ids,
            "gpuTypePriority": "availability",
            "gpuCount": 1,
            "imageName": request.image,
            "interruptible": request.interruptible,
            "containerDiskInGb": request.container_disk_gb,
            "volumeInGb": request.volume_gb,
            "volumeMountPath": "/workspace",
            "supportPublicIp": request.support_public_ip,
            "ports": request.ports,
        }
        if request.env:
            body["env"] = request.env
        return self._post_rest("/pods", body)

    def get_pod(self, pod_id: str) -> JsonDict:
        return self._get_rest(f"/pods/{pod_id}")

    def stop_pod(self, pod_id: str) -> JsonDict:
        return self._post_rest(f"/pods/{pod_id}/stop", {})

    def delete_pod(self, pod_id: str) -> JsonDict:
        return self._delete_rest(f"/pods/{pod_id}")

    def create_template(
        self,
        *,
        name: str,
        image: str,
        ports: Optional[Iterable[str]] = None,
        env: Optional[JsonDict] = None,
        is_serverless: bool = True,
    ) -> str:
        body: JsonDict = {
            "name": name,
            "imageName": image,
            "isServerless": is_serverless,
            "ports": list(ports) if ports is not None else ["80/http"],
        }
        if env:
            body["env"] = env
        result = self._post_rest("/templates", body)
        template_id = result.get("id")
        if not template_id:
            raise RuntimeError(f"RunPod template creation did not return an id: {result}")
        return template_id

    def create_endpoint(
        self,
        *,
        template_id: str,
        name: str,
        gpu_type_ids: Iterable[str],
        workers_min: int = 0,
        workers_max: int = 2,
        idle_timeout: int = 5,
    ) -> str:
        body = {
            "templateId": template_id,
            "name": name,
            "computeType": "GPU",
            "gpuTypeIds": list(gpu_type_ids),
            "gpuCount": 1,
            "workersMin": workers_min,
            "workersMax": workers_max,
            "idleTimeout": idle_timeout,
        }
        result = self._post_rest("/endpoints", body)
        endpoint_id = result.get("id")
        if not endpoint_id:
            raise RuntimeError(f"RunPod endpoint creation did not return an id: {result}")
        return endpoint_id

    def runsync(self, endpoint_id: str, payload: JsonDict) -> JsonDict:
        data = {"input": payload}
        return self._post_queue(f"{endpoint_id}/runsync", data)

    def run(self, endpoint_id: str, payload: JsonDict) -> str:
        data = {"input": payload}
        result = self._post_queue(f"{endpoint_id}/run", data)
        job_id = result.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod /run did not return a job id: {result}")
        return job_id

    def job_status(self, endpoint_id: str, job_id: str) -> JsonDict:
        return self._get_queue(f"{endpoint_id}/status/{job_id}")

    def wait_for_job(
        self,
        endpoint_id: str,
        job_id: str,
        *,
        timeout: float = 600.0,
        poll_interval: float = 5.0,
    ) -> JsonDict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.job_status(endpoint_id, job_id)
            if status.get("status") in {"COMPLETED", "FAILED"}:
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"RunPod job {job_id} did not complete within {timeout} seconds.")
