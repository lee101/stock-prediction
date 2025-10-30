"""
Typed helper functions for interacting with the Vast.ai REST API.

The focus is on covering the operations required for provisioning GPU capacity
for the market simulator: discovering rentable instances, creating contracts,
polling their status, and performing light-weight lifecycle management.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, Optional
import time

import requests

from .config import VastSettings


JsonDict = Dict[str, Any]


@dataclass(slots=True)
class OfferFilters:
    """Light-weight container for search filters."""

    gpu_name: Optional[str] = "RTX_3090"
    min_reliability: float = 0.98
    min_duration_hours: float = 6.0
    limit: int = 20
    max_price_per_hour: Optional[float] = None
    countries: Optional[Iterable[str]] = None


class VastClient:
    """Small wrapper over the Vast.ai REST API."""

    def __init__(self, settings: VastSettings, session: Optional[requests.Session] = None):
        self._settings = settings
        self._session = session or requests.Session()

    # --------------------------------------------------------------------- #
    # Low-level helpers
    # --------------------------------------------------------------------- #
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self._settings.base_url}{path}"

    def _post(self, path: str, payload: JsonDict) -> JsonDict:
        response = self._session.post(self._url(path), headers=self._headers(), data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

    def _put(self, path: str, payload: JsonDict) -> JsonDict:
        response = self._session.put(self._url(path), headers=self._headers(), data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

    def _get(self, path: str) -> JsonDict:
        response = self._session.get(self._url(path), headers=self._headers())
        response.raise_for_status()
        return response.json()

    def _delete(self, path: str) -> JsonDict:
        response = self._session.delete(self._url(path), headers=self._headers())
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search_offers(self, filters: OfferFilters) -> list[JsonDict]:
        """Return rentable offers that satisfy ``filters``."""
        payload: JsonDict = {
            "type": "on-demand",
            "limit": filters.limit,
            "order": [["dlperf_per_dphtotal", "desc"]],
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "reliability": {"gte": filters.min_reliability},
            "duration": {"gte": int(filters.min_duration_hours * 3600)},
        }
        if filters.gpu_name:
            payload["gpu_name"] = {"in": [filters.gpu_name]}
        if filters.max_price_per_hour is not None:
            payload["dph_total"] = {"lte": float(filters.max_price_per_hour)}
        if filters.countries:
            payload["geolocation"] = {"in": list(filters.countries)}
        data = self._post("/bundles/", payload)
        return data.get("offers", [])

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        disk_gb: int = 40,
        volume_gb: Optional[int] = None,
        label: str = "marketsimulator",
        bid_price: Optional[float] = None,
        portal_internal_port: int = 80,
        portal_external_port: int = 8000,
        runtype: str = "args",
        env: Optional[JsonDict] = None,
        onstart: Optional[str] = None,
    ) -> int:
        """
        Accept an offer and request a running instance.

        Returns the new contract identifier.
        """
        body: JsonDict = {
            "image": image,
            "disk": int(disk_gb),
            "env": self._build_env(env, portal_internal_port, portal_external_port),
            "runtype": runtype,
            "label": label,
            "target_state": "running",
            "cancel_unavail": True,
        }
        if onstart:
            body["onstart"] = onstart
        if volume_gb and volume_gb > 0:
            body["volume_info"] = {
                "create_new": True,
                "size": int(volume_gb),
                "mount_path": "/workspace/data",
                "name": f"{label}-vol",
            }
        if bid_price is not None:
            body["price"] = float(bid_price)
        result = self._put(f"/asks/{offer_id}/", body)
        new_id = result.get("new_contract")
        if not new_id:
            raise RuntimeError(f"Vast.ai did not return a contract id: {result}")
        return int(new_id)

    def get_instance(self, instance_id: int) -> JsonDict:
        return self._get(f"/instances/{instance_id}/").get("instances", {})

    def wait_for_status(
        self,
        instance_id: int,
        *,
        desired: str = "running",
        timeout: float = 900.0,
        poll_interval: float = 5.0,
    ) -> JsonDict:
        """Poll the instance until ``desired`` status is reached or timeout expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.get_instance(instance_id)
            if info.get("actual_status") == desired:
                return info
            time.sleep(poll_interval)
        raise TimeoutError(f"Instance {instance_id} failed to reach status {desired!r} within {timeout} seconds.")

    def attach_ssh_key(self, instance_id: int, public_key: str) -> JsonDict:
        payload = {"ssh_key": public_key.strip()}
        return self._post(f"/instances/{instance_id}/ssh/", payload)

    def change_bid(self, instance_id: int, price: float) -> JsonDict:
        payload = {"client_id": "me", "price": float(price)}
        return self._put(f"/instances/bid_price/{instance_id}/", payload)

    def update_state(self, instance_id: int, state: str) -> JsonDict:
        return self._put(f"/instances/{instance_id}/", {"state": state})

    def destroy_instance(self, instance_id: int) -> JsonDict:
        return self._delete(f"/instances/{instance_id}/")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_env(
        extra_env: Optional[JsonDict],
        portal_internal_port: int,
        portal_external_port: int,
    ) -> JsonDict:
        """
        Compose the environment required for portal access while honouring
        caller-supplied overrides.
        """
        env = {
            "PORTAL_CONFIG": f"localhost:{portal_external_port}:{portal_internal_port}:/:MarketSim",
            "OPEN_BUTTON_PORT": str(portal_external_port),
            "PORT": str(portal_internal_port),
        }
        if extra_env:
            env.update(extra_env)
        return env
