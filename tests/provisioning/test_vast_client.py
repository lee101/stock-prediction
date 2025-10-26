import json
from unittest.mock import Mock

import pytest

from marketsimulator.provisioning.config import VastSettings
from marketsimulator.provisioning.vast import OfferFilters, VastClient


def _response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_search_offers_builds_expected_payload():
    session = Mock()
    session.post.return_value = _response({"offers": [{"id": 1}]})
    client = VastClient(VastSettings(api_key="key", base_url="https://api"), session=session)

    filters = OfferFilters(
        gpu_name="RTX_4090",
        min_reliability=0.99,
        min_duration_hours=4,
        limit=5,
        max_price_per_hour=1.23,
        countries=["US", "CA"],
    )
    offers = client.search_offers(filters)

    assert offers == [{"id": 1}]
    session.post.assert_called_once()
    _, kwargs = session.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer key"
    payload = json.loads(kwargs["data"])
    assert payload["gpu_name"] == {"in": ["RTX_4090"]}
    assert payload["dph_total"] == {"lte": 1.23}
    assert payload["geolocation"] == {"in": ["US", "CA"]}
    assert payload["reliability"] == {"gte": 0.99}
    assert payload["duration"] == {"gte": 4 * 3600}


def test_create_instance_merges_environment_and_returns_id():
    session = Mock()
    session.put.return_value = _response({"new_contract": 4242})
    client = VastClient(VastSettings(api_key="x", base_url="https://api"), session=session)

    instance_id = client.create_instance(
        101,
        image="repo/image:tag",
        disk_gb=30,
        volume_gb=50,
        label="msim",
        bid_price=0.42,
        portal_internal_port=9000,
        portal_external_port=32000,
        env={"EXTRA": "1"},
        onstart="echo hello",
    )

    assert instance_id == 4242
    session.put.assert_called_once()
    _, kwargs = session.put.call_args
    payload = json.loads(kwargs["data"])
    assert payload["image"] == "repo/image:tag"
    assert payload["price"] == pytest.approx(0.42)
    assert payload["volume_info"]["size"] == 50
    # Env should include portal configuration and extra key.
    assert payload["env"]["PORT"] == "9000"
    assert payload["env"]["OPEN_BUTTON_PORT"] == "32000"
    assert payload["env"]["EXTRA"] == "1"
    assert payload["onstart"] == "echo hello"
