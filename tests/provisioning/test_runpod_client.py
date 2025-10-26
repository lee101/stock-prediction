import json
from unittest.mock import Mock

from marketsimulator.provisioning.config import RunPodSettings
from marketsimulator.provisioning.runpod import PodRequest, RunPodClient


def _response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_create_pod_posts_expected_payload():
    session = Mock()
    session.post.return_value = _response({"id": "pod-1"})
    client = RunPodClient(RunPodSettings(api_key="runpod", rest_base_url="https://rest", queue_base_url="https://queue"), session=session)

    request = PodRequest(
        name="marketsim",
        gpu_type_ids=["NVIDIA GeForce RTX 3090"],
        image="repo/image:tag",
        interruptible=True,
        volume_gb=100,
        container_disk_gb=60,
        ports=["22/tcp", "80/http"],
        env={"PORT": "80"},
    )
    client.create_pod(request)

    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args[0] == "https://rest/pods"
    payload = json.loads(kwargs["data"])
    assert payload["interruptible"] is True
    assert payload["volumeInGb"] == 100
    assert payload["env"] == {"PORT": "80"}


def test_create_template_validates_response():
    session = Mock()
    session.post.return_value = _response({"id": "template-1"})
    client = RunPodClient(
        RunPodSettings(
            api_key="token",
            rest_base_url="https://rest",
            queue_base_url="https://queue",
        ),
        session=session,
    )

    template_id = client.create_template(name="tpl", image="repo/image:tag", ports=["80/http"], env={"PORT": "80"})
    assert template_id == "template-1"

    args, kwargs = session.post.call_args
    assert args[0] == "https://rest/templates"
    payload = json.loads(kwargs["data"])
    assert payload["isServerless"] is True
    assert payload["env"] == {"PORT": "80"}
