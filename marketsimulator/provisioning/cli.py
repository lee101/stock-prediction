"""
Command line interface for provisioning GPU resources on Vast.ai and RunPod.

Usage (after exporting the required API keys):

    python -m marketsimulator.provisioning vast search
    python -m marketsimulator.provisioning runpod pod-create --name mypod --gpu-types "NVIDIA GeForce RTX 3090"
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
from typing import Optional

import typer

from .config import RunPodSettings, VastSettings
from .runpod import PodRequest, RunPodClient
from .vast import OfferFilters, VastClient


app = typer.Typer(help="Provision Vast.ai and RunPod capacity for the market simulator.")

vast_app = typer.Typer(help="Vast.ai automation commands.")
runpod_app = typer.Typer(help="RunPod automation commands.")


def _json_dump(data: object) -> None:
    typer.echo(json.dumps(data, indent=2, sort_keys=True, default=str))


def _coerce_list(argument: Optional[str]) -> list[str]:
    if not argument:
        return []
    return [item.strip() for item in argument.split(",") if item.strip()]


def _default_image() -> str:
    image = os.getenv("DOCKER_IMAGE")
    if not image:
        raise typer.BadParameter("Either supply --image or export DOCKER_IMAGE.")
    return image


# --------------------------------------------------------------------------- #
# Vast.ai commands
# --------------------------------------------------------------------------- #
@vast_app.command("search")
def vast_search(
    gpu: str = typer.Option("RTX_3090", help="Filter offers to the given GPU type."),
    min_reliability: float = typer.Option(0.98, help="Minimum host reliability score."),
    min_hours: float = typer.Option(6.0, help="Minimum remaining hours on the offer."),
    limit: int = typer.Option(20, help="Number of offers to return."),
    max_price_per_hour: Optional[float] = typer.Option(
        None, help="Maximum total price per hour (interruptible and on-demand)."
    ),
    countries: Optional[str] = typer.Option(
        None, help="Comma-separated list of country codes to include."
    ),
) -> None:
    """Search the Vast.ai marketplace for rentable offers."""
    client = VastClient(VastSettings.from_env())
    filters = OfferFilters(
        gpu_name=gpu,
        min_reliability=min_reliability,
        min_duration_hours=min_hours,
        limit=limit,
        max_price_per_hour=max_price_per_hour,
        countries=_coerce_list(countries) or None,
    )
    offers = client.search_offers(filters)
    if not offers:
        typer.echo("No offers found.")
        raise typer.Exit(code=1)
    _json_dump(offers)


@vast_app.command("rent")
def vast_rent(
    offer_id: int = typer.Argument(..., help="Offer identifier from the Vast marketplace search."),
    image: Optional[str] = typer.Option(None, help="Docker image to launch."),
    disk_gb: int = typer.Option(40, help="Container disk size (GB)."),
    volume_gb: Optional[int] = typer.Option(None, help="Optional persistent volume size (GB)."),
    label: str = typer.Option("marketsimulator", help="Label assigned to the instance."),
    bid: Optional[float] = typer.Option(None, help="Interruptible bid price (dollars per hour)."),
    wait: bool = typer.Option(True, help="Wait for the instance to reach RUNNING."),
    portal_external_port: int = typer.Option(8000, help="External port exposed via the portal."),
    portal_internal_port: int = typer.Option(80, help="Internal container port to expose."),
) -> None:
    """Accept a Vast.ai offer and boot the container image."""
    client = VastClient(VastSettings.from_env())
    instance_id = client.create_instance(
        offer_id,
        image=image or _default_image(),
        disk_gb=disk_gb,
        volume_gb=volume_gb,
        label=label,
        bid_price=bid,
        portal_external_port=portal_external_port,
        portal_internal_port=portal_internal_port,
    )
    typer.echo(f"Created instance {instance_id}.")
    if wait:
        info = client.wait_for_status(instance_id)
        typer.echo("Instance details:")
        _json_dump(info)


@vast_app.command("show")
def vast_show(instance_id: int = typer.Argument(...)) -> None:
    """Show current Vast.ai instance metadata."""
    client = VastClient(VastSettings.from_env())
    info = client.get_instance(instance_id)
    if not info:
        typer.echo("Instance not found.")
        raise typer.Exit(code=1)
    _json_dump(info)


@vast_app.command("attach-key")
def vast_attach_key(instance_id: int, public_key_path: pathlib.Path) -> None:
    """Attach an SSH public key to the instance."""
    public_key = public_key_path.read_text()
    client = VastClient(VastSettings.from_env())
    result = client.attach_ssh_key(instance_id, public_key)
    _json_dump(result)


@vast_app.command("change-bid")
def vast_change_bid(instance_id: int, price: float) -> None:
    """Adjust the bid price for an interruptible instance."""
    client = VastClient(VastSettings.from_env())
    result = client.change_bid(instance_id, price)
    _json_dump(result)


def _update_state(instance_id: int, *, state: str) -> None:
    client = VastClient(VastSettings.from_env())
    result = client.update_state(instance_id, state)
    _json_dump(result)


@vast_app.command("stop")
def vast_stop(instance_id: int) -> None:
    """Stop a Vast.ai instance."""
    _update_state(instance_id, state="stopped")


@vast_app.command("start")
def vast_start(instance_id: int) -> None:
    """Start a Vast.ai instance."""
    _update_state(instance_id, state="running")


@vast_app.command("destroy")
def vast_destroy(instance_id: int) -> None:
    """Terminate a Vast.ai instance."""
    client = VastClient(VastSettings.from_env())
    result = client.destroy_instance(instance_id)
    _json_dump(result)


@vast_app.command("ssh")
def vast_ssh(instance_id: int, key_path: Optional[pathlib.Path] = typer.Option(None, help="Optional SSH key path.")) -> None:
    """Print a ready-to-run SSH command for the instance."""
    client = VastClient(VastSettings.from_env())
    info = client.get_instance(instance_id)
    host = info.get("ssh_host") or info.get("public_ipaddr")
    port = info.get("ssh_port", 22)
    if not host:
        typer.echo("Instance is not exposing an SSH host yet.")
        raise typer.Exit(code=1)
    key_flag = f"-i {shlex.quote(str(key_path))} " if key_path else ""
    cmd = f"ssh {key_flag}-o StrictHostKeyChecking=no -p {port} root@{host}"
    typer.echo(cmd)


# --------------------------------------------------------------------------- #
# RunPod commands
# --------------------------------------------------------------------------- #
@runpod_app.command("gpu-types")
def runpod_gpu_types() -> None:
    """List available RunPod GPU type identifiers."""
    client = RunPodClient(RunPodSettings.from_env())
    data = client.list_gpu_types()
    _json_dump(data)


@runpod_app.command("pod-create")
def runpod_pod_create(
    name: str = typer.Option("marketsim-pod"),
    gpu_types: str = typer.Option(..., help="Comma-separated GPU type identifiers."),
    image: Optional[str] = typer.Option(None),
    interruptible: bool = typer.Option(False, help="Request an interruptible pod."),
    volume_gb: int = typer.Option(20),
    container_disk_gb: int = typer.Option(50),
    ports: Optional[str] = typer.Option(None, help="Comma-separated list of port mappings, e.g. '22/tcp,80/http'."),
    env: Optional[str] = typer.Option(None, help="Comma-separated KEY=VALUE overrides."),
) -> None:
    """Create a RunPod VM-like pod."""
    client = RunPodClient(RunPodSettings.from_env())
    parsed_env = None
    if env:
        parsed_env = {}
        for item in _coerce_list(env):
            if "=" not in item:
                raise typer.BadParameter(f"Invalid env entry {item!r}; expected KEY=VALUE.")
            key, value = item.split("=", 1)
            parsed_env[key] = value
    request = PodRequest(
        name=name,
        gpu_type_ids=_coerce_list(gpu_types),
        image=image or _default_image(),
        interruptible=interruptible,
        volume_gb=volume_gb,
        container_disk_gb=container_disk_gb,
        ports=_coerce_list(ports) or None,
        env=parsed_env,
    )
    result = client.create_pod(request)
    _json_dump(result)


@runpod_app.command("pod-get")
def runpod_pod_get(pod_id: str) -> None:
    """Fetch details for a RunPod pod."""
    client = RunPodClient(RunPodSettings.from_env())
    data = client.get_pod(pod_id)
    _json_dump(data)


@runpod_app.command("pod-stop")
def runpod_pod_stop(pod_id: str) -> None:
    client = RunPodClient(RunPodSettings.from_env())
    data = client.stop_pod(pod_id)
    _json_dump(data)


@runpod_app.command("pod-delete")
def runpod_pod_delete(pod_id: str) -> None:
    client = RunPodClient(RunPodSettings.from_env())
    data = client.delete_pod(pod_id)
    _json_dump(data)


@runpod_app.command("pod-ssh")
def runpod_pod_ssh(pod_id: str, key_path: Optional[pathlib.Path] = typer.Option(None)) -> None:
    """Print a ready-to-run SSH command for the pod."""
    client = RunPodClient(RunPodSettings.from_env())
    data = client.get_pod(pod_id)
    ip = data.get("publicIp")
    port_mappings = data.get("portMappings", {}) or {}
    ssh_port = port_mappings.get("22")
    if not (ip and ssh_port):
        typer.echo("Pod does not yet expose a public SSH endpoint.")
        raise typer.Exit(code=1)
    key_flag = f"-i {shlex.quote(str(key_path))} " if key_path else ""
    cmd = f"ssh {key_flag}-o StrictHostKeyChecking=no -p {ssh_port} root@{ip}"
    typer.echo(cmd)


@runpod_app.command("template-create")
def runpod_template_create(
    name: str = typer.Option("marketsim-template"),
    image: Optional[str] = typer.Option(None),
    ports: Optional[str] = typer.Option(None),
    env: Optional[str] = typer.Option(None),
) -> None:
    """Create a RunPod serverless template."""
    client = RunPodClient(RunPodSettings.from_env())
    parsed_env = None
    if env:
        parsed_env = {}
        for item in _coerce_list(env):
            if "=" not in item:
                raise typer.BadParameter(f"Invalid env entry {item!r}; expected KEY=VALUE.")
            key, value = item.split("=", 1)
            parsed_env[key] = value
    template_id = client.create_template(
        name=name,
        image=image or _default_image(),
        ports=_coerce_list(ports) or None,
        env=parsed_env,
    )
    typer.echo(template_id)


@runpod_app.command("endpoint-create")
def runpod_endpoint_create(
    template_id: str,
    name: str = typer.Option("marketsim-endpoint"),
    gpu_types: str = typer.Option(..., help="Comma-separated GPU type identifiers."),
    workers_min: int = typer.Option(0),
    workers_max: int = typer.Option(2),
    idle_timeout: int = typer.Option(5),
) -> None:
    """Create a RunPod serverless endpoint from a template."""
    client = RunPodClient(RunPodSettings.from_env())
    endpoint_id = client.create_endpoint(
        template_id=template_id,
        name=name,
        gpu_type_ids=_coerce_list(gpu_types),
        workers_min=workers_min,
        workers_max=workers_max,
        idle_timeout=idle_timeout,
    )
    typer.echo(endpoint_id)


@runpod_app.command("runsync")
def runpod_runsync(endpoint_id: str, symbol: str = "SPY", window: int = 256) -> None:
    """Invoke a RunPod serverless endpoint synchronously."""
    client = RunPodClient(RunPodSettings.from_env())
    payload = {"symbol": symbol, "window": window}
    result = client.runsync(endpoint_id, payload)
    _json_dump(result)


app.add_typer(vast_app, name="vast")
app.add_typer(runpod_app, name="runpod")


def main() -> None:  # pragma: no cover - thin wrapper for console entry.
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
