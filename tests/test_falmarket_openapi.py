import pytest

from falmarket.app import (
    MarketSimulatorApp,
    SimulationRequest,
    SimulationResponse,
)


@pytest.mark.integration
def test_simulation_endpoint_annotations_resolve() -> None:
    app = MarketSimulatorApp(_allow_init=True)
    schema = app.openapi()

    route_map = app.collect_routes()
    endpoint = next(
        handler for signature, handler in route_map.items() if signature.path == "/api/simulate"
    )

    assert endpoint.__annotations__["request"] is SimulationRequest
    assert endpoint.__annotations__["return"] is SimulationResponse
    body_schema = (
        schema["paths"]["/api/simulate"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    )
    assert body_schema.endswith("/SimulationRequest")
