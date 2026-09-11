from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.api.v1.chat import get_providers
from app.services.routing.circuit_breaker import get_circuit_breaker
from app.services.routing.store import get_metrics_store
from app.services.routing.manager import _STRATEGY_REGISTRY, _init_registry
from app.services.telemetry import get_telemetry

router = APIRouter(prefix="/v1", tags=["Observability"])


@router.get("/health/routing")
async def routing_health() -> JSONResponse:
    """
    §4.2: Routing health introspection endpoint.

    Returns per-provider circuit breaker state, in-flight request counts,
    EMA latency, and EMA error rate — giving operators a live view of the
    routing layer without having to grep logs.

    Example response:
    {
      "providers": {
        "openai":    { "circuit_state": "CLOSED",    "in_flight": 3, "ema_latency_ms": 412.5, "ema_error_rate": 0.02 },
        "anthropic": { "circuit_state": "HALF_OPEN",  "in_flight": 0, "ema_latency_ms": 650.1, "ema_error_rate": 0.18 }
      },
      "fallback_events": [ ... ]
    }
    """
    try:
        providers = get_providers()
    except ValueError:
        providers = []

    circuit_breaker = get_circuit_breaker()
    store = get_metrics_store()
    telemetry = get_telemetry()

    provider_health = {}
    for provider in providers:
        name = provider.get_provider_name()
        latency = store.get_latency(name)
        provider_health[name] = {
            "circuit_state": circuit_breaker.get_state(name).value,
            "in_flight": store.get_in_flight(name),
            "ema_latency_ms": round(latency * 1000, 2) if latency is not None else None,
            "ema_error_rate": round(store.get_error_rate(name), 4),
        }

    return JSONResponse(content={
        "providers": provider_health,
        "fallback_events": telemetry.get_fallback_events(),
    })

