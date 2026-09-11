from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services.telemetry import get_telemetry

router = APIRouter()

@router.get("/v1/telemetry", tags=["Observability"])
@router.get("/metrics", tags=["Observability"])
async def get_metrics():
    """
    Returns telemetry metrics including request counts per provider/model,
    Time-To-First-Chunk (TTFC) percentiles (p50, p95, p99), and fallback frequency.
    """
    telemetry = get_telemetry()
    return JSONResponse(content=telemetry.get_metrics())

