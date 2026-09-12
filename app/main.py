from fastapi import FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import CorrelationIdMiddleware
from app.api.v1.chat import router as chat_router, get_providers
from app.api.v1.models_api import router as models_router
from app.api.v1.routing import router as routing_router
from app.api.v1.telemetry import router as telemetry_router
from app.api.v1.health import router as health_router

# Initialize structured / correlation-aware logging
setup_logging(json_logs=settings.JSON_LOGS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Graceful teardown: close provider HTTP connection pools
    providers_getter = app.dependency_overrides.get(get_providers, get_providers)
    for provider in providers_getter():
        try:
            await provider.close()
        except Exception:
            pass

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, lifespan=lifespan)

# §2.5: Convert ValueError (e.g. "no provider API keys configured") raised during
# FastAPI dependency injection into HTTP 503 Service Unavailable.
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": f"Service Unavailable: {exc}"})

# Observability middleware
app.add_middleware(CorrelationIdMiddleware)

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(routing_router)
app.include_router(telemetry_router)
app.include_router(health_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

