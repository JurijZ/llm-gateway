from fastapi import APIRouter
from app.core.models import MODEL_MAPPING, PROVIDER_DEFAULTS

router = APIRouter(prefix="/v1")

@router.get("/models")
async def get_models():
    """
    Returns all available models and provider defaults.
    """
    return {
        "models": MODEL_MAPPING,
        "provider_defaults": PROVIDER_DEFAULTS
    }
