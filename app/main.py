from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.chat import router as chat_router
from app.api.v1.models_api import router as models_router
from app.api.v1.routing import router as routing_router

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(routing_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "app": settings.APP_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
