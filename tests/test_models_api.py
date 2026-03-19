from fastapi.testclient import TestClient
from app.main import app
from app.core.models import MODEL_MAPPING, PROVIDER_DEFAULTS

client = TestClient(app)

def test_get_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    
    assert "models" in data
    assert "provider_defaults" in data
    
    # Check that it matches the core models
    # Note: Tuple becomes List in JSON
    for model, info in MODEL_MAPPING.items():
        assert data["models"][model] == list(info)
        
    for provider, default in PROVIDER_DEFAULTS.items():
        assert data["provider_defaults"][provider] == default
