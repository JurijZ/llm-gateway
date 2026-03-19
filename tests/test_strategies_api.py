from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_strategies():
    response = client.get("/v1/strategies")
    assert response.status_code == 200
    data = response.json()
    
    assert "strategies" in data
    strategies = data["strategies"]
    assert len(strategies) >= 4
    
    strategy_ids = [s["id"] for s in strategies]
    assert "hardcoded" in strategy_ids
    assert "load_balance" in strategy_ids
    assert "latency" in strategy_ids
    assert "cost_latency" in strategy_ids
    
    for strategy in strategies:
        assert "id" in strategy
        assert "name" in strategy
        assert "description" in strategy
