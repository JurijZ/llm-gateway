from fastapi import APIRouter

router = APIRouter(prefix="/v1")

@router.get("/strategies")
async def get_strategies():
    """
    Returns all available routing strategies.
    """
    return {
        "strategies": [
            {
                "id": "hardcoded",
                "name": "Hardcoded Strategy",
                "description": "Returns the first provider in the list, or one matching the preference."
            },
            {
                "id": "load_balance",
                "name": "Least In-Flight Strategy",
                "description": "Returns the provider with the least in-flight requests."
            },
            {
                "id": "latency",
                "name": "Latency Based Strategy",
                "description": "Selects the provider with the lowest rolling average latency."
            },
            {
                "id": "cost_latency",
                "name": "Cost + Latency Tradeoff Strategy",
                "description": "Routes based on a composite score of latency, cost, and error rate."
            }
        ]
    }
