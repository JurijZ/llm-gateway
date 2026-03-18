# Implementation Plan - LLM Gateway

## Objective
Build a robust, FastAPI-based LLM Gateway that unifies access to OpenAI and Anthropic models. The service will feature streaming responses, dynamic routing strategies (load balancing, latency-based), and reliability mechanisms (retries, fallbacks).

## Tech Stack
- **Language:** Python 3.12+
- **Framework:** FastAPI
- **Package Manager:** uv
- **Async Runtime:** asyncio
- **Validation:** Pydantic
- **LLM SDKs:** `openai`, `anthropic`

## Proposed Architecture
The application will follow a modular service-based architecture to separate concerns between API handling, routing logic, and provider integrations.

```text
app/
├── api/
│   └── v1/
│       └── chat.py          # Endpoint definition
├── core/
│   ├── config.py            # Environment & Configuration (Pydantic Settings)
│   └── exceptions.py        # Custom exception handlers
├── models/
│   └── schemas.py           # Pydantic models for Request/Response
├── services/
│   ├── llm/
│   │   ├── base.py          # Abstract Base Class for Providers
│   │   ├── openai.py        # OpenAI Implementation
│   │   └── anthropic.py     # Anthropic Implementation
│   └── routing/
│       ├── manager.py       # Routing Manager Service
│       └── strategies.py    # Strategy Pattern implementations (Random, Latency, etc.)
└── main.py                  # App entrypoint
```

## Implementation Steps

### Phase 1: Project Setup & Core Configuration
1.  **Initialize Project:** Configure `pyproject.toml` with dependencies (`fastapi`, `uvicorn`, `pydantic-settings`, `openai`, `anthropic`, `python-dotenv`).
2.  **Configuration:** Implement `app/core/config.py` to handle API keys and default settings using `pydantic-settings`.
3.  **Skeleton:** Create the directory structure and a basic health check endpoint to verify setup.

### Phase 2: LLM Provider Abstraction
1.  **Interface:** Define an abstract base class `LLMProvider` in `app/services/llm/base.py` enforcing a `stream_chat` method.
2.  **OpenAI Integration:** Implement `OpenAIProvider` using the official SDK.
3.  **Anthropic Integration:** Implement `AnthropicProvider` using the official SDK.
4.  **Normalization:** Ensure both providers return a unified stream format.

### Phase 3: Routing Engine
1.  **Strategy Interface:** Define a `RoutingStrategy` interface.
2.  **Strategies Implementation:**
    -   **Hardcoded:** Returns a specific configured model.
    -   **Least In-Flight:** specific strategy to balance load (requires simple in-memory state tracking).
    -   **Latency-Based:** Strategy that selects provider based on historical latency (requires simple rolling average tracking).
3.  **Router Service:** Create `RouterManager` to select the appropriate provider based on the active strategy.

### Phase 4: Reliability Layer
1.  **Timeouts:** Implement `asyncio.wait_for` wrappers around provider calls to enforce strict timeouts.
2.  **Fallbacks:** Update `RouterManager` to handle exceptions. If the primary provider fails, automatically try the next best candidate.
3.  **Retries:** Implement a retry decorator or logic block for transient network errors.

### Phase 5: API Endpoint & Streaming
1.  **Endpoint:** Create `POST /chat` in `app/api/v1/chat.py`.
2.  **Request Model:** Define `ChatRequest` (message, optional model preference).
3.  **Streaming Response:** Use `StreamingResponse` to stream the generator yielded by the selected provider.

### Phase 6: Testing & Validation
1.  **Unit Tests:** Test routing logic and strategy selection.
2.  **Integration Tests:** Mock external API calls to verify the full flow from endpoint to provider.
3.  **Manual Verification:** Verify streaming with actual API keys.

## Verification Plan
- [ ] Server starts successfully with `uv run fastapi dev`.
- [ ] Health check returns 200 OK.
- [ ] `POST /chat` streams text from OpenAI.
- [ ] `POST /chat` streams text from Anthropic.
- [ ] Fallback logic works (simulate failure on primary).
- [ ] Latency routing updates based on mock performance data.
