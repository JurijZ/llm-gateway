# Gemini Context: LLM Gateway

This project is a FastAPI-based service designed to act as a unified gateway for multiple Large Language Model (LLM) providers (OpenAI, Anthropic). It focuses on streaming responses, intelligent routing strategies, and robust error handling.

## Project Overview

- **Purpose:** Provide a unified gateway for multiple LLMs via a set of versioned API endpoints.
- **API Endpoints:**
  - `POST /v1/chat`: Stream chat completions with dynamic provider/model selection and fallback. Supports `model_preference` and `routing_strategy`.
  - `GET /v1/models`: Returns a mapping of supported models to providers and their default models.
  - `GET /v1/strategies`: Lists available routing strategies (e.g., hardcoded, latency, cost_latency).
- **Core Technologies:** 
  - Python 3.13+
  - FastAPI
  - Pydantic
  - `uv` for package management
  - LLM SDKs (OpenAI, Anthropic)
  - Always use OpenAI Responses API. Never use OpenAI Completions API.
- **Key Features:**
  - Streaming chunk-by-chunk responses.
  - Intelligent Routing: Hardcoded, Least In-Flight, Latency-based, and Cost+Latency Tradeoff strategies.
  - Reliability: Timeouts, retries, and automatic provider fallback.
  - Observability: Request tracing (Planned).

## Architecture

The project aims for a clean FastAPI structure following best practices. It will likely involve:
- A routing layer to handle strategy selection.
- Provider abstractions for OpenAI and Anthropic.
- Middlewares for tracing and error handling.

## Building and Running

The project uses `uv` for dependency and environment management.

### Setup
```bash
# Install dependencies (once added to pyproject.toml)
uv sync
```

### Running the Server
```bash
# Start the FastAPI server (placeholder command)
uv run fastapi dev main.py
```

### Testing
```bash
# Run tests (once implemented)
uv run pytest
```

## Development Conventions

- **Type Safety:** Use Pydantic for request/response validation and strict type hinting.
- **Asynchronous Code:** Utilize `asyncio` and FastAPI's async capabilities for non-blocking I/O, especially for streaming.
- **Dependency Management:** All dependencies must be managed via `uv`. Use `uv add <package>` to add new libraries.
- **LLM Integration:** Use the latest official SDKs for Anthropic and OpenAI.
