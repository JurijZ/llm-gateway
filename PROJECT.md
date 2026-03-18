## Project Description

Create a FastAPI service that exposes a unified `POST /chat` endpoint. The endpoint should take a user message as input and stream a response from an LLM chunk by chunk as output. Conversation handling and follow-up messages are a nice-to-have but not part of the main scope. The Important aspect of this project is that the responses are streamed and handled by an **LLM Gateway.**

The LLM gateway should support some of these following properties:

- Support at least two LLM Providers (openai and anthropic); with the possiblity to use at least two different models on a single provider (e.g. opus and sonnet on anthropic; gpt5.2 and gpt5.4 on openai). Check the Internet for the latest model names.
- Three or more strategies to select to which provider and model to route a given `POST` request:
    - Straightforward hardcoded model (just send all requests to a given model)
    - Given a list of applicable model/providers; a strategy to spread out requests so that each individual model/provider has a comparable number of in-flight requests at any given point in time
    - Given a list of applicable models/providers, a strategy to continuously monitor which model/provider currently offers the best latency and route all requests to the provider with the lowest latency at that time.
    - Nice-to-have: feel free to propose alternative selection strategies to optimize for latency or other important metrics.
- Timeouts, retries and fallback strategies. In particular, the ability to specify a fallback for a given model to use in case the primary model fails; as well as correct handling of timeouts (think carefully about how to implement timeouts for streamed responses).
- Nice-to-have:
    - Request tracing for observability
    - Additional response types e.g. structured outputs and non-stream responses

We will not only judge the efficacy of your implementation, we are also looking for a very clean app structure that follows the FastAPI best practices. So consider that implementation and style equally matter. Remember to focus on the bigger picture of how the the FastAPI server app should look like!

## Tech Requirements

Required:

- **Python 3.12+**
- **FastAPI**
- **asyncio**
- **pydantic**

Use the latest SDKs from Anthropic and OpenAI.
Use uv for package management.