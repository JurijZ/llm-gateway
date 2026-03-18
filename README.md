

## Start coding agents
```bash
npx @google/gemini-cli
```

### Start FastAPI server
```bash
cd app/
uv run fastapi dev main.py
```

Swagger is on http://127.0.0.1:8000/docs

Most basic request body:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "How many planets in the solar system?"
    }
  ],
  "stream": true
}
```

If a value other than "openai" or "anthropic" is provided, the system will fall back to the default provider according to the active routing strategy.
```json
{
  "messages": [
    {
      "role": "user",
      "content": "How many planets in the solar system?"
    }
  ],
  "model_preference": "string",
  "stream": true
}
```