

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

Request body example:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "How many planets in the solar system?"
    }
  ],
  "model_preference": "gpt-5.2",
  "stream": true
}
```

If a model_preference is not provided, the system will fall back to the default according to the active routing strategy.
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
