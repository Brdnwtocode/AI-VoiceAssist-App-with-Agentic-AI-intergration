# AI Brain FastAPI Microservice

## Project layout

```
├── src/                 # Application package
│   ├── config.py        # App factory, env, LiteLLM router
│   ├── routes.py        # API routes + register_routes()
│   ├── main.py          # Uvicorn entry (static + /tester)
│   ├── nlu.py           # STT + NLU pipeline
│   ├── helpers.py
│   ├── models.py
│   ├── test_contract.py # Integration tests
│   └── static/
├── docs/
├── .env                 # Secrets (project root)
└── requirements.txt
```

Stateless FastAPI microservice for voice command processing:
1. STT with Whisper (`whisper-1`)
2. NLU with GPT tool-calling (`gpt-4o-mini`)
3. Returns strict JSON action payload for Next.js to persist

## 1) Environment Variables

Create `.env` in project root:

```env
OPENAI_API_KEY=your_openai_api_key
# Optional:
SONIOX_API_KEY=
ALLOWED_ORIGINS=*
```

## 2) Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

## 3) Run

From the project root:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
# or
python -m src.main
```

Tester UI: `http://localhost:8000/tester`

## 4) Endpoints

- `GET /health`
  - `200`: `{"status":"ok","api":"connected"}`

- `POST /api/v1/voice/process` (`multipart/form-data`)
  - `audio` OR `transcript` (one required)
    - `audio`: `audio/webm` or `audio/mp3`, max 10MB
    - `transcript`: pre-transcribed text (e.g. client-side realtime STT)
  - `context_type` (required: `NOTE`, `STACK`, `TASK`, or `CALENDAR`)
  - `context_id` (required UUID)
  - `cursor_position` (optional int, default `0`)
  - `dynamic_schema` (required for `STACK`)
  - `note_state` (optional JSON for `NOTE`)
  - `task_context` (optional JSON for `TASK`)

## 5) Error Contract

All errors are JSON:

```json
{"error": "human-readable description"}
```

Status codes used: `400`, `401`, `404`, `413`, `500`.