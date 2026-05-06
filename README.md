# AI Brain FastAPI Microservice

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

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 4) Endpoints

- `GET /health`
  - `200`: `{"status":"ok","openai":"connected"}`
  - `503`: `{"status":"error"}`

- `POST /api/v1/voice/process` (`multipart/form-data`)
  - `audio` (required, `audio/webm` or `audio/mp3`, max 10MB)
  - `context_type` (required: `NOTE` or `STACK`)
  - `context_id` (required UUID)
  - `cursor_position` (optional int, default `0`)
  - `dynamic_schema` (required for `STACK`)
  - `note_state` (required for `NOTE`)

## 5) Error Contract

All errors are JSON:

```json
{"error": "human-readable description"}
```

Status codes used: `400`, `401`, `404`, `413`, `500`.