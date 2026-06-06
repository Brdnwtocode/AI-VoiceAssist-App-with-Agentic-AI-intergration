# Contract test suite

Deterministic integration tests for `POST /api/v1/voice/process` and `GET /health`, aligned with `AI_MICROSERVICE_CONTRACT.md`.

## Quick start (Windows)

```bat
run_tests.bat
```

Creates `venv` if needed, installs `requirements.txt`, starts Uvicorn with `MOCK_OPENAI=1`, runs `src.test_contract`, then stops the listener on port 8000.

## Quick start (Unix)

```bash
chmod +x run_tests.sh
./run_tests.sh
```

## Environment

| Variable       | Meaning |
|----------------|---------|
| `SERVICE_URL`  | Base URL (default `http://127.0.0.1:8000`) |
| `MOCK_OPENAI`  | Must be `1` / `true` for mock suite; server skips Whisper and GPT and uses `X-Mock-*` headers instead |

## Running tests manually

Terminal A (mock mode — no OpenAI key required):

```bash
set MOCK_OPENAI=1          # Windows CMD
# export MOCK_OPENAI=1     # Unix
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Terminal B:

```bash
set SERVICE_URL=http://127.0.0.1:8000
python -m src.test_contract --mock-openai
```

## Live OpenAI (optional)

1. Unset `MOCK_OPENAI` and set a valid `OPENAI_API_KEY` in `.env`.
2. Start the server normally.
3. Run `python -m src.test_contract --real-openai`.

Deterministic LLM cases are **skipped** in this mode; validation, error paths, `/health`, and raw `413` checks still run. Full end-to-end STT+NLU checks require real audio and are inherently non-deterministic.

## Mock headers (`MOCK_OPENAI=1`)

The server reads `X-Mock-Transcript`, `X-Mock-Tool`, and `X-Mock-Args` instead of calling OpenAI. HTTP headers are restricted to ISO-8859-1; on some Windows setups, **non-ASCII characters in `X-Mock-Transcript` can make `httpx` raise `UnicodeEncodeError`**, so the bundled tests use ASCII mock transcripts. Vietnamese can still appear in JSON bodies (e.g. `content_to_insert`) without issue.

## Sample audio

`mock_audio.py` decodes `SILENT_WEBM_BASE64` into a tiny WebM-shaped file (or uses `ffmpeg` if installed). With `MOCK_OPENAI=1`, audio is not sent to Whisper.

Generate a temp file only:

```bash
python mock_audio.py
```

## What is exercised (mock mode)

- **NOTE**: `append` and `insert_at_cursor` content rules and `updatedData` shape.
- **STACK**: dynamic columns, name → id mapping in `data`, `temp_row_*` id.
- **`none`**: `updatedData` is `null`, fixed message.
- **400**: wrong audio MIME, missing `note_state`, missing `context_type`, invalid `context_type`, invalid UUID `context_id`.
- **413**: `Content-Length` larger than 10 MiB (middleware, raw HTTP).
- **`/health`**: `200` + `openai: connected` when mocked.
