import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Type
from uuid import UUID

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, create_model

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("voice_ai_microservice")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"audio/webm", "audio/mp3"}
OPENAI_STT_MODEL = "whisper-1"
OPENAI_NLU_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 5.0

MOCK_OPENAI = os.getenv("MOCK_OPENAI", "").strip().lower() in ("1", "true", "yes")

_api_key = os.getenv("OPENAI_API_KEY") or ""
client = openai.AsyncOpenAI(
    api_key=_api_key or "sk-mock-placeholder-not-used-when-mock-openai",
    timeout=REQUEST_TIMEOUT,
)

app = FastAPI(title="Voice AI Microservice", version="1.0.0")

_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
if _origins_env == "*":
    _cors_origins = ["*"]
else:
    _cors_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False if _cors_origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _http_error_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        try:
            return json.dumps(detail)
        except Exception:
            return str(detail)
    return str(detail)


@app.middleware("http")
async def enforce_max_content_length(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/api/v1/voice/process":
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > MAX_FILE_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={"error": "Payload too large"},
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid Content-Length header"},
                )
    return await call_next(request)


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": _http_error_detail(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_: Request, __: RequestValidationError):
    return JSONResponse(status_code=400, content={"error": "Invalid request"})


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(_: Request, exc: ValidationError):
    logger.warning("Pydantic validation error: %s", exc)
    return JSONResponse(
        status_code=400,
        content={"error": "Model generated invalid parameters – please retry"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


class UpdateNoteParams(BaseModel):
    content_to_insert: str = Field(..., description="The markdown text to insert.")
    action_type: Literal["append", "insert_at_cursor"] = Field(
        ...,
        description="append: add to end; insert_at_cursor: insert at cursor",
    )


class NoActionParams(BaseModel):
    pass


class ColumnDef(BaseModel):
    id: str
    name: str
    type: Optional[str] = "TEXT"


def data_type_to_optional(dtype: str) -> Any:
    mapping: Dict[str, Any] = {
        "TEXT": Optional[str],
        "INT": Optional[int],
        "FLOAT": Optional[float],
        "BOOLEAN": Optional[bool],
        "DATE": Optional[str],
        "SELECT": Optional[str],
    }
    return mapping.get(dtype.upper(), Optional[str])


def build_add_row_model(columns: List[Dict[str, Any]]) -> Type[BaseModel]:
    fields: Dict[str, Any] = {}
    for col in columns:
        name = col["name"]
        col_type_str = str(col.get("type", "TEXT")).upper()
        py = data_type_to_optional(col_type_str)
        fields[name] = (py, Field(default=None))
    return create_model("AddRowParams", **fields)


def build_column_mapping(columns: List[ColumnDef]) -> Dict[str, str]:
    return {col.name: col.id for col in columns}


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context_uuid(context_id: str) -> None:
    try:
        UUID(context_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid context_id UUID")


async def transcribe_audio(upload: UploadFile, request: Request) -> str:
    contents = await upload.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Payload too large")

    if MOCK_OPENAI:
        transcript = request.headers.get("x-mock-transcript")
        if not transcript:
            raise HTTPException(
                status_code=400,
                detail="Mock mode requires X-Mock-Transcript header",
            )
        return transcript

    buffer = BytesIO(contents)
    buffer.name = upload.filename or "audio.webm"
    try:
        result = await client.audio.transcriptions.create(
            model=OPENAI_STT_MODEL,
            file=buffer,
            language="vi",
        )
    except Exception:
        logger.exception("STT failed")
        raise HTTPException(status_code=500, detail="Speech-to-text failed")
    finally:
        buffer.close()

    text = (getattr(result, "text", None) or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No transcript generated from audio")
    return text


async def call_nlu_mock(request: Request, context_type: str, context_data: dict) -> dict:
    tool = (request.headers.get("x-mock-tool") or "").strip()
    raw_args = request.headers.get("x-mock-args") or "{}"
    if not tool:
        raise HTTPException(status_code=400, detail="Mock mode requires X-Mock-Tool header")
    try:
        arguments = json.loads(raw_args)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid X-Mock-Args JSON")
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="X-Mock-Args must be a JSON object")

    if tool == "update_note":
        if context_type != "NOTE":
            raise HTTPException(status_code=400, detail="X-Mock-Tool not valid for NOTE context")
        validated = UpdateNoteParams.model_validate(arguments)
        return {"action": "update_note", "params": validated.model_dump()}
    if tool == "add_stack_row":
        if context_type != "STACK":
            raise HTTPException(status_code=400, detail="X-Mock-Tool not valid for STACK context")
        AddRowModel = build_add_row_model(context_data["dynamic_schema"])
        validated = AddRowModel.model_validate(arguments)
        return {"action": "add_stack_row", "params": validated.model_dump()}
    if tool == "no_action":
        NoActionParams.model_validate(arguments)
        return {"action": "none", "params": {}}

    raise HTTPException(status_code=400, detail=f"Unknown X-Mock-Tool: {tool}")


async def call_nlu_live(transcript: str, context_type: str, context_data: dict) -> dict:
    if context_type == "NOTE":
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "update_note",
                    "description": "Modify the current note content",
                    "parameters": UpdateNoteParams.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "no_action",
                    "description": "When no data change is needed",
                    "parameters": NoActionParams.model_json_schema(),
                },
            },
        ]
        note = context_data["note_state"]
        cursor = context_data.get("cursor_position", 0)
        system_content = f"""You are an AI engine for a multimodal workspace. The user is speaking in Vietnamese or English.
You must interpret the command and call ONE of the available tools. Never reply with text.

Available tools:
- update_note: modify the current note.
- no_action: when the user's speech does not require any data change.

Current context_type: NOTE.
The note details:
Title: {note['title']}
Full content (Markdown):
{note['content']}
Cursor position (character index in content): {cursor}

Rules for update_note:
- Decide whether to append to the end or insert at cursor based on the command.
- If the user says "thêm vào cuối", "append", "thêm phía sau" → action_type = "append".
- If the user says "chèn vào đây", "insert at cursor", "thêm vào vị trí hiện tại" → action_type = "insert_at_cursor".
- The content_to_insert must be the cleaned up text that the user dictated."""
    else:
        columns = context_data["dynamic_schema"]
        col_desc = "\n".join(
            f"- {col['name']} : {col.get('type', 'TEXT')}" for col in columns
        )
        AddRowModel = build_add_row_model(columns)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_stack_row",
                    "description": "Add a new row to the table",
                    "parameters": AddRowModel.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "no_action",
                    "description": "When no row should be added",
                    "parameters": NoActionParams.model_json_schema(),
                },
            },
        ]
        system_content = f"""You are an AI engine for a multimodal workspace. The user is speaking in Vietnamese or English.
You must interpret the command and call ONE of the available tools. Never reply with text.

Available tools:
- add_stack_row: create a new row in the table.
- no_action: when no row should be added.

Current context_type: STACK.
The table has the following columns (name : type):
{col_desc}

Rules for add_stack_row:
- Extract values from the command. Map spoken attributes to column names.
- If a column is not mentioned, leave it null.
- Always output a valid function call with the provided parameters."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": transcript},
    ]

    try:
        response = await client.chat.completions.create(
            model=OPENAI_NLU_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=0.0,
        )
    except Exception:
        logger.exception("OpenAI NLU error")
        raise HTTPException(status_code=500, detail="Language understanding failed")

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise HTTPException(status_code=400, detail="No tool call received from LLM")

    tool_call = tool_calls[0]
    function_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Model generated invalid JSON arguments")

    if function_name == "update_note":
        validated = UpdateNoteParams.model_validate(arguments)
        return {"action": "update_note", "params": validated.model_dump()}
    if function_name == "add_stack_row":
        AddRowModel = build_add_row_model(context_data["dynamic_schema"])
        validated = AddRowModel.model_validate(arguments)
        return {"action": "add_stack_row", "params": validated.model_dump()}
    if function_name == "no_action":
        NoActionParams.model_validate(arguments)
        return {"action": "none", "params": {}}

    raise HTTPException(status_code=400, detail=f"Unknown function {function_name}")


async def call_nlu(
    request: Request,
    transcript: str,
    context_type: str,
    context_data: dict,
) -> dict:
    if MOCK_OPENAI:
        return await call_nlu_mock(request, context_type, context_data)
    return await call_nlu_live(transcript, context_type, context_data)


@app.get("/health")
async def health():
    if MOCK_OPENAI:
        return JSONResponse(content={"status": "ok", "openai": "connected"})
    if not _api_key:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "openai": "disconnected"},
        )
    try:
        await client.models.list()
        return JSONResponse(content={"status": "ok", "openai": "connected"})
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "openai": "disconnected"},
        )


@app.post("/api/v1/voice/process")
async def process_voice(
    request: Request,
    audio: UploadFile = File(...),
    context_type: str = Form(...),
    context_id: str = Form(...),
    cursor_position: int = Form(0),
    dynamic_schema: Optional[str] = Form(None),
    note_state: Optional[str] = Form(None),
):
    req_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    logger.info("[%s] voice/process start", req_id)

    parse_context_uuid(context_id)

    if audio.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid audio format")

    if context_type not in ("NOTE", "STACK"):
        raise HTTPException(status_code=400, detail="Invalid context_type")

    if context_type == "NOTE":
        if not note_state:
            raise HTTPException(status_code=400, detail="note_state required for NOTE context")
        try:
            note_data = json.loads(note_state)
            required_keys = {"id", "userId", "title", "content", "createdAt", "updatedAt"}
            if not required_keys.issubset(note_data.keys()):
                raise ValueError("Missing fields in note_state")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid note_state: {exc}") from exc
        columns: List[ColumnDef] = []
    else:
        if not dynamic_schema:
            raise HTTPException(status_code=400, detail="dynamic_schema required for STACK context")
        try:
            schema_list = json.loads(dynamic_schema)
            if not isinstance(schema_list, list):
                raise ValueError("Must be an array")
            columns = [ColumnDef.model_validate(col) for col in schema_list]
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid dynamic_schema: {exc}") from exc
        note_data = None

    transcript = await transcribe_audio(audio, request)

    context_data: Dict[str, Any] = {
        "context_type": context_type,
        "context_id": context_id,
        "cursor_position": cursor_position,
    }
    if context_type == "NOTE":
        context_data["note_state"] = note_data
    else:
        context_data["dynamic_schema"] = [c.model_dump() for c in columns]

    nlu_result = await call_nlu(request, transcript, context_type, context_data)

    action = nlu_result["action"]
    payload: Dict[str, Any] = {
        "transcript": transcript,
        "action": action,
        "success": True,
        "message": "",
    }

    if action == "update_note":
        params = nlu_result["params"]
        content_to_insert = params["content_to_insert"]
        action_type = params["action_type"]
        current_content = note_data["content"]
        if action_type == "append":
            new_content = current_content + "\n" + content_to_insert
        else:
            pos = max(0, min(int(cursor_position), len(current_content)))
            new_content = current_content[:pos] + content_to_insert + current_content[pos:]
        payload["updatedData"] = {
            "id": note_data["id"],
            "title": note_data["title"],
            "content": new_content,
            "createdAt": note_data["createdAt"],
            "updatedAt": utc_iso_z(),
        }
        payload["message"] = "Note updated"
    elif action == "add_stack_row":
        params = nlu_result["params"]
        col_mapping = build_column_mapping(columns)
        data: Dict[str, Any] = {}
        for col_name, value in params.items():
            if value is None:
                continue
            col_id = col_mapping.get(col_name)
            if col_id:
                data[col_id] = value
            else:
                logger.warning("Column name '%s' not in schema, ignoring", col_name)
        row_id = f"temp_row_{int(time.time() * 1000)}"
        payload["updatedData"] = {
            "id": row_id,
            "stackId": context_id,
            "data": data,
        }
        payload["message"] = "Row added"
    else:
        payload["updatedData"] = None
        payload["message"] = "No action recognized from command"

    logger.info(
        "[%s] voice/process done action=%s in %.0f ms",
        req_id,
        action,
        (time.perf_counter() - t0) * 1000,
    )
    return JSONResponse(content=payload)
