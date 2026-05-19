import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from config import (
    ALLOWED_CONTEXTS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE,
    MOCK_OPENAI,
    app,
    logger,
)
from helpers import (
    _http_error_detail,
    build_calendar_payload,
    build_none_payload,
    build_note_payload,
    build_stack_payload,
    build_task_payload,
    parse_context_uuid,
    validate_calendar_context,
    validate_note_context,
    validate_stack_context,
    validate_task_context,
)
from models import ColumnDef
from nlu import call_nlu, transcribe_audio


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


@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok", "api": "connected"})


@app.post("/api/v1/voice/process")
async def process_voice(
    request: Request,
    audio: UploadFile = File(...),
    context_type: str = Form(...),
    context_id: str = Form(...),
    cursor_position: int = Form(0),
    dynamic_schema: Optional[str] = Form(None),
    note_state: Optional[str] = Form(None),
    task_context: Optional[str] = Form(None),
):
    req_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    logger.info("[%s] voice/process start", req_id)

    parse_context_uuid(context_id)

    if audio.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid audio format")

    if context_type not in ALLOWED_CONTEXTS:
        raise HTTPException(status_code=400, detail="Invalid context_type")

    note_data: Optional[dict] = None
    columns: List[ColumnDef] = []
    task_context_data: dict = {}

    if context_type == "NOTE":
        note_data = validate_note_context(note_state, context_id)
    elif context_type == "STACK":
        columns = validate_stack_context(dynamic_schema)
    elif context_type == "TASK":
        task_context_data = validate_task_context(task_context)
    else:
        validate_calendar_context()

    contents = await audio.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Payload too large")

    if MOCK_OPENAI:
        transcript = request.headers.get("x-mock-transcript")
        if not transcript:
            raise HTTPException(
                status_code=400,
                detail="Mock mode requires X-Mock-Transcript header",
            )
    else:
        transcript = await transcribe_audio(audio.filename, audio.file, audio.content_type)

    context_data: Dict[str, Any] = {
        "context_type": context_type,
        "context_id": context_id,
        "cursor_position": cursor_position,
    }
    if context_type == "NOTE":
        context_data["note_state"] = note_data
    elif context_type == "STACK":
        context_data["dynamic_schema"] = [c.model_dump() for c in columns]
    elif context_type == "TASK":
        context_data["task_context"] = task_context_data

    nlu_result = await call_nlu(request, transcript, context_type, context_data)

    action = nlu_result["action"]
    conv_reply = nlu_result.get("reply")

    # AI PIPELINE BOUNDARY: Belt-and-suspenders guard against LLM hallucination.
    # run_resolver() already rejects cross-context actions, but if that guard is
    # ever bypassed, this is the last line of defense before a NoneType crash.
    if action == "update_note" and note_data is None:
        action = "none"
        conv_reply = "I cannot update a note because no note is currently open."
        nlu_result["action"] = "none"
        nlu_result["reply"] = conv_reply

    if action == "add_stack_row" and not dynamic_schema:
        action = "none"
        conv_reply = "I cannot add a row because no stack schema is available."
        nlu_result["action"] = "none"
        nlu_result["reply"] = conv_reply

    payload: Dict[str, Any] = {
        "transcript": transcript,
        "action": action,
        "success": True,
        "message": "",
        "updatedData": None,
        "reply": None,
    }

    if action == "update_note":
        updated_data, message, reply = build_note_payload(nlu_result, note_data, cursor_position)
    elif action == "add_stack_row":
        updated_data, message, reply = build_stack_payload(nlu_result, columns, context_id)
    elif action == "create_task":
        updated_data, message, reply = build_task_payload(nlu_result, task_context_data)
    elif action == "create_calendar_event":
        updated_data, message, reply = build_calendar_payload(nlu_result)
    else:
        updated_data, message, reply = build_none_payload(conv_reply)

    payload["updatedData"] = updated_data
    payload["message"] = message
    payload["reply"] = reply

    if payload["updatedData"] is not None:
        payload["reply"] = None
    elif payload.get("reply"):
        payload["updatedData"] = None

    logger.info(
        "[%s] voice/process done action=%s in %.0f ms",
        req_id,
        action,
        (time.perf_counter() - t0) * 1000,
    )
    return JSONResponse(content=payload)
