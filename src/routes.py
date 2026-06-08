import io
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import (
    ALLOWED_CONTEXTS,
    ALLOWED_MIME_TYPES,
    MAX_FILE_SIZE,
    MOCK_OPENAI,
    logger,
)
from .helpers import (
    _http_error_detail,
    build_bulk_update_stack_payload,
    build_calendar_payload,
    build_delete_row_payload,
    build_manage_tasks_payload,
    build_none_payload,
    build_note_payload,
    build_stack_payload,
    build_summarize_context_payload,
    build_task_payload,
    build_update_cell_payload,
    detect_context_mode,
    extract_stack_schema_from_item,
    normalize_audio_mime,
    parse_context_uuid,
    process_context,
    validate_calendar_context,
    validate_note_context,
    validate_stack_context,
    validate_task_context,
)
from .models import ColumnDef
from .nlu import call_nlu, transcribe_audio
from .replay import LiveEntry, store


def _capture_request_body(context_type, context_id, cursor_position, audio, transcript,
                          note_state, dynamic_schema, task_context, packed_context, request):
    headers = {k: v for k, v in request.headers.items()}
    form_fields = {}
    for key in ("context_type", "context_id", "cursor_position", "transcript",
                "note_state", "dynamic_schema", "task_context", "packed_context"):
        val = locals()[key]
        if val is not None:
            form_fields[key] = str(val)

    audio_info = None
    if audio is not None:
        audio_info = {
            "filename": audio.filename,
            "content_type": audio.content_type,
        }

    return {
        "method": "POST",
        "path": str(request.url.path),
        "query_string": str(request.url.query),
        "client": str(request.client) if request.client else None,
        "headers": headers,
        "form_fields": form_fields,
        "audio_info": audio_info,
        "body": {
            "context_type": context_type,
            "context_id": context_id,
            "cursor_position": cursor_position,
            "has_audio": audio is not None,
            "transcript": transcript,
        },
    }


def _capture_response_body(payload: dict):
    return {
        "content_type": "application/json",
        "body": payload,
    }


def _capture_error_body(body: dict):
    return {
        "content_type": "application/json",
        "body": body,
    }


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


async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        "HTTP %s %s %s — %s",
        exc.status_code,
        request.method,
        request.url.path,
        _http_error_detail(exc.detail),
    )
    body = {"error": _http_error_detail(exc.detail)}
    headers = {k: v for k, v in request.headers.items()}
    store.add(LiveEntry(
        request={
            "method": request.method,
            "path": str(request.url.path),
            "query_string": str(request.url.query),
            "client": str(request.client) if request.client else None,
            "headers": headers,
            "form_fields": {},
            "audio_info": None,
            "body": {},
        },
        response=_capture_error_body(body),
        duration_ms=0,
        status_code=exc.status_code,
    ))
    return JSONResponse(status_code=exc.status_code, content=body)


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    logger.warning(
        "Request validation failed %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(status_code=400, content={"error": "Invalid request"})


async def pydantic_validation_exception_handler(_: Request, exc: ValidationError):
    logger.warning("Pydantic validation error: %s", exc)
    return JSONResponse(
        status_code=400,
        content={"error": "Model generated invalid parameters – please retry"},
    )


async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    body = {"error": "Internal server error"}
    headers = {k: v for k, v in request.headers.items()}
    store.add(LiveEntry(
        request={
            "method": request.method,
            "path": str(request.url.path),
            "query_string": str(request.url.query),
            "client": str(request.client) if request.client else None,
            "headers": headers,
            "form_fields": {},
            "audio_info": None,
            "body": {},
        },
        response=_capture_error_body(body),
        duration_ms=0,
        status_code=500,
    ))
    return JSONResponse(status_code=500, content=body)


async def health():
    return JSONResponse(content={"status": "ok", "api": "connected"})


async def process_voice(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    transcript: Optional[str] = Form(None),
    context_type: Optional[str] = Form(None),
    context_id: Optional[str] = Form(None),
    cursor_position: int = Form(0),
    dynamic_schema: Optional[str] = Form(None),
    note_state: Optional[str] = Form(None),
    task_context: Optional[str] = Form(None),
    packed_context: Optional[str] = Form(None),
):
    import json
    req_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    audio_mime = normalize_audio_mime(audio.content_type) if audio else None
    logger.info(
        "[%s] voice/process start context_type=%s context_id=%s packed_context_len=%s "
        "has_audio=%s has_transcript=%s audio_mime_raw=%r audio_mime=%r mock=%s",
        req_id,
        context_type,
        context_id,
        len(packed_context) if packed_context else None,
        audio is not None,
        transcript is not None,
        audio.content_type if audio else None,
        audio_mime,
        MOCK_OPENAI,
    )

    if not packed_context and not (context_type and context_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid request"
        )

    processed_context: Optional[dict] = None
    note_data: Optional[dict] = None
    columns: List[ColumnDef] = []
    task_context_data: dict = {}

    if packed_context:
        try:
            packed_ctx = json.loads(packed_context)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid packed_context JSON: {exc}")

        processed_context = process_context(packed_ctx)
        items = processed_context.get("items", [])
        if not items:
            raise HTTPException(status_code=400, detail="packed_context must contain at least one item")
        
        # Primary context is the first item
        primary_item = items[0]
        context_type = primary_item.get("type")
        context_id = primary_item.get("id")

        # Detect and log context mode
        ctx_mode = detect_context_mode(processed_context)
        logger.info("[%s] context_mode=%s context_type=%s", req_id, ctx_mode, context_type)

        if context_type not in ALLOWED_CONTEXTS:
            logger.warning("[%s] rejected: invalid primary context_type=%r", req_id, context_type)
            raise HTTPException(status_code=400, detail="Invalid context_type")

        # Set default values for backward compatible check
        if context_type == "NOTE":
            note_data = {
                "id": primary_item.get("id"),
                "userId": "",
                "title": primary_item.get("title", ""),
                "content": primary_item.get("content", ""),
                "createdAt": primary_item.get("metadata", {}).get("createdAt") or "",
                "updatedAt": primary_item.get("metadata", {}).get("lastUpdated") or primary_item.get("metadata", {}).get("last_updated") or "",
            }
        elif context_type == "STACK":
            try:
                cols_json = extract_stack_schema_from_item(primary_item)
                columns = [ColumnDef.model_validate(col) for col in cols_json]
            except Exception:
                if dynamic_schema:
                    columns = validate_stack_context(dynamic_schema)
        elif context_type == "TASK":
            task_context_data = {
                "focusedTaskId": primary_item.get("id"),
                "focusedTaskTitle": primary_item.get("title")
            }
    else:
        # Legacy single context path
        if context_type not in ALLOWED_CONTEXTS:
            logger.warning("[%s] rejected: invalid context_type=%r", req_id, context_type)
            raise HTTPException(status_code=400, detail="Invalid context_type")

        if context_type == "NOTE":
            note_data = validate_note_context(note_state, context_id)
        elif context_type == "STACK":
            columns = validate_stack_context(dynamic_schema)
        elif context_type == "TASK":
            task_context_data = validate_task_context(task_context)
        else:
            validate_calendar_context()

    parse_context_uuid(context_id)

    if transcript is None and audio is None:
        logger.warning("[%s] rejected: missing audio and transcript", req_id)
        raise HTTPException(status_code=422, detail="Provide either 'transcript' or 'audio'")

    if audio and audio_mime not in ALLOWED_MIME_TYPES:
        logger.warning(
            "[%s] rejected: invalid audio format raw=%r normalized=%r allowed=%s",
            req_id,
            audio.content_type,
            audio_mime,
            sorted(ALLOWED_MIME_TYPES),
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format (got {audio.content_type!r}, expected audio/webm or audio/mp3)",
        )

    if transcript is not None:
        transcript = transcript.strip()
        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="Provide either 'transcript' or 'audio'",
            )
    else:
        contents = await audio.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Payload too large")

        if MOCK_OPENAI:
            transcript = request.headers.get("x-mock-transcript")
            if not transcript:
                logger.warning("[%s] rejected: MOCK_OPENAI set but X-Mock-Transcript missing", req_id)
                raise HTTPException(
                    status_code=400,
                    detail="Mock mode requires X-Mock-Transcript header",
                )
        else:
            logger.info("[%s] transcribing audio (%s bytes)", req_id, len(contents))
            transcript = await transcribe_audio(
                audio.filename or "audio.webm",
                io.BytesIO(contents),
                audio_mime or "audio/webm",
            )
            logger.info("[%s] transcript len=%d preview=%r", req_id, len(transcript), transcript[:80])

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

    nlu_result = await call_nlu(
        request,
        transcript,
        context_type,
        context_data,
        req_id=req_id,
        processed_context=processed_context,
        session_id=request.headers.get("x-session-id", req_id),
        user_id=request.headers.get("x-user-id", "default"),
    )

    action = nlu_result["action"]
    conv_reply = nlu_result.get("reply")

    # Before building payload, resolve note_data, columns, and task_context_data dynamically
    # based on the resolved action using matching items in the processed_context.
    if processed_context:
        if action == "update_note":
            target_item = next((item for item in processed_context.get("items", []) if item.get("type") == "NOTE"), None)
            if target_item:
                note_data = {
                    "id": target_item.get("id"),
                    "userId": "",
                    "title": target_item.get("title", ""),
                    "content": target_item.get("content", ""),
                    "createdAt": target_item.get("metadata", {}).get("createdAt") or "",
                    "updatedAt": target_item.get("metadata", {}).get("lastUpdated") or target_item.get("metadata", {}).get("last_updated") or "",
                }
                context_id = target_item.get("id")
        elif action in ("add_stack_row", "bulk_update_stack", "update_cell", "delete_row"):
            target_item = next((item for item in processed_context.get("items", []) if item.get("type") == "STACK"), None)
            if target_item:
                try:
                    cols_json = extract_stack_schema_from_item(target_item)
                    columns = [ColumnDef.model_validate(col) for col in cols_json]
                except Exception:
                    pass
                context_id = target_item.get("id")
        elif action in ("create_task", "manage_tasks"):
            target_item = next((item for item in processed_context.get("items", []) if item.get("type") in ("TASK", "TASKS")), None)
            if target_item:
                task_context_data = {
                    "focusedTaskId": target_item.get("id"),
                    "focusedTaskTitle": target_item.get("title")
                }

    # AI PIPELINE BOUNDARY: Belt-and-suspenders guard against LLM hallucination.
    # If the context is missing or is the dummy empty context ID, reject note/stack/task operations.
    is_dummy_context = context_id == "00000000-0000-0000-0000-000000000000"

    if action == "update_note" and (note_data is None or is_dummy_context):
        action = "none"
        conv_reply = "Please select the tabs or use @mentions in the text to add the relevant material to my context."
        nlu_result["action"] = "none"
        nlu_result["reply"] = conv_reply

    if action in ("add_stack_row", "bulk_update_stack", "update_cell", "delete_row") and (not columns or is_dummy_context):
        action = "none"
        conv_reply = "Please select the tabs or use @mentions in the text to add the relevant material to my context."
        nlu_result["action"] = "none"
        nlu_result["reply"] = conv_reply

    if action == "manage_tasks":
        params = nlu_result.get("params", {}) or {}
        action_type = params.get("action_type")
        # Creating a task is allowed without context, but updating/deleting requires active context.
        if action_type in ("update", "delete") and (not task_context_data or is_dummy_context):
            action = "none"
            conv_reply = "Please select the tabs or use @mentions in the text to add the relevant material to my context."
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
    elif action == "bulk_update_stack":
        updated_data, message, reply = build_bulk_update_stack_payload(nlu_result, columns, context_id)
    elif action == "update_cell":
        updated_data, message, reply = build_update_cell_payload(nlu_result, context_id)
    elif action == "delete_row":
        updated_data, message, reply = build_delete_row_payload(nlu_result, context_id)
    elif action == "create_task":
        updated_data, message, reply = build_task_payload(nlu_result, task_context_data)
    elif action == "manage_tasks":
        updated_data, message, reply = build_manage_tasks_payload(nlu_result, task_context_data)
    elif action == "summarize_context":
        updated_data, message, reply = build_summarize_context_payload(nlu_result)
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

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "[%s] voice/process done action=%s in %.0f ms",
        req_id,
        action,
        elapsed,
    )

    # Look up pipeline trace for LiveEntry linking
    pipeline_id = nlu_result.pop("_pipeline_id", None)
    pipeline_trace = store.get_pipeline_trace_obj(pipeline_id) if pipeline_id else None

    store.add(LiveEntry(
        request=_capture_request_body(
            context_type, context_id, cursor_position,
            audio, transcript, note_state, dynamic_schema,
            task_context, packed_context, request,
        ),
        response=_capture_response_body(payload),
        duration_ms=elapsed,
        status_code=200,
        pipeline_trace=pipeline_trace,
    ))
    return JSONResponse(content=payload)


def register_routes(application: FastAPI) -> None:
    """Attach middleware, exception handlers, and API routes to the FastAPI app."""
    application.middleware("http")(enforce_max_content_length)
    application.add_exception_handler(HTTPException, custom_http_exception_handler)
    application.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    application.add_exception_handler(ValidationError, pydantic_validation_exception_handler)
    application.add_exception_handler(Exception, global_exception_handler)
    application.get("/health")(health)
    application.post("/api/v1/voice/process")(process_voice)
