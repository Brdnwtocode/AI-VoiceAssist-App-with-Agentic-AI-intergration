"""
Records Automation Routes — POST /api/v1/records/automate

Stateless endpoint: receives audio/transcript, runs the extraction pipeline,
returns structured mutations for the BFF to stage and display.

Rules (per contract):
- Every top-level response field is mandatory (null/[]/"" for inapplicable)
- No DB access — process only what you receive
- Return HTTP 422 on malformed input, HTTP 500 on internal failure
"""

import time
import uuid
from typing import Optional

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import MAX_FILE_SIZE, MOCK_OPENAI, logger
from .helpers import normalize_audio_mime
from .records_models import (
    VALID_ACTIONS,
    AutomateResponse,
)
from .records_pipeline import run_records_pipeline
from .records_stt import transcribe_with_diarization
from .replay import LiveEntry, store


# ── Request/Response Capture Helpers ──────────────────────────────────────

def _capture_records_request(
    request: Request,
    audio: Optional[UploadFile],
    transcript: Optional[str],
    recording_id: Optional[str],
    user_id: Optional[str],
    action: str,
    has_audio: bool,
) -> dict:
    """Build a request snapshot for the live viewer."""
    headers = {k: v for k, v in request.headers.items()}
    form_fields = {}
    for key, val in [
        ("action", action),
        ("recording_id", recording_id or ""),
        ("user_id", user_id or ""),
        ("transcript", (transcript or "")[:200]),
    ]:
        if val:
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
            "mode": "automate",
            "action": action,
            "recording_id": recording_id or "",
            "user_id": user_id or "",
            "has_audio": has_audio,
            "has_transcript": transcript is not None,
            "transcript_preview": (transcript or "")[:200],
        },
    }


def _capture_records_response(payload: dict) -> dict:
    """Build a response snapshot for the live viewer."""
    return {
        "content_type": "application/json",
        "body": payload,
    }


async def records_automate(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    transcript: Optional[str] = Form(None),
    recording_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
    action: Optional[str] = Form("full_automate"),
):
    """Process a recording through the Agentic AI extraction pipeline.

    Returns structured mutations (notes, tasks, stacks, calendar, speakers, summary)
    for the BFF to display as staged suggestions.
    """
    req_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    has_audio = audio is not None

    # ── Validate mode ──────────────────────────────────────────────────
    if mode and mode != "automate":
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode: '{mode}'. Only 'automate' is supported.",
        )

    # ── Validate action hint ───────────────────────────────────────────
    if action not in VALID_ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action: '{action}'. Must be one of: {', '.join(sorted(VALID_ACTIONS))}",
        )

    # ── Validate: must have audio or transcript ─────────────────────────
    if transcript is None and audio is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'transcript' (string) or 'audio' (file), or both.",
        )

    # ── Validate user_id ───────────────────────────────────────────────
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=422,
            detail="'user_id' is required.",
        )

    audio_mime = normalize_audio_mime(audio.content_type) if audio else None
    final_transcript = ""
    speaker_labels = None

    logger.info(
        "[%s] records/automate start — action=%s has_audio=%s has_transcript=%s "
        "audio_mime=%r recording_id=%s user_id=%s mock=%s",
        req_id,
        action,
        has_audio,
        transcript is not None,
        audio_mime,
        recording_id,
        user_id,
        MOCK_OPENAI,
    )

    # ── Start pipeline trace ───────────────────────────────────────────
    trace = store.start_pipeline_trace(req_id, "", "RECORDS")

    # ── Resolve transcript ─────────────────────────────────────────────
    if transcript is not None:
        transcript = transcript.strip()

    if audio is not None:
        # Validate audio MIME type
        if audio_mime not in {"audio/webm", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/wav", "audio/flac"}:
            store.finish_pipeline_trace(req_id, "error")
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported audio format: {audio.content_type}. "
                        f"Expected: audio/webm, audio/mp3, audio/ogg, audio/wav, or audio/flac.",
            )

        contents = await audio.read()
        if len(contents) > MAX_FILE_SIZE:
            store.finish_pipeline_trace(req_id, "error")
            raise HTTPException(status_code=413, detail="Payload too large")

        if len(contents) == 0:
            store.finish_pipeline_trace(req_id, "error")
            raise HTTPException(
                status_code=422,
                detail="Audio file is empty.",
            )

        if MOCK_OPENAI:
            mock_transcript = request.headers.get("x-mock-transcript")
            if not mock_transcript:
                store.finish_pipeline_trace(req_id, "error")
                raise HTTPException(
                    status_code=400,
                    detail="Mock mode requires X-Mock-Transcript header",
                )
            final_transcript = mock_transcript
            store.add_pipeline_stage(req_id, "stt", "passed",
                                     {"mode": "mock", "transcript_len": len(final_transcript)},
                                     duration_ms=0)
            logger.info("[%s] mock mode — using X-Mock-Transcript (len=%d)", req_id, len(final_transcript))
        else:
            t_stt = time.perf_counter()
            logger.info("[%s] transcribing audio (%d bytes, %s)", req_id, len(contents), audio_mime)
            try:
                final_transcript, speaker_labels = await transcribe_with_diarization(
                    contents,
                    mime_type=audio_mime or "audio/webm",
                    filename=audio.filename or "recording.webm",
                )
                stt_elapsed = (time.perf_counter() - t_stt) * 1000
                store.add_pipeline_stage(req_id, "stt", "passed",
                                         {"transcript_len": len(final_transcript),
                                          "speaker_count": len(speaker_labels) if speaker_labels else 0},
                                         duration_ms=stt_elapsed)
            except Exception as exc:
                store.add_pipeline_stage(req_id, "stt", "failed", {"error": str(exc)[:200]})
                store.finish_pipeline_trace(req_id, "error")
                logger.exception("[%s] transcription failed", req_id)
                raise HTTPException(
                    status_code=500,
                    detail="Transcription service unavailable. Please try again or provide a transcript.",
                )

            logger.info(
                "[%s] transcription done — transcript_len=%d speaker_count=%s",
                req_id,
                len(final_transcript),
                len(speaker_labels) if speaker_labels else 0,
            )

        # If both audio and transcript provided, prefer the richer one
        if transcript and final_transcript:
            if len(final_transcript) < len(transcript) * 0.5:
                logger.warning(
                    "[%s] STT transcript (%d chars) is < 50%% of provided (%d chars) — using provided",
                    req_id, len(final_transcript), len(transcript),
                )
                final_transcript = transcript
            elif len(transcript) > len(final_transcript) * 1.5:
                logger.info(
                    "[%s] provided transcript (%d chars) >> STT (%d chars) — using STT result",
                    req_id, len(transcript), len(final_transcript),
                )
    elif transcript:
        final_transcript = transcript

    # ── Run the extraction pipeline ────────────────────────────────────
    if not final_transcript or not final_transcript.strip():
        logger.info("[%s] no transcript content — returning empty response", req_id)
        store.add_pipeline_stage(req_id, "assemble", "passed",
                                 {"summary": "", "note": "no", "tasks": 0, "stack": "no", "calendar": "no"},
                                 duration_ms=0)
        store.finish_pipeline_trace(req_id, "completed")
        response = AutomateResponse()
    else:
        try:
            response = await run_records_pipeline(
                transcript=final_transcript,
                action_hint=action,
                speaker_labels=speaker_labels,
                request_id=req_id,
            )
            # Emit assemble stage with final output summary
            store.add_pipeline_stage(req_id, "assemble", "passed",
                {"summary_len": len(response.summary),
                 "summary_preview": response.summary[:200] if response.summary else "",
                 "has_note": response.note_mutation is not None,
                 "task_count": len(response.task_mutations),
                 "has_stack": response.stack_mutation is not None,
                 "has_calendar": response.calendar_mutation is not None,
                 "speaker_count": len(response.speaker_labels) if response.speaker_labels else 0},
                duration_ms=0)
            store.finish_pipeline_trace(req_id, "completed")
        except Exception as exc:
            store.finish_pipeline_trace(req_id, "error")
            logger.exception("[%s] extraction pipeline failed", req_id)
            raise HTTPException(
                status_code=500,
                detail="AI extraction pipeline failed. Please try again.",
            )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "[%s] records/automate done in %.0f ms — summary=%d chars tasks=%d note=%s stack=%s calendar=%s speakers=%s",
        req_id,
        elapsed,
        len(response.summary),
        len(response.task_mutations),
        "yes" if response.note_mutation else "no",
        "yes" if response.stack_mutation else "no",
        "yes" if response.calendar_mutation else "no",
        len(response.speaker_labels) if response.speaker_labels else 0,
    )

    payload = response.model_dump()

    # ── Log to live viewer ─────────────────────────────────────────────
    store.add(LiveEntry(
        request=_capture_records_request(
            request, audio, transcript, recording_id, user_id, action, has_audio,
        ),
        response=_capture_records_response(payload),
        duration_ms=elapsed,
        status_code=200,
        pipeline_trace=store.get_pipeline_trace_obj(req_id),
    ))

    return JSONResponse(content=payload)


def register_records_routes(app) -> None:
    """Attach records automation routes to the FastAPI app."""
    app.post("/api/v1/records/automate")(records_automate)
