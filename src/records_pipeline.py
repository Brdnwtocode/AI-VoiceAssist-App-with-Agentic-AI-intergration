"""
Records Automation Pipeline — parallel expert extraction for long audio transcripts.

Flow:
  1. Receive full transcript (already transcribed if audio was provided)
  2. Run 5 experts in PARALLEL (asyncio.gather):
     - Summarize Expert   → summary (1-3 sentence recap)
     - Note Expert        → note_mutation (title + markdown content)
     - Task Expert        → task_mutations[] (action items)
     - Stack Expert       → stack_mutation (table extraction)
     - Calendar Expert    → calendar_mutation (event extraction)
  3. Validate each output against Pydantic models
  4. Assemble the final AutomateResponse

Each expert:
- Uses the same litellm model (RECORDS_EXTRACTOR_MODEL)
- Gets a tailored system prompt for its output type
- Fails independently — one failure doesn't block others
- Returns structured JSON only

Design:
- ~600 lines, transparent asyncio (no LangGraph dependency)
- Each expert has a 60s timeout (long transcripts need time)
- Pydantic validation catches malformed LLM output
- The action hint biases prompts but doesn't change the response shape
"""

import asyncio
import json
from typing import List, Optional, Tuple

import litellm
from pydantic import ValidationError

from .config import (
    LLM_TIMEOUT,
    RECORDS_EXTRACTOR_MODEL,
    RECORDS_EXTRACTOR_TIMEOUT,
    logger,
)
from .helpers import clean_json_output
from .records_models import (
    AutomateResponse,
    CalendarMutation,
    NoteMutation,
    SpeakerLabel,
    StackMutation,
    TaskMutation,
)
from .replay import store


# ═══════════════════════════════════════════════════════════════════════════
# Expert Prompt Templates
# ═══════════════════════════════════════════════════════════════════════════

_SUMMARIZE_SYSTEM = """You are a meeting summarizer. Read the transcript and produce a concise 1-3 sentence summary in the SAME LANGUAGE as the transcript.

Output ONLY a JSON object:
{"summary": "1-3 sentence recap here"}

Rules:
- Use the same language as the transcript (Vietnamese or English)
- If the transcript is empty or unintelligible, return {"summary": ""}
- Do NOT include markdown formatting in the JSON"""

_NOTE_SYSTEM = """You extract structured meeting notes from transcripts. Read the transcript and produce a note with a title and markdown body.

Output ONLY a JSON object:
{"note_mutation": {"title": "Note title", "content": "Markdown body", "folder_id": null}}

Rules:
- Use the same language as the transcript
- The content should be well-structured markdown (headings, bullets, bold for key points)
- folder_id is ALWAYS null (the BFF handles folder assignment)
- If there is no meaningful content for a note, return {"note_mutation": null}
- Do NOT include markdown formatting around the JSON itself"""

_TASK_SYSTEM = """You extract action items and to-dos from meeting transcripts. Read the transcript and produce a list of tasks.

Output ONLY a JSON object:
{"task_mutations": [{"title": "...", "description": "...", "status": "TODO", "priority": "MEDIUM", "assignee": null, "due_date": null}]}

Rules:
- Each task must have a clear, actionable title
- description should add context (who, what, why) — can be null if the title is self-explanatory
- status is always "TODO" (the user confirms later)
- priority: "HIGH" for urgent/deadline-driven, "MEDIUM" for important, "LOW" for nice-to-have
- assignee: extract the person's name if mentioned (e.g., "Anh Tuấn làm...") or leave null
- due_date: ISO 8601 string if a deadline is mentioned (e.g., "thứ 6 tuần này" → compute from context), or null
- If there are no tasks, return {"task_mutations": []}
- Use the same language as the transcript for titles/descriptions
- Do NOT include markdown formatting around the JSON"""

_STACK_SYSTEM = """You extract structured tabular data from meeting transcripts. Read the transcript and produce a stack (table) definition.

Output ONLY a JSON object:
{"stack_mutation": {"stack_id": null, "stack_name": "Table title", "columns": [{"name": "Col1", "type": "TEXT"}], "rows": [{"Col1": "value1"}]}}

Rules:
- stack_id is ALWAYS null (the BFF creates a new stack if needed)
- stack_name: a short descriptive title for the table
- columns: define each column with name and type (TEXT, INT, FLOAT, BOOLEAN, DATE, SELECT)
- rows: array of objects where keys match column names exactly
- Types: use TEXT for strings, INT for whole numbers, FLOAT for decimals, DATE for dates, BOOLEAN for true/false, SELECT for choice fields
- If no tabular data is found, return {"stack_mutation": null}
- Do NOT include markdown formatting around the JSON"""

_CALENDAR_SYSTEM = """You extract calendar events and meeting schedules from transcripts. Read the transcript and produce a calendar event.

Output ONLY a JSON object:
{"calendar_mutation": {"title": "Event title", "notes": "Optional details", "start_at": "ISO8601", "end_at": "ISO8601", "all_day": false}}

Rules:
- title: short event name
- notes: additional context (location, agenda, attendees) — can be null
- start_at / end_at: MUST be ISO 8601 datetime strings (e.g., "2026-06-15T14:00:00+07:00")
- If only a date is mentioned (no time), set all_day: true and use midnight for times
- If relative dates are used ("thứ 6 tuần này", "next Monday"), estimate based on the transcript context
- If no event is found, return {"calendar_mutation": null}
- Do NOT include markdown formatting around the JSON"""


# ═══════════════════════════════════════════════════════════════════════════
# Expert Call Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_user_message(transcript: str, action_hint: str) -> str:
    """Build the user message with transcript and optional focus hint."""
    hint_map = {
        "full_automate": "Extract ALL possible information: summary, notes, tasks, tables, and calendar events.",
        "summarize": "Focus primarily on producing a high-quality summary. Still check for tasks, notes, tables, and events.",
        "extract_tasks": "Focus primarily on extracting action items and tasks. Still check for other information.",
        "populate_stack": "Focus primarily on extracting tabular/structured data. Still check for other information.",
        "identify_speakers": "Speaker identification is handled separately. Focus on content extraction.",
        "create_calendar": "Focus primarily on extracting calendar events and schedules. Still check for other information.",
    }
    focus = hint_map.get(action_hint, hint_map["full_automate"])

    return f"""{focus}

Transcript:
---
{transcript}
---"""


def _extract_llm_metadata(response, model_requested: str) -> dict:
    """Extract model, token, and provider metadata from a litellm response."""
    meta = {}
    # Model actually used (resolved by litellm/router)
    actual_model = getattr(response, "model", None) or model_requested
    meta["model"] = actual_model
    # Provider extraction from model name (e.g., "openrouter/..." → "openrouter")
    if "/" in actual_model:
        meta["provider"] = actual_model.split("/")[0]
    else:
        meta["provider"] = "unknown"
    # Token usage
    usage = getattr(response, "usage", None)
    if usage:
        meta["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
        meta["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
        meta["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
    else:
        meta["prompt_tokens"] = 0
        meta["completion_tokens"] = 0
        meta["total_tokens"] = 0
    return meta


async def _call_expert(
    expert_name: str,
    system_prompt: str,
    user_message: str,
    model: str = None,
    temperature: float = 0.1,
    timeout: float = None,
) -> Tuple[str, Optional[dict], dict]:
    """Call an LLM expert and return (raw_response, parsed_json_or_None, llm_metadata).

    Each expert fails independently — never let one expert crash the pipeline.
    """
    model = model or RECORDS_EXTRACTOR_MODEL
    timeout = timeout or RECORDS_EXTRACTOR_TIMEOUT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=min(timeout, LLM_TIMEOUT),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error("[RecordsPipeline] %s expert timed out after %.0fs", expert_name, timeout)
        return "", None, {"model": model, "provider": model.split("/")[0] if "/" in model else "unknown", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": "timeout"}
    except Exception as exc:
        logger.error("[RecordsPipeline] %s expert LLM call failed: %s", expert_name, exc)
        return "", None, {"model": model, "provider": model.split("/")[0] if "/" in model else "unknown", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": str(exc)[:100]}

    raw = (response.choices[0].message.content or "").strip()
    meta = _extract_llm_metadata(response, model)
    logger.info("[RecordsPipeline] %s expert response len=%d model=%s tokens=%d",
                expert_name, len(raw), meta.get("model", "?"), meta.get("total_tokens", 0))

    try:
        parsed = clean_json_output(raw)
        return raw, parsed, meta
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("[RecordsPipeline] %s expert returned invalid JSON: %s — raw=%r", expert_name, exc, raw[:200])
        return raw, None, meta


# ═══════════════════════════════════════════════════════════════════════════
# Individual Expert Functions
# ═══════════════════════════════════════════════════════════════════════════

async def _summarize_expert(transcript: str, action_hint: str) -> Tuple[str, dict]:
    """Extract a 1-3 sentence summary. Returns (summary, llm_metadata)."""
    user_msg = _build_user_message(transcript, action_hint)
    _, parsed, meta = await _call_expert("summarize", _SUMMARIZE_SYSTEM, user_msg)
    if parsed and isinstance(parsed.get("summary"), str):
        return parsed["summary"].strip(), meta
    return "", meta


async def _note_expert(transcript: str, action_hint: str) -> Tuple[Optional[NoteMutation], dict]:
    """Extract a suggested note. Returns (note_mutation_or_none, llm_metadata)."""
    user_msg = _build_user_message(transcript, action_hint)
    _, parsed, meta = await _call_expert("note", _NOTE_SYSTEM, user_msg)
    if parsed is None:
        return None, meta
    nm = parsed.get("note_mutation")
    if nm is None:
        return None, meta
    if not isinstance(nm, dict):
        return None, meta
    if not nm.get("title") or not nm.get("content"):
        return None, meta
    try:
        return NoteMutation.model_validate(nm), meta
    except ValidationError as exc:
        logger.warning("[RecordsPipeline] note expert validation failed: %s", exc)
        return None, meta


async def _task_expert(transcript: str, action_hint: str) -> Tuple[List[TaskMutation], dict]:
    """Extract task/action items. Returns (task_mutations, llm_metadata)."""
    user_msg = _build_user_message(transcript, action_hint)
    _, parsed, meta = await _call_expert("task", _TASK_SYSTEM, user_msg)
    if parsed is None:
        return [], meta
    raw_tasks = parsed.get("task_mutations")
    if not isinstance(raw_tasks, list):
        return [], meta
    tasks: List[TaskMutation] = []
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue
        if not item.get("title"):
            continue
        try:
            tasks.append(TaskMutation.model_validate(item))
        except ValidationError as exc:
            logger.warning("[RecordsPipeline] task validation failed for %r: %s", item.get("title", "?"), exc)
    return tasks, meta


async def _stack_expert(transcript: str, action_hint: str) -> Tuple[Optional[StackMutation], dict]:
    """Extract a structured table. Returns (stack_mutation_or_none, llm_metadata)."""
    user_msg = _build_user_message(transcript, action_hint)
    _, parsed, meta = await _call_expert("stack", _STACK_SYSTEM, user_msg)
    if parsed is None:
        return None, meta
    sm = parsed.get("stack_mutation")
    if sm is None:
        return None, meta
    if not isinstance(sm, dict):
        return None, meta
    if not sm.get("stack_name"):
        return None, meta
    # Validate columns
    cols = sm.get("columns", [])
    if not isinstance(cols, list) or len(cols) == 0:
        return None, meta
    try:
        return StackMutation.model_validate(sm), meta
    except ValidationError as exc:
        logger.warning("[RecordsPipeline] stack expert validation failed: %s", exc)
        return None, meta


async def _calendar_expert(transcript: str, action_hint: str) -> Tuple[Optional[CalendarMutation], dict]:
    """Extract a calendar event. Returns (calendar_mutation_or_none, llm_metadata)."""
    user_msg = _build_user_message(transcript, action_hint)
    _, parsed, meta = await _call_expert("calendar", _CALENDAR_SYSTEM, user_msg)
    if parsed is None:
        return None, meta
    cm = parsed.get("calendar_mutation")
    if cm is None:
        return None, meta
    if not isinstance(cm, dict):
        return None, meta
    if not cm.get("title") or not cm.get("start_at") or not cm.get("end_at"):
        return None, meta
    try:
        return CalendarMutation.model_validate(cm), meta
    except ValidationError as exc:
        logger.warning("[RecordsPipeline] calendar expert validation failed: %s", exc)
        return None, meta


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline Entry Point
# ═══════════════════════════════════════════════════════════════════════════

async def run_records_pipeline(
    transcript: str,
    action_hint: str = "full_automate",
    speaker_labels: Optional[List[SpeakerLabel]] = None,
    request_id: str = "",
) -> AutomateResponse:
    """Run all 5 extraction experts in parallel and assemble the response.

    Args:
        transcript: The full transcribed text (already transcribed if audio was provided).
        action_hint: One of the valid action strings — biases expert prompts.
        speaker_labels: Optional diarization from STT (passed through, not LLM-generated).
        request_id: Pipeline trace ID for live viewer integration.

    Returns:
        AutomateResponse with all fields populated.
    """
    if not transcript or not transcript.strip():
        return AutomateResponse()

    logger.info(
        "[RecordsPipeline] starting parallel extraction — transcript_len=%d action=%s",
        len(transcript), action_hint,
    )

    t0 = asyncio.get_event_loop().time()

    # ── Run all 5 experts in parallel ─────────────────────────────────
    # Wrap each expert to emit pipeline stage traces with rich output previews + LLM metadata
    async def _traced_expert(name: str, coro, enrich_data_fn=None):
        t_start = asyncio.get_event_loop().time()
        try:
            result_tuple = await coro  # Each expert now returns (value, llm_metadata)
            elapsed = (asyncio.get_event_loop().time() - t_start) * 1000
            if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                value, llm_meta = result_tuple
            else:
                value, llm_meta = result_tuple, {}
            if request_id:
                data = {"elapsed_ms": round(elapsed, 1), "input_len": len(transcript)}
                # Include LLM metadata
                if llm_meta:
                    data["model"] = llm_meta.get("model", "?")
                    data["provider"] = llm_meta.get("provider", "?")
                    data["prompt_tokens"] = llm_meta.get("prompt_tokens", 0)
                    data["completion_tokens"] = llm_meta.get("completion_tokens", 0)
                    data["total_tokens"] = llm_meta.get("total_tokens", 0)
                if enrich_data_fn:
                    data.update(enrich_data_fn(value))
                store.add_pipeline_stage(request_id, name, "passed", data, duration_ms=elapsed)
            return result_tuple
        except Exception as exc:
            elapsed = (asyncio.get_event_loop().time() - t_start) * 1000
            if request_id:
                store.add_pipeline_stage(request_id, name, "failed",
                                         {"error": str(exc)[:200], "input_len": len(transcript)},
                                         duration_ms=elapsed)
            raise

    results = await asyncio.gather(
        _traced_expert("summarize_expert", _summarize_expert(transcript, action_hint),
                       enrich_data_fn=lambda v: {"has_summary": bool(v), "summary_preview": v[:200] if v else ""}),
        _traced_expert("note_expert", _note_expert(transcript, action_hint),
                       enrich_data_fn=lambda v: {"has_note": v is not None, "note_title": v.title[:100] if v else "", "content_len": len(v.content) if v else 0}),
        _traced_expert("task_expert", _task_expert(transcript, action_hint),
                       enrich_data_fn=lambda v: {"task_count": len(v), "first_task": v[0].title[:100] if v else ""}),
        _traced_expert("stack_expert", _stack_expert(transcript, action_hint),
                       enrich_data_fn=lambda v: {"has_stack": v is not None, "stack_name": v.stack_name[:100] if v else "", "col_count": len(v.columns) if v else 0, "row_count": len(v.rows) if v else 0}),
        _traced_expert("calendar_expert", _calendar_expert(transcript, action_hint),
                       enrich_data_fn=lambda v: {"has_event": v is not None, "event_title": v.title[:100] if v else "", "all_day": v.all_day if v else False}),
        return_exceptions=True,
    )

    elapsed = (asyncio.get_event_loop().time() - t0) * 1000

    # ── Unpack results (handle exceptions gracefully) ─────────────────
    # Each result is now (value, llm_metadata) tuple or an exception
    summary = ""
    note_mutation = None
    task_mutations: List[TaskMutation] = []
    stack_mutation = None
    calendar_mutation = None

    for i, result in enumerate(results):
        expert_names = ["summarize", "note", "task", "stack", "calendar"]
        name = expert_names[i] if i < len(expert_names) else f"expert_{i}"

        if isinstance(result, BaseException):
            logger.error("[RecordsPipeline] %s expert crashed: %s", name, result)
            continue

        # Unpack (value, metadata) tuple
        if isinstance(result, tuple) and len(result) == 2:
            value, _meta = result
        else:
            value = result

        if i == 0:
            summary = value if isinstance(value, str) else ""
        elif i == 1:
            note_mutation = value if isinstance(value, NoteMutation) or value is None else None
        elif i == 2:
            task_mutations = value if isinstance(value, list) else []
        elif i == 3:
            stack_mutation = value if isinstance(value, StackMutation) or value is None else None
        elif i == 4:
            calendar_mutation = value if isinstance(value, CalendarMutation) or value is None else None

    logger.info(
        "[RecordsPipeline] done in %.0f ms — summary=%d chars tasks=%d note=%s stack=%s calendar=%s",
        elapsed,
        len(summary),
        len(task_mutations),
        "yes" if note_mutation else "no",
        "yes" if stack_mutation else "no",
        "yes" if calendar_mutation else "no",
    )

    return AutomateResponse(
        note_mutation=note_mutation,
        task_mutations=task_mutations,
        stack_mutation=stack_mutation,
        calendar_mutation=calendar_mutation,
        speaker_labels=speaker_labels,
        summary=summary,
    )
