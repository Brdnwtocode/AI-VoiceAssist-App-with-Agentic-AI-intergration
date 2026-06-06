## Phase J Implementation Contract – FastAPI Microservice

**Version:** 2.0 – FINAL (Corrected after audit)  
**Target Repository:** `Brdnwtocode/AI-VoiceAssist-App-with-Agentic-AI-intergration`  
**Audience:** Coding AI agent (Python/FastAPI) with full access to the codebase.  
**Goal:** Extend the existing voice processing endpoint to support `TASK` and `CALENDAR` contexts, accept `task_context`, and return `create_task` and `create_calendar_event` actions.

**You must read the entire contract before writing any code.**

---

## 1. Prerequisites – Understand the Existing Code

The current `main.py` already implements:

- `POST /api/v1/voice/process` with `audio`, `context_type` (only `NOTE` or `STACK`), `context_id`, `cursor_position`, `dynamic_schema`, `note_state`.
- A `run_resolver()` function that builds a system prompt and calls an LLM to get a JSON with `action`, `params`, `reply`.
- A `ResolverLLMOutput` Pydantic model with `action: Literal["update_note", "add_stack_row", "none"]`.
- A `process_voice` function that validates, calls the resolver, and builds a final payload with `transcript`, `action`, `success`, `message`, `updatedData`, `reply`.

**You must not change any existing behaviour for `NOTE` or `STACK`.** You will only add new branches.

---

## 2. Implementation Steps – Exact Code Changes

All changes are in `main.py`. Follow the steps in order.

### Step 1 – Update Allowed Context Types

**Find** (around line 200):

```python
if context_type not in ("NOTE", "STACK"):
    raise HTTPException(status_code=400, detail="Invalid context_type")
```

**Replace with**:

```python
ALLOWED_CONTEXTS = ("NOTE", "STACK", "TASK", "CALENDAR")
if context_type not in ALLOWED_CONTEXTS:
    raise HTTPException(status_code=400, detail="Invalid context_type")
```

---

### Step 2 – Add `task_context` Form Parameter and Parse It

**Find the `process_voice` function signature** (around line 194). Add the new parameter:

```python
async def process_voice(
    request: Request,
    audio: UploadFile = File(...),
    context_type: str = Form(...),
    context_id: str = Form(...),
    cursor_position: int = Form(0),
    dynamic_schema: Optional[str] = Form(None),
    note_state: Optional[str] = Form(None),
    task_context: Optional[str] = Form(None),   # ← add this line
):
```

**After the validation block** (after the `if context_type not in ...`), add parsing:

```python
    task_context_data = None
    if task_context:
        try:
            task_context_data = json.loads(task_context)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid task_context JSON")
```

---

### Step 3 – Add New Pydantic Models

**Find the existing models** (after `class AddStackRowParams`). Add these:

```python
class CreateTaskParams(BaseModel):
    title: str = Field(..., description="Task title")
    description: str = Field(default="", description="Task description")
    status: Literal["TODO", "IN_PROGRESS", "DONE"] = Field(default="TODO")
    priority: Literal["LOW", "MEDIUM", "HIGH"] = Field(default="MEDIUM")
    assignee: Optional[str] = Field(default=None, description="Free text assignee")
    dueDate: Optional[str] = Field(default=None, description="ISO 8601 datetime string")
    parentId: Optional[str] = Field(default=None, description="UUID of parent task (for subtasks)")

class CreateCalendarEventParams(BaseModel):
    title: str = Field(..., description="Event title")
    notes: str = Field(default="", description="Optional notes")
    startAt: str = Field(..., description="ISO 8601 datetime string")
    endAt: str = Field(..., description="ISO 8601 datetime string")
    allDay: bool = Field(default=False)
    color: str = Field(default="#5645d4", pattern=r"^#[0-9A-Fa-f]{6}$")
```

> `Literal` is already imported. No extra import needed.

---

### Step 4 – Extend `ResolverLLMOutput.action`

**Find** (around line 79):

```python
class ResolverLLMOutput(BaseModel):
    action: Literal["update_note", "add_stack_row", "none"]
    params: dict
    reply: Optional[str]
```

**Change to**:

```python
    action: Literal["update_note", "add_stack_row", "create_task", "create_calendar_event", "none"]
```

---

### Step 5 – Update the Resolver System Prompt

**Find the `system = f"""...` string inside `run_resolver`** (around line 110). **Replace the entire string** with the following (preserve the `{trusted_block}`, `{rid}`, `{transcript}` placeholders – they are already there):

```python
    system = f"""You are the Resolver NLU for a multimodal workspace.
Return ONLY valid JSON (no markdown) with this exact shape:
{{ "action": "update_note" | "add_stack_row" | "create_task" | "create_calendar_event" | "none", "params": {{ ... }}, "reply": null or a string }}

Rules:
- NOTE context: you may use update_note (params: content_to_insert, action_type append|insert_at_cursor) or none.
- STACK context: you may use add_stack_row (params: column names from schema to values) or none.
- TASK context: you may use create_task or none.
  * If task_context is provided, and the user says "add subtask", set parentId to the focusedTaskId from task_context.
  * Otherwise create a root task (parentId = null).
  * For due dates, interpret relative expressions and return as UTC ISO 8601 string.
- CALENDAR context: you may use create_calendar_event or none.
  * Interpret relative time expressions ("tomorrow at 3pm", "next Monday") relative to today.
  * Default duration is 1 hour if endAt not specified.
- For data-changing actions, set reply to null. For none with only chit‑chat, set params to {{}} and put text in reply.

[TRUSTED CONTEXT]
{trusted_block}
The user transcript is enclosed between two unique markers below.
<<<{rid}_START>>>
{transcript}
<<<{rid}_END>>>
"""
```

---

### Step 6 – Add Dispatch in `run_resolver` (No Enrichment Here)

Inside `run_resolver`, after the `if action == "add_stack_row":` block, add **only validation and model conversion**. Do **not** try to access `task_context_data` – it is not in scope.

```python
    if action == "create_task":
        if context_type != "TASK":
            raise HTTPException(status_code=400, detail="Resolver action invalid for context")
        validated = CreateTaskParams.model_validate(params)
        # Note: parentId enrichment (from task_context) will be done later in process_voice
        return {"action": "create_task", "params": validated.model_dump(), "reply": None}

    if action == "create_calendar_event":
        if context_type != "CALENDAR":
            raise HTTPException(status_code=400, detail="Resolver action invalid for context")
        validated = CreateCalendarEventParams.model_validate(params)
        return {"action": "create_calendar_event", "params": validated.model_dump(), "reply": None}
```

---

### Step 7 – Add Dispatch in `process_voice` (Where Enrichment Happens)

In `process_voice`, after the resolver returns and you have `resolver_result`, **add two new branches** before the final `else` that sets `action = "none"`.

**Find the existing block** that looks like:

```python
    if resolver_result["action"] == "update_note":
        # ... builds updatedData
    elif resolver_result["action"] == "add_stack_row":
        # ... builds updatedData
    else:
        action = "none"
        updatedData = {}
```

**Add after the `add_stack_row` branch and before the `else`**:

```python
    elif resolver_result["action"] == "create_task":
        action = "create_task"
        updatedData = resolver_result["params"]
        # Enrich parentId from task_context if present and parentId is not set
        if task_context_data and not updatedData.get("parentId"):
            focused_id = task_context_data.get("focusedTaskId")
            if focused_id:
                updatedData["parentId"] = focused_id

    elif resolver_result["action"] == "create_calendar_event":
        action = "create_calendar_event"
        updatedData = resolver_result["params"]
```

The `updatedData` will be sent to the Next.js BFF exactly as shown (camelCase keys: `dueDate`, `parentId`, `startAt`, `endAt`). The Next.js store already expects these keys.

---

### Step 8 – (Optional) Update Mock Handler

**Find the `call_nlu_mock` function** (around line 160). Replace `transcript_lower` with `transcript.lower()` and add these branches:

```python
    transcript_lower = transcript.lower()
    if "update note" in transcript_lower or "append" in transcript_lower:
        return {"action": "update_note", "params": {"content_to_insert": transcript, "action_type": "append"}, "reply": None}
    elif "add row" in transcript_lower:
        return {"action": "add_stack_row", "params": {"dummy_column": "mock value"}, "reply": None}
    elif "add task" in transcript_lower:
        return {"action": "create_task", "params": {"title": transcript, "priority": "MEDIUM"}, "reply": None}
    elif "add event" in transcript_lower:
        return {"action": "create_calendar_event", "params": {"title": transcript, "startAt": "2026-01-01T10:00:00Z", "endAt": "2026-01-01T11:00:00Z"}, "reply": None}
    else:
        return {"action": "none", "params": {}, "reply": "I didn't understand that command."}
```

---

## 3. Verification Tests (To Be Run by the Mediator)

These tests **must be run after the changes** to confirm correctness. Replace `test.wav` with a real audio file or a silent 1‑second WAV.

**Test 1 – Task creation (root)**  
```bash
curl -X POST http://localhost:8000/api/v1/voice/process \
  -F "audio=@test.wav" \
  -F "context_type=TASK" \
  -F "context_id=00000000-0000-0000-0000-000000000000" \
  -F "task_context='{\"focusedTaskId\":null}'"
```
Expected: HTTP 200, JSON contains `"action":"create_task"` (or `"none"` if transcript doesn't trigger). No 400 error.

**Test 2 – Subtask creation (focused task)**  
```bash
curl -X POST http://localhost:8000/api/v1/voice/process \
  -F "audio=@test.wav" \
  -F "context_type=TASK" \
  -F "context_id=00000000-0000-0000-0000-000000000000" \
  -F "task_context='{\"focusedTaskId\":\"abc-123\"}'"
```
Expected: When the resolver returns `create_task` with `parentId` missing, the returned `updatedData` will have `parentId = "abc-123"`.

**Test 3 – Calendar event**  
```bash
curl -X POST http://localhost:8000/api/v1/voice/process \
  -F "audio=@test.wav" \
  -F "context_type=CALENDAR" \
  -F "context_id=00000000-0000-0000-0000-000000000000"
```
Expected: HTTP 200, `action` is either `"create_calendar_event"` or `"none"`.

**Test 4 – Existing NOTE context still works** (regression)  
```bash
curl -X POST http://localhost:8000/api/v1/voice/process \
  -F "audio=@test.wav" \
  -F "context_type=NOTE" \
  -F "context_id=some-note-uuid" \
  -F "note_state=..."
```
Expected: Same behaviour as before – returns `update_note` or `none`.

---

## 4. Completion Criteria

The AI agent’s job is done when:

1. All code changes described in Steps 1–8 are applied exactly.
2. The service starts without syntax errors (`python -m main` or `uvicorn main:app`).
3. The mediator reports that the verification tests in Section 3 pass (or accepts that the changes are correctly implemented even if the NLU needs tuning).
4. No existing `NOTE` or `STACK` functionality is broken.

The agent must output a short completion report listing which steps were performed and any deviations.

---

## 5. Notes for the Mediator

- The `context_id` in tests must be a valid UUID (e.g., `00000000-0000-0000-0000-000000000000`). The endpoint enforces UUID format.
- The `task_context` JSON sent from Next.js uses the key `focusedTaskId`. This contract uses that key. No further reconciliation is needed.
- After the changes are merged, you (the owner) must deploy the updated FastAPI service for the Next.js app to use Tasks/Calendar voice commands.

**End of Phase J Implementation Contract (Corrected)**
