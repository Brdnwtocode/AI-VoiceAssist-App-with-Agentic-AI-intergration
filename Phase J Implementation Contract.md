# Phase J Implementation Contract – FastAPI Microservice Extension

## For a Coding AI (Python/FastAPI) – Execute Exactly

**Contract Version:** 1.0 – FINAL  
**Target Repository:** `Brdnwtocode/AI-VoiceAssist-App-with-Agentic-AI-intergration` (FastAPI microservice)  
**Audience:** Coding AI agent with full access to the codebase.  
**Goal:** Extend the existing FastAPI voice processing endpoint to support `TASK` and `CALENDAR` contexts, accept `task_context`, and return `create_task` and `create_calendar_event` actions as required by the main Next.js application.

> **Rule:** Do not change any existing behavior for `NOTE` or `STACK` contexts. Add new functionality only.

---

## 1. Prerequisites – Codebase Familiarity

The agent must first understand the existing code. Key files:

- `main.py` – Contains the FastAPI app, the `/api/v1/voice/process` endpoint, the resolver logic, and the system prompt.
- `requirements.txt` – Already includes `fastapi`, `pydantic`, `python-multipart`, `litellm`, etc. No new dependencies required.

**Current endpoint behavior (relevant to this contract):**

- Accepts `audio`, `context_type` (only `"NOTE"` or `"STACK"` allowed), `context_id`, `cursor_position`, `dynamic_schema`, `note_state`.
- Validates `context_type` against a hardcoded tuple.
- Uses a **Resolver LLM** (no OpenAI tool‑calling) that outputs a JSON with `action`, `params`, `reply`.
- The resolver system prompt only knows `update_note` and `add_stack_row`.
- Returns actions that the Next.js BFF forwards to the store.

---

## 2. Implementation Steps – Exact Code Changes

All changes are to be made in `main.py`. Follow the steps in order.

### Step 1 – Update Allowed Context Types

**Find the line** (approx. line 200) that contains:

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

### Step 2 – Add `task_context` Form Parameter

**Find the function signature of `process_voice`** (approx. line 194). It currently has parameters like:

```python
async def process_voice(
    request: Request,
    audio: UploadFile = File(...),
    context_type: str = Form(...),
    context_id: str = Form(...),
    cursor_position: int = Form(0),
    dynamic_schema: Optional[str] = Form(None),
    note_state: Optional[str] = Form(None),
):
```

**Add a new parameter** after `note_state`:

```python
    task_context: Optional[str] = Form(None),
```

**Then, immediately after validating `context_type`, add the parsing logic** (approximately after the validation block). Insert:

```python
    task_context_data = None
    if task_context:
        try:
            task_context_data = json.loads(task_context)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid task_context JSON")
```

Make sure `import json` is present at the top of the file (it already is).

---

### Step 3 – Add New Pydantic Models for Task & Calendar

**Find the existing models** (near line 55, after `class UpdateNoteParams(BaseModel):` and `class AddStackRowParams(BaseModel):`).  

**Add these two new models** after the existing ones:

```python
class CreateTaskParams(BaseModel):
    title: str = Field(..., description="Task title")
    description: str = Field(default="", description="Task description")
    status: Literal["TODO", "IN_PROGRESS", "DONE"] = Field(default="TODO")
    priority: Literal["LOW", "MEDIUM", "HIGH"] = Field(default="MEDIUM")
    assignee: Optional[str] = Field(default=None, description="Free text assignee")
    dueDate: Optional[str] = Field(default=None, description="ISO 8601 datetime string")
    parentId: Optional[str] = Field(default=None, description="UUID of parent task for subtasks")


class CreateCalendarEventParams(BaseModel):
    title: str = Field(..., description="Event title")
    notes: str = Field(default="", description="Optional notes")
    startAt: str = Field(..., description="ISO 8601 datetime string")
    endAt: str = Field(..., description="ISO 8601 datetime string")
    allDay: bool = Field(default=False)
    color: str = Field(default="#5645d4", pattern=r"^#[0-9A-Fa-f]{6}$")
```

**Note:** The `Field` imports are already present. If `Literal` is not imported, add it: `from typing import Literal`.

---

### Step 4 – Extend the Resolver Output Enum

**Find the `ResolverLLMOutput` class** (approx. line 79). It has:

```python
class ResolverLLMOutput(BaseModel):
    action: Literal["update_note", "add_stack_row", "none"]
    params: dict
    reply: Optional[str]
```

**Change the `action` literal to**:

```python
    action: Literal["update_note", "add_stack_row", "create_task", "create_calendar_event", "none"]
```

---

### Step 5 – Replace the Resolver System Prompt

**Find the system prompt string** inside the `run_resolver` function (approx. lines 110‑123). It currently contains a block of text with `update_note` and `add_stack_row`.

**Replace the entire system prompt with the following** (preserve the `{trusted_block}`, `{rid}`, `{transcript}` placeholders – they are already there):

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

**Important:** The `trusted_block` variable is already defined earlier in the function. Do not modify its creation.

---

### Step 6 – Add Handling for `create_task` and `create_calendar_event` in the Resolver Response

**Find the block** after the resolver returns `action` and `params` (approx. line 142, after `if action == "add_stack_row":`). There is already validation and return for `update_note` and `add_stack_row`.

**Insert the following code** after the `add_stack_row` block and **before** the final return that raises an error for invalid actions:

```python
    if action == "create_task":
        if context_type != "TASK":
            raise HTTPException(status_code=400, detail="Resolver action invalid for context")
        validated = CreateTaskParams.model_validate(params)
        # If task_context contains a focusedTaskId and parentId is not explicitly set,
        # assume the user wants a subtask under the focused task.
        if task_context_data and validated.parentId is None:
            focused_id = task_context_data.get("focusedTaskId")
            if focused_id:
                validated.parentId = focused_id
        return {"action": "create_task", "params": validated.model_dump(), "reply": None}

    if action == "create_calendar_event":
        if context_type != "CALENDAR":
            raise HTTPException(status_code=400, detail="Resolver action invalid for context")
        validated = CreateCalendarEventParams.model_validate(params)
        return {"action": "create_calendar_event", "params": validated.model_dump(), "reply": None}
```

**Note:** The variables `task_context_data` and `context_type` are already in scope. Ensure the return dictionary uses `"action"` and `"params"` keys exactly as the existing code does for `update_note` and `add_stack_row`.

---

### Step 7 – (Optional but Recommended) Update the Mock Handler

If the environment uses `MOCK_OPENAI=1` for testing without calling a real LLM, update the `call_nlu_mock` function (approx. line 160). Add two new branches:

**Find the `if` block that checks `transcript.lower()` and adds branches for `"add note"` and `"add row"`.**  

Add:

```python
    elif "add task" in transcript_lower:
        return {"action": "create_task", "params": {"title": transcript, "priority": "MEDIUM"}, "reply": None}
    elif "add event" in transcript_lower:
        return {"action": "create_calendar_event", "params": {"title": transcript, "startAt": "2026-01-01T10:00:00Z", "endAt": "2026-01-01T11:00:00Z"}, "reply": None}
```

These are minimal examples; the mock is only for testing the flow.

---

## 3. Verification Checklist (To Be Performed by the Mediator/Owner)

After making the changes, the FastAPI service must be restarted. The mediator should run these tests and report the results to the CAi.

### 3.1 Endpoint Acceptance Test

**Command** (replace `test.wav` with any valid audio file or use `--form` with an empty file if the service doesn't require actual audio for testing – but the service expects a file; you can use a silent 1‑second WAV):

```bash
curl -X POST http://localhost:8000/api/v1/voice/process \
  -F "audio=@test.wav" \
  -F "context_type=TASK" \
  -F "context_id=none" \
  -F "task_context='{\"focusedTaskId\":null,\"focusedTaskTitle\":null}'"
```

**Expected:** HTTP 200 with JSON containing `{"action":"create_task",...}` (or `"action":"none"` if the transcript didn't trigger a task creation). The important part is **no 400 error about invalid context_type**.

### 3.2 Subtask Focus Test

```bash
curl -X POST http://localhost:8000/api/v1/voice/process \
  -F "audio=@test.wav" \
  -F "context_type=TASK" \
  -F "context_id=abc-123" \
  -F "task_context='{\"focusedTaskId\":\"abc-123\",\"focusedTaskTitle\":\"Fix bug\"}'"
```

**Expected:** If the transcript says "add subtask test", the returned `params` should have `parentId = "abc-123"`.

### 3.3 Calendar Context Test

```bash
curl -X POST http://localhost:8000/api/v1/voice/process \
  -F "audio=@test.wav" \
  -F "context_type=CALENDAR" \
  -F "context_id=none"
```

**Expected:** No validation error; the resolver may return `"action":"none"` or `"create_calendar_event"` depending on transcript. The key is that `context_type="CALENDAR"` is accepted.

### 3.4 Existing NOTE/STACK Regression Test

Run the same commands as before (with `context_type=NOTE` and `context_type=STACK`) to ensure they still work. They should return the same actions as before.

---

## 4. Completion Criteria

The CAi’s job is done when:

1. All code changes described in Steps 1–6 are applied exactly.
2. No existing functionality for `NOTE` or `STACK` is broken.
3. The service starts without syntax errors (run `python -m main` or `uvicorn main:app`).
4. The mediator reports that the verification tests in Section 3 pass (or the mediator accepts that the changes are correctly implemented even if the resolver’s NLU output requires tuning later – the contract is about the code structure, not the NLU accuracy).

**The CAi must output a short completion report listing which steps were performed and any deviations.**

---

## 5. Notes for the Mediator

- The CAi will not have access to the actual FastAPI logs or runtime; you must run the verification tests yourself.
- If any part of the contract is unclear, the CAi should halt and ask for clarification before writing code.
- The CAi should not modify any file other than `main.py` unless absolutely necessary (and only with explicit permission).
- After the changes are merged, you (the owner) should deploy the updated FastAPI service so that the Next.js app can use it for Tasks and Calendar voice commands.

**End of Phase J Implementation Contract**
