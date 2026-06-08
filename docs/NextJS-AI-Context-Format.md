# NextJS → FastAPI Context Format

> **Audience:** FastAPI / AI microservice team.  
> **Purpose:** This document describes exactly what the NextJS BFF sends to `POST /api/v1/voice/process`.  
> Use this to build your request parsers, write integration tests, and validate your AI prompt assembly.

---

## 1. Endpoint & Transport

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `{FASTAPI_URL}/api/v1/voice/process` |
| **Content-Type** | `multipart/form-data` |
| **Max audio size** | 10 MB (enforced by BFF before forwarding) |

### HTTP Headers

| Header | Required | Description |
|--------|----------|-------------|
| `x-user-id` | **Yes** | The authenticated user's UUID (from NextAuth session). Scopes memory/ConversationBuffer per user. |
| `x-session-id` | No | Opaque session token for per-tab conversation continuity. When absent, FastAPI should start a new session. |

---

## 2. FormData Fields

The BFF sends a `multipart/form-data` body. Fields are **snake_case**.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | `File` (binary) | * | Raw audio blob (webm/opus from browser `MediaRecorder`). Mutually exclusive with `transcript`. |
| `transcript` | `string` | * | Pre-transcribed text (used when STT happens client-side via Deepgram streaming). Mutually exclusive with `audio`. |
| `packed_context` | `string` (JSON) | **Yes** (primary) | JSON-serialized [`PackedContext`](#3-packed_context-structure). Contains 1–5 context items. |
| `cursor_position` | `string` (int) | No | Character offset in the active note's raw markdown. Only meaningful when primary context is a NOTE. |
| `task_context` | `string` (JSON) | No | Focused task snapshot. Only sent when primary context is a TASK. See [§4.3](#43-task-task_context). |
| `user_id` | `string` (UUID) | **Yes** | Redundant with `x-user-id` header; provided for convenience. |

\* Exactly one of `audio` or `transcript` must be present.

### Legacy fallback fields (only when `packed_context` is absent)

| Field | Type | Description |
|-------|------|-------------|
| `context_type` | `string` | One of: `NOTE`, `STACK`, `TASK` |
| `context_id` | `string` (UUID) | ID of the active tab/item |
| `note_state` | `string` | Full raw markdown of the note (sent ONLY when `packed_context` is absent and context is NOTE) |

---

## 3. `packed_context` Structure

```json
{
  "items": [ /* ContextItem[] — 1 to 5 items */ ],
  "packedAt": "2026-06-08T12:00:00.000Z",
  "totalItems": 2
}
```

Each `ContextItem` has this shape:

```json
{
  "type": "NOTE",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Meeting Notes — June 8",
  "content": "...",
  "metadata": { "...": "..." },
  "source": "active_tab"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"NOTE"` \| `"STACK"` \| `"TASK"` \| `"CALENDAR"` | The kind of workspace item |
| `id` | `string` (UUID) | Unique identifier |
| `title` | `string` | Human-readable title |
| `content` | `string` \| `object` \| absent | Depends on type — see [§4](#4-context-item-types) |
| `metadata` | `object` | Type-specific metadata |
| `source` | `"active_tab"` \| `"user_mention"` \| `"recent_activity"` | How this item was selected |

---

## 4. Context Item Types

### 4.1 NOTE

```json
{
  "type": "NOTE",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Meeting Notes — June 8",
  "content": "# Cuộc họp lúc 5h chiều\n\n- Điểm 1: ...\n- Điểm 2: ...\n\nKết luận: ...",
  "metadata": {
    "cursorPosition": 842,
    "title": "Meeting Notes — June 8"
  },
  "source": "active_tab"
}
```

- **`content`** is the **raw markdown string** of the entire note.  
- **`metadata.cursorPosition`** is the character offset where the user's cursor was when they spoke.  
  Use this to resolve positional commands like _"thêm ... vào cuối"_ or _"chèn ... sau dòng này"_.

### 4.2 STACK

```json
{
  "type": "STACK",
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "title": "Sprint Backlog",
  "content": {
    "schema": {
      "columns": [
        { "id": "col-uuid-1", "name": "Task", "type": "TEXT", "order": 0 },
        { "id": "col-uuid-2", "name": "Status", "type": "TEXT", "order": 1 },
        { "id": "col-uuid-3", "name": "Budget", "type": "INT", "order": 2 }
      ]
    },
    "stats": {
      "rowCount": 12,
      "columnCount": 3
    },
    "dataFormat": "csv",
    "data": "Task (TEXT),Status (TEXT),Budget (INT)\nThiết kế UI,TODO,1000\nViết API,IN_PROGRESS,2000\n..."
  },
  "metadata": {
    "dataFormat": "csv",
    "rowCount": 12
  },
  "source": "active_tab"
}
```

Key points:
- **`schema.columns`** — The column definitions. Each column has `id`, `name`, `type`.  
  Column types: `TEXT`, `INT`, `FLOAT`, `BOOLEAN`, `SELECT`, `DATE`.
- **`data`** — A **CSV string** (default) or Markdown table with the first `maxRowsForFullData` (100) rows.  
  Header format: `"ColumnName (TYPE)"` so the AI can distinguish identically-named columns of different types.  
  Row IDs are intentionally **omitted** — the AI works with data, not internal keys.
- **`stats`** — Total row/column counts for the AI to understand dataset size.

#### CSV format detail

```
Task (TEXT),Status (TEXT),Budget (INT)
Thiết kế UI,TODO,1000
Viết API,IN_PROGRESS,2000
```

- Headers use `"Name (TYPE)"` notation.
- BOOLEAN values are `"true"` / `"false"`.
- DATE values are ISO date portion only (`"2026-06-08"`).
- Null/empty cells are empty strings.

#### Markdown format (rare, when `dataFormat: "markdown"`)

```markdown
| Task (TEXT) | Status (TEXT) | Budget (INT) |
| --- | --- | --- |
| Thiết kế UI | TODO | 1000 |
| Viết API | IN_PROGRESS | 2000 |
```

### 4.3 TASK

```json
{
  "type": "TASK",
  "id": "task-uuid-abc-123",
  "title": "Mua sữa",
  "content": {
    "title": "Mua sữa",
    "description": "Mua sữa tươi không đường",
    "status": "TODO",
    "priority": "HIGH",
    "parentId": null,
    "children": [
      { "id": "sub-uuid-1", "title": "Kiểm tra giá", "status": "DONE" }
    ]
  },
  "metadata": {
    "isSubtask": false,
    "subtaskCount": 1
  },
  "source": "user_mention"
}
```

#### `task_context` (separate FormData field)

When the primary context is a TASK, the BFF also sends a **separate** `task_context` JSON string:

```json
{
  "focusedTaskId": "task-uuid-abc-123",
  "focusedTaskTitle": "Mua sữa"
}
```

This tells the AI which specific task is "in focus" (e.g., the one the user clicked on).

#### TASKS overview (when the Tasks tab is open, no specific task selected)

```json
{
  "type": "TASK",
  "id": "tasks-overview",
  "title": "Tasks",
  "metadata": {
    "taskCount": 25,
    "completedCount": 10
  },
  "source": "active_tab"
}
```

### 4.4 CALENDAR

```json
{
  "type": "CALENDAR",
  "id": "calendar-overview",
  "title": "Calendar",
  "content": {
    "dataFormat": "csv",
    "data": "title,startAt,endAt,allDay,color\nHọp team,2026-06-08T09:00:00Z,2026-06-08T10:00:00Z,false,#5645d4\n...",
    "eventCount": 8
  },
  "metadata": {
    "eventCount": 8
  },
  "source": "active_tab"
}
```

Calendar events are sent as CSV.

---

## 5. Complete Request Examples

### Example 1: Voice note editing (transcript mode)

```
POST /api/v1/voice/process
Content-Type: multipart/form-data
x-user-id: user-uuid-123
x-session-id: sess-abc-456

FormData:
  transcript = "thêm ghi chú cần review trước thứ 6 vào cuối"
  packed_context = {"items":[{"type":"NOTE","id":"550e8400-...","title":"Meeting Notes","content":"# Cuộc họp lúc 5h chiều\n\n...","metadata":{"cursorPosition":842,"title":"Meeting Notes"},"source":"active_tab"}],"packedAt":"2026-06-08T12:00:00.000Z","totalItems":1}
  cursor_position = "842"
  user_id = "user-uuid-123"
```

### Example 2: Voice stack row insertion (audio mode)

```
POST /api/v1/voice/process
Content-Type: multipart/form-data
x-user-id: user-uuid-123
x-session-id: sess-def-789

FormData:
  audio = <binary webm blob>
  packed_context = {"items":[{"type":"STACK","id":"660e8400-...","title":"Sprint Backlog","content":{"schema":{"columns":[{"id":"col-1","name":"Task","type":"TEXT","order":0},{"id":"col-2","name":"Status","type":"TEXT","order":1}]},"stats":{"rowCount":5,"columnCount":2},"dataFormat":"csv","data":"Task (TEXT),Status (TEXT)\nBug fix,TODO\n"},"metadata":{"dataFormat":"csv","rowCount":5},"source":"active_tab"}],"packedAt":"2026-06-08T12:00:00.000Z","totalItems":1}
  user_id = "user-uuid-123"
```

### Example 3: Multi-context with @mentions

```
FormData:
  transcript = "so sánh @Sprint Backlog với @Meeting Notes"
  packed_context = {"items":[
    {"type":"STACK","id":"660e8400-...","title":"Sprint Backlog","content":{...},"source":"user_mention"},
    {"type":"NOTE","id":"550e8400-...","title":"Meeting Notes","content":"# Cuộc họp...","source":"user_mention"}
  ],"packedAt":"2026-06-08T12:00:00.000Z","totalItems":2}
  user_id = "user-uuid-123"
```

---

## 6. Response Contract (what NextJS expects back)

FastAPI must return a JSON response matching this TypeScript interface:

```ts
interface VoiceResponse {
  action: string;                      // e.g. "update_note", "add_stack_row", "none"
  updatedData?: unknown | null;        // Present for data mutations
  aiReply?: string | null;             // Present for conversational replies
  error?: string;                      // Present on failure (HTTP 4xx/5xx)
}
```

**`updatedData` and `aiReply` are mutually exclusive** — a response must have one or the other, never both.

### 6.1 Update Note → ghost text

```json
{
  "action": "update_note",
  "success": true,
  "message": "Note update suggested",
  "updatedData": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "diff": {
      "action_type": "append",
      "content_to_insert": "Cần review trước thứ 6",
      "cursor_position": 842,
      "preview_surrounding": "…phiên họp lúc 5h chiều.││"
    }
  },
  "reply": null
}
```

- **`diff.action_type`**: `"append"` (end of note), `"insert"` (at cursor_position), `"replace"` (selection)
- **`diff.content_to_insert`**: Only the text to insert — NOT the full note
- **`diff.cursor_position`**: Character offset in the raw markdown
- NextJS renders this as **ghost text** at the cursor. User presses **Tab** to accept, **Esc** to dismiss.

### 6.2 Add Stack Row → ghost row

```json
{
  "action": "add_stack_row",
  "success": true,
  "message": "Row suggested",
  "updatedData": {
    "id": "temp_row_1717800000000",
    "stackId": "660e8400-e29b-41d4-a716-446655440001",
    "suggestionType": "ghost_row",
    "columnOrder": [
      { "id": "col-uuid-1", "name": "Task", "type": "TEXT" },
      { "id": "col-uuid-2", "name": "Status", "type": "TEXT" }
    ],
    "data": {
      "col-uuid-1": "Marketing Budget",
      "col-uuid-2": "TODO"
    }
  }
}
```

- `data` keys must match column `id` values from the `packed_context` schema.

### 6.3 Bulk Update Stack

```json
{
  "action": "bulk_update_stack",
  "success": true,
  "message": "Update suggested for 2 rows",
  "updatedData": {
    "stackId": "660e8400-e29b-41d4-a716-446655440001",
    "suggestionType": "cell_diff",
    "columnOrder": [
      { "id": "col-uuid-1", "name": "Task", "type": "TEXT" },
      { "id": "col-uuid-2", "name": "Status", "type": "TEXT" }
    ],
    "updates": [
      { "rowId": "row-uuid-1", "data": { "col-uuid-2": "DONE" } },
      { "rowId": "row-uuid-2", "data": { "col-uuid-2": "DONE" } }
    ]
  }
}
```

### 6.4 Update Cell

```json
{
  "action": "update_cell",
  "success": true,
  "message": "Cell edit suggested",
  "updatedData": {
    "stackId": "660e8400-e29b-41d4-a716-446655440001",
    "suggestionType": "cell_diff",
    "rowId": "row-uuid-1",
    "columnId": "col-uuid-2",
    "value": 5000
  }
}
```

### 6.5 Delete Row

```json
{
  "action": "delete_row",
  "success": true,
  "message": "Row deletion suggested",
  "updatedData": {
    "stackId": "660e8400-e29b-41d4-a716-446655440001",
    "suggestionType": "row_delete",
    "rowId": "row-uuid-1"
  }
}
```

### 6.6 Manage Tasks

**Create:**
```json
{
  "action": "manage_tasks",
  "success": true,
  "message": "Task creation suggested",
  "updatedData": {
    "suggestionType": "task_action",
    "action_type": "create",
    "title": "Mua sữa",
    "description": "Mua sữa tươi không đường",
    "priority": "HIGH",
    "dueDate": "2026-06-09T00:00:00Z"
  }
}
```

**Update:**
```json
{
  "action": "manage_tasks",
  "success": true,
  "message": "Task update suggested",
  "updatedData": {
    "suggestionType": "task_action",
    "action_type": "update",
    "task_id": "task-uuid-abc-123",
    "status": "DONE"
  }
}
```

**Delete:**
```json
{
  "action": "manage_tasks",
  "success": true,
  "message": "Task deletion suggested",
  "updatedData": {
    "suggestionType": "task_action",
    "action_type": "delete",
    "task_id": "task-uuid-abc-123"
  }
}
```

### 6.7 Conversational (no mutation)

```json
{
  "action": "none",
  "success": true,
  "message": null,
  "updatedData": null,
  "reply": "Không có gì! Bạn cần tôi giúp gì thêm không?"
}
```

### 6.8 Error

```json
{
  "error": "Command not recognized as a workspace action."
}
```
> HTTP status: **400**

---

## 7. Action → Context Mapping Reference

| User says | Primary Context | Expected `action` |
|-----------|----------------|-------------------|
| "thêm ghi chú ... vào cuối" | NOTE | `update_note` |
| "thêm dòng ..." | STACK | `add_stack_row` |
| "đổi tất cả trạng thái thành done" | STACK | `bulk_update_stack` |
| "đổi ô này thành 5000" | STACK (cell focused) | `update_cell` |
| "xóa dòng này đi" | STACK (row focused) | `delete_row` |
| "tạo task ..." | TASK / any | `manage_tasks` (create) |
| "đánh dấu task ... là done" | TASK | `manage_tasks` (update) |
| "xóa task ..." | TASK | `manage_tasks` (delete) |
| "cảm ơn nhé" | any | `none` |

---

## 8. Quick-Start Test Payload

Use this minimal JSON payload to test your FastAPI endpoint with `curl`:

```bash
curl -X POST http://localhost:8000/api/v1/voice/process \
  -H "x-user-id: test-user-001" \
  -F 'transcript=thêm ghi chú cần review trước thứ 6 vào cuối' \
  -F 'packed_context={"items":[{"type":"NOTE","id":"550e8400-e29b-41d4-a716-446655440000","title":"Meeting Notes","content":"# Cuộc họp lúc 5h chiều\n\n- Điểm 1: Thảo luận ngân sách\n- Điểm 2: Lên kế hoạch Q3\n\nKết luận: Cần họp lại vào thứ 6","metadata":{"cursorPosition":842,"title":"Meeting Notes"},"source":"active_tab"}],"packedAt":"2026-06-08T12:00:00.000Z","totalItems":1}' \
  -F 'cursor_position=842' \
  -F 'user_id=test-user-001'
```

**Expected response shape:**
```json
{
  "action": "update_note",
  "success": true,
  "updatedData": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "diff": {
      "action_type": "append",
      "content_to_insert": "Cần review trước thứ 6",
      "cursor_position": 842,
      "preview_surrounding": "…họp lại vào thứ 6││"
    }
  }
}
```
