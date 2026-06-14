# Agentic Automate — FastAPI Contract

> `POST /api/v1/records/automate`  
> FastAPI is stateless. No DB access. Process only what you receive.

---

## What FastAPI Receives

**Multipart form fields** from the BFF:

| Field | Type | Notes |
|-------|------|-------|
| `audio` | File (binary) | Raw audio blob (webm/opus). May be absent if transcript is provided. |
| `transcript` | string | Pre-transcribed text. May be empty if audio is provided. |
| `recording_id` | string | DB UUID. May be empty for unsaved recordings. |
| `user_id` | string | Authenticated user UUID. |
| `mode` | string | Always `"automate"`. Distinguishes from voice-command pipeline. |
| `action` | string | One of: `full_automate`, `summarize`, `extract_tasks`, `populate_stack`, `identify_speakers`, `create_calendar`. Hints which output to prioritize. |

**Headers**: `x-session-id`, `x-user-id`

---

## What FastAPI Must Return

**JSON** — snake_case keys. Every field below must be present in the response.

```json
{
  "note_mutation": null,
  "task_mutations": [],
  "stack_mutation": null,
  "calendar_mutation": null,
  "speaker_labels": null,
  "summary": ""
}
```

### Field specifications

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `note_mutation` | object or null | yes | A suggested note. `null` if not applicable. |
| `note_mutation.title` | string | if present | Note title. |
| `note_mutation.content` | string | if present | Markdown body. |
| `note_mutation.folder_id` | string or null | if present | Target folder UUID, or null. |
| `task_mutations` | array | yes | Suggested tasks. Empty array `[]` if none. |
| `task_mutations[].title` | string | yes | Task title. |
| `task_mutations[].description` | string or null | no | Optional details. |
| `task_mutations[].status` | string | no | `TODO`, `IN_PROGRESS`, or `DONE`. Default `TODO`. |
| `task_mutations[].priority` | string | no | `LOW`, `MEDIUM`, or `HIGH`. Default `MEDIUM`. |
| `task_mutations[].assignee` | string or null | no | Person responsible. |
| `task_mutations[].due_date` | string (ISO8601) or null | no | Deadline. |
| `stack_mutation` | object or null | yes | A suggested stack (table). `null` if not applicable. |
| `stack_mutation.stack_id` | string or null | no | Existing stack UUID to populate, or null for new. |
| `stack_mutation.stack_name` | string | if present | Stack name. |
| `stack_mutation.columns` | array | if present | Column definitions. |
| `stack_mutation.columns[].name` | string | yes | Column header. |
| `stack_mutation.columns[].type` | string | yes | `TEXT`, `INT`, `FLOAT`, `BOOLEAN`, `DATE`, or `SELECT`. |
| `stack_mutation.rows` | array | if present | Row data as `{"ColumnName":"value"}` objects. |
| `calendar_mutation` | object or null | yes | A suggested calendar event. `null` if not applicable. |
| `calendar_mutation.title` | string | if present | Event title. |
| `calendar_mutation.notes` | string or null | no | Optional details. |
| `calendar_mutation.start_at` | string (ISO8601) | if present | Start time. |
| `calendar_mutation.end_at` | string (ISO8601) | if present | End time. |
| `calendar_mutation.all_day` | boolean | no | Default `false`. |
| `speaker_labels` | array or null | yes | Speaker diarization. `null` if not applicable. |
| `speaker_labels[].speaker` | string | yes | Speaker identifier. |
| `speaker_labels[].segments` | array | yes | Timed segments for this speaker. |
| `speaker_labels[].segments[].start` | float | yes | Start time in seconds. |
| `speaker_labels[].segments[].end` | float | yes | End time in seconds. |
| `speaker_labels[].segments[].text` | string | yes | Transcribed text for this segment. |
| `summary` | string | yes | 1–3 sentence recap. Empty string `""` if nothing to summarize. |

---

## Rules

1. **Every top-level field is mandatory** in the JSON response. Use `null` for inapplicable objects, `[]` for empty arrays, `""` for empty strings.
2. **Do not auto-commit anything.** The BFF stores the response as-is on the Recording row, then the frontend asks the user to approve each mutation one by one.
3. **Return HTTP 422** on malformed input, **HTTP 500** on internal failure.
