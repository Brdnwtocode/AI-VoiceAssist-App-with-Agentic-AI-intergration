# Next.js <-> FastAPI Interface Contract

## 1. Overview & System Architecture
This contract defines the interface between:
- **Next.js Frontend/API**: Sends audio, context, and receives structured commands
- **Python FastAPI Microservice**: Handles Whisper transcription, GPT-4o reasoning, and returns structured action payloads
- **Next.js Voice Flow**: Uses `PushToTalk.tsx` to record audio → sends to `/api/voice/process` → receives action → executes Prisma writes

## 2. Relevant Prisma Schema

```prisma
model Note {
  id     String @id @default(uuid())
  userId String
  user   User   @relation(fields: [userId], references: [id], onDelete: Cascade)

  title   String
  content String @db.Text

  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}

model Stack {
  id     String @id @default(uuid())
  userId String
  user   User   @relation(fields: [userId], references: [id], onDelete: Cascade)

  name    String
  columns StackColumn[]
  rows    StackRow[]

  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}

enum DataType {
  TEXT
  INT
  FLOAT
  BOOLEAN
  DATE
  SELECT
}

model StackColumn {
  id      String   @id @default(uuid())
  stackId String
  stack   Stack    @relation(fields: [stackId], references: [id], onDelete: Cascade)

  name String
  type DataType?
}

model StackRow {
  id      String @id @default(uuid())
  stackId String
  stack   Stack  @relation(fields: [stackId], references: [id], onDelete: Cascade)

  data Json @db.JsonB // Format: { [columnId]: value }
}
```

## 3. Current Next.js Voice Flow (For Context)

### 3.1 PushToTalk Component (Frontend)
- Records audio using MediaRecorder (mimeType: `audio/webm`)
- Keyboard shortcut: Hold **Ctrl + Space** (or click/touch the button)
- Sends FormData to `/api/voice/process` with:
  - `audio`: Audio Blob
  - `contextType`: "NOTE" or "STACK"
  - `contextId`: Note/Stack UUID
  - `cursorPosition`: Cursor position in note (if NOTE context)

### 3.2 Next.js Voice API Route (`/app/api/voice/process/route.ts`)
Current implementation uses:
- OpenAI Whisper-1 for transcription (language: "vi" - Vietnamese)
- GPT-4o-mini for tool calling
- **Tools Available**:
  1. `update_note`: For note content manipulation
  2. `add_stack_row`: For adding rows to Stack tables

**GPT Prompt Used (from current implementation)**:
```
You are the AI engine for a multimodal workspace. The user is dictating commands in Vietnamese or English.
Current context type: ${contextType}.
Current state: ${JSON.stringify(contextData)}.
User's command (transcribed): "${transcript}"

Execute the user's intent by calling the appropriate tool. Do not respond with conversational text.
```

## 4. Next.js POST Request Shape (To FastAPI)

### Endpoint: `POST /api/v1/voice/process`
### Content-Type: `multipart/form-data`

### Form Fields:

| Field Name | Type | Required | Description |
|------------|------|----------|-------------|
| `audio` | File | Yes | Audio file (webm or mp3, max size: 10MB) |
| `contextType` | String | Yes | "NOTE" or "STACK" |
| `contextId` | String | Yes | UUID of the current note/stack |
| `cursorPosition` | String | No (if NOTE) | Current cursor position in note (string representation of integer) |

### dynamic_schema Example (for STACK context, when sending column definitions):
```json
[
  {
    "id": "clx123abc",
    "stackId": "clx456def",
    "name": "Task Name",
    "type": "TEXT"
  },
  {
    "id": "clx789ghi",
    "stackId": "clx456def",
    "name": "Priority",
    "type": "INT"
  },
  {
    "id": "clx012jkl",
    "stackId": "clx456def",
    "name": "Due Date",
    "type": "DATE"
  },
  {
    "id": "clx345mno",
    "stackId": "clx456def",
    "name": "Completed",
    "type": "BOOLEAN"
  }
]
```

## 5. Expected FastAPI Response Shape (The Contract)

All responses must be valid JSON with the following structure:

### Top-Level Response Keys:
| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `transcript` | String | Yes | Full Whisper transcription of the audio |
| `action` | String | Yes | Action name: "update_note", "add_stack_row", "update_stack_row", "delete_stack_row", "none" |
| `updatedData` | Object | No (if action is "none") | The data to be written to Prisma |
| `success` | Boolean | Yes | Indicates if the action was successful |
| `message` | String | No | Optional human-readable message |

---

### Example A: Note Action (update_note)
```json
{
  "transcript": "Thêm ghi chú về cuộc họp hôm nay",
  "action": "update_note",
  "updatedData": {
    "id": "note-uuid-here",
    "title": "My Note Title",
    "content": "Existing content\nMeeting notes from today's discussion...",
    "createdAt": "2026-05-07T00:00:00.000Z",
    "updatedAt": "2026-05-07T05:25:00.000Z"
  },
  "success": true,
  "message": "Note updated"
}
```

### update_note Payload Details (What GPT Should Return):
The FastAPI service should return the complete Note object after update, but if you need to generate the action parameters, use:
```json
{
  "content_to_insert": "string of markdown content",
  "action_type": "append | insert_at_cursor | replace_all"
}
```

---

### Example B: Stack Action (Add Row)
```json
{
  "transcript": "Thêm nhiệm vụ mua đồ ăn",
  "action": "add_stack_row",
  "updatedData": {
    "id": "temp_row_1234567890",
    "stackId": "stack-uuid-here",
    "data": {
      "clx123abc": "Buy groceries",
      "clx789ghi": 2,
      "clx012jkl": "2026-05-10",
      "clx345mno": false
    }
  },
  "success": true,
  "message": "Row added"
}
```

### add_stack_row Payload Details:
GPT should return **column names mapped to values**, and the service maps them to column IDs:
```json
{
  "data": {
    "Task Name": "Buy groceries",
    "Priority": 2,
    "Due Date": "2026-05-10",
    "Completed": false
  }
}
```

---

### Example C: Stack Action (Update Row)
```json
{
  "transcript": "Cập nhật nhiệm vụ mua đồ ăn",
  "action": "update_stack_row",
  "updatedData": {
    "id": "row-uuid-here",
    "stackId": "stack-uuid-here",
    "data": {
      "clx123abc": "Buy groceries and clean the house",
      "clx789ghi": 1
    }
  },
  "success": true,
  "message": "Row updated"
}
```

---

### Example D: Stack Action (Delete Row)
```json
{
  "transcript": "Xóa nhiệm vụ mua đồ ăn",
  "action": "delete_stack_row",
  "updatedData": null,
  "success": true,
  "message": "Row deleted"
}
```

---

### Example E: No Action Recognized
```json
{
  "transcript": "Chào bạn",
  "action": "none",
  "updatedData": null,
  "success": true,
  "message": "No action recognized from command"
}
```

## 6. Error Handling

FastAPI should return appropriate HTTP status codes and JSON error payloads:

### Error Response Shape:
```json
{
  "error": "Human-readable error message"
}
```

### Status Codes:
- **400**: Bad Request (invalid audio, missing fields, too large)
- **401**: Unauthorized (user not authenticated)
- **404**: Not Found (note/stack not found or doesn't belong to user)
- **500**: Internal Server Error (transcription or reasoning failed)
