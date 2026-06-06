# FastAPI Context Grabber Interface - Implementation Guide

**Date**: June 6, 2026  
**Status**: Implementation Ready  
**For**: FastAPI Microservice Team

---

## 🎯 Overview

The frontend now sends **dual-mode context** for efficient AI processing:
- **Mode 1 (Precision Edit)**: Schema + focused cell only (~270 tokens)
- **Mode 2 (Full Data)**: Schema + CSV/Markdown data (~800 tokens)

**Your job**: Update FastAPI to parse both modes and use the data correctly.

---

## 📦 New Context Structure

### What FastAPI Receives:

```json
{
  "packed_context": "{\"items\":[{\"type\":\"STACK\",\"id\":\"...\",\"title\":\"...\",\"content\":{...},\"metadata\":{...}}],\"packedAt\":\"...\",\"totalItems\":1}",
  "contextType": "STACK",
  "contextId": "107c8c18-...",
  "cursorPosition": "0"
}
```

### Parsed `packed_context` Structure:

#### **Mode 1: Precision Edit (90% of commands)**
```json
{
  "type": "STACK",
  "id": "107c8c18-9ce0-4e15-ad77-1ffae69934eb",
  "title": "asd",
  "source": "active_tab",
  "content": {
    "schema": {
      "columns": [
        {
          "id": "2e74e82c-a87c-4a63-846c-9a37c5b3c79a",
          "name": "Name",
          "type": "TEXT"
        },
        {
          "id": "6e2ffef7-8fde-404d-aeb8-fe73b3c7ef09",
          "name": "Revenue",
          "type": "INT"
        }
      ]
    },
    "stats": {
      "rowCount": 14,
      "columnCount": 8
    },
    "focusedTarget": {
      "rowId": "row_7",
      "columnId": "6e2ffef7-8fde-404d-aeb8-fe73b3c7ef09",
      "currentValue": 50000,
      "rowIndex": 6,
      "columnIndex": 1
    }
  },
  "metadata": {
    "commandType": "single_edit",
    "editMode": "single_cell"
  }
}
```

#### **Mode 2: Full Data (Summarization/Insights)**
```json
{
  "type": "STACK",
  "id": "107c8c18-9ce0-4e15-ad77-1ffae69934eb",
  "title": "asd",
  "source": "active_tab",
  "content": {
    "schema": {
      "columns": [...]
    },
    "stats": {
      "rowCount": 14,
      "columnCount": 8
    },
    "dataFormat": "csv",
    "data": "id,Name,Revenue\nrow_1,Acme,50000\nrow_2,Globex,85000\n..."
  },
  "metadata": {
    "commandType": "summarize",
    "editMode": "full_data",
    "dataFormat": "csv"
  }
}
```

---

## 🔍 How to Detect Context Mode

### Step 1: Parse `packed_context`
```python
import json

def parse_packed_context(request: Request) -> dict:
    packed_context_str = request.form.get("packed_context")
    if packed_context_str:
        return json.loads(packed_context_str)
    return {}
```

### Step 2: Check `editMode` in metadata
```python
def detect_context_mode(packed_context: dict) -> str:
    """Returns 'precision' or 'full_data'"""
    items = packed_context.get("items", [])
    if not items:
        return "unknown"
    
    metadata = items[0].get("metadata", {})
    edit_mode = metadata.get("editMode", "")
    
    if edit_mode == "single_cell":
        return "precision"
    elif edit_mode == "full_data":
        return "full_data"
    else:
        return "schema_only"
```

---

## 📊 How to Parse Data Formats

### CSV Format (Most Common)
```python
import csv
import io

def parse_csv_data(csv_string: str) -> list[dict]:
    """Parse CSV string into list of dicts"""
    reader = csv.DictReader(io.StringIO(csv_string))
    return [row for row in reader]
```

**Example usage:**
```python
# In your voice process endpoint
packed_context = parse_packed_context(request)

for item in packed_context.get("items", []):
    if item["type"] == "STACK":
        content = item["content"]
        
        # Check if full data is included
        if "data" in content and content.get("dataFormat") == "csv":
            rows = parse_csv_data(content["data"])
            # Now you have: [{"id": "row_1", "Name": "Acme", "Revenue": "50000"}, ...]
```

### Markdown Table Format
```python
import pandas as pd
import io

def parse_markdown_table(md_string: str) -> list[dict]:
    """Parse Markdown table into list of dicts"""
    lines = [line for line in md_string.strip().split("\n") if line.startswith("|")]
    if len(lines) < 3:  # Need header, separator, at least 1 data row
        return []
    
    # Remove separator line (---|---|)
    lines = [lines[0]] + lines[2:]
    
    # Convert to CSV-like format
    csv_lines = []
    for line in lines:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        csv_lines.append(",".join([f'"{cell}"' if "," in cell else cell for cell in cells]))
    
    csv_string = "\n".join(csv_lines)
    df = pd.read_csv(io.StringIO(csv_string))
    return df.to_dict("records")
```

---

## 🎯 How to Use Focused Target (Precision Mode)

### When `editMode == "single_cell"`:
```python
def handle_precision_edit(packed_context: dict, transcript: str) -> dict:
    """Handle single cell edit commands"""
    item = packed_context["items"][0]
    content = item["content"]
    
    # Extract focused target
    focused = content.get("focusedTarget", {})
    row_id = focused.get("rowId")
    column_id = focused.get("columnId")
    current_value = focused.get("currentValue")
    
    # Build LLM prompt with precision context
    prompt = f"""
User command: "{transcript}"

Context (Precision Edit Mode):
- Stack: {item['title']} (ID: {item['id']})
- Focused Cell: rowId={row_id}, columnId={column_id}
- Current Value: {current_value}

Schema:
{json.dumps(content['schema'], indent=2)}

Instructions:
1. The user wants to edit the focused cell
2. Return the NEW value for this specific cell
3. Use the rowId and columnId from above

Response format:
{{
  "action": "update_cell",
  "stackId": "{item['id']}",
  "rowId": "{row_id}",
  "columnId": "{column_id}",
  "value": <new value>
}}
"""
    
    # Call LLM
    response = call_llm(prompt)
    return json.loads(response)
```

---

## 📝 Updated LLM Prompt Template

### For Precision Edit (Mode 1):
```
You are editing a stack table. The user has a specific cell focused.

Stack: {title} (ID: {stack_id})
Columns: {columns}
Total Rows: {row_count}

Focused Cell:
- Row ID: {row_id}
- Column ID: {column_id}
- Current Value: {current_value}

User command: "{transcript}"

Instructions:
1. The user wants to edit the FOCUSED CELL only
2. Return the new value for this cell
3. Use the rowId and columnId from "Focused Cell" section
4. Do NOT try to edit other cells

Response format (JSON):
{{
  "action": "update_cell",
  "stackId": "{stack_id}",
  "rowId": "{row_id}",
  "columnId": "{column_id}",
  "value": <new value>
}}
```

### For Full Data (Mode 2):
```
You are analyzing a stack table. The user wants insights or summary.

Stack: {title} (ID: {stack_id})
Columns: {columns}
Total Rows: {row_count}

Data ({data_format} format):
{data}

User command: "{transcript}"

Instructions:
1. Parse the {data_format} data above
2. Fulfill the user's request (summarize, find insights, etc.)
3. If you need to update data, return the specific rowId and columnId

Response format (JSON):
For summary:
{{
  "action": "summary",
  "content": "<summary text>"
}}

For bulk update:
{{
  "action": "bulk_update",
  "stackId": "{stack_id}",
  "updates": [
    {{"rowId": "...", "columnId": "...", "value": ...}},
    ...
  ]
}}
```

---

## 🔄 Response Format Expectations

### For Precision Edit (Single Cell):
```json
{
  "action": "update_cell",
  "stackId": "107c8c18-...",
  "rowId": "row_7",
  "columnId": "6e2ffef7-...",
  "value": 75000
}
```

### For Add Row:
```json
{
  "action": "add_row",
  "stackId": "107c8c18-...",
  "data": {
    "2e74e82c-...": "New Company",
    "6e2ffef7-...": 100000
  }
}
```

### For Delete Row:
```json
{
  "action": "delete_row",
  "stackId": "107c8c18-...",
  "rowId": "row_7"
}
```

### For Summary:
```json
{
  "action": "summary",
  "content": "This stack contains 14 companies with total revenue of $1.2M. The top 3 revenue companies are..."
}
```

### For Bulk Update:
```json
{
  "action": "bulk_update",
  "stackId": "107c8c18-...",
  "updates": [
    {"rowId": "row_1", "columnId": "6e2ffef7-...", "value": 55000},
    {"rowId": "row_2", "columnId": "6e2ffef7-...", "value": 93500}
  ]
}
```

---

## ⚙️ Implementation Steps

### Step 1: Update `/api/v1/voice/process` Endpoint
```python
@app.post("/api/v1/voice/process")
async def process_voice_command(request: Request):
    # Parse form data
    transcript = request.form.get("transcript", "")
    packed_context_str = request.form.get("packed_context", "{}")
    context_type = request.form.get("contextType", "")
    context_id = request.form.get("contextId", "")
    
    # Parse packed context
    try:
        packed_context = json.loads(packed_context_str)
    except:
        packed_context = {}
    
    # Detect context mode
    context_mode = detect_context_mode(packed_context)
    
    # Build LLM prompt based on mode
    if context_mode == "precision":
        prompt = build_precision_prompt(packed_context, transcript)
    elif context_mode == "full_data":
        prompt = build_full_data_prompt(packed_context, transcript)
    else:
        prompt = build_schema_only_prompt(packed_context, transcript)
    
    # Call LLM
    llm_response = call_llm(prompt)
    
    # Parse LLM response
    response = json.loads(llm_response)
    
    return {
        "action": response.get("action"),
        "updatedData": response.get("updatedData") or response,
        "aiReply": response.get("content") if response.get("action") == "summary" else None
    }
```

### Step 2: Implement Helper Functions
```python
def build_precision_prompt(packed_context: dict, transcript: str) -> str:
    """Build prompt for precision edit mode"""
    item = packed_context["items"][0]
    content = item["content"]
    focused = content.get("focusedTarget", {})
    
    return f"""
User command: "{transcript}"

Context (Precision Edit):
- Stack: {item['title']} (ID: {item['id']})
- Focused Cell: rowId={focused.get('rowId')}, columnId={focused.get('columnId')}
- Current Value: {focused.get('currentValue')}

Schema:
{json.dumps(content['schema'], indent=2)}

Instructions: Return JSON with action="update_cell" and the new value for the focused cell.
"""

def build_full_data_prompt(packed_context: dict, transcript: str) -> str:
    """Build prompt for full data mode"""
    item = packed_context["items"][0]
    content = item["content"]
    
    data_format = content.get("dataFormat", "csv")
    data = content.get("data", "")
    
    return f"""
User command: "{transcript}"

Context (Full Data):
- Stack: {item['title']} (ID: {item['id']})
- Data Format: {data_format}

Schema:
{json.dumps(content['schema'], indent=2)}

Data:
{data}

Instructions: Analyze the data and fulfill the user's request.
"""
```

---

## 🧪 Testing Cases

### Test 1: Precision Edit
**Input:**
```
transcript: "Update this cell to 75000"
packed_context: (Mode 1 - precision edit)
```

**Expected LLM Response:**
```json
{
  "action": "update_cell",
  "stackId": "107c8c18-...",
  "rowId": "row_7",
  "columnId": "6e2ffef7-...",
  "value": 75000
}
```

### Test 2: Summarization
**Input:**
```
transcript: "Summarize this stack"
packed_context: (Mode 2 - full data as CSV)
```

**Expected LLM Response:**
```json
{
  "action": "summary",
  "content": "This stack contains 14 companies with total revenue of $1.2M. The top performer is Globex with $85K revenue..."
}
```

### Test 3: Add Row
**Input:**
```
transcript: "Add a new company called Acme with revenue 100000"
packed_context: (Mode 1 - schema only)
```

**Expected LLM Response:**
```json
{
  "action": "add_row",
  "stackId": "107c8c18-...",
  "data": {
    "2e74e82c-...": "Acme",
    "6e2ffef7-...": 100000
  }
}
```

---

## 📋 Checklist for FastAPI Team

### ✅ **Must Implement:**
1. [ ] Parse `packed_context` from form data
2. [ ] Detect context mode (`editMode` in metadata)
3. [ ] Parse CSV/Markdown data formats
4. [ ] Use `focusedTarget` for precision edits
5. [ ] Update LLM prompt templates
6. [ ] Handle all response formats

### ✅ **Nice to Have:**
1. [ ] Token estimation for context
2. [ ] Automatic mode detection (if frontend fails to set `editMode`)
3. [ ] Fallback to JSON parsing if CSV/Markdown fails
4. [ ] Logging for context mode usage

---

## 📞 Contact

If you have questions about the frontend implementation, check:
- `lib/context/packer.ts` - Main packer logic
- `lib/context/commandDetector.ts` - Command type detection
- `lib/context/dataFormatter.ts` - CSV/Markdown formatting

---

**End of FastAPI Interface Document**
