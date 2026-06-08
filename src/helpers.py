import csv
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Type
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError, create_model

from .config import logger
from .models import ColumnDef


def normalize_audio_mime(content_type: Optional[str]) -> Optional[str]:
    """Strip codec parameters (e.g. audio/webm;codecs=opus -> audio/webm)."""
    if not content_type:
        return None
    base = content_type.split(";")[0].strip().lower()
    if base == "audio/mpeg":
        return "audio/mp3"
    return base


def _http_error_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        try:
            return json.dumps(detail)
        except Exception:
            return str(detail)
    return str(detail)


def data_type_to_optional(dtype: str) -> Any:
    mapping: Dict[str, Any] = {
        "TEXT": Optional[str],
        "INT": Optional[int],
        "FLOAT": Optional[float],
        "BOOLEAN": Optional[bool],
        "DATE": Optional[str],
        "SELECT": Optional[str],
    }
    return mapping.get(dtype.upper(), Optional[str])


def fix_llm_output(raw_json: dict) -> dict:
    """
    Post-process LLM JSON output to fix common mistakes before Pydantic validation.
    
    Fixes:
    - Invalid action_type values (e.g., "insert_at_top" -> "insert_at_cursor")
    - Missing required fields
    - Common typos in field names
    """
    if not isinstance(raw_json, dict):
        return raw_json
    
    action = raw_json.get("action")
    params = raw_json.get("params", {})
    
    # Fix update_note action_type
    if action == "update_note" and isinstance(params, dict):
        action_type = params.get("action_type", "")
        
        # Map invalid values to valid ones
        action_type_mapping = {
            "insert_at_top": "insert_at_cursor",
            "insert_top": "insert_at_cursor",
            "top": "insert_at_cursor",
            "insert_at_beginning": "insert_at_cursor",
            "beginning": "insert_at_cursor",
            "insert": "insert_at_cursor",
            "prepend": "insert_at_cursor",
            # Keep valid values
            "append": "append",
            "insert_at_cursor": "insert_at_cursor",
        }
        
        if action_type in action_type_mapping:
            fixed_type = action_type_mapping[action_type]
            if fixed_type != action_type:
                logger.warning(
                    "[fix_llm_output] Fixed action_type: '%s' -> '%s'",
                    action_type, fixed_type
                )
                params["action_type"] = fixed_type
                raw_json["params"] = params
        else:
            # Default to "append" if completely invalid
            logger.warning(
                "[fix_llm_output] Unknown action_type: '%s', defaulting to 'append'",
                action_type
            )
            params["action_type"] = "append"
            raw_json["params"] = params
    
    return raw_json


@lru_cache(maxsize=512)
def get_dynamic_model(schema_str: str) -> Type[BaseModel]:
    """LRU-cached dynamic Pydantic model for STACK row params (schema_str = JSON column array)."""
    columns = json.loads(schema_str)
    fields: Dict[str, Any] = {}
    for col in columns:
        name = col["name"]
        col_type_str = str(col["type"]).upper()
        py = data_type_to_optional(col_type_str)
        fields[name] = (py, Field(default=None))
    key = hashlib.sha256(schema_str.encode("utf-8")).hexdigest()[:16]
    return create_model(f"StackRowDyn_{key}", **fields)


def build_add_row_model(columns: List[Dict[str, Any]]) -> Type[BaseModel]:
    return get_dynamic_model(json.dumps(columns))


def clean_json_output(raw_output: str) -> dict:
    s = (raw_output or "").strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        s = s[first_nl + 1 :] if first_nl != -1 else s[3:]
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]
    
    parsed = json.loads(s)
    
    # Fix common LLM output mistakes before returning
    if isinstance(parsed, dict):
        parsed = fix_llm_output(parsed)
    
    return parsed


def build_column_mapping(columns: List[ColumnDef]) -> Dict[str, str]:
    return {col.name: col.id for col in columns}


# ── Context-Grabber Interface: Mode Detection & Data Parsing ──────────────

def detect_context_mode(packed_context: dict) -> str:
    """Detect the context mode from packed_context metadata.

    Returns one of: 'precision', 'full_data', 'schema_only', 'unknown'
    """
    items = packed_context.get("items", [])
    if not items:
        return "unknown"

    metadata = items[0].get("metadata", {})
    edit_mode = metadata.get("editMode", "")

    if edit_mode == "single_cell":
        return "precision"
    elif edit_mode == "full_data":
        return "full_data"
    elif edit_mode:
        return "schema_only"
    else:
        # Fallback: auto-detect from content shape
        content = items[0].get("content", {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                content = {}
        if content.get("focusedTarget"):
            return "precision"
        if content.get("data") and content.get("dataFormat"):
            return "full_data"
        if content.get("schema"):
            return "schema_only"
        return "unknown"


def parse_csv_data(csv_string: str) -> List[Dict[str, str]]:
    """Parse a CSV string into a list of dicts (column name → value)."""
    reader = csv.DictReader(io.StringIO(csv_string))
    return [dict(row) for row in reader]


def parse_markdown_table(md_string: str) -> List[Dict[str, str]]:
    """Parse a Markdown table string into a list of dicts."""
    lines = [line for line in md_string.strip().split("\n") if line.startswith("|")]
    if len(lines) < 3:  # header, separator, at least 1 data row
        return []

    # Remove separator line (e.g. |---|---|)
    data_lines = [lines[0]] + lines[2:]

    # Extract headers from first line
    header_cells = [cell.strip() for cell in data_lines[0].split("|")[1:-1]]

    rows = []
    for line in data_lines[1:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) == len(header_cells):
            rows.append(dict(zip(header_cells, cells)))

    return rows


def extract_stack_schema_from_item(item: dict) -> List[Dict[str, Any]]:
    """Extract STACK columns from a packed_context item's content.

    Handles the Context-Grabber v2 structure: content.schema.columns
    Falls back to content.columns for backward compatibility.
    """
    content = item.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return []

    # v2 path: content.schema.columns (Context-Grabber interface)
    schema = content.get("schema", {})
    if isinstance(schema, dict) and "columns" in schema:
        return schema["columns"]

    # v1 fallback: content.columns (legacy)
    if "columns" in content:
        return content["columns"]

    return []


def extract_focused_target(item: dict) -> Optional[Dict[str, Any]]:
    """Extract the focusedTarget from a packed_context item's content."""
    content = item.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
    return content.get("focusedTarget")


def extract_content_data(item: dict) -> Tuple[Optional[str], Optional[str]]:
    """Extract (dataFormat, data) from a packed_context item's content.
    Returns (data_format, data_string) or (None, None).
    """
    content = item.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None, None
    data_format = content.get("dataFormat")
    data = content.get("data")
    return data_format, data


def extract_content_stats(item: dict) -> Dict[str, Any]:
    """Extract stats (rowCount, columnCount) from a packed_context item's content."""
    content = item.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {}
    return content.get("stats", {})


def _default_note_state(context_id: str) -> dict:
    return {
        "id": context_id,
        "userId": "",
        "title": "",
        "content": "",
        "createdAt": "",
        "updatedAt": "",
    }


def validate_note_context(note_state: Optional[str], context_id: str) -> dict:
    """Parse note_state JSON or return a stub when absent/empty (never HTTP 400)."""
    default = _default_note_state(context_id)
    if note_state is None:
        return default

    stripped = note_state.strip()
    if not stripped or stripped.lower() in ("null", "undefined"):
        return default

    try:
        note_data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        logger.warning(
            "note_state JSON invalid (%s), using default stub — preview=%r",
            exc,
            stripped[:120],
        )
        return default

    if not isinstance(note_data, dict):
        logger.warning("note_state is not a JSON object, using default stub")
        return default

    required_keys = {"id", "userId", "title", "content", "createdAt", "updatedAt"}
    if not required_keys.issubset(note_data.keys()):
        logger.warning(
            "note_state missing required fields %s, using default stub",
            required_keys - note_data.keys(),
        )
        return default

    return note_data


def validate_stack_context(dynamic_schema: Optional[str]) -> List[ColumnDef]:
    if not dynamic_schema:
        raise HTTPException(status_code=400, detail="dynamic_schema required for STACK context")
    try:
        schema_list = json.loads(dynamic_schema)
        if not isinstance(schema_list, list):
            raise ValueError("Must be an array")
        return [ColumnDef.model_validate(col) for col in schema_list]
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid dynamic_schema: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid dynamic_schema: {exc}") from exc


def validate_task_context(task_context: Optional[str] = None) -> dict:
    if not task_context:
        return {}
    try:
        data = json.loads(task_context)
        if not isinstance(data, dict):
            raise ValueError("Must be a JSON object")
        return data
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid task_context JSON") from None
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid task_context: {exc}") from exc


def validate_calendar_context() -> dict:
    return {}


def build_note_payload(
    nlu_result: dict,
    note_data: dict,
    cursor_position: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Build a surgical diff payload for note updates.

    Instead of returning the ENTIRE note content (old behavior), returns only
    the proposed insertion + surrounding context. NextJS renders this as an
    inline suggestion (ghost text) at the cursor position — like VS Code's
    inline completions. The user accepts (Tab) or dismisses (Esc).

    Response shape:
    {
      "id": "note-uuid",
      "diff": {
        "action_type": "append" | "insert_at_cursor",
        "content_to_insert": "the new text",
        "cursor_position": 42,
        "preview_surrounding": "…context before││context after…"
      }
    }
    """
    import re as _re

    params = nlu_result["params"]
    content_to_insert = params["content_to_insert"]
    action_type = params["action_type"]
    current_content = note_data.get("content", "")

    def _clean_preview(text: str) -> str:
        """Sanitize markdown/HTML for a clean single-line preview snippet."""
        # Strip HTML tags like <br />, <div>, etc.
        text = _re.sub(r'<[^>]+>', '', text)
        # Collapse whitespace (newlines, tabs, multiple spaces) into single spaces
        text = _re.sub(r'\s+', ' ', text)
        return text.strip()

    # Build preview: 40 chars before + marker + 40 chars after insertion point
    preview_before = ""
    preview_after = ""
    insert_pos = cursor_position

    if action_type == "append":
        insert_pos = len(current_content)
        raw_before = current_content[-60:] if len(current_content) > 60 else current_content
        preview_before = _clean_preview(raw_before)
        preview_after = ""
    else:
        pos = max(0, min(int(cursor_position), len(current_content)))
        insert_pos = pos
        raw_before = current_content[max(0, pos - 40):pos]
        raw_after = current_content[pos:pos + 40]
        preview_before = _clean_preview(raw_before)
        preview_after = _clean_preview(raw_after)

    # Join with cursor marker — preview_before + "││" + preview_after
    preview = (preview_before + "││" + preview_after).strip()

    # If the cleaned preview is empty (e.g., only whitespace/HTML in original),
    # provide a minimal fallback so the frontend still has context.
    if not preview_before and not preview_after:
        preview = "││"

    updated_data = {
        "id": note_data["id"],
        "diff": {
            "action_type": action_type,
            "content_to_insert": content_to_insert,
            "cursor_position": insert_pos,
            "preview_surrounding": preview[:200],
        },
    }
    return updated_data, "Note update suggested", None


def build_stack_payload(
    nlu_result: dict,
    columns: List[ColumnDef],
    context_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Build a ghost-row suggestion payload for stack inserts.

    Returns ONLY the proposed row data + column ordering metadata.
    NextJS renders this as a ghost/faded row at the bottom of the table.
    The user accepts (click) or dismisses.

    Column values are mapped from names → UUIDs. Column ordering follows
    the schema strictly so NextJS can align cells without guessing.
    """
    params = nlu_result["params"]
    col_mapping = build_column_mapping(columns)

    # Build ordered data: follow column schema order strictly
    ordered_columns: List[Dict[str, str]] = []
    data: Dict[str, Any] = {}
    for col in columns:
        ordered_columns.append({"id": col.id, "name": col.name, "type": col.type})
        if col.name in params and params[col.name] is not None:
            data[col.id] = params[col.name]

    row_id = f"temp_row_{int(time.time() * 1000)}"
    updated_data = {
        "id": row_id,
        "stackId": context_id,
        "suggestionType": "ghost_row",
        "columnOrder": ordered_columns,  # Schema order — NextJS aligns cells by this
        "data": data,                     # { [columnUuid]: value }
    }
    return updated_data, "Row suggested", None


def build_task_payload(
    nlu_result: dict,
    task_context_data: dict,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Build task creation suggestion payload (create_task action)."""
    updated_data = dict(nlu_result["params"])
    if task_context_data and not updated_data.get("parentId"):
        focused_id = task_context_data.get("focusedTaskId")
        if focused_id:
            updated_data["parentId"] = focused_id
    updated_data["suggestionType"] = "task_action"
    return updated_data, "Task creation suggested", None


def build_bulk_update_stack_payload(
    nlu_result: dict,
    columns: List[ColumnDef],
    context_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Build a surgical bulk-update payload for stack rows.

    Each update targets a specific row + specific columns. Column ordering
    metadata is included so NextJS can render diffs cell-by-cell.
    """
    params = nlu_result["params"]
    col_mapping = build_column_mapping(columns)

    # Column order reference for NextJS
    ordered_columns: List[Dict[str, str]] = [
        {"id": col.id, "name": col.name, "type": col.type} for col in columns
    ]

    mapped_updates = []
    for item in params.get("updates", []):
        row_id = item.get("row_id")
        col_values = item.get("column_values", {})
        mapped_data = {}
        for col_name, val in col_values.items():
            if val is None:
                continue
            col_id = col_mapping.get(col_name)
            if col_id:
                mapped_data[col_id] = val
            else:
                logger.warning("Column name '%s' not in schema, ignoring", col_name)
        mapped_updates.append({"rowId": row_id, "data": mapped_data})

    updated_data = {
        "stackId": context_id,
        "suggestionType": "cell_diff",
        "columnOrder": ordered_columns,
        "updates": mapped_updates,
    }
    return updated_data, f"Update suggested for {len(mapped_updates)} rows", None


def build_update_cell_payload(
    nlu_result: dict,
    context_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Build payload for single-cell precision edit (update_cell action).

    Surgical: targets exactly one cell by rowId + columnId.
    NextJS highlights the cell with the proposed new value inline.
    """
    params = nlu_result["params"]
    row_id = params.get("row_id", "")
    column_id = params.get("column_id", "")
    value = params.get("value")

    updated_data = {
        "stackId": context_id,
        "suggestionType": "cell_diff",
        "rowId": row_id,
        "columnId": column_id,
        "value": value,
    }
    return updated_data, "Cell edit suggested", None


def build_delete_row_payload(
    nlu_result: dict,
    context_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Build payload for row deletion suggestion (delete_row action).

    NextJS highlights the row in red/with a strikethrough until confirmed.
    """
    params = nlu_result["params"]
    row_id = params.get("row_id", "")

    updated_data = {
        "stackId": context_id,
        "suggestionType": "row_delete",
        "rowId": row_id,
    }
    return updated_data, "Row deletion suggested", None


def build_manage_tasks_payload(
    nlu_result: dict,
    task_context_data: dict,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    updated_data = dict(nlu_result["params"])
    action_type = updated_data.get("action_type")

    if action_type == "create":
        if task_context_data and not updated_data.get("parentId"):
            focused_id = task_context_data.get("focusedTaskId")
            if focused_id:
                updated_data["parentId"] = focused_id
        msg = "Task creation suggested"
    elif action_type == "update":
        msg = "Task update suggested"
    elif action_type == "delete":
        msg = "Task deletion suggested"
    else:
        msg = "Task action suggested"

    updated_data["suggestionType"] = "task_action"
    return updated_data, msg, None


def build_summarize_context_payload(
    nlu_result: dict,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    summary = nlu_result["params"].get("summary")
    return None, "Context summarized", summary


def build_calendar_payload(
    nlu_result: dict,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    data = dict(nlu_result["params"])
    data["suggestionType"] = "calendar_event"
    return data, "Calendar event suggested", None


def build_none_payload(
    conv_reply: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    if conv_reply:
        return None, None, conv_reply
    return None, "No action recognized from command", None


def utc_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context_uuid(context_id: str) -> None:
    try:
        UUID(context_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid context_id UUID")


def process_context(packed_context: dict) -> dict:
    """
    Process context without DB access.
    - Truncate large content safely (parse JSON, truncate fields, re-serialize)
    - Validate structure
    - Add processing metadata
    """
    MAX_CONTENT_LENGTH = 2000
    processed_items = []
    warnings = []
    for item in packed_context.get("items", []):
        # Validate required fields
        if not item.get("type") or not item.get("id"):
            warnings.append(f"Missing required fields (type or id) in item: {item}")
            continue

        # Safely truncate content: if it's JSON, parse → truncate data field → re-serialize
        raw_content = item.get("content")
        if raw_content is not None:
            if isinstance(raw_content, str) and len(raw_content) > MAX_CONTENT_LENGTH:
                try:
                    parsed = json.loads(raw_content)
                    if isinstance(parsed, dict):
                        # Truncate the 'data' field if present (CSV/Markdown payload)
                        if "data" in parsed and isinstance(parsed["data"], str) and len(parsed["data"]) > MAX_CONTENT_LENGTH:
                            parsed["data"] = parsed["data"][:MAX_CONTENT_LENGTH] + "\n...[truncated]"
                            parsed["_truncated"] = True
                        item["content"] = json.dumps(parsed, ensure_ascii=False)
                    else:
                        item["content"] = raw_content[:MAX_CONTENT_LENGTH] + "...[truncated]"
                except (json.JSONDecodeError, TypeError):
                    item["content"] = raw_content[:MAX_CONTENT_LENGTH] + "...[truncated]"
                item["metadata"] = item.get("metadata", {})
                item["metadata"]["truncated"] = True
            elif isinstance(raw_content, dict):
                # Already a dict — truncate data field if too long
                if "data" in raw_content and isinstance(raw_content["data"], str) and len(raw_content["data"]) > MAX_CONTENT_LENGTH:
                    raw_content["data"] = raw_content["data"][:MAX_CONTENT_LENGTH] + "\n...[truncated]"
                    raw_content["_truncated"] = True
                    item["metadata"] = item.get("metadata", {})
                    item["metadata"]["truncated"] = True

        # Add processing metadata
        item["metadata"] = item.get("metadata", {})
        item["metadata"]["processed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        processed_items.append(item)

    return {
        "items": processed_items,
        "packedAt": packed_context.get("packedAt"),
        "totalItems": len(processed_items),
        "warnings": warnings,
    }

