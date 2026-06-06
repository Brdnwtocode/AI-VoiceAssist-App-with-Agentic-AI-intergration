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
    return json.loads(s)


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
    params = nlu_result["params"]
    content_to_insert = params["content_to_insert"]
    action_type = params["action_type"]
    current_content = note_data["content"]
    if action_type == "append":
        new_content = current_content + "\n" + content_to_insert
    else:
        pos = max(0, min(int(cursor_position), len(current_content)))
        new_content = current_content[:pos] + content_to_insert + current_content[pos:]
    updated_data = {
        "id": note_data["id"],
        "title": note_data["title"],
        "content": new_content,
        "createdAt": note_data["createdAt"],
        "updatedAt": utc_iso_z(),
    }
    return updated_data, "Note updated", None


def build_stack_payload(
    nlu_result: dict,
    columns: List[ColumnDef],
    context_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    params = nlu_result["params"]
    col_mapping = build_column_mapping(columns)
    data: Dict[str, Any] = {}
    for col_name, value in params.items():
        if value is None:
            continue
        col_id = col_mapping.get(col_name)
        if col_id:
            data[col_id] = value
        else:
            logger.warning("Column name '%s' not in schema, ignoring", col_name)
    row_id = f"temp_row_{int(time.time() * 1000)}"
    updated_data = {
        "id": row_id,
        "stackId": context_id,
        "data": data,
    }
    return updated_data, "Row added", None


def build_task_payload(
    nlu_result: dict,
    task_context_data: dict,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    updated_data = dict(nlu_result["params"])
    if task_context_data and not updated_data.get("parentId"):
        focused_id = task_context_data.get("focusedTaskId")
        if focused_id:
            updated_data["parentId"] = focused_id
    return updated_data, "Task created", None


def build_bulk_update_stack_payload(
    nlu_result: dict,
    columns: List[ColumnDef],
    context_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    params = nlu_result["params"]
    col_mapping = build_column_mapping(columns)
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
        "updates": mapped_updates,
    }
    return updated_data, f"Updated {len(mapped_updates)} rows", None


def build_update_cell_payload(
    nlu_result: dict,
    context_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Build payload for single-cell precision edit (update_cell action)."""
    params = nlu_result["params"]
    row_id = params.get("row_id", "")
    column_id = params.get("column_id", "")
    value = params.get("value")

    updated_data = {
        "stackId": context_id,
        "rowId": row_id,
        "columnId": column_id,
        "value": value,
    }
    return updated_data, "Cell updated", None


def build_delete_row_payload(
    nlu_result: dict,
    context_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Build payload for row deletion (delete_row action)."""
    params = nlu_result["params"]
    row_id = params.get("row_id", "")

    updated_data = {
        "stackId": context_id,
        "rowId": row_id,
    }
    return updated_data, "Row deleted", None


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
        msg = "Task created"
    elif action_type == "update":
        msg = "Task updated"
    elif action_type == "delete":
        msg = "Task deleted"
    else:
        msg = "Task action processed"
        
    return updated_data, msg, None


def build_summarize_context_payload(
    nlu_result: dict,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    summary = nlu_result["params"].get("summary")
    return None, "Context summarized", summary


def build_calendar_payload(
    nlu_result: dict,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    return dict(nlu_result["params"]), "Calendar event created", None


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

