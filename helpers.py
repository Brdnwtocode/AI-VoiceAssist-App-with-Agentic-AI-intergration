import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Type
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError, create_model

from config import logger
from models import ColumnDef


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


def validate_note_context(note_state: Optional[str], context_id: str) -> dict:
    if note_state:
        try:
            note_data = json.loads(note_state)
            required_keys = {"id", "userId", "title", "content", "createdAt", "updatedAt"}
            if not required_keys.issubset(note_data.keys()):
                raise ValueError("Missing fields in note_state")
            return note_data
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid note_state: {exc}") from exc
    return {
        "id": context_id,
        "userId": "",
        "title": "",
        "content": "",
        "createdAt": "",
        "updatedAt": "",
    }


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
