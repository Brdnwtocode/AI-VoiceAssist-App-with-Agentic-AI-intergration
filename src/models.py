from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UpdateNoteParams(BaseModel):
    content_to_insert: str = Field(..., description="The markdown text to insert.")
    action_type: Literal["append", "insert_at_cursor"] = Field(
        ...,
        description="append: add to end; insert_at_cursor: insert at cursor",
    )


class NoActionParams(BaseModel):
    reply: Optional[str] = Field(default=None, description="Conversational reply.")


class ColumnDef(BaseModel):
    id: str = Field(...)
    name: str = Field(...)
    type: str = Field(...)

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


class SummarizeContextParams(BaseModel):
    summary: str = Field(..., description="The summary or synthesis of the context materials.")


class UpdateCellParams(BaseModel):
    """Precision edit: update a single focused cell in a STACK."""
    stack_id: str = Field(..., description="The ID of the stack to update.")
    row_id: str = Field(..., description="The row ID of the focused cell.")
    column_id: str = Field(..., description="The column ID of the focused cell.")
    value: Any = Field(..., description="The new value for the cell.")


class DeleteRowParams(BaseModel):
    """Delete a row from a STACK."""
    stack_id: str = Field(..., description="The ID of the stack.")
    row_id: str = Field(..., description="The row ID to delete.")


class AddRowParams(BaseModel):
    """Add a new row to a STACK (column-id keyed data)."""
    stack_id: str = Field(..., description="The ID of the stack.")
    data: Dict[str, Any] = Field(..., description="Column ID → value mapping for the new row.")


class BulkUpdateStackParams(BaseModel):
    stack_id: str = Field(..., description="The ID of the stack to update.")
    updates: List[Dict[str, Any]] = Field(
        ...,
        description="List of updates, where each update contains 'row_id' and 'column_values' mapping column names to values."
    )


class ManageTasksParams(BaseModel):
    action_type: Literal["create", "update", "delete"] = Field(..., description="The action to perform on tasks.")
    task_id: Optional[str] = Field(default=None, description="The ID of the task to update or delete.")
    title: Optional[str] = Field(default=None, description="Task title.")
    description: Optional[str] = Field(default=None, description="Task description.")
    status: Optional[Literal["TODO", "IN_PROGRESS", "DONE"]] = Field(default=None)
    priority: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = Field(default=None)
    assignee: Optional[str] = Field(default=None, description="Free text assignee.")
    dueDate: Optional[str] = Field(default=None, description="ISO 8601 datetime string.")
    parentId: Optional[str] = Field(default=None, description="UUID of parent task (for subtasks).")


class ResolverLLMOutput(BaseModel):
    action: Literal[
        "update_note",
        "add_stack_row",
        "bulk_update_stack",
        "manage_tasks",
        "summarize_context",
        "create_calendar_event",
        "update_cell",
        "delete_row",
        "none",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)
    reply: Optional[str] = None

    @field_validator("params", mode="before")
    @classmethod
    def _params_object(cls, v: Any) -> Dict[str, Any]:
        return v if isinstance(v, dict) else {}

    model_config = ConfigDict(extra="ignore")
