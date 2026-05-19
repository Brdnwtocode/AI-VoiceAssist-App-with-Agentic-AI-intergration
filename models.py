from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UpdateNoteParams(BaseModel):
    content_to_insert: str = Field(..., description="The markdown text to insert.")
    action_type: Literal["append", "insert_at_cursor"] = Field(
        ...,
        description="append: add to end; insert_at_cursor: insert at cursor",
    )


class NoActionParams(BaseModel):
    pass


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


class ResolverLLMOutput(BaseModel):
    action: Literal["update_note", "add_stack_row", "create_task", "create_calendar_event", "none"]
    params: Dict[str, Any] = Field(default_factory=dict)
    reply: Optional[str] = None

    @field_validator("params", mode="before")
    @classmethod
    def _params_object(cls, v: Any) -> Dict[str, Any]:
        return v if isinstance(v, dict) else {}

    model_config = ConfigDict(extra="ignore")
