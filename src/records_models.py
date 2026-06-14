"""
Pydantic models for the POST /api/v1/records/automate contract.

Every top-level response field is mandatory per the contract:
- Objects use `null` when not applicable
- Arrays use `[]` when empty
- Strings use `""` when empty
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Request Validation ─────────────────────────────────────────────────────

VALID_ACTIONS = {
    "full_automate", "summarize", "extract_tasks",
    "populate_stack", "identify_speakers", "create_calendar",
}


# ── Response Models ────────────────────────────────────────────────────────

class NoteMutation(BaseModel):
    """A suggested note from the recording."""
    title: str = Field(..., description="Note title")
    content: str = Field(..., description="Markdown body of the note")
    folder_id: Optional[str] = Field(default=None, description="Target folder UUID, or null")


class TaskMutation(BaseModel):
    """A single suggested task extracted from the recording."""
    title: str = Field(..., description="Task title")
    description: Optional[str] = Field(default=None, description="Optional task details")
    status: Literal["TODO", "IN_PROGRESS", "DONE"] = Field(default="TODO")
    priority: Literal["LOW", "MEDIUM", "HIGH"] = Field(default="MEDIUM")
    assignee: Optional[str] = Field(default=None, description="Person responsible")
    due_date: Optional[str] = Field(default=None, description="ISO 8601 deadline, or null")


class StackColumn(BaseModel):
    """A column definition within a stack mutation."""
    name: str = Field(..., description="Column header")
    type: Literal["TEXT", "INT", "FLOAT", "BOOLEAN", "DATE", "SELECT"] = Field(
        ..., description="Column data type"
    )


class StackMutation(BaseModel):
    """A suggested stack (table) extracted from the recording."""
    stack_id: Optional[str] = Field(default=None, description="Existing stack UUID, or null for new")
    stack_name: str = Field(..., description="Stack name/title")
    columns: List[StackColumn] = Field(default_factory=list, description="Column definitions")
    rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Row data as {ColumnName: value} objects",
    )


class CalendarMutation(BaseModel):
    """A suggested calendar event from the recording."""
    title: str = Field(..., description="Event title")
    notes: Optional[str] = Field(default=None, description="Optional event details")
    start_at: str = Field(..., description="ISO 8601 start time")
    end_at: str = Field(..., description="ISO 8601 end time")
    all_day: bool = Field(default=False)


class SpeakerSegment(BaseModel):
    """A timed transcript segment for one speaker."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")


class SpeakerLabel(BaseModel):
    """Speaker diarization result for one speaker."""
    speaker: str = Field(..., description="Speaker identifier (e.g. 'Speaker 1')")
    segments: List[SpeakerSegment] = Field(
        default_factory=list,
        description="Timed segments for this speaker",
    )


class AutomateResponse(BaseModel):
    """Complete response for POST /api/v1/records/automate.

    Every top-level field MUST be present. Use null/[]/\"\" for inapplicable values.
    """
    note_mutation: Optional[NoteMutation] = Field(
        default=None,
        description="A suggested note. null if not applicable.",
    )
    task_mutations: List[TaskMutation] = Field(
        default_factory=list,
        description="Suggested tasks. Empty array [] if none.",
    )
    stack_mutation: Optional[StackMutation] = Field(
        default=None,
        description="A suggested stack. null if not applicable.",
    )
    calendar_mutation: Optional[CalendarMutation] = Field(
        default=None,
        description="A suggested calendar event. null if not applicable.",
    )
    speaker_labels: Optional[List[SpeakerLabel]] = Field(
        default=None,
        description="Speaker diarization. null if not applicable.",
    )
    summary: str = Field(
        default="",
        description="1-3 sentence recap. Empty string \"\" if nothing to summarize.",
    )
