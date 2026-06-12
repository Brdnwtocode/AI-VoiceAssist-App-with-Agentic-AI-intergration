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
        "create_task",
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


# ── Multi-Expert Orchestration Models ─────────────────────────────────────

class SafetyVerdict(BaseModel):
    """Output from the safety gate (absorbed sentinel)."""
    safe: bool
    reason: str = ""


class ComplexityAssessment(BaseModel):
    """Router decision: should the orchestrator fan out to all experts?"""
    complexity: Literal["simple", "complex"]
    reasoning: str = ""


class ContrarianOutput(BaseModel):
    """Contrarian expert: challenges assumptions, flags risks, breaks sycophancy."""
    critique: str = Field(..., description="What the primary interpretation might be missing")
    risk: Literal["low", "medium", "high"] = Field(..., description="Risk level of acting on the primary interpretation")
    alternative_action: Optional[str] = Field(default=None, description="Alternative action to consider, if any")


class ResearchOutput(BaseModel):
    """Research expert: grounds the command against workspace state."""
    relevant_context: str = Field(default="", description="Key facts from workspace state relevant to this command")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence that workspace state has been fully considered")
    data_gaps: List[str] = Field(default_factory=list, description="Information gaps that prevent confident resolution")
    research_findings: str = Field(default="", description="Synthesized research content (from web search) that downstream actions can use verbatim")
    sources: List[str] = Field(default_factory=list, description="Source URLs backing the research findings")


class ConversationOutput(BaseModel):
    """Conversation expert: extracts intent, tone, and language cues."""
    intent: str = Field(..., description="The user's underlying intent in natural language")
    tone: Literal["command", "query", "chitchat"] = Field(default="command")
    language: Literal["vi", "en", "mixed"] = Field(default="vi")
    has_ambiguity: bool = Field(default=False, description="Whether the command has multiple plausible interpretations")


class DeliberationResult(BaseModel):
    """Aggregated output from all experts after fan-out/fan-in."""
    contrarian: Optional[ContrarianOutput] = None
    research: Optional[ResearchOutput] = None
    conversation: Optional[ConversationOutput] = None
    synthesis_notes: str = Field(default="", description="Key points for the Resolver to consider")


class OrchestratorDecision(BaseModel):
    """Final decision from the Master Orchestrator."""
    should_deliberate: bool = Field(..., description="Whether multi-expert deliberation was performed")
    complexity: str = Field(default="simple")
    safety_verdict: SafetyVerdict = Field(default_factory=lambda: SafetyVerdict(safe=True, reason=""))
    deliberation: Optional[DeliberationResult] = None
    directive: str = Field(default="", description="Instructions passed to the Resolver for final synthesis")


# ── Expert Intermediate Reasoning Models ──────────────────────────────────

class ContrarianReasoning(BaseModel):
    """Intermediate reasoning trace from the Contrarian expert's structured template."""
    deconstructed_command: str = Field(default="", description="How the command was parsed")
    edge_cases_considered: List[str] = Field(default_factory=list, description="Edge cases evaluated")
    sycophancy_risks: List[str] = Field(default_factory=list, description="Sycophancy patterns detected")
    risk_assessment: str = Field(default="", description="Risk level rationale")


class ResearchReasoning(BaseModel):
    """Intermediate reasoning trace from the Research expert's grounding template."""
    workspace_inventory: str = Field(default="", description="What data was found in workspace")
    entity_mapping: Dict[str, str] = Field(default_factory=dict, description="User entities → workspace data mapping")
    web_search_performed: bool = Field(default=False)
    web_search_query: Optional[str] = Field(default=None)
    web_results_count: int = Field(default=0)
    confidence_rationale: str = Field(default="", description="Why the confidence score was assigned")


class ConversationReasoning(BaseModel):
    """Intermediate reasoning trace from the Conversation expert's analysis."""
    lang_detection: Dict[str, Any] = Field(default_factory=dict)
    tone_classification: Dict[str, Any] = Field(default_factory=dict)
    stt_corrections_applied: List[str] = Field(default_factory=list)
    ambiguities_detected: List[str] = Field(default_factory=list)
    intent_rationale: str = Field(default="", description="How the intent was derived")


# ── Planning Node Models ──────────────────────────────────────────────────

class PlanStep(BaseModel):
    """A single step in the execution plan."""
    step: int = Field(..., description="Step number (1-based)")
    action: str = Field(..., description="The action to take: update_note, add_stack_row, manage_tasks, etc.")
    description: str = Field(..., description="What this step accomplishes in natural language")
    params_hint: Dict[str, Any] = Field(default_factory=dict, description="Hint for what params to include")
    depends_on: List[int] = Field(default_factory=list, description="Step numbers this step depends on (empty for independent steps)")
    context_required: str = Field(default="", description="What context data this step needs")


class ExecutionPlan(BaseModel):
    """Structured execution plan produced by the Planning Node."""
    overall_goal: str = Field(..., description="The user's overall goal in one clear sentence")
    reasoning: str = Field(default="", description="Why this plan structure was chosen")
    steps: List[PlanStep] = Field(default_factory=list, description="Ordered execution steps")
    is_multi_step: bool = Field(default=False, description="Whether this is truly multi-step or could be handled in one action")
    fallback_action: str = Field(default="none", description="What to do if planning fails or is unnecessary")


# ── Reflection Pattern Models ─────────────────────────────────────────────

class ReflectionOutput(BaseModel):
    """Output from the Reflection Node — critiques the Resolver's output."""
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Quality score: 1.0 = perfect, 0.0 = unusable")
    issues: List[str] = Field(default_factory=list, description="Specific problems found in the resolver output")
    suggestions: List[str] = Field(default_factory=list, description="Concrete suggestions for improvement")
    needs_refinement: bool = Field(default=False, description="Whether the resolver should retry with these suggestions")
    critique_summary: str = Field(default="", description="One-sentence summary of the reflection")


class ReflectionReasoning(BaseModel):
    """Trace of the reflection node's decision process."""
    action_valid: bool = Field(default=True)
    params_complete: bool = Field(default=True)
    context_respected: bool = Field(default=True)
    reply_appropriate: bool = Field(default=True)
    hallutination_detected: bool = Field(default=False)
    iteration: int = Field(default=0, description="Which refinement iteration this is (0 = first pass)")
    threshold_met: bool = Field(default=True)
