"""
LangGraph Agent State — typed dictionary flowing through the orchestration graph.

Uses TypedDict with Annotated reducers for proper fan-in semantics:
when multiple parallel expert nodes write to the same state, the reducer
controls how values merge (replace vs. append).

Design:
- All fields are Optional with sensible defaults
- The `messages` field uses operator.add reducer (append-only) for audit trail
- Expert outputs use simple replacement (last writer wins — but since experts
  write to different keys, no conflicts occur in practice)
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from .models import (
    ComplexityAssessment,
    ContrarianOutput,
    ContrarianReasoning,
    ConversationOutput,
    ConversationReasoning,
    DeliberationResult,
    ExecutionPlan,
    ReflectionOutput,
    ReflectionReasoning,
    ResearchOutput,
    ResearchReasoning,
    SafetyVerdict,
)


class AgentState(TypedDict, total=False):
    """Complete state flowing through the LangGraph orchestration graph.

    Fields are grouped by pipeline phase:
    - Input: set before graph invocation
    - Safety: written by safety_gate node
    - Routing: written by complexity_router node
    - Expert outputs: written by parallel expert nodes
    - Synthesis: written by synthesizer node
    - Output: written by resolver node
    """

    # ── Input fields (set by caller before ainvoke) ──
    transcript: str
    context_type: str
    context_id: str
    note_state: Optional[str]
    dynamic_schema: Optional[str]
    task_context_data: Optional[str]
    processed_context: Optional[dict]
    cursor_position: int
    session_id: str  # For memory scoping
    user_id: str     # Authenticated user ID for profile isolation

    # ── Memory (injected at graph entry, used by resolver) ──
    memory_context: str  # Conversation history + user profile + similar interactions

    # ── Phase 1: Safety Gate ──
    safety_verdict: SafetyVerdict

    # ── Phase 2: Complexity Router ──
    complexity_assessment: ComplexityAssessment
    should_deliberate: bool
    fan_out_targets: List[str]  # Which experts to invoke: ["contrarian", "research", "conversation"]

    # ── Phase 3: Expert Outputs (written by parallel Send nodes) ──
    contrarian_output: Optional[ContrarianOutput]
    contrarian_reasoning: Optional[ContrarianReasoning]
    research_output: Optional[ResearchOutput]
    research_reasoning: Optional[ResearchReasoning]
    conversation_output: Optional[ConversationOutput]
    conversation_reasoning: Optional[ConversationReasoning]

    # ── Phase 3b: Planning Node (parallel with experts) ──
    execution_plan: Optional[ExecutionPlan]

    # ── Phase 4: Synthesis ──
    deliberation_result: Optional[DeliberationResult]
    orchestrator_directive: str

    # ── Phase 5: Resolver Output ──
    nlu_result: Optional[dict]
    nlu_raw_response: str  # Raw LLM response for debugging

    # ── Phase 6: Reflection Pattern (critique → refine loop) ──
    reflection_output: Optional[ReflectionOutput]
    reflection_reasoning: Optional[ReflectionReasoning]
    refinement_count: int  # How many times resolver has been retried (0 = first pass)
    max_refinements: int   # Cap at 3 to bound latency

    # ── Flow control ──
    error: Optional[str]
    is_blocked: bool

    # ── Pipeline tracing (for Live Viewer) ──
    pipeline_request_id: str

    # ── Audit trail (messages appended by each node) ──
    messages: Annotated[List[Dict[str, Any]], operator.add]


def create_initial_state(
    transcript: str,
    context_type: str,
    context_id: str,
    note_state: Optional[str] = None,
    dynamic_schema: Optional[str] = None,
    task_context_data: Optional[str] = None,
    processed_context: Optional[dict] = None,
    cursor_position: int = 0,
    session_id: str = "",
    memory_context: str = "",
    user_id: str = "default",
    pipeline_request_id: str = "",
) -> AgentState:
    """Factory for a clean initial state before graph invocation.

    All optional fields are set to sensible defaults so nodes don't
    need to check for key existence.
    """
    return AgentState(
        # Input
        transcript=transcript,
        context_type=context_type,
        context_id=context_id,
        note_state=note_state,
        dynamic_schema=dynamic_schema,
        task_context_data=task_context_data,
        processed_context=processed_context,
        cursor_position=cursor_position,
        session_id=session_id,
        memory_context=memory_context,
        user_id=user_id,
        pipeline_request_id=pipeline_request_id,
        # Safety (populated by node)
        safety_verdict=SafetyVerdict(safe=True, reason=""),
        # Routing
        complexity_assessment=ComplexityAssessment(complexity="simple", reasoning=""),
        should_deliberate=False,
        fan_out_targets=[],
        # Expert outputs
        contrarian_output=None,
        contrarian_reasoning=None,
        research_output=None,
        research_reasoning=None,
        conversation_output=None,
        conversation_reasoning=None,
        # Planning
        execution_plan=None,
        # Synthesis
        deliberation_result=None,
        orchestrator_directive="",
        # Resolver
        nlu_result=None,
        nlu_raw_response="",
        # Reflection
        reflection_output=None,
        reflection_reasoning=None,
        refinement_count=0,
        max_refinements=3,
        # Flow
        error=None,
        is_blocked=False,
        # Audit
        messages=[],
    )
