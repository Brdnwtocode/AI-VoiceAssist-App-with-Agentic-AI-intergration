"""
LangGraph Builder — constructs the orchestration StateGraph.

Graph topology:

    START
      │
      ▼
  safety_gate ──(blocked)──► END (error)
      │ (safe)
      ▼
  complexity_router ──(simple)──► resolver
      │ (complex)                    │
      ▼                              ▼
  ┌─── Send(contrarian) ──┐     reflection ◄──┐
  ├─── Send(research)   ──┤        │            │
  ├─── Send(conversation)─┤        ├(pass)──► END
  └─── Send(planner)    ──┘        │(refine)
      │    │    │    │             ▼
      └────┼────┼────┘         resolver (retry, max 3)
           ▼
       synthesizer
           │
           ▼
       resolver ──► reflection (critique → refine loop)

Key LangGraph features:
- StateGraph with AgentState TypedDict
- Conditional edges for routing & reflection loop
- Send API for parallel execution (4 experts + planner)
- Node-level error handling
- Messages[] audit trail
"""

from typing import Any, Dict, List, Literal

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from .config import logger
from .graph_nodes import (
    complexity_router_node,
    contrarian_expert_node,
    conversation_expert_node,
    planner_node,
    reflection_node,
    research_expert_node,
    resolver_node,
    safety_gate_node,
    synthesizer_node,
)
from .graph_state import AgentState, create_initial_state
from .memory import get_memory_manager


# ═══════════════════════════════════════════════════════════════════════════
# Conditional Edge Functions
# ═══════════════════════════════════════════════════════════════════════════

def route_after_safety(state: AgentState) -> Literal["complexity_router", END]:
    """After safety gate: continue to router if safe, else END."""
    if state.get("is_blocked", False):
        logger.warning("[Graph] Safety blocked — ending pipeline")
        return END
    logger.info("[Graph] Safety passed → routing to complexity_router")
    return "complexity_router"


def route_after_router(state: AgentState) -> List[Send] | Literal["resolver"]:
    """After complexity router: fan-out to experts+planner (complex) or go to resolver (simple).

    Uses LangGraph Send API for parallel fan-out:
    - Returns list[Send] when complex → parallel expert + planner execution
    - Returns "resolver" string when simple → skip experts
    """
    if state.get("should_deliberate", False):
        targets = state.get("fan_out_targets", ["contrarian", "research", "conversation"])
        logger.info("[Graph] Complex → fanning out to %d experts + planner", len(targets))

        sends: List[Send] = []
        for target in targets:
            if target == "contrarian":
                sends.append(Send("contrarian_expert", state))
            elif target == "research":
                sends.append(Send("research_expert", state))
            elif target == "conversation":
                sends.append(Send("conversation_expert", state))
            elif target == "planner":
                sends.append(Send("planner", state))

        # Always include planner when deliberating
        sends.append(Send("planner", state))

        return sends

    logger.info("[Graph] Simple → skipping experts, direct to resolver")
    return "resolver"


def route_after_reflection(state: AgentState) -> Literal["resolver", END]:
    """After reflection: loop back to resolver if refinement needed, else END.

    Implements the Reflexion pattern with bounded retries.
    """
    reflection = state.get("reflection_output")
    refinement_count = state.get("refinement_count", 0)
    max_refinements = state.get("max_refinements", 3)

    if reflection and reflection.needs_refinement and refinement_count < max_refinements:
        logger.info(
            "[Graph] Reflection: needs refinement (score=%.2f, iteration=%d/%d)",
            reflection.score, refinement_count + 1, max_refinements,
        )
        return "resolver"

    logger.info(
        "[Graph] Reflection: accepted (score=%.2f, iteration=%d)",
        reflection.score if reflection else 0, refinement_count,
    )
    return END


# ═══════════════════════════════════════════════════════════════════════════
# Graph Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_orchestrator_graph() -> StateGraph:
    """Build and compile the LangGraph orchestration state machine.

    Returns a compiled StateGraph ready for ainvoke() or astream().

    Usage:
        graph = build_orchestrator_graph()
        initial_state = create_initial_state(transcript=..., context_type=..., ...)
        result = await graph.ainvoke(initial_state)
        nlu_output = result["nlu_result"]
    """
    # ── Create graph with AgentState ──
    builder = StateGraph(AgentState)

    # ── Add nodes ──
    builder.add_node("safety_gate", safety_gate_node)
    builder.add_node("complexity_router", complexity_router_node)
    builder.add_node("contrarian_expert", contrarian_expert_node)
    builder.add_node("research_expert", research_expert_node)
    builder.add_node("conversation_expert", conversation_expert_node)
    builder.add_node("planner", planner_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("resolver", resolver_node)
    builder.add_node("reflection", reflection_node)

    # ── Add edges ──

    # Entry point
    builder.add_edge(START, "safety_gate")

    # Safety gate → conditional: continue or block
    builder.add_conditional_edges(
        "safety_gate",
        route_after_safety,
        {
            "complexity_router": "complexity_router",
            END: END,
        },
    )

    # Complexity router → conditional: simple (resolver) or complex (Send fan-out)
    builder.add_conditional_edges(
        "complexity_router",
        route_after_router,
        {
            "resolver": "resolver",
        },
    )

    # Expert + Planner nodes → synthesizer (fan-in after parallel execution)
    builder.add_edge("contrarian_expert", "synthesizer")
    builder.add_edge("research_expert", "synthesizer")
    builder.add_edge("conversation_expert", "synthesizer")
    builder.add_edge("planner", "synthesizer")

    # Synthesizer → resolver
    builder.add_edge("synthesizer", "resolver")

    # Resolver → reflection (reflexion loop entry)
    builder.add_edge("resolver", "reflection")

    # Reflection → conditional: loop back to resolver or END
    builder.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "resolver": "resolver",
            END: END,
        },
    )

    # ── Compile with MemorySaver for checkpointing ──
    checkpointer = MemorySaver()
    logger.info("LangGraph orchestrator compiled with MemorySaver checkpointing")
    return builder.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Runner
# ═══════════════════════════════════════════════════════════════════════════

# Singleton compiled graph (lazy-init)
_graph: StateGraph | None = None


def _get_graph() -> StateGraph:
    """Get or create the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_orchestrator_graph()
    return _graph


async def run_graph(
    transcript: str,
    context_type: str,
    context_id: str,
    note_state: str | None = None,
    dynamic_schema: str | None = None,
    task_context_data: str | None = None,
    processed_context: dict | None = None,
    cursor_position: int = 0,
    session_id: str = "",
    user_id: str = "default",
) -> dict:
    """Run the full LangGraph orchestration pipeline and return the NLU result.

    This is the main entry point — replaces the old call_nlu_live().

    Now with memory integration:
    - Loads conversation history + user profile before graph invocation
    - Injects memory context into the resolver node's prompt
    - Saves the completed exchange to short-term + long-term memory
    - Uses MemorySaver for graph-level checkpointing (pause/resume/replay)

    Args:
        transcript: Transcribed user speech
        context_type: NOTE, STACK, TASK, TASKS, CALENDAR
        context_id: UUID of the primary context item
        note_state: Serialized note JSON
        dynamic_schema: Serialized stack schema JSON
        task_context_data: Serialized task JSON
        processed_context: Full packed_context from Context-Grabber
        cursor_position: Cursor position in note
        session_id: Session identifier for memory scoping

    Returns:
        NLU result dict: {"action": str, "params": dict, "reply": str|None}
    """
    import time as _time
    import uuid as _uuid
    t0 = _time.perf_counter()

    # ── Start pipeline trace for Live Viewer ──
    from .replay import store
    pipeline_request_id = str(_uuid.uuid4())
    store.start_pipeline_trace(pipeline_request_id, transcript, context_type)

    # ── Load memory context (scoped to this user + session) ──
    memory = get_memory_manager(session_id=session_id, user_id=user_id)
    memory_context = await memory.get_context_for_prompt(transcript, context_type)

    # ── Build initial state ──
    initial_state = create_initial_state(
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
    )

    logger.info(
        "[Graph] Invoking pipeline: context=%s transcript_len=%d session=%s memory=%d",
        context_type, len(transcript), session_id[:12] if session_id else "none",
        len(memory_context),
    )

    # ── Run graph with checkpointing ──
    config = {"configurable": {"thread_id": session_id or "default"}}
    graph = _get_graph()
    final_state = await graph.ainvoke(initial_state, config)

    # ── Check for safety block ──
    if final_state.get("is_blocked", False):
        logger.warning("[Graph] Pipeline blocked by safety gate")
        store.finish_pipeline_trace(pipeline_request_id, "blocked")
        return {"action": None, "params": {}, "reply": None, "_pipeline_id": pipeline_request_id}

    # ── Extract NLU result ──
    nlu_result = final_state.get("nlu_result")
    error = final_state.get("error")

    if error and not nlu_result:
        logger.error("[Graph] Pipeline error: %s", error)
        store.finish_pipeline_trace(pipeline_request_id, "error")
        return {"action": "none", "params": {}, "reply": None, "_pipeline_id": pipeline_request_id}

    if nlu_result is None:
        logger.error("[Graph] No NLU result produced")
        store.finish_pipeline_trace(pipeline_request_id, "error")
        return {"action": "none", "params": {}, "reply": None, "_pipeline_id": pipeline_request_id}

    # ── Save to memory (short-term + long-term) ──
    elapsed = (_time.perf_counter() - t0) * 1000
    try:
        # Extract language from conversation expert if available
        conv = final_state.get("conversation_output")
        language = conv.language if conv else "vi"
        complexity = final_state.get("complexity_assessment")
        complexity_str = complexity.complexity if complexity else "simple"

        await memory.record_exchange(
            transcript=transcript,
            context_type=context_type,
            action=nlu_result.get("action", "none"),
            reply=nlu_result.get("reply"),
            language=language,
            complexity=complexity_str,
            duration_ms=elapsed,
            success=error is None,
        )
        logger.info("[Graph] Exchange saved to memory (%.0f ms)", elapsed)
    except Exception as mem_exc:
        logger.warning("[Graph] Failed to save to memory: %s", mem_exc)

    logger.info(
        "[Graph] Pipeline complete: action=%s messages=%d elapsed=%.0fms",
        nlu_result.get("action", "unknown"),
        len(final_state.get("messages", [])),
        elapsed,
    )

    # ── Finalize pipeline trace ──
    store.finish_pipeline_trace(pipeline_request_id, "completed")

    # Attach pipeline ID to result for LiveEntry linking
    nlu_result["_pipeline_id"] = pipeline_request_id
    return nlu_result
