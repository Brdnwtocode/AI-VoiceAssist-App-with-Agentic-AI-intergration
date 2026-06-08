"""
Master Orchestrator — replaces the old sentinel layer.

Flow:
  1. Safety Gate (absorbed sentinel) — same UUID delimiter, same 400 on block
  2. @Maximus Detection — explicit user escalation to deliberation
  3. Complexity Router — fast decision: fan-out or single-expert?
  4. Fan-out (if complex) — parallel Contrarian + Research + Conversation
  5. Directive generation — builds structured instructions for the Resolver

The Orchestrator does NOT replace the Resolver. It augments it with
structured expert context, enabling the Resolver to make better decisions.

Design decisions:
- Custom asyncio.gather (not LangGraph) — transparent, ~80 lines, thesis-friendly
- Safety gate stays FIRST inside the orchestrator — do not remove it
- Each expert fails independently — one failure doesn't block the pipeline
- Complexity router uses heuristic shortcuts for obviously simple commands
  before falling back to LLM routing, saving ~300ms on simple commands
"""

from typing import Optional

from .config import logger
from .experts import (
    run_all_experts,
    run_complexity_router,
    run_safety_gate,
)
from .models import (
    ComplexityAssessment,
    DeliberationResult,
    OrchestratorDecision,
    SafetyVerdict,
)


async def run_orchestrator(
    transcript: str,
    context_type: str,
    note_state: Optional[str] = None,
    dynamic_schema: Optional[str] = None,
    task_context_data: Optional[str] = None,
    processed_context: Optional[dict] = None,
) -> OrchestratorDecision:
    """Master Orchestrator — the central decision-making layer.

    Replaces the old `run_sentinel` call in call_nlu_live.
    Returns an OrchestratorDecision that downstream code uses to
    decide how to call the Resolver.

    Args:
        transcript: The transcribed user speech (Vietnamese or English)
        context_type: NOTE, STACK, TASK, TASKS, CALENDAR, or none
        note_state: Serialized note JSON (NOTE context)
        dynamic_schema: Serialized stack schema JSON (STACK context)
        task_context_data: Serialized task JSON (TASK context)
        processed_context: Full packed_context from Context-Grabber

    Returns:
        OrchestratorDecision with safety verdict, complexity, and optional
        deliberation results + synthesis directive for the Resolver.
    """
    # ── Phase 1: Safety Gate (absorbed sentinel) ──────────────────────────
    logger.info(
        "Orchestrator: running safety gate (transcript_len=%d, context=%s)",
        len(transcript), context_type,
    )
    safety = await run_safety_gate(transcript)

    if not safety.safe:
        logger.warning(
            "Orchestrator: SAFETY BLOCK — reason=%s preview=%r",
            safety.reason,
            transcript[:120],
        )
        return OrchestratorDecision(
            should_deliberate=False,
            complexity="blocked",
            safety_verdict=safety,
        )
    logger.info("Orchestrator: safety gate passed")

    # ── Phase 2: Complexity Routing ───────────────────────────────────────
    logger.info("Orchestrator: running complexity router")
    complexity = await run_complexity_router(transcript, context_type)
    logger.info(
        "Orchestrator: complexity=%s reasoning=%s",
        complexity.complexity, complexity.reasoning,
    )

    # ── Phase 3: Simple path — skip experts, go straight to Resolver ──────
    if complexity.complexity == "simple":
        return OrchestratorDecision(
            should_deliberate=False,
            complexity="simple",
            safety_verdict=safety,
            directive="",
        )

    # ── Phase 4: Complex path — fan out to all experts ────────────────────
    logger.info("Orchestrator: fanning out to Contrarian + Research + Conversation")
    deliberation = await run_all_experts(
        transcript=transcript,
        context_type=context_type,
        note_state=note_state,
        dynamic_schema=dynamic_schema,
        task_context_data=task_context_data,
        processed_context=processed_context,
    )

    # ── Phase 5: Build directive for Resolver ─────────────────────────────
    directive = _build_resolver_directive(transcript, context_type, deliberation)
    logger.info(
        "Orchestrator: deliberation complete — synthesis_notes_len=%d",
        len(deliberation.synthesis_notes),
    )

    return OrchestratorDecision(
        should_deliberate=True,
        complexity="complex",
        safety_verdict=safety,
        deliberation=deliberation,
        directive=directive,
    )


def _build_resolver_directive(
    transcript: str,
    context_type: str,
    deliberation: DeliberationResult,
) -> str:
    """Build the structured directive that augments the Resolver's prompt.

    This is the key engineering piece — the directive injects expert findings
    as structured context, not free-form text, making the Resolver's output
    deterministic and testable.
    """
    parts = [
        "─── EXPERT DELIBERATION RESULTS ───",
        "",
        "The following structured analysis was performed by three independent expert agents.",
        "Consider ALL of the following when formulating your response.",
        "",
    ]

    # Contrarian findings
    c = deliberation.contrarian
    if c:
        parts.append(f"CONTRARIAN EXPERT:")
        parts.append(f"  Risk Level: {c.risk.upper()}")
        parts.append(f"  Critique: {c.critique}")
        if c.alternative_action:
            parts.append(f"  Alternative Action to Consider: {c.alternative_action}")
        parts.append("")

    # Research findings
    r = deliberation.research
    if r:
        parts.append(f"RESEARCH EXPERT:")
        parts.append(f"  Workspace Confidence: {r.confidence:.0%}")
        parts.append(f"  Relevant Context Found: {r.relevant_context}")
        if r.data_gaps:
            parts.append(f"  DATA GAPS (information missing from workspace):")
            for gap in r.data_gaps:
                parts.append(f"    - {gap}")
        parts.append("")

    # Conversation findings
    conv = deliberation.conversation
    if conv:
        parts.append(f"CONVERSATION EXPERT:")
        parts.append(f"  Intent: {conv.intent}")
        parts.append(f"  Tone: {conv.tone}")
        parts.append(f"  Language: {conv.language}")
        parts.append(f"  Has Ambiguity: {'YES — consider multiple interpretations' if conv.has_ambiguity else 'No'}")
        parts.append("")

    # Synthesis summary
    parts.append(f"SYNTHESIS NOTES:")
    parts.append(f"  {deliberation.synthesis_notes}")
    parts.append("")
    parts.append("─── END EXPERT DELIBERATION ───")
    parts.append("")
    parts.append("INSTRUCTION: Use the expert findings above to improve your response.")
    parts.append("If the Contrarian flagged HIGH risk, be cautious with destructive actions.")
    parts.append("If the Research expert found DATA GAPS, acknowledge what you don't know.")
    parts.append("If the Conversation expert flagged AMBIGUITY, ask for clarification if needed.")

    return "\n".join(parts)
