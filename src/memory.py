"""
Memory System — Short-Term & Long-Term Memory for the Agentic Pipeline.

Short-Term Memory (ConversationBuffer):
  - Sliding window of last N exchanges (user command + assistant response)
  - Injected into resolver prompts for follow-up awareness
  - Session-scoped, cleared on session end
  - In-memory only (Python deque)

Long-Term Memory:
  - UserProfile: persisted preferences → Neon PostgreSQL (user_profiles table)
  - InteractionStore: durable command log → Neon PostgreSQL (interactions table)
  - Falls back to local JSON files if DATABASE_URL is not configured

Integration with LangGraph:
  - ConversationBuffer feeds into AgentState before graph invocation
  - After resolver completes, the exchange is saved to buffer + PostgreSQL
  - MemorySaver provides graph-level checkpointing (pause/resume/replay)
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import DB_ENABLED, PROJECT_ROOT, logger

# ── Local JSON fallback paths ──
_MEMORY_DIR = PROJECT_ROOT / ".memory"
_MEMORY_DIR.mkdir(exist_ok=True)
PROFILE_PATH = _MEMORY_DIR / "user_profile.json"
INTERACTIONS_PATH = _MEMORY_DIR / "interactions.jsonl"


# ═══════════════════════════════════════════════════════════════════════════
# Short-Term Memory: Conversation Buffer (unchanged — in-memory only)
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationTurn:
    """A single turn in the conversation (one user command + one assistant response)."""
    timestamp: str
    transcript: str
    context_type: str
    context_id: str
    action: str
    reply: Optional[str]
    language: str
    complexity: str


class ConversationBuffer:
    """Sliding-window short-term memory for the current session.

    Stores the last N turns. Injected into resolver prompts so the
    assistant can understand follow-up commands like "làm lại", "đổi lại",
    "không phải cái đó", "undo that", etc.

    Thread-safe for async usage (deque is thread-safe for appends/pops).
    """

    def __init__(self, max_turns: int = 10, session_id: str = ""):
        self.max_turns = max_turns
        self.session_id = session_id or f"session_{int(time.time())}"
        self._buffer: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._context_switch_count: Dict[str, int] = {}  # Track context switches
        self._last_context_type: Optional[str] = None

    def add_turn(
        self,
        transcript: str,
        context_type: str,
        context_id: str,
        action: str,
        reply: Optional[str] = None,
        language: str = "vi",
        complexity: str = "simple",
    ) -> None:
        """Record a completed exchange."""
        turn = ConversationTurn(
            timestamp=datetime.now(timezone.utc).isoformat(),
            transcript=transcript,
            context_type=context_type,
            context_id=context_id,
            action=action,
            reply=reply,
            language=language,
            complexity=complexity,
        )
        self._buffer.append(turn)

        # Track context switches
        if self._last_context_type and self._last_context_type != context_type:
            key = f"{self._last_context_type}→{context_type}"
            self._context_switch_count[key] = self._context_switch_count.get(key, 0) + 1
        self._last_context_type = context_type

        logger.debug(
            "[Memory] Buffer turn added: action=%s context=%s buffer_size=%d/%d",
            action, context_type, len(self._buffer), self.max_turns,
        )

    def get_history(self, max_turns: int = 5) -> List[ConversationTurn]:
        """Get the last N turns for prompt injection."""
        return list(self._buffer)[-max_turns:]

    def format_for_prompt(self, max_turns: int = 5) -> str:
        """Format recent conversation history for injection into the resolver prompt."""
        history = self.get_history(max_turns)
        if not history:
            return "[No conversation history — this is the first turn.]"

        lines = ["─── RECENT CONVERSATION HISTORY ───", ""]
        for i, turn in enumerate(history, 1):
            lines.append(f"Turn {i} ({turn.timestamp[:19]}):")
            lines.append(f"  User said: \"{turn.transcript}\"")
            lines.append(f"  Context: {turn.context_type} | Lang: {turn.language}")
            lines.append(f"  Assistant did: {turn.action}")
            if turn.reply:
                lines.append(f"  Assistant replied: \"{turn.reply}\"")
            lines.append("")

        # Add context switch awareness
        if self._context_switch_count:
            switches = sorted(self._context_switch_count.items(), key=lambda x: -x[1])
            lines.append("Context switch patterns this session:")
            for pattern, count in switches[:3]:
                lines.append(f"  {pattern}: {count} time(s)")

        return "\n".join(lines)

    def get_last_action(self) -> Optional[str]:
        """Get the last action taken (for undo/redo awareness)."""
        if self._buffer:
            return self._buffer[-1].action
        return None

    def get_last_context_type(self) -> Optional[str]:
        """Get the last context type used."""
        return self._last_context_type

    def clear(self) -> None:
        """Reset the buffer for a new session."""
        self._buffer.clear()
        self._context_switch_count.clear()
        self._last_context_type = None

    def size(self) -> int:
        return len(self._buffer)


# ═══════════════════════════════════════════════════════════════════════════
# Long-Term Memory: User Profile (Neon PostgreSQL + JSON fallback)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    """Persisted user preferences and learned patterns.

    Primary store: Neon PostgreSQL (user_profiles table)
    Fallback: .memory/user_profile.json
    """
    user_id: str = "default"
    preferred_language: str = "vi"
    frequently_used_actions: Dict[str, int] = field(default_factory=dict)
    frequently_used_contexts: Dict[str, int] = field(default_factory=dict)
    common_workflows: List[Dict[str, Any]] = field(default_factory=list)
    tone_preference: str = "command"
    total_interactions: int = 0
    known_facts: Dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class UserProfileStore:
    """Load/save user profile from Neon PostgreSQL, with JSON fallback."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self._profile: Optional[UserProfile] = None

    async def load(self) -> UserProfile:
        """Load profile from Neon DB, fall back to JSON, or create default."""
        if self._profile:
            return self._profile

        # Try Neon PostgreSQL first
        if DB_ENABLED:
            try:
                from .database import get_or_create_profile
                data = await get_or_create_profile(self.user_id)
                if data:
                    # Parse JSONB fields that may come back as JSON strings from the DB driver
                    def _parse_jsonb(raw, default):
                        """Parse a JSONB value that might be a JSON string or already a dict/list."""
                        if isinstance(raw, str):
                            try:
                                return json.loads(raw)
                            except (json.JSONDecodeError, TypeError):
                                return default
                        if isinstance(raw, (dict, list)):
                            return raw
                        return default

                    self._profile = UserProfile(
                        user_id=data.get("user_id", self.user_id),
                        preferred_language=data.get("preferred_language", "vi"),
                        frequently_used_actions=_parse_jsonb(data.get("frequently_used_actions"), {}),
                        frequently_used_contexts=_parse_jsonb(data.get("frequently_used_contexts"), {}),
                        common_workflows=_parse_jsonb(data.get("common_workflows"), []),
                        tone_preference=data.get("tone_preference", "command"),
                        total_interactions=data.get("total_interactions", 0),
                        known_facts=_parse_jsonb(data.get("known_facts"), {}),
                        created_at=str(data.get("created_at", "")),
                        updated_at=str(data.get("updated_at", "")),
                    )
                    logger.info("[Memory] Profile loaded from Neon: %d interactions, %d known facts, lang=%s",
                               self._profile.total_interactions,
                               len(self._profile.known_facts),
                               self._profile.preferred_language)
                    return self._profile
            except Exception as exc:
                logger.warning("[Memory] Neon profile load failed: %s — trying JSON fallback", exc)

        # Fallback to JSON file
        if PROFILE_PATH.exists():
            try:
                data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
                self._profile = UserProfile(**data)
                logger.info("[Memory] Profile loaded from JSON fallback")
                return self._profile
            except Exception as exc:
                logger.warning("[Memory] JSON profile load failed: %s", exc)

        self._profile = UserProfile(user_id=self.user_id)
        return self._profile

    async def save(self) -> None:
        """Persist to Neon DB (primary) AND JSON (always, as backup)."""
        if not self._profile:
            return
        self._profile.updated_at = datetime.now(timezone.utc).isoformat()

        # Neon PostgreSQL (primary)
        if DB_ENABLED:
            try:
                from .database import update_profile
                # save() is called from learn_fact(), not from record_interaction(),
                # so we use a lightweight upsert for facts only
                from .database import update_profile_facts
                await update_profile_facts(self.user_id, dict(self._profile.known_facts))
            except Exception as exc:
                logger.warning("[Memory] Neon profile save failed: %s", exc)

        # JSON fallback (always write, even if Neon succeeded — belt and suspenders)
        try:
            tmp_path = PROFILE_PATH.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(self._profile.__dict__, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(PROFILE_PATH)
        except Exception as exc:
            logger.warning("[Memory] JSON profile save failed: %s", exc)

    async def record_interaction(
        self, action: str, context_type: str, language: str = "vi",
    ) -> None:
        """Update profile with a completed interaction."""
        profile = await self.load()

        # Update in-memory
        profile.total_interactions += 1
        profile.frequently_used_actions[action] = profile.frequently_used_actions.get(action, 0) + 1
        profile.frequently_used_contexts[context_type] = profile.frequently_used_contexts.get(context_type, 0) + 1

        # Auto-detect preferred language
        if language != "mixed" and profile.total_interactions > 10 and language != profile.preferred_language:
            profile.preferred_language = language

        # Persist to Neon
        if DB_ENABLED:
            try:
                from .database import update_profile
                await update_profile(
                    user_id=self.user_id,
                    action=action,
                    context_type=context_type,
                    language=language,
                )
                return
            except Exception as exc:
                logger.warning("[Memory] Neon profile update failed: %s", exc)

        # JSON fallback (save every 10 interactions)
        if profile.total_interactions % 10 == 0:
            await self.save()

    async def learn_fact(self, key: str, value: str) -> None:
        """Extract and persist a personal fact about the user (name, preference, etc.)."""
        profile = await self.load()
        if key in profile.known_facts and profile.known_facts[key] == value:
            return  # Already known
        profile.known_facts[key] = value
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        logger.info("[Memory] Learned fact: %s = %s", key, value)
        # Save immediately for facts (don't wait for batch)
        if DB_ENABLED:
            try:
                from .database import update_profile_facts
                await update_profile_facts(self.user_id, dict(profile.known_facts))
            except Exception:
                pass
        await self.save()

    async def extract_facts_from_transcript(self, transcript: str) -> Dict[str, str]:
        """Extract personal facts from a user transcript using lightweight heuristics.

        Detects patterns like:
        - "tên tôi là X" / "my name is X" / "tôi tên là X"
        - "tôi thích X" / "I like X" / "I prefer X"
        - "tôi làm việc ở X" / "I work at X"
        - "tôi là X" / "I am X" (role/introduction)

        Returns a dict of {fact_key: fact_value} for newly discovered facts.
        """
        import re
        facts: Dict[str, str] = {}
        t = transcript.strip()

        # Name patterns (Vietnamese + English)
        name_patterns = [
            (r'tên (?:của )?(?:tôi|mình|em|anh|chị) là ([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)(?:[,\.!?]|$)', 'user_name'),
            (r'(?:tôi|mình|em|anh|chị) tên là ([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)(?:[,\.!?]|$)', 'user_name'),
            (r'my name is ([A-Za-z][A-Za-z\s]{1,30}?)(?:[,\.!?]|$)', 'user_name'),
            (r"(?:i am|i'm) ([A-Za-z][A-Za-z\s]{1,30}?)(?:[,\.!?]|$)", 'user_name'),
            (r'gọi (?:tôi|mình|em|anh|chị) là ([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)(?:[,\.!?]|$)', 'user_name'),
        ]
        for pattern, fact_key in name_patterns:
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                if len(name) >= 2 and len(name) <= 30 and not name.lower() in ('a', 'an', 'the', 'not', 'just', 'here', 'there', 'your', 'my'):
                    facts[fact_key] = name
                    break

        # Preference patterns
        pref_patterns = [
            (r'(?:tôi|mình) thích ([^,.!?]{2,40}?)(?:[,.!?]|$)', 'likes'),
            (r'i like ([^,.!?]{2,40}?)(?:[,.!?]|$)', 'likes'),
            (r'(?:tôi|mình) (?:làm việc|học) (?:ở|tại) ([^,.!?]{2,40}?)(?:[,.!?]|$)', 'workplace'),
            (r'i (?:work|study) (?:at|in) ([^,.!?]{2,40}?)(?:[,.!?]|$)', 'workplace'),
            (r'(?:tôi|mình) là (?:một |1 )?(sinh viên|học sinh|giáo viên|kỹ sư|bác sĩ|developer|designer|manager|lập trình viên|nhà phát triển|nhà thiết kế)(?:[,.!?]|$)', 'role'),
            (r'i am (?:a |an )?(student|teacher|engineer|doctor|developer|designer|manager)(?:[,.!?]|$)', 'role'),
        ]
        for pattern, fact_key in pref_patterns:
            if fact_key in facts:
                continue
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if len(val) >= 2 and len(val) <= 40:
                    facts[fact_key] = val

        return facts

    async def get_preferences_for_prompt(self) -> str:
        """Format user preferences for injection into prompts."""
        profile = await self.load()

        lines = ["─── USER PROFILE ───"]

        if profile.total_interactions < 3:
            lines.append(f"Interactions so far: {profile.total_interactions}")
        else:
            top_actions = sorted(profile.frequently_used_actions.items(), key=lambda x: -x[1])[:5]
            top_contexts = sorted(profile.frequently_used_contexts.items(), key=lambda x: -x[1])[:5]
            lines.append(f"Preferred language: {profile.preferred_language}")
            lines.append(f"Total interactions: {profile.total_interactions}")
            lines.append(f"Top actions: {', '.join(f'{a} ({c}x)' for a, c in top_actions)}")
            lines.append(f"Top contexts: {', '.join(f'{c} ({n}x)' for c, n in top_contexts)}")
            lines.append(f"Typical tone: {profile.tone_preference}")

        # Always include known facts, even for new users
        if profile.known_facts:
            lines.append("Known facts about this user:")
            for key, value in profile.known_facts.items():
                lines.append(f"  - {key}: {value}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Long-Term Memory: Interaction Store (Neon PostgreSQL + JSONL fallback)
# ═══════════════════════════════════════════════════════════════════════════

class InteractionStore:
    """Durable log of all successful interactions.

    Primary store: Neon PostgreSQL (interactions table)
    Fallback: .memory/interactions.jsonl

    Features:
    - Keyword + full-text search for similar commands
    - Session-scoped retrieval
    - Training data generation
    """

    def __init__(self):
        pass  # No local state needed — all operations hit DB or fallback

    async def log(
        self,
        session_id: str,
        transcript: str,
        context_type: str,
        action: str,
        params: dict,
        reply: Optional[str],
        language: str,
        complexity: str,
        duration_ms: float,
        user_id: str = "default",
        success: bool = True,
    ) -> None:
        """Append an interaction to Neon DB (primary) or JSONL (fallback)."""
        # Neon PostgreSQL
        if DB_ENABLED:
            try:
                from .database import insert_interaction
                row_id = await insert_interaction(
                    session_id=session_id,
                    user_id=user_id,
                    transcript=transcript,
                    context_type=context_type,
                    action=action,
                    params=params,
                    reply=reply,
                    language=language,
                    complexity=complexity,
                    duration_ms=duration_ms,
                    success=success,
                )
                if row_id is not None:
                    return
            except Exception as exc:
                logger.warning("[Memory] Neon interaction log failed: %s — using JSONL fallback", exc)

        # JSONL fallback
        record = {
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transcript": transcript,
            "context_type": context_type,
            "action": action,
            "params": params,
            "reply": reply,
            "language": language,
            "complexity": complexity,
            "duration_ms": round(duration_ms, 1),
            "success": success,
        }
        try:
            with open(INTERACTIONS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[Memory] JSONL interaction log failed: %s", exc)

    async def find_similar(
        self, transcript: str, context_type: str, user_id: str = "default", max_results: int = 5,
    ) -> List[dict]:
        """Find past interactions similar to the current one.

        Uses Neon PostgreSQL full-text search (gin index) or ILIKE fallback.
        """
        # Neon PostgreSQL
        if DB_ENABLED:
            try:
                from .database import find_similar_interactions
                results = await find_similar_interactions(
                    transcript=transcript,
                    context_type=context_type,
                    user_id=user_id,
                    max_results=max_results,
                )
                if results:
                    return results
            except Exception as exc:
                logger.warning("[Memory] Neon similar-search failed: %s", exc)

        # JSONL fallback: simple keyword overlap
        if not INTERACTIONS_PATH.exists():
            return []

        keywords = set(transcript.lower().split())
        if len(keywords) < 2:
            return []

        scored: List[Tuple[float, dict]] = []
        try:
            with open(INTERACTIONS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ctx_bonus = 2.0 if record.get("context_type") == context_type else 0.0
                    record_words = set(record.get("transcript", "").lower().split())
                    overlap = len(keywords & record_words)
                    score = overlap + ctx_bonus
                    if score > 1:
                        scored.append((score, record))
        except Exception as exc:
            logger.warning("[Memory] JSONL search failed: %s", exc)
            return []

        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:max_results]]

    async def format_similar_for_prompt(
        self, transcript: str, context_type: str, user_id: str = "default", max_results: int = 3,
    ) -> str:
        """Format similar past interactions for injection into prompts."""
        similar = await self.find_similar(transcript, context_type, user_id, max_results)
        if not similar:
            return ""

        lines = ["─── SIMILAR PAST INTERACTIONS ───", ""]
        for i, rec in enumerate(similar, 1):
            ts = str(rec.get("timestamp", ""))[:19]
            lines.append(
                f"{i}. [{ts}] \"{rec['transcript'][:150]}\" → "
                f"{rec['action']} ({rec['context_type']})"
            )
        lines.append("")
        lines.append("Use these as reference for how similar commands were handled.")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Memory Manager — orchestrates short + long term memory
# ═══════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """Central memory coordinator for the agentic pipeline.

    Usage:
        manager = MemoryManager(session_id="abc123")
        history = await manager.get_context_for_prompt(transcript, context_type)
        # ... after resolver completes ...
        await manager.record_exchange(transcript, context_type, action, reply, ...)
    """

    def __init__(self, session_id: str = "", user_id: str = "default"):
        self.session_id = session_id or f"session_{int(time.time())}_{os.urandom(4).hex()}"
        self.user_id = user_id
        self.buffer = ConversationBuffer(max_turns=10, session_id=self.session_id)
        self.profile_store = UserProfileStore(user_id=user_id)
        self.interaction_store = InteractionStore()

    async def get_context_for_prompt(
        self, transcript: str, context_type: str,
    ) -> str:
        """Build the full memory context block for injection into the resolver prompt.

        Includes:
        - Recent conversation history (short-term, in-memory)
        - User profile preferences (long-term, Neon DB)
        - Similar past interactions (long-term, Neon DB)
        """
        parts = []

        # Short-term: conversation history (always include, even if empty)
        history = self.buffer.format_for_prompt(max_turns=5)
        if history and history != "[No conversation history — this is the first turn.]":
            parts.append(history)

        # Long-term: user preferences (always include — useful even on first turn)
        profile = await self.profile_store.get_preferences_for_prompt()
        parts.append(profile)

        # Long-term: similar interactions
        similar = await self.interaction_store.format_similar_for_prompt(
            transcript, context_type, self.user_id, max_results=3,
        )
        if similar:
            parts.append(similar)

        return "\n\n".join(parts)

    async def record_exchange(
        self,
        transcript: str,
        context_type: str,
        action: str,
        reply: Optional[str],
        language: str,
        complexity: str,
        duration_ms: float,
        params: dict = None,
        success: bool = True,
    ) -> None:
        """Record a completed exchange in all memory stores."""
        params = params or {}

        # Short-term buffer (synchronous)
        self.buffer.add_turn(
            transcript=transcript,
            context_type=context_type,
            context_id="",
            action=action,
            reply=reply,
            language=language,
            complexity=complexity,
        )

        # Long-term profile (async — Neon DB)
        await self.profile_store.record_interaction(
            action=action,
            context_type=context_type,
            language=language,
        )

        # ── Extract personal facts from transcript ──
        try:
            new_facts = await self.profile_store.extract_facts_from_transcript(transcript)
            for key, value in new_facts.items():
                await self.profile_store.learn_fact(key, value)
        except Exception as fact_exc:
            logger.debug("[Memory] Fact extraction skipped: %s", fact_exc)

        # Long-term interaction log (async — Neon DB)
        await self.interaction_store.log(
            session_id=self.session_id,
            transcript=transcript,
            context_type=context_type,
            action=action,
            params=params,
            reply=reply,
            language=language,
            complexity=complexity,
            duration_ms=duration_ms,
            user_id=self.user_id,
            success=success,
        )

    def clear_session(self) -> None:
        """Clear short-term memory (start fresh session)."""
        self.buffer.clear()
        logger.info("[Memory] Session cleared: %s", self.session_id)


# ═══════════════════════════════════════════════════════════════════════════
# Global Memory Manager (singleton per session)
# ═══════════════════════════════════════════════════════════════════════════

_sessions: Dict[str, MemoryManager] = {}


def get_memory_manager(session_id: str = "", user_id: str = "default") -> MemoryManager:
    """Get or create a MemoryManager for the given session."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]

    manager = MemoryManager(session_id=session_id, user_id=user_id)
    if session_id:
        _sessions[session_id] = manager
        # Cleanup old sessions (keep last 50)
        if len(_sessions) > 50:
            oldest = min(_sessions.keys(), key=lambda k: _sessions[k].buffer.size())
            del _sessions[oldest]

    return manager
