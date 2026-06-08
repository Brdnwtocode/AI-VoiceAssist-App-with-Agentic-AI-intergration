"""
Neon PostgreSQL Database Layer — connection pool + schema + queries.

Uses asyncpg for high-performance async PostgreSQL access.
Connection string format:
  postgresql://user:password@ep-xxxx.us-east-2.aws.neon.tech/dbname?sslmode=require

Tables:
  - interactions       — durable log of every resolved command
  - user_profiles      — per-user preferences and learned patterns

All operations are non-blocking (async) to fit the FastAPI + LangGraph pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

from .config import DATABASE_URL, DB_ENABLED, logger

# ── Global connection pool (lazy-init) ──
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> Optional[asyncpg.Pool]:
    """Get or create the asyncpg connection pool.

    Returns None if DATABASE_URL is not configured.
    """
    global _pool
    if not DB_ENABLED:
        return None
    if _pool is None:
        try:
            _pool = await asyncpg.create_pool(
                dsn=DATABASE_URL,
                min_size=1,
                max_size=5,
                command_timeout=10.0,
            )
            logger.info("Neon PostgreSQL pool created (min=1, max=5)")
        except Exception as exc:
            logger.error("Failed to create Neon pool: %s — long-term memory disabled", exc)
            return None
    return _pool


async def close_pool() -> None:
    """Close the connection pool on shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Neon PostgreSQL pool closed")


# ═══════════════════════════════════════════════════════════════════════════
# Schema Migration
# ═══════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Interactions: durable log of every resolved voice command
CREATE TABLE IF NOT EXISTS interactions (
    id              BIGSERIAL PRIMARY KEY,
    session_id      VARCHAR(64)  NOT NULL,
    user_id         VARCHAR(64)  NOT NULL DEFAULT 'default',
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    transcript      TEXT         NOT NULL,
    context_type    VARCHAR(20)  NOT NULL,
    action          VARCHAR(50)  NOT NULL,
    params          JSONB        NOT NULL DEFAULT '{}',
    reply           TEXT,
    language        VARCHAR(10)  NOT NULL DEFAULT 'vi',
    complexity      VARCHAR(20)  NOT NULL DEFAULT 'simple',
    duration_ms     DOUBLE PRECISION,
    success         BOOLEAN      NOT NULL DEFAULT true
);

-- User profiles: per-user preferences and learned patterns
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id                 VARCHAR(64) PRIMARY KEY DEFAULT 'default',
    preferred_language      VARCHAR(10)  NOT NULL DEFAULT 'vi',
    frequently_used_actions JSONB        NOT NULL DEFAULT '{}',
    frequently_used_contexts JSONB       NOT NULL DEFAULT '{}',
    common_workflows        JSONB        NOT NULL DEFAULT '[]',
    tone_preference         VARCHAR(20)  NOT NULL DEFAULT 'command',
    total_interactions      INTEGER      NOT NULL DEFAULT 0,
    known_facts             JSONB        NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Migration: add known_facts column if it doesn't exist (idempotent)
DO $$ BEGIN
    ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS known_facts JSONB NOT NULL DEFAULT '{}';
EXCEPTION WHEN duplicate_column THEN NULL;
END $$;

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_interactions_session
    ON interactions (session_id);

CREATE INDEX IF NOT EXISTS idx_interactions_user_ts
    ON interactions (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_interactions_context
    ON interactions (context_type);

CREATE INDEX IF NOT EXISTS idx_interactions_action
    ON interactions (action);

-- Full-text search index for finding similar commands
CREATE INDEX IF NOT EXISTS idx_interactions_transcript_fts
    ON interactions USING gin (to_tsvector('simple', transcript));
"""


async def migrate_schema() -> bool:
    """Run schema migration — creates tables if they don't exist.

    Safe to call multiple times (uses IF NOT EXISTS).
    Returns True if migration succeeded, False otherwise.
    """
    pool = await get_pool()
    if pool is None:
        return False

    try:
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("Neon schema migration complete — tables ready")
        return True
    except Exception as exc:
        logger.error("Neon schema migration failed: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Interaction Queries
# ═══════════════════════════════════════════════════════════════════════════

async def insert_interaction(
    session_id: str,
    user_id: str,
    transcript: str,
    context_type: str,
    action: str,
    params: dict,
    reply: Optional[str],
    language: str,
    complexity: str,
    duration_ms: float,
    success: bool = True,
) -> Optional[int]:
    """Insert a completed interaction and return its ID."""
    pool = await get_pool()
    if pool is None:
        return None

    try:
        async with pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO interactions
                    (session_id, user_id, transcript, context_type, action,
                     params, reply, language, complexity, duration_ms, success)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                session_id,
                user_id,
                transcript,
                context_type,
                action,
                json.dumps(params, ensure_ascii=False),
                reply,
                language,
                complexity,
                duration_ms,
                success,
            )
        return row_id
    except Exception as exc:
        logger.warning("Failed to insert interaction: %s", exc)
        return None


async def find_similar_interactions(
    transcript: str,
    context_type: str,
    user_id: str = "default",
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Find past interactions similar to the current command.

    Uses PostgreSQL full-text search + context_type matching.
    Falls back to simple ILIKE if FTS index isn't available.
    """
    pool = await get_pool()
    if pool is None:
        return []

    try:
        async with pool.acquire() as conn:
            # Try FTS first; fall back to ILIKE if no results
            rows = await conn.fetch(
                """
                SELECT transcript, context_type, action, reply, language, timestamp
                FROM interactions
                WHERE user_id = $1
                  AND context_type = $2
                  AND to_tsvector('simple', transcript) @@ plainto_tsquery('simple', $3)
                ORDER BY timestamp DESC
                LIMIT $4
                """,
                user_id, context_type, transcript, int(max_results),
            )

            # Fallback: keyword ILIKE if FTS returned nothing
            if not rows:
                keywords = [w for w in transcript.split() if len(w) > 2]
                if keywords:
                    ilike_clauses = " OR ".join(
                        f"transcript ILIKE '%' || ${i+3} || '%'"
                        for i in range(len(keywords))
                    )
                    rows = await conn.fetch(
                        f"""
                        SELECT transcript, context_type, action, reply, language, timestamp
                        FROM interactions
                        WHERE user_id = $1
                          AND context_type = $2
                          AND ({ilike_clauses})
                        ORDER BY timestamp DESC
                        LIMIT $3
                        """,
                        user_id, context_type, *keywords, int(max_results),
                    )

            return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Failed to search interactions: %s", exc)
        return []


async def get_recent_interactions(
    session_id: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Get the most recent interactions for a session."""
    pool = await get_pool()
    if pool is None:
        return []

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT transcript, context_type, action, reply, language, timestamp
                FROM interactions
                WHERE session_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """,
                session_id, int(max_results),
            )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("Failed to get recent interactions: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# User Profile Queries
# ═══════════════════════════════════════════════════════════════════════════

async def get_or_create_profile(user_id: str = "default") -> Dict[str, Any]:
    """Get user profile or create default if not exists."""
    pool = await get_pool()
    if pool is None:
        return _default_profile(user_id)

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM user_profiles WHERE user_id = $1", user_id
            )
            if row is None:
                profile = _default_profile(user_id)
                await conn.execute(
                    """
                    INSERT INTO user_profiles (user_id, preferred_language)
                    VALUES ($1, $2)
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    user_id, profile["preferred_language"],
                )
                return profile
            return dict(row)
    except Exception as exc:
        logger.warning("Failed to get profile: %s", exc)
        return _default_profile(user_id)


async def update_profile(
    user_id: str,
    action: str,
    context_type: str,
    language: str,
) -> None:
    """Increment interaction counts and update preferences in the profile."""
    pool = await get_pool()
    if pool is None:
        return

    try:
        async with pool.acquire() as conn:
            # Upsert profile with atomic increments
            await conn.execute(
                """
                INSERT INTO user_profiles (user_id, preferred_language, total_interactions,
                    frequently_used_actions, frequently_used_contexts)
                VALUES ($1, $5, 1, $2::jsonb, $3::jsonb)
                ON CONFLICT (user_id) DO UPDATE SET
                    total_interactions = user_profiles.total_interactions + 1,
                    frequently_used_actions = user_profiles.frequently_used_actions
                        || jsonb_build_object($4, COALESCE((user_profiles.frequently_used_actions->>$4)::int, 0) + 1),
                    frequently_used_contexts = user_profiles.frequently_used_contexts
                        || jsonb_build_object($6, COALESCE((user_profiles.frequently_used_contexts->>$6)::int, 0) + 1),
                    preferred_language = CASE
                        WHEN user_profiles.total_interactions > 10 AND $5 != 'mixed'
                        THEN $5
                        ELSE user_profiles.preferred_language
                    END,
                    updated_at = NOW()
                """,
                user_id,
                json.dumps({action: 1}),
                json.dumps({context_type: 1}),
                action,
                language,
                context_type,
            )
    except Exception as exc:
        logger.warning("Failed to update profile: %s", exc)


def _default_profile(user_id: str = "default") -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "preferred_language": "vi",
        "frequently_used_actions": {},
        "frequently_used_contexts": {},
        "common_workflows": [],
        "tone_preference": "command",
        "total_interactions": 0,
        "known_facts": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def update_profile_facts(user_id: str, facts: Dict[str, str]) -> None:
    """Update the known_facts JSONB column for a user profile."""
    pool = await get_pool()
    if pool is None:
        return

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_profiles (user_id, known_facts)
                VALUES ($1, $2::jsonb)
                ON CONFLICT (user_id) DO UPDATE SET
                    known_facts = user_profiles.known_facts || $2::jsonb,
                    updated_at = NOW()
                """,
                user_id,
                json.dumps(facts),
            )
    except Exception as exc:
        logger.warning("Failed to update profile facts: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════

async def init_db() -> bool:
    """Initialize the database: create pool + run migrations.

    Call once at app startup. Safe to call even if DATABASE_URL is unset.
    """
    if not DB_ENABLED:
        logger.info("Neon DB not configured — long-term memory using local JSON fallback")
        return False

    pool = await get_pool()
    if pool is None:
        return False

    ok = await migrate_schema()
    if ok:
        logger.info("Neon database initialized — long-term memory active")
    return ok
