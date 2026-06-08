import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from litellm import Router

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("voice_ai_microservice")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTEXTS = ("NOTE", "STACK", "TASK", "CALENDAR", "TASKS")
ALLOWED_MIME_TYPES = {"audio/webm", "audio/mp3", "audio/mpeg"}
RESOLVER_PRIMARY = "gemini/gemini-2.5-flash"
# Fallback chain: OpenRouter first, then Groq as last resort
RESOLVER_FALLBACKS = [
    "openrouter/openai/gpt-oss-120b:free",
    "groq/llama-3.3-70b-versatile",
]
SENTINEL_MODEL = "groq/llama-3.1-8b-instant"

# ── Multi-Expert Orchestration Config ─────────────────────────────────────
# Experts use fast 8B models for structured short outputs (~300-800ms each).
# The Resolver (Gemini 2.5 Flash) synthesizes their findings.
# This keeps total latency under budget: safety(~300ms) + router(~200ms)
# + experts(parallel ~500ms) + resolver(~1.2s) ≈ 2.2s after STT.
EXPERT_MODEL = "groq/llama-3.1-8b-instant"
EXPERT_TIMEOUT = 15.0  # Experts should be fast — tight timeout

LLM_TIMEOUT = 30.0

# ── Neon PostgreSQL (long-term memory) ──
DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_ENABLED = bool(DATABASE_URL and DATABASE_URL != "postgresql://neondb_owner:your_password@ep-your-project.us-east-2.aws.neon.tech/neondb?sslmode=require")

MOCK_OPENAI = os.getenv("MOCK_OPENAI", "").strip().lower() in ("1", "true", "yes")

router = Router(model_list=[
    {
        "model_name": "sentinel",
        "litellm_params": {
            "model": "groq/llama-3.1-8b-instant"
        }
    },
    {
        "model_name": "resolver",
        "litellm_params": {
            "model": "gemini/gemini-2.5-flash",
            "fallbacks": [
                "openrouter/openai/gpt-oss-120b:free",
                "groq/llama-3.3-70b-versatile"
            ]
        }
    },
    {
        "model_name": "openrouter/openai/gpt-oss-120b:free",
        "litellm_params": {
            "model": "openrouter/openai/gpt-oss-120b:free",
            "api_key": os.environ.get("OPENROUTER_API_KEY")
        }
    },
    {
        "model_name": "groq/llama-3.3-70b-versatile",
        "litellm_params": {
            "model": "groq/llama-3.3-70b-versatile"
        }
    },
    {
        "model_name": "expert",
        "litellm_params": {
            "model": "groq/llama-3.1-8b-instant",
            "fallbacks": ["groq/llama-3.3-70b-versatile"]
        }
    },
    {
        "model_name": "stt-router",
        "litellm_params": {
            "model": "deepgram/nova-2",
            "api_key": os.environ.get("DEEPGRAM_API_KEY"),
            "timeout": 2.5
        }
    },
    {
        "model_name": "stt-router",
        "litellm_params": {
            "model": "groq/whisper-large-v3",
            "api_key": os.environ.get("GROQ_API_KEY")
        }
    }
])

app = FastAPI(title="Voice AI Microservice", version="1.0.0")

_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
if _origins_env == "*":
    _cors_origins = ["*"]
else:
    _cors_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False if _cors_origins == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routes import register_routes

register_routes(app)
