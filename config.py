import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from litellm import Router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("voice_ai_microservice")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTEXTS = ("NOTE", "STACK", "TASK", "CALENDAR")
ALLOWED_MIME_TYPES = {"audio/webm", "audio/mp3"}
RESOLVER_PRIMARY = "gemini/gemini-2.5-flash"
RESOLVER_FALLBACK = "groq/llama-3.3-70b-versatile"
SENTINEL_MODEL = "groq/llama-3.1-8b-instant"
LLM_TIMEOUT = 30.0

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
            "fallbacks": ["groq/llama-3.3-70b-versatile"]
        }
    },
    {
        "model_name": "groq/llama-3.3-70b-versatile",
        "litellm_params": {
            "model": "groq/llama-3.3-70b-versatile"
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
