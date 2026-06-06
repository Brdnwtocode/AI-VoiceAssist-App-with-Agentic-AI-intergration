import asyncio
import json
import time
from collections import deque
from typing import Any, Dict, List, Optional

MAX_ENTRIES = 500


class LiveEntry:
    def __init__(
        self,
        request: dict,
        response: dict,
        duration_ms: float,
        status_code: int,
    ):
        self.id = str(time.time_ns())
        self.timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.request = request
        self.response = response
        self.duration_ms = duration_ms
        self.status_code = status_code

    def _build_summary(self) -> str:
        body = self.request.get("body", {})
        if isinstance(body, dict):
            ctx = body.get("context_type", "")
            transcript = body.get("transcript", "")
            has_audio = body.get("has_audio", False)
            parts = []
            if ctx:
                parts.append(f"[{ctx}]")
            if transcript:
                preview = transcript[:80]
                parts.append(f'"{preview}"')
                if len(transcript) > 80:
                    parts[-1] += "..."
            if has_audio:
                parts.append("(audio)")
            if parts:
                return " ".join(parts)
        return self.request.get("path", "")

    def to_summary(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "method": self.request.get("method", "POST"),
            "path": self.request.get("path", ""),
            "status_code": self.status_code,
            "duration_ms": round(self.duration_ms, 1),
            "summary": self._build_summary(),
        }

    def to_full(self) -> dict:
        summary = self._build_summary()
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "summary": summary,
            "method": self.request.get("method", "POST"),
            "path": self.request.get("path", ""),
            "request": {
                "method": self.request.get("method", "POST"),
                "path": self.request.get("path", ""),
                "query_string": self.request.get("query_string", ""),
                "client": self.request.get("client"),
                "headers": self.request.get("headers", {}),
                "form_fields": self.request.get("form_fields", {}),
                "audio_info": self.request.get("audio_info"),
            },
            "response": {
                "content_type": self.response.get("content_type", "application/json"),
                "body": self.response.get("body", {}),
            },
            "duration_ms": round(self.duration_ms, 1),
            "status_code": self.status_code,
        }


class LiveStore:
    def __init__(self):
        self._entries: deque = deque(maxlen=MAX_ENTRIES)
        self._events: asyncio.Queue = asyncio.Queue()

    def add(self, entry: LiveEntry) -> None:
        self._entries.append(entry)
        self._events.put_nowait(entry)

    def get_all(self) -> List[dict]:
        return [e.to_summary() for e in self._entries]

    def get_by_id(self, entry_id: str) -> Optional[dict]:
        for e in self._entries:
            if e.id == entry_id:
                return e.to_full()
        return None

    async def subscribe(self):
        while True:
            entry = await self._events.get()
            yield f"data: {json.dumps(entry.to_full())}\n\n"


store = LiveStore()
