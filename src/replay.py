import asyncio
import json
import time
from collections import deque
from typing import Any, Dict, List, Optional

MAX_ENTRIES = 500
MAX_PIPELINE_TRACES = 200


class PipelineTrace:
    """A trace of a single AI pipeline execution through the LangGraph stages."""

    def __init__(self, request_id: str, transcript: str, context_type: str):
        self.id = request_id
        self.timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.transcript = transcript[:200]
        self.context_type = context_type
        self.stages: List[Dict[str, Any]] = []
        self.status = "running"  # running, completed, blocked, error
        self.total_duration_ms: float = 0
        self._start_time = time.perf_counter()

    def add_stage(self, name: str, status: str, data: dict = None, duration_ms: float = 0):
        """Record a pipeline stage execution."""
        self.stages.append({
            "name": name,
            "status": status,  # passed, failed, skipped, running
            "data": data or {},
            "duration_ms": round(duration_ms, 1),
            "timestamp": time.strftime("%H:%M:%S", time.localtime()),
        })

    def finalize(self, status: str):
        """Mark the trace as completed."""
        self.status = status
        self.total_duration_ms = round((time.perf_counter() - self._start_time) * 1000, 1)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "transcript": self.transcript,
            "context_type": self.context_type,
            "stages": self.stages,
            "status": self.status,
            "total_duration_ms": self.total_duration_ms,
        }


class LiveEntry:
    def __init__(
        self,
        request: dict,
        response: dict,
        duration_ms: float,
        status_code: int,
        pipeline_trace: Optional[PipelineTrace] = None,
    ):
        self.id = str(time.time_ns())
        self.timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.request = request
        self.response = response
        self.duration_ms = duration_ms
        self.status_code = status_code
        self.pipeline_trace = pipeline_trace

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
        resp_body = self.response.get("body", {})
        action = ""
        pipeline_status = ""
        if isinstance(resp_body, dict):
            action = resp_body.get("action", "")
        if self.pipeline_trace:
            pipeline_status = self.pipeline_trace.status
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "method": self.request.get("method", "POST"),
            "path": self.request.get("path", ""),
            "status_code": self.status_code,
            "duration_ms": round(self.duration_ms, 1),
            "summary": self._build_summary(),
            "action": action,
            "pipeline_status": pipeline_status,
        }

    def to_full(self) -> dict:
        summary = self._build_summary()
        resp_body = self.response.get("body", {})
        action = ""
        if isinstance(resp_body, dict):
            action = resp_body.get("action", "")
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "summary": summary,
            "method": self.request.get("method", "POST"),
            "path": self.request.get("path", ""),
            "action": action,
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
            "pipeline_trace": self.pipeline_trace.to_dict() if self.pipeline_trace else None,
        }


class LiveStore:
    def __init__(self):
        self._entries: deque = deque(maxlen=MAX_ENTRIES)
        self._events: asyncio.Queue = asyncio.Queue()
        self._pipeline_traces: deque = deque(maxlen=MAX_PIPELINE_TRACES)
        self._pipeline_events: asyncio.Queue = asyncio.Queue()
        self._active_traces: Dict[str, PipelineTrace] = {}

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

    # ── Pipeline Tracing ──

    def start_pipeline_trace(self, request_id: str, transcript: str, context_type: str) -> PipelineTrace:
        """Create a new pipeline trace for a request."""
        trace = PipelineTrace(request_id, transcript, context_type)
        self._active_traces[request_id] = trace
        self._pipeline_events.put_nowait({"type": "pipeline_start", "trace": trace.to_dict()})
        return trace

    def add_pipeline_stage(self, request_id: str, name: str, status: str, data: dict = None, duration_ms: float = 0):
        """Record a pipeline stage and emit SSE event."""
        trace = self._active_traces.get(request_id)
        if trace:
            trace.add_stage(name, status, data, duration_ms)
            self._pipeline_events.put_nowait({
                "type": "pipeline_stage",
                "request_id": request_id,
                "stage": {"name": name, "status": status, "data": data or {}, "duration_ms": round(duration_ms, 1)},
            })

    def finish_pipeline_trace(self, request_id: str, status: str):
        """Finalize a pipeline trace."""
        trace = self._active_traces.pop(request_id, None)
        if trace:
            trace.finalize(status)
            self._pipeline_traces.append(trace)
            self._pipeline_events.put_nowait({"type": "pipeline_end", "trace": trace.to_dict()})

    def get_pipeline_traces(self) -> List[dict]:
        return [t.to_dict() for t in self._pipeline_traces]

    def get_pipeline_trace(self, request_id: str) -> Optional[dict]:
        trace = self._active_traces.get(request_id)
        if trace:
            return trace.to_dict()
        for t in self._pipeline_traces:
            if t.id == request_id:
                return t.to_dict()
        return None

    def get_pipeline_trace_obj(self, request_id: str) -> Optional[PipelineTrace]:
        """Get the raw PipelineTrace object (for linking to LiveEntry)."""
        trace = self._active_traces.get(request_id)
        if trace:
            return trace
        for t in self._pipeline_traces:
            if t.id == request_id:
                return t
        return None

    async def subscribe(self):
        while True:
            entry = await self._events.get()
            yield f"data: {json.dumps(entry.to_full())}\n\n"

    async def subscribe_pipeline(self):
        """SSE stream for pipeline trace events."""
        while True:
            event = await self._pipeline_events.get()
            yield f"data: {json.dumps(event)}\n\n"


store = LiveStore()
