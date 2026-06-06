import sys
from pathlib import Path

# Support `python src/main.py` from the project root (not only `python -m src.main`).
if __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    __package__ = "src"

from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import app
from .replay import store

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Service index — use /health for probes (Next.js, load balancers, etc.)."""
    return JSONResponse(
        content={
            "service": "voice-ai-microservice",
            "status": "ok",
            "health": "/health",
            "tester": "/tester",
            "live-viewer": "/live-viewer",
            "api": "/api/v1/voice/process",
        }
    )


@app.get("/tester", include_in_schema=False)
async def tester_page():
    tester_path = Path(__file__).parent / "static" / "tester.html"
    return FileResponse(tester_path, media_type="text/html")


@app.get("/live-viewer", include_in_schema=False)
async def live_viewer_page():
    viewer_path = Path(__file__).parent / "static" / "live_viewer.html"
    return FileResponse(viewer_path, media_type="text/html")


@app.get("/api/live-viewer/entries", include_in_schema=False)
async def live_viewer_entries():
    return JSONResponse(content=store.get_all())


@app.get("/api/live-viewer/entries/{entry_id}", include_in_schema=False)
async def live_viewer_entry(entry_id: str):
    entry = store.get_by_id(entry_id)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": "Entry not found"})
    return JSONResponse(content=entry)


@app.get("/api/live-viewer/stream", include_in_schema=False)
async def live_viewer_stream():
    return StreamingResponse(
        store.subscribe(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
