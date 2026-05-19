from fastapi import FastAPI
from fastapi.responses import FileResponse
from config import app
import routes  # Triggers route registration via decorators
from fastapi.staticfiles import StaticFiles

# Mount static directory for other static assets (if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Explicit route for the tester page
@app.get("/tester", include_in_schema=False)
async def tester_page():
    from pathlib import Path
    tester_path = Path(__file__).parent / "static" / "tester.html"
    return FileResponse(tester_path, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)