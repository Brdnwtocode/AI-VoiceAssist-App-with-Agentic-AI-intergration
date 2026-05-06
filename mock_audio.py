"""
Generate a tiny audio file for multipart tests.

Default output is WebM-shaped bytes (EBML header + padding). With MOCK_OPENAI=1 the
service never sends audio to Whisper, so this file is sufficient for contract tests.

If ffmpeg is on PATH, you can optionally emit a short silent WebM/MP3.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from typing import Optional

# Minimal WebM-shaped payload (EBML 0x1A45DFA3 + stub + NUL padding). Decode at runtime.
# With MOCK_OPENAI=1 the service does not call Whisper; use ffmpeg below or a real clip for live STT.
SILENT_WEBM_BASE64 = "GkXfo4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="

_WEBM_SHELL = base64.b64decode(SILENT_WEBM_BASE64)


def _write_temp(suffix: str, data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return path


def _try_ffmpeg_webm(duration_sec: float = 2.0) -> Optional[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    out = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    out.close()
    try:
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r=48000:cl=mono",
                "-t",
                str(duration_sec),
                "-c:a",
                "libopus",
                "-b:a",
                "48k",
                "-y",
                out.name,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(out.name, "rb") as handle:
            return handle.read()
    except Exception:
        return None
    finally:
        try:
            os.unlink(out.name)
        except OSError:
            pass


def create_test_audio(
    suffix: str = ".webm",
    prefer_ffmpeg: bool = True,
) -> str:
    """
    Create a temp audio file and return its path.
    Prefer silent WebM via ffmpeg when available; otherwise embedded shell bytes.
    """
    if prefer_ffmpeg and suffix.lower().endswith(".webm"):
        data = _try_ffmpeg_webm()
        if data:
            return _write_temp(".webm", data)

    if suffix.lower().endswith(".webm"):
        return _write_temp(".webm", _WEBM_SHELL)

    return _write_temp(suffix, _WEBM_SHELL)


if __name__ == "__main__":
    print(create_test_audio())
