"""
Contract integration tests for the Voice AI FastAPI microservice.

Default mode (--mock-openai): expects server started with MOCK_OPENAI=1 and uses
X-Mock-* headers so OpenAI is never required.

Live mode (--real-openai): server must have MOCK_OPENAI unset and a valid OPENAI_API_KEY.
Only non-deterministic LLM tests are skipped; validation and /health behavior are still checked.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

import mock_audio

SERVICE_URL = os.getenv("SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")

NOTE_ID = "550e8400-e29b-41d4-a716-446655440001"
STACK_ID = "550e8400-e29b-41d4-a716-446655440002"
USER_ID = "550e8400-e29b-41d4-a716-446655440099"


def _note_state(content: str = "Old content") -> Dict[str, Any]:
    return {
        "id": NOTE_ID,
        "userId": USER_ID,
        "title": "My Note",
        "content": content,
        "createdAt": "2026-05-07T00:00:00.000Z",
        "updatedAt": "2026-05-07T00:00:00.000Z",
    }


def _mock_headers(transcript: str, tool: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    h: Dict[str, str] = {
        "X-Mock-Transcript": transcript,
        "X-Mock-Tool": tool,
    }
    if args is None:
        h["X-Mock-Args"] = "{}"
    else:
        h["X-Mock-Args"] = json.dumps(args)
    return h


def assert_success_contract(body: Dict[str, Any]) -> None:
    for key in ("transcript", "action", "success", "message", "updatedData", "reply"):
        assert key in body, f"missing key {key}: {body!r}"
    assert isinstance(body["transcript"], str)
    assert isinstance(body["action"], str)
    assert body["success"] is True
    assert body["message"] is None or isinstance(body["message"], str)
    ud = body["updatedData"]
    assert ud is None or isinstance(ud, dict)
    assert body["reply"] is None or isinstance(body["reply"], str)
    assert not (ud is not None and body["reply"] is not None), (
        "updatedData and reply must not both be set"
    )


def assert_error_response(resp: httpx.Response, expected_status: int) -> None:
    assert resp.status_code == expected_status, f"{resp.status_code} {resp.text}"
    data = resp.json()
    assert data.keys() == {"error"}, data
    assert isinstance(data["error"], str)


def _connect_or_fail(client: httpx.Client) -> None:
    try:
        r = client.get(f"{SERVICE_URL}/health", timeout=5.0)
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"Cannot reach {SERVICE_URL}. Start the server (e.g. run_tests.bat) — {exc}"
        ) from exc
    if r.status_code not in (200, 503):
        raise SystemExit(f"Unexpected /health status {r.status_code}: {r.text}")


def test_health(use_mock: bool, client: httpx.Client) -> None:
    r = client.get(f"{SERVICE_URL}/health", timeout=10.0)
    if use_mock:
        assert r.status_code == 200, r.text
        body = r.json()
        assert body == {"status": "ok", "api": "connected"}, body
    else:
        assert r.status_code in (200, 503)
        body = r.json()
        assert "status" in body
        if r.status_code == 200:
            assert body.get("api") == "connected"


def test_note_update_append(audio_path: str, client: httpx.Client) -> None:
    note = _note_state()
    headers = _mock_headers(
        "Append meeting notes to the end (dictated).",
        "update_note",
        {"content_to_insert": "Meeting notes from today", "action_type": "append"},
    )
    with open(audio_path, "rb") as f:
        files = {"audio": ("t.webm", f, "audio/webm")}
        data = {
            "context_type": "NOTE",
            "context_id": NOTE_ID,
            "cursor_position": "5",
            "note_state": json.dumps(note),
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "update_note"
    assert body["message"] == "Note updated"
    ud = body["updatedData"]
    assert ud is not None
    for k in ("id", "title", "content", "createdAt", "updatedAt"):
        assert k in ud
    assert "userId" not in ud
    assert ud["id"] == NOTE_ID
    assert body.get("reply") is None
    expected = note["content"] + "\n" + "Meeting notes from today"
    assert ud["content"] == expected


def test_note_insert_at_cursor(audio_path: str, client: httpx.Client) -> None:
    note = _note_state("0123456789")
    headers = _mock_headers(
        "Insert at cursor position now.",
        "update_note",
        {"content_to_insert": "XX", "action_type": "insert_at_cursor"},
    )
    with open(audio_path, "rb") as f:
        files = {"audio": ("t.webm", f, "audio/webm")}
        data = {
            "context_type": "NOTE",
            "context_id": NOTE_ID,
            "cursor_position": "3",
            "note_state": json.dumps(note),
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "update_note"
    assert body["updatedData"]["content"] == "012XX3456789"


def test_stack_add_row(audio_path: str, client: httpx.Client) -> None:
    schema = [
        {"id": "col1", "name": "Task Name", "type": "TEXT"},
        {"id": "col2", "name": "Priority", "type": "INT"},
        {"id": "col3", "name": "Due Date", "type": "DATE"},
        {"id": "col4", "name": "Completed", "type": "BOOLEAN"},
    ]
    headers = _mock_headers(
        "Add stack row: buy groceries task.",
        "add_stack_row",
        {
            "Task Name": "Buy groceries",
            "Priority": 2,
            "Due Date": "2026-05-10",
            "Completed": False,
        },
    )
    with open(audio_path, "rb") as f:
        files = {"audio": ("t.webm", f, "audio/webm")}
        data = {
            "context_type": "STACK",
            "context_id": STACK_ID,
            "dynamic_schema": json.dumps(schema),
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "add_stack_row"
    assert body["message"] == "Row added"
    ud = body["updatedData"]
    assert ud["stackId"] == STACK_ID
    assert ud["id"].startswith("temp_row_")
    d = ud["data"]
    assert d["col1"] == "Buy groceries"
    assert d["col2"] == 2
    assert d["col3"] == "2026-05-10"
    assert d["col4"] is False


def test_none_action(audio_path: str, client: httpx.Client) -> None:
    note = _note_state()
    headers = _mock_headers("Hello, no data change.", "no_action", {})
    with open(audio_path, "rb") as f:
        files = {"audio": ("t.webm", f, "audio/webm")}
        data = {
            "context_type": "NOTE",
            "context_id": NOTE_ID,
            "note_state": json.dumps(note),
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "none"
    assert body["updatedData"] is None
    assert body["message"] == "No action recognized from command"


def test_invalid_audio_mime(audio_path: str, client: httpx.Client) -> None:
    note = _note_state()
    headers = _mock_headers("x", "no_action", {})
    with open(audio_path, "rb") as f:
        files = {"audio": ("x.txt", f, "text/plain")}
        data = {
            "context_type": "NOTE",
            "context_id": NOTE_ID,
            "note_state": json.dumps(note),
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    assert_error_response(r, 400)
    assert "invalid" in r.json()["error"].lower()


def test_missing_note_state(audio_path: str, client: httpx.Client) -> None:
    headers = _mock_headers("x", "no_action", {})
    with open(audio_path, "rb") as f:
        files = {"audio": ("t.webm", f, "audio/webm")}
        data = {
            "context_type": "NOTE",
            "context_id": NOTE_ID,
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "none"
    assert body["updatedData"] is None


def test_missing_context_type(audio_path: str, client: httpx.Client) -> None:
    note = _note_state()
    headers = _mock_headers("x", "no_action", {})
    with open(audio_path, "rb") as f:
        files = {"audio": ("t.webm", f, "audio/webm")}
        data = {
            "context_id": NOTE_ID,
            "note_state": json.dumps(note),
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    assert_error_response(r, 400)
    assert r.json()["error"] == "Invalid request"


def test_invalid_context_type(audio_path: str, client: httpx.Client) -> None:
    note = _note_state()
    headers = _mock_headers("x", "no_action", {})
    with open(audio_path, "rb") as f:
        files = {"audio": ("t.webm", f, "audio/webm")}
        data = {
            "context_type": "INVALID",
            "context_id": NOTE_ID,
            "note_state": json.dumps(note),
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    assert_error_response(r, 400)


def test_invalid_context_id_uuid(audio_path: str, client: httpx.Client) -> None:
    note = _note_state()
    headers = _mock_headers("x", "no_action", {})
    with open(audio_path, "rb") as f:
        files = {"audio": ("t.webm", f, "audio/webm")}
        data = {
            "context_type": "NOTE",
            "context_id": "not-a-uuid",
            "note_state": json.dumps(note),
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    assert_error_response(r, 400)


def test_oversized_content_length(service_url: str) -> None:
    parsed = urlparse(service_url)
    host = parsed.hostname or "127.0.0.1"
    scheme = (parsed.scheme or "http").lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    boundary = "----OversizedBoundary"
    body = f"--{boundary}--\r\n".encode("ascii")
    head = (
        f"POST /api/v1/voice/process HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
        f"Content-Length: 12000000\r\n"
        f"Connection: close\r\n"
        "\r\n"
    ).encode("ascii")
    payload = head + body

    if scheme == "https":
        import ssl

        ctx = ssl.create_default_context()
        conn = ctx.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM), server_hostname=host)
        conn.connect((host, port))
    else:
        conn = socket.create_connection((host, port), timeout=10.0)
    try:
        conn.sendall(payload)
        buf = conn.recv(65536)
    finally:
        conn.close()

    first = buf.split(b"\r\n", 1)[0].decode("latin1", errors="replace")
    assert "413" in first, first
    # Optional: body contains JSON error
    if b"{" in buf:
        lower = buf.decode("latin1", errors="replace").lower()
        assert "payload" in lower or "large" in lower or "error" in lower


def run_tests(use_mock: bool) -> int:
    audio_path = mock_audio.create_test_audio(prefer_ffmpeg=False)
    try:
        tests: List[Tuple[str, Callable[[], None]]] = []

        with httpx.Client() as client:
            _connect_or_fail(client)

            tests.append(("health", lambda: test_health(use_mock, client)))

            if use_mock:
                tests.append(("note_append", lambda: test_note_update_append(audio_path, client)))
                tests.append(("note_insert_cursor", lambda: test_note_insert_at_cursor(audio_path, client)))
                tests.append(("stack_add_row", lambda: test_stack_add_row(audio_path, client)))
                tests.append(("none_action", lambda: test_none_action(audio_path, client)))

            tests.append(("invalid_mime", lambda: test_invalid_audio_mime(audio_path, client)))
            tests.append(("missing_note_state", lambda: test_missing_note_state(audio_path, client)))
            tests.append(("missing_context_type", lambda: test_missing_context_type(audio_path, client)))
            tests.append(("invalid_context_type", lambda: test_invalid_context_type(audio_path, client)))
            tests.append(("invalid_context_uuid", lambda: test_invalid_context_id_uuid(audio_path, client)))
            tests.append(("oversized_cl", lambda: test_oversized_content_length(SERVICE_URL)))

            passed = 0
            failed: List[str] = []
            for name, fn in tests:
                try:
                    fn()
                    print(f"PASS  {name}")
                    passed += 1
                except AssertionError as exc:
                    print(f"FAIL  {name}: {exc}")
                    failed.append(name)
                except Exception as exc:
                    print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
                    failed.append(name)

            print("-" * 50)
            print(f"Passed {passed}/{len(tests)}")
            if failed:
                print("Failed:", ", ".join(failed))
                return 1
            return 0
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice AI microservice contract tests")
    mx = parser.add_mutually_exclusive_group()
    mx.add_argument(
        "--mock-openai",
        action="store_true",
        help="Expect server with MOCK_OPENAI=1 (default)",
    )
    mx.add_argument(
        "--real-openai",
        action="store_true",
        help="Server without mock; skips deterministic LLM cases",
    )
    args = parser.parse_args()
    use_mock = not args.real_openai
    if not args.mock_openai and not args.real_openai:
        use_mock = True

    global SERVICE_URL
    SERVICE_URL = os.getenv("SERVICE_URL", "http://127.0.0.1:8000").rstrip("/")

    print(f"SERVICE_URL={SERVICE_URL}  mock_mode={use_mock}")
    code = run_tests(use_mock)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
