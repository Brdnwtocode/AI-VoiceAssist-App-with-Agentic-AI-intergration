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

from . import mock_audio

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


def test_note_update_append_transcript_only(client: httpx.Client) -> None:
    note = _note_state()
    headers = _mock_headers(
        "Append meeting notes to the end (dictated).",
        "update_note",
        {"content_to_insert": "Meeting notes from today", "action_type": "append"},
    )
    data = {
        "transcript": "Append meeting notes to the end (dictated).",
        "context_type": "NOTE",
        "context_id": NOTE_ID,
        "cursor_position": "5",
        "note_state": json.dumps(note),
    }
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data=data,
        headers=headers,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "update_note"
    assert body["transcript"] == "Append meeting notes to the end (dictated)."


def test_missing_audio_and_transcript(client: httpx.Client) -> None:
    note = _note_state()
    headers = _mock_headers("x", "no_action", {})
    data = {
        "context_type": "NOTE",
        "context_id": NOTE_ID,
        "note_state": json.dumps(note),
    }
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data=data,
        headers=headers,
        timeout=60.0,
    )
    assert_error_response(r, 422)
    assert "transcript" in r.json()["error"].lower() or "audio" in r.json()["error"].lower()


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


def test_hallucinated_update_note(audio_path: str, client: httpx.Client) -> None:
    """Verify the server does not crash when the LLM hallucinates update_note in STACK context."""
    schema = [{"id": "col1", "name": "Task", "type": "TEXT"}]
    headers = _mock_headers(
        "Add a task called review.",
        "update_note",  # hallucinated — wrong tool for STACK context
        {"content_to_insert": "review", "action_type": "append"},
    )
    headers["X-Mock-Hallucinate-Update-Note"] = "1"
    with open(audio_path, "rb") as f:
        files = {"audio": ("t.webm", f, "audio/webm")}
        data = {
            "context_type": "STACK",
            "context_id": STACK_ID,
            "dynamic_schema": json.dumps(schema),
            # note_state intentionally absent — STACK context, no open note
        }
        r = client.post(
            f"{SERVICE_URL}/api/v1/voice/process",
            files=files,
            data=data,
            headers=headers,
            timeout=60.0,
        )
    # Must not 500. Must gracefully return none with a conversational reply.
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    body = r.json()
    assert body["success"] is True
    assert body["action"] == "none"
    assert body["updatedData"] is None
    assert body["reply"] is not None and len(body["reply"]) > 0


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


def test_packed_context_note_update(client: httpx.Client) -> None:
    packed = {
        "items": [
            {
                "type": "NOTE",
                "id": NOTE_ID,
                "title": "My Note",
                "content": "0123456789",
                "metadata": {
                    "lastUpdated": "2026-06-02T10:30:00Z"
                },
                "source": "active_tab"
            }
        ],
        "packedAt": "2026-06-02T10:30:05Z"
    }
    headers = _mock_headers(
        "Append XX to the note.",
        "update_note",
        {"content_to_insert": "XX", "action_type": "append"},
    )
    data = {
        "packed_context": json.dumps(packed),
        "transcript": "Append XX to the note."
    }
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data=data,
        headers=headers,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "update_note"
    assert body["updatedData"]["content"] == "0123456789\nXX"
    assert body["updatedData"]["id"] == NOTE_ID


def test_packed_context_stack_add_row(client: httpx.Client) -> None:
    schema = [
        {"id": "col1", "name": "Task Name", "type": "TEXT"},
        {"id": "col2", "name": "Priority", "type": "INT"},
    ]
    packed = {
        "items": [
            {
                "type": "STACK",
                "id": STACK_ID,
                "title": "Tasks Stack",
                "content": json.dumps({"name": "Tasks Stack", "columns": schema, "rowCount": 2}),
                "metadata": {},
                "source": "active_tab"
            }
        ],
        "packedAt": "2026-06-02T10:30:05Z"
    }
    headers = _mock_headers(
        "Add stack row: buy groceries.",
        "add_stack_row",
        {
            "Task Name": "Buy groceries",
            "Priority": 2,
        },
    )
    data = {
        "packed_context": json.dumps(packed),
        "transcript": "Add stack row: buy groceries."
    }
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data=data,
        headers=headers,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "add_stack_row"
    ud = body["updatedData"]
    assert ud["stackId"] == STACK_ID
    assert ud["id"].startswith("temp_row_")
    d = ud["data"]
    assert d["col1"] == "Buy groceries"
    assert d["col2"] == 2


def test_packed_context_multicontext_validation(client: httpx.Client) -> None:
    schema = [
        {"id": "col1", "name": "Task Name", "type": "TEXT"},
    ]
    packed = {
        "items": [
            {
                "type": "NOTE",
                "id": NOTE_ID,
                "title": "My Note",
                "content": "0123456789",
                "source": "active_tab"
            },
            {
                "type": "STACK",
                "id": STACK_ID,
                "title": "Tasks Stack",
                "content": json.dumps({"name": "Tasks Stack", "columns": schema, "rowCount": 2}),
                "source": "user_mention"
            }
        ]
    }
    headers = _mock_headers(
        "Add stack row: buy groceries.",
        "add_stack_row",
        {
            "Task Name": "Buy groceries",
        },
    )
    data = {
        "packed_context": json.dumps(packed),
        "transcript": "Add stack row: buy groceries."
    }
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data=data,
        headers=headers,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "add_stack_row"
    ud = body["updatedData"]
    assert ud["stackId"] == STACK_ID
    assert ud["data"]["col1"] == "Buy groceries"


def test_packed_context_missing_or_invalid(client: httpx.Client) -> None:
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": "{invalid", "transcript": "Hello"},
    )
    assert r.status_code == 400
    assert "Invalid packed_context JSON" in r.text

    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps({"items": []}), "transcript": "Hello"},
    )
    assert r.status_code == 400
    assert "packed_context must contain at least one item" in r.text

    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"transcript": "Hello"},
    )
    assert r.status_code == 400
    assert "Invalid request" in r.text


def test_summarize_context(client: httpx.Client) -> None:
    headers = _mock_headers(
        "Summarize my workspace.",
        "summarize_context",
        {"summary": "You have a note about cooking and a stack with active rows."},
    )
    packed = {
        "items": [
            {
                "type": "NOTE",
                "id": NOTE_ID,
                "title": "Cooking",
                "content": "Cooking recipes",
            }
        ]
    }
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Summarize my workspace."},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["action"] == "summarize_context"
    assert body["reply"] == "You have a note about cooking and a stack with active rows."
    assert body["updatedData"] is None


def test_bulk_update_stack(client: httpx.Client) -> None:
    schema = [
        {"id": "col-status", "name": "Status", "type": "SELECT"},
        {"id": "col-due", "name": "Due Date", "type": "DATE"},
    ]
    packed = {
        "items": [
            {
                "type": "STACK",
                "id": STACK_ID,
                "title": "My Stack",
                "content": json.dumps({"name": "My Stack", "columns": schema, "rows": []}),
            }
        ]
    }
    headers = _mock_headers(
        "Complete all stack items.",
        "bulk_update_stack",
        {
            "stack_id": STACK_ID,
            "updates": [
                {"row_id": "row-1", "column_values": {"Status": "DONE"}},
                {"row_id": "row-2", "column_values": {"Status": "DONE", "Due Date": "2026-06-02"}},
            ],
        },
    )
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Complete all stack items."},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["action"] == "bulk_update_stack"
    ud = body["updatedData"]
    assert ud["stackId"] == STACK_ID
    assert len(ud["updates"]) == 2
    assert ud["updates"][0] == {"rowId": "row-1", "data": {"col-status": "DONE"}}
    assert ud["updates"][1] == {"rowId": "row-2", "data": {"col-status": "DONE", "col-due": "2026-06-02"}}


def test_manage_tasks_actions(client: httpx.Client) -> None:
    # 1. Create task
    headers = _mock_headers(
        "Create task.",
        "manage_tasks",
        {"action_type": "create", "title": "Buy milk"},
    )
    packed = {"items": [{"type": "TASKS", "id": NOTE_ID}]}
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Create task."},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "manage_tasks"
    assert body["updatedData"]["action_type"] == "create"
    assert body["updatedData"]["title"] == "Buy milk"

    # 2. Update task
    headers = _mock_headers(
        "Complete task.",
        "manage_tasks",
        {"action_type": "update", "task_id": "task-1", "status": "DONE"},
    )
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Complete task."},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "manage_tasks"
    assert body["updatedData"]["action_type"] == "update"
    assert body["updatedData"]["status"] == "DONE"

    # 3. Delete task
    headers = _mock_headers(
        "Delete task.",
        "manage_tasks",
        {"action_type": "delete", "task_id": "task-1"},
    )
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Delete task."},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "manage_tasks"
    assert body["updatedData"]["action_type"] == "delete"
    assert body["updatedData"]["task_id"] == "task-1"


def test_context_guidance_fallback(client: httpx.Client) -> None:
    headers = _mock_headers(
        "Edit stack.",
        "no_action",
        {"reply": "Please select the tabs or use @mentions in the text to add the relevant material to my context."},
    )
    packed = {
        "items": [
            {
                "type": "NOTE",
                "id": NOTE_ID,
                "title": "Cooking",
                "content": "Cooking recipes",
            }
        ]
    }
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Edit stack."},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "none"
    assert "Please select the tabs" in body["reply"]


def test_empty_context_fallback_dummy_uuid(client: httpx.Client) -> None:
    headers = _mock_headers(
        "Edit stack.",
        "update_note",
        {
            "content_to_insert": "Buy groceries",
            "action_type": "append",
        },
    )
    packed = {
        "items": [
            {
                "type": "NOTE",
                "id": "00000000-0000-0000-0000-000000000000",
                "title": "No active context",
                "content": "",
            }
        ]
    }
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Edit stack."},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] == "none"
    assert "Please select the tabs" in body["reply"]


# ── Context-Grabber Interface v2 Tests ──────────────────────────────────

def test_update_cell_precision_edit(client: httpx.Client) -> None:
    """Test single-cell precision edit via update_cell action."""
    schema = [
        {"id": "col-name", "name": "Name", "type": "TEXT"},
        {"id": "col-revenue", "name": "Revenue", "type": "INT"},
    ]
    packed = {
        "items": [
            {
                "type": "STACK",
                "id": STACK_ID,
                "title": "Companies",
                "content": json.dumps({
                    "schema": {"columns": schema},
                    "stats": {"rowCount": 14, "columnCount": 2},
                    "focusedTarget": {
                        "rowId": "row_7",
                        "columnId": "col-revenue",
                        "currentValue": 50000,
                        "rowIndex": 6,
                        "columnIndex": 1,
                    },
                }),
                "metadata": {
                    "commandType": "single_edit",
                    "editMode": "single_cell",
                },
            }
        ],
        "packedAt": "2026-06-06T10:00:00Z",
    }
    headers = _mock_headers(
        "Update this cell to 75000",
        "update_cell",
        {
            "stack_id": STACK_ID,
            "row_id": "row_7",
            "column_id": "col-revenue",
            "value": 75000,
        },
    )
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Update this cell to 75000"},
        headers=headers,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "update_cell"
    ud = body["updatedData"]
    assert ud["stackId"] == STACK_ID
    assert ud["rowId"] == "row_7"
    assert ud["columnId"] == "col-revenue"
    assert ud["value"] == 75000


def test_delete_row(client: httpx.Client) -> None:
    """Test row deletion via delete_row action."""
    schema = [
        {"id": "col-name", "name": "Name", "type": "TEXT"},
    ]
    packed = {
        "items": [
            {
                "type": "STACK",
                "id": STACK_ID,
                "title": "Companies",
                "content": json.dumps({
                    "schema": {"columns": schema},
                    "stats": {"rowCount": 14, "columnCount": 1},
                    "dataFormat": "csv",
                    "data": "id,Name\nrow_1,Acme\nrow_2,Globex\n",
                }),
                "metadata": {
                    "commandType": "summarize",
                    "editMode": "full_data",
                },
            }
        ],
    }
    headers = _mock_headers(
        "Delete row row_1",
        "delete_row",
        {"stack_id": STACK_ID, "row_id": "row_1"},
    )
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Delete row row_1"},
        headers=headers,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "delete_row"
    ud = body["updatedData"]
    assert ud["stackId"] == STACK_ID
    assert ud["rowId"] == "row_1"


def test_packed_context_v2_schema_format(client: httpx.Client) -> None:
    """Test that STACK schema is correctly extracted from v2 content.schema.columns format."""
    schema = [
        {"id": "col-a", "name": "Company", "type": "TEXT"},
        {"id": "col-b", "name": "Value", "type": "INT"},
    ]
    packed = {
        "items": [
            {
                "type": "STACK",
                "id": STACK_ID,
                "title": "V2 Stack",
                # v2 format: content.schema.columns (not content.columns)
                "content": json.dumps({
                    "schema": {"columns": schema},
                    "stats": {"rowCount": 5, "columnCount": 2},
                }),
                "metadata": {"editMode": "schema_only"},
            }
        ],
    }
    headers = _mock_headers(
        "Add row for Acme with value 100",
        "add_stack_row",
        {"Company": "Acme", "Value": 100},
    )
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Add row for Acme with value 100"},
        headers=headers,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "add_stack_row"
    ud = body["updatedData"]
    assert ud["stackId"] == STACK_ID
    # Column names are mapped to column IDs via schema
    assert ud["data"]["col-a"] == "Acme"
    assert ud["data"]["col-b"] == 100


def test_packed_context_precision_mode_detection(client: httpx.Client) -> None:
    """Test that precision edit mode is properly detected and routed."""
    schema = [
        {"id": "col-x", "name": "Status", "type": "SELECT"},
    ]
    packed = {
        "items": [
            {
                "type": "STACK",
                "id": STACK_ID,
                "title": "Tasks",
                "content": json.dumps({
                    "schema": {"columns": schema},
                    "focusedTarget": {
                        "rowId": "row_3",
                        "columnId": "col-x",
                        "currentValue": "TODO",
                        "rowIndex": 2,
                        "columnIndex": 0,
                    },
                }),
                "metadata": {"editMode": "single_cell"},
            }
        ],
    }
    # Use bulk_update_stack as tool but verify update_cell is the expected path for single_cell
    headers = _mock_headers(
        "Mark this as DONE",
        "update_cell",
        {"stack_id": STACK_ID, "row_id": "row_3", "column_id": "col-x", "value": "DONE"},
    )
    r = client.post(
        f"{SERVICE_URL}/api/v1/voice/process",
        data={"packed_context": json.dumps(packed), "transcript": "Mark this as DONE"},
        headers=headers,
        timeout=60.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert_success_contract(body)
    assert body["action"] == "update_cell"
    assert body["updatedData"]["value"] == "DONE"


def run_tests(use_mock: bool) -> int:
    audio_path = mock_audio.create_test_audio(prefer_ffmpeg=False)
    try:
        tests: List[Tuple[str, Callable[[], None]]] = []

        with httpx.Client() as client:
            _connect_or_fail(client)

            tests.append(("health", lambda: test_health(use_mock, client)))

            if use_mock:
                tests.append(
                    ("note_append_transcript", lambda: test_note_update_append_transcript_only(client)),
                )
                tests.append(("note_append", lambda: test_note_update_append(audio_path, client)))
                tests.append(("note_insert_cursor", lambda: test_note_insert_at_cursor(audio_path, client)))
                tests.append(("stack_add_row", lambda: test_stack_add_row(audio_path, client)))
                tests.append(("none_action", lambda: test_none_action(audio_path, client)))
                tests.append(
                    ("hallucinated_update_note", lambda: test_hallucinated_update_note(audio_path, client)),
                )
                tests.append(("packed_context_note_update", lambda: test_packed_context_note_update(client)))
                tests.append(("packed_context_stack_add_row", lambda: test_packed_context_stack_add_row(client)))
                tests.append(("packed_context_multicontext_validation", lambda: test_packed_context_multicontext_validation(client)))
                tests.append(("packed_context_missing_or_invalid", lambda: test_packed_context_missing_or_invalid(client)))
                tests.append(("summarize_context", lambda: test_summarize_context(client)))
                tests.append(("bulk_update_stack", lambda: test_bulk_update_stack(client)))
                tests.append(("manage_tasks_actions", lambda: test_manage_tasks_actions(client)))
                tests.append(("context_guidance_fallback", lambda: test_context_guidance_fallback(client)))
                tests.append(("empty_context_fallback_dummy_uuid", lambda: test_empty_context_fallback_dummy_uuid(client)))
                # Context-Grabber v2 tests
                tests.append(("update_cell_precision_edit", lambda: test_update_cell_precision_edit(client)))
                tests.append(("delete_row", lambda: test_delete_row(client)))
                tests.append(("packed_context_v2_schema", lambda: test_packed_context_v2_schema_format(client)))
                tests.append(("precision_mode_detection", lambda: test_packed_context_precision_mode_detection(client)))

            tests.append(("missing_audio_transcript", lambda: test_missing_audio_and_transcript(client)))
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
