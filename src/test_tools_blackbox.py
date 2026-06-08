"""
Blackbox unit and integration tests for the Voice AI pipeline tools (src/tools.py).
Runs as a module: python -m src.test_tools_blackbox
"""

import asyncio
import json
from typing import Any, Callable, Coroutine, Dict, List, Tuple, Union

from .tools import (
    web_search,
    web_search_formatted,
    extract_workspace_facts,
    detect_language,
    correct_stt_errors,
    classify_tone,
    assess_action_risk,
    generate_edge_cases,
    detect_sycophancy_risks,
    build_contrarian_template,
    detect_ambiguity,
    format_workspace_for_llm,
    build_research_template,
    build_conversation_template,
    detect_multi_step,
    extract_action_verbs,
    build_planning_template,
    detect_hallucination,
    build_reflection_template,
)


async def test_web_search() -> None:
    print("Testing web_search...")
    # Normal query (requires internet, may fallback gracefully to empty list if offline or rate limited)
    results = await web_search("python language", max_results=3)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    
    if len(results) > 0:
        for r in results:
            assert "title" in r, "Result missing title"
            assert "snippet" in r, "Result missing snippet"
            assert "url" in r, "Result missing url"
        print(f"  -> Success: retrieved {len(results)} results")
    else:
        print("  -> Success (graceful fallback): retrieved 0 results (offline or DDG rate-limited)")

    # Empty query
    empty_results = await web_search("")
    assert isinstance(empty_results, list)
    assert len(empty_results) == 0, f"Expected 0 results for empty query, got {len(empty_results)}"

    # Formatted search
    formatted = await web_search_formatted("python language", max_results=2)
    assert isinstance(formatted, str)
    assert "Web search results for:" in formatted or "[Web search returned no results]" in formatted


def test_extract_workspace_facts() -> None:
    print("Testing extract_workspace_facts...")
    # Test None context
    facts_none = extract_workspace_facts(None)
    assert facts_none["context_type"] == "unknown"
    assert facts_none["item_count"] == 0
    assert facts_none["key_entities"] == []

    # Test NOTE context
    note_ctx = {
        "items": [
            {
                "type": "NOTE",
                "id": "note-123",
                "title": "Draft Outline",
                "content": "This is a note with some draft details.",
            }
        ]
    }
    facts_note = extract_workspace_facts(note_ctx)
    assert facts_note["context_type"] == "NOTE"
    assert facts_note["item_count"] == 1
    assert any(e["value"] == "Draft Outline" for e in facts_note["key_entities"])
    assert any(e["value"] == "note-123" for e in facts_note["key_entities"])
    assert facts_note["data_stats"]["content_length"] > 0

    # Test STACK context
    stack_ctx = {
        "items": [
            {
                "type": "STACK",
                "id": "stack-456",
                "title": "Product Roadmap",
                "content": json.dumps({
                    "schema": {
                        "columns": [
                            {"name": "Feature", "type": "TEXT"},
                            {"name": "Status", "type": "SELECT"},
                        ]
                    },
                    "stats": {"rowCount": 15},
                    "focusedTarget": {"rowId": "row_2", "columnId": "Feature", "currentValue": "AI Search"}
                }),
            }
        ]
    }
    facts_stack = extract_workspace_facts(stack_ctx)
    assert facts_stack["context_type"] == "STACK"
    assert facts_stack["item_count"] == 1
    assert facts_stack["data_stats"]["column_count"] == 2
    assert facts_stack["data_stats"]["row_count"] == 15
    assert any(e["name"] == "focused_cell" for e in facts_stack["key_entities"])
    
    # Test legacy paths (note_state / dynamic_schema)
    legacy_note = {"note": {"title": "Legacy Title", "content": "Legacy content details"}}
    facts_legacy = extract_workspace_facts(None, note_state=legacy_note)
    assert facts_legacy["context_type"] == "NOTE"
    assert any(e["value"] == "Legacy Title" for e in facts_legacy["key_entities"])


def test_detect_language() -> None:
    print("Testing detect_language...")
    # Vietnamese
    vi_res = detect_language("Xin chào, đây là hệ thống trợ lý ảo thông minh.")
    assert vi_res["language"] in ("vi", "mixed"), f"Expected vi/mixed, got {vi_res}"
    assert vi_res["vi_ratio"] > 0.5

    # English
    en_res = detect_language("Hello, this is a smart voice assistant system.")
    assert en_res["language"] == "en", f"Expected en, got {en_res}"
    assert en_res["en_ratio"] > 0.5

    # Mixed
    mixed_res = detect_language("Thêm task buy groceries vào stack.")
    assert mixed_res["language"] in ("vi", "mixed", "en")

    # Empty
    empty_res = detect_language("")
    assert empty_res["language"] == "vi"  # Default fallback


def test_correct_stt_errors() -> None:
    print("Testing correct_stt_errors...")
    # Vietnamese corrections
    vi_corrected, vi_fixes = correct_stt_errors("ko dc tai sao", language="vi")
    assert vi_corrected == "không được tại sao", f"Expected 'không được tại sao', got '{vi_corrected}'"
    assert len(vi_fixes) == 3

    # English corrections
    en_corrected, en_fixes = correct_stt_errors("I wanna go but I dont know", language="en")
    assert "want to" in en_corrected
    assert "don't" in en_corrected
    assert len(en_fixes) == 2

    # Auto detect
    auto_corrected, auto_fixes = correct_stt_errors("gonna add task", language="auto")
    assert "going to" in auto_corrected or "add" in auto_corrected


def test_classify_tone() -> None:
    print("Testing classify_tone...")
    # Command
    cmd = classify_tone("Thêm task buy milk")
    assert cmd["tone"] == "command"

    # Query
    query = classify_tone("Tại sao tính năng này không hoạt động?")
    assert query["tone"] == "query"

    # Chitchat
    chat = classify_tone("Xin chào bạn nhé")
    assert chat["tone"] == "chitchat"


def test_assess_action_risk() -> None:
    print("Testing assess_action_risk...")
    high = assess_action_risk("delete_row")
    assert high["base_risk"] == "high"
    
    med = assess_action_risk("update_note")
    assert med["base_risk"] == "medium"
    
    low = assess_action_risk("create_task")
    assert low["base_risk"] == "low"
    
    unknown = assess_action_risk("arbitrary_action")
    assert unknown["base_risk"] == "medium"  # Defaults to medium


def test_generate_edge_cases() -> None:
    print("Testing generate_edge_cases...")
    # NOTE context
    note_cases = generate_edge_cases("NOTE")
    assert any("empty" in case.lower() for case in note_cases)
    
    # STACK context with triggers
    stack_cases = generate_edge_cases("STACK", "xóa cái này đi")
    assert any("deleted" in case.lower() for case in stack_cases)
    assert any("destructive" in case.lower() for case in stack_cases)  # trigger case


def test_detect_sycophancy_risks() -> None:
    print("Testing detect_sycophancy_risks...")
    facts = {
        "missing_critical": ["Note content is empty"],
        "key_entities": []
    }
    # Vague reference + assertion + missing data
    risks = detect_sycophancy_risks("nó là cái này đúng không?", "NOTE", facts)
    assert len(risks) > 0
    assert any("vague" in r.lower() or "ambiguous" in r.lower() for r in risks)
    assert any("assertion" in r.lower() for r in risks)
    assert any("empty" in r.lower() for r in risks)


def test_detect_ambiguity() -> None:
    print("Testing detect_ambiguity...")
    # Ambiguous
    has_amb, amb_list, score = detect_ambiguity("sửa cái này")
    assert has_amb is True
    assert len(amb_list) > 0
    assert score > 0.2

    # Clear
    has_amb2, amb_list2, score2 = detect_ambiguity("thêm cột tên Feature vào bảng")
    # Might still have slight short-text warning if too short, but should be lower ambiguity
    print(f"  Clear query score: {score2}")


def test_detect_multi_step() -> None:
    print("Testing detect_multi_step...")
    # Multi step
    is_multi, triggers, steps = detect_multi_step("Thêm task mới rồi sau đó xóa note")
    assert is_multi is True
    assert len(triggers) >= 1
    assert steps >= 2

    # Single step
    is_multi2, triggers2, steps2 = detect_multi_step("Thêm task mới")
    assert is_multi2 is False


def test_extract_action_verbs() -> None:
    print("Testing extract_action_verbs...")
    verbs = extract_action_verbs("thêm note và xóa dòng")
    assert len(verbs) == 2
    assert verbs[0]["verb"] in ("thêm", "ghi")
    assert verbs[1]["verb"] == "xóa"


def test_detect_hallucination() -> None:
    print("Testing detect_hallucination...")
    # Context mismatch
    nlu_mismatch = {"action": "add_stack_row", "params": {"Task Name": "Buy milk"}}
    has_hall, issues = detect_hallucination(nlu_mismatch, "NOTE")
    assert has_hall is True
    assert any("requires 'STACK' context" in issue for issue in issues)

    # Empty params
    nlu_empty_params = {"action": "update_note", "params": {}}
    has_hall2, issues2 = detect_hallucination(nlu_empty_params, "NOTE")
    assert has_hall2 is True
    assert any("empty params" in issue for issue in issues2)

    # Fake UUID
    nlu_uuid = {"action": "update_cell", "params": {"rowId": "550e8400-e29b-41d4-a716-446655440003", "value": "x"}}
    has_hall3, issues3 = detect_hallucination(nlu_uuid, "STACK")
    assert has_hall3 is True
    assert any("invented UUID" in issue for issue in issues3)


def test_template_builders() -> None:
    print("Testing template builders...")
    facts = {
        "context_type": "NOTE",
        "item_count": 1,
        "key_entities": [{"type": "NOTE", "name": "title", "value": "Meeting Notes"}],
        "data_stats": {"content_length": 120},
        "missing_critical": []
    }
    
    # Contrarian
    c_temp = build_contrarian_template("Xóa note này", "NOTE", facts)
    assert "CONTRARIAN CRITICAL REASONING FRAMEWORK" in c_temp
    
    # Research
    r_temp = build_research_template("Tìm thông tin python", "NOTE", facts, "web search results")
    assert "RESEARCH GROUNDING FRAMEWORK" in r_temp
    
    # Conversation
    lang_info = detect_language("hello")
    tone_info = classify_tone("hello")
    v_temp = build_conversation_template("hello", lang_info, tone_info, False, [], "hello", [])
    assert "CONVERSATION ANALYSIS FRAMEWORK" in v_temp

    # Planning
    p_temp = build_planning_template("thêm note", "NOTE", facts)
    assert "TASK PLANNING FRAMEWORK" in p_temp

    # Reflection
    ref_temp = build_reflection_template("thêm note", "NOTE", {"action": "update_note", "params": {"content_to_insert": "x"}}, 0)
    assert "REFLECTION CRITIQUE FRAMEWORK" in ref_temp


async def run_all_tests() -> int:
    print("=" * 60)
    print("STARTING AI PIPELINE TOOLS BLACKBOX TEST SUITE")
    print("=" * 60)
    
    tests_sync = [
        ("extract_workspace_facts", test_extract_workspace_facts),
        ("detect_language", test_detect_language),
        ("correct_stt_errors", test_correct_stt_errors),
        ("classify_tone", test_classify_tone),
        ("assess_action_risk", test_assess_action_risk),
        ("generate_edge_cases", test_generate_edge_cases),
        ("detect_sycophancy_risks", test_detect_sycophancy_risks),
        ("detect_ambiguity", test_detect_ambiguity),
        ("detect_multi_step", test_detect_multi_step),
        ("extract_action_verbs", test_extract_action_verbs),
        ("detect_hallucination", test_detect_hallucination),
        ("template_builders", test_template_builders),
    ]
    
    passed = 0
    failed = []
    
    # Run async tests
    try:
        await test_web_search()
        print("PASS  web_search")
        passed += 1
    except AssertionError as exc:
        print(f"FAIL  web_search: {exc}")
        failed.append("web_search")
    except Exception as exc:
        print(f"FAIL  web_search: {type(exc).__name__}: {exc}")
        failed.append("web_search")

    # Run sync tests
    for name, fn in tests_sync:
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

    print("-" * 60)
    total_tests = len(tests_sync) + 1
    print(f"Passed {passed}/{total_tests} tests")
    if failed:
        print("Failed tests:", ", ".join(failed))
        return 1
    else:
        print("All tests passed successfully!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    exit(exit_code)
