"""
Expert Tools — functional capabilities backing the multi-expert agents.

Each tool is a concrete, callable function (not just a prompt) that
gives experts real capabilities:
- Research Expert: web search, workspace data extraction, fact verification
- Contrarian Expert: risk matrix, edge-case enumeration, sycophancy detection
- Conversation Expert: language detection, tone classification, STT correction
"""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from .config import logger


# ═══════════════════════════════════════════════════════════════════════════
# Web Search Tool (Research Expert)
# ═══════════════════════════════════════════════════════════════════════════

# Patterns stripped from transcripts when building a search query:
# trigger mentions, action clauses ("and add it to the note"), politeness.
_QUERY_NOISE_PATTERNS = [
    r"@\w+[,:]?\s*",                                            # @Maximus mention
    r"\b(and|then|và|rồi|sau đó)\s+(add|insert|put|write|save|append|ghi|thêm|viết|lưu|chèn)\b.*$",  # trailing action clause
    r"\b(add|insert|put|write|save|append)\s+(it|this|that|the results?|them)\s+(in|into|to|on)\b.*$",
    r"\b(ghi|thêm|viết|lưu|chèn)\s+(nó|cái này|kết quả)?\s*(vào|lên)\b.*$",
    r"^\s*(please|hãy|làm ơn|giúp tôi|giúp mình|can you|could you)\s+",
    r"^\s*(research|search|look up|find out|tìm hiểu|tra cứu|tìm kiếm|nghiên cứu)\s+(about|on|for|về|thông tin về)?\s*",
]


def build_search_query(transcript: str, max_len: int = 120) -> str:
    """Distill a raw voice transcript into a clean web-search query.

    Removes trigger mentions (@Maximus), trailing action clauses
    ("...and add it to the note"), politeness prefixes, and research verbs,
    leaving just the topic. Falls back to the cleaned transcript if the
    result would be empty.
    """
    q = transcript.strip()
    for pattern in _QUERY_NOISE_PATTERNS:
        q = re.sub(pattern, " ", q, flags=re.IGNORECASE).strip()
    q = re.sub(r"\s{2,}", " ", q).strip(" ,.;:!?")
    if len(q) < 3:  # stripped too much — use de-mentioned transcript
        q = re.sub(r"@\w+[,:]?\s*", "", transcript).strip()
    return q[:max_len]


async def web_search(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
) -> List[Dict[str, str]]:
    """Search the web. Primary: `ddgs` library (robust DuckDuckGo client).
    Fallbacks: DuckDuckGo HTML scrape → DuckDuckGo Lite → Wikipedia REST API.

    Returns a list of dicts: {"title": ..., "snippet": ..., "url": ...}
    No API key required. Falls back gracefully on network errors.
    """
    results: List[Dict[str, str]] = []
    if not query or not query.strip():
        return results

    # ── Primary: ddgs library (maintained DuckDuckGo client) ──
    try:
        from ddgs import DDGS

        def _ddgs_search() -> List[Dict[str, str]]:
            out = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, region=region, max_results=max_results):
                    out.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    })
            return out

        results = await asyncio.to_thread(_ddgs_search)
        if results:
            logger.info("Web search (ddgs) returned %d results for: %s", len(results), query[:80])
            return results
    except ImportError:
        logger.debug("ddgs library not installed — falling back to HTML scrape")
    except Exception as exc:
        logger.warning("ddgs search failed: %s — falling back", exc)

    # ── Fallback 1: DuckDuckGo HTML scrape ──
    try:
        import httpx

        async with httpx.AsyncClient(timeout=8.0) as client:
            # Try the HTML search first (more reliable than instant answer)
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query, "kl": region},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                follow_redirects=True,
            )

            if resp.status_code == 200:
                # Parse HTML results with regex (avoid BeautifulSoup dependency)
                html = resp.text
                # Extract result blocks: each has a "result__snippet" and "result__url"
                snippet_pattern = re.compile(
                    r'<a[^>]*class="result__snippet"[^>]*>\s*(.*?)\s*</a>',
                    re.DOTALL,
                )
                url_pattern = re.compile(
                    r'<a[^>]*class="result__url"[^>]*>\s*(.*?)\s*</a>',
                    re.DOTALL,
                )
                title_pattern = re.compile(
                    r'<a[^>]*class="result__a"[^>]*>\s*(.*?)\s*</a>',
                    re.DOTALL,
                )

                snippets = [re.sub(r"<[^>]+>", "", s).strip() for s in snippet_pattern.findall(html)]
                urls = [re.sub(r"<[^>]+>", "", u).strip() for u in url_pattern.findall(html)]
                titles = [re.sub(r"<[^>]+>", "", t).strip() for t in title_pattern.findall(html)]

                for i in range(min(len(titles), max_results)):
                    results.append({
                        "title": titles[i] if i < len(titles) else "",
                        "snippet": snippets[i] if i < len(snippets) else "",
                        "url": urls[i] if i < len(urls) else "",
                    })

                if results:
                    logger.info("Web search returned %d results for: %s", len(results), query[:80])
                    return results
    except ImportError:
        logger.warning("httpx not installed — web search disabled")
    except Exception as exc:
        logger.warning("DuckDuckGo HTML search failed: %s", exc)

    # ── Fallback: try DuckDuckGo Lite ──
    try:
        import httpx

        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                follow_redirects=True,
            )
            if resp.status_code == 200:
                # Lite returns simple HTML with <a> links
                link_pattern = re.compile(
                    r'<a[^>]*href="([^"]*uddg=([^"&]*)[^"]*)"[^>]*>(.*?)</a>',
                    re.DOTALL,
                )
                matches = link_pattern.findall(resp.text)
                for i, match in enumerate(matches[:max_results]):
                    try:
                        from urllib.parse import unquote
                        url = unquote(match[1])
                    except Exception:
                        url = match[0]
                    title = re.sub(r"<[^>]+>", "", match[2]).strip()
                    if title and url:
                        results.append({
                            "title": title,
                            "snippet": "",
                            "url": url,
                        })
                if results:
                    return results
    except Exception as exc:
        logger.warning("DuckDuckGo Lite fallback failed: %s", exc)

    # ── Fallback 3: Wikipedia REST search (very reliable, topical queries) ──
    try:
        import httpx

        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                "https://en.wikipedia.org/w/rest.php/v1/search/page",
                params={"q": query, "limit": max_results},
                headers={"User-Agent": "VoiceAssistResearch/1.0"},
            )
            if resp.status_code == 200:
                for page in resp.json().get("pages", []):
                    desc = page.get("description") or ""
                    excerpt = re.sub(r"<[^>]+>", "", page.get("excerpt") or "")
                    results.append({
                        "title": page.get("title", ""),
                        "snippet": (desc + " — " + excerpt).strip(" —"),
                        "url": f"https://en.wikipedia.org/wiki/{quote_plus(page.get('key', ''))}",
                    })
                if results:
                    logger.info("Web search (wikipedia) returned %d results for: %s", len(results), query[:80])
                    return results
    except Exception as exc:
        logger.warning("Wikipedia fallback failed: %s", exc)

    # ── Last resort: return empty ──
    logger.warning("Web search returned 0 results for: %s", query[:80])
    return results


async def web_search_with_meta(query: str, max_results: int = 5) -> Tuple[str, int, List[str]]:
    """Search the web; return (formatted_text, result_count, source_urls)."""
    results = await web_search(query, max_results)
    if not results:
        return "[Web search returned no results]", 0, []

    lines = [f"Web search results for: \"{query}\"", ""]
    urls: List[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r["snippet"]:
            lines.append(f"   {r['snippet'][:300]}")
        if r["url"]:
            lines.append(f"   URL: {r['url']}")
            urls.append(r["url"])
        lines.append("")
    return "\n".join(lines), len(results), urls


async def web_search_formatted(query: str, max_results: int = 5) -> str:
    """Search the web and return a formatted string for LLM consumption."""
    text, _, _ = await web_search_with_meta(query, max_results)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# Workspace Data Extraction (Research Expert)
# ═══════════════════════════════════════════════════════════════════════════

def extract_workspace_facts(
    processed_context: Optional[dict],
    note_state: Optional[str] = None,
    dynamic_schema: Optional[str] = None,
    task_context_data: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract structured facts from workspace state for the Research expert.

    Returns a dictionary with:
    - context_type: the primary context type
    - item_count: how many items in context
    - key_entities: named entities found (titles, IDs, column names)
    - data_stats: row/column counts, content length
    - missing_critical: what's absent that might be needed
    """
    facts: Dict[str, Any] = {
        "context_type": "unknown",
        "item_count": 0,
        "key_entities": [],
        "data_stats": {},
        "missing_critical": [],
    }

    if processed_context:
        items = processed_context.get("items", [])
        facts["item_count"] = len(items)

        for item in items:
            item_type = item.get("type", "")
            if not facts["context_type"] or facts["context_type"] == "unknown":
                facts["context_type"] = item_type

            # Extract entity names
            title = item.get("title", "")
            if title:
                facts["key_entities"].append({"type": item_type, "name": "title", "value": title})

            item_id = item.get("id", "")
            if item_id:
                facts["key_entities"].append({"type": item_type, "name": "id", "value": item_id})

            # Extract content stats
            content = item.get("content", {})
            content_is_dict = False
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                    content_is_dict = isinstance(content, dict)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(content, dict):
                content_is_dict = True

            # Stack schema
            schema = content.get("schema", {}) if content_is_dict else {}
            if isinstance(schema, dict) and "columns" in schema:
                cols = schema["columns"]
                facts["data_stats"]["column_count"] = len(cols)
                facts["data_stats"]["column_names"] = [c.get("name", "") for c in cols]
                facts["key_entities"].append({
                    "type": item_type,
                    "name": "columns",
                    "value": ", ".join(c.get("name", "") for c in cols),
                })

            # Data stats
            stats = content.get("stats", {}) if content_is_dict else {}
            if stats:
                facts["data_stats"]["row_count"] = stats.get("rowCount", 0)

            # Focused target
            focused = content.get("focusedTarget", {}) if content_is_dict else {}
            if focused:
                facts["key_entities"].append({
                    "type": item_type,
                    "name": "focused_cell",
                    "value": f"row={focused.get('rowId', '')}, col={focused.get('columnId', '')}, current={focused.get('currentValue', '')}",
                })

            # Note content length
            if isinstance(content, str) and len(content) > 10:
                facts["data_stats"]["content_length"] = len(content)
            elif content_is_dict:
                note_content = content.get("content", "")
                if isinstance(note_content, str) and len(note_content) > 10:
                    facts["data_stats"]["content_length"] = len(note_content)

        # Detect missing critical data
        context_type = facts["context_type"]
        if context_type == "NOTE" and facts["data_stats"].get("content_length", 0) == 0:
            facts["missing_critical"].append("Note content is empty or not provided")
        if context_type == "STACK" and "column_count" not in facts.get("data_stats", {}):
            facts["missing_critical"].append("Stack schema (columns) not provided")
        if context_type in ("TASK", "TASKS") and facts["item_count"] == 0:
            facts["missing_critical"].append("No task data in context")

    else:
        # Legacy path
        if note_state:
            try:
                note = json.loads(note_state) if isinstance(note_state, str) else note_state
                facts["context_type"] = "NOTE"
                content = note.get("note", {}).get("content", "") if isinstance(note.get("note"), dict) else ""
                facts["data_stats"]["content_length"] = len(content) if content else 0
                if not content:
                    facts["missing_critical"].append("Note content is empty")
                title = note.get("note", {}).get("title", "") if isinstance(note.get("note"), dict) else ""
                if title:
                    facts["key_entities"].append({"type": "NOTE", "name": "title", "value": title})
            except (json.JSONDecodeError, TypeError):
                pass

        if dynamic_schema:
            try:
                schema = json.loads(dynamic_schema) if isinstance(dynamic_schema, str) else dynamic_schema
                facts["context_type"] = facts["context_type"] or "STACK"
                facts["data_stats"]["column_count"] = len(schema) if isinstance(schema, list) else 0
            except (json.JSONDecodeError, TypeError):
                pass

    return facts


# ═══════════════════════════════════════════════════════════════════════════
# Language Detection (Conversation Expert)
# ═══════════════════════════════════════════════════════════════════════════

# Vietnamese character ranges and common words
_VI_CHARS = set(
    "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ"
)

_VI_COMMON_WORDS = {
    "và", "của", "có", "không", "được", "trong", "cho", "những", "này",
    "một", "các", "khi", "là", "tôi", "với", "để", "đã", "sẽ", "đang",
    "phải", "rất", "nên", "tại", "bị", "vào", "ra", "lên", "nếu", "thì",
    "làm", "nói", "đi", "biết", "thấy", "muốn", "cần", "hay", "cũng",
}

_EN_COMMON_WORDS = {
    "the", "is", "are", "was", "were", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "might", "must", "shall", "this", "that",
    "these", "those", "a", "an", "and", "or", "but", "if", "then", "else",
    "when", "where", "why", "how", "what", "who", "which", "with", "from",
    "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
}


def detect_language(text: str) -> Dict[str, Any]:
    """Detect language of text (vi / en / mixed) with confidence score.

    Uses character-level analysis (Vietnamese diacritics) + word-level heuristics.
    Works on short transcripts (3-50 words) typical of voice commands.

    Returns: {"language": "vi"|"en"|"mixed", "confidence": 0.0-1.0,
              "vi_ratio": float, "en_ratio": float, "method": str}
    """
    if not text or not text.strip():
        return {"language": "vi", "confidence": 0.5, "vi_ratio": 0.5, "en_ratio": 0.5, "method": "empty"}

    text = text.strip()
    words = text.split()

    # ── Method 1: Character-level diacritic detection ──
    vi_char_count = sum(1 for c in text if c in _VI_CHARS)
    total_chars = len(text)
    vi_char_ratio = vi_char_count / max(total_chars, 1)

    # ── Method 2: Word-level dictionary lookup ──
    word_lower = [w.lower().strip(".,!?;:()[]{}\"'") for w in words]
    vi_word_hits = sum(1 for w in word_lower if w in _VI_COMMON_WORDS)
    en_word_hits = sum(1 for w in word_lower if w in _EN_COMMON_WORDS)
    total_known = vi_word_hits + en_word_hits
    vi_word_ratio = vi_word_hits / max(total_known, 1) if total_known > 0 else 0.5
    en_word_ratio = en_word_hits / max(total_known, 1) if total_known > 0 else 0.5

    # ── Method 3: ASCII vs non-ASCII heuristic ──
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_chars / max(total_chars, 1)

    # ── Combine signals ──
    # Strong diacritic signal → Vietnamese
    if vi_char_ratio > 0.05:
        vi_score = min(1.0, vi_char_ratio * 1.5 + vi_word_ratio * 0.3)
        en_score = 1.0 - vi_score
    elif vi_word_hits > en_word_hits:
        vi_score = 0.5 + vi_word_ratio * 0.4
        en_score = 1.0 - vi_score
    elif en_word_hits > vi_word_hits:
        en_score = 0.5 + en_word_ratio * 0.4
        vi_score = 1.0 - en_score
    else:
        # Ambiguous — check ASCII ratio
        if ascii_ratio > 0.95 and len(words) <= 5:
            # Short, all-ASCII — could be either; default to vi for this app
            vi_score = 0.5
            en_score = 0.5
        else:
            vi_score = 0.5
            en_score = 0.5

    # Determine language
    if vi_score > 0.7:
        language = "vi"
        confidence = vi_score
    elif en_score > 0.7:
        language = "en"
        confidence = en_score
    else:
        language = "mixed"
        confidence = max(vi_score, en_score)

    return {
        "language": language,
        "confidence": round(confidence, 3),
        "vi_ratio": round(vi_score, 3),
        "en_ratio": round(en_score, 3),
        "method": "char+word hybrid",
    }


# ═══════════════════════════════════════════════════════════════════════════
# STT Error Correction (Conversation Expert)
# ═══════════════════════════════════════════════════════════════════════════

# Common Vietnamese STT error patterns (Deepgram + Whisper known issues)
_STT_CORRECTIONS_VI: List[Tuple[str, str]] = [
    # Word-level corrections
    (r"\bdc\b", "được"),         # "dc" → "được" (teencode)
    (r"\bko\b", "không"),        # "ko" → "không"
    (r"\bkhg\b", "không"),       # "khg" → "không"
    (r"\bvs\b", "vậy"),          # "vs" → "vậy" (sometimes STT confusion)
    (r"\bnch\b", "nên"),         # abbreviated "nên"
    # Tone mark recovery (common Whisper issue)
    (r"\btai sao\b", "tại sao"),
    (r"\bvi sao\b", "vì sao"),
    (r"\bdc khong\b", "được không"),
    # Number normalization
    (r"\b(\d+) gio\b", r"\1 giờ"),
    (r"\bhom nay\b", "hôm nay"),
    (r"\bngay mai\b", "ngày mai"),
    # English STT in Vietnamese context
    (r"\badd task\b", "thêm task"),
    (r"\bdelete task\b", "xóa task"),
]

_STT_CORRECTIONS_EN: List[Tuple[str, str]] = [
    # Common Whisper English errors
    (r"\bwanna\b", "want to"),
    (r"\bgonna\b", "going to"),
    (r"\bgotta\b", "got to"),
    (r"\bkindra\b", "kind of"),
    (r"\bsorta\b", "sort of"),
    (r"\bdunno\b", "don't know"),
    (r"\bcant\b", "can't"),
    (r"\bdont\b", "don't"),
    (r"\bwont\b", "won't"),
    # Number normalization
    (r"\b(\d+) pm\b", r"\1 PM"),
    (r"\b(\d+) am\b", r"\1 AM"),
]


def correct_stt_errors(transcript: str, language: str = "auto") -> Tuple[str, List[str]]:
    """Apply STT error correction patterns to a transcript.

    Returns (corrected_transcript, list_of_fixes_applied).
    """
    fixes: List[str] = []
    corrected = transcript

    if language == "auto":
        lang_info = detect_language(transcript)
        language = lang_info["language"]

    # Apply Vietnamese corrections
    if language in ("vi", "mixed"):
        for pattern, replacement in _STT_CORRECTIONS_VI:
            new_text = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            if new_text != corrected:
                fixes.append(f"vi: '{pattern}' → '{replacement}'")
                corrected = new_text

    # Apply English corrections
    if language in ("en", "mixed"):
        for pattern, replacement in _STT_CORRECTIONS_EN:
            new_text = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            if new_text != corrected:
                fixes.append(f"en: '{pattern}' → '{replacement}'")
                corrected = new_text

    if fixes:
        logger.info("STT corrections applied: %s", fixes)

    return corrected, fixes


# ═══════════════════════════════════════════════════════════════════════════
# Tone Classification (Conversation Expert)
# ═══════════════════════════════════════════════════════════════════════════

# Command/imperative indicators
_COMMAND_PATTERNS_VI = [
    r"\b(thêm|xóa|sửa|đổi|tạo|viết|ghi|chỉnh|xem|mở|đóng|gửi|lưu|copy|dán)\b",
    r"\b(cho tôi|cho mình|giúp tôi|hãy|làm ơn)\b",
]

_COMMAND_PATTERNS_EN = [
    r"\b(add|delete|remove|update|create|write|change|edit|open|close|save|send|copy|paste|move)\b",
    r"\b(please|can you|could you|would you|I need|I want)\b",
]

_QUERY_PATTERNS_VI = [
    r"\b(tại sao|vì sao|như thế nào|làm sao|bao nhiêu|khi nào|ở đâu|có nên|nên không|cái nào)\b",
    r"\b(phân tích|giải thích|so sánh|đánh giá|tóm tắt|tổng hợp|dự đoán|đề xuất)\b",
    r"\?$",
]

_QUERY_PATTERNS_EN = [
    r"\b(what|why|how|when|where|who|which|should I|can I|could you explain)\b",
    r"\b(analyze|explain|compare|summarize|evaluate|review|suggest|recommend|predict)\b",
    r"\?$",
]

_CHITCHAT_PATTERNS_VI = [
    r"\b(xin chào|chào|hi|hello|hey|cảm ơn|thanks|cám ơn|bye|tạm biệt|goodbye)\b",
    r"\b(khỏe không|thế nào|ăn cơm chưa|đang làm gì)\b",
]

_CHITCHAT_PATTERNS_EN = [
    r"\b(hi|hello|hey|good morning|good afternoon|good evening|how are you|thanks|thank you|bye|goodbye|see you)\b",
]


def classify_tone(transcript: str, language: str = "auto") -> Dict[str, Any]:
    """Classify the tone of a transcript as command, query, or chitchat.

    Returns: {"tone": "command"|"query"|"chitchat", "confidence": float, "indicators": [...]}
    """
    if language == "auto":
        lang_info = detect_language(transcript)
        language = lang_info["language"]

    text = transcript.strip().lower()
    indicators: List[str] = []

    # Score each category
    cmd_score = 0.0
    query_score = 0.0
    chitchat_score = 0.0

    patterns_vi = _COMMAND_PATTERNS_VI if language in ("vi", "mixed") else []
    patterns_en = _COMMAND_PATTERNS_EN if language in ("en", "mixed") else []
    for p in patterns_vi + patterns_en:
        if re.search(p, text):
            cmd_score += 1.0
            indicators.append(f"cmd: {p}")

    patterns_vi = _QUERY_PATTERNS_VI if language in ("vi", "mixed") else []
    patterns_en = _QUERY_PATTERNS_EN if language in ("en", "mixed") else []
    for p in patterns_vi + patterns_en:
        if re.search(p, text):
            query_score += 1.5  # Query words are stronger signals than commands
            indicators.append(f"query: {p}")

    patterns_vi = _CHITCHAT_PATTERNS_VI if language in ("vi", "mixed") else []
    patterns_en = _CHITCHAT_PATTERNS_EN if language in ("en", "mixed") else []
    for p in patterns_vi + patterns_en:
        if re.search(p, text):
            chitchat_score += 1.5  # Chitchat words are strong signals
            indicators.append(f"chitchat: {p}")

    # Normalize
    total = cmd_score + query_score + chitchat_score
    if total == 0:
        # Default: short text without clear indicators → command (most common in workspace)
        return {"tone": "command", "confidence": 0.5, "indicators": ["default: no clear tone markers"]}

    cmd_score /= total
    query_score /= total
    chitchat_score /= total

    if chitchat_score > 0.4:
        tone = "chitchat"
        conf = chitchat_score
    elif query_score > cmd_score:
        tone = "query"
        conf = query_score
    else:
        tone = "command"
        conf = cmd_score

    return {
        "tone": tone,
        "confidence": round(conf, 3),
        "indicators": indicators[:5],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Risk Assessment Matrix (Contrarian Expert)
# ═══════════════════════════════════════════════════════════════════════════

_RISK_MATRIX: Dict[str, Dict[str, Any]] = {
    "delete_row": {
        "base_risk": "high",
        "concern": "Irreversible data deletion",
        "mitigations": ["Confirm row exists", "Show preview before deletion", "Allow undo"],
    },
    "update_cell": {
        "base_risk": "medium",
        "concern": "Data overwrite — previous value lost",
        "mitigations": ["Verify target cell exists", "Log previous value"],
    },
    "bulk_update_stack": {
        "base_risk": "high",
        "concern": "Mass data mutation — multiple rows affected at once",
        "mitigations": ["Confirm row count", "Show diff preview", "Batch confirm"],
    },
    "add_stack_row": {
        "base_risk": "low",
        "concern": "Duplicate rows possible if command repeated",
        "mitigations": ["Check for duplicates before adding"],
    },
    "update_note": {
        "base_risk": "medium",
        "concern": "Content overwrite at cursor or append with wrong position",
        "mitigations": ["Show cursor position", "Confirm append vs insert"],
    },
    "manage_tasks": {
        "base_risk": "medium",
        "concern": "Task state mutation — status/priority/assignee changes",
        "mitigations": ["Verify task exists", "Log state transition"],
    },
    "create_task": {
        "base_risk": "low",
        "concern": "Duplicate task creation",
        "mitigations": ["Check for similar existing tasks"],
    },
    "create_calendar_event": {
        "base_risk": "low",
        "concern": "Wrong date/time interpretation from relative expressions",
        "mitigations": ["Confirm parsed datetime", "Check for conflicts"],
    },
    "summarize_context": {
        "base_risk": "low",
        "concern": "Summary may omit critical information",
        "mitigations": ["Review summary completeness"],
    },
    "research": {
        "base_risk": "low",
        "concern": "Web-sourced content may be inaccurate or outdated",
        "mitigations": ["Cite sources", "Ground findings only in retrieved results"],
    },
    "none": {
        "base_risk": "low",
        "concern": "User intent not actioned — may need follow-up",
        "mitigations": ["Provide clear reply explaining why no action taken"],
    },
}


def assess_action_risk(action_type: str) -> Dict[str, Any]:
    """Look up the risk profile for a given action type."""
    return _RISK_MATRIX.get(action_type, {
        "base_risk": "medium",
        "concern": "Unknown action type — proceed with caution",
        "mitigations": ["Verify action is valid for context"],
    })


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case Generator (Contrarian Expert)
# ═══════════════════════════════════════════════════════════════════════════

_EDGE_CASE_TEMPLATES: Dict[str, List[str]] = {
    "NOTE": [
        "What if the note is currently empty?",
        "What if the cursor is at position 0 vs end of document?",
        "What if the note contains code blocks or special formatting?",
        "What if the user meant a different note?",
        "What if this is an edit, not an append?",
    ],
    "STACK": [
        "What if the target row was deleted between command and execution?",
        "What if the column name is misspelled or ambiguous?",
        "What if the value type doesn't match the column schema?",
        "What if the user is referencing a row by content, not ID?",
        "What if this should be a bulk operation, not single-cell?",
        "What if the stack has filters applied, hiding relevant rows?",
    ],
    "TASK": [
        "What if the task was already completed/deleted?",
        "What if the due date is in the past?",
        "What if the parent task doesn't exist (for subtasks)?",
        "What if the assignee name is ambiguous?",
        "What if priority should be inferred from deadline proximity?",
    ],
    "CALENDAR": [
        "What if the time is ambiguous (AM/PM, timezone)?",
        "What if the event conflicts with an existing event?",
        "What if 'tomorrow' means different things at 11:59 PM?",
        "What if the duration is unspecified?",
        "What if this is a recurring event?",
    ],
    "none": [
        "What if the user expects a factual answer, not an opinion?",
        "What if the query requires real-time data (weather, news, stocks)?",
        "What if the question has a false premise?",
        "What if the user is testing the assistant's boundaries?",
        "What if this is a multi-turn conversation, not a single query?",
    ],
}


def generate_edge_cases(context_type: str, transcript: str = "") -> List[str]:
    """Generate relevant edge-case questions based on context type.

    Returns a list of edge-case questions the Contrarian should consider.
    """
    bases = _EDGE_CASE_TEMPLATES.get(context_type, _EDGE_CASE_TEMPLATES["none"])

    # Add transcript-specific edge cases
    specific = []
    t = transcript.lower()

    if any(w in t for w in ["tất cả", "all", "mọi", "every"]):
        specific.append("⚠️ Bulk operation keyword detected — confirm scope (all vs selected)")
    if any(w in t for w in ["xóa", "delete", "remove", "clear"]):
        specific.append("⚠️ Destructive action keyword — confirm before executing")
    if any(w in t for w in ["cái này", "nó", "this", "that", "it"]):
        specific.append("⚠️ Ambiguous pronoun — what exactly does 'this'/'nó' refer to?")
    if any(w in t for w in ["không", "not", "don't", "đừng"]):
        specific.append("⚠️ Negation detected — ensure polarity is correctly interpreted")

    return bases + specific


# ═══════════════════════════════════════════════════════════════════════════
# Sycophancy Detection (Contrarian Expert)
# ═══════════════════════════════════════════════════════════════════════════

_SYCOPHANCY_INDICATORS = [
    "Direct agreement without verification",
    "Assumes user's premise is correct",
    "Doesn't question ambiguous references",
    "Fills in missing information with assumptions",
    "Prefers action over clarification when ambiguous",
    "Doesn't flag data that contradicts user's statement",
    "Accepts user's terminology without mapping to schema",
]


def detect_sycophancy_risks(
    transcript: str,
    context_type: str,
    workspace_facts: Dict[str, Any],
) -> List[str]:
    """Identify specific sycophancy risks for a given command.

    Returns a list of sycophancy risk descriptions the Contrarian should address.
    """
    risks = []

    # Check: user makes claims about data that isn't in workspace
    data_gaps = workspace_facts.get("missing_critical", [])
    if data_gaps:
        risks.append(
            f"User may be assuming data that isn't present: {', '.join(data_gaps)}. "
            "Without verification, the assistant will hallucinate."
        )

    # Check: user uses vague references
    t = transcript.lower()
    vague_references = ["cái này", "cái đó", "nó", "this", "that", "it", "the", "đó"]
    found_vague = [ref for ref in vague_references if ref in t.split()]
    if found_vague:
        risks.append(
            f"Ambiguous references detected ({', '.join(found_vague)}). "
            "Assistant may resolve these incorrectly without clarification."
        )

    # Check: user asserts facts
    assertion_patterns = [
        r"\b(nó là|nó đang|hiện tại|currently it|it is|it has|the data shows)\b",
    ]
    for pattern in assertion_patterns:
        if re.search(pattern, t):
            risks.append(
                "User is making an assertion about current state. "
                "Assistant may agree without verifying against workspace."
            )
            break

    # Check: yes/no question bias
    yes_no_patterns = [r"\b(có phải|đúng không|phải không|right\?|correct\?|is it\?)\b"]
    for pattern in yes_no_patterns:
        if re.search(pattern, t):
            risks.append(
                "Yes/no question format may bias assistant toward agreement. "
                "Independent verification needed."
            )
            break

    if not risks:
        risks.append("No strong sycophancy signals detected — but verify assumptions anyway.")

    return risks


# ═══════════════════════════════════════════════════════════════════════════
# Critical Reasoning Template Builder (Contrarian Expert)
# ═══════════════════════════════════════════════════════════════════════════

def build_contrarian_template(
    transcript: str,
    context_type: str,
    workspace_facts: Dict[str, Any],
    likely_action: str = "unknown",
    execution_plan: Any = None,  # Optional[ExecutionPlan]
) -> str:
    """Build a structured critical reasoning template for the Contrarian.

    NOW PLAN-AWARE: when execution_plan is provided, the framework includes
    the plan steps and asks the contrarian to critique EACH step.
    """
    risk_profile = assess_action_risk(likely_action)
    edge_cases = generate_edge_cases(context_type, transcript)
    sycophancy_risks = detect_sycophancy_risks(transcript, context_type, workspace_facts)
    data_gaps = workspace_facts.get("missing_critical", [])
    entities = workspace_facts.get("key_entities", [])

    # ── Build plan context section ──
    plan_context = ""
    if execution_plan and execution_plan.steps:
        plan_lines = ["─── EXECUTION PLAN (from Planner Node) ───",
                      f"Overall goal: {execution_plan.overall_goal}",
                      f"Reasoning: {execution_plan.reasoning}",
                      f"Multi-step: {execution_plan.is_multi_step}",
                      "Steps:"]
        for s in execution_plan.steps:
            deps = f" (depends on step(s) {s.depends_on})" if s.depends_on else ""
            plan_lines.append(
                f"  Step {s.step}: [{s.action}] {s.description}{deps}"
                + (f" — params_hint: {json.dumps(s.params_hint, ensure_ascii=False)}" if s.params_hint else "")
            )
        plan_context = "\n".join(plan_lines)

    template = f"""=== CONTRARIAN CRITICAL REASONING FRAMEWORK ===

STEP 0 — REVIEW THE EXECUTION PLAN (IF PROVIDED)
{plan_context if plan_context else '[No plan — single action assessment only]'}

STEP 1 — DECONSTRUCT THE COMMAND
- Raw transcript: "{transcript}"
- Context type: {context_type}
- Known entities in workspace: {json.dumps(entities[:5], ensure_ascii=False) if entities else 'None'}
- Data gaps: {json.dumps(data_gaps, ensure_ascii=False) if data_gaps else 'None identified'}

STEP 2 — CHALLENGE THE PLAN (or the obvious interpretation if no plan)
For each edge case below, ask: "Could the plan fail because of this?"

Edge cases to consider:
{chr(10).join(f'- {ec}' for ec in edge_cases)}

STEP 3 — DETECT SYCOPHANCY PATTERNS
The assistant tends to: agree without verification, assume user correctness,
fill gaps with assumptions. For THIS command, the specific risks are:

{chr(10).join(f'- {sr}' for sr in sycophancy_risks)}

STEP 4 — RISK ASSESSMENT
- If any plan step is destructive (delete, overwrite, bulk mutate) → HIGH risk
- If a plan step is reversible but suboptimal → MEDIUM risk
- If all steps are straightforward and safe → LOW risk

Likely action: {likely_action}
Action risk profile: {json.dumps(risk_profile, ensure_ascii=False)}

STEP 5 — OUTPUT
Based on the analysis above, produce a JSON output with:
- critique: Per-step critique — reference step numbers. What could go wrong?
- risk: "low" | "medium" | "high"
- alternative_action: A specific different action, or null if none is plausible

Remember: Your job is NOT to block the user. It's to make the assistant THINK
before acting. Critique the PLAN, not just the raw utterance.
"""
    return template


# ═══════════════════════════════════════════════════════════════════════════
# Intent Disambiguation (Conversation Expert)
# ═══════════════════════════════════════════════════════════════════════════

_AMBIGUITY_PATTERNS: List[Tuple[str, str]] = [
    # Vietnamese ambiguous references
    (r"\bcái này\b", "Ambiguous: 'cái này' — which specific item?"),
    (r"\bcái đó\b", "Ambiguous: 'cái đó' — which previously mentioned item?"),
    (r"\bnó\b", "Ambiguous: 'nó' — unclear antecedent"),
    (r"\bđây\b", "Ambiguous: 'đây' — what exactly is 'here'?"),
    (r"\bở đó\b", "Ambiguous: 'ở đó' — which location?"),
    (r"\bngười đó\b", "Ambiguous: 'người đó' — which person?"),
    # English ambiguous references
    (r"\bthis one\b", "Ambiguous: 'this one' — which specific item?"),
    (r"\bthat one\b", "Ambiguous: 'that one' — which previously referenced item?"),
    (r"\bit\b", "Ambiguous: 'it' — unclear antecedent"),
    (r"\bthey\b", "Ambiguous: 'they' — who?"),
    (r"\bthere\b", "Ambiguous: 'there' — which location?"),
    # Action ambiguity
    (r"\bđổi\b(?!.*(?:màu|tên|số|ngày|giờ|thành))", "Ambiguous: 'đổi' — change what aspect?"),
    (r"\bsửa\b(?!.*(?:lỗi|bài|file|nội dung|cột|dòng))", "Ambiguous: 'sửa' — edit what exactly?"),
]


def detect_ambiguity(transcript: str) -> Tuple[bool, List[str], float]:
    """Detect ambiguous references and unclear intent in a transcript.

    Returns: (has_ambiguity, list_of_ambiguities_found, ambiguity_score)
    """
    text = transcript.strip().lower()
    found: List[str] = []

    for pattern, description in _AMBIGUITY_PATTERNS:
        if re.search(pattern, text):
            found.append(description)

    # Additional heuristic: very short commands are often ambiguous
    words = text.split()
    if len(words) <= 2 and not any(
        w in text for w in ["xin chào", "hi", "hello", "cảm ơn", "thanks", "bye"]
    ):
        found.append(f"Very short command ({len(words)} words) — likely missing context")

    # Score: 0 = clear, 1 = highly ambiguous
    score = min(1.0, len(found) * 0.25 + (0.3 if len(words) <= 2 else 0))

    return len(found) > 0, found, round(score, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Data Formatter (Research Expert)
# ═══════════════════════════════════════════════════════════════════════════

def format_workspace_for_llm(workspace_facts: Dict[str, Any]) -> str:
    """Format extracted workspace facts into a concise LLM-readable summary."""
    lines = [
        "=== WORKSPACE STATE SUMMARY ===",
        f"Context type: {workspace_facts.get('context_type', 'unknown')}",
        f"Items in context: {workspace_facts.get('item_count', 0)}",
        "",
    ]

    stats = workspace_facts.get("data_stats", {})
    if stats:
        lines.append("Data Statistics:")
        for k, v in stats.items():
            if isinstance(v, list):
                lines.append(f"  {k}: {', '.join(str(x) for x in v)}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")

    entities = workspace_facts.get("key_entities", [])
    if entities:
        lines.append("Key Entities:")
        for e in entities:
            lines.append(f"  [{e.get('type', '?')}] {e.get('name', '?')}: {e.get('value', '?')}")
        lines.append("")

    gaps = workspace_facts.get("missing_critical", [])
    if gaps:
        lines.append("⚠️ MISSING CRITICAL DATA:")
        for g in gaps:
            lines.append(f"  - {g}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Structured Research Template Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_research_template(
    transcript: str,
    context_type: str,
    workspace_facts: Dict[str, Any],
    web_results: str = "",
    execution_plan: Any = None,  # Optional[ExecutionPlan]
) -> str:
    """Build a structured research analysis template.

    NOW PLAN-AWARE: when execution_plan is provided, the framework asks the
    research expert to ground EACH step against available data.
    """
    workspace_summary = format_workspace_for_llm(workspace_facts)

    # ── Build plan context section ──
    plan_context = ""
    if execution_plan and execution_plan.steps:
        plan_lines = ["─── EXECUTION PLAN (from Planner Node) ───",
                      f"Overall goal: {execution_plan.overall_goal}"]
        for s in execution_plan.steps:
            plan_lines.append(
                f"  Step {s.step}: [{s.action}] {s.description}"
                + f" — needs context: {s.context_required or 'any'}"
            )
        plan_context = "\n".join(plan_lines)

    template = f"""=== RESEARCH GROUNDING FRAMEWORK ===

STEP 0 — REVIEW THE EXECUTION PLAN (IF PROVIDED)
{plan_context if plan_context else '[No plan — ground the raw command directly]'}

STEP 1 — IDENTIFY WHAT THE USER IS ASKING
Transcript: "{transcript}"
Context type: {context_type}

STEP 2 — INVENTORY WORKSPACE STATE
{workspace_summary}

STEP 3 — MAP EACH PLAN STEP TO AVAILABLE DATA
For each step in the plan (or the raw command if no plan):
- Is this entity present in the workspace? → YES: cite it / NO: flag as data gap
- Is the action valid for this context type? → YES: confirm / NO: flag as mismatch
- Are all required parameters available? → YES: note them / NO: list what's missing
- TAG each gap with the step number: "Step 2 needs: column schema"

STEP 4 — EXTERNAL KNOWLEDGE (if applicable)
{web_results if web_results else '[No web search performed — command is workspace-scoped]'}

STEP 5 — CONFIDENCE CALIBRATION
- 0.9-1.0: All data present for all plan steps
- 0.5-0.8: Gaps exist in some steps
- 0.0-0.4: Critical data missing, cannot ground the command

STEP 6 — OUTPUT JSON
{{
  "relevant_context": "Specific facts from workspace state (with actual values, reference step numbers)",
  "confidence": <0.0 to 1.0>,
  "data_gaps": ["list specific missing data points per step, e.g. 'Step 2 (add_stack_row): column schema not provided'"],
  "research_findings": "If web results are present above: synthesize them into 3-6 sentences of usable, factual content (cite key facts/dates/numbers). This text will be inserted into the user's workspace verbatim, so write it as polished prose. Empty string if no web results."
}}

IMPORTANT: If you found data in workspace state, CITE IT SPECIFICALLY.
Reference PLAN STEP NUMBERS when identifying gaps.
Do not say 'the note contains information' — say what the information IS.
If a data_gap exists, say exactly what's missing, not 'some data is missing'.
If web results exist, research_findings MUST be non-empty and grounded ONLY in those results.
"""
    return template


# ═══════════════════════════════════════════════════════════════════════════
# Conversation Analysis Template Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_conversation_template(
    transcript: str,
    lang_info: Dict[str, Any],
    tone_info: Dict[str, Any],
    has_ambiguity: bool,
    ambiguity_list: List[str],
    stt_corrected: str,
    stt_fixes: List[str],
    execution_plan: Any = None,  # Optional[ExecutionPlan]
) -> str:
    """Build a structured conversation analysis template.

    NOW PLAN-AWARE: when execution_plan is provided, the framework asks the
    conversation expert to verify the plan matches the user's intent.
    """
    # ── Build plan context section ──
    plan_context = ""
    if execution_plan and execution_plan.steps:
        plan_lines = ["─── EXECUTION PLAN (from Planner Node) ───",
                      f"Overall goal: {execution_plan.overall_goal}"]
        for s in execution_plan.steps:
            plan_lines.append(f"  Step {s.step}: [{s.action}] {s.description}")
        plan_context = "\n".join(plan_lines)

    return f"""=== CONVERSATION ANALYSIS FRAMEWORK ===

STEP 0 — REVIEW THE EXECUTION PLAN (IF PROVIDED)
{plan_context if plan_context else '[No plan — analyze raw command directly]'}

STEP 1 — RAW INPUT
Original transcript: "{transcript}"
STT-corrected transcript: "{stt_corrected}"
STT fixes applied: {json.dumps(stt_fixes, ensure_ascii=False) if stt_fixes else 'None'}

STEP 2 — LANGUAGE DETECTION
Language: {lang_info.get('language', 'unknown')} (confidence: {lang_info.get('confidence', 0)})
Vietnamese character ratio: {lang_info.get('vi_ratio', 0)}
English word ratio: {lang_info.get('en_ratio', 0)}
Method: {lang_info.get('method', 'unknown')}

STEP 3 — TONE CLASSIFICATION
Detected tone: {tone_info.get('tone', 'unknown')} (confidence: {tone_info.get('confidence', 0)})
Indicators found: {json.dumps(tone_info.get('indicators', []), ensure_ascii=False)}

STEP 4 — AMBIGUITY CHECK (including plan-intent alignment)
Has ambiguity: {'YES' if has_ambiguity else 'No'}
Ambiguities detected:
{chr(10).join(f'- {a}' for a in ambiguity_list) if ambiguity_list else '- None'}

STEP 4b — PLAN-INTENT ALIGNMENT (if plan provided)
Compare the plan's overall_goal against the transcript:
- Does the plan capture ALL actions the user requested?
- Does the plan do anything the user did NOT ask for?
- Are the step actions appropriate for this context type?

STEP 5 — OUTPUT JSON
{{
  "intent": "Clear, specific statement of what the user wants (in English)",
  "tone": "{tone_info.get('tone', 'command')}",
  "language": "{lang_info.get('language', 'vi')}",
  "has_ambiguity": {'true' if has_ambiguity else 'false'}
}}

IMPORTANT:
- 'intent' must be specific: not 'manage tasks' but 'create a task titled X due on Y'
- If a plan is provided and it doesn't match the intent, set has_ambiguity=true
- If the language is 'mixed', note which parts are which in the intent
- If STT corrections were applied, use the CORRECTED transcript for intent extraction
- Be generous with has_ambiguity — it's safer to flag than to assume
"""


# ═══════════════════════════════════════════════════════════════════════════
# Planning Node Tools
# ═══════════════════════════════════════════════════════════════════════════

# Multi-step command indicators (Vietnamese + English)
_MULTI_STEP_TRIGGERS_VI = [
    r"\b(sau đó|rồi|trước khi|tiếp theo|cuối cùng|đầu tiên|sau khi|trước đó)\b",
    r"\b(và|cùng lúc|đồng thời|cả hai|cũng như)\b",
    r"\b(thì|rồi thì|xong rồi|làm xong)\b",
]

_MULTI_STEP_TRIGGERS_EN = [
    r"\b(then|after that|before|next|finally|first|after|subsequently)\b",
    r"\b(and also|as well as|both|simultaneously|meanwhile)\b",
    r"\b(once|when|afterwards|following that)\b",
]

# Atomic action verbs mapped to their action types
_ACTION_TYPE_MAP: Dict[str, str] = {
    # Vietnamese
    "thêm": "add_stack_row", "tạo": "create_task", "viết": "update_note",
    "ghi": "update_note", "xóa": "delete_row", "sửa": "update_cell",
    "đổi": "update_cell", "cập nhật": "bulk_update_stack",
    "lên lịch": "create_calendar_event", "đặt lịch": "create_calendar_event",
    "tóm tắt": "summarize_context", "tổng hợp": "summarize_context",
    "quản lý": "manage_tasks", "hoàn thành": "manage_tasks",
    # Research/lookup verbs (count as a step: gather info before acting)
    "nghiên cứu": "research", "tìm hiểu": "research", "tra cứu": "research",
    # English
    "add": "add_stack_row", "create": "create_task", "write": "update_note",
    "delete": "delete_row", "remove": "delete_row", "update": "update_cell",
    "change": "update_cell", "edit": "update_cell",
    "schedule": "create_calendar_event", "summarize": "summarize_context",
    "complete": "manage_tasks", "finish": "manage_tasks",
    "research": "research", "look up": "research", "find out": "research",
    "insert": "update_note", "append": "update_note", "save": "update_note",
}

# Context-aware override: a generic "add/thêm" verb means different actions
# depending on the active workspace context.
_CONTEXT_ACTION_OVERRIDES: Dict[str, Dict[str, str]] = {
    "NOTE": {"add_stack_row": "update_note", "update_cell": "update_note",
             "delete_row": "update_note", "create_task": "update_note",
             "bulk_update_stack": "update_note"},
    "STACK": {"update_note": "add_stack_row", "create_task": "add_stack_row"},
    "TASK": {"add_stack_row": "manage_tasks", "update_cell": "manage_tasks",
             "update_note": "manage_tasks", "delete_row": "manage_tasks",
             "create_task": "manage_tasks"},
    "TASKS": {"add_stack_row": "manage_tasks", "update_cell": "manage_tasks",
              "update_note": "manage_tasks", "delete_row": "manage_tasks",
              "create_task": "manage_tasks"},
    "CALENDAR": {"add_stack_row": "create_calendar_event",
                 "create_task": "create_calendar_event",
                 "update_note": "create_calendar_event"},
}


def resolve_action_for_context(action_type: str, context_type: str) -> str:
    """Map a verb-derived action type to the correct action for the active context.

    e.g. "thêm"/"add" maps to add_stack_row by default, but in a NOTE context
    the right action is update_note; in TASK context it's manage_tasks.
    """
    overrides = _CONTEXT_ACTION_OVERRIDES.get(context_type, {})
    return overrides.get(action_type, action_type)


def detect_multi_step(transcript: str) -> Tuple[bool, List[str], int]:
    """Detect if a command requires multiple sequential steps.

    Returns: (is_multi_step, trigger_phrases_found, estimated_step_count)
    """
    t = transcript.lower()
    triggers: List[str] = []

    for pattern in _MULTI_STEP_TRIGGERS_VI + _MULTI_STEP_TRIGGERS_EN:
        matches = re.findall(pattern, t)
        triggers.extend(matches)

    # Count distinct action verbs
    action_count = 0
    for verb in _ACTION_TYPE_MAP:
        if re.search(rf"\b{verb}\b", t):
            action_count += 1

    # Heuristic: sequential indicators OR 3+ action verbs → multi-step
    is_multi = len(triggers) >= 2 or action_count >= 2
    estimated_steps = max(1, min(action_count + len(triggers), 10))

    return is_multi, triggers, estimated_steps


def extract_action_verbs(transcript: str) -> List[Dict[str, str]]:
    """Extract action verbs with their mapped action types from a transcript.

    Returns: [{"verb": "thêm", "action_type": "add_stack_row", "position": 0}, ...]
    """
    t = transcript.lower()
    found: List[Dict[str, str]] = []

    for verb, action_type in _ACTION_TYPE_MAP.items():
        for match in re.finditer(rf"\b{verb}\b", t):
            found.append({
                "verb": match.group(),
                "action_type": action_type,
                "position": match.start(),
            })

    # Sort by position in transcript
    found.sort(key=lambda x: x["position"])
    return found


def build_planning_template(
    transcript: str,
    context_type: str,
    workspace_facts: Dict[str, Any],
) -> str:
    """Build a structured planning template for the Planning Node.

    The Planner runs SEQUENTIALLY FIRST — before any expert agents.
    It decomposes complex commands into ordered executable steps.
    Experts (contrarian, research, conversation) will later receive
    this plan and critique/enrich it.

    This gives the LLM a structured framework to decompose a complex
    command into an ordered sequence of executable steps.
    """
    is_multi, triggers, est_steps = detect_multi_step(transcript)
    action_verbs = extract_action_verbs(transcript)
    entities = workspace_facts.get("key_entities", [])
    data_gaps = workspace_facts.get("missing_critical", [])

    template = f"""=== TASK PLANNING FRAMEWORK ===

STEP 1 — UNDERSTAND THE GOAL
Transcript: "{transcript}"
Context type: {context_type}
Detected as multi-step: {'YES' if is_multi else 'No'}
Multi-step triggers found: {json.dumps(triggers[:5], ensure_ascii=False) if triggers else 'None'}
Estimated step count: {est_steps}

STEP 2 — IDENTIFY ATOMIC ACTIONS
Action verbs detected (in order):
{chr(10).join(f'- "{v["verb"]}" → {v["action_type"]} (position {v["position"]})' for v in action_verbs) if action_verbs else '- No clear action verbs detected — infer from context'}

STEP 3 — AVAILABLE CONTEXT
Known entities: {json.dumps(entities[:5], ensure_ascii=False) if entities else 'None'}
Data gaps: {json.dumps(data_gaps, ensure_ascii=False) if data_gaps else 'None'}

STEP 4 — BUILD THE PLAN
For each step, specify:
- step: sequential number
- action: one of [update_note, add_stack_row, bulk_update_stack, update_cell, delete_row, manage_tasks, create_task, summarize_context, create_calendar_event, none]
- description: what this step accomplishes
- params_hint: hints for parameter values (not exact params — the Resolver fills those in)
- depends_on: list of step numbers this step depends on (empty if independent)
- context_required: what data from the workspace this step needs

STEP 5 — OUTPUT JSON
{{
  "overall_goal": "The user's goal in one clear sentence",
  "reasoning": "Why this plan structure was chosen",
  "steps": [
    {{
      "step": 1,
      "action": "action_type",
      "description": "what this does",
      "params_hint": {{}},
      "depends_on": [],
      "context_required": ""
    }}
  ],
  "is_multi_step": true/false,
  "fallback_action": "none"
}}

RULES:
- You are the FIRST node in the deliberation pipeline. Experts will critique your plan later.
- If the command is NOT truly multi-step, set is_multi_step=false and use fallback_action for the single action.
- Steps should be ordered logically — step N can only depend on steps < N.
- params_hint should be hints, not exact values. The Resolver fills in exact params.
- For single-step commands, return one step and is_multi_step=false.
- The plan should be EXECUTABLE — each step must be a valid action with clear inputs.
- Be decisive. A clear plan with minor imperfections is better than an overly cautious 'none'.
"""
    return template


# ═══════════════════════════════════════════════════════════════════════════
# Reflection Pattern Tools
# ═══════════════════════════════════════════════════════════════════════════

# Known hallucination patterns to check
_HALLUCINATION_CHECKS = [
    (r"rowId.*[a-fA-F0-9]{8}-[a-fA-F0-9]{4}", "Contains invented UUID in rowId"),
    (r"task_id.*[a-fA-F0-9]{8}-[a-fA-F0-9]{4}", "Contains invented UUID in task_id"),
    (r"stack_id.*[a-fA-F0-9]{8}-[a-fA-F0-9]{4}", "Contains invented UUID in stack_id"),
    (r'"value":\s*"[^"]{200,}"', "Very long value string — possible hallucinated content"),
    (r'"content_to_insert":\s*"[^"]{500,}"', "Very long note content — possible hallucination"),
]

# Context mismatch checks
_CONTEXT_MISMATCH_CHECKS = [
    ("action", "update_note", "NOTE"),
    ("action", "add_stack_row", "STACK"),
    ("action", "bulk_update_stack", "STACK"),
    ("action", "update_cell", "STACK"),
    ("action", "delete_row", "STACK"),
    ("action", "manage_tasks", "TASK"),
    ("action", "create_task", "TASK"),
    ("action", "create_calendar_event", "CALENDAR"),
]


def detect_hallucination(nlu_result: dict, context_type: str) -> Tuple[bool, List[str]]:
    """Detect common LLM hallucination patterns in the resolver output.

    Checks for:
    - Invented UUIDs (the LLM making up IDs)
    - Excessively long values (hallucinated content)
    - Context/action mismatches

    Returns: (has_hallucination, list of issues found)
    """
    issues: List[str] = []
    result_str = json.dumps(nlu_result, ensure_ascii=False)

    # Check for hallucination patterns
    for pattern, desc in _HALLUCINATION_CHECKS:
        if re.search(pattern, result_str):
            issues.append(desc)

    # Check for context/action mismatch
    action = nlu_result.get("action", "")
    for field, expected_action, required_context in _CONTEXT_MISMATCH_CHECKS:
        if action == expected_action and context_type != required_context:
            # Only flag if context really doesn't match (some actions work across contexts)
            if action not in ("summarize_context", "none"):
                issues.append(
                    f"Action '{action}' requires '{required_context}' context but got '{context_type}'"
                )
                break

    # Check for empty params when action requires them
    params = nlu_result.get("params", {})
    if action not in ("none", "summarize_context") and (not params or params == {}):
        issues.append(f"Action '{action}' has empty params — likely incomplete")

    # Check for missing required fields
    if action == "update_note" and "content_to_insert" not in params:
        issues.append("update_note missing 'content_to_insert'")
    if action == "manage_tasks" and "action_type" not in params:
        issues.append("manage_tasks missing 'action_type'")
    if action == "create_calendar_event" and "startAt" not in params:
        issues.append("create_calendar_event missing 'startAt'")

    return len(issues) > 0, issues


def build_reflection_template(
    transcript: str,
    context_type: str,
    nlu_result: dict,
    iteration: int,
    expert_outputs: Dict[str, Any] = None,
) -> str:
    """Build a structured reflection/critique template.

    This implements the Reflexion pattern: the LLM critiques its own output
    and suggests improvements. If the score is below threshold, the resolver
    is re-invoked with the critique as additional context.
    """
    action = nlu_result.get("action", "unknown")
    params = nlu_result.get("params", {})
    reply = nlu_result.get("reply")

    has_hallu, hallu_issues = detect_hallucination(nlu_result, context_type)

    # Expert context summary
    expert_context = ""
    if expert_outputs:
        parts = []
        if expert_outputs.get("contrarian"):
            c = expert_outputs["contrarian"]
            if c.risk != "low":
                parts.append(f"Contrarian flagged {c.risk.upper()} risk: {c.critique[:150]}")
        if expert_outputs.get("research"):
            r = expert_outputs["research"]
            if r.data_gaps:
                parts.append(f"Research found data gaps: {', '.join(r.data_gaps[:3])}")
        if expert_outputs.get("conversation"):
            v = expert_outputs["conversation"]
            if v.has_ambiguity:
                parts.append(f"Conversation flagged ambiguity: {v.intent[:150]}")
        expert_context = "\n".join(parts)

    template = f"""=== REFLECTION CRITIQUE FRAMEWORK ===
Iteration: {iteration + 1} (max 3)

STEP 1 — REVIEW THE INPUT
User said: "{transcript}"
Context type: {context_type}

STEP 2 — REVIEW THE OUTPUT
Action chosen: {action}
Params: {json.dumps(params, ensure_ascii=False)[:500]}
Reply: {reply or 'null'}

STEP 3 — HALLUCINATION CHECK
Pre-computed checks:
{chr(10).join(f'- {h}' for h in hallu_issues) if hallu_issues else '- No hallucination issues detected'}

STEP 4 — EXPERT FEEDBACK TO CONSIDER
{expert_context if expert_context else '[No expert deliberation — no cross-reference available]'}

STEP 5 — QUALITY ASSESSMENT
Evaluate the output on these dimensions:
1. Action correctness: Is "{action}" the right action for "{transcript}"? (score 0-1)
2. Parameter completeness: Are all required params present and correctly typed? (score 0-1)
3. Context respect: Does the action respect the context_type "{context_type}"? (score 0-1)
4. Reply appropriateness: If reply is set, is it natural and helpful? (score 0-1)

STEP 6 — OUTPUT JSON
{{
  "score": <0.0-1.0 overall quality>,
  "issues": ["specific problems found"],
  "suggestions": ["concrete fixes for the resolver"],
  "needs_refinement": true/false,
  "critique_summary": "one-sentence evaluation"
}}

THRESHOLDS:
- score >= 0.85: Output is good → needs_refinement=false
- score 0.6-0.84: Minor issues → needs_refinement=true (one more try)
- score < 0.6: Significant problems → needs_refinement=true (retry with strong guidance)

RULES:
- Be honest but not pedantic. Don't flag trivial formatting issues.
- If the action is "none", that's valid — don't penalize unless the command clearly needed action.
- Consider the expert feedback: if Contrarian flagged HIGH risk and the resolver proceeded anyway, that's a major issue.
- On iteration 3 (last attempt), accept the output even if imperfect.
- Each suggestion must be ACTIONABLE — the resolver should know exactly what to fix.
"""
    return template
