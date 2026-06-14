"""
Google Speech-to-Text integration for long audio transcription with speaker diarization.

Handles:
- Long audio chunking (> 10 MB files split into manageable segments)
- Speaker diarization via Google STT v1
- Fallback to existing litellm STT router when Google is unavailable
- Vietnamese (vi-VN) + English (en-US) language support

Architecture:
- Files <= 10 MB: Google sync Recognize with diarization (no GCS needed)
- Files > 10 MB: chunked → parallel Google sync Recognize → merge segments
- No GCS bucket required for the chunked path

Google STT config is read from environment (3 methods, checked in order):
  1. GOOGLE_CREDENTIALS_JSON — the raw JSON content of the service account key
     (best for EC2/Docker: paste the entire JSON as an env var)
  2. GOOGLE_APPLICATION_CREDENTIALS — path to a service account JSON file
     (best for local dev: download the key and point to it)
  3. GOOGLE_APPLICATION_CREDENTIALS — if the value itself looks like valid JSON,
     it's treated as inline credentials (same as #1)
- GOOGLE_STT_ENABLED: "true"/"1" to force-enable (optional)
- GOOGLE_STT_LANGUAGE: override language code (default: vi-VN)
"""

import asyncio
import io
import json as _json
import math
import os
from typing import List, Optional, Tuple

from .config import logger
from .records_models import SpeakerLabel, SpeakerSegment

# ── Chunking constants ────────────────────────────────────────────────────
# Google sync Recognize limit: 10 MB or 1 minute.
# We use 5 MB / 50 seconds chunks for safety margin.
MAX_CHUNK_BYTES = 5 * 1024 * 1024   # 5 MB per chunk
MAX_CHUNK_SECS = 50                  # ~50 seconds per chunk at typical webm bitrate
BYTES_PER_SECOND_ESTIMATE = 16_000   # Conservative webm/opus estimate (16 KB/s)


_google_stt_client = None
_google_available = False
_google_check_done = False


def _resolve_credentials() -> Optional[dict]:
    """Resolve Google service account credentials from environment.

    Checks in order:
      1. GOOGLE_CREDENTIALS_JSON — inline JSON string (best for EC2/Docker)
      2. GOOGLE_APPLICATION_CREDENTIALS — file path (best for local dev)
      3. GOOGLE_APPLICATION_CREDENTIALS — if it looks like JSON, treat it as inline

    Returns the parsed service account dict, or None.
    """
    # ── Method 1: GOOGLE_CREDENTIALS_JSON (inline JSON, deployment-friendly) ──
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()
    if creds_json:
        try:
            key_data = _json.loads(creds_json)
            if key_data.get("project_id") and key_data.get("client_email"):
                logger.info("Google STT: using GOOGLE_CREDENTIALS_JSON (inline) — project=%s", key_data["project_id"])
                return key_data
            else:
                logger.warning("Google STT: GOOGLE_CREDENTIALS_JSON is set but missing project_id/client_email")
        except _json.JSONDecodeError:
            logger.warning("Google STT: GOOGLE_CREDENTIALS_JSON is not valid JSON")

    # ── Method 2 & 3: GOOGLE_APPLICATION_CREDENTIALS ──
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not creds_path:
        return None

    # Method 3: if the value looks like JSON, treat it as inline credentials
    if creds_path.startswith("{") and creds_path.endswith("}"):
        try:
            key_data = _json.loads(creds_path)
            if key_data.get("project_id") and key_data.get("client_email"):
                logger.info("Google STT: GOOGLE_APPLICATION_CREDENTIALS is inline JSON — project=%s", key_data["project_id"])
                return key_data
        except _json.JSONDecodeError:
            pass  # Not JSON after all, treat as file path below

    # Method 2: file path
    if not os.path.isfile(creds_path):
        logger.error(
            "Google STT: GOOGLE_APPLICATION_CREDENTIALS file not found: %s. "
            "Download your key from GCP Console → IAM & Admin → Service Accounts → Keys. "
            "For EC2/Docker, use GOOGLE_CREDENTIALS_JSON instead (paste the entire JSON).",
            creds_path,
        )
        return None

    try:
        with open(creds_path, 'r') as f:
            key_data = _json.load(f)
        project_id = key_data.get("project_id", "")
        if not project_id:
            logger.error("Google STT: credentials file is missing 'project_id': %s", creds_path)
            return None
        logger.info("Google STT: using credentials file — project=%s account=%s",
                     project_id, key_data.get("client_email", "?"))
        return key_data
    except Exception as exc:
        logger.error("Google STT: failed to read credentials file %s: %s", creds_path, exc)
        return None


def _get_google_client():
    """Lazy-init Google Speech-to-Text client. Returns None if unavailable."""
    global _google_stt_client, _google_available, _google_check_done

    if _google_check_done:
        return _google_stt_client

    _google_check_done = True

    enabled = os.getenv("GOOGLE_STT_ENABLED", "").strip().lower() in ("1", "true", "yes")

    key_data = _resolve_credentials()
    if key_data is None:
        if enabled:
            logger.warning(
                "Google STT: GOOGLE_STT_ENABLED=true but no credentials found. "
                "Set GOOGLE_CREDENTIALS_JSON='{...}' (EC2/Docker) or "
                "GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json (local dev)."
            )
        else:
            logger.info(
                "Google STT: no credentials configured — "
                "falling back to litellm STT router (no diarization). "
                "To enable: set GOOGLE_CREDENTIALS_JSON (EC2) or "
                "GOOGLE_APPLICATION_CREDENTIALS (local)."
            )
        return None

    try:
        from google.cloud import speech_v1 as speech
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_info(key_data)
        _google_stt_client = speech.SpeechClient(credentials=credentials)
        _google_available = True
        logger.info("Google STT client initialized successfully")
        return _google_stt_client
    except ImportError:
        logger.error(
            "Google STT: google-cloud-speech package not installed. "
            "Run: pip install google-cloud-speech"
        )
        return None
    except Exception as exc:
        logger.error(
            "Google STT: client init failed: %s. "
            "Ensure the Speech-to-Text API is enabled: "
            "https://console.cloud.google.com/apis/library/speech.googleapis.com",
            exc,
        )
        return None


def _estimate_chunk_count(audio_bytes: bytes, mime_type: str) -> int:
    """Estimate how many chunks we need based on byte size and estimated duration."""
    by_size = math.ceil(len(audio_bytes) / MAX_CHUNK_BYTES)
    estimated_duration_secs = len(audio_bytes) / BYTES_PER_SECOND_ESTIMATE
    by_duration = max(1, math.ceil(estimated_duration_secs / MAX_CHUNK_SECS))
    return max(by_size, by_duration)


def _split_audio_bytes(audio_bytes: bytes, num_chunks: int) -> List[bytes]:
    """Split raw audio bytes into roughly equal chunks.

    NOTE: This is a naive byte-split. For production, use a proper audio
    library (pydub/ffmpeg) to split on frame boundaries. WebM/Opus and MP3
    have frame headers — splitting mid-frame may cause decoding artifacts.
    For now we accept this limitation and rely on Google STT's resilience.
    """
    if num_chunks <= 1:
        return [audio_bytes]

    chunk_size = len(audio_bytes) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(audio_bytes)
        chunks.append(audio_bytes[start:end])
    return chunks


def _detect_language_code(mime_type: str, filename: str) -> str:
    """Detect the best language code for STT. Defaults to vi-VN (primary use case)."""
    # Can be extended with a fast language-detection pass
    # For now, default to Vietnamese as the primary target
    override = os.getenv("GOOGLE_STT_LANGUAGE", "").strip()
    if override:
        return override
    return "vi-VN"


async def _transcribe_chunk_google(
    client,
    audio_bytes: bytes,
    mime_type: str,
    language_code: str,
    chunk_index: int,
    enable_diarization: bool = True,
    diarization_speaker_count: int = 5,
) -> Tuple[str, List[SpeakerSegment], List[dict]]:
    """Transcribe a single audio chunk with Google STT sync API.

    Returns (full_text, list_of_speaker_segments, raw_segments_with_speaker_info).
    The raw_segments include 'speaker' key for diarization grouping.
    """
    from google.cloud.speech_v1 import RecognitionAudio, RecognitionConfig

    # Map mime types to Google encoding enums
    encoding_map = {
        "audio/webm": RecognitionConfig.AudioEncoding.WEBM_OPUS,
        "audio/ogg": RecognitionConfig.AudioEncoding.OGG_OPUS,
        "audio/mp3": RecognitionConfig.AudioEncoding.MP3,
        "audio/mpeg": RecognitionConfig.AudioEncoding.MP3,
        "audio/wav": RecognitionConfig.AudioEncoding.LINEAR16,
        "audio/flac": RecognitionConfig.AudioEncoding.FLAC,
    }
    encoding = encoding_map.get(mime_type, RecognitionConfig.AudioEncoding.WEBM_OPUS)

    diarization_config = None
    if enable_diarization:
        from google.cloud.speech_v1 import SpeakerDiarizationConfig
        diarization_config = SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=1,
            max_speaker_count=diarization_speaker_count,
        )

    config = RecognitionConfig(
        encoding=encoding,
        language_code=language_code,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        model="latest_long" if len(audio_bytes) > 1_000_000 else "latest_short",
        diarization_config=diarization_config,
    )

    audio = RecognitionAudio(content=audio_bytes)

    try:
        response = await asyncio.to_thread(
            client.recognize, config=config, audio=audio
        )
    except Exception as exc:
        logger.error("Google STT chunk %d failed: %s", chunk_index, exc)
        raise

    full_text_parts = []
    raw_segments: List[dict] = []  # Use plain dicts to avoid Pydantic attr-setting issues
    current_speaker: Optional[str] = None
    current_start: float = 0.0
    current_end: float = 0.0
    current_words: List[str] = []

    def _flush_current():
        nonlocal current_speaker, current_start, current_end, current_words
        if current_speaker is not None and current_words:
            raw_segments.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_words),
            })
        current_speaker = None
        current_start = 0.0
        current_end = 0.0
        current_words = []

    for result in response.results:
        if not result.alternatives:
            continue
        alt = result.alternatives[0]
        full_text_parts.append(alt.transcript)

        # Build speaker segments from word-level diarization
        for word in alt.words:
            speaker_tag = str(word.speaker_tag) if word.speaker_tag else "0"
            speaker_label = f"Speaker {speaker_tag}"
            word_start = word.start_time.total_seconds() if word.start_time else 0.0
            word_end = word.end_time.total_seconds() if word.end_time else 0.0

            # Offset by chunk index to keep global timeline
            chunk_offset = chunk_index * MAX_CHUNK_SECS
            word_start += chunk_offset
            word_end += chunk_offset

            if (current_speaker == speaker_label
                    and (word_start - current_end) < 2.0):
                # Extend current segment
                current_end = word_end
                current_words.append(word.word)
            else:
                # Flush previous segment, start new one
                _flush_current()
                current_speaker = speaker_label
                current_start = word_start
                current_end = word_end
                current_words = [word.word]

    _flush_current()

    # Convert raw dicts to SpeakerSegment models
    segments = [
        SpeakerSegment(start=s["start"], end=s["end"], text=s["text"])
        for s in raw_segments
    ]

    return " ".join(full_text_parts), segments, raw_segments


async def transcribe_with_diarization(
    audio_bytes: bytes,
    mime_type: str = "audio/webm",
    filename: str = "recording.webm",
) -> Tuple[str, Optional[List[SpeakerLabel]]]:
    """Transcribe audio with speaker diarization.

    Primary: Google Cloud Speech-to-Text with diarization enabled.
    Fallback: litellm STT router (no diarization).

    Returns (full_transcript, optional_speaker_labels).
    """
    client = _get_google_client()

    if client is not None:
        return await _transcribe_google(client, audio_bytes, mime_type, filename)

    # Fallback to existing STT router
    logger.info("Google STT unavailable — falling back to litellm STT router (no diarization)")
    return await _transcribe_fallback(audio_bytes, mime_type, filename)


async def _transcribe_google(
    client,
    audio_bytes: bytes,
    mime_type: str,
    filename: str,
) -> Tuple[str, Optional[List[SpeakerLabel]]]:
    """Full Google STT pipeline with chunking and diarization merging."""
    language_code = _detect_language_code(mime_type, filename)
    num_chunks = _estimate_chunk_count(audio_bytes, mime_type)

    logger.info(
        "Google STT: transcribing %d bytes (%s) language=%s chunks=%d",
        len(audio_bytes), mime_type, language_code, num_chunks,
    )

    # For small files, single call with diarization
    if num_chunks <= 1 and len(audio_bytes) <= MAX_CHUNK_BYTES:
        try:
            full_text, segments, raw_segments = await _transcribe_chunk_google(
                client, audio_bytes, mime_type, language_code, chunk_index=0,
            )
            speaker_labels = _build_speaker_labels(raw_segments)
            return full_text, speaker_labels
        except Exception as exc:
            logger.error("Google STT single-chunk failed: %s — falling back", exc)
            return await _transcribe_fallback(audio_bytes, mime_type, filename)

    # For large files: chunk and transcribe in parallel batches
    chunks = _split_audio_bytes(audio_bytes, num_chunks)
    logger.info("Google STT: split into %d chunks (~%d bytes each)", len(chunks), len(chunks[0]) if chunks else 0)

    # Process in parallel batches of 4 to avoid rate limits
    semaphore = asyncio.Semaphore(4)
    all_raw_segments: List[dict] = []
    all_texts: List[str] = []

    async def process_chunk(i: int, chunk: bytes) -> Tuple[int, str, List[dict]]:
        async with semaphore:
            try:
                text, _, raw_segs = await _transcribe_chunk_google(
                    client, chunk, mime_type, language_code, chunk_index=i,
                )
                return i, text, raw_segs
            except Exception as exc:
                logger.error("Google STT chunk %d/%d failed: %s", i + 1, len(chunks), exc)
                return i, "", []

    tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    # Merge results in chunk order
    results.sort(key=lambda r: r[0])
    for _, text, raw_segs in results:
        all_texts.append(text)
        all_raw_segments.extend(raw_segs)

    full_text = " ".join(all_texts)
    speaker_labels = _build_speaker_labels(all_raw_segments) if all_raw_segments else None

    return full_text, speaker_labels


def _build_speaker_labels(raw_segments: List[dict]) -> List[SpeakerLabel]:
    """Group raw segments (dicts with 'speaker' key) into SpeakerLabel models."""
    speaker_map: dict = {}
    for s in raw_segments:
        speaker = s.get("speaker", "Speaker 1")
        if speaker not in speaker_map:
            speaker_map[speaker] = SpeakerLabel(speaker=speaker, segments=[])
        seg = SpeakerSegment(start=s["start"], end=s["end"], text=s["text"])
        speaker_map[speaker].segments.append(seg)
    return list(speaker_map.values())


async def _transcribe_fallback(
    audio_bytes: bytes,
    mime_type: str,
    filename: str,
) -> Tuple[str, Optional[List[SpeakerLabel]]]:
    """Fallback transcription using the existing litellm STT router (no diarization)."""
    from .nlu import transcribe_audio

    try:
        transcript = await transcribe_audio(
            filename, io.BytesIO(audio_bytes), mime_type,
        )
        return transcript, None  # No speaker labels from fallback
    except Exception as exc:
        logger.exception("Fallback STT failed")
        raise
