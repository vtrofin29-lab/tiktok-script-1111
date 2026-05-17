def opacity_fade(t, start, dur, fade_in, fade_out):
    """Return opacity (0..1) at time t for a clip that starts at `start`, has duration `dur`,
    fades in over `fade_in` seconds and fades out over `fade_out` seconds."""
    if t < start:
        return 0.0
    rel = t - start
    # fade in
    if fade_in and rel < fade_in:
        return float(rel) / float(fade_in)
    # fade out
    if fade_out and rel > (dur - fade_out):
        val = float(dur - rel) / float(fade_out)
        return max(0.0, min(1.0, val))
    return 1.0


#!/usr/bin/env python3
"""
tiktok_full_gui.py

Versiune completă integrată:
- GUI responsivă cu PanedWindow
- Mini-preview cu draggable crop lines
- Timeline slider + time label
- Save/Load crop settings
- Job queue (multiple jobs) + unique outputs (_1, _2 ... if exist)
- Pre-render foreground via ffmpeg (NVENC if available)
- Robust Whisper loading (retry + cleanup + fallback to 'small')
- Caption grouping (WORDS_PER_GROUP) + Bangers font loading + diacritics normalization
- Static blurred background from a mid frame (fast)
- Monitored export with stall detection and fallback attempts
- Auto-open output on success
"""
import os


def find_ttf_in_tiktok(font_name, search_root=r"C:\tiktok"):
    """
    Search recursively in search_root for a .ttf/.otf file whose filename matches font_name.
    Matching is case-insensitive and ignores non-alphanumeric characters (so '-' and spaces are ignored).
    Returns the first matching path or None.
    """
    try:
        if not font_name:
            return None
        import re, os
        def _norm(s):
            return re.sub(r'[^a-z0-9]', '', s.lower())
        token = _norm(font_name)
        for root, dirs, files in os.walk(search_root):
            for fn in files:
                if fn.lower().endswith((".ttf", ".otf")):
                    name_only = os.path.splitext(fn)[0]
                    name_norm = _norm(name_only)
                    if not token:
                        continue
                    if token in name_norm or name_norm in token:
                        return os.path.join(root, fn)
        return None
    except Exception:
        return None


def validate_font_path_matches_name(font_path, font_name):
    """
    Check if a font file path corresponds to the expected font name.
    Returns True if the path matches the font name (case-insensitive).
    This prevents stale font paths from being used when user changes fonts.
    """
    if not font_path or not font_name:
        return False
    try:
        import re, os
        font_name_norm = re.sub(r'[^a-z0-9]', '', font_name.lower())
        path_basename = os.path.basename(font_path).lower()
        path_name_norm = re.sub(r'[^a-z0-9]', '', os.path.splitext(path_basename)[0])
        return font_name_norm in path_name_norm or path_name_norm in font_name_norm
    except Exception:
        return False


def get_validated_font(selected_font_name, selected_font_path):
    """
    Get the correct font identifier to use, validating that path matches name.
    Returns font path if it matches the font name, otherwise returns the font name.
    """
    if selected_font_path and selected_font_name:
        if validate_font_path_matches_name(selected_font_path, selected_font_name):
            return selected_font_path
        else:
            # Mismatch - path is from old font. Use font name for search.
            return selected_font_name
    elif selected_font_path:
        return selected_font_path
    elif selected_font_name:
        return selected_font_name
    return None


import sys
import threading
import queue
import subprocess
import math
import time
import tempfile
import shutil
import json
import gc

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from pathlib import Path

import numpy as np
# --- Pillow compatibility shim: ensure Image.ANTIALIAS exists for older code / moviepy ---
try:
    from PIL import Image as _Image_for_compat
    if not hasattr(_Image_for_compat, 'ANTIALIAS'):
        try:
            # Pillow >= 10: use Resampling.LANCZOS
            _Image_for_compat.ANTIALIAS = _Image_for_compat.Resampling.LANCZOS
        except Exception:
            # fallback to Image.LANCZOS if available
            if hasattr(_Image_for_compat, 'LANCZOS'):
                _Image_for_compat.ANTIALIAS = _Image_for_compat.LANCZOS
            else:
                # last resort: set to None (resizing will still work but quality setting not applied)
                _Image_for_compat.ANTIALIAS = None
except Exception:
    pass

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageTk

# External heavy deps: moviepy (loaded eagerly), whisper+torch (loaded lazily on first use)
from moviepy.editor import (
    VideoFileClip, CompositeVideoClip, AudioFileClip, ImageClip,
    concatenate_videoclips, concatenate_audioclips
)
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.video.fx.all import speedx
from moviepy.audio.fx.all import audio_fadeout

# whisper and torch are loaded lazily (inside _load_whisper_model_with_retries /
# _release_whisper_model) so the app starts faster and these heavy modules
# (~5-10 s import time) are only pulled in when transcription is actually needed.

# -------- TRANSLATION & AI VOICE MODULES --------
try:
    from google trans import Translator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    Translator = None

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    gTTS = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

# -------- TRANSLATION FUNCTIONS --------
def translate_text(text, target_language='en', log=None):
    """
    Translate text to target language using Google Translate API.
    
    Args:
        text: Text to translate
        target_language: Target language code (e.g., 'en', 'es', 'fr', 'ro')
        log: Optional logging function
    
    Returns:
        Translated text or original text if translation fails
    """
    if not TRANSLATION_AVAILABLE:
        if log:
            log("[TRANSLATE] googletrans not available - skipping translation")
        return text
    
    if not text or not text.strip():
        return text
    
    try:
        try:
            translator = Translator(timeout=15)
        except TypeError:
            translator = Translator()
        result = translator.translate(text, dest=target_language)
        if result and result.text:
            if log:
                log(f"[TRANSLATE] '{text[:50]}...' -> '{result.text[:50]}...' ({target_language})")
            return result.text
        return text
    except Exception as e:
        if log:
            log(f"[TRANSLATE ERROR] Failed to translate: {e}")
        return text


def _openai_translate_segments(segments, target_language='en', log=None):
    """
    Translate caption segments using OpenAI for natural,
    context-aware translations that avoid repetition.
    
    Sends all segments as numbered lines so the model can see full context
    and produce natural translations (e.g. using pronouns, "the same", etc.
    instead of repeating identical phrases).
    
    Args:
        segments: List of caption segments with 'text' keys
        target_language: Target language code
        log: Optional logging function
    
    Returns:
        List of translated text strings (same order as input segments),
        or None if translation fails (caller should fall back).
    """
    api_key = globals().get('OPENAI_API_KEY')
    if not api_key:
        return None
    
    if not REQUESTS_AVAILABLE:
        if log:
            log("[OpenAI TRANSLATE] requests library not available")
        return None
    
    if not text or not text.strip():
        return text
    
    try:
        try:
            translator = Translator(timeout=15)
        except TypeError:
            translator = Translator()
        result = translator.translate(text, dest=target_language)
        if result and result.text:
            if log:
                log(f"[TRANSLATE] '{text[:50]}...' -> '{result.text[:50]}...' ({target_language})")
            return result.text
        return text
    except Exception as e:
        if log:
            log(f"[TRANSLATE ERROR] Failed to translate: {e}")
        return text


def translate_segments(segments, target_language='en', log=None):
    """
    Translate all caption segments to target language.
    
    Uses OpenAI when OPENAI_API_KEY is set for natural,
    context-aware translations that avoid repetition.
    Falls back to googletrans batch translation (with ||| separator
    for context), then per-segment translation as last resort.
    
    Args:
        segments: List of caption segments from Whisper
        target_language: Target language code
        log: Optional logging function
    
    Returns:
        List of segments with translated text
    """
    if target_language == 'none':
        return segments
    
    if not segments:
        return segments
    
    if log:
        log(f"[TRANSLATE] Translating {len(segments)} segments to {target_language}...")
    
    # --- Strategy 1: OpenAI contextual translation (best quality) ---
    api_key = globals().get('OPENAI_API_KEY')
    if api_key and REQUESTS_AVAILABLE:
        if log:
            log(f"[TRANSLATE] Using OpenAI {globals().get('OPENAI_MODEL', 'gpt-4o-mini')} (API key: ...{api_key[-4:]})")
            log(f"[TRANSLATE] Check API usage at: https://platform.openai.com/usage")
        openai_results = _openai_translate_segments(segments, target_language, log=log)
        if openai_results:
            translated = []
            for i, seg in enumerate(segments):
                new_seg = seg.copy()
                new_seg["original_text"] = seg.get("text", "")
                new_seg["text"] = openai_results[i]
                translated.append(new_seg)
            if log:
                log(f"[TRANSLATE] ✓ OpenAI {globals().get('OPENAI_MODEL', 'gpt-4o-mini')} translation complete!")
            return translated
        if log:
            log("[TRANSLATE] ✗ OpenAI translation failed ✗ falling back to googletrans...")
            log("[TRANSLATE] Check logs above for details (429=needs billing credit, other=see error)")
    elif not api_key and log:
        log("[TRANSLATE] No OpenAI API key set - using googletrans (free, lower quality)")
    
    if not TRANSLATION_AVAILABLE:
        if log:
            log("[TRANSLATE] googletrans not available - skipping translation")
        return segments
    
    # --- Strategy 2: Batch googletrans with ||| separator for context ---
    try:
        texts = [seg.get("text", "").strip() for seg in segments]
        batch_text = " ||| ".join(texts)
        
        try:
            translator = Translator(timeout=15)
        except TypeError:
            translator = Translator()
        result = translator.translate(batch_text, dest=target_language)
        
        if result and result.text:
            parts = result.text.split("|||")
            if len(parts) >= len(segments):
                translated = []
                for i, seg in enumerate(segments):
                    new_seg = seg.copy()
                    new_seg["original_text"] = seg.get("text", "")
                    new_seg["text"] = parts[i].strip()
                    translated.append(new_seg)
                if log:
                    log("[TRANSLATE] Batch translation complete!")
                return _reduce_translation_repetition(translated, log=log)
            else:
                if log:
                    log(f"[TRANSLATE] Batch split mismatch ({len(parts)} vs {len(segments)}), falling back to per-segment")
        else:
            if log:
                log("[TRANSLATE] Batch translation returned empty result, falling back to per-segment")
    except Exception as e:
        if log:
            log(f"[TRANSLATE] Batch translation failed ({e}), falling back to per-segment")
    
    # --- Strategy 3: Per-segment fallback ---
    translated = []
    fail_count = 0
    for i, seg in enumerate(segments):
        try:
            original_text = seg.get("text", "")
            translated_text = translate_text(original_text, target_language, log=None)
            
            new_seg = seg.copy()
            new_seg["text"] = translated_text
            new_seg["original_text"] = original_text
            translated.append(new_seg)
            
            # Track when translation returns the same text (likely failed silently)
            if translated_text == original_text and original_text.strip():
                fail_count += 1
            else:
                fail_count = 0  # Reset on success
        except Exception as e:
            fail_count += 1
            if log:
                log(f"[TRANSLATE ERROR] Failed segment {i}: {e}")
            translated.append(seg)
        
        # If too many segments fail in a row, stop trying (googletrans is broken)
        if fail_count >= 3:
            if log:
                log(f"[TRANSLATE] ✗ {fail_count} consecutive failures ✗ googletrans appears broken, keeping remaining segments untranslated")
            for remaining_seg in segments[i+1:]:
                new_seg = remaining_seg.copy()
                new_seg["original_text"] = remaining_seg.get("text", "")
                translated.append(new_seg)
            break
    
    if log:
        if fail_count > 0:
            log(f"[TRANSLATE] Translation complete with {fail_count} failed segment(s) (kept original text)")
        else:
            log(f"[TRANSLATE] Translation complete!")
    
    return _reduce_translation_repetition(translated, log=log)


def _reduce_translation_repetition(segments, log=None):
    """
    Post-process translated segments to reduce repetitive phrasing.
    
    When consecutive segments share substantial text, replaces the
    repeated portion with shorter references (e.g., 'la fel', 'the same',
    'de asemena', etc.) based on target language detection.
    
    This is used for the googletrans path which lacks contextual awareness.
    OpenAI translations already handle this via the system prompt.
    """
    if not segments or len(segments) < 2:
        return segments
    
    import re
    
    for i in range(1, len(segments)):
        prev_text = segments[i - 1].get("text", "").strip()
        curr_text = segments[i].get("text", "").strip()
        
        if not prev_text or not curr_text:
            continue
        
        # Normalize for comparison: lowercase, strip punctuation
        def _norm(t):
            if not isinstance(t, str):
                return ''
            return re.sub(r'[^\w\s]', '', t.lower()).strip()
        
        prev_norm = _norm(prev_text)
        curr_norm = _norm(curr_text)
        
        # Skip very short segments (less than 4 words)
        prev_words = prev_norm.split()
        curr_words = curr_norm.split()
        if len(curr_words) < 4 or len(prev_words) < 4:
            continue
        
        # Check if segments are very similar (>70% word overlap)
        prev_set = set(prev_words)
        curr_set = set(curr_words)
        if not prev_set or not curr_set:
            continue
        overlap = len(prev_set & curr_set) / max(len(prev_set), len(curr_set))
        
        if overlap < 0.7:
            continue
        
        # Find the differing words between segments
        diff_words = []
        for w in curr_words:
            if w not in prev_set:
                diff_words.append(w)
        
        if not diff_words:
            # Segments are essentially identical - use "la fel" / "the same"
            segments[i]["text"] = _build_short_reference(curr_text, prev_text, diff_words=[])
        elif len(diff_words) <= 2:
            # Only 1-2 words differ - build a concise reference
            segments[i]["text"] = _build_short_reference(curr_text, prev_text, diff_words=diff_words)
        # else: too many differences, keep original
    
    if log:
        log("[TRANSLATE] Applied anti-repetition post-processing")
    
    return segments


def _build_short_reference(curr_text, prev_text, diff_words):
    """
    Build a shorter version of curr_text that references prev_text.
    
    Examples:
    - "He was 20" / "She was 20" -> "She, too" or "She la fel"
    - "El avea 20 de ani" / "Ea avea 20 de ani" -> "Ea la fel"
    """
    import re
    
    # Detect language heuristics based on common words (word boundary matching)
    words_set = set(re.findall(r'\b\w+\b', curr_text.lower()))
    
    # Romanian detection
    ro_markers = {'îi', 'este', 'avea', 'ani', 'de', 'la', 'că', 'pentru'}
    is_romanian = len(words_set & ro_markers) >= 2
    
    # Spanish detection
    es_markers = {'ella', 'también', 'tenía', 'año', 'de', 'lo', 'como'}
    is_spanish = not is_romanian and len(words_set & es_markers) >= 2
    
    # French detection
    fr_markers = {'elle', 'aussi', 'avait', 'mais', 'comme', 'les'}
    is_french = not is_romanian and not is_spanish and len(words_set & fr_markers) >= 2
    
    # German detection
    de_markers = {'sie', 'auch', 'hatte', 'aber', 'wie', 'und'}
    is_german = not is_romanian and not is_spanish and not is_french and len(words_set & de_markers) >= 2
    
    if not diff_words:
        # Identical segments
        if is_romanian:
            return "La fel"
        elif is_spanish:
            return "Lo mismo"
        elif is_french:
            return "Pareil"
        elif is_german:
            return "Genauso"
        else:
            return "The same"
    
    # Build reference with the differing subject + short connector
    subject = " ".join(diff_words)
    
    if is_romanian:
        return f"{subject.capitalize()} la fel"
    elif is_spanish:
        return f"{subject.capitalize()} también"
    elif is_french:
        return f"{subject.capitalize()} aussi"
    elif is_german:
        return f"{subject.capitalize()} auch"
    else:
        return f"{subject.capitalize()} too"


# ---- GenAI Pro API constraints
GENAI_PRO_API_BASE = 'https://genai pro.io/api'  # API base URL – endpoints live under /api/v1/labs/task
GENAI_PRO_MIN_SPEED = 0.7      # Minimum TTS speed accepted by the API
GENAI_PRO_MAX_SPEED = 1.2      # Maximum TTS speed accepted by the API
GENAI_PRO_MAX_WAIT_SECONDS = float('inf')    # Max time to wait for a TTS task (infinite)

# Number of voice jobs to submit to GenAI Pro simultaneously.
# Higher = faster overall queue, but too many concurrent tasks may get rate-limited.
# GenAI Pro doesn't publish a hard limit; 3 is a safe default.
VOICE_SUBMISSION_CONCURRENCY = 3


def _genai pro_session(requests_module):
    """Return a requests.Session that preserves the Authorization header across redirects.
    Python's default behaviour strips it when following cross-domain redirects, which
    causes genai pro.io to report token_is_empty even when a valid JWT is supplied."""
    class _KeepAuthSession(requests_module.Session):
        def rebuild_auth(self, prepared_request, response):
            # Do NOT strip the Authorization header on redirect
            pass
    return _KeepAuthSession()


def _find_task_in_genai pro_response(data, task_id):
    """
    Locate a specific task in any response shape returned by genai pro.io.
    Handles multiple formats:
      - Per-task dict:              {"id": "...", "status": "...", "result": "..."}
      - Wrapped single dict:        {"data": {"id": "...", "status": "..."}, "success": true}
      - Wrapped list:               {"tasks": [...], "total": N, ...}
      - Bare list:                  [{"id": "...", ...}, ...]
    Returns the task dict or None.
    """
    if isinstance(data, dict):
        # Case 1: looks like a task object itself (has known id field at top level)
        tid = data.get('id') or data.get('task_id')
        if tid is not None and (task_id is None or str(tid) == str(task_id)):
            return data
        # Case 2: unwrap envelope keys – value may be a list OR a single task dict
        for key in ('tasks', 'data', 'items', 'results', 'task'):
            inner = data.get(key)
            if isinstance(inner, list):
                for task in inner:
                    if isinstance(task, dict):
                        tid = task.get('id') or task.get('task_id')
                        if tid is not None and str(tid) == str(task_id):
                            return task
            elif isinstance(inner, dict):
                # Single task wrapped in an envelope key
                tid = inner.get('id') or inner.get('task_id')
                if tid is not None and (task_id is None or str(tid) == str(task_id)):
                    return inner
        # Case 3: the response has 'status' but no matching id field
        # (per-task endpoint may omit the id in the body)
        if 'status' in data:
            return data
    elif isinstance(data, list):
        for task in data:
            if isinstance(task, dict):
                tid = task.get('id') or task.get('task_id')
                if tid is not None and str(tid) == str(task_id):
                    return task
    return None


def _select_submission_task(tasks, expected_input=None, expected_voice_id=None, exclude_task_id=None):
    """Select the best task candidate from a submission/list response."""
    if not isinstance(tasks, list):
        return None
    candidates = [t for t in tasks if isinstance(t, dict)]
    if not candidates:
        return None

    # When retrying, exclude the known stale task so we don't re-select it.
    if exclude_task_id is not None:
        exclude_str = str(exclude_task_id)
        candidates = [t for t in candidates
                          if str(t.get('id') or t.get('task_id') or '') != exclude_str]
        if not candidates:
            return None

    # Prefer exact input+voice match (for nonce re-submit flows)
    if expected_input is not None:
        exact = [t for t in candidates if t.get('input') == expected_input]
        if expected_voice_id is not None:
            exact_voice = [t for t in exact if t.get('voice_id') == expected_voice_id]
            if exact_voice:
                exact = exact_voice
        if exact:
            candidates = exact
        else:
            # Fallback: compare after removing zero-width chars.
            def _norm(s):
                if not isinstance(s, str):
                    return ''
                return s.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\ufeff', '').strip()
            norm_expected = _norm(expected_input)
            approx = [t for t in candidates if _norm(t.get('input')) == norm_expected]
            if expected_voice_id is not None:
                approx_voice = [t for t in approx if t.get('voice_id') == expected_voice_id]
                if approx_voice:
                    approx = approx_voice
            if approx:
                candidates = approx

    # Prefer most recent task when list contains historical entries.
    def _time_key(task):
        return (
            str(task.get('updated_at') or ''),
            str(task.get('created_at') or ''),
            str(task.get('id') or task.get('task_id') or '')
        )

    return max(candidates, key=_time_key)


def _extract_submission_task_id(task_data, expected_input=None, expected_voice_id=None, exclude_task_id=None):
    """Extract a task_id from GenAI Pro submit response across all known shapes."""
    if isinstance(task_data, dict):
        task_id = task_data.get('task_id') or task_data.get('id')
        if task_id:
            return task_id, task_data

        if isinstance(task_data.get('data'), dict):
            inner = task_data['data']
            task_id = inner.get('task_id') or inner.get('id')
            if task_id:
                return task_id, inner

        selected = _select_submission_task(
            task_data.get('tasks'),
            expected_input=expected_input,
            expected_voice_id=expected_voice_id,
            exclude_task_id=exclude_task_id
        )
        if selected:
            return selected.get('task_id') or selected.get('id'), selected

    elif isinstance(task_data, list):
        selected = _select_submission_task(
            task_data,
            expected_input=expected_input,
            expected_voice_id=expected_voice_id,
            exclude_task_id=exclude_task_id
        )
        if selected:
            return selected.get('task_id') or selected.get('id'), selected

    return None, None


def _refresh_genai pro_result_url(session, headers, task_id, log=None, log_prefix='[GenAI Pro]'):
    """Re-fetch an existing task and return the newest result/audio URL, if available."""
    if not task_id:
        return None
    if log:
        log(f"{log_prefix} Refreshing task result URL from existing task_id before nonce re-submit...")
    urls = (
        f'{GENAI_PRO_API_BASE}/v1/labs/task/{task_id}',
        f'{GENAI_PRO_API_BASE}/v1/labs/task?limit=100',
    )
    for url in urls:
        try:
            resp = session.get(url, headers=headers, timeout=60)
            if resp.status_code not in (200, 201, 202):
                continue
            data = resp.json()
            task = _find_task_in_genai pro_response(data, task_id)
            if not isinstance(task, dict):
                continue
            refreshed_url = (
                task.get('result') or task.get('output_url') or task.get('audio_url') or
                task.get('result_url') or task.get('file_url') or task.get('url')
            )
            if isinstance(refreshed_url, str) and refreshed_url.startswith('http'):
                return refreshed_url
        except Exception:
            continue
    return None


def _build_genai pro_nonce_input(base_input, nonce_attempt=1):
    """Build an invisible, progressively stronger nonce marker for GenAI Pro dedupe bypass."""
    if not isinstance(base_input, str):
        base_input = '' if base_input is None else str(base_input)
    try:
        n = int(nonce_attempt)
    except Exception:
        n = 1
    n = max(1, n)
    # Mix multiple zero-width code points so retries are distinct even if one is normalized away.
    # Place nonce at the start and early inside the text as well (not just tail) so it still
    # affects dedupe even when very long text is truncated server-side.
    prefix = ('\u2060' * n) + ('\u200B' if n % 2 else '\u200C')
    mid = ('\u200B' * n) + ('\u200C' if n % 2 else '\u200D')
    suffix = ('\u200D' * min(n, 4))
    text = base_input.rstrip()
    if not text:
        return prefix + mid + suffix
    insert_at = min(max(1, len(text) // 3), 64)
    return prefix + text[:insert_at] + mid + text[insert_at:] + suffix


_GENAI_PRO_FALLBACK_MODELS = [
    'eleven_turbo_v2_5',
    'eleven_multilingual_v2',
    'eleven_turbo_v2',
    'eleven_flash_v2_5',
    'eleven_flash_v2',
    'eleven_turbo_v2_5',
]


def _build_genai pro_retry_payload(base_payload, retry_attempt=1):
    """Build a fresh retry payload that avoids server-side dedupe collisions."""
    payload = dict(base_payload or {})
    try:
        attempt = max(1, int(retry_attempt))
    except Exception:
        attempt = 1

    # Always change text with a stronger invisible nonce.
    payload['input'] = _build_genai pro_nonce_input(payload.get('input'), attempt)

    # Cycle through model IDs – the API deduplicates on (text, voice_id, model_id)
    # so switching models reliably forces a new task even when invisible-char nonces
    # are normalized away by the server.
    payload['model_id'] = _GENAI_PRO_FALLBACK_MODELS[(attempt - 1) % len(_GENAI_PRO_FALLBACK_MODELS)]

    # Toggle use_speaker_boost on alternating attempts for an additional dedupe dimension.
    payload['use_speaker_boost'] = bool(attempt % 2)

    # Also slightly vary synthesis knobs so retries are unique even if input is normalized.
    sign = 1 if (attempt % 2) else -1

    def _to_float(value, default):
        try:
            return float(value)
        except Exception:
            return float(default)

    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))

    base_speed = _to_float(payload.get('speed', 1.0), 1.0)
    base_stability = _to_float(payload.get('stability', 0.5), 0.5)
    base_similarity = _to_float(payload.get('similarity', 0.75), 0.75)
    base_style = _to_float(payload.get('style', 0.0), 0.0)
    step = min(0.05, 0.01 * attempt)

    payload['speed'] = round(_clamp(base_speed + (sign * step), GENAI_PRO_MIN_SPEED, GENAI_PRO_MAX_SPEED), 3)
    payload['stability'] = round(_clamp(base_stability + (sign * step), 0.0, 1.0), 3)
    payload['similarity'] = round(_clamp(base_similarity - (sign * step), 0.0, 1.0), 3)
    payload['style'] = round(_clamp(base_style + (0.5 * step), 0.0, 1.0), 3)
    return payload


def _normalize_genai pro_api_key(api_key):
    """Normalize a GenAI Pro JWT pasted by the user."""
    if api_key is None:
        return None
    key = str(api_key).strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key or None


def generate_tts_with_genai pro(text, language='en', output_path=None, api_key=None, log=None):
    """
    Generate Text-to-Speech audio using GenAI Pro API.
    
    Args:
        text: Text to convert to speech
        language: Language code for TTS
        output_path: Path to save audio file (temp file if None)
        api_key: GenAI Pro API key (JWT token)
        log: Optional logging function
    
    Returns:
        Path to generated audio file or None if failed
    """
    if not REQUESTS_AVAILABLE:
        if log:
            log("[GenAI Pro] requests library not available")
        return None
    
    api_key = _normalize_genai pro_api_key(api_key)
    if not api_key:
        if log:
            log("[GenAI Pro] No API key provided")
        return None
    
    if not text or not text.strip():
        return None
    
    try:
        import requests
        import time

        session = _genai pro_session(requests)
        
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.mp3', prefix='genai pro_tts_')
            os.close(fd)
        
        # Map language codes to voice IDs (you can expand this mapping)
        # Check if a custom voice ID is specified, otherwise use language defaults
        global TTS_VOICE_ID
        if TTS_VOICE_ID and TTS_VOICE_ID != 'auto':
            voice_id = TTS_VOICE_ID
        else:
            voice_map = {
                'en': 'uju3wxzG5OhpWcoi3SMy',  # Default English voice
                'es': 'uju3wxzG5OhpWcoi3SMy',  # Using same for now, can be customized
                'fr': 'uju3wxzG5OhpWcoi3SMy',
                'de': 'uju3wxzG5OhpWcoi3SMy',
                'it': 'uju3wxzG5OhpWcoi3SMy',
                'pt': 'uju3wxzG5OhpWcoi3SMy',
                'ro': 'uju3wxzG5OhpWcoi3SMy',
                'ru': 'uju3wxzG5OhpWcoi3SMy',
                'zh': 'uju3wxzG5OhpWcoi3SMy',
                'ja': 'uju3wxzG5OhpWcoi3SMy',
                'ko': 'uju3wxzG5OhpWcoi3SMy',
            }
            voice_id = voice_map.get(language, 'uju3wxzG5OhpWcoi3SMy')
        
        # GenAI Pro API requires speed between 0.7 and 1.2
        tts_speed = max(GENAI_PRO_MIN_SPEED, min(GENAI_PRO_MAX_SPEED, globals().get('TTS_SPEED', 1.0)))

        task_payload = {
            'input': text,
            'voice_id': voice_id,
            'model_id': 'eleven_turbo_v2_5',  # Fast model
            'speed': tts_speed,
            'style': 0.0,
            'use_speaker_boost': False,
            'similarity': 0.75,
            'stability': 0.5
        }
        nonce_retry_count = 0
        max_nonce_retries = 6
        
        if log:
            token_preview = api_key[:8] + "..." if len(api_key) > 8 else "(short)"
            log(f"[GenAI Pro] Submitting TTS task ({len(text)} chars)... [token: {token_preview}]")
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Step 1: Submit TTS task
        if log:
            log(f"[GenAI Pro] Submitting TTS task ({len(text)} chars)... [token: {token_preview}]")
        
        response = session.post(
            f'{GENAI_PRO_API_BASE}/v1/labs/task',
            headers=headers,
            json=task_payload,
            timeout=30
        )
        
        if response.status_code not in (200, 201, 202):
            if log:
                err_body = response.text[:300]
                log(f"[GenAI Pro ERROR] Task submission failed: {response.status_code} - {err_body}")
                if response.status_code == 401:
                    log("[GenAI Pro ERROR] ✗ Authentication failed ✗ your GenAI Pro JWT token has expired or is invalid.")
                    log("[GenAI Pro ERROR]    Please log in again at https://genai pro.io, copy your new JWT token,")
                    log("[GenAI Pro ERROR]      and paste it in the 'TTS API Key' field, then click 'Save API Key'.")
            return None
        
        try:
            task_data = response.json()
        except Exception:
            if log:
                log(f"[GenAI Pro ERROR] Non-JSON submission response (status {response.status_code}), first 200 bytes: {response.content[:200]}")
            return None

        if log:
            log(f"[GenAI Pro] Submission response ({response.status_code}): {str(task_data)[:400]}")

        # Handle synchronous response: API may return audio URL directly (no polling needed)
        direct_audio_url = (task_data.get('audio_url') or task_data.get('output_url') or
                            task_data.get('result_url') or task_data.get('file_url') or
                            task_data.get('url'))
        # Also unwrap one level of "data" envelope for direct URL
        if not direct_audio_url and isinstance(task_data.get('data'), dict):
            inner = task_data['data']
            direct_audio_url = (inner.get('audio_url') or inner.get('output_url') or
                               inner.get('result_url') or inner.get('file_url') or
                               inner.get('url'))
        if direct_audio_url and direct_audio_url.startswith('http'):
            if log:
                log(f"[GenAI Pro] ◆ Synchronous response ✗ audio URL returned immediately, downloading...")
            try:
                audio_response = requests.get(direct_audio_url, timeout=120)
                if audio_response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(audio_response.content)
                    if log:
                        log(f"[GenAI Pro] ✓ TTS generated successfully (sync): {output_path}")
                    return output_path
                elif log:
                    log(f"[GenAI Pro ERROR] Direct audio download failed: HTTP {audio_response.status_code}")
            except Exception as dl_err:
                if log:
                    log(f"[GenAI Pro] ✗ Direct audio download error: {dl_err}")
            return None

        # GenAI Pro might return direct id, nested id, or a paginated list of tasks.
        task_id, selected_submission_task = _extract_submission_task_id(
            task_data,
            expected_input=text,
            expected_voice_id=voice_id
        )
        _submission_tasks = task_data['tasks'] if isinstance(task_data.get('tasks'), list) else []

        if not task_id:
            if log:
                log(f"[GenAI Pro ERROR] No task_id in response: {str(task_data)[:400]}")
            return None

        if log:
            log(f"[GenAI Pro] Task submitted with ID: {task_id}")

        # If the submission response already contains a completed task with an audio URL, download immediately
        if _submission_tasks and selected_submission_task:
            _sub = selected_submission_task
            _sub_status = _sub.get('status', '').lower()
            _sub_result = (_sub.get('result') or _sub.get('audio_url') or
                          _sub.get('output_url') or _sub.get('file_url') or _sub.get('url') or '')
            if ((_sub_status in ['completed', 'done', 'success', 'succeeded', 'finished'] or _sub_result)
                    and isinstance(_sub_result, str) and _sub_result.startswith('http')):
                if log:
                    log(f"[GenAI Pro] ◆ Task already completed in submission response ✗ downloading immediately...")
                try:
                    audio_response = requests.get(_sub_result, timeout=120)
                    if audio_response.status_code == 200:
                        with open(output_path, 'wb') as f:
                            f.write(audio_response.content)
                        if log:
                            log(f"[GenAI Pro] ✓ TTS generated successfully (cached): {output_path}")
                        return output_path
                    elif log:
                        log(f"[GenAI Pro ERROR] Cached audio download failed: HTTP {audio_response.status_code} ✗ result URL likely expired, re-submitting with nonce")
                except Exception as dl_err:
                    if log:
                        log(f"[GenAI Pro ERROR] Cached audio download error: {dl_err} ✗ result URL likely expired, re-submitting with nonce")
                    refreshed_url = _refresh_genai pro_result_url(
                        session=session,
                        headers=headers,
                        task_id=task_id,
                        log=log,
                        log_prefix='[GenAI Pro]'
                    )
                    if refreshed_url:
                        try:
                            audio_response = requests.get(refreshed_url, timeout=120)
                            if audio_response.status_code == 200:
                                with open(output_path, 'wb') as f:
                                    f.write(audio_response.content)
                                if log:
                                    log(f"[GenAI Pro] ✓ TTS generated successfully (refreshed task URL): {output_path}")
                                return output_path
                            elif log:
                                log(f"[GenAI Pro ERROR] Refreshed URL download failed: HTTP {audio_response.status_code} ✗ trying nonce re-submit")
                        except Exception as refresh_dl_err:
                            if log:
                                log(f"[GenAI Pro ERROR] Refreshed URL download error: {refresh_dl_err} ✗ trying nonce re-submit")
                        # The cached result URL is expired/broken ✗ force a fresh task via nonce and let polling handle it
                        if log:
                            log("[GenAI Pro] ✗ Result URL expired ✗ re-submitting with nonce to bypass cache...")
                        try:
                            if nonce_retry_count >= max_nonce_retries:
                                if log:
                                    log(f"[GenAI Pro ERROR] Reached max nonce retries ({max_nonce_retries}) without a downloadable file")
                                return None
                            nonce_retry_count += 1
                            nonce_payload = _build_genai pro_retry_payload(task_payload, nonce_retry_count)
                            nonce_resp = session.post(
                                f'{GENAI_PRO_API_BASE}/v1/labs/task',
                                headers=headers,
                                json=nonce_payload,
                                timeout=30
                            )
                            if nonce_resp.status_code == 200:
                                nonce_data = nonce_resp.json()
                                new_id, _ = _extract_submission_task_id(
                                    nonce_data,
                                    expected_input=nonce_payload.get('input'),
                                    expected_voice_id=voice_id,
                                    exclude_task_id=task_id
                                )
                                if not new_id:
                                    if log:
                                        log(f"[GenAI Pro ERROR] Nonce re-submission had no task_id: {nonce_data}")
                                    continue
                                if str(new_id) == str(task_id):
                                    if log:
                                        log(f"[GenAI Pro] Re-submission returned same task ID ({new_id}); retrying with stronger nonce...")
                                    continue
                                task_id = new_id
                                task_payload = nonce_payload
                                # Update task_payload so any further re-submits keep the nonce
                                if log:
                                    log(f"[GenAI Pro] ✓ Fresh task submitted: {new_id}. Waiting for completion...")
                                i = 0
                                consecutive_non200 = 0
                                break  # resume while-True with fresh task_id
                            if log:
                                log(f"[GenAI Pro ERROR] Re-submission failed: HTTP {nonce_resp.status_code} - {nonce_resp.text[:200]}")
                        except Exception as retry_err:
                            if log:
                                log(f"[GenAI Pro ERROR] Re-submission error: {retry_err}")
                        return None
        
        # Step 2: Poll for task completion
        poll_interval = 2  # Poll every 2 seconds
        max_wait_seconds = GENAI_PRO_MAX_WAIT_SECONDS

        if log:
            log(f"[GenAI Pro] Waiting for audio generation... (max {max_wait_seconds // 60}m)")
        consecutive_non200 = 0  # Track repeated failures on per-task endpoint
        i = 0
        while True:
            time.sleep(poll_interval)
            
            elapsed_seconds = (i + 1) * poll_interval
            if elapsed_seconds > max_wait_seconds:
                if log:
                    log(f"[GenAI Pro ERROR] ✕ Timed out after {max_wait_seconds // 60} minutes ✗ aborting")
                return None

            # Show progress every 10 seconds or at start
            if i == 0 or elapsed_seconds % 10 == 0:
                elapsed_mins = elapsed_seconds // 60
                elapsed_secs = elapsed_seconds % 60
                if log:
                    log(f"[GenAI Pro] ✲ Waiting: {elapsed_mins}m {elapsed_secs}s elapsed - Still processing...")
            if elapsed_seconds > 0 and elapsed_seconds % 1800 == 0 and log:
                log(f"[GenAI Pro] ✓✓ Still waiting after {elapsed_seconds // 60} minutes.")
            
            # Try per-task endpoint first; fall back to list endpoint after repeated failures
            use_list_fallback = consecutive_non200 >= 3
            if use_list_fallback:
                poll_url = f'{GENAI_PRO_API_BASE}/v1/labs/task?limit=100'
            else:
                poll_url = f'{GENAI_PRO_API_BASE}/v1/labs/task/{task_id}'

            try:
                status_response = session.get(
                    poll_url,
                    headers=headers,
                    timeout=60
                )
            except Exception as poll_err:
                if log:
                    log(f"[GenAI Pro] ✗ Network error during status check: {poll_err} ✗ retrying...")
                i += 1
                continue
            
            if status_response.status_code not in (200, 201, 202):
                consecutive_non200 += 1
                if log:
                    body_preview = status_response.text[:200] if status_response.text else '(empty)'
                    log(f"[GenAI Pro] ✗ Status check HTTP {status_response.status_code}: {body_preview}")
                    if consecutive_non200 == 3:
                        log("[GenAI Pro] ✗ Per-task endpoint unavailable ✗ switching to list endpoint fallback")
                i += 1
                continue
            
            consecutive_non200 = 0  # Reset on success

            try:
                raw_data = status_response.json()
            except Exception:
                if log:
                    log("[GenAI Pro] ✗ Invalid JSON in status response ✗ retrying...")
                i += 1
                continue
            
            our_task = _find_task_in_genai pro_response(raw_data, task_id)
            if our_task is None:
                if i == 0 and log:
                    log(f"[GenAI Pro] Task {task_id[:12] if len(task_id) > 12 else task_id} not found yet ✗ queued... (raw: {str(raw_data)[:300]})")
                elif i % 30 == 0 and log and i > 0:
                    log(f"[GenAI Pro] Still waiting for task {task_id[:12] if len(task_id) > 12 else task_id}... (raw: {str(raw_data)[:200]})")
                i += 1
                continue

            status = our_task.get('status', '').lower()
            result = our_task.get('result', '')
            
            # Check if task is complete - either by status OR by result field being populated
            # GenAI Pro fills the 'result' field with audio URL when done
            is_complete = (status in ['completed', 'done', 'success', 'succeeded', 'finished'] or 
                          (result and result != ''))
            
            if is_complete:
                elapsed_seconds = (i + 1) * poll_interval
                elapsed_mins = elapsed_seconds // 60
                elapsed_secs = elapsed_seconds % 60
                if log:
                    log(f"[GenAI Pro] ✓ Audio generation complete! (took {elapsed_mins}m {elapsed_secs}s)")
                
                # Try multiple possible field names for the audio URL, including 'result'
                audio_url = (our_task.get('result') or
                            our_task.get('output_url') or 
                            our_task.get('audio_url') or
                            our_task.get('result_url') or
                            our_task.get('file_url') or
                            our_task.get('url'))
                
                if not audio_url:
                    if log:
                        log(f"[GenAI Pro ERROR] Task completed but no audio URL found (status={status})")
                    return None
                
                # Step 3: Download the audio file (with retries)
                if log:
                    log("[GenAI Pro] 🎵 Downloading audio file...")
                
                download_ok = False
                for dl_attempt in range(5):
                    try:
                        audio_response = requests.get(audio_url, timeout=600)
                        if audio_response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                f.write(audio_response.content)
                            download_ok = True
                            break
                        else:
                            if log:
                                log(f"[GenAI Pro] ✗ Download attempt {dl_attempt+1}/5 failed: HTTP {audio_response.status_code}")
                    except Exception as dl_err:
                        if log:
                            log(f"[GenAI Pro] ✗ Download attempt {dl_attempt+1}/5 error: {dl_err}")
                        time.sleep(3)
                
                if download_ok:
                    if log:
                        log(f"[GenAI Pro] ✓ TTS generated successfully: {output_path}")
                    return output_path
                else:
                    refreshed_url = _refresh_genai pro_result_url(
                        session=session,
                        headers=headers,
                        task_id=task_id,
                        log=log,
                        log_prefix='[GenAI Pro]'
                    )
                    if refreshed_url:
                        if log:
                            log("[GenAI Pro] 🎵 Retrying download with refreshed task URL...")
                        for refresh_attempt in range(3):
                            try:
                                refresh_resp = requests.get(refreshed_url, timeout=600)
                                if refresh_resp.status_code == 200:
                                    with open(output_path, 'wb') as f:
                                        f.write(refresh_resp.content)
                                    if log:
                                        log(f"[GenAI Pro] ✓ TTS generated successfully (refreshed task URL): {output_path}")
                                    return output_path
                                elif log:
                                    log(f"[GenAI Pro] Refreshed URL download failed: HTTP {refresh_resp.status_code} ✗ result URL likely expired, re-submitting with nonce")
                            except Exception as refresh_dl_err:
                                if log:
                                    log(f"[GenAI Pro] ✗ Refreshed URL download error: {refresh_dl_err} ✗ re-submitting with nonce")
                        # Result URL has expired (CDN cleanup after a few days).
                        # The API deduplicates by text+voice, so re-submitting the identical
                        # payload would return the same stale task_id. Add an invisible
                        # zero-width space (U+200B) as a nonce to force a genuinely fresh task.
                        if log:
                            log("[GenAI Pro] ✗ Result URL expired ✗ re-submitting with nonce to bypass cache...")
                        try:
                            if nonce_retry_count >= max_nonce_retries:
                                if log:
                                    log(f"[GenAI Pro ERROR] Reached max nonce retries ({max_nonce_retries}) without a downloadable file")
                                return None
                            nonce_retry_count += 1
                            nonce_payload = _build_genai pro_retry_payload(task_payload, nonce_retry_count)
                            retry_resp = session.post(
                                f'{GENAI_PRO_API_BASE}/v1/labs/task',
                                headers=headers,
                                json=nonce_payload,
                                timeout=30
                            )
                            if retry_resp.status_code == 200:
                                retry_data = retry_resp.json()
                                new_id, _ = _extract_submission_task_id(
                                    retry_data,
                                    expected_input=nonce_payload.get('input'),
                                    expected_voice_id=voice_id,
                                    exclude_task_id=task_id
                                )
                                if new_id:
                                    if str(new_id) == str(task_id):
                                        if log:
                                            log(f"[GenAI Pro] Re-submission returned same task ID ({new_id}); retrying with stronger nonce...")
                                        continue
                                    task_id = new_id
                                    task_payload = nonce_payload
                                    # Update task_payload so any further re-submits keep the nonce
                                    if log:
                                        log(f"[GenAI Pro] ✓ Fresh task submitted: {new_id}. Waiting for completion...")
                                    i = 0
                                    consecutive_non200 = 0
                                    continue  # resume while-True with fresh task_id
                                if log:
                                    log(f"[GenAI Pro ERROR] Re-submission response had no task_id: {retry_data}")
                            else:
                                if log:
                                    log(f"[GenAI Pro ERROR] Re-submission failed: HTTP {retry_resp.status_code} - {retry_resp.text[:200]}")
                        except Exception as retry_err:
                            if log:
                                log(f"[GenAI Pro ERROR] Re-submission error: {retry_err}")
                        return None
                
                # Result URL has expired (CDN cleanup after a few days).
                # The API deduplicates by text+voice, so re-submitting the identical
                # payload would return the same stale task_id. Add an invisible
                # zero-width space (U+200B) as a nonce to force a genuinely fresh task.
                if log:
                    log("[GenAI Pro] ✗ Result URL expired ✗ re-submitting with nonce to bypass cache...")
                try:
                    if nonce_retry_count >= max_nonce_retries:
                        if log:
                            log(f"[GenAI Pro ERROR] Reached max nonce retries ({max_nonce_retries}) without a downloadable file")
                        return None
                    nonce_retry_count += 1
                    nonce_payload = _build_genai pro_retry_payload(task_payload, nonce_retry_count)
                    retry_resp = session.post(
                        f'{GENAI_PRO_API_BASE}/v1/labs/task',
                        headers=headers,
                        json=nonce_payload,
                        timeout=30
                    )
                    if retry_resp.status_code == 200:
                        retry_data = retry_resp.json()
                        new_id, _ = _extract_submission_task_id(
                            retry_data,
                            expected_input=nonce_payload.get('input'),
                            expected_voice_id=voice_id,
                            exclude_task_id=task_id
                        )
                        if new_id:
                            if str(new_id) == str(task_id):
                                if log:
                                    log(f"[GenAI Pro] Re-submission returned same task ID ({new_id}); retrying with stronger nonce...")
                                continue
                            task_id = new_id
                            task_payload = nonce_payload
                            # Update task_payload so any further re-submits keep the nonce
                            if log:
                                log(f"[GenAI Pro] ✓ Fresh task submitted: {new_id}. Waiting for completion...")
                            i = 0
                            consecutive_non200 = 0
                            continue  # resume while-True with fresh task_id
                        if log:
                            log(f"[GenAI Pro ERROR] Re-submission response had no task_id: {retry_data}")
                    else:
                        if log:
                            log(f"[GenAI Pro ERROR] Re-submission failed: HTTP {retry_resp.status_code} - {retry_resp.text[:200]}")
                except Exception as retry_err:
                    if log:
                        log(f"[GenAI Pro ERROR] Re-submission error: {retry_err}")
                return None
            
            # Show status periodically for non-completed tasks
            elif i % 10 == 0 and log and i > 0:
                log(f"[GenAI Pro] Status: {status!r} (still processing...)")
            
            i += 1
        
    except Exception as e:
        if log:
            import traceback as _tb
            log(f"[GenAI Pro ERROR] Exception: {e}")
            log(_tb.format_exc())
        return None


def _submit_genai pro_task(text, language='en', api_key=None, log=None):
    """Submit a TTS task to GenAI Pro API and return task_id or None."""
