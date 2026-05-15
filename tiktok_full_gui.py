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
- GUI responsive cu PanedWindow
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

# ----------------- TRANSLATION & AI VOICE MODULES -----------------
try:
    from googletrans import Translator
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

# ----------------- TRANSLATION FUNCTIONS -----------------
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
    
    import requests as _requests
    
    # Build numbered lines from segment texts
    lines = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"{i+1}. {text}")
        else:
            lines.append(f"{i+1}. ")
    
    numbered_text = "\n".join(lines)
    
    # Language name mapping for clearer prompts
    lang_names = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian',
        'zh-cn': 'Chinese (Simplified)', 'ja': 'Japanese', 'ko': 'Korean',
        'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi', 'tr': 'Turkish',
        'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish', 'da': 'Danish',
    }
    lang_name = lang_names.get(target_language, target_language)
    
    # System prompt: always use the professional translator rules
    system_prompt = (
        f"You are a professional translator.\n\n"
        f"Rules:\n"
        f"- Detect the source language automatically.\n"
        f"- Translate the text into {lang_name}.\n"
        f"- Use natural, fluent language (not word-for-word translation).\n"
        f"- Keep the same number of lines as the original text.\n"
        f"- Preserve numbering like \"1. 2. 3.\"\n"
        f"- Return only the translated text."
    )
    
    # User message: custom prompt (if set from UI) + numbered text to translate
    custom = globals().get('TRANSLATION_CUSTOM_PROMPT', '').strip()
    if custom:
        # Replace {language} placeholder with actual target language name
        user_content = custom.replace('{language}', lang_name) + "\n\n" + numbered_text
    else:
        user_content = numbered_text
    
    if log:
        log(f"[OpenAI TRANSLATE] Sending {len(segments)} segments to {globals().get('OPENAI_MODEL', 'gpt-4o-mini')} for {lang_name} translation...")
        if custom:
            log(f"[OpenAI TRANSLATE] Using CUSTOM user prompt: {custom[:120]}{'...' if len(custom) > 120 else ''}")
        else:
            log(f"[OpenAI TRANSLATE] Using default translator prompt")
    
    import time as _time
    import re
    max_retries = 5
    # Backoff schedule: 10s, 20s, 40s, 60s, 120s — total ~250s to outlast per-minute rate-limit windows
    backoff_schedule = [10, 20, 40, 60, 120]
    retryable_status_codes = {429, 500, 502, 503, 504}
    
    for attempt in range(max_retries):
        try:
            response = _requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': globals().get('OPENAI_MODEL', 'gpt-4o-mini'),
                    'temperature': 0.2,
                    'top_p': 1,
                    'frequency_penalty': 0,
                    'presence_penalty': 0,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_content}
                    ]
                },
                timeout=120
            )
            
            if response.status_code in retryable_status_codes:
                # Respect Retry-After header if present, otherwise use backoff schedule
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait = min(int(retry_after), 120)
                    except (ValueError, TypeError):
                        wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                else:
                    wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
                if log:
                    log(f"[OpenAI TRANSLATE] HTTP {response.status_code} — retry {attempt+1}/{max_retries} in {wait}s...")
                    if response.status_code == 429 and attempt == 0:
                        log("[OpenAI TRANSLATE] TIP: If this persists, add billing credit at https://platform.openai.com/settings/organization/billing")
                _time.sleep(wait)
                continue
            
            if response.status_code != 200:
                if log:
                    log(f"[OpenAI TRANSLATE ERROR] API returned {response.status_code}: {response.text[:200]}")
                return None
            
            data = response.json()
            reply = data['choices'][0]['message']['content'].strip()
            
            # Parse numbered lines from response
            result_lines = reply.split('\n')
            translated_texts = [''] * len(segments)
            
            for line in result_lines:
                line = line.strip()
                if not line:
                    continue
                # Match "1. text", "1) text", "1: text", "1- text", or "1 text"
                m = re.match(r'^(\d+)[.\)\:\-]?\s+(.*)', line)
                if m:
                    idx = int(m.group(1)) - 1  # Convert 1-based to 0-based
                    if 0 <= idx < len(segments):
                        translated_texts[idx] = m.group(2).strip()
            
            # Verify we got translations for most segments
            filled = sum(1 for t in translated_texts if t)
            
            # Fallback: if numbered parsing failed, try plain line-by-line mapping
            if filled < len(segments) * 0.5:
                non_empty_lines = [l.strip() for l in result_lines if l.strip()]
                if len(non_empty_lines) == len(segments):
                    if log:
                        log(f"[OpenAI TRANSLATE] Numbered parsing got {filled}/{len(segments)}, using line-by-line mapping instead")
                    for i, line in enumerate(non_empty_lines):
                        # Strip any leading numbering from the line
                        cleaned = re.sub(r'^(\d+)[.\)\:\-]?\s*', '', line).strip()
                        translated_texts[i] = cleaned if cleaned else line.strip()
                    filled = sum(1 for t in translated_texts if t)
            
            if filled < len(segments) * 0.5:
                if log:
                    log(f"[OpenAI TRANSLATE WARNING] Only {filled}/{len(segments)} lines parsed, falling back")
                return None
            
            # Fill any missing translations with originals
            for i in range(len(segments)):
                if not translated_texts[i]:
                    translated_texts[i] = segments[i].get("text", "").strip()
            
            if log:
                log(f"[OpenAI TRANSLATE] Successfully translated {filled}/{len(segments)} segments")
            
            return translated_texts
            
        except (_requests.exceptions.Timeout, _requests.exceptions.ConnectionError) as e:
            # Network errors are retryable
            wait = backoff_schedule[min(attempt, len(backoff_schedule) - 1)]
            if log:
                log(f"[OpenAI TRANSLATE] Network error: {e} — retry {attempt+1}/{max_retries} in {wait}s...")
            _time.sleep(wait)
            continue
        except Exception as e:
            if log:
                log(f"[OpenAI TRANSLATE ERROR] {e}")
            return None
    
    # All retries exhausted
    if log:
        log("[OpenAI TRANSLATE ERROR] All retries failed (rate limited). Add billing credit at https://platform.openai.com/settings/organization/billing")
    return None


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
            log("[TRANSLATE] ⚠ OpenAI translation failed — falling back to googletrans...")
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
                log(f"[TRANSLATE] ⚠️ {fail_count} consecutive failures — googletrans appears broken, keeping remaining segments untranslated")
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
    'de asemenea', etc.) based on target language detection.
    
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
        def _normalize(t):
            return re.sub(r'[^\w\s]', '', t.lower()).strip()
        
        prev_norm = _normalize(prev_text)
        curr_norm = _normalize(curr_text)
        
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
        # Try to identify what changed (e.g., "el" -> "ea", "he" -> "she")
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
    ro_markers = {'și', 'este', 'avea', 'ani', 'de', 'la', 'că', 'pentru'}
    is_romanian = len(words_set & ro_markers) >= 2
    
    # Spanish detection
    es_markers = {'ella', 'también', 'tenía', 'años', 'pero', 'como'}
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


# ----------------- AI VOICE REPLACEMENT FUNCTIONS -----------------

def _openai_tts_generate(text, language='en', output_path=None, log=None):
    """
    Generate Text-to-Speech audio using OpenAI's TTS API (/v1/audio/speech).
    Uses the same OPENAI_API_KEY that is used for translations.
    
    Args:
        text: Text to convert to speech
        language: Language code (used to select voice)
        output_path: Path to save audio file (temp file if None)
        log: Optional logging function
    
    Returns:
        Path to generated audio file or None if failed
    """
    api_key = globals().get('OPENAI_API_KEY')
    if not api_key:
        return None
    
    if not REQUESTS_AVAILABLE:
        if log:
            log("[OpenAI TTS] requests library not available")
        return None
    
    if not text or not text.strip():
        return None
    
    try:
        import requests as _requests
        
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.mp3', prefix='openai_tts_')
            os.close(fd)
        
        # Select voice based on language — all OpenAI TTS voices are multilingual,
        # but some voice tones suit certain language groups better (cosmetic preference)
        voice_map = {
            'en': 'alloy',   'es': 'nova',    'fr': 'shimmer',
            'de': 'onyx',    'it': 'nova',    'pt': 'nova',
            'ro': 'alloy',   'ru': 'onyx',    'zh': 'nova',
            'zh-cn': 'nova', 'ja': 'shimmer', 'ko': 'shimmer',
            'ar': 'onyx',    'hi': 'alloy',   'tr': 'onyx',
            'pl': 'alloy',   'nl': 'alloy',   'sv': 'shimmer',
            'da': 'shimmer',
        }
        voice = voice_map.get(language, 'alloy')
        
        if log:
            log(f"[OpenAI TTS] Generating speech with voice '{voice}' ({len(text)} chars)...")
        
        response = _requests.post(
            'https://api.openai.com/v1/audio/speech',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'tts-1',
                'input': text,
                'voice': voice,
                'response_format': 'mp3'
            },
            timeout=120
        )
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            if log:
                log(f"[OpenAI TTS] ✓ Generated audio: {output_path} ({len(response.content)} bytes)")
            return output_path
        elif response.status_code == 429:
            if log:
                log("[OpenAI TTS] ⚠ Rate limited (429) — will fall back to other TTS engines")
                log("[OpenAI TTS] TIP: Add billing credit at https://platform.openai.com/settings/organization/billing")
            return None
        else:
            if log:
                err_text = response.text[:300]
                log(f"[OpenAI TTS ERROR] API returned {response.status_code}: {err_text}")
            return None
    except Exception as e:
        if log:
            log(f"[OpenAI TTS ERROR] {e}")
        return None


# GenAI Pro API constraints
GENAIPRO_API_BASE = 'https://genaipro.io/api'  # API base URL — endpoints live under /api/v1/labs/task
GENAIPRO_MIN_SPEED = 0.7     # Minimum TTS speed accepted by the API
GENAIPRO_MAX_SPEED = 1.2     # Maximum TTS speed accepted by the API
GENAIPRO_MAX_WAIT_SECONDS = 600  # Max time to wait for a TTS task (10 minutes)

# Number of voice jobs to submit to GenAI Pro simultaneously.
# Higher = faster overall queue, but too many concurrent tasks may get rate-limited.
# GenAI Pro doesn't publish a hard limit; 3 is a safe default.
VOICE_SUBMISSION_CONCURRENCY = 3


def _genaipro_session(requests_module):
    """Return a requests.Session that preserves the Authorization header across redirects.
    Python's default behaviour strips it when following cross-domain redirects, which
    causes genaipro.io to report token_is_empty even when a valid JWT is supplied."""
    class _KeepAuthSession(requests_module.Session):
        def rebuild_auth(self, prepared_request, response):
            # Do NOT strip the Authorization header on redirect
            pass
    return _KeepAuthSession()


def _find_task_in_genaipro_response(data, task_id):
    """
    Locate a specific task in any response shape returned by genaipro.io.
    Handles multiple formats:
      - Per-task dict:          {"id": "...", "status": "...", "result": "..."}
      - Wrapped single dict:    {"data": {"id": "...", "status": "..."}, "success": true}
      - Wrapped list:           {"tasks": [...], "total": N, ...}
      - Bare list:              [{"id": "...", ...}, ...]
    Returns the task dict or None.
    """
    if isinstance(data, dict):
        # Case 1: looks like a task object itself (has known id field at top level)
        tid = data.get('id') or data.get('task_id')
        if tid is not None and (task_id is None or str(tid) == str(task_id)):
            return data
        # Case 2: unwrap envelope keys — value may be a list OR a single task dict
        for key in ('tasks', 'data', 'items', 'results', 'task'):
            inner = data.get(key)
            if isinstance(inner, list):
                for task in inner:
                    tid = task.get('id') or task.get('task_id')
                    if tid is not None and str(tid) == str(task_id):
                        return task
            elif isinstance(inner, dict):
                # Single task wrapped in an envelope key
                tid = inner.get('id') or inner.get('task_id')
                if tid is not None and (task_id is None or str(tid) == str(task_id)):
                    return inner
                # Even if id doesn't match, return if it has a status field (per-task endpoint)
                if 'status' in inner and task_id is None:
                    return inner
        # Case 3: the response has 'status' but no matching id field
        # (per-task endpoint may omit the id in the body)
        if 'status' in data:
            return data
    elif isinstance(data, list):
        for task in data:
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


def _refresh_genaipro_result_url(session, headers, task_id, log=None, log_prefix='[GenAI Pro]'):
    """Re-fetch an existing task and return the newest result/audio URL, if available."""
    if not task_id:
        return None
    if log:
        log(f"{log_prefix} Refreshing task result URL from existing task_id before nonce re-submit...")
    urls = (
        f'{GENAIPRO_API_BASE}/v1/labs/task/{task_id}',
        f'{GENAIPRO_API_BASE}/v1/labs/task?limit=100',
    )
    for url in urls:
        try:
            resp = session.get(url, headers=headers, timeout=60)
            if resp.status_code not in (200, 201, 202):
                continue
            data = resp.json()
            task = _find_task_in_genaipro_response(data, task_id)
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


def _build_genaipro_nonce_input(base_input, nonce_attempt=1):
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


_GENAIPRO_FALLBACK_MODELS = [
    'eleven_turbo_v2_5',
    'eleven_multilingual_v2',
    'eleven_turbo_v2',
    'eleven_flash_v2_5',
    'eleven_flash_v2',
    'eleven_turbo_v2_5',
]


def _build_genaipro_retry_payload(base_payload, retry_attempt=1):
    """Build a fresh retry payload that avoids server-side dedupe collisions."""
    payload = dict(base_payload or {})
    try:
        attempt = max(1, int(retry_attempt))
    except Exception:
        attempt = 1

    # Always change text with a stronger invisible nonce.
    payload['input'] = _build_genaipro_nonce_input(payload.get('input'), attempt)

    # Cycle through model IDs — the API deduplicates on (text, voice_id, model_id)
    # so switching models reliably forces a new task even when invisible-char nonces
    # are normalized away by the server.
    payload['model_id'] = _GENAIPRO_FALLBACK_MODELS[(attempt - 1) % len(_GENAIPRO_FALLBACK_MODELS)]

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

    payload['speed'] = round(_clamp(base_speed + (sign * step), GENAIPRO_MIN_SPEED, GENAIPRO_MAX_SPEED), 3)
    payload['stability'] = round(_clamp(base_stability + (sign * step), 0.0, 1.0), 3)
    payload['similarity'] = round(_clamp(base_similarity - (sign * step), 0.0, 1.0), 3)
    payload['style'] = round(_clamp(base_style + (0.5 * step), 0.0, 1.0), 3)
    return payload


def _normalize_genaipro_api_key(api_key):
    """Normalize a GenAI Pro JWT pasted by the user."""
    if api_key is None:
        return None
    key = str(api_key).strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key or None


def generate_tts_with_genaipro(text, language='en', output_path=None, api_key=None, log=None):
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
    
    api_key = _normalize_genaipro_api_key(api_key)
    if not api_key:
        if log:
            log("[GenAI Pro] No API key provided")
        return None
    
    if not text or not text.strip():
        return None
    
    try:
        import requests
        import time

        session = _genaipro_session(requests)
        
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.mp3', prefix='genaipro_tts_')
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
        
        # Step 1: Submit TTS task
        if log:
            token_preview = api_key[:8] + "..." if len(api_key) > 8 else "(short)"
            log(f"[GenAI Pro] Submitting TTS task ({len(text)} chars)... [token: {token_preview}]")
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # GenAI Pro API requires speed between 0.7 and 1.2
        tts_speed = max(GENAIPRO_MIN_SPEED, min(GENAIPRO_MAX_SPEED, globals().get('TTS_SPEED', 1.0)))

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
        
        response = session.post(
            f'{GENAIPRO_API_BASE}/v1/labs/task',
            headers=headers,
            json=task_payload,
            timeout=30
        )
        
        if response.status_code not in (200, 201, 202):
            if log:
                err_body = response.text[:300]
                log(f"[GenAI Pro ERROR] Task submission failed: {response.status_code} - {err_body}")
                if response.status_code == 401:
                    log("[GenAI Pro ERROR] ❌ Authentication failed — your GenAI Pro JWT token has expired or is invalid.")
                    log("[GenAI Pro ERROR] ➡️  Please log in again at https://genaipro.io, copy your new JWT token,")
                    log("[GenAI Pro ERROR]     and paste it in the 'TTS API Key' field, then click 'Save API Key'.")
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
                log(f"[GenAI Pro] 🎯 Synchronous response — audio URL returned immediately, downloading...")
            try:
                audio_response = requests.get(direct_audio_url, timeout=120)
                if audio_response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(audio_response.content)
                    if log:
                        log(f"[GenAI Pro] ✅ TTS generated successfully (sync): {output_path}")
                    return output_path
                elif log:
                    log(f"[GenAI Pro ERROR] Direct audio download failed: HTTP {audio_response.status_code}")
            except Exception as dl_err:
                if log:
                    log(f"[GenAI Pro ERROR] Direct audio download error: {dl_err}")
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
                    log(f"[GenAI Pro] 🎯 Task already completed in submission response — downloading immediately...")
                try:
                    audio_response = requests.get(_sub_result, timeout=120)
                    if audio_response.status_code == 200:
                        with open(output_path, 'wb') as f:
                            f.write(audio_response.content)
                        if log:
                            log(f"[GenAI Pro] ✅ TTS generated successfully (cached): {output_path}")
                        return output_path
                    elif log:
                        log(f"[GenAI Pro ERROR] Cached audio download failed: HTTP {audio_response.status_code} — result URL likely expired, re-submitting with nonce")
                except Exception as dl_err:
                    if log:
                        log(f"[GenAI Pro ERROR] Cached audio download error: {dl_err} — re-submitting with nonce")
                refreshed_url = _refresh_genaipro_result_url(
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
                                log(f"[GenAI Pro] ✅ TTS generated successfully (refreshed task URL): {output_path}")
                            return output_path
                        elif log:
                            log(f"[GenAI Pro] Refreshed task URL download failed: HTTP {audio_response.status_code} — trying nonce re-submit")
                    except Exception as refresh_dl_err:
                        if log:
                            log(f"[GenAI Pro] Refreshed task URL download error: {refresh_dl_err} — trying nonce re-submit")
                # The cached result URL is expired/broken — force a fresh task via nonce and let polling handle it
                fresh_task_found = False
                while nonce_retry_count < max_nonce_retries:
                    try:
                        nonce_retry_count += 1
                        nonce_payload = _build_genaipro_retry_payload(task_payload, nonce_retry_count)
                        nonce_resp = session.post(
                            f'{GENAIPRO_API_BASE}/v1/labs/task',
                            headers=headers,
                            json=nonce_payload,
                            timeout=30
                        )
                        if nonce_resp.status_code not in (200, 201, 202):
                            if log:
                                log(f"[GenAI Pro ERROR] Nonce re-submission failed: HTTP {nonce_resp.status_code}")
                            continue
                        nonce_data = nonce_resp.json()
                        nonce_id, _ = _extract_submission_task_id(
                            nonce_data,
                            expected_input=nonce_payload.get('input'),
                            expected_voice_id=voice_id,
                            exclude_task_id=task_id
                        )
                        if not nonce_id:
                            if log:
                                log(f"[GenAI Pro ERROR] Nonce re-submission had no task_id: {nonce_data}")
                            continue
                        if str(nonce_id) == str(task_id):
                            if log:
                                log(f"[GenAI Pro] Nonce re-submission returned same task ID ({nonce_id}); retrying with stronger nonce...")
                            continue
                        task_id = nonce_id
                        task_payload = nonce_payload
                        fresh_task_found = True
                        if log:
                            log(f"[GenAI Pro] ✓ Fresh task submitted (expired-cache bypass): {nonce_id}. Polling...")
                        break
                    except Exception as nonce_err:
                        if log:
                            log(f"[GenAI Pro ERROR] Nonce re-submission error: {nonce_err}")
                if not fresh_task_found:
                    if log:
                        log(f"[GenAI Pro ERROR] Could not obtain a fresh task after {nonce_retry_count} nonce retries")
                    return None
        
        # Step 2: Poll for task completion
        poll_interval = 2  # Poll every 2 seconds
        max_wait_seconds = GENAIPRO_MAX_WAIT_SECONDS

        if log:
            log(f"[GenAI Pro] Waiting for audio generation... (max {max_wait_seconds // 60}m)")
        consecutive_non200 = 0  # Track repeated failures on per-task endpoint
        i = 0
        while True:
            time.sleep(poll_interval)
            
            elapsed_seconds = (i + 1) * poll_interval
            if elapsed_seconds > max_wait_seconds:
                if log:
                    log(f"[GenAI Pro ERROR] ⏰ Timed out after {max_wait_seconds // 60} minutes — aborting")
                return None

            # Show progress every 10 seconds or at start
            if i == 0 or elapsed_seconds % 10 == 0:
                elapsed_mins = elapsed_seconds // 60
                elapsed_secs = elapsed_seconds % 60
                if log:
                    log(f"[GenAI Pro] ⏳ Waiting: {elapsed_mins}m {elapsed_secs}s elapsed - Still processing...")
            if elapsed_seconds > 0 and elapsed_seconds % 1800 == 0 and log:
                log(f"[GenAI Pro] ℹ️ Still waiting after {elapsed_seconds // 60} minutes.")
            
            # Try per-task endpoint first; fall back to list endpoint after repeated failures
            use_list_fallback = consecutive_non200 >= 3
            if use_list_fallback:
                poll_url = f'{GENAIPRO_API_BASE}/v1/labs/task?limit=100'
            else:
                poll_url = f'{GENAIPRO_API_BASE}/v1/labs/task/{task_id}'

            try:
                status_response = session.get(
                    poll_url,
                    headers=headers,
                    timeout=60
                )
            except Exception as poll_err:
                if log:
                    log(f"[GenAI Pro] ⚠️ Network error during status check: {poll_err} — retrying...")
                i += 1
                continue
            
            if status_response.status_code not in (200, 201, 202):
                consecutive_non200 += 1
                if log:
                    body_preview = status_response.text[:200] if status_response.text else '(empty)'
                    log(f"[GenAI Pro] ⚠️ Status check HTTP {status_response.status_code}: {body_preview}")
                    if consecutive_non200 == 3:
                        log("[GenAI Pro] ⚠️ Per-task endpoint unavailable — switching to list endpoint fallback")
                i += 1
                continue
            
            consecutive_non200 = 0  # Reset on success

            try:
                raw_data = status_response.json()
            except Exception:
                if log:
                    log("[GenAI Pro] ⚠️ Invalid JSON in status response — retrying...")
                i += 1
                continue
            
            our_task = _find_task_in_genaipro_response(raw_data, task_id)
            if our_task is None:
                if i == 0 and log:
                    log(f"[GenAI Pro] Task {task_id[:12] if len(task_id) > 12 else task_id} not found yet — queued... (raw: {str(raw_data)[:300]})")
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
                    log(f"[GenAI Pro] 📥 Downloading audio file...")
                
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
                                log(f"[GenAI Pro] ⚠️ Download attempt {dl_attempt+1}/5 failed: HTTP {audio_response.status_code}")
                    except Exception as dl_err:
                        if log:
                            log(f"[GenAI Pro] ⚠️ Download attempt {dl_attempt+1}/5 error: {dl_err}")
                    time.sleep(3)
                
                if download_ok:
                    if log:
                        log(f"[GenAI Pro] ✅ TTS generated successfully: {output_path}")
                    return output_path
                else:
                    refreshed_url = _refresh_genaipro_result_url(
                        session=session,
                        headers=headers,
                        task_id=task_id,
                        log=log,
                        log_prefix='[GenAI Pro]'
                    )
                    if refreshed_url:
                        if log:
                            log("[GenAI Pro] 📥 Retrying download with refreshed task URL...")
                        for refresh_attempt in range(3):
                            try:
                                refresh_resp = requests.get(refreshed_url, timeout=600)
                                if refresh_resp.status_code == 200:
                                    with open(output_path, 'wb') as f:
                                        f.write(refresh_resp.content)
                                    if log:
                                        log(f"[GenAI Pro] ✅ TTS generated successfully (refreshed task URL): {output_path}")
                                    return output_path
                                elif log:
                                    log(f"[GenAI Pro] ⚠️ Refreshed URL attempt {refresh_attempt+1}/3 failed: HTTP {refresh_resp.status_code}")
                            except Exception as refresh_dl_err:
                                if log:
                                    log(f"[GenAI Pro] ⚠️ Refreshed URL attempt {refresh_attempt+1}/3 error: {refresh_dl_err}")
                            if refresh_attempt < 2:
                                time.sleep(2)
                    # Result URL has expired (CDN cleanup after a few days).
                    # The API deduplicates by text+voice, so re-submitting the identical
                    # payload would return the same stale task_id. Add an invisible
                    # zero-width space (U+200B) as a nonce to force a genuinely fresh task.
                    if log:
                        log(f"[GenAI Pro] ⚠️ Result URL expired — re-submitting with nonce to bypass cache...")
                    try:
                        if nonce_retry_count >= max_nonce_retries:
                            if log:
                                log(f"[GenAI Pro ERROR] Reached max nonce retries ({max_nonce_retries}) without a downloadable file")
                            return None
                        nonce_retry_count += 1
                        nonce_payload = _build_genaipro_retry_payload(task_payload, nonce_retry_count)
                        retry_resp = session.post(
                            f'{GENAIPRO_API_BASE}/v1/labs/task',
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
                                # Update task_payload so any further re-submits keep the nonce
                                task_payload = nonce_payload
                                i = 0
                                consecutive_non200 = 0
                                if log:
                                    log(f"[GenAI Pro] ✓ Fresh task submitted: {new_id}. Waiting for completion...")
                                continue  # resume while-True with fresh task_id
                            if log:
                                log(f"[GenAI Pro ERROR] Re-submission response had no task_id: {retry_data}")
                        else:
                            if log:
                                log(f"[GenAI Pro ERROR] Re-submission failed: HTTP {retry_resp.status_code} — {retry_resp.text[:200]}")
                    except Exception as retry_err:
                        if log:
                            log(f"[GenAI Pro ERROR] Re-submission error: {retry_err}")
                    return None
            
            elif status in ['failed', 'error', 'cancelled', 'canceled']:
                if log:
                    log(f"[GenAI Pro ERROR] ❌ Task failed with status '{status}' (task JSON: {str(our_task)[:300]})")
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


def _submit_genaipro_task(text, language='en', api_key=None, log=None):
    """
    Submit a TTS task to GenAI Pro API WITHOUT waiting for completion.
    Returns (task_id, headers) on success, or (None, None) on failure.
    """
    if not REQUESTS_AVAILABLE:
        if log:
            log("[GenAI Pro Submit] requests library not available")
        return None, None
    api_key = _normalize_genaipro_api_key(api_key)
    if not api_key:
        if log:
            log("[GenAI Pro Submit] No API key provided")
        return None, None
    if not text or not text.strip():
        return None, None

    try:
        import requests

        session = _genaipro_session(requests)

        global TTS_VOICE_ID
        if TTS_VOICE_ID and TTS_VOICE_ID != 'auto':
            voice_id = TTS_VOICE_ID
        else:
            voice_map = {
                'en': 'uju3wxzG5OhpWcoi3SMy',
                'es': 'uju3wxzG5OhpWcoi3SMy',
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

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        task_payload = {
            'input': text,
            'voice_id': voice_id,
            'model_id': 'eleven_turbo_v2_5',
            'speed': max(GENAIPRO_MIN_SPEED, min(GENAIPRO_MAX_SPEED, globals().get('TTS_SPEED', 1.0))),
            'style': 0.0,
            'use_speaker_boost': False,
            'similarity': 0.75,
            'stability': 0.5
        }

        if log:
            token_preview = api_key[:8] + "..." if len(api_key) > 8 else "(short)"
            log(f"[GenAI Pro Submit] Submitting TTS task ({len(text)} chars)... [token: {token_preview}]")

        response = session.post(
            f'{GENAIPRO_API_BASE}/v1/labs/task',
            headers=headers,
            json=task_payload,
            timeout=30
        )

        if response.status_code not in (200, 201, 202):
            if log:
                err_body = response.text[:300]
                log(f"[GenAI Pro Submit ERROR] {response.status_code} - {err_body}")
                if response.status_code == 401:
                    log("[GenAI Pro Submit ERROR] ❌ Authentication failed — your GenAI Pro JWT token has expired or is invalid.")
                    log("[GenAI Pro Submit ERROR] ➡️  Please log in again at https://genaipro.io, copy your new JWT token,")
                    log("[GenAI Pro Submit ERROR]     and paste it in the 'TTS API Key' field, then click 'Save API Key'.")
            return None, None

        try:
            task_data = response.json()
        except Exception:
            if log:
                log(f"[GenAI Pro Submit ERROR] Non-JSON response (status {response.status_code}): {response.content[:200]}")
            return None, None

        if log:
            log(f"[GenAI Pro Submit] Response ({response.status_code}): {str(task_data)[:400]}")

        task_id, _ = _extract_submission_task_id(
            task_data,
            expected_input=text,
            expected_voice_id=voice_id
        )

        if not task_id:
            if log:
                log(f"[GenAI Pro Submit ERROR] No task_id in response: {str(task_data)[:400]}")
            return None, None

        if log:
            log(f"[GenAI Pro Submit] ✓ Task submitted: {task_id}")

        return task_id, {
            'headers': headers,
            'task_payload': task_payload,
            'voice_id': voice_id,
            'api_key': api_key,
        }

    except Exception as e:
        if log:
            import traceback as _tb
            log(f"[GenAI Pro Submit ERROR] {e}")
            log(_tb.format_exc())
        return None, None


def _poll_and_download_genaipro(task_id, headers, output_path=None, log=None):
    """
    Poll GenAI Pro for a specific task's completion and download the audio.
    Returns path to audio file or None.
    """
    if not REQUESTS_AVAILABLE:
        return None

    try:
        import requests
        import time

        session = _genaipro_session(requests)

        poll_context = headers if isinstance(headers, dict) and 'headers' in headers else {}
        request_headers = poll_context.get('headers', headers)
        task_payload = dict(poll_context.get('task_payload') or {})
        voice_id = poll_context.get('voice_id')
        nonce_retry_count = 0
        max_nonce_retries = 6

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.mp3', prefix='genaipro_poll_')
            os.close(fd)

        poll_interval = 2
        max_wait_seconds = GENAIPRO_MAX_WAIT_SECONDS
        tag = task_id[:8] if len(task_id) > 8 else task_id
        if log:
            log(f"[GenAI Pro Poll] Waiting for task {tag}... (max {max_wait_seconds // 60}m)")

        i = 0
        consecutive_non200 = 0
        while True:
            time.sleep(poll_interval)

            elapsed_seconds = (i + 1) * poll_interval
            if elapsed_seconds > max_wait_seconds:
                if log:
                    log(f"[GenAI Pro Poll ERROR] ⏰ Timed out after {max_wait_seconds // 60} minutes — aborting")
                return None

            if i == 0 or elapsed_seconds % 10 == 0:
                elapsed_mins = elapsed_seconds // 60
                elapsed_secs = elapsed_seconds % 60
                if log:
                    log(f"[GenAI Pro Poll] ⏳ {tag}... {elapsed_mins}m {elapsed_secs}s elapsed")

            if elapsed_seconds > 0 and elapsed_seconds % 1800 == 0 and log:
                log(f"[GenAI Pro Poll] ℹ️ Still waiting after {elapsed_seconds // 60} min")

            # Try per-task endpoint first; fall back to list endpoint after repeated failures
            use_list_fallback = consecutive_non200 >= 3
            if use_list_fallback:
                poll_url = f'{GENAIPRO_API_BASE}/v1/labs/task?limit=100'
            else:
                poll_url = f'{GENAIPRO_API_BASE}/v1/labs/task/{task_id}'

            try:
                status_response = session.get(
                    poll_url,
                    headers=request_headers,
                    timeout=30
                )
            except Exception as poll_net_err:
                if log:
                    log(f"[GenAI Pro Poll] ⚠️ Network error during status check: {poll_net_err} — retrying...")
                i += 1
                continue

            if status_response.status_code not in (200, 201, 202):
                consecutive_non200 += 1
                if log:
                    body_preview = status_response.text[:200] if status_response.text else '(empty)'
                    log(f"[GenAI Pro Poll] ⚠️ Status check HTTP {status_response.status_code}: {body_preview}")
                    if consecutive_non200 == 3:
                        log("[GenAI Pro Poll] ⚠️ Per-task endpoint unavailable — switching to list endpoint fallback")
                i += 1
                continue

            consecutive_non200 = 0

            try:
                raw_data = status_response.json()
            except Exception:
                if log:
                    log("[GenAI Pro Poll] ⚠️ Invalid JSON in status response — retrying...")
                i += 1
                continue

            our_task = _find_task_in_genaipro_response(raw_data, task_id)
            if our_task is None:
                if i == 0 and log:
                    log(f"[GenAI Pro Poll] Task not found yet — may still be queued... (raw: {str(raw_data)[:300]})")
                elif i % 30 == 0 and log and i > 0:
                    log(f"[GenAI Pro Poll] Still waiting for task {tag}... (raw: {str(raw_data)[:200]})")
                i += 1
                continue

            status = our_task.get('status', '').lower()
            result = our_task.get('result', '')

            is_complete = (status in ['completed', 'done', 'success', 'succeeded', 'finished'] or
                           (result and result != ''))

            if is_complete:
                if log:
                    elapsed_seconds = (i + 1) * poll_interval
                    log(f"[GenAI Pro Poll] ✓ Task complete! ({elapsed_seconds}s)")

                audio_url = (our_task.get('result') or our_task.get('output_url') or
                            our_task.get('audio_url') or our_task.get('result_url') or
                            our_task.get('file_url') or our_task.get('url'))

                if not audio_url:
                    if log:
                        log(f"[GenAI Pro Poll ERROR] No audio URL in completed task (status={status})")
                    return None

                if log:
                    log(f"[GenAI Pro Poll] 📥 Downloading audio...")

                download_ok = False
                for dl_attempt in range(5):
                    try:
                        audio_response = requests.get(audio_url, timeout=60)
                        if audio_response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                f.write(audio_response.content)
                            download_ok = True
                            break
                        else:
                            if log:
                                log(f"[GenAI Pro Poll] ⚠️ Download attempt {dl_attempt+1}/5 HTTP {audio_response.status_code}")
                    except Exception as dl_err:
                        if log:
                            log(f"[GenAI Pro Poll] ⚠️ Download attempt {dl_attempt+1}/5 error: {dl_err}")
                    if dl_attempt < 4:
                        time.sleep(3)

                if download_ok:
                    if log:
                        log(f"[GenAI Pro Poll] ✅ Audio downloaded: {output_path}")
                    return output_path
                else:
                    refreshed_url = _refresh_genaipro_result_url(
                        session=session,
                        headers=request_headers,
                        task_id=task_id,
                        log=log,
                        log_prefix='[GenAI Pro Poll]'
                    )
                    if refreshed_url:
                        if log:
                            log("[GenAI Pro Poll] 📥 Retrying download with refreshed task URL...")
                        for refresh_attempt in range(3):
                            try:
                                refresh_resp = requests.get(refreshed_url, timeout=60)
                                if refresh_resp.status_code == 200:
                                    with open(output_path, 'wb') as f:
                                        f.write(refresh_resp.content)
                                    if log:
                                        log(f"[GenAI Pro Poll] ✅ Audio downloaded (refreshed task URL): {output_path}")
                                    return output_path
                                elif log:
                                    log(f"[GenAI Pro Poll] ⚠️ Refreshed URL attempt {refresh_attempt+1}/3 HTTP {refresh_resp.status_code}")
                            except Exception as refresh_dl_err:
                                if log:
                                    log(f"[GenAI Pro Poll] ⚠️ Refreshed URL attempt {refresh_attempt+1}/3 error: {refresh_dl_err}")
                            if refresh_attempt < 2:
                                time.sleep(2)
                    if not task_payload:
                        if log:
                            log(f"[GenAI Pro Poll ERROR] ❌ Download failed after 5 attempts (URL may have expired)")
                        return None

                    if nonce_retry_count >= max_nonce_retries:
                        if log:
                            log(f"[GenAI Pro Poll ERROR] Reached max nonce retries ({max_nonce_retries}) without a downloadable file")
                        return None

                    if log:
                        log(f"[GenAI Pro Poll] ⚠️ Result URL expired — re-submitting with nonce to bypass cache...")

                    try:
                        nonce_retry_count += 1
                        nonce_payload = _build_genaipro_retry_payload(task_payload, nonce_retry_count)
                        retry_resp = session.post(
                            f'{GENAIPRO_API_BASE}/v1/labs/task',
                            headers=request_headers,
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
                                        log(f"[GenAI Pro Poll] Re-submission returned same task ID ({new_id}); retrying with stronger nonce...")
                                    continue
                                task_id = new_id
                                task_payload = nonce_payload
                                i = 0
                                consecutive_non200 = 0
                                tag = task_id[:8] if len(task_id) > 8 else task_id
                                if log:
                                    log(f"[GenAI Pro Poll] ✓ Fresh task submitted: {new_id}. Waiting for completion...")
                                continue
                            if log:
                                log(f"[GenAI Pro Poll ERROR] Re-submission response had no task_id: {retry_data}")
                        else:
                            if log:
                                log(f"[GenAI Pro Poll ERROR] Re-submission failed: HTTP {retry_resp.status_code} — {retry_resp.text[:200]}")
                    except Exception as retry_err:
                        if log:
                            log(f"[GenAI Pro Poll ERROR] Re-submission error: {retry_err}")
                    return None

            elif status in ['failed', 'error', 'cancelled', 'canceled']:
                if log:
                    log(f"[GenAI Pro Poll ERROR] ❌ Task failed: {status!r} (task JSON: {str(our_task)[:300]})")
                return None
            elif i % 10 == 0 and log and i > 0:
                log(f"[GenAI Pro Poll] Status: {status!r} (still processing...)")

            i += 1

    except Exception as e:
        if log:
            import traceback as _tb
            log(f"[GenAI Pro Poll ERROR] {e}")
            log(_tb.format_exc())
        return None


def remove_silence_from_audio(audio_path, output_path=None, log=None, min_silence_ms=300):
    """
    Remove long silences from audio file while keeping natural speech pauses.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to save processed audio (temp file if None)
        log: Optional logging function
        min_silence_ms: Minimum silence length in milliseconds to remove (default: 300ms)
    
    Returns:
        Tuple of (output_path, silence_map) where silence_map is a list of
        (original_start, original_end, new_start, new_end) for each kept segment
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
    except ImportError:
        if log:
            log("[SILENCE] pydub not available - skipping silence removal")
        return audio_path, []
    
    try:
        if log:
            log(f"[SILENCE] Loading audio: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        if log:
            log(f"[SILENCE] Original duration: {len(audio)/1000.0:.2f}s")
        
        # Detect non-silent chunks (balanced - remove gaps based on user setting)
        # min_silence_len: minimum length of silence to consider (default 300ms)
        # silence_thresh: volume threshold for silence (audio.dBFS - 16 = fairly sensitive)
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_ms,  # User-configurable threshold
            silence_thresh=audio.dBFS - 16  # Detect even quiet parts as non-silent
        )
        
        if not nonsilent_ranges:
            if log:
                log("[SILENCE] No non-silent segments found!")
            return audio_path, []
        
        if log:
            log(f"[SILENCE] Found {len(nonsilent_ranges)} non-silent segments")
        
        # Add padding around each non-silent range to avoid cutting word
        # boundaries.  150 ms is enough to preserve soft consonant tails
        # and the onset of the next word.
        pad_ms = 150
        audio_len = len(audio)
        padded = []
        for s, e in nonsilent_ranges:
            padded.append((max(0, s - pad_ms), min(audio_len, e + pad_ms)))
        
        # Merge overlapping / adjacent ranges produced by the padding
        merged = [padded[0]]
        for s, e in padded[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        
        if log:
            log(f"[SILENCE] After padding & merge: {len(merged)} segments (pad={pad_ms}ms)")
        
        # Concatenate all non-silent segments
        output_audio = AudioSegment.empty()
        silence_map = []
        new_position = 0
        
        for i, (start_ms, end_ms) in enumerate(merged):
            segment = audio[start_ms:end_ms]
            output_audio += segment
            
            # Track mapping: original time -> new time
            segment_duration = end_ms - start_ms
            silence_map.append((
                start_ms / 1000.0,  # original start in seconds
                end_ms / 1000.0,    # original end in seconds
                new_position / 1000.0,  # new start in seconds
                (new_position + segment_duration) / 1000.0  # new end in seconds
            ))
            new_position += segment_duration
        
        if log:
            orig_dur = len(audio) / 1000.0
            new_dur = len(output_audio) / 1000.0
            removed = orig_dur - new_dur
            log(f"[SILENCE] New duration: {new_dur:.2f}s (removed {removed:.2f}s of silence)")
        
        # Save processed audio
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.mp3', prefix='audio_no_silence_')
            os.close(fd)
        
        output_audio.export(output_path, format="mp3")
        
        if log:
            log(f"[SILENCE] ✅ Saved silence-removed audio: {output_path}")
        
        return output_path, silence_map
        
    except Exception as e:
        if log:
            log(f"[SILENCE ERROR] Failed to remove silence: {e}")
        return audio_path, []

def map_timestamps_after_silence_removal(segments, silence_map, log=None):
    """
    Re-map caption segment timestamps after silence removal.
    
    Args:
        segments: List of caption segments with 'start' and 'end' times
        silence_map: List of (orig_start, orig_end, new_start, new_end) from remove_silence_from_audio
        log: Optional logging function
    
    Returns:
        List of segments with updated 'start' and 'end' times
    """
    if not silence_map:
        if log:
            log("[TIMESTAMP] No silence map - returning original segments")
        return segments
    
    if log:
        log(f"[TIMESTAMP] Re-mapping {len(segments)} segments to compressed audio timeline")
    
    def map_time(original_time):
        """Map an original timestamp to the new timeline after silence removal."""
        # Find which segment contains this timestamp
        for orig_start, orig_end, new_start, new_end in silence_map:
            if orig_start <= original_time <= orig_end:
                # Time falls within this segment
                # Calculate relative position within the segment
                relative_pos = (original_time - orig_start) / (orig_end - orig_start) if (orig_end - orig_start) > 0 else 0
                # Map to new timeline
                return new_start + relative_pos * (new_end - new_start)
        
        # If time doesn't fall in any segment, find the closest segment
        # (this handles edge cases and slight timestamp mismatches)
        closest_segment = min(silence_map, key=lambda seg: min(abs(seg[0] - original_time), abs(seg[1] - original_time)))
        orig_start, orig_end, new_start, new_end = closest_segment
        
        if original_time < orig_start:
            return new_start  # Before first segment
        else:
            return new_end  # After last segment
    
    # Create new segments with mapped timestamps
    mapped_segments = []
    for seg in segments:
        new_seg = seg.copy()
        orig_start = seg.get('start', 0)
        orig_end = seg.get('end', 0)
        
        new_seg['start'] = map_time(orig_start)
        new_seg['end'] = map_time(orig_end)
        
        # If segment has word-level timestamps, map those too
        if 'words' in seg and seg['words']:
            new_words = []
            for word in seg['words']:
                new_word = word.copy()
                if 'start' in word:
                    new_word['start'] = map_time(word['start'])
                if 'end' in word:
                    new_word['end'] = map_time(word['end'])
                new_words.append(new_word)
            new_seg['words'] = new_words
        
        mapped_segments.append(new_seg)
    
    if log:
        orig_total = segments[-1]['end'] if segments else 0
        new_total = mapped_segments[-1]['end'] if mapped_segments else 0
        log(f"[TIMESTAMP] Timeline compressed from {orig_total:.2f}s to {new_total:.2f}s")
    
    return mapped_segments

def _get_genaipro_api_key():
    """Return GenAI Pro API key from tts_config.json, or None if unavailable."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "tts_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return _normalize_genaipro_api_key(config.get("api_key", ""))
    except Exception:
        pass
    return None

def generate_tts_audio(text, language='en', output_path=None, log=None):
    """
    Generate Text-to-Speech audio using GenAI Pro exclusively.
    
    Args:
        text: Text to convert to speech
        language: Language code for TTS
        output_path: Path to save audio file (temp file if None)
        log: Optional logging function
    
    Returns:
        Path to generated audio file or None if failed
    """
    if not text or not text.strip():
        return None
    
    api_key = _get_genaipro_api_key()
    if not api_key:
        if log:
            config_path = os.path.join(os.path.dirname(__file__), "tts_config.json")
            log(f"[TTS] ❌ No GenAI Pro API key found in: {config_path}")
            log("[TTS] ➡️  Enter your JWT token in the 'TTS API Key' field and click 'Save API Key'.")
        return None
    
    if log:
        log("="*60)
        log("[TTS] 🎙️  STARTING AI VOICE GENERATION")
        log(f"[TTS] Using GenAI Pro API for high-quality synthesis")
        log(f"[TTS] Text length: {len(text)} characters")
        log(f"[TTS] Language: {language}")
        log("="*60)
    result = generate_tts_with_genaipro(text, language, output_path, api_key, log)
    if result:
        if log:
            log("="*60)
            log("[TTS] ✅ AI VOICE GENERATION COMPLETE!")
            log(f"[TTS] Audio file ready: {result}")
            log("="*60)
        return result
    
    # --- Fallback voice retry ---
    fallback_id = globals().get('TTS_FALLBACK_VOICE_ID', 'auto')
    primary_id = globals().get('TTS_VOICE_ID', 'auto')
    if fallback_id and fallback_id != 'auto' and fallback_id != primary_id:
        if log:
            log("")
            log("="*60)
            log("[TTS] ⚠️  PRIMARY VOICE FAILED — TRYING FALLBACK VOICE")
            log(f"[TTS] Fallback voice ID: {fallback_id}")
            log("="*60)
        # Temporarily swap to fallback voice
        old_voice = globals().get('TTS_VOICE_ID')
        globals()['TTS_VOICE_ID'] = fallback_id
        try:
            result = generate_tts_with_genaipro(text, language, output_path, api_key, log)
        finally:
            globals()['TTS_VOICE_ID'] = old_voice  # Restore primary voice
        if result:
            if log:
                log("="*60)
                log("[TTS] ✅ FALLBACK VOICE GENERATION COMPLETE!")
                log(f"[TTS] Audio file ready: {result}")
                log("="*60)
            return result
        if log:
            log("[TTS] ❌ Fallback voice also failed.")
    
    if log:
        log("[TTS] ❌ GenAI Pro TTS failed. No voice generated.")
    return None

def replace_voice_with_tts(caption_segments, language='en', log=None):
    """
    Generate AI voice audio from caption segments.
    
    Args:
        caption_segments: List of caption segments with translated text
        language: Language code for TTS
        log: Optional logging function
    
    Returns:
        Path to generated audio file or None if failed
    """
    # Check if GenAI Pro TTS is available
    if not _get_genaipro_api_key():
        if log:
            log("[TTS] ⚠️  No GenAI Pro API key found in tts_config.json")
        return None
    
    if log:
        log(f"[TTS] Generating AI voice from {len(caption_segments)} segments...")
    
    try:
        # NOTE: This implementation combines all segments into one continuous audio track.
        # For better synchronization, a future enhancement would be to:
        # 1. Generate TTS for each segment individually
        # 2. Concatenate audio clips with proper timing based on segment start/end times
        # 3. Add silence between segments to match original timing
        # Current approach works well when video speed is adjusted to match audio duration.
        
        # Combine all segment texts
        full_text = " ".join([seg.get("text", "") for seg in caption_segments])
        
        # Generate TTS audio
        output_path = generate_tts_audio(full_text, language=language, log=log)
        
        if output_path and log:
            log(f"[TTS] Voice replacement complete: {output_path}")
        
        return output_path
    except Exception as e:
        if log:
            log(f"[TTS ERROR] Failed to replace voice: {e}")
        return None

# ----------------- SETTINGS (defaults) -----------------
WIDTH = 1080
HEIGHT = 1920
IS_4K_MODE = False  # Track resolution mode for proper scaling

CROP_TOP_RATIO = 0.30
CROP_BOTTOM_RATIO = 0.35

# Minimum foreground height as a fraction of canvas height.
# Prevents the foreground from being a tiny strip when crop is heavy on landscape videos.
# If the width-scaled foreground would be shorter than this, it scales up more (zooms in, clips sides).
MIN_FG_HEIGHT_RATIO = 0.35  # Foreground fills at least 35% of canvas height (672px on 1920px canvas)

VOICE_GAIN = 1.5  # Default: 1.5x — slight boost for clear voice over music
MUSIC_GAIN = 0.4  # Default: 0.4x — audible background music, not overwhelming
TTS_SPEED = 1.0   # Default TTS speed multiplier (0.5–2.0, 1.0 = normal)

def _gain_to_db_str(gain):
    """Convert linear gain multiplier to dB string (CapCut-style display)."""
    if gain <= 0:
        return "Mute"
    db = 20.0 * math.log10(gain)
    if db >= 0:
        return f"+{db:.1f} dB"
    return f"{db:.1f} dB"

def _gain_to_display(gain):
    """Convert gain to human-friendly display: percentage + dB."""
    if gain <= 0:
        return "🔇 Mute"
    pct = int(round(gain * 100))
    db = 20.0 * math.log10(gain)
    if db >= 0:
        return f"{pct}%  (+{db:.1f} dB)"
    return f"{pct}%  ({db:.1f} dB)"
CAPTION_FONT_PREFERRED = "Bangers"
CAPTION_FONT_SIZE = 56

# Mapping of font filenames to proper font family names (for when fontTools is unavailable)
# This allows FFmpeg ASS subtitles to use correct font family without fontTools dependency
FONT_FILENAME_TO_FAMILY = {
    # Arial variants
    "arial-black": "Arial Black",
    "arialblack": "Arial Black",
    "arial_black": "Arial Black",
    "arialbl": "Arial Black",
    "ariblk": "Arial Black",
    "arial": "Arial",
    "arialbd": "Arial",
    "ariali": "Arial",
    "arialbi": "Arial",
    # Bangers
    "bangers": "Bangers",
    "bangers-regular": "Bangers",
    # Impact
    "impact": "Impact",
    # Comic Sans
    "comic": "Comic Sans MS",
    "comicsans": "Comic Sans MS",
    "comicsansms": "Comic Sans MS",
    # Times New Roman
    "times": "Times New Roman",
    "timesnewroman": "Times New Roman",
    "timesbd": "Times New Roman",
    "timesi": "Times New Roman",
    # Courier New
    "cour": "Courier New",
    "courier": "Courier New",
    "couriernew": "Courier New",
    # Verdana
    "verdana": "Verdana",
    "verdanab": "Verdana",
    "verdanai": "Verdana",
    # Georgia
    "georgia": "Georgia",
    "georgiab": "Georgia",
    "georgiai": "Georgia",
    # Tahoma
    "tahoma": "Tahoma",
    "tahomabd": "Tahoma",
    # Trebuchet MS
    "trebuc": "Trebuchet MS",
    "trebuchet": "Trebuchet MS",
    "trebuchetms": "Trebuchet MS",
    # Segoe UI
    "segoeui": "Segoe UI",
    "segoeuib": "Segoe UI",
    "segoeuii": "Segoe UI",
    # Popular Google Fonts
    "roboto": "Roboto",
    "roboto-regular": "Roboto",
    "roboto-bold": "Roboto",
    "robotobold": "Roboto",
    "opensans": "Open Sans",
    "open-sans": "Open Sans",
    "opensans-regular": "Open Sans",
    "opensansbold": "Open Sans",
    "lato": "Lato",
    "lato-regular": "Lato",
    "lato-bold": "Lato",
    "montserrat": "Montserrat",
    "montserrat-regular": "Montserrat",
    "montserrat-bold": "Montserrat",
    "oswald": "Oswald",
    "oswald-regular": "Oswald",
    "poppins": "Poppins",
    "poppins-regular": "Poppins",
    "raleway": "Raleway",
    "raleway-regular": "Raleway",
    "ubuntu": "Ubuntu",
    "ubuntu-regular": "Ubuntu",
    "bebasneue": "Bebas Neue",
    "bebas-neue": "Bebas Neue",
    "bebas": "Bebas Neue",
    "anton": "Anton",
    "anton-regular": "Anton",
    "sourcecodepro": "Source Code Pro",
    "source-code-pro": "Source Code Pro",
    "playfairdisplay": "Playfair Display",
    "playfair-display": "Playfair Display",
    "merriweather": "Merriweather",
    "nunito": "Nunito",
    "quicksand": "Quicksand",
    "comfortaa": "Comfortaa",
    "pacifico": "Pacifico",
    "lobster": "Lobster",
    "dancingscript": "Dancing Script",
    "dancing-script": "Dancing Script",
}

# Debug: force saving caption images and force-visible overlay for testing
CAPTION_DEBUG_SAVE_IMAGES = True
CAPTION_DEBUG_FORCE_VISIBLE = True
CAPTION_PADDING = 200

# Default caption colors (RGBA)
CAPTION_TEXT_COLOR = (255, 255, 255, 255)
CAPTION_STROKE_COLOR = (0, 0, 0, 150)
# Stroke width (pixels) for caption border
CAPTION_STROKE_WIDTH = max(1, int(CAPTION_FONT_SIZE * 0.05))

# Video zoom scale (user-controllable)
VIDEO_ZOOM_SCALE = 1.0  # 1.0 = auto-fit, <1.0 = zoom out, >1.0 = zoom in

# Translation and AI Voice settings
TRANSLATION_ENABLED = False
TARGET_LANGUAGE = 'none'  # 'none', 'en', 'es', 'fr', 'ro', etc.
USE_AI_VOICE_REPLACEMENT = False
TTS_LANGUAGE = 'en'
TTS_VOICE_ID = 'auto'  # 'auto' or specific voice ID
TTS_FALLBACK_VOICE_ID = 'auto'  # Backup voice ID used when primary voice fails (policy violation, crash, etc.)
# Mapping from translation language codes to TTS language codes (they differ for some languages)
TRANS_TO_TTS_LANG = {'zh-cn': 'zh', 'zh-tw': 'zh'}
# Premium TTS API keys (for better quality voices)
# Supported: 'elevenlabs', 'openai', 'azure'
TTS_ENGINE = 'gtts'  # Options: 'gtts' (free, basic), 'elevenlabs', 'openai', 'azure'
ELEVENLABS_API_KEY = None
OPENAI_API_KEY = None
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model for translation; user can change via UI dropdown
TRANSLATION_CUSTOM_PROMPT = ""  # User-defined prompt for OpenAI translation; {language} is auto-replaced
STOP_REQUESTED = False
AZURE_SPEECH_KEY = None
AZURE_SPEECH_REGION = None

FPS = 24

MUSIC_FADEOUT_SECONDS = 0.8
MUSIC_LOOP_ALLOWED = True

STATIC_BG_BLUR_RADIUS = 25
BG_SCALE_EXTRA = 1.08
DIM_FACTOR = 1.0  # 1.0 = no dimming (full brightness)
BG_BRIGHTNESS_BOOST = 0.08  # +8% brightness compensation for NV12/format conversion color loss

USE_GPU_IF_AVAILABLE = True
PREFERRED_NVENC_CODEC = "h264_nvenc"
USE_HARDWARE_DECODING = True  # Enable GPU-accelerated decoding
NVENC_PRESET_SPEED = "p1"  # p1=fastest, p7=slowest/best quality. Using p1 for maximum export speed

CAPTION_RAISE = 420
CAPTION_Y_OFFSET = 0  # Vertical offset in pixels (negative = move up, positive = move down)
TEMPLATE_WORDS = {1: 1, 2: 2, 3: 3}
CAPTION_TEMPLATE = 2  # 1, 2 sau 3 cuvinte pe rand
CAPTION_CASE = "none"  # "none", "upper", "lower" – caption text case transform

# Maximum captions for FFmpeg drawtext filters before switching to ASS subtitle file
# (avoids command line length limits and improves performance with many captions)
MAX_DRAWTEXT_CAPTIONS = 100

REQUIRE_FONT_BANGERS = False

CROP_SETTINGS_FILE = "crop_settings.json"

CREATED_OUTPUTS = set()

# ----------------- FONT / DIACRITICS / UTIL -----------------
FONT_CANDIDATES = ["Bangers-Regular.ttf", "Bangers.ttf", "bangers.ttf", "Bangers.otf", "Bangers-Regular.otf"]

# Custom font directory (can be set via environment variable TIKTOK_FONT_DIR)
CUSTOM_FONT_ROOT = os.environ.get("TIKTOK_FONT_DIR", r"C:\tiktok")


# --- Custom dark CanvasSlider widget ---
class CanvasSlider(tk.Canvas):
    def __init__(self, master, from_=0.0, to=1.0, length=400, command=None, initial=0.0, bg='#161b22', trough='#21262d', slider_color='#58a6ff', height=24, **kwargs):
        # accept optional tk.Variable to sync value externally
        self.tk_variable = kwargs.pop('variable', None)
        # remove other widget-specific kwargs
        for _k in ('orient','resolution','sliderlength','from_','to'):
            kwargs.pop(_k, None)
        super().__init__(master, width=length, height=height, bg=bg, highlightthickness=0, **kwargs)
        self.from_ = float(from_)
        self.to = float(to)
        self.command = command
        self.length = int(length)
        self.height = int(height)
        self.trough_color = trough
        self.slider_color = slider_color
        self.bg = bg
        self.knob_radius = max(8, self.height//2 - 2)
        self.knob_pos = 0  # pixel position
        self.value = float(initial)
        # sync variable if provided
        try:
            if self.tk_variable is not None:
                try:
                    self.tk_variable.set(self.value)
                except Exception:
                    pass
        except Exception:
            pass
        # position knob according to initial value
        try:
            self.knob_pos = self._value_to_pos(self.value)
        except Exception:
            self.knob_pos = 0
        self.bind("<Button-1>", self._click)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Configure>", self._on_resize)
        self.redraw()

    def _on_resize(self, event):
        self.length = max(50, event.width)
        try:
            self.knob_pos = self._value_to_pos(self.value)
        except Exception:
            pass
        self.redraw()

    def _value_to_pos(self, v):
        frac = 0.0 if self.to == self.from_ else (v - self.from_) / (self.to - self.from_)
        return int(self.knob_radius + frac * (self.length - 2*self.knob_radius))

    def _pos_to_value(self, x):
        frac = min(1.0, max(0.0, (x - self.knob_radius) / float(max(1, self.length - 2*self.knob_radius))))
        return self.from_ + frac * (self.to - self.from_)

    def set(self, v):
        try:
            v = float(v)
        except Exception:
            return
        v = min(self.to, max(self.from_, v))
        self.value = v
        self.knob_pos = self._value_to_pos(v)
        # sync to tk variable if present
        try:
            if self.tk_variable is not None:
                try:
                    self.tk_variable.set(self.value)
                except Exception:
                    pass
        except Exception:
            pass
        self.redraw()
        # call command callback when value is set programmatically
        if self.command:
            try:
                self.command(self.value)
            except Exception:
                pass

    def get(self):
        return self.value

    def _click(self, event):
        x = event.x
        val = self._pos_to_value(x)
        self.set(val)

    def _drag(self, event):
        x = event.x
        val = self._pos_to_value(x)
        self.set(val)

    def redraw(self):
        self.delete("all")
        # draw trough
        y = self.height // 2
        self.create_rectangle(self.knob_radius, y - 6, self.length - self.knob_radius, y + 6, fill=self.trough_color, outline=self.trough_color)
        # draw progress
        self.create_rectangle(self.knob_radius, y - 6, self.knob_pos, y + 6, fill=self.slider_color, outline=self.slider_color)
        # draw knob (circle)
        self.create_oval(self.knob_pos - self.knob_radius, y - self.knob_radius, self.knob_pos + self.knob_radius, y + self.knob_radius, fill=self.slider_color, outline='')

def _common_font_dirs():
    dirs = [os.getcwd()]
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(os.path.join(windir, "Fonts"))
        # Add custom font directory for user fonts
        dirs.append(CUSTOM_FONT_ROOT)
        dirs.append(os.path.join(CUSTOM_FONT_ROOT, "fonts"))
    else:
        dirs += ["/usr/share/fonts", "/usr/share/fonts/truetype", "/usr/local/share/fonts", "/Library/Fonts", "/System/Library/Fonts"]
    return dirs

def find_font_file_recursive(preferred_name=None):
    names_lower = set()
    if preferred_name:
        names_lower.add(preferred_name.lower())
    for n in FONT_CANDIDATES:
        names_lower.add(n.lower())
    if preferred_name:
        if os.path.isabs(preferred_name) and os.path.isfile(preferred_name):
            return os.path.abspath(preferred_name)
        if os.path.isfile(preferred_name):
            return os.path.abspath(preferred_name)
    for name in FONT_CANDIDATES:
        if os.path.isfile(name):
            return os.path.abspath(name)
    for base in _common_font_dirs():
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            for f in files:
                fl = f.lower()
                tried_path = os.path.join(root, f)
                if fl in names_lower:
                    return os.path.abspath(tried_path)
                if preferred_name and preferred_name.lower() in fl:
                    return os.path.abspath(tried_path)
                if "bangers" in fl:
                    return os.path.abspath(tried_path)
    return None

LOADED_FONT = None
LOADED_FONT_PATH = None
LOADED_FONT_FAMILY = None  # Store the actual font family name for ASS subtitles

def clear_font_cache():
    """Clear the global font cache. Call when user selects a new font."""
    global LOADED_FONT, LOADED_FONT_PATH, LOADED_FONT_FAMILY
    LOADED_FONT = None
    LOADED_FONT_PATH = None
    LOADED_FONT_FAMILY = None

def get_font_family_name(font_path, log_func=None):
    """
    Extract the actual font family name from a font file.
    
    Args:
        font_path: Path to the font file (.ttf, .otf)
        log_func: Optional logging function
        
    Returns:
        Font family name or filename without extension if extraction fails
    """
    if not font_path or not os.path.exists(font_path):
        if log_func:
            log_func(f"[FONT-EXTRACT] Invalid font path: {font_path}")
        if font_path:
            return os.path.splitext(os.path.basename(font_path))[0]
        return None
    
    try:
        from fontTools.ttLib import TTFont
        if log_func:
            log_func(f"[FONT-EXTRACT] Opening font file: {font_path}")
        
        font = TTFont(font_path)
        name_table = font['name']
        
        # Try different name IDs in order of preference:
        # nameID 1 = Font Family Name
        # nameID 16 = Typographic Family Name (preferred for complex font families)
        # nameID 4 = Full Font Name
        family_name = None
        
        for name_id in (16, 1, 4):
            for record in name_table.names:
                if record.nameID == name_id:
                    try:
                        name = record.toUnicode()
                        if name and name.strip():
                            family_name = name.strip()
                            if log_func:
                                log_func(f"[FONT-EXTRACT] Found nameID {name_id}: '{family_name}'")
                            break
                    except Exception as e:
                        if log_func:
                            log_func(f"[FONT-EXTRACT] Error decoding nameID {name_id}: {e}")
            if family_name:
                break
        
        font.close()
        
        if family_name:
            if log_func:
                log_func(f"[FONT-EXTRACT] ✓ Extracted font family: '{family_name}'")
            return family_name
        else:
            if log_func:
                log_func(f"[FONT-EXTRACT] No family name found in name table, using filename")
            
    except ImportError as e:
        if log_func:
            log_func(f"[FONT-EXTRACT] fontTools not available: {e}")
    except Exception as e:
        if log_func:
            log_func(f"[FONT-EXTRACT] Error reading font: {e}")
    
    # Fallback: try to use FONT_FILENAME_TO_FAMILY mapping
    fallback_name = os.path.splitext(os.path.basename(font_path))[0]
    
    # Normalize filename for lookup: lowercase, remove spaces/hyphens variations
    lookup_key = fallback_name.lower().replace(" ", "").replace("_", "-")
    
    if lookup_key in FONT_FILENAME_TO_FAMILY:
        mapped_name = FONT_FILENAME_TO_FAMILY[lookup_key]
        if log_func:
            log_func(f"[FONT-EXTRACT] Found in mapping: '{lookup_key}' → '{mapped_name}'")
        return mapped_name
    
    # Try without hyphens too
    lookup_key_no_hyphen = lookup_key.replace("-", "")
    if lookup_key_no_hyphen in FONT_FILENAME_TO_FAMILY:
        mapped_name = FONT_FILENAME_TO_FAMILY[lookup_key_no_hyphen]
        if log_func:
            log_func(f"[FONT-EXTRACT] Found in mapping (no hyphen): '{lookup_key_no_hyphen}' → '{mapped_name}'")
        return mapped_name
    
    if log_func:
        log_func(f"[FONT-EXTRACT] Using fallback name: '{fallback_name}'")
    return fallback_name

def load_preferred_font_cached(preferred, size, log=None):
    global LOADED_FONT, LOADED_FONT_PATH, LOADED_FONT_FAMILY
    # If a font is already loaded and the preferred request matches the cached path/family, reuse it.
    try:
        if LOADED_FONT is not None:
            if not preferred:
                try:
                    if log:
                        log(f"[FONT] Using cached font: {LOADED_FONT_PATH}")
                except Exception:
                    pass
                return LOADED_FONT
            # normalize comparison: absolute path or family marker
            try:
                pref_norm = os.path.abspath(preferred) if os.path.isabs(preferred) else preferred
            except Exception:
                pref_norm = preferred
            if LOADED_FONT_PATH:
                # exact path match
                try:
                    if os.path.isabs(str(LOADED_FONT_PATH)) and os.path.isabs(str(pref_norm)) and os.path.abspath(str(LOADED_FONT_PATH)) == os.path.abspath(str(pref_norm)):
                        try:
                            if log:
                                log(f"[FONT] Using cached font (path match): {LOADED_FONT_PATH}")
                        except Exception:
                            pass
                        return LOADED_FONT
                except Exception:
                    pass
                # family name match marker: LOADED_FONT_PATH may be 'family:<name>'
                try:
                    if isinstance(LOADED_FONT_PATH, str) and LOADED_FONT_PATH.lower().startswith('family:') and preferred and preferred.lower() in LOADED_FONT_PATH.lower():
                        try:
                            if log:
                                log(f"[FONT] Using cached font (family match): {LOADED_FONT_PATH}")
                        except Exception:
                            pass
                        return LOADED_FONT
                except Exception:
                    pass
            # otherwise fall through and attempt to load the requested preferred font
    except Exception:
        pass

    if preferred:
        if os.path.isfile(preferred):
            try:
                LOADED_FONT = ImageFont.truetype(preferred, size)
                LOADED_FONT_PATH = os.path.abspath(preferred)
                LOADED_FONT_FAMILY = get_font_family_name(preferred)
                if log: log(f"[FONT] ✓ Loaded font from file: {LOADED_FONT_PATH} (family: {LOADED_FONT_FAMILY})")
                return LOADED_FONT
            except Exception:
                pass
        found = find_font_file_recursive(preferred)
        if found:
            try:
                LOADED_FONT = ImageFont.truetype(found, size)
                LOADED_FONT_PATH = found
                LOADED_FONT_FAMILY = get_font_family_name(found)
                if log: log(f"[FONT] ✓ Loaded font (searched): {found} (family: {LOADED_FONT_FAMILY})")
                return LOADED_FONT
            except Exception:
                pass
        try:
            LOADED_FONT = ImageFont.truetype(preferred, size)
            LOADED_FONT_PATH = f"family:{preferred}"
            LOADED_FONT_FAMILY = preferred  # Family name was passed directly
            if log: log(f"[FONT] ✓ Loaded font by family name: {preferred}")
            return LOADED_FONT
        except Exception:
            pass
    found = find_font_file_recursive(None)
    if found:
        try:
            LOADED_FONT = ImageFont.truetype(found, size)
            LOADED_FONT_PATH = found
            LOADED_FONT_FAMILY = get_font_family_name(found)
            if log: log(f"[FONT] ✓ Loaded candidate font: {found} (family: {LOADED_FONT_FAMILY})")
            return LOADED_FONT
        except Exception:
            pass
    try:
        LOADED_FONT = ImageFont.truetype("DejaVuSans.ttf", size)
        LOADED_FONT_PATH = "DejaVuSans.ttf (fallback)"
        LOADED_FONT_FAMILY = "DejaVu Sans"
        if log: log("[FONT] WARNING: Preferred font not found — using DejaVuSans.ttf fallback.")
        return LOADED_FONT
    except Exception:
        pass
    if log: log("[FONT] WARNING: No TrueType fonts available. Using default bitmap font.")
    LOADED_FONT = ImageFont.load_default()
    LOADED_FONT_PATH = "default"
    LOADED_FONT_FAMILY = None
    return LOADED_FONT

def normalize_text(s: str) -> str:
    if not s: return s
    trans = str.maketrans({
        'ă': 'a', 'Ă': 'A','â': 'a', 'Â': 'A','î': 'i', 'Î': 'I',
        'ș': 's', 'Ș': 'S', 'ş': 's', 'Ş': 'S','ț': 't', 'Ț': 'T','ţ': 't', 'Ţ': 'T'
    })
    try:
        return s.translate(trans)
    except Exception:
        repl = s
        for k, v in trans.items():
            repl = repl.replace(k, v)
        return repl

# ----------------- logging helper -----------------
class QueueWriter:
    def __init__(self, q: queue.Queue):
        self.q = q
    def write(self, s):
        if s and not s.isspace():
            self.q.put(str(s))
    def flush(self):
        pass

# ----------------- ffmpeg / pre-render / export helpers -----------------
_nvenc_cache = {}  # Cache NVENC test results to avoid repeated probing
_nvenc_diag = {}   # Store diagnostic info for logging
_gpu_filters_cache = None  # Cache GPU filter availability test

def ffmpeg_supports_nvenc(codec_name="h264_nvenc"):
    """Test if NVENC encoding actually works (not just listed in encoders)."""
    if codec_name in _nvenc_cache:
        return _nvenc_cache[codec_name]
    try:
        # Step 1: Check encoder is listed
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"],
                                      stderr=subprocess.STDOUT, text=True, timeout=15)
        if codec_name not in out:
            _nvenc_cache[codec_name] = False
            _nvenc_diag[codec_name] = f"Codec '{codec_name}' not listed in FFmpeg encoders"
            return False
        # Step 2: Actually test encoding a tiny dummy frame
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="nvenc_test_")
        tmp_path = os.path.join(tmp_dir, "test.mp4")
        try:
            test_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                "-c:v", codec_name, "-t", "0.1", tmp_path
            ]
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
            works = (result.returncode == 0)
            if not works:
                _nvenc_diag[codec_name] = f"Test encode failed: {result.stderr.strip()[:200]}"
            else:
                _nvenc_diag[codec_name] = "OK"
        finally:
            try:
                import shutil
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        _nvenc_cache[codec_name] = works
        return works
    except Exception as e:
        _nvenc_cache[codec_name] = False
        _nvenc_diag[codec_name] = f"Exception: {e}"
        return False

def get_nvenc_diagnostic():
    """Return diagnostic info about GPU/NVENC availability."""
    lines = []
    lines.append(f"  USE_GPU_IF_AVAILABLE = {USE_GPU_IF_AVAILABLE}")
    lines.append(f"  USE_HARDWARE_DECODING = {USE_HARDWARE_DECODING}")
    lines.append(f"  PREFERRED_NVENC_CODEC = {PREFERRED_NVENC_CODEC}")
    lines.append(f"  NVENC_PRESET_SPEED = {NVENC_PRESET_SPEED}")
    # Test primary codec
    primary_ok = ffmpeg_supports_nvenc(PREFERRED_NVENC_CODEC)
    diag = _nvenc_diag.get(PREFERRED_NVENC_CODEC, "not tested")
    lines.append(f"  {PREFERRED_NVENC_CODEC}: {'✓ AVAILABLE' if primary_ok else '✗ UNAVAILABLE'} ({diag})")
    # Test fallback codecs
    for alt in ["hevc_nvenc", "h264_nvenc"]:
        if alt != PREFERRED_NVENC_CODEC:
            alt_ok = ffmpeg_supports_nvenc(alt)
            alt_diag = _nvenc_diag.get(alt, "not tested")
            lines.append(f"  {alt}: {'✓ AVAILABLE' if alt_ok else '✗ UNAVAILABLE'} ({alt_diag})")
    # GPU filter availability
    gpu_filters = ffmpeg_gpu_filters_available()
    lines.append(f"  GPU filters (scale_cuda/overlay_cuda): {'✓ AVAILABLE' if gpu_filters else '✗ UNAVAILABLE'}")
    return "\n".join(lines)

def ffmpeg_gpu_filters_available():
    """Test if FFmpeg supports scale_cuda and overlay_cuda GPU filters."""
    global _gpu_filters_cache
    if _gpu_filters_cache is not None:
        return _gpu_filters_cache
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-filters"],
            stderr=subprocess.STDOUT, text=True, timeout=15
        )
        has_scale = "scale_cuda" in out
        has_overlay = "overlay_cuda" in out
        _gpu_filters_cache = has_scale and has_overlay
        return _gpu_filters_cache
    except Exception:
        _gpu_filters_cache = False
        return False

def get_export_settings():
    audio_bitrate = "192k"
    threads = 4  # Use 4 threads for better CPU utilization (was 0/auto)
    libx264_params = ["-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]  # Changed from slow to ultrafast
    libx264_codec = "libx264"
    nvenc_params = ["-rc", "constqp", "-qp", "22", "-b:v", "0", "-preset", NVENC_PRESET_SPEED, "-multipass", "0", "-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "+faststart"]
    if USE_GPU_IF_AVAILABLE and ffmpeg_supports_nvenc(PREFERRED_NVENC_CODEC):
        return PREFERRED_NVENC_CODEC, nvenc_params, threads, audio_bitrate
    return libx264_codec, libx264_params, threads, audio_bitrate

def reencode_with_libx264(input_path, output_path, log=None):
    # Use hardware acceleration if available
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    
    # Add hardware decoding
    if USE_HARDWARE_DECODING and USE_GPU_IF_AVAILABLE and ffmpeg_supports_nvenc(PREFERRED_NVENC_CODEC):
        cmd.extend(["-hwaccel", "cuda"])
    
    cmd.extend(["-i", input_path])
    
    # Use NVENC if available, otherwise use faster CPU preset
    if USE_GPU_IF_AVAILABLE and ffmpeg_supports_nvenc(PREFERRED_NVENC_CODEC):
        cmd.extend(["-c:v", PREFERRED_NVENC_CODEC, "-rc", "constqp", "-qp", "22", "-b:v", "0", 
                   "-preset", NVENC_PRESET_SPEED, "-multipass", "0",
                   "-pix_fmt", "yuv420p", "-profile:v", "high"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", 
                   "-pix_fmt", "yuv420p", "-profile:v", "high", "-threads", "0"])
    
    cmd.extend(["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path])
    
    if log: log(f"[ffmpeg] Re-encoding to: {output_path} (GPU={'NVENC' if USE_GPU_IF_AVAILABLE and ffmpeg_supports_nvenc(PREFERRED_NVENC_CODEC) else 'No'})")
    try:
        subprocess.check_call(cmd)
        if log: log("[ffmpeg] Re-encode completed.")
        return True
    except subprocess.CalledProcessError as e:
        if log: log(f"[ffmpeg] Re-encode failed: {e}")
        return False

def probe_file_with_ffmpeg(path):
    cmd = ["ffmpeg", "-hide_banner", "-i", path]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=30)
    except Exception as e:
        return False, f"Probe error: {e}"
    bad_signals = ["Invalid data found", "moov atom not found", "Error while decoding", "corrupt", "missing"]
    for sig in bad_signals:
        if sig.lower() in err.lower():
            return False, err
    return True, err

def pre_render_foreground_ffmpeg(input_path, out_path, crop_x, crop_y, crop_w, crop_h, scale_w, scale_h, fps, use_nvenc, log):
    gpu_filters = use_nvenc and ffmpeg_gpu_filters_available()
    
    # Build command with hardware acceleration if available
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    
    if USE_HARDWARE_DECODING and use_nvenc:
        cmd.extend(["-hwaccel", "cuda"])
        if gpu_filters:
            cmd.extend(["-hwaccel_output_format", "cuda"])
    
    cmd.extend(["-i", input_path, "-an"])
    
    if gpu_filters and USE_HARDWARE_DECODING:
        # GPU path: -hwaccel_output_format cuda keeps decoded frames in CUDA memory.
        # crop is a CPU-only filter, so we must hwdownload first, then crop on CPU,
        # then hwupload_cuda back to GPU for scale_cuda.
        vf = f"hwdownload,format=nv12,crop={crop_w}:{crop_h}:{crop_x}:{crop_y},hwupload_cuda,scale_cuda={scale_w}:{scale_h},hwdownload,format=nv12,setsar=1:1"
        if log: log(f"[ffmpeg] Using GPU-accelerated scale_cuda for pre-render")
    else:
        vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={scale_w}:{scale_h}:flags=lanczos,setsar=1:1"
    
    cmd.extend(["-vf", vf, "-r", str(int(fps))])
    
    if use_nvenc and ffmpeg_supports_nvenc(PREFERRED_NVENC_CODEC):
        codec = PREFERRED_NVENC_CODEC
        # Use fastest preset for pre-render (p1 = maximum speed)
        vparams = ["-c:v", codec, "-rc", "constqp", "-qp", "22", "-b:v", "0", "-preset", "p1", "-multipass", "0"]
    else:
        codec = "libx264"
        vparams = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-threads", "0"]  # Use all CPU cores
    
    cmd.extend(vparams + ["-pix_fmt", "yuv420p", out_path])
    
    if log: log(f"[ffmpeg] Pre-render starting -> {os.path.basename(out_path)} (nvenc={use_nvenc}, hwaccel={USE_HARDWARE_DECODING and use_nvenc})")
    try:
        subprocess.check_call(cmd)
        if log: log(f"[ffmpeg] Pre-render completed: {out_path}")
        return True
    except subprocess.CalledProcessError as e:
        if log:
            log(f"[ffmpeg] Pre-render FAILED (return code {e.returncode}).")
            log(" ".join(cmd))
            log(str(e))
        return False
    except Exception as e:
        if log: log(f"[ffmpeg] Pre-render exception: {e}")
        return False

def make_unique_output_path(requested_path, log=None):
    base = os.path.abspath(requested_path)
    dirn, fname = os.path.split(base)
    name, ext = os.path.splitext(fname)
    candidate = base
    if candidate in CREATED_OUTPUTS or os.path.exists(candidate):
        i = 1
        while True:
            new_name = f"{name}_{i}{ext}"
            candidate = os.path.join(dirn, new_name)
            if candidate not in CREATED_OUTPUTS and not os.path.exists(candidate):
                break
            i += 1
        if log: log(f"[output] '{requested_path}' exists -> using '{candidate}' instead")
    else:
        if log: log(f"[output] Using requested output: {candidate}")
    CREATED_OUTPUTS.add(candidate)
    return candidate

# ----------------- Whisper robust loading & transcription -----------------
# Global Whisper model cache — loads the ~3GB model ONCE, reuses for all transcriptions.
# Without caching, each transcribe_captions() call loads a fresh model on GPU.
# In multi-job pipelines, this quickly exhausts GPU memory (3GB × N simultaneous loads)
# and causes 100% GPU usage from memory thrashing without actual speed benefit.
_whisper_model_cache = {}  # {model_name: (model, device)}
_whisper_transcription_lock = threading.Lock()  # Serialize GPU transcription (Whisper isn't thread-safe)

def _get_cached_whisper_model(model_name="large", tries=3, log=None):
    """Load Whisper model once and cache it. Thread-safe: called within _whisper_transcription_lock."""
    global _whisper_model_cache
    # Note: This function is always called within _whisper_transcription_lock (from transcribe_captions).
    # The lock ensures only one thread can check/update the cache at a time.
    if model_name in _whisper_model_cache:
        model, device = _whisper_model_cache[model_name]
        if log:
            log(f"[whisper] ✓ Using cached model '{model_name}' on {device.upper()} (no reload needed)")
        return model, device
    if log:
        log(f"[whisper] Loading model '{model_name}' for the first time (will be cached for reuse)...")
    model, device = _load_whisper_model_with_retries(model_name, tries=tries, log=log)
    _whisper_model_cache[model_name] = (model, device)
    return model, device

def _release_whisper_model(log=None):
    """Release cached Whisper model and free GPU memory for FFmpeg NVENC export."""
    global _whisper_model_cache
    with _whisper_transcription_lock:
        if _whisper_model_cache:
            if log:
                log(f"[whisper] 🧹 Releasing cached Whisper model(s) to free GPU memory for export...")
            _whisper_model_cache.clear()
            gc.collect()
            try:
                import torch  # lazy import — only needed when releasing GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if log:
                        log(f"[whisper] ✓ GPU memory freed (torch.cuda.empty_cache)")
            except Exception:
                pass
        else:
            if log:
                log(f"[whisper] No cached model to release")

def _find_and_remove_corrupted_whisper_models(model_name, log=None):
    removed = []
    possible_roots = [
        os.path.join(os.path.expanduser("~"), ".cache", "whisper"),
        os.path.join(os.path.expanduser("~"), ".cache"),
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.getcwd(), ".cache"),
    ]
    possible_roots.append(os.path.expanduser("~"))
    lower_name = model_name.lower()
    for root in possible_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                fn_lower = fn.lower()
                full = os.path.join(dirpath, fn)
                if lower_name in fn_lower and (fn_lower.endswith(".pt") or fn_lower.endswith(".bin") or ".part" in fn_lower or fn_lower.endswith(".tmp")):
                    try:
                        os.remove(full)
                        removed.append(full)
                        if log: log(f"[whisper-cleanup] removed: {full}")
                    except Exception as e:
                        if log: log(f"[whisper-cleanup] failed to remove {full}: {e}")
    return removed

def _load_whisper_model_with_retries(model_name="large-v3", tries=3, log=None):
    import torch    # lazy import — heavy module, only loaded when transcription is needed
    import whisper  # lazy import — heavy module, only loaded when transcription is needed
    last_exc = None
    
    # Detect GPU availability for Whisper with improved detection
    device = "cpu"  # Default to CPU
    cuda_available = False
    
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                device = "cuda"
                if log:
                    gpu_name = torch.cuda.get_device_name(0)
                    log(f"[whisper] GPU detected: {gpu_name}")
                    log(f"[whisper] Will use CUDA acceleration for transcription")
            else:
                if log:
                    log(f"[whisper] CUDA available but no GPU devices found - will use CPU")
        else:
            if log:
                log(f"[whisper] CUDA not available in PyTorch - will use CPU (slower)")
                log(f"[whisper] For GPU support, install PyTorch with CUDA:")
                log(f"[whisper]   RTX 5070/5080/5090: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130")
                log(f"[whisper]   RTX 4060-4090:      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except Exception as e:
        if log:
            log(f"[whisper] GPU detection failed ({str(e)}) - will use CPU")
    
    for attempt in range(1, tries + 1):
        try:
            if log: log(f"[whisper] Loading model '{model_name}' on {device.upper()} (attempt {attempt}/{tries})...")
            model = whisper.load_model(model_name, device=device)
            if log: log(f"[whisper] Model '{model_name}' loaded successfully on {device.upper()}.")
            return model, device  # Return both model and device used
        except RuntimeError as e:
            msg = str(e).lower()
            error_str = str(e)
            last_exc = e
            if "sha256" in msg or "checksum" in msg or "downloaded but the sha256" in msg:
                if log: log(f"[whisper] Checksum error for '{model_name}': {e}")
                removed = _find_and_remove_corrupted_whisper_models(model_name, log=log)
                if removed:
                    if log: log(f"[whisper] Removed {len(removed)} suspected corrupted file(s). Retrying download...")
                    time.sleep(1.0 + attempt * 0.5)
                    continue
                else:
                    if log: log("[whisper] Could not find model file(s) to remove — retrying download anyway.")
                    time.sleep(1.0 + attempt * 0.5)
                    continue
            elif "cuda" in msg and device == "cuda":
                # CUDA error - check if it's architecture mismatch
                if "no kernel image" in error_str or ("kernel" in error_str and "sm" in error_str):
                    # Architecture mismatch - PyTorch built without support for this GPU
                    if log:
                        log("[whisper] ========================================")
                        log("[whisper] ERROR: PyTorch Architecture Mismatch!")
                        log("[whisper] ========================================")
                        log("[whisper] PyTorch was built without support for your GPU architecture")
                        log("[whisper] ")
                        log("[whisper] SOLUTION: Reinstall PyTorch with proper GPU support:")
                        log("[whisper]   pip uninstall torch torchvision torchaudio")
                        log("[whisper]   RTX 5070/5080/5090 (Blackwell):")
                        log("[whisper]     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130")
                        log("[whisper]   RTX 4060/4070/4080/4090 (Ada Lovelace):")
                        log("[whisper]     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                        log("[whisper] ")
                        log("[whisper] Or as temporary workaround, using CPU (slower)...")
                        log("[whisper] ========================================")
                    device = "cpu"
                    if attempt < tries:
                        time.sleep(0.5)
                        continue
                else:
                    # Other CUDA error - try falling back to CPU
                    if log: log(f"[whisper] CUDA error loading model: {e}")
                    if log: log(f"[whisper] Falling back to CPU...")
                    device = "cpu"
                    if attempt < tries:
                        time.sleep(0.5)
                        continue
            else:
                if log: log(f"[whisper] RuntimeError loading model '{model_name}': {e}")
                raise
        except Exception as e:
            last_exc = e
            if log: log(f"[whisper] Unexpected error while loading model '{model_name}': {e}")
            time.sleep(0.5 + attempt * 0.5)
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Could not load whisper model '{model_name}' for unknown reason.")

def transcribe_captions(voice_path, log=None, translate_to=None):
    """
    Transcribe audio to text captions using Whisper, with optional translation.
    
    Uses a global model cache to avoid re-loading the ~3GB Whisper model on GPU
    for every call. Serializes GPU transcription via _whisper_transcription_lock
    to prevent concurrent GPU usage that causes 100% GPU with no speed benefit.
    
    Args:
        voice_path: Path to audio file
        log: Optional logging function
        translate_to: Target language code for translation (None or 'none' to skip)
    
    Returns:
        List of caption segments
    """
    def _default_log(s):
        try:
            print(s)
        except Exception:
            pass
    log_fn = _default_log if log is None else log
    
    # Serialize GPU transcription — only one Whisper transcription at a time.
    # Without this lock, the interleaved voice pipeline can run multiple transcriptions
    # concurrently (submitter + completion workers), each trying to use GPU simultaneously.
    # This causes GPU memory thrashing and 100% usage without actual speed benefit.
    with _whisper_transcription_lock:
        # Use cached Whisper model (loaded once, reused for all transcriptions)
        # Using 'large' (not large-v3) for accurate word-level timestamps needed for caption sync
        # Medium/small models are faster but timestamps not accurate enough, causing caption-voice mismatch
        try:
            model, device = _get_cached_whisper_model("large", tries=3, log=log_fn)
        except Exception as e_large:
            log_fn(f"[whisper] Failed to load 'large' model after retries: {e_large}")
            log_fn("[whisper] Falling back to 'medium' model (faster but less accurate timestamps).")
            try:
                model, device = _get_cached_whisper_model("medium", tries=2, log=log_fn)
            except Exception as e_medium:
                log_fn(f"[whisper] Failed to load 'medium' model as well: {e_medium}")
                raise RuntimeError("Whisper models unavailable. Verifică conexiunea la internet și spațiul pe disc.") from e_medium
        
        # Show appropriate message based on actual device being used
        if device == "cuda":
            log_fn("[whisper] Transcribing audio with word-level timestamps (GPU accelerated)...")
        else:
            log_fn("[whisper] Transcribing audio with word-level timestamps (CPU mode)...")
        
        # Enable word_timestamps for precise caption synchronization
        # Use FP16 on GPU for faster inference (2x speedup with minimal quality loss)
        # Only enable FP16 if actually running on GPU
        use_fp16 = (device == "cuda")
        if use_fp16:
            log_fn("[whisper] Using FP16 precision on GPU for faster transcription (2x speedup)")
        
        result = model.transcribe(voice_path, word_timestamps=True, fp16=use_fp16)
        log_fn("[whisper] Transcription finished.")
    
    # Post-processing outside the lock (doesn't need GPU)
    segments = result["segments"]
    
    # Log auto-detected source language from Whisper
    detected_lang = result.get("language", "unknown")
    log_fn(f"[whisper] Detected source language: {detected_lang}")
    
    # Apply translation if requested
    # Translation is enabled when translate_to is specified and not 'none'
    if translate_to and translate_to != 'none':
        log_fn(f"[TRANSCRIBE] Auto-detected source language: {detected_lang}")
        log_fn(f"[TRANSCRIBE] Translating from {detected_lang} → {translate_to}...")
        segments = translate_segments(segments, target_language=translate_to, log=log_fn)
    
    return segments

# ----------------- caption generation -----------------
def generate_caption_image(text, preferred_font=None, log=None):
    """
    Generate caption image with visible background and proper transparency settings.
    
    FIXED ISSUES:
    - Changed bubble background from transparent (0,0,0,0) to semi-opaque white (255,255,255,220)
    - Removed duplicate shadow drawing code
    - Added logging for caption rendering errors
    - Normalized color values to ensure valid RGBA ranges
    """
    try:
        if log:
            log(f"[CAPTION-GEN] ═══════════════════════════════════════════════")
            log(f"[CAPTION-GEN] Starting caption generation")
            log(f"[CAPTION-GEN] Text: '{text[:80]}{'...' if len(text) > 80 else ''}'")
            log(f"[CAPTION-GEN] Text length: {len(text)} characters")
            log(f"[CAPTION-GEN] Preferred font: {preferred_font or 'default'}")
    except Exception:
        pass
    
    try:
        font = load_preferred_font_cached(preferred_font or CAPTION_FONT_PREFERRED, CAPTION_FONT_SIZE, log=log)
        try:
            if log:
                font_info = f"{LOADED_FONT_PATH}" if LOADED_FONT_PATH else "unknown"
                log(f"[CAPTION-GEN] Font loaded successfully: {font_info}")
                log(f"[CAPTION-GEN] Font size: {CAPTION_FONT_SIZE}px")
        except Exception:
            pass
    except Exception as e:
        try:
            if log:
                log(f"[CAPTION-GEN ERROR] Failed to load font: {e}")
        except Exception:
            pass
        raise
    
    text = normalize_text(text)
    
    def measure_text(draw, s):
        try:
            l, t, r, b = draw.textbbox((0,0), s, font=font)
            return (r-l, b-t)
        except Exception:
            return font.getsize(s)
    
    tmp_img = Image.new("RGBA", (WIDTH, 10), (0,0,0,0))
    draw_tmp = ImageDraw.Draw(tmp_img)
    max_text_width = WIDTH - 160
    words = text.split()
    lines, current = [], ""
    for w in words:
        test = (current + " " + w).strip()
        w_width, _ = measure_text(draw_tmp, test)
        if w_width <= max_text_width:
            current = test
        else:
            if current: lines.append(current)
            current = w
    if current: lines.append(current)
    
    line_heights = []
    max_line_w = 0
    for line in lines:
        lw, lh = measure_text(draw_tmp, line)
        line_heights.append(lh)
        max_line_w = max(max_line_w, lw)
    
    line_spacing = int(CAPTION_FONT_SIZE * 0.18)
    padding_x = 24; padding_y = 24  # increased vertical padding to avoid bottom cutoff
    extra_bottom_margin = int(CAPTION_FONT_SIZE * 0.35)  # extra transparent margin at bottom
    total_height = sum(line_heights) + line_spacing*(len(lines)-1) + padding_y*2 + extra_bottom_margin
    bubble_width = min(WIDTH - 80, max_line_w + padding_x*2)
    
    try:
        if log:
            log(f"[CAPTION-GEN] Image dimensions: {WIDTH}x{total_height + 8 + extra_bottom_margin}")
            log(f"[CAPTION-GEN] Number of lines: {len(lines)}")
            log(f"[CAPTION-GEN] Bubble width: {bubble_width}px, height: {total_height - extra_bottom_margin}px")
            log(f"[CAPTION-GEN] Padding: x={padding_x}px, y={padding_y}px")
    except Exception:
        pass
    
    img = Image.new("RGBA", (WIDTH, total_height + 8 + extra_bottom_margin), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    bubble_x0 = (WIDTH - bubble_width)//2; bubble_x1 = bubble_x0 + bubble_width
    bubble_y0 = 4; bubble_y1 = bubble_y0 + (total_height - extra_bottom_margin)  # bubble excludes the extra bottom margin
    
    # Draw shadow (FIXED: removed duplicate code)
    shadow = Image.new("RGBA", (bubble_width, total_height - extra_bottom_margin), (0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    try:
        radius = int(padding_y*0.8)
        sd.rounded_rectangle([0,0,bubble_width,total_height - extra_bottom_margin], radius=radius, fill=(0,0,0,200))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))
        img.paste(shadow, (bubble_x0+0, bubble_y0+2), shadow)
        try:
            if log:
                log(f"[CAPTION-GEN] Shadow drawn successfully with rounded corners, radius={radius}px")
        except Exception:
            pass
    except Exception as e:
        try:
            if log:
                log(f"[CAPTION-GEN WARNING] Rounded shadow failed, using rectangle: {e}")
        except Exception:
            pass
        sd.rectangle([0,0,bubble_width,total_height - extra_bottom_margin], fill=(0,0,0,140))
        sd_im = shadow.filter(ImageFilter.GaussianBlur(radius=6))
        img.paste(sd_im, (bubble_x0+0, bubble_y0+2), sd_im)
    
    # Draw bubble background - TRANSPARENT (user requested no white box, just text)
    # Background is now transparent - text visibility comes from strong stroke outline
    bubble_fill = (0, 0, 0, 0)  # Fully transparent - no white box behind text
    try:
        draw.rounded_rectangle([bubble_x0,bubble_y0,bubble_x1,bubble_y1], radius=int(padding_y*0.8), fill=bubble_fill)
        try:
            if log:
                log(f"[CAPTION-GEN] Background: TRANSPARENT (no white box - text only with stroke)")
        except Exception:
            pass
    except Exception as e:
        try:
            if log:
                log(f"[CAPTION-GEN WARNING] Rounded bubble failed, using rectangle: {e}")
        except Exception:
            pass
        draw.rectangle([bubble_x0,bubble_y0,bubble_x1,bubble_y1], fill=bubble_fill)
    
    # Normalize and validate color values (FIXED: ensure colors are in valid range)
    def normalize_color(color):
        """Ensure RGBA color tuple is in valid range (0-255)."""
        return tuple(max(0, min(255, int(c))) for c in color)
    
    try:
        stroke_w = int(CAPTION_STROKE_WIDTH)
    except Exception:
        stroke_w = max(1, int(CAPTION_FONT_SIZE * 0.05))
    
    # Increase stroke width for better visibility without white background
    # Make stroke at least 3 pixels for good visibility
    stroke_w = max(3, stroke_w)
    
    try:
        stroke_fill = normalize_color(tuple(CAPTION_STROKE_COLOR))
    except Exception:
        stroke_fill = (0,0,0,150)
    
    # Make stroke fully opaque (255 alpha) for better visibility without background
    stroke_fill = (stroke_fill[0], stroke_fill[1], stroke_fill[2], 255)
    
    try:
        text_fill = normalize_color(tuple(CAPTION_TEXT_COLOR))
    except Exception:
        text_fill = (255,255,255,255)
    
    try:
        if log:
            log(f"[CAPTION-GEN] Text color: RGBA{text_fill}")
            log(f"[CAPTION-GEN] Stroke color: RGBA{stroke_fill}")
            log(f"[CAPTION-GEN] Stroke width: {stroke_w}px (enhanced for visibility)")
    except Exception:
        pass
    
    # Draw text
    y = bubble_y0 + padding_y
    for line_idx, line in enumerate(lines):
        lw, lh = measure_text(draw, line)
        x = bubble_x0 + (bubble_width - lw)//2
        try:
            draw.text((x,y), line, font=font, fill=text_fill, stroke_width=stroke_w, stroke_fill=stroke_fill)
        except TypeError as e:
            try:
                if log:
                    log(f"[CAPTION-GEN WARNING] stroke_width not supported, using manual stroke: {e}")
            except Exception:
                pass
            # Fallback for older Pillow versions
            offs = [(dx, dy) for dx in (-stroke_w,0,stroke_w) for dy in (-stroke_w,0,stroke_w)]
            for dx, dy in offs:
                if dx==0 and dy==0: continue
                draw.text((x+dx, y+dy), line, font=font, fill=stroke_fill)
            draw.text((x,y), line, font=font, fill=text_fill)
        except Exception as e:
            try:
                if log:
                    log(f"[CAPTION-GEN ERROR] Failed to draw text line {line_idx}: {e}")
            except Exception:
                pass
            raise
        y += lh + line_spacing
    
    try:
        if log:
            log(f"[CAPTION-GEN] ✓ Caption image generated successfully!")
            log(f"[CAPTION-GEN] Final image size: {img.size[0]}x{img.size[1]}px")
            log(f"[CAPTION-GEN] ═══════════════════════════════════════════════")
    except Exception:
        pass
    
    return img
# ----------------- preview helpers (with timeline support) -----------------
def extract_and_scale_frame(video_path, time_sec=None, desired_width=360):
    # video_path may be a path (str) or a VideoFileClip instance. If it's a clip, reuse it.
    created_clip = False
    if isinstance(video_path, VideoFileClip):
        clip = video_path
    else:
        clip = VideoFileClip(video_path)
        created_clip = True
    if time_sec is None:
        t = min(max(0.001, clip.duration / 2.0), clip.duration - 0.001)
    else:
        t = min(max(0.0, float(time_sec)), max(0.001, clip.duration - 0.001))
    frame = clip.get_frame(t)
    if created_clip:
        try:
            clip.close()
        except Exception:
            pass
    img = Image.fromarray(frame).convert("RGB")
    w, h = img.size
    scale = desired_width / w
    new_w = desired_width
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img, scale

def overlay_crop_on_image(base_img, top_ratio, bottom_ratio, bar_color=(0,0,0,140), line_color=(0,0,0,255)):
    img_rgba = base_img.convert("RGBA")
    w, h = img_rgba.size
    top_y = int(round(h * top_ratio))
    bottom_y = int(round(h * (1.0 - bottom_ratio)))
    if top_y >= bottom_y - 2:
        if top_y > h//2:
            top_y = max(0, bottom_y - 2)
        else:
            bottom_y = min(h, top_y + 2)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    if top_y > 0:
        draw.rectangle([0, 0, w, top_y], fill=bar_color)
    if bottom_y < h:
        draw.rectangle([0, bottom_y, w, h], fill=bar_color)
    line_thickness = max(1, int(round(2 * (w/360))))
    draw.rectangle([0, top_y-line_thickness//2, w, top_y+line_thickness//2], fill=line_color)
    draw.rectangle([0, bottom_y-line_thickness//2, w, bottom_y+line_thickness//2], fill=line_color)
    composed = Image.alpha_composite(img_rgba, overlay)
    return composed

# ----------------- Save / Load crop settings -----------------
def save_crop_settings(top_pct, bottom_pct, filepath=CROP_SETTINGS_FILE):
    data = {"top_pct": float(top_pct), "bottom_pct": float(bottom_pct)}
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True, filepath
    except Exception as e:
        return False, str(e)

def load_crop_settings(filepath=CROP_SETTINGS_FILE):
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None

# ----------------- Export composer with monitoring & fallback -----------------
def _make_ffmpeg_params_for_codec(codec):
    """
    Get FFmpeg encoding parameters for the specified codec.
    
    Note: These are OUTPUT encoding parameters only. MoviePy passes these to FFmpeg
    for the final encoding step. Hardware decoding (-hwaccel) doesn't work here
    because MoviePy reads/processes frames in Python before encoding.
    
    The main export bottleneck is MoviePy's Python frame processing, not encoding.
    GPU acceleration is already being used optimally via:
    - pre_render_foreground_ffmpeg() uses -hwaccel cuda for foreground (works!)
    - NVENC encoding below (works!)
    """
    if codec in ("h264_nvenc", "hevc_nvenc"):
        # GPU encoding with NVENC - optimized for maximum speed
        return [
            "-rc", "constqp",          # Constant QP for fastest encoding
            "-qp", "22",               # Quality level (lower = better, 22 is good balance)
            "-b:v", "0",               # Let QP control quality
            "-preset", NVENC_PRESET_SPEED,  # p1 for max speed (configurable)
            "-multipass", "0",         # Disable multipass for fastest encoding
            "-pix_fmt", "yuv420p",     # Standard pixel format
            "-profile:v", "high",      # H.264 High profile
            "-movflags", "+faststart"  # Web streaming optimization
        ]
    else:
        # CPU encoding with libx264 - fallback option
        return [
            "-preset", "ultrafast",    # Fastest CPU preset
            "-crf", "20",              # Constant quality
            "-pix_fmt", "yuv420p",     # Standard pixel format
            "-movflags", "+faststart"  # Web streaming optimization
        ]

# ----------------- FFmpeg Fast Export Functions -----------------

def _calculate_text_lines(text, max_chars=40, words_per_line=None):
    """
    Calculate line breaks for text to fit within max_chars per line or words per line.
    Breaks at word boundaries when possible.
    
    Args:
        text: String to wrap
        max_chars: Maximum characters per line (used if words_per_line is None)
        words_per_line: Maximum words per line (overrides max_chars if provided)
        
    Returns:
        List of text lines
    """
    # Handle empty or whitespace-only text
    if not text or not text.strip():
        return [""]
    
    words = text.split()
    
    # If words_per_line is specified, group by word count instead of characters
    if words_per_line and words_per_line > 0:
        if not words:
            return [""]
        lines = []
        for i in range(0, len(words), words_per_line):
            line_words = words[i:i + words_per_line]
            lines.append(' '.join(line_words))
        return lines
    
    # Otherwise use character-based wrapping (original logic)
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_len = len(word)
        # +1 for space between words
        needed = word_len if not current_line else current_length + 1 + word_len
        
        if needed <= max_chars:
            current_line.append(word)
            current_length = needed
        else:
            # Start new line
            if current_line:
                lines.append(' '.join(current_line))
            # Handle very long words
            if word_len > max_chars:
                lines.append(word[:max_chars])
                current_line = [word[max_chars:]] if len(word) > max_chars else []
                current_length = len(word[max_chars:]) if len(word) > max_chars else 0
            else:
                current_line = [word]
                current_length = word_len
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines if lines else [text]


def _escape_ffmpeg_text(text):
    """
    Escape special characters for FFmpeg drawtext filter.
    
    Args:
        text: String to escape
        
    Returns:
        Escaped string safe for FFmpeg
    """
    # FFmpeg drawtext special characters that need escaping
    text = text.replace('\\', '\\\\')  # Backslash first!
    text = text.replace("'", "\\'")    # Single quote
    text = text.replace(':', '\\:')    # Colon
    text = text.replace('%', '\\%')    # Percent
    text = text.replace('[', '\\[')    # Square brackets
    text = text.replace(']', '\\]')
    return text


def _rgba_to_hex(rgba_tuple):
    """
    Convert RGBA tuple to hex color string for FFmpeg.
    
    Args:
        rgba_tuple: (R, G, B, A) tuple with values 0-255
        
    Returns:
        Hex color string like '#RRGGBB' or '0xRRGGBB'
    """
    try:
        r, g, b = int(rgba_tuple[0]), int(rgba_tuple[1]), int(rgba_tuple[2])
        # FFmpeg accepts #RRGGBB or 0xRRGGBB format
        return f"0x{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "0xFFFFFF"  # Default to white


def _generate_ass_subtitle_file(caption_segments, output_path, font_name="Arial", fontsize=56, 
                                text_color_rgba=(255, 255, 255, 255), stroke_width=3,
                                stroke_color_rgba=(0, 0, 0, 150),
                                video_width=1080, video_height=1920, y_offset=0, log_fn=None):
    """
    Generate an ASS (Advanced SubStation Alpha) subtitle file for word-by-word captions.
    
    Args:
        caption_segments: List of caption dictionaries with 'text', 'start', 'end'
        output_path: Path where to save the .ass file
        font_name: Font name (e.g., "Bangers", "Arial")
        fontsize: Font size in pixels
        text_color_rgba: Text color as RGBA tuple
        stroke_width: Outline/border width
        stroke_color_rgba: Stroke/outline color as RGBA tuple
        video_width: Video width for PlayResX
        video_height: Video height for PlayResY
        y_offset: Vertical offset in pixels (negative = move up, positive = move down)
        log_fn: Optional logging function
        
    Returns:
        Path to generated ASS file
    """
    import re
    
    # Convert RGBA to ASS color format (BGR in hex with alpha)
    # ASS uses &HAABBGGRR format
    r, g, b, a = text_color_rgba
    # Convert alpha: 255 = fully opaque = 00, 0 = fully transparent = FF
    alpha_hex = f"{255 - int(a):02X}"
    text_color_ass = f"&H{alpha_hex}{b:02X}{g:02X}{r:02X}"
    # Convert stroke color RGBA to ASS format
    sr, sg, sb, sa = stroke_color_rgba
    stroke_alpha_hex = f"{255 - int(sa):02X}"
    outline_color_ass = f"&H{stroke_alpha_hex}{sb:02X}{sg:02X}{sr:02X}"
    
    # Calculate MarginV from y_offset
    # y_offset is negative when moving up from bottom, positive when moving down
    # In ASS, MarginV is the margin from bottom edge (where Alignment=2 positions from)
    # 
    # MoviePy uses: y = HEIGHT - img_clip.h + y_offset
    #   y_offset = -618 → caption is 618px UP from bottom
    #   y_offset = 0    → caption is at bottom
    #   y_offset = +100 → caption is 100px BELOW bottom (off-screen)
    #
    # For ASS with Alignment=2 (bottom center):
    #   MarginV = 0   → text at very bottom
    #   MarginV = 618 → text 618px up from bottom
    #
    # So: MarginV = abs(y_offset) when y_offset is negative
    #     MarginV = 0 when y_offset >= 0 (at or below bottom)
    if y_offset < 0:
        margin_v = abs(y_offset)
    else:
        margin_v = 0  # At bottom or below (clip to bottom)
    
    if log_fn:
        log_fn(f"[ASS] Caption Y offset: {y_offset}px → MarginV: {margin_v}px")
        log_fn(f"[ASS] Font size in ASS: {fontsize}px")
    
    # ASS file header
    ass_content = f"""[Script Info]
Title: Generated Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{fontsize},{text_color_ass},&H00FFFFFF,{outline_color_ass},&H00000000,0,0,0,0,100,100,0,0,1,{stroke_width},0,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Add each caption as a dialogue line
    for segment in caption_segments:
        # Convert time to ASS format (H:MM:SS.CS where CS is centiseconds)
        start_time = segment['start']
        end_time = segment['end']
        
        start_h = int(start_time // 3600)
        start_m = int((start_time % 3600) // 60)
        start_s = int(start_time % 60)
        start_cs = int((start_time % 1) * 100)
        start_str = f"{start_h}:{start_m:02d}:{start_s:02d}.{start_cs:02d}"
        
        end_h = int(end_time // 3600)
        end_m = int((end_time % 3600) // 60)
        end_s = int(end_time % 60)
        end_cs = int((end_time % 1) * 100)
        end_str = f"{end_h}:{end_m:02d}:{end_s:02d}.{end_cs:02d}"
        
        # Normalize diacritics and escape text for ASS (replace newlines with \N)
        text = normalize_text(segment['text']).replace('\n', '\\N')
        
        ass_content += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n"
    
    # Write ASS file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    return output_path


def _build_caption_drawtext_filter(caption_text, start_time, end_time, video_width, video_height, 
                                   fontsize=56, font_path=None, text_color="0xFFFFFF", 
                                   stroke_color="0x000000", stroke_width=3, words_per_line=None,
                                   y_offset=0):
    """
    Build FFmpeg drawtext filter for a single caption with custom styling.
    
    Args:
        caption_text: Text to display
        start_time: Start time in seconds
        end_time: End time in seconds
        video_width: Video width in pixels
        video_height: Video height in pixels
        fontsize: Font size in pixels
        font_path: Path to custom font file (optional)
        text_color: Text color in hex format (e.g., '0xFFFFFF')
        stroke_color: Stroke/border color in hex format (e.g., '0x000000')
        stroke_width: Stroke width in pixels
        words_per_line: Maximum words per line (optional, overrides character-based wrapping)
        y_offset: Vertical offset in pixels (negative = move up, positive = move down).
                  Matches CAPTION_Y_OFFSET: 0 = bottom, -618 = 618px up from bottom.
        
    Returns:
        FFmpeg drawtext filter string
    """
    # Normalize diacritics and calculate text wrapping with word-based grouping if specified
    lines = _calculate_text_lines(normalize_text(caption_text), max_chars=40, words_per_line=words_per_line)
    
    # Escape text for FFmpeg
    escaped_lines = [_escape_ffmpeg_text(line) for line in lines]
    text_with_newlines = '\\n'.join(escaped_lines)
    
    # Position: use y_offset to match preview/MoviePy positioning
    # MoviePy uses: y = HEIGHT - img_clip.h + y_offset
    # FFmpeg drawtext equivalent: y = h - text_h + y_offset
    # y_offset = 0 → text at bottom, y_offset = -618 → text 618px up from bottom
    # 'th' is the rendered text height in FFmpeg drawtext expressions
    # Small upward correction: FFmpeg th is slightly smaller than PIL text height,
    # causing exported captions to appear a few pixels lower than preview.
    y_correction = int(fontsize * 0.08)
    y_position = f"h-th+({y_offset})-{y_correction}"
    
    # Build drawtext filter with custom styling
    filter_parts = [
        f"drawtext=text='{text_with_newlines}'",
        f"fontsize={fontsize}",
        f"fontcolor={text_color}",
        f"borderw={stroke_width}",
        f"bordercolor={stroke_color}",
        f"x=(w-text_w)/2",  # Center horizontally
        f"y={y_position}",   # Bottom positioning with offset
        f"enable='gte(t,{start_time:.3f})*lt(t,{end_time:.3f})'"  # Timing (exclusive end to avoid overlap)
    ]
    
    # Add custom font if provided
    if font_path and os.path.exists(font_path):
        # Escape font path for FFmpeg (handle spaces and special chars)
        escaped_font_path = font_path.replace('\\', '/').replace(':', '\\:')
        filter_parts.insert(1, f"fontfile='{escaped_font_path}'")
    
    return ':'.join(filter_parts)


def _build_all_caption_filters(caption_segments, video_width, video_height, 
                               font_path=None, text_color="0xFFFFFF", 
                               stroke_color="0x000000", stroke_width=3, words_per_line=None, fontsize=56,
                               y_offset=0):
    """
    Build all caption drawtext filters and chain them together with custom styling.
    
    Args:
        caption_segments: List of caption dictionaries with 'text', 'start', 'end'
        video_width: Video width in pixels
        video_height: Video height in pixels
        font_path: Path to custom font file (optional)
        text_color: Text color in hex format
        stroke_color: Stroke color in hex format
        stroke_width: Stroke width in pixels
        words_per_line: Maximum words per line (optional)
        fontsize: Font size in pixels
        y_offset: Vertical offset in pixels (negative = move up, positive = move down)
        
    Returns:
        Complete filter_complex string for all captions
    """
    if not caption_segments:
        return None
    
    filters = []
    
    for i, segment in enumerate(caption_segments):
        caption_filter = _build_caption_drawtext_filter(
            caption_text=segment['text'],
            start_time=segment['start'],
            end_time=segment['end'],
            video_width=video_width,
            video_height=video_height,
            fontsize=fontsize,
            font_path=font_path,
            text_color=text_color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            words_per_line=words_per_line,
            y_offset=y_offset
        )
        filters.append(caption_filter)
    
    # Chain all filters together
    return ','.join(filters)


def _build_ffmpeg_effect_filters(effect_settings, log_fn=None):
    """
    Build FFmpeg filter string for video effects (to match MoviePy effects).
    
    FFmpeg equivalents:
    - Sharpness: unsharp filter
    - Saturation: eq filter (saturation parameter)
    - Contrast: eq filter (contrast parameter)
    - Brightness: eq filter (brightness parameter)
    - Vintage: colorbalance + noise filters
    
    Args:
        effect_settings: Dictionary with effect settings
        log_fn: Optional logging function
        
    Returns:
        FFmpeg filter string or empty string if no effects
    """
    if not effect_settings:
        return ""
    
    effect_filters = []
    active_effects = []
    
    # Check for sharpness effect (Resilience)
    if effect_settings.get('effect_sharpness', False):
        intensity = effect_settings.get('effect_sharpness_intensity', 1.5)
        # unsharp filter: luma_msize_x:luma_msize_y:luma_amount
        # intensity 1.5 -> luma_amount 1.5, intensity 2.0 -> luma_amount 2.0
        luma_amount = min(5.0, max(0.5, intensity))  # Clamp between 0.5 and 5.0
        effect_filters.append(f"unsharp=5:5:{luma_amount}")
        active_effects.append(f"Sharpness ({intensity}x)")
    
    # Build eq filter parameters for saturation, contrast, brightness
    eq_params = []
    
    # Check for saturation effect (Vibrance)
    if effect_settings.get('effect_saturation', False):
        intensity = effect_settings.get('effect_saturation_intensity', 1.3)
        # eq saturation: 1.0 = normal, >1.0 = more saturated
        saturation = min(3.0, max(0.0, intensity))
        eq_params.append(f"saturation={saturation}")
        active_effects.append(f"Saturation ({intensity}x)")
    
    # Check for contrast effect (HDR)
    if effect_settings.get('effect_contrast', False):
        intensity = effect_settings.get('effect_contrast_intensity', 1.2)
        # eq contrast: 1.0 = normal, >1.0 = more contrast
        contrast = min(2.0, max(-2.0, intensity))
        eq_params.append(f"contrast={contrast}")
        active_effects.append(f"Contrast ({intensity}x)")
    
    # Check for brightness effect
    if effect_settings.get('effect_brightness', False):
        intensity = effect_settings.get('effect_brightness_intensity', 1.15)
        # eq brightness: 0.0 = normal, >0 = brighter, <0 = darker
        # Convert multiplier to additive: 1.15 -> 0.15
        brightness = min(1.0, max(-1.0, intensity - 1.0))
        eq_params.append(f"brightness={brightness}")
        active_effects.append(f"Brightness ({intensity}x)")
    
    # Add eq filter if any eq parameters were set
    if eq_params:
        effect_filters.append(f"eq={':'.join(eq_params)}")
    
    # Check for vintage effect
    if effect_settings.get('effect_vintage', False):
        grain_intensity = effect_settings.get('effect_vintage_intensity', 0.3)
        # Vintage effect: sepia tone via colorbalance + optional noise
        # colorbalance for warm sepia tones: increase red/yellow in highlights/midtones
        # Note: intensity 0.3 -> moderate vintage look
        shadow_red = min(1.0, max(-1.0, grain_intensity * 0.3))
        midtone_red = min(1.0, max(-1.0, grain_intensity * 0.2))
        highlight_yellow = min(1.0, max(-1.0, grain_intensity * 0.15))
        effect_filters.append(f"colorbalance=rs={shadow_red}:gs=-{shadow_red*0.5}:bs=-{shadow_red}:rm={midtone_red}:gm=0:bm=-{midtone_red*0.5}:rh=0:gh={highlight_yellow}:bh=-{highlight_yellow}")
        active_effects.append(f"Vintage ({grain_intensity}x)")
    
    # Log active effects
    if log_fn and active_effects:
        log_fn(f"[EXPORT] Applying FFmpeg effects: {', '.join(active_effects)}")
    
    # Combine all effect filters
    if effect_filters:
        return ','.join(effect_filters)
    return ""


def _generate_capcut_metadata():
    """Generate CapCut/vicut-style metadata key-value pairs for MP4 export.

    Returns a list of (key, value) tuples suitable for FFmpeg ``-metadata`` flags.
    Each call produces unique random IDs so every exported video looks like a
    distinct CapCut project.
    """
    import uuid
    import random

    video_id = str(uuid.uuid4())
    music_ids = f"{uuid.uuid4()},{uuid.uuid4()}"
    # Real CapCut effect IDs are 19-digit integers in the 7.3–7.5×10¹⁸ range
    effect_id1 = str(random.randint(7300000000000000000, 7499999999999999999))
    effect_id2 = str(random.randint(7300000000000000000, 7499999999999999999))

    artwork = {
        "data": {
            "editType": "default",
            "infoStickerId": "",
            "is_ai_lyric": 0,
            "is_aimusic_mv": 0,
            "is_use_ai_image_generation": 0,
            "is_use_ai_video_generation": 0,
            "is_use_aimusic_bgm": 0,
            "is_use_aimusic_vocal": 0,
            "is_use_graph_chart": 0,
            "is_use_jichuang_mode_in_ai_writer": 0,
            "is_use_relight": 0,
            "is_use_vc_sing_clone": 1,
            "is_use_voice_clone": "0",
            "motion_blur_cnt": 0,
            "musicId": music_ids,
            "os": "windows",
            "product": "vicut",
            "stickerId": "",
            "videoEffectId": f"{effect_id1},{effect_id2}",
            "videoId": video_id,
            "videoParams": {
                "be": 0, "ef": 2, "ft": 2, "ma": 0, "me": 0,
                "mu": 0, "re": 0, "sp": 2, "st": 0, "te": 0,
                "tx": 0, "v": 0, "vs": 47
            }
        },
        "source_platform": "desktop",
        "source_type": "vicut"
    }

    return [
        ("Artwork", json.dumps(artwork, separators=(',', ':'))),
        ("Hw", "1"),
        ("Bitrate", "28000000"),
        ("Maxrate", "0"),
        ("Te_Is_Reencode", "1"),
        ("Mp_4_Data_Incomplete", "false"),
    ]


def _export_with_ffmpeg_filters(bg_path, fg_path, caption_segments, audio_path, output_path, video_width, video_height, log_fn, effect_settings=None, mirror_video=False, target_duration=None, preferred_font=None, words_per_caption=2, text_color_rgba=None, stroke_color_rgba=None, stroke_width=None, font_size=None, blur_radius=None, dim_factor=None, bg_scale_extra=None, crop_top_ratio=None, crop_bottom_ratio=None, caption_y_offset=None, force_cpu=False):
    """
    Fast export using pure FFmpeg complex filters.
    2-3x faster than MoviePy's Python frame processing.
    
    Args:
        bg_path: Path to background image/video
        fg_path: Path to foreground video
        caption_segments: List of caption dictionaries
        audio_path: Path to audio file
        output_path: Path for output video
        video_width: Video width
        video_height: Video height
        log_fn: Logging function
        effect_settings: Dictionary with video effect settings (sharpness, saturation, contrast, etc.)
        mirror_video: Whether to apply horizontal flip to the foreground video
        target_duration: Target duration for the final video (for speed adjustment)
        preferred_font: Custom font path or name to use for captions
        words_per_caption: Number of words per caption group
        text_color_rgba: RGBA tuple for text color
        stroke_color_rgba: RGBA tuple for stroke/border color
        stroke_width: Width of text stroke in pixels
        font_size: Font size in pixels
        
    Returns:
        True if successful, False otherwise
    """
    try:
        log_fn("[EXPORT] ═══════════════════════════════════════════════════")
        log_fn("[EXPORT] Using fast FFmpeg filter-based export...")
        log_fn(f"[EXPORT] Building filter chain for {len(caption_segments)} caption segments...")
        if mirror_video:
            log_fn("[EXPORT] Mirror/flip: enabled")
        if target_duration:
            log_fn(f"[EXPORT] Target duration: {target_duration:.2f}s")
        
        # Resolve blur_radius and dim_factor defaults
        if blur_radius is None:
            blur_radius = globals().get('STATIC_BG_BLUR_RADIUS', 25)
        # Always use global DIM_FACTOR — ignore stale values from saved presets/jobs
        dim_factor = globals().get('DIM_FACTOR', 1.0)
        if bg_scale_extra is None:
            bg_scale_extra = globals().get('STATIC_BG_SCALE_EXTRA', 1.08)
        # Resolve crop ratios for background — same crop as foreground
        if crop_top_ratio is None:
            crop_top_ratio = globals().get('CROP_TOP_RATIO', 0.30)
        if crop_bottom_ratio is None:
            crop_bottom_ratio = globals().get('CROP_BOTTOM_RATIO', 0.35)
        
        # Use passed parameters, fall back to globals, then defaults
        try:
            # Font path - find actual font file path for FFmpeg
            font_path = None
            
            # Priority 1: Try the passed preferred_font
            if preferred_font:
                log_fn(f"[EXPORT] Font: Received preferred_font = '{preferred_font}'")
                
                # Check if it's already a valid file path
                if os.path.exists(preferred_font):
                    font_path = preferred_font
                    log_fn(f"[EXPORT] Font: Using passed font file: {font_path}")
                else:
                    # It's a font name, search for the font file
                    # Try find_ttf_in_tiktok first (searches custom font directory)
                    found = None
                    try:
                        found = find_ttf_in_tiktok(preferred_font, search_root=CUSTOM_FONT_ROOT)
                    except Exception as e:
                        log_fn(f"[EXPORT] Font: find_ttf_in_tiktok error: {e}")
                    
                    if not found:
                        # Fallback to general font search
                        found = find_font_file_recursive(preferred_font)
                    
                    if found:
                        font_path = found
                        log_fn(f"[EXPORT] Font: Found font by search: {font_path}")
                    else:
                        log_fn(f"[EXPORT] Font: Could not find file for '{preferred_font}'")
            
            # Priority 2: Fall back to LOADED_FONT_PATH if preferred_font didn't work
            if not font_path:
                loaded_path = globals().get('LOADED_FONT_PATH', None)
                if loaded_path:
                    loaded_path_str = str(loaded_path)
                    log_fn(f"[EXPORT] Font: Falling back to LOADED_FONT_PATH = '{loaded_path_str}'")
                    # Check if it's an actual file path
                    if os.path.exists(loaded_path_str):
                        font_path = loaded_path
                        log_fn(f"[EXPORT] Font: Using cached font file: {font_path}")
                    elif loaded_path_str.startswith("family:"):
                        # Extract font name and search for the actual file
                        font_name = loaded_path_str.replace("family:", "")
                        log_fn(f"[EXPORT] Font: Searching for font by family name: {font_name}")
                        # Try find_ttf_in_tiktok first
                        found = None
                        try:
                            found = find_ttf_in_tiktok(font_name, search_root=CUSTOM_FONT_ROOT)
                        except Exception as e:
                            log_fn(f"[EXPORT] Font: find_ttf_in_tiktok error: {e}")
                        if not found:
                            found = find_font_file_recursive(font_name)
                        if found:
                            font_path = found
                            log_fn(f"[EXPORT] Font: Found font file: {font_path}")
                        else:
                            log_fn(f"[EXPORT] Font: Could not find font file for '{font_name}', will use default")
                    else:
                        # Try to find font by name
                        found = find_font_file_recursive(loaded_path_str)
                        if found:
                            font_path = found
                            log_fn(f"[EXPORT] Font: Found font file by search: {font_path}")
            
            # Text color - use passed value or default
            if text_color_rgba is None:
                text_color_rgba = globals().get('CAPTION_TEXT_COLOR', (255, 255, 255, 255))
            
            # Stroke color - use passed value or default to black
            if stroke_color_rgba is None:
                stroke_color_rgba = globals().get('CAPTION_STROKE_COLOR', (0, 0, 0, 255))
            
            # Stroke width - use passed value or default
            if stroke_width is None:
                stroke_width = globals().get('CAPTION_STROKE_WIDTH', 3)
            
            # Font size - use passed value or default
            if font_size is None:
                font_size = globals().get('CAPTION_FONT_SIZE', 56)
            
            # Convert colors to hex for FFmpeg
            text_color_hex = _rgba_to_hex(text_color_rgba)
            stroke_color_hex = _rgba_to_hex(stroke_color_rgba)
            
            if font_path:
                log_fn(f"[EXPORT] Using custom font: {font_path}")
            else:
                log_fn(f"[EXPORT] Using default FFmpeg font (no custom font found)")
            log_fn(f"[EXPORT] Text color: {text_color_hex} (RGBA: {text_color_rgba})")
            log_fn(f"[EXPORT] Stroke color: {stroke_color_hex}")
            log_fn(f"[EXPORT] Stroke width: {stroke_width}px")
            log_fn(f"[EXPORT] Font size: {font_size}px")
            if words_per_caption:
                log_fn(f"[EXPORT] Words per caption: {words_per_caption}")
        except Exception as e:
            log_fn(f"[EXPORT] Warning: Could not get custom settings, using defaults: {e}")
            font_path = None
            text_color_hex = "0xFFFFFF"
            stroke_color_hex = "0x000000"
            stroke_width = 3
            font_size = 56
            words_per_caption = 2
        
        # For many captions (>MAX_DRAWTEXT_CAPTIONS), use subtitle file approach to avoid command line length limits
        # Otherwise use drawtext filters for better compatibility
        use_subtitle_file = len(caption_segments) > MAX_DRAWTEXT_CAPTIONS
        ass_subtitle_path = None
        
        # Get caption Y offset from parameter, falling back to global
        if caption_y_offset is None:
            caption_y_offset = globals().get('CAPTION_Y_OFFSET', 0)
        
        if use_subtitle_file:
            log_fn(f"[EXPORT] Using ASS subtitle file for {len(caption_segments)} captions (efficient for many captions)")
            # Generate ASS subtitle file in temp directory
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="tiktok_subtitles_")
            ass_subtitle_path = os.path.join(temp_dir, "captions.ass")
            
            # Get font family name for ASS subtitle
            # ASS files need the actual font family name, not the filename
            font_name = "Arial"  # Default
            
            # ALWAYS extract font family from the actual font file being used
            # This ensures we use the correct font even if LOADED_FONT_FAMILY is stale
            if font_path and os.path.exists(font_path):
                # Extract font family from font file with logging for debugging
                extracted_family = get_font_family_name(font_path, log_func=log_fn)
                if extracted_family:
                    font_name = extracted_family
                    log_fn(f"[EXPORT] Using font family: {font_name}")
                else:
                    # Fallback to filename without extension
                    font_name = os.path.splitext(os.path.basename(font_path))[0]
                    log_fn(f"[EXPORT] Using font filename as family: {font_name}")
            else:
                # Fallback: try to get from global LOADED_FONT_FAMILY
                loaded_family = globals().get('LOADED_FONT_FAMILY', None)
                if loaded_family:
                    font_name = loaded_family
                    log_fn(f"[EXPORT] Using cached font family for ASS: {font_name}")
            
            
            # Log the actual values being used for ASS generation
            log_fn(f"[ASS-GEN] Generating ASS with: font_name={font_name}, font_size={font_size}, y_offset={caption_y_offset}")
            
            _generate_ass_subtitle_file(
                caption_segments, 
                ass_subtitle_path,
                font_name=font_name,
                fontsize=font_size,
                text_color_rgba=text_color_rgba,
                stroke_width=stroke_width,
                stroke_color_rgba=stroke_color_rgba,
                video_width=video_width,
                video_height=video_height,
                y_offset=caption_y_offset,
                log_fn=log_fn
            )
            log_fn(f"[EXPORT] ASS subtitle file generated: {ass_subtitle_path}")
            caption_filters = None
        else:
            log_fn(f"[EXPORT] Using drawtext filters for {len(caption_segments)} captions")
            # Build caption filters with custom styling and words per caption
            caption_filters = _build_all_caption_filters(
                caption_segments, video_width, video_height,
                font_path=font_path,
                text_color=text_color_hex,
                stroke_color=stroke_color_hex,
                stroke_width=stroke_width,
                words_per_line=words_per_caption,
                fontsize=font_size,
                y_offset=caption_y_offset
            )
        
        # Build video effect filters (to match MoviePy effects)
        effect_filter_str = _build_ffmpeg_effect_filters(effect_settings, log_fn)
        
        # Use blur/dim/scale settings from function parameters (already set by caller).
        # Do NOT re-read globals here — that would override the values passed by
        # process_single_job, making per-job/preset dim_factor useless.
        
        # Build complete filter chain
        # Use the foreground video itself as the blurred background (instead of a static image)
        # Input [0:v] = foreground video (used for both background and foreground)
        # Input [1:a] = audio
        
        # Calculate speed adjustment factor if target_duration is set
        speed_factor = None
        if target_duration and target_duration > 0:
            # We need to probe the foreground video duration to calculate speed factor
            try:
                probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", fg_path]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
                if probe_result.returncode == 0 and probe_result.stdout.strip():
                    fg_duration = float(probe_result.stdout.strip())
                    if abs(fg_duration - target_duration) > 0.05:
                        speed_factor = fg_duration / target_duration
                        log_fn(f"[EXPORT] Video speed adjustment: {fg_duration:.2f}s → {target_duration:.2f}s (factor: {speed_factor:.4f})")
                else:
                    log_fn(f"[EXPORT] ffprobe failed (rc={probe_result.returncode}), skipping speed adjustment")
            except Exception as e:
                log_fn(f"[EXPORT] Could not probe video duration for speed adjustment: {e}")
        
        # Build the setpts expression for speed adjustment
        setpts_filter = ""
        # Track whether we're slowing down (need looping) or speeding up
        needs_stream_loop = False
        if speed_factor and speed_factor > 0:
            # setpts=PTS/factor speeds up (factor>1) or slows down (factor<1)
            setpts_filter = f"setpts=PTS/{speed_factor:.6f},"
            log_fn(f"[EXPORT] ✓ Speed adjustment filter: setpts=PTS/{speed_factor:.6f}")
            if speed_factor < 1.0:
                # Slowing down: video becomes longer, need looping to avoid running out of frames
                needs_stream_loop = True
                log_fn(f"[EXPORT] Video is being slowed down (factor={speed_factor:.4f}) - looping enabled")
        
        # Build background from video: scale to FILL canvas (preserving aspect ratio), apply blur and dim
        # FFmpeg boxblur approximates Gaussian blur; halving the radius gives similar visual results.
        # Minimum of 5 ensures visible blur even with small radius settings.
        box_blur_val = max(5, blur_radius // 2)
        # brightness adjustment: dim_factor 1.0 means no dimming (full brightness)
        # FFmpeg eq filter brightness is additive (-1.0 to 1.0), so: brightness = dim_factor - 1.0
        # Note: NV12 format conversion and downscale-blur-upscale can lose ~8% brightness,
        # so we add a small compensation boost (BG_BRIGHTNESS_BOOST)
        eq_brightness = dim_factor - 1.0 + BG_BRIGHTNESS_BOOST
        log_fn(f"[EXPORT]   Background: blur={blur_radius}, dim_factor={dim_factor:.2f}, eq_brightness={eq_brightness:+.2f}")
        
        # Ensure target dimensions are even (required for yuv420p pixel format)
        bg_target_w = int(video_width * bg_scale_extra) & ~1  # Ensure even
        bg_target_h = int(video_height * bg_scale_extra) & ~1
        
        # ── TWO-PASS STRATEGY ──
        # Pass 1: Pre-render blurred background video to a temp file.
        #         Heavy CPU filters (boxblur, scale, crop, eq) run here.
        # Pass 2: Final encode uses two simple inputs (bg + fg) with just
        #         overlay + captions. GPU NVENC can encode at full speed
        #         because the filter chain is lightweight.
        import tempfile
        bg_temp_dir = tempfile.mkdtemp(prefix="tiktok_bg_")
        bg_prerendered_path = os.path.join(bg_temp_dir, "bg_blurred.mp4")
        
        # Determine GPU availability and select codec
        if force_cpu:
            use_gpu = False
            gpu_filters = False
            nvenc_codec = PREFERRED_NVENC_CODEC
            log_fn("[EXPORT] ⚠️ CPU-only fallback mode (GPU disabled for this attempt)")
        else:
            use_gpu = USE_GPU_IF_AVAILABLE and ffmpeg_supports_nvenc(PREFERRED_NVENC_CODEC)
            nvenc_codec = PREFERRED_NVENC_CODEC
            if not use_gpu and USE_GPU_IF_AVAILABLE:
                # Try fallback NVENC codecs
                for alt_codec in ["hevc_nvenc", "h264_nvenc"]:
                    if alt_codec != PREFERRED_NVENC_CODEC and ffmpeg_supports_nvenc(alt_codec):
                        use_gpu = True
                        nvenc_codec = alt_codec
                        log_fn(f"[EXPORT] Primary codec {PREFERRED_NVENC_CODEC} unavailable, using fallback: {alt_codec}")
                        break
            
            gpu_filters = use_gpu and ffmpeg_gpu_filters_available()
        
        # Log GPU diagnostic info
        log_fn(f"[EXPORT] ═══ GPU DIAGNOSTIC ═══")
        log_fn(f"[EXPORT] GPU encoding: {'✓ ENABLED (' + nvenc_codec + ')' if use_gpu else '✗ DISABLED (using CPU libx264)'}")
        log_fn(f"[EXPORT] GPU filters: {'✓ ENABLED (scale_cuda/overlay_cuda)' if gpu_filters else '✗ DISABLED (using CPU filters)'}")
        if not use_gpu:
            log_fn(f"[EXPORT] {get_nvenc_diagnostic()}")
        log_fn(f"[EXPORT] ═══════════════════")
        
        # Build background video filter chain
        # Background uses the SAME crop as foreground, then zooms to fill the canvas.
        # This makes the blurred background match the foreground content area.
        #
        # Optimization: downscale-blur-upscale trick for much faster blur processing.
        # Instead of boxblur=12 on 1080x1920 (2M pixels, ~1.25B ops/frame),
        # downscale 4x → smaller blur → upscale back (~200x fewer operations).
        blur_down_w = max(video_width // 4, 2) & ~1   # 1080→270, ensure even
        blur_down_h = max(video_height // 4, 2) & ~1  # 1920→480, ensure even
        small_blur = max(2, box_blur_val // 3)         # 12→4, proportionally reduced
        # Encode background at blur_down resolution for ALL resolutions.
        # The blur already destroys detail above blur_down resolution, so encoding
        # at a higher resolution wastes GPU encoder cycles for no quality benefit.
        # Pass 2 upscales the background before compositing.
        bg_encode_w = blur_down_w
        bg_encode_h = blur_down_h
        bg_needs_upscale = (bg_encode_w != video_width or bg_encode_h != video_height)
        # Add eq brightness filter: handles both dimming and NV12 brightness compensation
        eq_part = f"eq=brightness={eq_brightness:.2f}," if abs(eq_brightness) > 0.001 else ""
        # Build user crop filter for background (same crop as foreground)
        keep_ratio = 1.0 - crop_top_ratio - crop_bottom_ratio
        bg_crop_part = ""
        if keep_ratio < 0.99:
            bg_crop_part = f"crop=iw:ih*{keep_ratio:.4f}:0:ih*{crop_top_ratio:.4f},"
            log_fn(f"[EXPORT]   Background crop: top={crop_top_ratio*100:.1f}%, bottom={crop_bottom_ratio*100:.1f}% (keeping {keep_ratio*100:.1f}%)")

        # ── Extra background crop when spot blur is active ──
        # When spot blur is enabled, the background (blurred version of same video)
        # still shows the content the spot blur is trying to hide. To prevent this,
        # crop the background to exclude the vertical range of the spot blur area,
        # keeping the larger portion (above or below), then zoom in to fill the frame.
        _blur_ov = effect_settings or {}
        if _blur_ov.get('blur_overlay_enabled', False):
            _by_pct = float(_blur_ov.get('blur_overlay_y', 10))
            _bh_pct = float(_blur_ov.get('blur_overlay_h', 15))
            _blur_top = _by_pct / 100.0           # top edge of blur area (fraction)
            _blur_bottom = (_by_pct + _bh_pct) / 100.0  # bottom edge (fraction)
            # Portion above vs below the blur area
            _above = _blur_top                     # fraction of video above blur
            _below = 1.0 - _blur_bottom            # fraction of video below blur
            if _above >= _below and _above > 0.15:
                # Keep the portion above the blur area
                bg_crop_part = f"crop=iw:ih*{_above:.4f}:0:0,"
                log_fn(f"[EXPORT]   Background spot-blur crop: keeping top {_above*100:.1f}% (above blur at y={_by_pct:.0f}%)")
            elif _below > 0.15:
                # Keep the portion below the blur area
                bg_crop_part = f"crop=iw:ih*{_below:.4f}:0:ih*{_blur_bottom:.4f},"
                log_fn(f"[EXPORT]   Background spot-blur crop: keeping bottom {_below*100:.1f}% (below blur at y+h={_blur_bottom*100:.0f}%)")
            else:
                # Blur area is too large to crop around – keep default crop
                log_fn(f"[EXPORT]   Background spot-blur crop: blur area too large ({_bh_pct:.0f}%), using default crop")

        if gpu_filters and USE_HARDWARE_DECODING and not needs_stream_loop:
            # GPU-accelerated decode + CPU boxblur: identical quality to CPU-only path.
            # GPU handles fast decode via -hwaccel cuda, then hwdownload transfers frames
            # to CPU where the same crop→scale→boxblur chain runs as the CPU path.
            # NVENC handles encoding at the end. This ensures the blur looks identical
            # to the CPU path while still benefiting from GPU decode/encode speed.
            bg_vf = (
                f"{setpts_filter}hwdownload,format=nv12,"
                f"{bg_crop_part}"
                f"scale={video_width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={video_width}:{video_height},"
                f"scale={blur_down_w}:{blur_down_h},"
                f"boxblur={small_blur}:{small_blur},"
                f"{eq_part}"
                f"scale={bg_encode_w}:{bg_encode_h},setsar=1:1"
            )
            log_fn(f"[EXPORT] Pass 1 using GPU decode + CPU boxblur={small_blur} (blur_down={blur_down_w}x{blur_down_h}, encode={bg_encode_w}x{bg_encode_h})")
        else:
            bg_vf = (
                f"{setpts_filter}{bg_crop_part}"
                f"scale={video_width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={video_width}:{video_height},"
                f"scale={blur_down_w}:{blur_down_h},"
                f"boxblur={small_blur}:{small_blur},"
                f"{eq_part}"
                f"scale={bg_encode_w}:{bg_encode_h},setsar=1:1"
            )
        
        # ── Spot blur on background ── DISABLED ──
        # The background is already fully blurred (boxblur), so applying an
        # additional spot-blur rectangle creates a visible double-blur artifact.
        # The foreground spot blur is sufficient to hide the target area.
        bg_spot_blur = ""

        bg_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        # GPU hardware decoding for background pre-render.
        # Use -hwaccel cuda for fast decode even though the blur filter chain runs on
        # CPU (hwdownload transfers frames). GPU decode is still faster than CPU decode,
        # and the background is a one-time pre-render so the transfer overhead is minimal.
        if use_gpu and USE_HARDWARE_DECODING and gpu_filters and not needs_stream_loop:
            bg_cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
        if needs_stream_loop:
            bg_cmd.extend(["-stream_loop", "-1"])
        # Add filter threading for CPU filter processing.
        # Use all available CPU cores for maximum throughput.
        total_cores = os.cpu_count() or 4
        if use_gpu:
            cpu_threads = max(2, total_cores // 2)
        else:
            cpu_threads = total_cores
        bg_cmd.extend(["-filter_threads", str(cpu_threads), "-filter_complex_threads", str(cpu_threads)])
        # When spot blur is enabled on the background, we need -filter_complex
        # because the split→crop→overlay graph requires named streams.
        if bg_spot_blur:
            bg_cmd.extend(["-i", bg_path, "-an", "-filter_complex", bg_vf + bg_spot_blur])
        else:
            bg_cmd.extend(["-i", bg_path, "-an", "-vf", bg_vf])
        
        # Use NVENC for bg pre-render if available, else CPU ultrafast
        # Background is blurred — use constqp with high QP for fastest encoding.
        # Higher QP/CRF for the intermediate is safe because the heavy blur
        # hides any quality difference, and the file is only consumed once.
        if use_gpu:
            bg_cmd.extend(["-c:v", nvenc_codec, "-preset", "p1", "-rc", "constqp", "-qp", "34", "-b:v", "0", "-multipass", "0"])
        else:
            bg_cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-threads", "0"])
        
        # Limit bg to same duration as output
        bg_duration_limit = None
        if target_duration and target_duration > 0:
            bg_duration_limit = target_duration
        else:
            try:
                dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
                dur_result = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=30)
                if dur_result.returncode == 0 and dur_result.stdout.strip():
                    bg_duration_limit = float(dur_result.stdout.strip())
            except Exception:
                pass
        if bg_duration_limit:
            bg_cmd.extend(["-t", f"{bg_duration_limit:.3f}"])
        
        bg_cmd.extend(["-pix_fmt", "yuv420p", bg_prerendered_path])
        
        blur_mode = "GPU decode + CPU boxblur" if (gpu_filters and not needs_stream_loop) else f"CPU boxblur={small_blur}"
        blur_detail = f"(downscale {video_width}→{blur_down_w}, boxblur={small_blur})"
        log_fn("[EXPORT] Pass 1/2: Pre-rendering blurred background video...")
        log_fn(f"[EXPORT]   Encoder: {'NVENC (' + nvenc_codec + ')' if use_gpu else 'CPU (libx264)'}, blur={blur_mode} {blur_detail}, encode={bg_encode_w}x{bg_encode_h}, dim={eq_brightness:.2f}")
        log_fn(f"[EXPORT]   Command: {' '.join(bg_cmd)}")
        bg_timeout = max(300, int((bg_duration_limit or 60) * 5))  # 5x video duration, min 5 min
        bg_result = subprocess.run(bg_cmd, capture_output=True, text=True, timeout=bg_timeout)
        if bg_result.returncode != 0:
            log_fn(f"[EXPORT] ⚠️ Background pre-render failed: {bg_result.stderr}")
            try:
                import shutil
                shutil.rmtree(bg_temp_dir)
            except Exception:
                pass
            return False
        log_fn("[EXPORT] ✓ Background pre-rendered successfully")
        
        # ── Pass 2: Final encode ──
        # Two inputs: [0:v]=bg (pre-rendered blurred), [1:v]=fg (original), [2:a]=audio
        # Lightweight filter: just overlay + captions. GPU can run at full speed.
        
        # Decide if we can use full GPU pipeline for Pass 2.
        # overlay_cuda requires both inputs in CUDA memory and no CPU-only filters after it.
        # Subtitles (ass/drawtext) are CPU-only, so if we have captions we use a HYBRID
        # pipeline: GPU overlay_cuda for the expensive bg+fg composite, then hwdownload
        # for CPU caption burn-in. This offloads the overlay to GPU and frees CPU cores.
        has_captions = (use_subtitle_file and ass_subtitle_path) or caption_filters
        use_gpu_overlay = gpu_filters and not has_captions and not mirror_video
        # Hybrid GPU: use overlay_cuda even with captions, then hwdownload for CPU captions
        use_gpu_hybrid = gpu_filters and has_captions and not mirror_video
        # When hybrid GPU is active and hardware decoding is enabled, the foreground
        # [1:v] will be decoded on GPU (CUDA frames) via per-input -hwaccel cuda.
        # In that case, skip hwupload_cuda for fg since it's already in CUDA memory.
        fg_hwaccel_decode = use_gpu and USE_HARDWARE_DECODING and use_gpu_hybrid
        
        if use_gpu_overlay:
            # Full GPU pipeline: hwupload both inputs → overlay_cuda
            # No captions/effects = pure GPU path
            fg_prep = f"[1:v]{setpts_filter}hwupload_cuda[fg_ready]"
            # If background was encoded at reduced resolution, upscale before overlay
            bg_scale_prefix = f"[0:v]scale={video_width}:{video_height}," if bg_needs_upscale else "[0:v]"
            filter_parts = [
                fg_prep,
                f"{bg_scale_prefix}hwupload_cuda[bg_cuda]",
                "[bg_cuda][fg_ready]overlay_cuda=x=(W-w)/2:y=(H-h)/2,hwdownload,format=nv12"
            ]
            if effect_filter_str:
                # Effects are CPU-only; data is already in CPU format (nv12) after hwdownload
                filter_parts[-1] += f",{effect_filter_str}"
                log_fn(f"[EXPORT] ✓ GPU overlay + CPU effects")
            if bg_needs_upscale:
                log_fn(f"[EXPORT] ✓ Background upscale {bg_encode_w}x{bg_encode_h} → {video_width}x{video_height} in Pass 2")
            log_fn("[EXPORT] ✓ Using GPU-accelerated overlay_cuda (full GPU pipeline)")
        elif use_gpu_hybrid:
            # Hybrid GPU pipeline: GPU overlay_cuda for bg+fg, then hwdownload for
            # CPU-only caption/subtitle burn-in. This offloads the most expensive filter
            # (overlay) to GPU, freeing CPU cores for caption rendering.
            # When fg_hwaccel_decode is true, [1:v] is already CUDA but hwupload_cuda
            # handles CUDA→CUDA as a no-op, so we keep it for compatibility.
            fg_prep = f"[1:v]{setpts_filter}hwupload_cuda[fg_ready]"
            bg_scale_prefix = f"[0:v]scale={video_width}:{video_height}," if bg_needs_upscale else "[0:v]"
            filter_parts = [
                fg_prep,
                f"{bg_scale_prefix}hwupload_cuda[bg_cuda]",
                "[bg_cuda][fg_ready]overlay_cuda=x=(W-w)/2:y=(H-h)/2,hwdownload,format=nv12"
            ]
            if bg_needs_upscale:
                log_fn(f"[EXPORT] ✓ Background upscale {bg_encode_w}x{bg_encode_h} → {video_width}x{video_height} in Pass 2")
            if fg_hwaccel_decode:
                log_fn("[EXPORT] ✓ Hybrid GPU pipeline: GPU decode fg → overlay_cuda → hwdownload → CPU captions")
            else:
                log_fn("[EXPORT] ✓ Hybrid GPU pipeline: overlay_cuda → hwdownload → CPU captions")
        else:
            # CPU overlay pipeline (when mirror is needed or no GPU filters)
            if mirror_video:
                fg_prep = f"[1:v]{setpts_filter}hflip[fg_ready]"
                log_fn("[EXPORT] ✓ Mirror/flip filter added")
            else:
                fg_prep = f"[1:v]{setpts_filter}copy[fg_ready]"
            
            # If background was encoded at reduced resolution, upscale before overlay
            if bg_needs_upscale:
                bg_prep = f"[0:v]scale={video_width}:{video_height}[bg_up]"
                filter_parts = [fg_prep, bg_prep, "[bg_up][fg_ready]overlay=x=(W-w)/2:y=(H-h)/2"]
                log_fn(f"[EXPORT] ✓ Background upscale {bg_encode_w}x{bg_encode_h} → {video_width}x{video_height} in Pass 2")
            else:
                # Combine: bg + fg overlay
                filter_parts = [fg_prep, "[0:v][fg_ready]overlay=x=(W-w)/2:y=(H-h)/2"]
        
        # --- Blur overlay (cover-up region) ---
        # If blur overlay is enabled in effect_settings, split the composited stream,
        # crop+blur a region, and overlay it back to cover that area.
        # The user sets blur coordinates in the preview which shows the ORIGINAL video
        # frame. In the export, the foreground is cropped+scaled and centered on the
        # canvas, so we must transform the coordinates accordingly.
        blur_ov = effect_settings or {}
        if blur_ov.get('blur_overlay_enabled', False):
            # Convert to yuv420p to prevent green chroma artifacts from boxblur
            filter_parts[-1] += ",format=yuv420p"
            
            # Probe foreground video dimensions to calculate its offset in the
            # composited frame. The foreground is centered via overlay=x=(W-w)/2:y=(H-h)/2.
            fg_w, fg_h = video_width, video_height  # fallback
            try:
                dim_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream=width,height",
                           "-of", "csv=p=0:s=x", fg_path]
                dim_result = subprocess.run(dim_cmd, capture_output=True, text=True, timeout=30)
                if dim_result.returncode == 0 and 'x' in dim_result.stdout.strip():
                    parts = dim_result.stdout.strip().split('x')
                    fg_w = int(parts[0])
                    fg_h = int(parts[1])
                    log_fn(f"[EXPORT] Blur overlay: probed foreground {fg_w}x{fg_h}")
            except Exception as e:
                log_fn(f"[EXPORT] Blur overlay: could not probe fg dimensions ({e}), using canvas size")
            
            # Foreground offset in the composited frame (centered overlay)
            fg_x_off = max(0, (video_width - fg_w) // 2)
            fg_y_off = max(0, (video_height - fg_h) // 2)
            
            # keep_ratio is the fraction of the original video height that remains
            # after top/bottom crop. Used to map original-video Y% → foreground Y%.
            keep_ratio = max(0.01, 1.0 - crop_top_ratio - crop_bottom_ratio)
            
            bx_pct = float(blur_ov.get('blur_overlay_x', 10))
            by_pct = float(blur_ov.get('blur_overlay_y', 10))
            bw_pct = float(blur_ov.get('blur_overlay_w', 20))
            bh_pct = float(blur_ov.get('blur_overlay_h', 15))
            b_intensity = int(blur_ov.get('blur_overlay_intensity', 20))
            
            # Transform coordinates from original-video-relative to composited-frame.
            # X: no crop in X direction, so simply offset by foreground position.
            # Y: the top crop_top_ratio of the original is removed, and the visible
            #    area (keep_ratio) maps to fg_h. So original Y% must be adjusted:
            #    canvas_y = fg_y_off + (by_pct/100 - crop_top_ratio) / keep_ratio * fg_h
            bx_px = max(0, fg_x_off + int(fg_w * bx_pct / 100.0)) & ~1
            by_px = max(0, fg_y_off + int(fg_h * (by_pct / 100.0 - crop_top_ratio) / keep_ratio)) & ~1
            bw_px = max(2, int(fg_w * bw_pct / 100.0)) & ~1
            bh_px = max(2, int(fg_h * bh_pct / (100.0 * keep_ratio))) & ~1
            
            # Clamp to fit within frame
            if bx_px + bw_px > video_width:
                bw_px = (video_width - bx_px) & ~1
            if by_px + bh_px > video_height:
                bh_px = (video_height - by_px) & ~1
            b_blur = max(2, b_intensity)
            # Clamp boxblur radius to respect YUV420p chroma plane limits.
            # For YUV420p, chroma is half luma in both dimensions. FFmpeg requires
            # boxblur radius <= min(chroma_w, chroma_h) / 2 = min(crop_w, crop_h) / 4.
            safe_blur_limit = max(2, min(bw_px, bh_px) // 4)
            if b_blur > safe_blur_limit:
                log_fn(f"[EXPORT] ⚠️ Blur overlay radius {b_blur} clamped to {safe_blur_limit} (crop {bw_px}x{bh_px} limit)")
                b_blur = safe_blur_limit
            # Label the current composited stream and split it
            filter_parts[-1] += "[_bo_pre]"
            filter_parts.append(f"[_bo_pre]split[_bo_main][_bo_copy]")
            filter_parts.append(f"[_bo_copy]crop={bw_px}:{bh_px}:{bx_px}:{by_px},boxblur={b_blur}:2[_bo_blurred]")
            filter_parts.append(f"[_bo_main][_bo_blurred]overlay={bx_px}:{by_px}")
            log_fn(f"[EXPORT] ✓ Blur overlay: pos=({bx_px},{by_px}) size={bw_px}x{bh_px} blur={b_blur} fg={fg_w}x{fg_h} offset=({fg_x_off},{fg_y_off})")

        # Apply captions/subtitles and effects for CPU and hybrid GPU pipelines.
        # In both cases, frames are in CPU memory at this point:
        # - CPU path: frames were never on GPU
        # - Hybrid GPU path: frames are in CPU after hwdownload,format=nv12
        if not use_gpu_overlay:
            if use_subtitle_file and ass_subtitle_path:
                # Use ass subtitle filter (burns subtitles into video)
                # Escape path for Windows
                escaped_ass_path = ass_subtitle_path.replace('\\', '/').replace(':', '\\:')
                
                # If we have a custom font, specify the fonts directory so FFmpeg can find it
                # The fontsdir option tells FFmpeg where to look for fonts used in the ASS file
                if font_path and os.path.exists(font_path):
                    font_dir = os.path.dirname(font_path)
                    escaped_font_dir = font_dir.replace('\\', '/').replace(':', '\\:')
                    ass_filter = f"ass='{escaped_ass_path}':fontsdir='{escaped_font_dir}'"
                    log_fn(f"[EXPORT] ASS filter using font directory: {font_dir}")
                else:
                    ass_filter = f"ass='{escaped_ass_path}'"
                
                # Append ASS subtitle filter after overlay
                filter_parts[-1] += f",{ass_filter}"
            else:
                if caption_filters:
                    filter_parts[-1] += "," + caption_filters
            
            # Add video effects to filter chain (same as MoviePy applies)
            if effect_filter_str:
                filter_parts[-1] += "," + effect_filter_str
                log_fn(f"[EXPORT] ✓ Video effects added to FFmpeg filter chain")
        
        # Force exact output resolution and reset SAR to avoid narrow/stretched video.
        # Some source videos have non-1:1 SAR that carries through the filter chain,
        # causing the output to appear narrow even with correct pixel dimensions.
        filter_parts[-1] += f",scale={video_width}:{video_height},setsar=1:1"
        
        # Label the filter output for mapping
        filter_parts[-1] += "[vout]"
        
        # Join all filter parts with semicolons
        filter_chain = ";".join(filter_parts)
        
        # Log input information for debugging
        log_fn(f"[EXPORT] Pass 2/2: Final encode with lightweight filter chain")
        log_fn(f"[EXPORT] Background (pre-rendered): {bg_prerendered_path} ({bg_encode_w}x{bg_encode_h}{' → upscale to ' + str(video_width) + 'x' + str(video_height) if bg_needs_upscale else ''})")
        log_fn(f"[EXPORT] Foreground: {fg_path}")
        log_fn(f"[EXPORT] Audio: {audio_path}")
        log_fn(f"[EXPORT] Output: {output_path} ({video_width}x{video_height})")
        log_fn(f"[EXPORT] Filter chain: {filter_chain[:300]}...")
        
        # Determine explicit output duration to prevent FFmpeg from hanging.
        # -shortest does NOT work reliably with -filter_complex; we must use -t.
        output_duration = bg_duration_limit  # Already probed during bg pre-render
        if not output_duration:
            if target_duration and target_duration > 0:
                output_duration = target_duration
            else:
                try:
                    dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
                    dur_result = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=30)
                    if dur_result.returncode == 0 and dur_result.stdout.strip():
                        output_duration = float(dur_result.stdout.strip())
                        log_fn(f"[EXPORT] Audio duration probed: {output_duration:.2f}s")
                except Exception as e:
                    log_fn(f"[EXPORT] Could not probe audio duration: {e}")
        
        # Build FFmpeg command - two video inputs + audio
        # [0:v] = pre-rendered blurred background (lightweight, already processed)
        # [1:v] = foreground video (just needs overlay + captions)
        # [2:a] = audio
        cmd = ["ffmpeg", "-y"]
        
        # GPU hardware decoding: offloads video decoding from CPU to GPU.
        # Full GPU: hwaccel for all inputs, decoded frames stay in CUDA memory.
        # Hybrid GPU: per-input hwaccel for foreground only (input [1:v]).
        #   Background [0:v] is tiny (e.g. 540×960) and decoded on CPU.
        #   Foreground [1:v] is the main video and benefits from GPU decoding.
        #   This reduces CPU load by ~20-30% (video decoding offloaded to GPU).
        # fg_hwaccel_decode was already set above (before filter chain construction).
        if use_gpu and USE_HARDWARE_DECODING and use_gpu_overlay:
            cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
            log_fn("[EXPORT] ✓ GPU hardware decoding + GPU overlay (full GPU pipeline)")
        elif fg_hwaccel_decode:
            log_fn("[EXPORT] ✓ Hybrid GPU pipeline: GPU decode fg → GPU overlay_cuda → CPU captions → GPU NVENC")
        elif use_gpu and use_gpu_hybrid:
            log_fn("[EXPORT] ✓ Hybrid GPU pipeline: CPU decode → GPU overlay_cuda → CPU captions → GPU NVENC")
        elif use_gpu:
            log_fn("[EXPORT] ✓ GPU NVENC encoding (CPU decode → CPU filters → GPU encode)")
        
        # Input [0:v]: background (CPU decoded, small resolution)
        cmd.extend(["-i", bg_prerendered_path])
        # Input [1:v]: foreground video (GPU decoded in hybrid mode for less CPU load)
        if fg_hwaccel_decode:
            cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                        "-i", fg_path])
        else:
            cmd.extend(["-i", fg_path])
        # Input [2:a]: audio
        cmd.extend(["-i", audio_path])
        
        # Use multiple threads for CPU filter processing (captions, effects, etc.).
        # Use all available CPU cores for maximum throughput.
        if not use_gpu_overlay:
            total_cores = os.cpu_count() or 4
            if use_gpu:
                # GPU active: use half the cores for filters, GPU handles encoding.
                cpu_threads = max(2, total_cores // 2)
            else:
                # CPU-only: use all available cores for maximum throughput.
                cpu_threads = total_cores
            cmd.extend(["-filter_threads", str(cpu_threads), "-filter_complex_threads", str(cpu_threads)])
            log_fn(f"[EXPORT] ✓ CPU filter threading: {cpu_threads} threads")
        
        cmd.extend([
            "-filter_complex", filter_chain,
            "-map", "[vout]",
            "-map", "2:a",                # Audio from input 2
        ])
        
        # Explicit duration limit prevents hanging (critical with -filter_complex)
        if output_duration and output_duration > 0:
            cmd.extend(["-t", f"{output_duration:.3f}"])
            log_fn(f"[EXPORT] Output duration limited to {output_duration:.2f}s")
        
        cmd.extend([
            "-shortest",
            "-c:a", "aac",
            "-b:a", "192k",
        ])
        
        # Add video encoding parameters - GPU NVENC or CPU libx264
        if use_gpu:
            log_fn(f"[EXPORT] ✓ GPU NVENC encoding: {nvenc_codec}, preset={NVENC_PRESET_SPEED}")
            cmd.extend([
                "-c:v", nvenc_codec,
                "-rc", "constqp",
                "-qp", "22",
                "-b:v", "0",
                "-preset", NVENC_PRESET_SPEED,
                "-multipass", "0",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high"
            ])
        else:
            log_fn("[EXPORT] Using CPU encoding (libx264)...")
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-threads", "0"
            ])
        
        # Inject CapCut / vicut metadata so the file looks like a genuine
        # CapCut desktop export when inspected with ExifTool or similar tools.
        capcut_meta = _generate_capcut_metadata()
        for meta_key, meta_val in capcut_meta:
            cmd.extend(["-metadata", f"{meta_key}={meta_val}"])
        log_fn(f"[EXPORT] ✓ CapCut metadata injected ({len(capcut_meta)} tags)")

        cmd.extend([
            "-movflags", "+faststart+use_metadata_tags",
            "-brand", "isom",
            output_path
        ])
        
        log_fn("[EXPORT] Executing FFmpeg final encode...")
        log_fn(f"[EXPORT] Full command: {' '.join(cmd)}")
        
        # Run FFmpeg
        final_timeout = max(600, int((output_duration or 60) * 5))  # 5x duration, min 10 min
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=final_timeout
        )
        
        # Clean up background temp file
        try:
            import shutil
            shutil.rmtree(bg_temp_dir)
        except Exception:
            pass
        
        if result.returncode == 0:
            log_fn("[EXPORT] ✅ Fast FFmpeg export completed successfully!")
            return True
        else:
            log_fn(f"[EXPORT] ❌ FFmpeg failed with return code {result.returncode}")
            log_fn(f"[EXPORT] FFmpeg stderr output:")
            log_fn(f"{result.stderr[-1500:]}")
            return False
            
    except Exception as e:
        log_fn(f"[EXPORT] ❌ FFmpeg export exception: {e}")
        import traceback
        log_fn(f"[EXPORT] Traceback: {traceback.format_exc()}")
        return False


# ----------------- Video Effects (CapCut-style) -----------------
def apply_video_effects(frame, effect_settings):
    """
    Apply CapCut-style effects to a video frame.
    
    Args:
        frame: numpy array or PIL Image representing the frame
        effect_settings: dict containing effect parameters
        
    Returns:
        Modified frame as numpy array or PIL Image (same type as input)
    """
    import numpy as np
    
    # Convert to PIL if needed
    is_numpy = isinstance(frame, np.ndarray)
    if is_numpy:
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
    else:
        img = frame
    
    # Apply Sharpness/Resilience effect
    if effect_settings.get('effect_sharpness', False):
        intensity = effect_settings.get('effect_sharpness_intensity', 1.5)
        sharpness = ImageEnhance.Sharpness(img)
        img = sharpness.enhance(intensity)
    
    # Apply Saturation/Vibrance effect
    if effect_settings.get('effect_saturation', False):
        intensity = effect_settings.get('effect_saturation_intensity', 1.3)
        color = ImageEnhance.Color(img)
        img = color.enhance(intensity)
    
    # Apply Contrast/HDR effect
    if effect_settings.get('effect_contrast', False):
        intensity = effect_settings.get('effect_contrast_intensity', 1.2)
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(intensity)
    
    # Apply Brightness effect
    if effect_settings.get('effect_brightness', False):
        intensity = effect_settings.get('effect_brightness_intensity', 1.15)
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(intensity)
    
    # Apply Vintage/Film Grain effect
    if effect_settings.get('effect_vintage', False):
        grain_intensity = effect_settings.get('effect_vintage_intensity', 0.3)
        # Add film grain noise
        width, height = img.size
        noise = np.random.normal(0, grain_intensity * 25, (height, width, 3))
        img_array = np.array(img).astype('float')
        img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
        
        # Apply slight sepia tone for vintage look
        sepia_r = np.array([[0.393 + 0.607 * (1 - grain_intensity), 0.769 - 0.769 * (1 - grain_intensity), 0.189 - 0.189 * (1 - grain_intensity)]])
        sepia_g = np.array([[0.349 - 0.349 * (1 - grain_intensity), 0.686 + 0.314 * (1 - grain_intensity), 0.168 - 0.168 * (1 - grain_intensity)]])
        sepia_b = np.array([[0.272 - 0.272 * (1 - grain_intensity), 0.534 - 0.534 * (1 - grain_intensity), 0.131 + 0.869 * (1 - grain_intensity)]])
        
        img_array = np.array(img)
        r = np.dot(img_array, sepia_r.T)
        g = np.dot(img_array, sepia_g.T)
        b = np.dot(img_array, sepia_b.T)
        sepia_img = np.dstack([r, g, b])
        img = Image.fromarray(np.clip(sepia_img, 0, 255).astype('uint8'), 'RGB')
    
    # Convert back to numpy if needed
    if is_numpy:
        return np.array(img)
    return img


def apply_blur_overlay_to_preview(img, effect_settings):
    """
    Apply blur overlay (cover-up region) to a PIL Image for real-time preview.
    
    Crops the specified region, applies GaussianBlur, and pastes it back.
    This gives a real-time preview of the blur overlay on the mini canvas.
    
    Args:
        img: PIL Image
        effect_settings: dict with blur_overlay_enabled, blur_overlay_x/y/w/h (percentages),
                         and blur_overlay_intensity
    Returns:
        Modified PIL Image with blur overlay applied (or original if not enabled)
    """
    if not effect_settings or not effect_settings.get('blur_overlay_enabled', False):
        return img
    
    try:
        w, h = img.size
        x_pct = float(effect_settings.get('blur_overlay_x', 10)) / 100.0
        y_pct = float(effect_settings.get('blur_overlay_y', 10)) / 100.0
        w_pct = float(effect_settings.get('blur_overlay_w', 20)) / 100.0
        h_pct = float(effect_settings.get('blur_overlay_h', 15)) / 100.0
        intensity = int(effect_settings.get('blur_overlay_intensity', 20))
        
        # Convert percentages to pixel coordinates
        x1 = max(0, int(round(w * x_pct)))
        y1 = max(0, int(round(h * y_pct)))
        x2 = min(w, int(round(w * (x_pct + w_pct))))
        y2 = min(h, int(round(h * (y_pct + h_pct))))
        
        # Ensure valid region
        if x2 <= x1 or y2 <= y1:
            return img
        
        # Crop the region, blur it, paste it back
        img = img.copy()
        region = img.crop((x1, y1, x2, y2))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=intensity))
        img.paste(blurred, (x1, y1))
        return img
    except Exception:
        return img


def compose_final_video_with_static_blurred_bg(video_clip, audio_clip, caption_segments, output_path, preferred_font=None, log=None, blur_radius=STATIC_BG_BLUR_RADIUS, bg_scale_extra=BG_SCALE_EXTRA, dim_factor=DIM_FACTOR, words_per_caption=2, effect_settings=None, pre_rendered_fg_path=None, mirror_video=False, target_duration=None, original_video_path=None, crop_top_ratio=None, crop_bottom_ratio=None, caption_text_color=None, caption_stroke_color=None, caption_stroke_width=None, caption_font_size=None, caption_y_offset=None):
    """
    Compose final video with blurred background and caption overlays.
    
    FIXED ISSUES:
    - Removed duplicated and nested caption generation loops
    - Fixed alpha channel handling for caption masks
    - Added logging for caption composition errors
    - Verified caption layer creation and positioning
    - Ensured captions are properly composited onto final video
    
    Args:
        pre_rendered_fg_path: Optional path to pre-rendered foreground video for fast FFmpeg export
        mirror_video: Whether to apply horizontal flip to the video
        target_duration: Target duration for the final video (for speed adjustment in FFmpeg)
    """
    try:
        log("Composing final video (with monitored export).")
    except Exception:
        print("Composing final video (no log callable).")
    
    try:
        log(f"[compose] Starting composition with {len(caption_segments)} caption segments")
        if pre_rendered_fg_path:
            log(f"[compose] Pre-rendered foreground available: {pre_rendered_fg_path}")
        if mirror_video:
            log(f"[compose] Mirror video: enabled")
        if target_duration:
            log(f"[compose] Target duration: {target_duration:.2f}s")
    except Exception:
        pass
    
    # Background creation is deferred until needed (MoviePy fallback only).
    # The primary FFmpeg export path creates the blurred background via native boxblur/eq filters,
    # so this expensive per-frame MoviePy processing is skipped when FFmpeg export succeeds.
    bg_static = None  # Will be created lazily if MoviePy fallback is needed

    # foreground: zoom to fill width (left/right borders) with user-controllable zoom
    # Calculate scale to fill the canvas width while maintaining aspect ratio
    try:
        # Get user's zoom setting
        zoom_factor = globals().get('VIDEO_ZOOM_SCALE', 1.0)
        
        # Scale to fill width (left/right borders)
        width_scale = WIDTH / video_clip.w
        
        # Ensure minimum height coverage (same logic as processing_job)
        base_scale_factor = 1.03
        estimated_height = video_clip.h * width_scale * base_scale_factor
        min_fg_height = HEIGHT * MIN_FG_HEIGHT_RATIO
        if estimated_height < min_fg_height and video_clip.h > 0:
            height_scale = min_fg_height / video_clip.h
            fg_base_scale = max(width_scale, height_scale)
        else:
            fg_base_scale = width_scale
        
        # Apply base scale factor and user's zoom factor
        fg_scale = fg_base_scale * base_scale_factor * zoom_factor
    except Exception:
        fg_scale = 1.0
    
    # Remove original audio from video clip to prevent duplicate audio
    # Only the mixed_audio (TTS + music) should be used
    fg = video_clip.without_audio().resize(fg_scale).set_position(("center", "center")).set_duration(video_clip.duration)
    
    # Apply video effects (CapCut-style) if enabled
    if effect_settings and any([
        effect_settings.get('effect_sharpness', False),
        effect_settings.get('effect_saturation', False),
        effect_settings.get('effect_contrast', False),
        effect_settings.get('effect_brightness', False),
        effect_settings.get('effect_vintage', False)
    ]):
        try:
            log(f"[EFFECTS] Applying CapCut-style effects to video...")
            active_effects = []
            if effect_settings.get('effect_sharpness'): active_effects.append('Resilience')
            if effect_settings.get('effect_saturation'): active_effects.append('Vibrance')
            if effect_settings.get('effect_contrast'): active_effects.append('HDR')
            if effect_settings.get('effect_brightness'): active_effects.append('Brightness')
            if effect_settings.get('effect_vintage'): active_effects.append('Vintage')
            log(f"[EFFECTS] Active effects: {', '.join(active_effects)}")
            
            # Apply effects to each frame
            fg = fg.fl_image(lambda frame: apply_video_effects(frame, effect_settings))
            log(f"[EFFECTS] ✓ Effects applied successfully")
        except Exception as e:
            log(f"[EFFECTS] Warning: Could not apply effects: {e}")
    
    try:
        log(f"[compose] Foreground scaled: {fg_scale:.3f}x (fills canvas completely)")
    except Exception:
        pass

    # Collect caption timing data (lightweight - fast, no PIL rendering)
    # MoviePy caption clips are created lazily only if FFmpeg export fails
    caption_clips = []
    caption_data_for_ffmpeg = []  # Collect word-group caption data for FFmpeg export
    MIN_GROUP_DURATION = 0.25
    
    # Use the words_per_caption parameter (from UI control)
    tpl = words_per_caption
    
    try:
        log(f"[COMPOSE] Using {tpl} word(s) per caption (CapCut-style)")
    except Exception:
        pass
    
    for seg_idx, segment in enumerate(caption_segments):
        try:
            start_t = float(segment.get("start", segment.get("start_time", 0)))
            end_t = float(segment.get("end", segment.get("end_time", start_t + 3)))
            seg_dur = max(0.05, end_t - start_t)
            text = normalize_text(segment.get("text", "").strip())
            
            if not text:
                continue
            
            # Use word-level timestamps if available (CapCut-style auto-captions)
            word_data = segment.get("words", [])
            
            if word_data:
                groups_with_timing = []
                for word_idx in range(0, len(word_data), tpl):
                    word_group = word_data[word_idx:word_idx + tpl]
                    grp_text = " ".join([w.get("word", "").strip() for w in word_group])
                    grp_start = word_group[0].get("start", start_t)
                    grp_end = word_group[-1].get("end", end_t)
                    groups_with_timing.append({
                        "text": grp_text,
                        "start": grp_start,
                        "end": grp_end
                    })
            else:
                words = text.split()
                groups_with_timing = []
                raw_group_dur = seg_dur / max(1, len(words) // tpl + (1 if len(words) % tpl else 0))
                for i in range(0, len(words), tpl):
                    grp_text = " ".join(words[i:i+tpl])
                    g_start = start_t + (i // tpl) * raw_group_dur
                    g_end = min(end_t, g_start + raw_group_dur)
                    groups_with_timing.append({
                        "text": grp_text,
                        "start": g_start,
                        "end": g_end
                    })
            
            for group_data in groups_with_timing:
                grp_text = group_data["text"]
                # Strip punctuation from captions
                grp_text = grp_text.replace(",", "").replace(";", "").replace(".", "").replace("!", "").replace("?", "").replace("…", "").replace(":", "")
                # Apply caption case transformation
                _ccase = globals().get('CAPTION_CASE', 'none')
                if _ccase == 'upper':
                    grp_text = grp_text.upper()
                elif _ccase == 'lower':
                    grp_text = grp_text.lower()
                g_start = group_data["start"]
                g_dur = max(MIN_GROUP_DURATION, group_data["end"] - group_data["start"])
                if g_start >= end_t:
                    g_start = max(start_t, end_t - g_dur)
                
                caption_data_for_ffmpeg.append({
                    'text': grp_text,
                    'start': g_start,
                    'end': g_start + g_dur
                })
                    
        except Exception:
            continue
    
    # Prevent overlapping captions: clamp each caption's end before next caption's start.
    # Use a small gap (10ms) to ensure no frame shows both captions simultaneously,
    # even with ASS centisecond truncation or drawtext inclusive-end timing.
    for i in range(len(caption_data_for_ffmpeg) - 1):
        next_start = caption_data_for_ffmpeg[i + 1]['start']
        if caption_data_for_ffmpeg[i]['end'] > next_start - 0.01:
            caption_data_for_ffmpeg[i]['end'] = max(caption_data_for_ffmpeg[i]['start'], next_start - 0.01)
    
    try:
        log(f"[COMPOSE] ═══════════════════════════════════════════════")
        log(f"[COMPOSE] Total caption segments prepared: {len(caption_data_for_ffmpeg)}")
        log(f"[COMPOSE] ═══════════════════════════════════════════════")
    except Exception:
        pass
    
    # FFmpeg-only export pipeline (faster than MoviePy)
    ffmpeg_export_successful = False
    
    # Determine which caption data to use for FFmpeg
    captions_for_ffmpeg = caption_data_for_ffmpeg if caption_data_for_ffmpeg else caption_segments
    
    # Note: FFmpeg export handles many captions (>MAX_DRAWTEXT_CAPTIONS) via ASS subtitle files
    # which is efficient and avoids command line length limits. No need to disable FFmpeg.
    if len(captions_for_ffmpeg) > MAX_DRAWTEXT_CAPTIONS:
        log(f"[EXPORT] Many captions detected ({len(captions_for_ffmpeg)}) - will use ASS subtitle file for efficient FFmpeg rendering")
    
    MAX_FFMPEG_RETRIES = 2
    last_ffmpeg_error = None
    for ffmpeg_attempt in range(MAX_FFMPEG_RETRIES):
        if globals().get('STOP_REQUESTED', False):
            log("[STOP] ⛔ Processing stopped by user.")
            return False
        try:
            import tempfile
            # On second attempt, force CPU-only mode as fallback
            use_cpu_fallback = (ffmpeg_attempt > 0)
            if ffmpeg_attempt == 0:
                log("[EXPORT] Starting FFmpeg export...")
            else:
                log(f"[EXPORT] Retrying FFmpeg export (attempt {ffmpeg_attempt + 1}/{MAX_FFMPEG_RETRIES}) with CPU-only fallback...")
            if not caption_segments:
                log("[EXPORT] Note: No captions to render (video will have no text overlay)")
            
            temp_dir = tempfile.mkdtemp(prefix="tiktok_ffmpeg_export_")
            
            # Save audio to temp file (WAV = instant write, no MP3 encoding overhead)
            audio_temp_path = os.path.join(temp_dir, "audio.wav")
            audio_clip.write_audiofile(audio_temp_path, fps=44100, codec='pcm_s16le', verbose=False, logger=None)
            log(f"[EXPORT] Audio saved to: {audio_temp_path}")
            
            # Get foreground video path - use pre-rendered path if available
            fg_video_path = None
            if pre_rendered_fg_path and os.path.exists(pre_rendered_fg_path):
                fg_video_path = pre_rendered_fg_path
                log(f"[EXPORT] Using pre-rendered foreground: {fg_video_path}")
            elif hasattr(fg, 'filename') and fg.filename and os.path.exists(fg.filename):
                fg_video_path = fg.filename
                log(f"[EXPORT] Using foreground from clip: {fg_video_path}")
            else:
                # Need to save foreground video first
                fg_video_path = os.path.join(temp_dir, "foreground.mp4")
                log(f"[EXPORT] No pre-rendered foreground - saving to: {fg_video_path}")
                fg.write_videofile(fg_video_path, fps=FPS, codec='libx264', audio=False, verbose=False, logger=None, preset='ultrafast')
            
            # Determine background source path
            # Use original (uncropped) video for background blur - NOT the pre-rendered foreground.
            # The foreground is already cropped/scaled; using it as bg would show wrong crop and bad blur.
            bg_source_path = original_video_path if original_video_path and os.path.exists(original_video_path) else fg_video_path
            if bg_source_path != fg_video_path:
                log(f"[EXPORT] Using original video for background blur: {os.path.basename(bg_source_path)}")
            else:
                log(f"[EXPORT] ⚠️ No original video path available, using foreground as background source")
            
            # Export with FFmpeg filters
            ffmpeg_export_successful = _export_with_ffmpeg_filters(
                bg_path=bg_source_path,
                fg_path=fg_video_path,
                caption_segments=captions_for_ffmpeg,
                audio_path=audio_temp_path,
                output_path=output_path,
                video_width=WIDTH,
                video_height=HEIGHT,
                log_fn=log,
                effect_settings=effect_settings,
                mirror_video=mirror_video,
                target_duration=target_duration,
                preferred_font=preferred_font,
                words_per_caption=words_per_caption,
                text_color_rgba=caption_text_color if caption_text_color is not None else globals().get('CAPTION_TEXT_COLOR'),
                stroke_color_rgba=caption_stroke_color if caption_stroke_color is not None else globals().get('CAPTION_STROKE_COLOR'),
                stroke_width=caption_stroke_width if caption_stroke_width is not None else globals().get('CAPTION_STROKE_WIDTH'),
                font_size=caption_font_size if caption_font_size is not None else globals().get('CAPTION_FONT_SIZE'),
                blur_radius=blur_radius,
                dim_factor=dim_factor,
                bg_scale_extra=bg_scale_extra,
                crop_top_ratio=crop_top_ratio,
                crop_bottom_ratio=crop_bottom_ratio,
                caption_y_offset=caption_y_offset,
                force_cpu=use_cpu_fallback
            )
            
            if ffmpeg_export_successful:
                log("[EXPORT] ✅ FFmpeg export completed successfully!")
                # Clean up temp files
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass
                return True
            else:
                last_ffmpeg_error = "FFmpeg returned non-zero exit code (check log above for stderr details)"
                log(f"[EXPORT] ⚠️ FFmpeg export failed (attempt {ffmpeg_attempt + 1}/{MAX_FFMPEG_RETRIES}) - see FFmpeg stderr output above")
                
        except Exception as e:
            last_ffmpeg_error = str(e)
            log(f"[EXPORT] ⚠️ FFmpeg export exception (attempt {ffmpeg_attempt + 1}/{MAX_FFMPEG_RETRIES}): {e}")
            import traceback
            log(f"[EXPORT] Traceback: {traceback.format_exc()}")
    
    # FFmpeg failed after all retries - raise error (do NOT fall back to MoviePy)
    if not ffmpeg_export_successful:
        error_msg = f"FFmpeg export failed after {MAX_FFMPEG_RETRIES} attempts. Last error: {last_ffmpeg_error}"
        log(f"[EXPORT] ❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # MoviePy fallback (disabled - kept as emergency backup, only reachable if code above is modified)
    if False:
        try:
            log("[EXPORT] Using MoviePy export (fallback or FFmpeg disabled)...")
            # Verify audio clip before compositing
            if audio_clip is None:
                try:
                    log(f"[COMPOSE WARNING] ⚠️ audio_clip is None! Final video will have no audio!")
                except Exception:
                    pass
            else:
                try:
                    log(f"[COMPOSE] Audio clip duration: {audio_clip.duration:.2f}s")
                except Exception:
                    pass
            
            # Create blurred video background lazily (only for MoviePy fallback path)
            # This per-frame processing is expensive, so we skip it when FFmpeg export succeeds
            log("[COMPOSE] Creating blurred video background for MoviePy export...")
            img_w, img_h = video_clip.w, video_clip.h
            scale_needed = max(WIDTH / img_w, HEIGHT / img_h) * bg_scale_extra
            new_w = int(img_w * scale_needed)
            new_h = int(img_h * scale_needed)
            left_bg = max(0, (new_w - WIDTH) // 2)
            top_bg = max(0, (new_h - HEIGHT) // 2)
            
            def make_blurred_bg_frame(get_frame, t):
                """Apply blur and dim to each frame for the video background."""
                frame = get_frame(t)
                img = Image.fromarray(frame)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                img = img.crop((left_bg, top_bg, left_bg + WIDTH, top_bg + HEIGHT))
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                # Apply brightness: dim_factor < 1.0 darkens, > 1.0 brightens
                # Add brightness compensation for color loss during processing
                effective_brightness = dim_factor * (1.0 + BG_BRIGHTNESS_BOOST)
                if abs(effective_brightness - 1.0) > 0.01:
                    img = ImageEnhance.Brightness(img).enhance(effective_brightness)
                return np.array(img)
            
            bg_static = video_clip.fl(make_blurred_bg_frame).set_duration(video_clip.duration)
            log(f"[COMPOSE] Video background created: {WIDTH}x{HEIGHT}, blur={blur_radius}, dim={dim_factor}")
            
            # Create MoviePy caption clips lazily (only for MoviePy fallback)
            # Uses the same caption_data_for_ffmpeg list (text/timing data) to build PIL-based clips
            log(f"[COMPOSE] Creating {len(caption_data_for_ffmpeg)} MoviePy caption clips (PIL rendering)...")
            for cap_data in caption_data_for_ffmpeg:
                try:
                    pil_img = generate_caption_image(cap_data['text'], preferred_font=preferred_font, log=log)
                    if pil_img is None:
                        continue
                    pil_rgba = pil_img.convert("RGBA")
                    alpha_arr_check = np.array(pil_rgba.split()[-1])
                    if not np.any(alpha_arr_check > 0):
                        continue
                    rgb_arr = np.array(pil_rgba.convert("RGB"))
                    alpha_arr = alpha_arr_check.astype("float32") / 255.0
                    g_start = cap_data['start']
                    g_dur = max(0.25, cap_data['end'] - cap_data['start'])
                    img_clip = ImageClip(rgb_arr).set_start(g_start).set_duration(g_dur)
                    mask_clip = ImageClip(alpha_arr, ismask=True).set_start(g_start).set_duration(g_dur)
                    img_clip = img_clip.set_mask(mask_clip)
                    y_offset = globals().get('CAPTION_Y_OFFSET', 0)
                    if y_offset == 0:
                        img_clip = img_clip.set_position(("center", "bottom"))
                    else:
                        img_clip = img_clip.set_position(lambda t: ("center", HEIGHT - img_clip.h + y_offset))
                    caption_clips.append(img_clip)
                except Exception:
                    continue
            log(f"[COMPOSE] ✓ Created {len(caption_clips)} caption clips")
            
            final = CompositeVideoClip([bg_static, fg] + caption_clips, size=(WIDTH, HEIGHT)).set_audio(audio_clip)
            try:
                log(f"[COMPOSE] ✓ Final composition created successfully")
                log(f"[COMPOSE] Layers: background + foreground + {len(caption_clips)} caption overlays")
            except Exception:
                pass
        except Exception as e:
            try:
                log(f"[COMPOSE ERROR] Failed to create CompositeVideoClip: {e}")
                import traceback
                log(f"[COMPOSE ERROR] Traceback: {traceback.format_exc()}")
            except Exception:
                pass
            raise

        # export monitoring and fallback
        codec_try_order = []
        if USE_GPU_IF_AVAILABLE and ffmpeg_supports_nvenc(PREFERRED_NVENC_CODEC):
            codec_try_order.append(PREFERRED_NVENC_CODEC)
            log(f"[EXPORT] GPU acceleration enabled for export (NVENC: {PREFERRED_NVENC_CODEC})")
        else:
            log("[EXPORT] Using CPU encoding for export (libx264)")
        codec_try_order.append("libx264")

        STALL_TIMEOUT = 90
        CHECK_INTERVAL = 8
        MAX_ATTEMPTS_PER_CODEC = 1

        def _run_write(final_clip, out_path, codec_name, ffmpeg_params, threads_setting, result_dict):
            try:
                final_clip.write_videofile(
                    out_path,
                    fps=FPS,
                    codec=codec_name,
                    audio_codec="aac",
                    audio_bitrate="192k",
                    threads=threads_setting,
                    ffmpeg_params=ffmpeg_params,
                    verbose=False,
                    logger="bar"
                )
                result_dict["ok"] = True
            except Exception as e:
                result_dict["ok"] = False
                result_dict["error"] = str(e)
            finally:
                try:
                    final_clip.close()
                except Exception:
                    pass
                gc.collect()

        last_error = None
        for codec in codec_try_order:
            for attempt in range(MAX_ATTEMPTS_PER_CODEC):
                ffmpeg_params = _make_ffmpeg_params_for_codec(codec)
                threads_setting = os.cpu_count() or 4  # Use all CPU cores for maximum export speed
                # Log GPU usage status for user clarity
                gpu_status = "GPU (NVENC)" if codec in ("h264_nvenc", "hevc_nvenc") else "CPU (libx264)"
                log(f"[EXPORT] Starting export with {gpu_status}")
                log(f"[EXPORT] Codec: {codec}, Params: {ffmpeg_params}, Threads: {threads_setting}, Attempt: {attempt+1}")
                result = {"ok": False, "error": None}
                writer_thread = threading.Thread(target=_run_write, args=(final, output_path, codec, ffmpeg_params, threads_setting, result), daemon=True)
                writer_thread.start()

                prev_size = -1
                stalled_since = None
                start_t = time.time()
                while writer_thread.is_alive():
                    time.sleep(CHECK_INTERVAL)
                    try:
                        cur_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    except Exception:
                        cur_size = 0
                    log(f"[export-monitor] {codec} out_size={cur_size} bytes (elapsed {int(time.time()-start_t)}s)")
                    if cur_size > prev_size:
                        prev_size = cur_size
                        stalled_since = None
                    else:
                        if stalled_since is None:
                            stalled_since = time.time()
                        else:
                            if (time.time() - stalled_since) > STALL_TIMEOUT:
                                log(f"[export-monitor] Detected stall for {STALL_TIMEOUT}s. Attempting to abort and retry.")
                                try:
                                    os.remove(output_path)
                                    log("[export-monitor] Removed partial output file.")
                                except Exception as e_rm:
                                    log(f"[export-monitor] Could not remove partial file: {e_rm}")
                                result["ok"] = False
                                result["error"] = f"Stalled export (no growth for {STALL_TIMEOUT}s)"
                                break
                writer_thread.join(timeout=1.0)
                if result.get("ok"):
                    gpu_used = "GPU (NVENC)" if codec in ("h264_nvenc", "hevc_nvenc") else "CPU (libx264)"
                    log(f"[EXPORT] ✅ Export finished successfully using {gpu_used}")
                    ok_probe, probe_out = probe_file_with_ffmpeg(output_path)
                    if ok_probe:
                        log("Export OK — ffmpeg probe didn't find fatal errors.")
                        try:
                            if os.name == "nt":
                                os.startfile(output_path)
                            elif sys.platform == "darwin":
                                subprocess.Popen(["open", output_path])
                            else:
                                subprocess.Popen(["xdg-open", output_path])
                        except Exception as eopen:
                            log(f"Could not open output automatically: {eopen}")
                        return True
                    else:
                        log("[export-monitor] Probe reported issues, attempting re-encode.")
                        last_error = probe_out
                        reencoded = output_path.replace(".mp4", "_reencoded.mp4")
                        success = reencode_with_libx264(output_path, reencoded, log)
                        if success:
                            try:
                                os.replace(reencoded, output_path)
                                log("Re-encoded succeeded and replaced original output.")
                                return True
                            except Exception:
                                log(f"Re-encoded saved at: {reencoded}")
                                return True
                        else:
                            log("Re-encode failed.")
                else:
                    last_error = result.get("error")
                    log(f"[export-monitor] Export attempt failed/aborted: {last_error}")
                    time.sleep(1.0)
                    continue

        log(f"[export-monitor] All export attempts exhausted. Last error: {last_error}")
        return False

# ----------------- Processing pipeline (single job) -----------------
def crop_precise_top_bottom_return_cropped(video_clip, log, top_ratio=None, bottom_ratio=None):
    tr = CROP_TOP_RATIO if top_ratio is None else top_ratio
    br = CROP_BOTTOM_RATIO if bottom_ratio is None else bottom_ratio
    log(f"Cropping using ratios top={tr:.4f}, bottom={br:.4f} (return cropped clip).")
    original_width, original_height = video_clip.size
    crop_top = int(original_height * tr)
    crop_bottom = int(original_height * br)
    crop_y_start = crop_top
    crop_y_end = original_height - crop_bottom
    cropped_video = video_clip.crop(x1=0, y1=crop_y_start, x2=original_width, y2=crop_y_end)
    log(f"Crop done. Cropped size: {cropped_video.size}, duration: {cropped_video.duration:.2f}s")
    return cropped_video

def _compose_with_pref_font(preferred_font, video_clip, audio_clip, caption_segments, output_path, log, blur_radius=STATIC_BG_BLUR_RADIUS, bg_scale_extra=BG_SCALE_EXTRA, dim_factor=DIM_FACTOR, words_per_caption=2, effect_settings=None, pre_rendered_fg_path=None, mirror_video=False, target_duration=None, original_video_path=None, crop_top_ratio=None, crop_bottom_ratio=None, caption_text_color=None, caption_stroke_color=None, caption_stroke_width=None, caption_font_size=None, caption_y_offset=None):
    """Helper to temporarily override global CAPTION_FONT_PREFERRED for the duration of compose."""
    old = globals().get('CAPTION_FONT_PREFERRED')
    try:
        if preferred_font:
            globals()['CAPTION_FONT_PREFERRED'] = preferred_font
            try:
                if log: log(f"[FONT] Overriding CAPTION_FONT_PREFERRED -> {preferred_font}")
            except Exception:
                pass
        # call compose with keyword args to avoid positional mismatch
        return compose_final_video_with_static_blurred_bg(video_clip=video_clip, audio_clip=audio_clip, caption_segments=caption_segments, output_path=output_path, preferred_font=preferred_font, log=log, blur_radius=blur_radius, bg_scale_extra=bg_scale_extra, dim_factor=dim_factor, words_per_caption=words_per_caption, effect_settings=effect_settings, pre_rendered_fg_path=pre_rendered_fg_path, mirror_video=mirror_video, target_duration=target_duration, original_video_path=original_video_path, crop_top_ratio=crop_top_ratio, crop_bottom_ratio=crop_bottom_ratio, caption_text_color=caption_text_color, caption_stroke_color=caption_stroke_color, caption_stroke_width=caption_stroke_width, caption_font_size=caption_font_size, caption_y_offset=caption_y_offset)
    finally:
        try:
            if preferred_font and old is not None:
                globals()['CAPTION_FONT_PREFERRED'] = old
                try:
                    if log: log(f"[FONT] Restored CAPTION_FONT_PREFERRED -> {old}")
                except Exception:
                    pass
        except Exception:
            pass



def adjust_video_speed(video_clip, audio_duration, log, max_change=2.0):
    old_dur = video_clip.duration
    target = float(audio_duration)
    if abs(old_dur - target) < 0.01:
        log("Video already approximately matches audio duration.")
        return video_clip.set_duration(target)
    factor = old_dur / target
    log(f"Adjust video: {old_dur:.2f}s -> {target:.2f}s ; speedx factor = {factor:.4f}")
    min_factor = 1.0 / max_change
    max_factor = max_change
    if factor < min_factor or factor > max_factor:
        log(f"Factor {factor:.4f} out of bounds ({min_factor:.3f}..{max_factor:.3f}). Using fallback.")
        if old_dur < target:
            loops = int(np.ceil(target / old_dur))
            log(f"Looping video {loops} times then subclip to {target:.2f}s")
            looped = concatenate_videoclips([video_clip] * loops).subclip(0, target)
            return looped.set_duration(target)
        else:
            log("Trimming video to target duration.")
            return video_clip.subclip(0, target).set_duration(target)
    adjusted = video_clip.fx(speedx, factor).set_duration(target)
    log(f"Speed adjusted by factor {factor:.4f}. New duration: {adjusted.duration:.2f}s")
    return adjusted

def make_music_match_duration(music_clip, target_duration, log, music_gain=None):
    if music_gain is None:
        music_gain = MUSIC_GAIN
    log(f"[AUDIO] Applying music volume: {music_gain:.2f}x ({_gain_to_display(music_gain)})")
    if music_clip.duration <= 0.01:
        raise ValueError("Music clip invalid / durată zero.")
    if abs(music_clip.duration - target_duration) < 0.01:
        return music_clip.volumex(music_gain).set_duration(target_duration)
    if music_clip.duration < target_duration:
        loops = int(np.ceil(target_duration / music_clip.duration))
        log(f"Music too short ({music_clip.duration:.2f}s). Looping {loops} times to reach {target_duration:.2f}s")
        looped = concatenate_audioclips([music_clip] * loops).subclip(0, target_duration)
        return looped.volumex(music_gain)
    else:
        log(f"Music longer ({music_clip.duration:.2f}s). Trimming to {target_duration:.2f}s and applying fadeout {MUSIC_FADEOUT_SECONDS}s.")
        trimmed = music_clip.subclip(0, target_duration)
        trimmed = trimmed.fx(audio_fadeout, MUSIC_FADEOUT_SECONDS)
        return trimmed.volumex(music_gain).set_duration(target_duration)

def process_single_job(video_path, voice_path, music_path, requested_output_path, q, preferred_font=None, custom_top_ratio=None, custom_bottom_ratio=None, mirror_video=False, words_per_caption=2, use_4k=False, blur_radius=None, bg_scale_extra=None, dim_factor=None, effect_settings=None, use_ai_voice=None, target_language=None, translation_enabled=None, tts_language=None, silence_threshold_ms=300, caption_text_color=None, caption_stroke_color=None, caption_stroke_width=None, caption_font_size=None, caption_y_offset=None, pre_generated_voice=None, voice_gain=None, music_gain=None):
    def log(s):
        q.put(str(s))
    def _check_stop():
        if globals().get('STOP_REQUESTED', False):
            log("[STOP] ⛔ Processing stopped by user.")
            raise InterruptedError("Stop requested by user")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = QueueWriter(q)
    temp_fg = None
    
    # Log received crop parameters for debugging
    log(f"[DEBUG] Received custom_top_ratio: {custom_top_ratio}")
    log(f"[DEBUG] Received custom_bottom_ratio: {custom_bottom_ratio}")
    
    # Use explicit gain parameters if provided, otherwise fall back to globals
    if voice_gain is None:
        voice_gain = globals().get('VOICE_GAIN', 1.5)
    if music_gain is None:
        music_gain = globals().get('MUSIC_GAIN', 0.4)
    log(f"[AUDIO] Voice gain: {voice_gain:.2f}x ({_gain_to_display(voice_gain)})")
    log(f"[AUDIO] Music gain: {music_gain:.2f}x ({_gain_to_display(music_gain)})")
    
    # Set defaults for effects if not provided
    if blur_radius is None:
        blur_radius = globals().get('STATIC_BG_BLUR_RADIUS', 25)
    if bg_scale_extra is None:
        bg_scale_extra = globals().get('BG_SCALE_EXTRA', 1.08)
    # Always use global DIM_FACTOR — ignore stale values from saved presets/jobs
    # (there is no UI control for dim_factor, so saved values are always stale)
    dim_factor = globals().get('DIM_FACTOR', 1.0)
    
    # Determine if AI voice should be used - prefer parameter over global
    if use_ai_voice is None:
        use_ai_voice = globals().get('USE_AI_VOICE_REPLACEMENT', False)
    
    # Determine language settings - prefer parameters over globals
    if target_language is None:
        target_language = globals().get('TARGET_LANGUAGE', 'none')
    if translation_enabled is None:
        translation_enabled = globals().get('TRANSLATION_ENABLED', False)
    if tts_language is None:
        tts_language = globals().get('TTS_LANGUAGE', 'en')
    
    # Auto-sync TTS language with translation target language
    # When both translation and TTS are enabled, TTS should speak in the translated language
    if translation_enabled and use_ai_voice and target_language and target_language != 'none':
        tts_lang_synced = TRANS_TO_TTS_LANG.get(target_language, target_language)
        tts_language = tts_lang_synced
    
    # Set 4K mode if requested
    # NOTE: Using global state for IS_4K_MODE. This is safe because:
    # 1. Jobs are processed sequentially (one at a time) via queue_worker
    # 2. Single jobs via on_run_single run in separate threads but don't overlap
    # 3. The old value is saved and restored in the finally block
    old_is_4k = globals().get('IS_4K_MODE', False)
    if use_4k:
        globals()['IS_4K_MODE'] = True
        log("[RESOLUTION] Job set to 4K mode (2160x3840)")
    else:
        globals()['IS_4K_MODE'] = False
        log("[RESOLUTION] Job set to HD mode (1080x1920)")
    
    try:
        load_preferred_font_cached(preferred_font or CAPTION_FONT_PREFERRED, CAPTION_FONT_SIZE, log=log)
        if REQUIRE_FONT_BANGERS and (LOADED_FONT_PATH is None or ("bangers" not in str(LOADED_FONT_PATH).lower())):
            log("Error: Bangers font required but not found. Aborting job.")
            return

        output_path = make_unique_output_path(requested_output_path, log=log)

        if not os.path.exists(video_path):
            log(f"Error: VIDEO missing: {video_path}")
            return
        
        # Voice is now optional - extract from video if not provided
        voice_from_video = False
        if not voice_path or not os.path.exists(voice_path):
            log("[VOICE] No separate voice file provided - extracting audio from video...")
            voice_from_video = True
            # Extract audio from video to a temporary file
            temp_voice_path = tempfile.mktemp(suffix='.mp3', prefix='extracted_voice_')
            try:
                video_clip_for_audio = VideoFileClip(video_path)
                if video_clip_for_audio.audio is None:
                    log("[VOICE ERROR] Video has no audio track!")
                    if not use_ai_voice:
                        log("[VOICE ERROR] Cannot proceed without voice audio or AI voice enabled.")
                        return
                    log("[VOICE] Will generate AI voice from text/captions only.")
                    voice_path = None
                else:
                    video_clip_for_audio.audio.write_audiofile(temp_voice_path, logger=None)
                    voice_path = temp_voice_path
                    log(f"[VOICE] ✓ Extracted audio from video: {temp_voice_path}")
                    video_clip_for_audio.close()
            except Exception as e:
                log(f"[VOICE ERROR] Failed to extract audio from video: {e}")
                if not use_ai_voice:
                    return
                log("[VOICE] Will proceed with AI voice generation only.")
                voice_path = None
        
        if not os.path.exists(music_path):
            log(f"Error: MUSIC missing: {music_path}")
            return

        # Load video first to get dimensions
        _check_stop()
        original_clip = VideoFileClip(video_path)
        cropped = crop_precise_top_bottom_return_cropped(original_clip, log, top_ratio=custom_top_ratio, bottom_ratio=custom_bottom_ratio)

        orig_w, orig_h = original_clip.size
        crop_top = int(orig_h * (CROP_TOP_RATIO if custom_top_ratio is None else custom_top_ratio))
        crop_bottom = int(orig_h * (CROP_BOTTOM_RATIO if custom_bottom_ratio is None else custom_bottom_ratio))
        crop_h = orig_h - crop_top - crop_bottom
        crop_w = orig_w
        crop_x = 0
        crop_y = crop_top
        
        # Log crop values for debugging
        log(f"[CROP] Custom crop enabled: {custom_top_ratio is not None or custom_bottom_ratio is not None}")
        log(f"[CROP] Top ratio: {custom_top_ratio if custom_top_ratio is not None else CROP_TOP_RATIO} ({crop_top}px)")
        log(f"[CROP] Bottom ratio: {custom_bottom_ratio if custom_bottom_ratio is not None else CROP_BOTTOM_RATIO} ({crop_bottom}px)")
        log(f"[CROP] Original: {orig_w}x{orig_h}, Cropped: {crop_w}x{crop_h}")

        try:
            # TikTok format: scale foreground to fill canvas WIDTH
            # Foreground fills the width, background visible above/below
            width_scale = WIDTH / cropped.w
        except Exception:
            width_scale = 1.0
        
        # Get user's zoom setting
        user_zoom = globals().get('VIDEO_ZOOM_SCALE', 1.0)
        
        # TikTok layout: foreground fills canvas width, blurred bg shows above/below
        # width_scale ensures the foreground matches the 1080px canvas width
        # For landscape videos (width_scale < 1.0): scales down to fit canvas width
        # For portrait videos (width_scale > 1.0): scales up to fill canvas width
        is_4k = globals().get('IS_4K_MODE', False)
        base_scale_factor = 1.03
        
        # Ensure the foreground fills a minimum height of the canvas.
        # Without this, landscape videos with heavy crop (e.g., 65%) result in a tiny
        # content strip (11% of canvas). With this, the content zooms in to fill at
        # least MIN_FG_HEIGHT_RATIO of the canvas (sides are clipped by overlay centering).
        min_fg_height = HEIGHT * MIN_FG_HEIGHT_RATIO
        estimated_height = crop_h * width_scale * base_scale_factor
        if estimated_height < min_fg_height and crop_h > 0:
            # Content would be too thin — scale up to meet minimum height
            height_scale = min_fg_height / crop_h
            fg_base_scale = max(width_scale, height_scale)
            log(f"[SCALE] Content too thin ({int(estimated_height)}px < {int(min_fg_height)}px min), using height-based scale: {height_scale:.3f}")
        else:
            fg_base_scale = width_scale
        
        # Apply base scale factor (slight overscan to avoid black edges)
        fit_scale = fg_base_scale * base_scale_factor
        
        # Apply user's zoom factor directly - no capping
        # User zoom of 1.08x means 8% larger than the auto-fit size
        fg_scale = fit_scale * user_zoom
        
        log(f"[SCALE] width_scale={width_scale:.3f}, fit_scale={fit_scale:.3f}, user_zoom={user_zoom:.2f}, final={fg_scale:.3f}")
        
        scale_w = max(2, int(round(crop_w * fg_scale)) & ~1)  # Ensure even (required for yuv420p)
        scale_h = max(2, int(round(crop_h * fg_scale)) & ~1)
        
        # Get caption offset for logging
        y_offset = globals().get('CAPTION_Y_OFFSET', 0)
        offset_desc = f"moving {abs(y_offset)}px {'UP' if y_offset < 0 else 'DOWN'} from bottom" if y_offset != 0 else "at BOTTOM (default)"
        
        # Enhanced detailed logging - now that we have all values
        log("═══════════════ PROCESSING JOB ═══════════════")
        log(f"VIDEO: {os.path.basename(video_path)} ({orig_w}x{orig_h}, {original_clip.duration:.1f}s)")
        voice_label = os.path.basename(voice_path) if voice_path else "(AI voice only)"
        log(f"VOICE: {voice_label} (volume: {voice_gain:.2f}x, {_gain_to_display(voice_gain)})")
        log(f"MUSIC: {os.path.basename(music_path)} (volume: {music_gain:.2f}x, {_gain_to_display(music_gain)})")
        
        # Font information
        font_info = "default"
        if LOADED_FONT_PATH:
            font_info = f"{os.path.basename(LOADED_FONT_PATH)}"
            try:
                if LOADED_FONT_FAMILY:
                    font_info += f" ({LOADED_FONT_FAMILY})"
            except:
                pass
        log(f"FONT: {font_info}")
        
        # Crop information
        crop_top_pct = (custom_top_ratio if custom_top_ratio is not None else CROP_TOP_RATIO) * 100
        crop_bottom_pct = (custom_bottom_ratio if custom_bottom_ratio is not None else CROP_BOTTOM_RATIO) * 100
        
        # Check NVENC support before logging
        use_nvenc = USE_GPU_IF_AVAILABLE and ffmpeg_supports_nvenc(PREFERRED_NVENC_CODEC)
        
        log(f"CROP: Top {crop_top_pct:.1f}%, Bottom {crop_bottom_pct:.1f}% → {crop_w}x{crop_h}")
        log(f"ZOOM: User setting {user_zoom:.2f}x")
        log(f"SCALE: {fg_scale:.2f}x → {scale_w}x{scale_h}")
        log(f"CAPTION OFFSET: {y_offset}px ({offset_desc})")
        log(f"MIRROR: {'Enabled' if mirror_video else 'Disabled'}")
        log(f"OUTPUT: {os.path.basename(output_path)} ({WIDTH}x{HEIGHT}, {FPS}fps{',' if use_nvenc else ''}{' NVENC' if use_nvenc else ''})")
        log("══════════════════════════════════════════════")

        temp_dir = tempfile.mkdtemp(prefix="tiktok_prerender_")
        temp_fg = os.path.join(temp_dir, "fg_prerender.mp4")
        log(f"Attempting ffmpeg pre-render -> {os.path.basename(temp_fg)} (nvenc={use_nvenc})")
        prer_ok = pre_render_foreground_ffmpeg(video_path, temp_fg, crop_x, crop_y, crop_w, crop_h, scale_w, scale_h, FPS, use_nvenc, log)

        if prer_ok and os.path.exists(temp_fg):
            log("Using pre-rendered foreground clip.")
            fg_clip = VideoFileClip(temp_fg)
        else:
            log("Pre-render failed or missing — falling back to MoviePy in-memory crop/resize.")
            fg_clip = cropped.resize(fg_scale).set_position(("center", "center")).set_duration(cropped.duration)
        
        # Remove original audio from video clip to prevent duplicate audio
        # Only the mixed_audio (voice/TTS + music) should be used
        log("Removing original audio from video clip...")
        fg_clip = fg_clip.without_audio()
        log("✓ Original video audio removed (will use voice/TTS + music only)")
        
        # Apply mirror if enabled
        if mirror_video:
            log("Applying horizontal mirror/flip to video...")
            from moviepy.video.fx.mirror_x import mirror_x
            fg_clip = mirror_x(fg_clip)
            log("✓ Video mirrored successfully")

        # Handle audio based on whether we have a voice file
        if voice_path and os.path.exists(voice_path):
            voice_clip = AudioFileClip(voice_path).volumex(voice_gain)
            music_clip = AudioFileClip(music_path)
            target_duration = voice_clip.duration
            log(f"Voice duration (target): {target_duration:.2f}s")
            log(f"[AUDIO] Applying voice volume: {voice_gain:.2f}x ({_gain_to_display(voice_gain)})")
            music_matched = make_music_match_duration(music_clip, target_duration, log, music_gain=music_gain)
            mixed_audio = CompositeAudioClip([music_matched, voice_clip.set_start(0)]).set_duration(target_duration)
            
            synced_video = adjust_video_speed(fg_clip, mixed_audio.duration, log, max_change=2.0)
            
            # Transcribe captions ONLY if AI voice replacement is NOT enabled
            # If AI voice is enabled, we'll transcribe from the TTS audio later
            if not use_ai_voice:
                log("[CAPTION] Transcribing captions from original voice...")
                _check_stop()
                caption_segments = transcribe_captions(
                    voice_path, 
                    log, 
                    translate_to=target_language if translation_enabled else None
                )
            else:
                # AI voice is enabled — check if we already have a pre-generated voice
                # with caption segments (from the batch parallel pipeline).  In that case
                # there is NO need to load Whisper and transcribe again — the captions are
                # already inside pre_generated_voice and will be applied at line ~5050.
                if pre_generated_voice and 'caption_segments' in pre_generated_voice:
                    log("[CAPTION] ⚡ Pre-generated voice includes captions — skipping Whisper transcription")
                    caption_segments = pre_generated_voice['caption_segments']
                    if not caption_segments:
                        log("[CAPTION] ⚠️ Pre-generated captions list is empty (Whisper found no words in TTS audio)")
                else:
                    log("[CAPTION] Deferring caption generation until after TTS voice is created...")
                    # Generate initial caption segments from original voice for TTS generation
                    caption_segments = transcribe_captions(
                        voice_path, 
                        log, 
                        translate_to=target_language if translation_enabled else None
                    )
                    # Release Whisper model immediately after transcription to free GPU memory
                    # for the NVENC video encoding that follows
                    _release_whisper_model(log=log)
        else:
            # No voice file - use video duration as target
            log("[NO VOICE] Using video duration as target")
            target_duration = fg_clip.duration
            music_clip = AudioFileClip(music_path)
            music_matched = make_music_match_duration(music_clip, target_duration, log, music_gain=music_gain)
            mixed_audio = music_matched.set_duration(target_duration)
            synced_video = fg_clip
            caption_segments = []
            
            # If AI voice is enabled, we still need to generate it even without a voice file
            # We'll need to create dummy caption segments from user input or skip captions
            if use_ai_voice:
                log("[NO VOICE + AI TTS] Will generate AI voice without captions")
                # For now, we'll skip caption generation when there's no voice to transcribe
                # The TTS will be generated below if caption_segments exists or can be created
            else:
                log("[NO VOICE] No captions will be generated (no voice to transcribe)")
        
        # Track if TTS was successfully applied
        tts_successfully_applied = False
        
        # Optional: Replace voice with AI-generated TTS or generate from scratch
        if use_ai_voice:
            # Check if voice was pre-generated by the parallel pipeline
            if pre_generated_voice and pre_generated_voice.get('compressed_tts_path'):
                log("")
                log("━"*60)
                log("[AI VOICE] ⚡ USING PRE-GENERATED VOICE (parallel pipeline)")
                log(f"[AI VOICE] TTS audio: {pre_generated_voice['compressed_tts_path']}")
                log(f"[AI VOICE] Duration: {pre_generated_voice['tts_duration']:.2f}s")
                log(f"[AI VOICE] Captions: {len(pre_generated_voice.get('caption_segments', []))} segments")
                log("━"*60)
                log("")
                
                try:
                    compressed_tts_path = pre_generated_voice['compressed_tts_path']
                    caption_segments = pre_generated_voice['caption_segments']
                    tts_duration = pre_generated_voice['tts_duration']
                    
                    # Load the pre-generated TTS audio
                    tts_clip = AudioFileClip(compressed_tts_path).volumex(voice_gain)
                    
                    log(f"[AI VOICE] TTS voice duration: {tts_duration:.2f}s")
                    log(f"[AI VOICE] Keeping TTS voice at original speed (natural sound)")
                    
                    # Decide final duration: match video to voice, or let
                    # video play at natural speed when it is much longer.
                    original_video_dur = fg_clip.duration
                    speed_factor = original_video_dur / tts_duration if tts_duration > 0.01 else 1.0
                    
                    if speed_factor > 2.0:
                        # Video is much longer than TTS — keep video at natural speed
                        final_duration = original_video_dur
                        log(f"[AI VOICE] Video ({original_video_dur:.2f}s) >> TTS ({tts_duration:.2f}s) — keeping full video")
                        log(f"[AI VOICE] TTS voice ends at {tts_duration:.2f}s, music continues to {final_duration:.2f}s")
                        music_matched = make_music_match_duration(music_clip, final_duration, log, music_gain=music_gain)
                        mixed_audio = CompositeAudioClip([music_matched, tts_clip.set_start(0)]).set_duration(final_duration)
                        synced_video = fg_clip.set_duration(final_duration)
                    else:
                        # Video can be reasonably adjusted — sync to TTS duration
                        final_duration = tts_duration
                        log(f"[AI VOICE] Adjusting video speed to match TTS duration...")
                        music_matched = make_music_match_duration(music_clip, final_duration, log, music_gain=music_gain)
                        mixed_audio = CompositeAudioClip([music_matched, tts_clip.set_start(0)]).set_duration(final_duration)
                        synced_video = adjust_video_speed(fg_clip, final_duration, log, max_change=2.0)
                    
                    log("")
                    log("━"*60)
                    log("[AI VOICE] ✅ PRE-GENERATED VOICE APPLIED!")
                    log(f"[AI VOICE] Final audio duration: {mixed_audio.duration:.2f}s")
                    log(f"[AI VOICE] Final video duration: {synced_video.duration:.2f}s")
                    log("━"*60)
                    log("")
                    
                    tts_successfully_applied = True
                    log("[AI VOICE] 🎯 TTS FLAG SET: Pre-generated TTS audio will be used in final video")
                    
                except Exception as e:
                    log(f"[AI VOICE ERROR] ❌ Failed to use pre-generated TTS audio: {e}")
                    import traceback
                    log(traceback.format_exc())
                    log("[AI VOICE ERROR] Falling back to generating voice inline...")
                    pre_generated_voice = None  # Clear so we fall through to inline generation
                    tts_successfully_applied = False
            
            # Inline voice generation (normal path or fallback from failed pre-generated)
            if not tts_successfully_applied and not pre_generated_voice:
                if caption_segments:
                    log("")
                    log("━"*60)
                    log("[AI VOICE] 🎵 GENERATING AI VOICE REPLACEMENT")
                    log(f"[AI VOICE] Segments to synthesize: {len(caption_segments)}")
                    log(f"[AI VOICE] TTS language: {tts_language}")
                    if translation_enabled and target_language and target_language != 'none':
                        log(f"[AI VOICE] ✓ Using translated text ({target_language}) for TTS")
                    log("━"*60)
                    log("")
                    
                    tts_audio_path = replace_voice_with_tts(
                        caption_segments, 
                        language=tts_language,
                        log=log
                    )
                    if tts_audio_path:
                        # Replace voice_clip with TTS audio
                        try:
                            log("")
                            log("[AI VOICE] 🔊 Integrating AI voice into video...")
                            
                            # STEP 1: Remove all silences from TTS audio FIRST for continuous speech
                            log("")
                            log("[AI VOICE] 📝 Removing silences from TTS audio for continuous playback...")
                            log(f"[AI VOICE] Using silence threshold: {silence_threshold_ms}ms")
                            compressed_tts_path, silence_map = remove_silence_from_audio(
                                tts_audio_path, 
                                output_path=None,
                                log=log,
                                min_silence_ms=silence_threshold_ms
                            )
                            
                            # STEP 2: Re-transcribe captions from the SILENCE-REMOVED audio
                            # This ensures captions match exactly with the final compressed audio
                            # CapCut-style: transcribe from final audio for perfect sync
                            log("")
                            log("[AI VOICE] 📝 TRANSCRIBING CAPTIONS FROM SILENCE-REMOVED TTS AUDIO")
                            log("[AI VOICE] CapCut-style: Captions generated from final compressed audio...")
                            translated_caption_segments = list(caption_segments)  # Save translated text before re-transcription
                            caption_segments = transcribe_captions(
                                compressed_tts_path,  # Use compressed audio instead of original
                                log, 
                                translate_to=None  # Already translated during TTS generation
                            )
                            log(f"[AI VOICE] ✓ Generated {len(caption_segments)} caption segments with perfect timing")
                            
                            # Preserve translated text from before re-transcription.
                            # Whisper re-transcription gives perfect timing but may auto-detect
                            # the wrong language, producing source-language text instead of the
                            # target-language translation. Map the original translated text
                            # onto the re-timed segments.
                            if translated_caption_segments and caption_segments:
                                if len(caption_segments) == len(translated_caption_segments):
                                    for re_seg, orig_seg in zip(caption_segments, translated_caption_segments):
                                        re_seg['text'] = orig_seg.get('text', re_seg.get('text', ''))
                                        if 'original_text' in orig_seg:
                                            re_seg['original_text'] = orig_seg['original_text']
                                    # Filter out segments with empty translated text (prevents
                                    # original-language Whisper text from leaking through).
                                    caption_segments = [s for s in caption_segments if s.get('text', '').strip()]
                                    log(f"[AI VOICE] ✓ Mapped translated text to {len(caption_segments)} re-timed segments (1:1)")
                                else:
                                    log(f"[AI VOICE] ⚠ Segment count changed ({len(translated_caption_segments)} → {len(caption_segments)}) — redistributing translated text")
                                    all_translated_words = []
                                    for seg in translated_caption_segments:
                                        all_translated_words.extend(seg.get('text', '').split())
                                    if all_translated_words:
                                        total_words = len(all_translated_words)
                                        n_segs = len(caption_segments)
                                        base_per_seg = total_words // n_segs
                                        remainder = total_words % n_segs
                                        word_idx = 0
                                        for i, seg in enumerate(caption_segments):
                                            count = base_per_seg + (1 if i < remainder else 0)
                                            seg['text'] = ' '.join(all_translated_words[word_idx:word_idx + count])
                                            word_idx += count
                                        # Remove segments that received no translated words — they
                                        # still contain the original-language Whisper text and would
                                        # appear as untranslated captions at the end of the video.
                                        caption_segments = [s for s in caption_segments if s.get('text', '').strip()]
                                        log(f"[AI VOICE] ✓ Distributed {total_words} translated words across {len(caption_segments)} re-timed segments")
                                    else:
                                        # All translated text was empty — clear re-transcription
                                        # segments so no original-language text leaks through.
                                        log("[AI VOICE] ⚠ No translated words available — clearing re-transcribed segments")
                                        caption_segments = []
                            elif translated_caption_segments and not caption_segments:
                                log("[AI VOICE] ⚠ Re-transcription empty — using original translated segments")
                                caption_segments = translated_caption_segments
                            log("")
                            
                            # Release Whisper model to free GPU memory for NVENC video encoding
                            _release_whisper_model(log=log)
                            
                            # No need for timestamp remapping - captions already match the compressed audio!
                            # Do NOT extend last caption — let the video play silently past
                            # the last spoken word so no foreign-language text leaks through.
                            
                            # Load the silence-removed TTS audio
                            tts_clip = AudioFileClip(compressed_tts_path).volumex(voice_gain)
                            
                            # Use TTS duration as the new target - DO NOT speed up/slow down the voice
                            tts_duration = tts_clip.duration
                            log(f"[AI VOICE] TTS voice duration (after silence removal): {tts_duration:.2f}s")
                            log(f"[AI VOICE] Keeping TTS voice at original speed (natural sound)")
                            
                            # Decide final duration: try to match video to voice,
                            # but if the video is much longer (would need >2× speed-up)
                            # just let the video play at its natural speed so the user
                            # sees the full content.  TTS voice plays for its portion,
                            # then only music continues for the rest.
                            original_video_dur = fg_clip.duration
                            speed_factor = original_video_dur / tts_duration if tts_duration > 0.01 else 1.0
                            
                            if speed_factor > 2.0:
                                # Video is much longer than TTS — keep video at natural speed
                                final_duration = original_video_dur
                                log(f"[AI VOICE] Video ({original_video_dur:.2f}s) >> TTS ({tts_duration:.2f}s) — keeping full video")
                                log(f"[AI VOICE] TTS voice ends at {tts_duration:.2f}s, music continues to {final_duration:.2f}s")
                                music_matched = make_music_match_duration(music_clip, final_duration, log, music_gain=music_gain)
                                mixed_audio = CompositeAudioClip([music_matched, tts_clip.set_start(0)]).set_duration(final_duration)
                                synced_video = fg_clip.set_duration(final_duration)
                            else:
                                # Video can be reasonably adjusted — sync to TTS duration
                                final_duration = tts_duration
                                log(f"[AI VOICE] Adjusting video speed to match TTS duration...")
                                music_matched = make_music_match_duration(music_clip, final_duration, log, music_gain=music_gain)
                                mixed_audio = CompositeAudioClip([music_matched, tts_clip.set_start(0)]).set_duration(final_duration)
                                synced_video = adjust_video_speed(fg_clip, final_duration, log, max_change=2.0)
                            
                            log("")
                            log("━"*60)
                            log("[AI VOICE] ✅ VOICE REPLACEMENT SUCCESSFUL!")
                            log("[AI VOICE] Voice plays continuously (silences removed)")
                            log("[AI VOICE] Captions synchronized with word timestamps")
                            log(f"[AI VOICE] Final audio duration: {mixed_audio.duration:.2f}s")
                            log(f"[AI VOICE] Final video duration: {synced_video.duration:.2f}s")
                            log("━"*60)
                            log("")
                            
                            # Mark that TTS was successfully applied
                            tts_successfully_applied = True
                            log("[AI VOICE] 🎯 TTS FLAG SET: TTS audio will be used in final video")
                            
                        except Exception as e:
                            log(f"[AI VOICE ERROR] ❌ Failed to use TTS audio: {e}")
                            import traceback
                            log(traceback.format_exc())
                            log("[AI VOICE ERROR] Falling back to original voice/music audio")
                            tts_successfully_applied = False
                    else:
                        log("[AI VOICE] ⚠️ TTS audio generation returned None - using original audio")
                        tts_successfully_applied = False
                else:
                    log("[AI VOICE] No captions available - AI voice replacement skipped")
                    log("[AI VOICE] Tip: Provide voice file with audio for automatic transcription and AI voice generation")
                    tts_successfully_applied = False

        # Final verification of what audio is being used
        log("")
        log("━"*60)
        log("[FINAL COMPOSITION] Preparing to create final video...")
        if tts_successfully_applied:
            log("[FINAL COMPOSITION] 🎤 USING AI-GENERATED TTS VOICE")
            log(f"[FINAL COMPOSITION] ✓ TTS voice successfully integrated")
        else:
            log("[FINAL COMPOSITION] 🎵 Using original voice/music audio")
            log(f"[FINAL COMPOSITION] (AI voice was not enabled or failed)")
        log(f"[FINAL COMPOSITION] Video duration: {synced_video.duration:.2f}s")
        log(f"[FINAL COMPOSITION] Audio duration: {mixed_audio.duration:.2f}s")
        log(f"[FINAL COMPOSITION] Caption segments: {len(caption_segments)}")
        if temp_fg and os.path.exists(temp_fg):
            log(f"[FINAL COMPOSITION] Pre-rendered foreground: {temp_fg}")
        log("━"*60)
        log("")
        
        # Pass mirror_video and target_duration for FFmpeg export optimization
        # Calculate target duration from audio or video, with fallback
        target_duration = None
        if mixed_audio and hasattr(mixed_audio, 'duration') and mixed_audio.duration:
            target_duration = mixed_audio.duration
        elif synced_video and hasattr(synced_video, 'duration') and synced_video.duration:
            target_duration = synced_video.duration
        
        ok = _compose_with_pref_font(preferred_font, synced_video, mixed_audio, caption_segments, output_path, log, blur_radius=blur_radius, bg_scale_extra=bg_scale_extra, dim_factor=dim_factor, words_per_caption=words_per_caption, effect_settings=effect_settings, pre_rendered_fg_path=temp_fg, mirror_video=mirror_video, target_duration=target_duration, original_video_path=video_path, crop_top_ratio=custom_top_ratio, crop_bottom_ratio=custom_bottom_ratio, caption_text_color=caption_text_color, caption_stroke_color=caption_stroke_color, caption_stroke_width=caption_stroke_width, caption_font_size=caption_font_size, caption_y_offset=caption_y_offset)
        if ok:
            log(f"Job finished successfully. Output: {output_path}")
        else:
            log("Job finished with errors.")
    except InterruptedError:
        log("[STOP] ⛔ Job was stopped by user.")
    except Exception as e:
        q.put(f"Exception: {e}")
        import traceback
        q.put(traceback.format_exc())
    finally:
            # Robust cleanup: close any MoviePy/AudioClip objects we created and remove temporary directories.
        def _safe_close(obj, name=None):
            try:
                if obj is None:
                    return
                if hasattr(obj, "close"):
                    obj.close()
            except Exception as e:
                try:
                    q.put(f"[cleanup] Failed to close {name or str(obj)}: {e}")
                except Exception:
                    pass
    
        try:
            _safe_close(globals().get("fg_clip") if "fg_clip" in globals() else locals().get("fg_clip"), "fg_clip")
            _safe_close(locals().get("fg_clip"), "fg_clip")
            _safe_close(locals().get("original_clip"), "original_clip")
            _safe_close(locals().get("voice_clip"), "voice_clip")
            _safe_close(locals().get("music_clip"), "music_clip")
            _safe_close(locals().get("music_matched"), "music_matched")
            _safe_close(locals().get("mixed_audio"), "mixed_audio")
            _safe_close(locals().get("synced_video"), "synced_video")
        except Exception:
            pass
    
        # Remove temp dir if it was created
        try:
            if 'temp_dir' in locals() and temp_dir and os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    q.put(f"[cleanup] Removed temp dir: {temp_dir}")
                except Exception as e_rm:
                    q.put(f"[cleanup] Could not remove temp dir {temp_dir}: {e_rm}")
        except Exception:
            pass
        
        # Remove temp voice file if it was created
        try:
            if 'temp_voice_path' in locals() and temp_voice_path and os.path.exists(temp_voice_path):
                try:
                    os.remove(temp_voice_path)
                    q.put(f"[cleanup] Removed temp voice file: {temp_voice_path}")
                except Exception as e_rm:
                    q.put(f"[cleanup] Could not remove temp voice file {temp_voice_path}: {e_rm}")
        except Exception:
            pass

    # restore stdout/stderr no matter what
    try:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    except Exception:
        pass
    
    # Restore IS_4K_MODE
    try:
        globals()['IS_4K_MODE'] = old_is_4k
    except Exception:
        pass


def _prepare_voice_for_job(job, job_index, total_jobs, q):
    """
    Prepare voice/TTS audio for a single job (Phase 1 of parallel pipeline).
    
    This extracts voice from video if needed, transcribes it, generates TTS audio,
    removes silences, and re-transcribes for perfect caption sync.
    
    Returns a dict with pre-generated voice data, or None if voice generation is
    not needed (AI voice disabled) or failed.
    
    The returned dict contains:
        - compressed_tts_path: Path to silence-removed TTS audio file
        - caption_segments: Caption segments from re-transcribed TTS audio
        - tts_duration: Duration of the final TTS audio
    """
    def log(s):
        q.put(str(s))
    
    use_ai_voice = job.get("use_ai_voice", False)
    if not use_ai_voice:
        log(f"[VOICE PREP {job_index}/{total_jobs}] AI voice not enabled — skipping parallel voice generation")
        return None
    
    video_path = job["video"]
    voice_path = job.get("voice", "")
    
    # Determine language settings
    target_language = job.get("target_language", 'none')
    translation_enabled = job.get("translation_enabled", False)
    tts_language = job.get("tts_language", 'en')
    silence_threshold_ms = job.get("silence_threshold_ms", 300)
    
    # Auto-sync TTS language with translation target language
    if translation_enabled and use_ai_voice and target_language and target_language != 'none':
        tts_lang_synced = TRANS_TO_TTS_LANG.get(target_language, target_language)
        tts_language = tts_lang_synced
    
    log(f"")
    log(f"[VOICE PREP {job_index}/{total_jobs}] 🎤 Starting voice generation for: {os.path.basename(video_path)}")
    
    temp_voice_path = None
    try:
        # Step 1: Extract voice from video if not provided
        actual_voice_path = voice_path
        if not actual_voice_path or not os.path.exists(actual_voice_path):
            log(f"[VOICE PREP {job_index}/{total_jobs}] Extracting audio from video...")
            fd, temp_voice_path = tempfile.mkstemp(suffix='.mp3', prefix=f'voice_prep_{job_index}_')
            os.close(fd)
            try:
                video_clip_for_audio = VideoFileClip(video_path)
                if video_clip_for_audio.audio is None:
                    log(f"[VOICE PREP {job_index}/{total_jobs}] ⚠️ Video has no audio track!")
                    video_clip_for_audio.close()
                    return None
                video_clip_for_audio.audio.write_audiofile(temp_voice_path, logger=None)
                actual_voice_path = temp_voice_path
                log(f"[VOICE PREP {job_index}/{total_jobs}] ✓ Extracted audio from video")
                video_clip_for_audio.close()
            except Exception as e:
                log(f"[VOICE PREP {job_index}/{total_jobs}] ❌ Failed to extract audio: {e}")
                return None
        
        # Step 2: Transcribe original audio to get text for TTS
        log(f"[VOICE PREP {job_index}/{total_jobs}] 📝 Transcribing original audio...")
        caption_segments = transcribe_captions(
            actual_voice_path,
            log,
            translate_to=target_language if translation_enabled else None
        )
        
        if not caption_segments:
            log(f"[VOICE PREP {job_index}/{total_jobs}] ⚠️ No caption segments from transcription")
            return None
        
        log(f"[VOICE PREP {job_index}/{total_jobs}] ✓ Got {len(caption_segments)} caption segments")
        
        # Step 3: Generate TTS audio
        log(f"[VOICE PREP {job_index}/{total_jobs}] 🎵 Generating AI voice (TTS language: {tts_language})...")
        tts_audio_path = replace_voice_with_tts(
            caption_segments,
            language=tts_language,
            log=log
        )
        
        if not tts_audio_path:
            log(f"[VOICE PREP {job_index}/{total_jobs}] ❌ TTS generation failed")
            return None
        
        log(f"[VOICE PREP {job_index}/{total_jobs}] ✓ TTS audio generated")
        
        # Step 4: Remove silences from TTS
        log(f"[VOICE PREP {job_index}/{total_jobs}] 🔇 Removing silences (threshold: {silence_threshold_ms}ms)...")
        compressed_tts_path, silence_map = remove_silence_from_audio(
            tts_audio_path,
            output_path=None,
            log=log,
            min_silence_ms=silence_threshold_ms
        )
        
        # Step 5: Re-transcribe from silence-removed TTS for perfect caption sync
        log(f"[VOICE PREP {job_index}/{total_jobs}] 📝 Re-transcribing from compressed TTS...")
        final_caption_segments = transcribe_captions(
            compressed_tts_path,
            log,
            translate_to=None  # Already translated during TTS generation
        )
        
        # Step 5.5: Map translated text onto re-transcription segments.
        # Whisper may auto-detect the wrong language; replace with correct translation.
        if caption_segments and final_caption_segments:
            if len(final_caption_segments) == len(caption_segments):
                for re_seg, orig_seg in zip(final_caption_segments, caption_segments):
                    re_seg['text'] = orig_seg.get('text', re_seg.get('text', ''))
                    if 'original_text' in orig_seg:
                        re_seg['original_text'] = orig_seg['original_text']
            else:
                all_translated_words = []
                for seg in caption_segments:
                    all_translated_words.extend(seg.get('text', '').split())
                if all_translated_words:
                    total_words = len(all_translated_words)
                    n_segs = len(final_caption_segments)
                    base_per_seg = total_words // n_segs
                    remainder = total_words % n_segs
                    word_idx = 0
                    for i, seg in enumerate(final_caption_segments):
                        count = base_per_seg + (1 if i < remainder else 0)
                        seg['text'] = ' '.join(all_translated_words[word_idx:word_idx + count])
                        word_idx += count
                else:
                    final_caption_segments = []
            # Filter segments with empty text (prevents original-language leaks)
            final_caption_segments = [s for s in final_caption_segments if s.get('text', '').strip()]
        elif caption_segments and not final_caption_segments:
            final_caption_segments = caption_segments
        
        # Get TTS duration (do NOT extend last caption — let video play
        # silently past the last spoken word so no foreign-language text leaks).
        from moviepy.editor import AudioFileClip as _AudioFileClip
        tts_clip_probe = _AudioFileClip(compressed_tts_path)
        tts_duration = tts_clip_probe.duration
        tts_clip_probe.close()
        
        log(f"[VOICE PREP {job_index}/{total_jobs}] ✅ Voice ready! Duration: {tts_duration:.2f}s, Captions: {len(final_caption_segments)}")
        
        return {
            'compressed_tts_path': compressed_tts_path,
            'caption_segments': final_caption_segments,
            'tts_duration': tts_duration,
            'original_tts_path': tts_audio_path,
        }
    except Exception as e:
        log(f"[VOICE PREP {job_index}/{total_jobs}] ❌ Voice preparation failed: {e}")
        import traceback
        log(traceback.format_exc())
        return None
    finally:
        # Clean up temp voice file (but NOT the TTS files — those are needed by the video step)
        if temp_voice_path and os.path.exists(temp_voice_path):
            try:
                os.remove(temp_voice_path)
            except Exception:
                pass


def _submit_voice_for_job(job, job_index, total_jobs, q, release_whisper=True):
    """
    Phase 1 of non-blocking voice pipeline: Extract audio, transcribe, submit TTS.
    Does NOT wait for TTS completion — returns immediately after API submission.

    Args:
        release_whisper: If True (default), call _release_whisper_model() after transcription
            to free GPU memory.  Set to False when running multiple concurrent submitters so
            only the caller (which knows when ALL submitters are done) releases the model.
    
    Returns a submission dict with task info, or None if not needed/failed.
    The dict contains:
        - needs_polling: True if GenAI Pro task was submitted (needs polling later)
        - task_id, headers: GenAI Pro task info (only if needs_polling)
        - tts_audio_path: Path to already-generated TTS audio (only if NOT needs_polling)
        - caption_segments: Transcribed caption segments
        - tts_language: Language used for TTS
        - silence_threshold_ms: Silence threshold for post-processing
    """
    def log(s):
        q.put(str(s))

    use_ai_voice = job.get("use_ai_voice", False)
    if not use_ai_voice:
        log(f"[VOICE SUBMIT {job_index}/{total_jobs}] AI voice not enabled — skipping")
        return None

    video_path = job["video"]
    voice_path = job.get("voice", "")

    target_language = job.get("target_language", 'none')
    translation_enabled = job.get("translation_enabled", False)
    tts_language = job.get("tts_language", 'en')
    silence_threshold_ms = job.get("silence_threshold_ms", 300)

    if translation_enabled and use_ai_voice and target_language and target_language != 'none':
        tts_lang_synced = TRANS_TO_TTS_LANG.get(target_language, target_language)
        tts_language = tts_lang_synced

    log(f"")
    log(f"[VOICE SUBMIT {job_index}/{total_jobs}] 🎤 Preparing: {os.path.basename(video_path)}")

    temp_voice_path = None
    try:
        # Step 1: Extract voice from video if not provided
        actual_voice_path = voice_path
        if not actual_voice_path or not os.path.exists(actual_voice_path):
            log(f"[VOICE SUBMIT {job_index}/{total_jobs}] Extracting audio from video...")
            fd, temp_voice_path = tempfile.mkstemp(suffix='.mp3', prefix=f'voice_submit_{job_index}_')
            os.close(fd)
            try:
                video_clip_for_audio = VideoFileClip(video_path)
                if video_clip_for_audio.audio is None:
                    log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ⚠️ Video has no audio track!")
                    video_clip_for_audio.close()
                    return None
                video_clip_for_audio.audio.write_audiofile(temp_voice_path, logger=None)
                actual_voice_path = temp_voice_path
                log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ✓ Extracted audio")
                video_clip_for_audio.close()
            except Exception as e:
                log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ❌ Audio extraction failed: {e}")
                return None

        # Step 2: Transcribe original audio (GPU) then translate (CPU/network separately)
        # Split into two steps so we can release the Whisper model (~3GB GPU) before
        # potentially long OpenAI translation retries (which are CPU/network only).
        # This prevents 99% GPU usage during translation timeouts.
        log(f"[VOICE SUBMIT {job_index}/{total_jobs}] 📝 Transcribing audio...")
        caption_segments = transcribe_captions(
            actual_voice_path,
            log,
            translate_to=None  # Translate separately after releasing GPU
        )

        # Release Whisper model to free ~3GB GPU memory before translation.
        # Translation is CPU/network only and can take minutes if OpenAI rate-limits.
        # The model will be reloaded on next transcribe_captions() call if needed.
        # When running multiple concurrent submitters, the caller handles the release
        # once all submitters are done (release_whisper=False).
        if release_whisper:
            _release_whisper_model(log=log)

        # Now translate if needed (no GPU required)
        if translation_enabled and target_language and target_language != 'none' and caption_segments:
            log(f"[VOICE SUBMIT {job_index}/{total_jobs}] 🌐 Translating to {target_language}...")
            caption_segments = translate_segments(caption_segments, target_language=target_language, log=log)

        if not caption_segments:
            log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ⚠️ No transcription results")
            return None

        log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ✓ Got {len(caption_segments)} segments")

        # Step 3: Submit TTS (non-blocking for GenAI Pro)
        full_text = " ".join([seg.get("text", "") for seg in caption_segments])

        # Try GenAI Pro API key
        api_key = _get_genaipro_api_key()

        if api_key:
            # Submit to GenAI Pro (fast HTTP POST, no waiting for result)
            log(f"[VOICE SUBMIT {job_index}/{total_jobs}] 🚀 Submitting to GenAI Pro...")
            task_id, headers = _submit_genaipro_task(full_text, tts_language, api_key, log)
            if task_id:
                log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ✅ Submitted! Task: {task_id[:12]}... → moving to next job")
                return {
                    'needs_polling': True,
                    'task_id': task_id,
                    'headers': headers,
                    'api_key': api_key,
                    'caption_segments': caption_segments,
                    'tts_language': tts_language,
                    'silence_threshold_ms': silence_threshold_ms,
                    'full_text': full_text,
                }
            # --- Fallback voice retry for submission ---
            fallback_id = globals().get('TTS_FALLBACK_VOICE_ID', 'auto')
            primary_id = globals().get('TTS_VOICE_ID', 'auto')
            if fallback_id and fallback_id != 'auto' and fallback_id != primary_id:
                log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ⚠️ PRIMARY VOICE FAILED — trying fallback voice...")
                old_voice = globals().get('TTS_VOICE_ID')
                globals()['TTS_VOICE_ID'] = fallback_id
                try:
                    task_id, headers = _submit_genaipro_task(full_text, tts_language, api_key, log)
                finally:
                    globals()['TTS_VOICE_ID'] = old_voice
                if task_id:
                    log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ✅ Fallback voice submitted! Task: {task_id[:12]}...")
                    return {
                        'needs_polling': True,
                        'task_id': task_id,
                        'headers': headers,
                        'api_key': api_key,
                        'caption_segments': caption_segments,
                        'tts_language': tts_language,
                        'silence_threshold_ms': silence_threshold_ms,
                        'full_text': full_text,
                    }
            log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ❌ GenAI Pro submission failed (primary + fallback)")

        log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ❌ Voice generation failed — please check your GenAI Pro API key in tts_config.json")
        return None

    except Exception as e:
        log(f"[VOICE SUBMIT {job_index}/{total_jobs}] ❌ Submission failed: {e}")
        import traceback
        log(traceback.format_exc())
        return None
    finally:
        if temp_voice_path and os.path.exists(temp_voice_path):
            try:
                os.remove(temp_voice_path)
            except Exception:
                pass


def _complete_voice_for_job(submission, job_index, total_jobs, q):
    """
    Phase 2 of non-blocking voice pipeline: Poll for TTS completion (if GenAI Pro),
    remove silences, re-transcribe for perfect caption sync.
    
    Returns voice data dict (same format as _prepare_voice_for_job) or None.
    """
    def log(s):
        q.put(str(s))

    if not submission:
        return None

    try:
        silence_threshold_ms = submission.get('silence_threshold_ms', 300)

        # Step 1: Get TTS audio path
        if submission.get('needs_polling'):
            # Poll GenAI Pro for completion (this is the waiting part)
            log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ⏳ Waiting for GenAI Pro...")
            tts_audio_path = _poll_and_download_genaipro(
                submission['task_id'],
                submission['headers'],
                output_path=None,
                log=log
            )
            if not tts_audio_path:
                # --- Fallback voice retry for async path ---
                fallback_id = globals().get('TTS_FALLBACK_VOICE_ID', 'auto')
                primary_id = globals().get('TTS_VOICE_ID', 'auto')
                if fallback_id and fallback_id != 'auto' and fallback_id != primary_id:
                    log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ⚠️ PRIMARY VOICE FAILED — trying fallback voice ({fallback_id[:12]}...)")
                    old_voice = globals().get('TTS_VOICE_ID')
                    globals()['TTS_VOICE_ID'] = fallback_id
                    try:
                        fb_task_id, fb_headers = _submit_genaipro_task(
                            submission.get('full_text', ''),
                            submission.get('tts_language', 'en'),
                            submission.get('api_key'),
                            log
                        )
                        if fb_task_id:
                            log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ⏳ Waiting for fallback voice...")
                            tts_audio_path = _poll_and_download_genaipro(fb_task_id, fb_headers, output_path=None, log=log)
                    finally:
                        globals()['TTS_VOICE_ID'] = old_voice
                if not tts_audio_path:
                    log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ❌ GenAI Pro failed (primary + fallback)")
                    return None
        else:
            tts_audio_path = submission.get('tts_audio_path')

        if not tts_audio_path:
            return None

        log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ✓ TTS audio ready")

        # Step 2: Remove silences
        log(f"[VOICE COMPLETE {job_index}/{total_jobs}] 🔇 Removing silences (threshold: {silence_threshold_ms}ms)...")
        compressed_tts_path, silence_map = remove_silence_from_audio(
            tts_audio_path,
            output_path=None,
            log=log,
            min_silence_ms=silence_threshold_ms
        )

        # Step 3: Re-transcribe from silence-removed TTS for perfect caption sync
        log(f"[VOICE COMPLETE {job_index}/{total_jobs}] 📝 Re-transcribing compressed audio...")
        final_caption_segments = transcribe_captions(
            compressed_tts_path,
            log,
            translate_to=None
        )

        # Step 3b: Preserve the original translated text from the submission phase.
        # Re-transcription gives us perfect timing sync with the silence-removed audio,
        # but Whisper may auto-detect the wrong language and produce source-language text
        # instead of the target-language text. The caption_segments from _submit_voice_for_job()
        # contain the correct OpenAI/googletrans translated text — map it onto re-timed segments.
        original_translated = submission.get('caption_segments', [])
        if original_translated and final_caption_segments:
            if len(final_caption_segments) == len(original_translated):
                for re_seg, orig_seg in zip(final_caption_segments, original_translated):
                    re_seg['text'] = orig_seg.get('text', re_seg.get('text', ''))
                    if 'original_text' in orig_seg:
                        re_seg['original_text'] = orig_seg['original_text']
                # Filter out segments with empty translated text (prevents
                # original-language Whisper text from leaking through).
                final_caption_segments = [s for s in final_caption_segments if s.get('text', '').strip()]
                log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ✓ Mapped translated text to {len(final_caption_segments)} re-timed segments (1:1)")
            else:
                log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ⚠ Segment count changed ({len(original_translated)} translated → {len(final_caption_segments)} re-transcribed) — redistributing translated text")
                # Segment counts differ: merge all translated text and redistribute
                # across the re-transcribed timing for correct caption display
                all_translated_words = []
                for seg in original_translated:
                    words = seg.get('text', '').split()
                    all_translated_words.extend(words)
                if all_translated_words:
                    total_words = len(all_translated_words)
                    n_segs = len(final_caption_segments)
                    base_per_seg = total_words // n_segs
                    remainder = total_words % n_segs
                    word_idx = 0
                    for i, seg in enumerate(final_caption_segments):
                        count = base_per_seg + (1 if i < remainder else 0)
                        seg['text'] = ' '.join(all_translated_words[word_idx:word_idx + count])
                        word_idx += count
                    # Remove segments that received no translated words — they
                    # still contain the original-language Whisper text and would
                    # appear as untranslated captions at the end of the video.
                    final_caption_segments = [s for s in final_caption_segments if s.get('text', '').strip()]
                    log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ✓ Distributed {total_words} translated words across {len(final_caption_segments)} re-timed segments")
                else:
                    # All translated text was empty — clear re-transcription
                    # segments so no original-language text leaks through.
                    log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ⚠ No translated words available — clearing re-transcribed segments")
                    final_caption_segments = []
        elif original_translated and not final_caption_segments:
            # Re-transcription produced nothing — fall back to original translated segments
            log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ⚠ Re-transcription empty — using original translated segments")
            final_caption_segments = original_translated

        # Get TTS duration (do NOT extend last caption — let video play
        # silently past the last spoken word so no foreign-language text leaks).
        from moviepy.editor import AudioFileClip as _AudioFileClip
        tts_clip_probe = _AudioFileClip(compressed_tts_path)
        tts_duration = tts_clip_probe.duration
        tts_clip_probe.close()

        log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ✅ Voice ready! Duration: {tts_duration:.2f}s, Captions: {len(final_caption_segments)}")

        # Release Whisper model after re-transcription to free GPU memory
        _release_whisper_model(log=log)

        return {
            'compressed_tts_path': compressed_tts_path,
            'caption_segments': final_caption_segments,
            'tts_duration': tts_duration,
            'original_tts_path': tts_audio_path,
        }
    except Exception as e:
        log(f"[VOICE COMPLETE {job_index}/{total_jobs}] ❌ Completion failed: {e}")
        import traceback
        log(traceback.format_exc())
        # Release Whisper model even on failure
        _release_whisper_model(log=log)
        return None


def _extract_effect_settings(job):
    """Extract effect_settings dict from a job dict."""
    return {
        'effect_sharpness': job.get("effect_sharpness", False),
        'effect_sharpness_intensity': job.get("effect_sharpness_intensity", 1.5),
        'effect_saturation': job.get("effect_saturation", False),
        'effect_saturation_intensity': job.get("effect_saturation_intensity", 1.3),
        'effect_contrast': job.get("effect_contrast", False),
        'effect_contrast_intensity': job.get("effect_contrast_intensity", 1.2),
        'effect_brightness': job.get("effect_brightness", False),
        'effect_brightness_intensity': job.get("effect_brightness_intensity", 1.15),
        'effect_vintage': job.get("effect_vintage", False),
        'effect_vintage_intensity': job.get("effect_vintage_intensity", 0.3),
        'blur_overlay_enabled': job.get("blur_overlay_enabled", False),
        'blur_overlay_x': job.get("blur_overlay_x", 10.0),
        'blur_overlay_y': job.get("blur_overlay_y", 10.0),
        'blur_overlay_w': job.get("blur_overlay_w", 20.0),
        'blur_overlay_h': job.get("blur_overlay_h", 15.0),
        'blur_overlay_intensity': job.get("blur_overlay_intensity", 20)
    }


def _run_video_job(job, job_index, total_jobs, q, pre_generated_voice=None):
    """Run a single video processing job (Phase 2 of parallel pipeline)."""
    def log(s):
        q.put(str(s))
    log(f"\n===== START JOB {job_index}/{total_jobs} =====")
    # Release Whisper model before export to free GPU memory for NVENC encoding.
    # IMPORTANT: Skip when pre_generated_voice is provided (parallel pipeline).
    # In the parallel pipeline, completion workers call _release_whisper_model()
    # themselves after each re-transcription.  Calling it here would block the
    # main (export) thread on _whisper_transcription_lock while another job's
    # completion worker is re-transcribing, causing ALL voices to finish before
    # ANY export starts — the opposite of the desired interleaved behaviour.
    # For the single-job / no-pre-voice path, release normally so FFmpeg NVENC
    # gets the GPU memory it needs.
    if not pre_generated_voice:
        _release_whisper_model(log=log)
    effect_settings = _extract_effect_settings(job)
    process_single_job(job["video"], job["voice"], job["music"], job["output"], q, job.get("font"),
                       custom_top_ratio=job.get("custom_top_ratio"),
                       custom_bottom_ratio=job.get("custom_bottom_ratio"),
                       mirror_video=job.get("mirror_video", False),
                       words_per_caption=job.get("words_per_caption", 2),
                       use_4k=job.get("use_4k", False),
                       blur_radius=job.get("blur_radius"),
                       bg_scale_extra=job.get("bg_scale_extra"),
                       dim_factor=job.get("dim_factor"),
                       effect_settings=effect_settings,
                       use_ai_voice=job.get("use_ai_voice", False),
                       target_language=job.get("target_language", 'none'),
                       translation_enabled=job.get("translation_enabled", False),
                       tts_language=job.get("tts_language", 'en'),
                       silence_threshold_ms=job.get("silence_threshold_ms", 300),
                       caption_text_color=job.get("caption_text_color"),
                       caption_stroke_color=job.get("caption_stroke_color"),
                       caption_stroke_width=job.get("caption_stroke_width"),
                       caption_font_size=job.get("caption_font_size"),
                       caption_y_offset=job.get("caption_y_offset"),
                       pre_generated_voice=pre_generated_voice,
                       voice_gain=job.get("voice_gain"),
                       music_gain=job.get("music_gain"))
    log(f"===== END JOB {job_index} =====\n")


def queue_worker(jobs, q):
    def log(s):
        q.put(str(s))
    log(f"[QUEUE] Starting queue with {len(jobs)} job(s).")
    
    # Check if any jobs use AI voice — if so, use non-blocking voice pipeline
    any_ai_voice = any(job.get("use_ai_voice", False) for job in jobs)
    
    if not any_ai_voice or len(jobs) <= 1:
        # No AI voice jobs or single job — use simple sequential processing
        for i, job in enumerate(jobs, start=1):
            _run_video_job(job, i, len(jobs), q)
        # Final cleanup: release Whisper model to free GPU memory
        _release_whisper_model(log=log)
        log("[QUEUE_DONE]")
        return
    
    # === INTERLEAVED NON-BLOCKING VOICE PIPELINE ===
    # Voice submissions and completions run concurrently:
    #   - Up to VOICE_SUBMISSION_CONCURRENCY submitter threads run simultaneously:
    #     each extracts audio, transcribes (serialised via Whisper lock) and POSTs
    #     to GenAI Pro, then immediately starts a completion thread.
    #   - Completion threads: poll GenAI + silence removal + re-transcribe (all parallel)
    #   - Main thread: processes videos as soon as each voice becomes ready (any order)
    
    total = len(jobs)
    
    log("")
    log("━"*60)
    log("[QUEUE] 🚀 INTERLEAVED NON-BLOCKING VOICE PIPELINE")
    log(f"[QUEUE] {total} jobs — submitting voices & processing videos concurrently")
    concurrency_preview = max(1, globals().get('VOICE_SUBMISSION_CONCURRENCY', 3))
    log(f"[QUEUE] Concurrent voice submissions: {concurrency_preview} | Completions: parallel")
    log("━"*60)
    log("")
    
    voice_results = [None] * total
    voice_done_events = [threading.Event() for _ in range(total)]
    any_voice_ready = threading.Event()
    completion_threads = []
    completion_threads_lock = threading.Lock()
    
    def _completion_worker(idx, submission):
        """Thread: poll for TTS completion + post-process (silence removal + re-transcribe)."""
        try:
            if submission is None:
                voice_results[idx] = None
            else:
                voice_results[idx] = _complete_voice_for_job(submission, idx + 1, total, q)
        except Exception as e:
            q.put(f"[VOICE COMPLETE {idx+1}/{total}] ❌ Unexpected error: {e}")
            voice_results[idx] = None
        finally:
            voice_done_events[idx].set()
            any_voice_ready.set()

    # ── Concurrent submission pool ──────────────────────────────────────────
    # Up to VOICE_SUBMISSION_CONCURRENCY jobs are submitted to GenAI Pro at
    # the same time.  Whisper transcription still serialises internally (GPU
    # is not thread-safe), but audio extraction and TTS API POSTs overlap.
    concurrency = max(1, globals().get('VOICE_SUBMISSION_CONCURRENCY', 3))
    _submit_sem = threading.Semaphore(concurrency)
    # Counts how many submitter threads are still running (used to know when
    # the Whisper model can be released).
    _active_submitters_lock = threading.Lock()
    _active_submitters = [0]        # mutable int inside a list for closure

    log(f"[QUEUE] ⚡ Concurrent voice submissions: {concurrency} at a time")

    def _submit_one(idx):
        """Thread: submit one voice job (bounded by _submit_sem semaphore)."""
        with _submit_sem:
            try:
                job = jobs[idx]

                if not job.get("use_ai_voice", False):
                    log(f"[QUEUE] Job {idx+1}/{total}: no AI voice — ready immediately")
                    voice_done_events[idx].set()
                    any_voice_ready.set()
                    return

                log(f"\n[QUEUE] 📤 Submitting voice {idx+1}/{total}...")
                # release_whisper=False: let the orchestrator release the model
                # once ALL submitters have finished (avoids reloading between jobs).
                sub = _submit_voice_for_job(job, idx + 1, total, q, release_whisper=False)

                if sub is None:
                    voice_done_events[idx].set()
                    any_voice_ready.set()
                else:
                    t = threading.Thread(target=_completion_worker, args=(idx, sub))
                    with completion_threads_lock:
                        completion_threads.append(t)
                    t.start()
                    log(f"[QUEUE] ✓ Job {idx+1} completion thread started")

            except Exception as _err:
                import traceback as _tb
                log(f"[QUEUE SUBMITTER {idx+1}] ❌ Error: {_err}")
                log(_tb.format_exc())
                if not voice_done_events[idx].is_set():
                    voice_done_events[idx].set()
                any_voice_ready.set()
            finally:
                with _active_submitters_lock:
                    _active_submitters[0] -= 1
                    remaining = _active_submitters[0]
                # When the last submitter finishes, release Whisper model
                if remaining == 0:
                    log(f"\n[QUEUE] ═══ ALL {total} VOICES SUBMITTED ═══")
                    _release_whisper_model(log=log)

    # Launch all submitter threads (semaphore limits actual concurrency)
    submit_threads = []
    with _active_submitters_lock:
        _active_submitters[0] = total
    for _idx in range(total):
        _t = threading.Thread(target=_submit_one, args=(_idx,))
        submit_threads.append(_t)
        _t.start()

    # ── legacy alias so the join below still works ──────────────────────────
    def _dummy_join():
        for _t in submit_threads:
            _t.join()
    submitter_thread = threading.Thread(target=_dummy_join)
    submitter_thread.start()

    # Process videos as each voice becomes ready (main thread)
    processed = [False] * total
    processed_count = 0
    
    while processed_count < total:
        # Find the first unprocessed job whose voice is ready
        ready_idx = None
        
        # Quick scan for any already-done voice
        for idx in range(total):
            if not processed[idx] and voice_done_events[idx].is_set():
                ready_idx = idx
                break
        
        if ready_idx is None:
            # No voice is ready yet — wait for the shared signal
            any_voice_ready.wait()
            any_voice_ready.clear()
            # Now scan to find which one(s) finished
            for idx in range(total):
                if not processed[idx] and voice_done_events[idx].is_set():
                    ready_idx = idx
                    break
        
        if ready_idx is None:
            # Defensive: should not happen, but avoid infinite loop
            continue
        
        # Process this job's video
        job = jobs[ready_idx]
        voice_data = voice_results[ready_idx]
        
        if voice_data:
            log(f"\n[QUEUE] 🎬 Voice for job {ready_idx + 1} is ready — starting video!")
        else:
            log(f"\n[QUEUE] 🎬 Job {ready_idx + 1} ready — starting video (no pre-gen voice)")
        
        _run_video_job(job, ready_idx + 1, total, q, pre_generated_voice=voice_data)
        
        processed[ready_idx] = True
        processed_count += 1
        log(f"[QUEUE] ✓ Completed {processed_count}/{total} jobs")
    
    # Wait for submitter and all completion threads to finish
    submitter_thread.join()
    with completion_threads_lock:
        threads_to_join = list(completion_threads)
    for t in threads_to_join:
        t.join(timeout=10.0)
        if t.is_alive():
            log(f"[QUEUE] ⚠️ Completion thread still running after timeout")
    
    # Final cleanup: release Whisper model to free GPU memory
    _release_whisper_model(log=log)
    
    log("")
    log("━"*60)
    log("[QUEUE] ✅ ALL JOBS COMPLETE (interleaved voice pipeline)")
    log("━"*60)
    log("")
    log("[QUEUE_DONE]")

# ----------------- GUI: responsive layout with PanedWindow -----------------
def seconds_to_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

class App:
    # --- Modern Dark Theme Color Palette ---
    BG_DARK = '#0d1117'          # Main background (GitHub-dark inspired)
    BG_SECONDARY = '#161b22'     # Secondary panels / cards
    BG_TERTIARY = '#1c2333'      # Input fields / entries
    BG_ELEVATED = '#21262d'      # Elevated surfaces (buttons, headers)
    BORDER_COLOR = '#30363d'     # Subtle borders
    BORDER_LIGHT = '#3d444d'     # Lighter borders for focus
    TEXT_PRIMARY = '#e6edf3'     # Primary text (soft white)
    TEXT_SECONDARY = '#8b949e'   # Secondary / hint text
    TEXT_MUTED = '#6e7681'       # Muted text
    ACCENT_BLUE = '#58a6ff'      # Primary accent (links, active)
    ACCENT_GREEN = '#3fb950'     # Success / Run buttons
    ACCENT_PURPLE = '#bc8cff'    # Secondary accent
    ACCENT_RED = '#f85149'       # Danger / Remove
    ACCENT_ORANGE = '#d29922'    # Warning / highlight
    SELECT_BG = '#1f6feb'        # Selection background
    HOVER_BG = '#292e36'         # Hover state
    PROGRESS_BG = '#58a6ff'      # Progress bar fill

    def __init__(self, root):
        self.root = root
        root.title("TikTok Auto Studio")
        # --- Lock system for scroll protection ---
        self._lock_vars = {}           # {lock_key: BooleanVar}
        self._lock_labels = {}         # {lock_key: label widget}
        self._lock_target_widgets = {} # {lock_key: (widget, is_combo_readonly)}
        self._left_canvas = None       # set later during UI build
        # --- Apply modern dark UI theme ---
        try:
            style = ttk.Style()
            try:
                style.theme_use('clam')
            except Exception:
                pass

            # Core widget styles
            style.configure('TFrame', background=self.BG_DARK)
            style.configure('TLabel', background=self.BG_DARK, foreground=self.TEXT_PRIMARY,
                          font=('Segoe UI', 9))

            # --- Base TButton: visible border + hover highlight ---
            style.configure('TButton', background=self.BG_ELEVATED, foreground=self.TEXT_PRIMARY,
                          borderwidth=1, bordercolor=self.BORDER_COLOR,
                          focuscolor=self.ACCENT_BLUE, relief='solid',
                          font=('Segoe UI', 9), padding=(10, 4))
            style.map('TButton',
                      background=[('active', self.HOVER_BG), ('pressed', self.BG_TERTIARY)],
                      foreground=[('disabled', self.TEXT_MUTED)],
                      bordercolor=[('active', self.ACCENT_BLUE), ('focus', self.ACCENT_BLUE)])

            # --- TEntry: visible border + focus glow ---
            style.configure('TEntry', fieldbackground=self.BG_TERTIARY, foreground=self.TEXT_PRIMARY,
                          borderwidth=1, bordercolor=self.BORDER_COLOR, lightcolor=self.BORDER_COLOR,
                          darkcolor=self.BORDER_COLOR, padding=4, insertcolor=self.TEXT_PRIMARY)
            style.map('TEntry',
                      fieldbackground=[('focus', '#1c2333'), ('readonly', self.BG_SECONDARY)],
                      bordercolor=[('focus', self.ACCENT_BLUE)],
                      lightcolor=[('focus', self.ACCENT_BLUE)],
                      darkcolor=[('focus', self.ACCENT_BLUE)])

            style.configure('TCheckbutton', background=self.BG_DARK, foreground=self.TEXT_PRIMARY,
                          font=('Segoe UI', 9), indicatormargin=4)
            style.map('TCheckbutton',
                      background=[('active', self.BG_DARK)],
                      indicatorcolor=[('selected', self.ACCENT_BLUE), ('!selected', self.BG_TERTIARY)])
            style.configure('TMenubutton', background=self.BG_ELEVATED, foreground=self.TEXT_PRIMARY,
                          borderwidth=1, bordercolor=self.BORDER_COLOR, padding=(8, 3))
            style.map('TMenubutton',
                      bordercolor=[('active', self.ACCENT_BLUE)])
            style.configure('Horizontal.TProgressbar', troughcolor=self.BG_SECONDARY,
                          background=self.PROGRESS_BG, borderwidth=0)

            # --- TCombobox: visible border + focus glow ---
            style.configure('TCombobox', fieldbackground=self.BG_TERTIARY, foreground=self.TEXT_PRIMARY,
                          selectbackground=self.SELECT_BG, selectforeground=self.TEXT_PRIMARY,
                          borderwidth=1, bordercolor=self.BORDER_COLOR, lightcolor=self.BORDER_COLOR,
                          darkcolor=self.BORDER_COLOR, padding=4, arrowcolor=self.TEXT_SECONDARY)
            style.map('TCombobox',
                      fieldbackground=[('readonly', self.BG_TERTIARY)],
                      selectbackground=[('readonly', self.SELECT_BG)],
                      bordercolor=[('focus', self.ACCENT_BLUE), ('active', self.ACCENT_BLUE)],
                      lightcolor=[('focus', self.ACCENT_BLUE)],
                      darkcolor=[('focus', self.ACCENT_BLUE)],
                      arrowcolor=[('active', self.ACCENT_BLUE)])

            # --- TSpinbox: visible border + focus glow ---
            style.configure('TSpinbox', fieldbackground=self.BG_TERTIARY, foreground=self.TEXT_PRIMARY,
                          borderwidth=1, bordercolor=self.BORDER_COLOR, lightcolor=self.BORDER_COLOR,
                          darkcolor=self.BORDER_COLOR, arrowcolor=self.TEXT_SECONDARY, padding=3)
            style.map('TSpinbox',
                      bordercolor=[('focus', self.ACCENT_BLUE), ('active', self.ACCENT_BLUE)],
                      lightcolor=[('focus', self.ACCENT_BLUE)],
                      darkcolor=[('focus', self.ACCENT_BLUE)],
                      arrowcolor=[('active', self.ACCENT_BLUE)])

            style.configure('TSeparator', background=self.BORDER_COLOR)
            style.configure('TScrollbar', background=self.BG_SECONDARY, troughcolor=self.BG_DARK,
                          borderwidth=0, arrowcolor=self.TEXT_MUTED)
            style.map('TScrollbar',
                      background=[('active', self.BG_ELEVATED)])
            style.configure('TPanedwindow', background=self.BG_DARK)
            style.configure('TLabelframe', background=self.BG_DARK, foreground=self.ACCENT_BLUE,
                          borderwidth=1, relief='groove')
            style.configure('TLabelframe.Label', background=self.BG_DARK, foreground=self.ACCENT_BLUE,
                          font=('Segoe UI', 10, 'bold'))

            # --- Accent button styles (with border) ---
            style.configure('Accent.TButton', background=self.ACCENT_BLUE, foreground='#ffffff',
                          borderwidth=1, bordercolor='#4090e0', relief='solid',
                          font=('Segoe UI', 9, 'bold'), padding=(12, 5))
            style.map('Accent.TButton',
                      background=[('active', '#4090e0'), ('pressed', '#3070c0')],
                      bordercolor=[('active', '#58a6ff'), ('pressed', '#3070c0')])

            style.configure('Success.TButton', background=self.ACCENT_GREEN, foreground='#ffffff',
                          borderwidth=1, bordercolor='#2ea043', relief='solid',
                          font=('Segoe UI', 9, 'bold'), padding=(12, 5))
            style.map('Success.TButton',
                      background=[('active', '#2ea043'), ('pressed', '#238636')],
                      bordercolor=[('active', '#3fb950'), ('pressed', '#238636')])

            style.configure('Danger.TButton', background=self.ACCENT_RED, foreground='#ffffff',
                          borderwidth=1, bordercolor='#da3633', relief='solid',
                          font=('Segoe UI', 9, 'bold'), padding=(12, 5))
            style.map('Danger.TButton',
                      background=[('active', '#da3633'), ('pressed', '#b62324')],
                      bordercolor=[('active', '#f85149'), ('pressed', '#b62324')])

            # --- Section header labels ---
            style.configure('SectionHeader.TLabel', background=self.BG_DARK,
                          foreground=self.ACCENT_BLUE, font=('Segoe UI', 10, 'bold'))
            style.configure('SubHeader.TLabel', background=self.BG_DARK,
                          foreground=self.TEXT_SECONDARY, font=('Segoe UI', 8))
            style.configure('Hint.TLabel', background=self.BG_DARK,
                          foreground=self.TEXT_MUTED, font=('Segoe UI', 8))
            style.configure('Emoji.TLabel', background=self.BG_DARK,
                          foreground=self.TEXT_PRIMARY, font=('Segoe UI', 11))

            # --- Warning button (orange – reset / caution actions) ---
            style.configure('Warning.TButton', background=self.ACCENT_ORANGE, foreground='#ffffff',
                          borderwidth=1, bordercolor='#b88a1e', relief='solid',
                          font=('Segoe UI', 9, 'bold'), padding=(12, 5))
            style.map('Warning.TButton',
                      background=[('active', '#b88a1e'), ('pressed', '#9e7519')],
                      bordercolor=[('active', '#d29922'), ('pressed', '#9e7519')])

            # --- Info button (purple – toggle / info actions) ---
            style.configure('Info.TButton', background=self.ACCENT_PURPLE, foreground='#ffffff',
                          borderwidth=1, bordercolor='#a371e0', relief='solid',
                          font=('Segoe UI', 9, 'bold'), padding=(12, 5))
            style.map('Info.TButton',
                      background=[('active', '#a371e0'), ('pressed', '#8957c0')],
                      bordercolor=[('active', '#bc8cff'), ('pressed', '#8957c0')])

            # --- Bordered secondary button (Browse, Choose, Custom, etc.) ---
            style.configure('Bordered.TButton', background=self.BG_ELEVATED,
                          foreground=self.TEXT_PRIMARY, borderwidth=1,
                          bordercolor=self.BORDER_LIGHT, relief='solid',
                          font=('Segoe UI', 9), padding=(10, 4))
            style.map('Bordered.TButton',
                      background=[('active', self.HOVER_BG), ('pressed', self.BG_TERTIARY)],
                      foreground=[('disabled', self.TEXT_MUTED)],
                      bordercolor=[('active', self.ACCENT_BLUE), ('focus', self.ACCENT_BLUE)])

        except Exception:
            pass

        try:
            root.configure(bg=self.BG_DARK)
        except Exception:
            pass

        # Helper to force dark colors for native tk widgets that ttk styles don't affect.
        app_ref = self
        def apply_dark_theme_to(root_widget):
            try:
                import tkinter as _tk
                from tkinter import ttk as _ttk
            except Exception:
                return

            def _apply(w):
                # set native tk widget colors
                try:
                    if isinstance(w, _tk.Listbox) or isinstance(w, _tk.Text):
                        w.config(bg=app_ref.BG_SECONDARY, fg=app_ref.TEXT_PRIMARY,
                                insertbackground=app_ref.TEXT_PRIMARY,
                                selectbackground=app_ref.SELECT_BG,
                                selectforeground='#ffffff',
                                highlightbackground=app_ref.BORDER_COLOR,
                                highlightcolor=app_ref.ACCENT_BLUE,
                                highlightthickness=1,
                                relief='solid', bd=1)
                    elif isinstance(w, _tk.Canvas):
                        # Don't override preview/color canvases that have specific bg
                        cur_bg = str(w.cget('bg'))
                        if cur_bg in ('#141414', '#0b0b0b'):
                            w.config(bg=app_ref.BG_SECONDARY)
                    elif isinstance(w, _tk.Scrollbar):
                        try:
                            w.config(bg=app_ref.BG_SECONDARY,
                                    troughcolor=app_ref.BG_DARK,
                                    activebackground=app_ref.BG_ELEVATED,
                                    highlightbackground=app_ref.BG_DARK,
                                    relief='flat', bd=0)
                        except Exception:
                            w.config(bg=app_ref.BG_SECONDARY)
                    elif isinstance(w, _tk.Entry):
                        try:
                            w.config(bg=app_ref.BG_TERTIARY, fg=app_ref.TEXT_PRIMARY,
                                    insertbackground=app_ref.TEXT_PRIMARY,
                                    relief='solid', bd=1,
                                    highlightthickness=1,
                                    highlightbackground=app_ref.BORDER_COLOR,
                                    highlightcolor=app_ref.ACCENT_BLUE)
                        except Exception:
                            pass
                    elif isinstance(w, _tk.Scale):
                        try:
                            w.config(bg=app_ref.BG_DARK, fg=app_ref.TEXT_PRIMARY,
                                    troughcolor=app_ref.BG_TERTIARY,
                                    activebackground=app_ref.ACCENT_BLUE,
                                    highlightbackground=app_ref.BORDER_COLOR,
                                    highlightcolor=app_ref.ACCENT_BLUE,
                                    highlightthickness=1,
                                    relief='flat', bd=0, sliderrelief='flat')
                        except Exception:
                            pass
                except Exception:
                    pass
                # recurse into children
                try:
                    for c in w.winfo_children():
                        _apply(c)
                except Exception:
                    pass

            _apply(root_widget)

        # schedule to apply after mainloop starts so all widgets exist
        try:
            root.after(150, lambda: apply_dark_theme_to(root))
        except Exception:
            pass

        except Exception:
            pass

        self.q = queue.Queue()

        try:
            root.state('zoomed')
        except Exception:
            pass

        pw = ttk.PanedWindow(root, orient="horizontal")
        pw.pack(fill="both", expand=True, padx=4, pady=4)
        try:
            pw.config(style='TPanedwindow')
            pw.config(bg=self.BG_DARK)
        except Exception:
            pass

        # Create a container frame for the left side with scrollbar
        left_container = ttk.Frame(pw)
        try:
            left_container.configure(style='TFrame')
            left_container.config(bg=self.BG_DARK)
        except Exception:
            pass
        
        # Create Canvas and Scrollbar for scrollable left panel
        left_canvas = tk.Canvas(left_container, bg=self.BG_DARK, highlightthickness=0)
        self._left_canvas = left_canvas
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        
        # Create the actual frame that will contain all controls
        left_frame = ttk.Frame(left_canvas, padding=10)
        try:
            left_frame.configure(style='TFrame')
            left_frame.config(bg=self.BG_DARK)
        except Exception:
            pass
        left_frame.columnconfigure(1, weight=1)
        
        # Pack scrollbar and canvas
        left_scrollbar.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)
        
        # Create window in canvas for the frame
        canvas_frame = left_canvas.create_window((0, 0), window=left_frame, anchor="nw")
        
        # Configure canvas scrolling
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        # Update scroll region when frame changes size
        def configure_scroll_region(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            # Also update the canvas window width to match canvas width
            canvas_width = left_canvas.winfo_width()
            left_canvas.itemconfig(canvas_frame, width=canvas_width)
        
        left_frame.bind("<Configure>", configure_scroll_region)
        left_canvas.bind("<Configure>", configure_scroll_region)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def bind_mousewheel(event):
            left_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            left_canvas.unbind_all("<MouseWheel>")
        
        left_canvas.bind('<Enter>', bind_mousewheel)
        left_canvas.bind('<Leave>', unbind_mousewheel)
        
        pw.add(left_container, weight=1)

        right_outer = ttk.PanedWindow(pw, orient="vertical")
        pw.add(right_outer, weight=2)

        preview_frame = ttk.Frame(right_outer, padding=10)
        try:
            preview_frame.configure(style='TFrame')
            preview_frame.config(bg=self.BG_DARK)
        except Exception:
            pass
        right_outer.add(preview_frame, weight=3)

        bottom_frame = ttk.Frame(right_outer, padding=10)
        try:
            bottom_frame.configure(style='TFrame')
            bottom_frame.config(bg=self.BG_DARK)
        except Exception:
            pass
        right_outer.add(bottom_frame, weight=2)

        # left controls
        row = 0

        # ══════════ FILE SELECTION ══════════
        ttk.Label(left_frame, text="📁  File Selection", style='SectionHeader.TLabel').grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        ttk.Label(left_frame, text="Video:").grid(row=row, column=0, sticky="w")
        self.video_var = tk.StringVar(value="")
        self.video_entry = ttk.Entry(left_frame, textvariable=self.video_var)
        self.video_entry.grid(row=row, column=1, sticky="we", padx=(6,0))
        ttk.Button(left_frame, text="Browse…", style='Bordered.TButton', command=self.browse_video).grid(row=row, column=2, padx=6)
        row += 1

        ttk.Label(left_frame, text="Voice:").grid(row=row, column=0, sticky="w")
        self.voice_var = tk.StringVar(value="")
        self.voice_entry = ttk.Entry(left_frame, textvariable=self.voice_var)
        self.voice_entry.grid(row=row, column=1, sticky="we", padx=(6,0))
        ttk.Button(left_frame, text="Browse…", style='Bordered.TButton', command=self.browse_voice).grid(row=row, column=2, padx=6)
        row += 1

        ttk.Label(left_frame, text="Music:").grid(row=row, column=0, sticky="w")
        self.music_var = tk.StringVar(value="")
        self.music_entry = ttk.Entry(left_frame, textvariable=self.music_var)
        self.music_entry.grid(row=row, column=1, sticky="we", padx=(6,0))
        ttk.Button(left_frame, text="Browse…", style='Bordered.TButton', command=self.browse_music).grid(row=row, column=2, padx=6)
        row += 1

        ttk.Label(left_frame, text="Output:").grid(row=row, column=0, sticky="w")
        self.output_var = tk.StringVar(value="final_tiktok.mp4")
        self.output_entry = ttk.Entry(left_frame, textvariable=self.output_var)
        self.output_entry.grid(row=row, column=1, sticky="we", padx=(6,0))
        ttk.Button(left_frame, text="Choose…", style='Bordered.TButton', command=self.choose_output).grid(row=row, column=2, padx=6)
        row += 1

        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        # ══════════ VIDEO SETTINGS ══════════
        ttk.Label(left_frame, text="⚙️  Video Settings", style='SectionHeader.TLabel').grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        self.use_custom_crop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Use custom crop for this job", variable=self.use_custom_crop_var).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        
        self.mirror_video_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Mirror video horizontally", variable=self.mirror_video_var).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        
        self.use_4k_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Export in 4K resolution (2160x3840)", variable=self.use_4k_var, command=self.on_4k_toggle).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # --- Voice Volume Control ---
        _lbl_voice = ttk.Label(left_frame, text="Voice volume:")
        _lbl_voice.grid(row=row, column=0, sticky="e")
        self.voice_gain_var = tk.DoubleVar(value=VOICE_GAIN)
        self.voice_gain_scale = tk.Scale(left_frame, from_=0.0, to=5.0, resolution=0.01, orient='horizontal', length=200, showvalue=0, variable=self.voice_gain_var, command=self.on_voice_gain_changed)
        self.voice_gain_scale.grid(row=row, column=1, padx=(6,0))
        self.voice_gain_label = ttk.Label(left_frame, text=_gain_to_display(self.voice_gain_var.get()), width=22)
        self.voice_gain_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self._register_lockable("voice_gain", _lbl_voice, self.voice_gain_scale)
        row += 1

        # --- Music Volume Control ---
        _lbl_music = ttk.Label(left_frame, text="Music volume:")
        _lbl_music.grid(row=row, column=0, sticky="e")
        self.music_gain_var = tk.DoubleVar(value=MUSIC_GAIN)
        self.music_gain_scale = tk.Scale(left_frame, from_=0.0, to=2.0, resolution=0.01, orient='horizontal', length=200, showvalue=0, variable=self.music_gain_var, command=self.on_music_gain_changed)
        self.music_gain_scale.grid(row=row, column=1, padx=(6,0))
        self.music_gain_label = ttk.Label(left_frame, text=_gain_to_display(self.music_gain_var.get()), width=22)
        self.music_gain_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self._register_lockable("music_gain", _lbl_music, self.music_gain_scale)
        row += 1

        # --- Preview Mix Button ---
        audio_preview_frame = ttk.Frame(left_frame)
        audio_preview_frame.grid(row=row, column=0, columnspan=3, sticky="we", pady=(4, 0))
        ttk.Button(audio_preview_frame, text="🔊 Preview Mix", style='Bordered.TButton', command=self._preview_audio_mix).pack(side="left", padx=(4, 8))
        ttk.Label(audio_preview_frame, text="Listen to voice + music at current volumes", font=('Segoe UI', 8)).pack(side="left")
        row += 1

        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        # ══════════ TRANSLATION & AI VOICE ══════════
        ttk.Label(left_frame, text="🎤  Translation & AI Voice", style='SectionHeader.TLabel').grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        self.translation_enabled_var = tk.BooleanVar(value=TRANSLATION_ENABLED)
        ttk.Checkbutton(left_frame, text="Enable translation", variable=self.translation_enabled_var, 
                       command=self.on_translation_toggle).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        _lbl_target_lang = ttk.Label(left_frame, text="Target language:")
        _lbl_target_lang.grid(row=row, column=0, sticky="e")
        self.target_language_var = tk.StringVar(value=TARGET_LANGUAGE)
        languages = ['none', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ro', 'ru', 'zh-cn', 'ja', 'ko']
        self.language_combo = ttk.Combobox(left_frame, textvariable=self.target_language_var, values=languages, state='readonly', width=10)
        self.language_combo.grid(row=row, column=1, sticky="w", padx=(6,0))
        self.language_combo.bind('<<ComboboxSelected>>', self.on_language_selected)
        ttk.Label(left_frame, text="(for captions)").grid(row=row, column=2, sticky='w', padx=(4,0))
        self._register_lockable("target_language", _lbl_target_lang, self.language_combo, is_combo_readonly=True)
        row += 1

        # OpenAI API Key for natural contextual translations
        ttk.Label(left_frame, text="OpenAI Key:").grid(row=row, column=0, sticky="e")
        openai_frame = ttk.Frame(left_frame)
        openai_frame.grid(row=row, column=1, columnspan=2, sticky="we", padx=(6,0))
        # Try to load saved OpenAI key and model from config file
        saved_openai_key = ""
        saved_openai_model = OPENAI_MODEL
        try:
            openai_config_path = os.path.join(os.path.dirname(__file__), "openai_config.json")
            if os.path.exists(openai_config_path):
                with open(openai_config_path, 'r') as f:
                    openai_cfg = json.load(f)
                    saved_openai_key = openai_cfg.get("openai_api_key", "")
                    if saved_openai_key:
                        globals()['OPENAI_API_KEY'] = saved_openai_key
                        print("[OpenAI] API key loaded from openai_config.json")
                        print("[OpenAI] NOTE: API calls do NOT appear on chat.openai.com")
                        print("[OpenAI] Check your API usage at: https://platform.openai.com/usage")
                    saved_model = openai_cfg.get("openai_model", "")
                    if saved_model:
                        saved_openai_model = saved_model
                        globals()['OPENAI_MODEL'] = saved_model
                        print(f"[OpenAI] Model loaded from config: {saved_model}")
        except Exception:
            pass
        self.openai_key_var = tk.StringVar(value=saved_openai_key)
        self.openai_key_entry = ttk.Entry(openai_frame, textvariable=self.openai_key_var, show="*", width=22)
        self.openai_key_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(openai_frame, text="Set", style='Bordered.TButton', command=self._apply_openai_key, width=4).pack(side="left", padx=(4,0))
        ttk.Button(openai_frame, text="Save", style='Bordered.TButton', command=self._save_openai_key, width=5).pack(side="left", padx=(4,0))
        ttk.Button(openai_frame, text="Verify", style='Bordered.TButton', command=self._verify_openai_key, width=6).pack(side="left", padx=(4,0))
        row += 1

        # OpenAI Model selector for translation
        _lbl_model = ttk.Label(left_frame, text="Model:")
        _lbl_model.grid(row=row, column=0, sticky="e")
        model_frame = ttk.Frame(left_frame)
        model_frame.grid(row=row, column=1, columnspan=2, sticky="we", padx=(6,0))
        self.openai_model_var = tk.StringVar(value=saved_openai_model)
        model_choices = [
            "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "o4-mini", "o3-mini",
        ]
        self.openai_model_combo = ttk.Combobox(model_frame, textvariable=self.openai_model_var,
                                                values=model_choices, width=18)
        self.openai_model_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(model_frame, text="Set", style='Bordered.TButton', command=self._apply_openai_model, width=4).pack(side="left", padx=(4,0))
        ttk.Button(model_frame, text="Save", style='Bordered.TButton', command=self._save_openai_model, width=5).pack(side="left", padx=(4,0))
        self._block_widget_scroll(self.openai_model_combo)
        self._register_lockable("openai_model", _lbl_model, self.openai_model_combo, is_combo_readonly=False)
        row += 1

        # Custom translation prompt (uses {language} placeholder for selected target language)
        ttk.Label(left_frame, text="Prompt:").grid(row=row, column=0, sticky="ne")
        prompt_frame = ttk.Frame(left_frame)
        prompt_frame.grid(row=row, column=1, columnspan=2, sticky="we", padx=(6,0))
        self.translation_prompt_var = tk.StringVar(value=TRANSLATION_CUSTOM_PROMPT)
        self.translation_prompt_entry = ttk.Entry(prompt_frame, textvariable=self.translation_prompt_var, width=30)
        self.translation_prompt_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(prompt_frame, text="Set", style='Bordered.TButton', command=self._apply_translation_prompt, width=4).pack(side="left", padx=(4,0))
        row += 1
        prompt_hint = ttk.Label(left_frame, text="  Use {language} for auto target lang", font=('Segoe UI', 7))
        prompt_hint.grid(row=row, column=1, columnspan=2, sticky='w', padx=(6,0))
        row += 1

        self.use_ai_voice_var = tk.BooleanVar(value=USE_AI_VOICE_REPLACEMENT)
        ttk.Checkbutton(left_frame, text="Replace voice with AI (TTS)", variable=self.use_ai_voice_var,
                       command=self.on_ai_voice_toggle).grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        _lbl_tts_lang = ttk.Label(left_frame, text="TTS language:")
        _lbl_tts_lang.grid(row=row, column=0, sticky="e")
        self.tts_language_var = tk.StringVar(value=TTS_LANGUAGE)
        tts_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ro', 'ru', 'zh', 'ja', 'ko']
        self.tts_combo = ttk.Combobox(left_frame, textvariable=self.tts_language_var, values=tts_languages, state='readonly', width=10)
        self.tts_combo.grid(row=row, column=1, sticky="w", padx=(6,0))
        self.tts_combo.bind('<<ComboboxSelected>>', self.on_tts_language_selected)
        ttk.Label(left_frame, text="(voice output)").grid(row=row, column=2, sticky='w', padx=(4,0))
        self._register_lockable("tts_language", _lbl_tts_lang, self.tts_combo, is_combo_readonly=True)
        row += 1

        # Voice selection dropdown - shows voices for selected TTS language
        _lbl_tts_voice = ttk.Label(left_frame, text="Voice:")
        _lbl_tts_voice.grid(row=row, column=0, sticky="e")
        
        # Available voices per language (from GenAI Pro API)
        self.voice_options = {
            'en': ['Auto (Default)', 'Voice 1', 'Voice 2', 'Voice 3'],
            'es': ['Auto (Default)', 'Voice 1', 'Voice 2'],
            'fr': ['Auto (Default)', 'Voice 1', 'Voice 2'],
            'de': ['Auto (Default)', 'Voice 1'],
            'it': ['Auto (Default)', 'Voice 1'],
            'pt': ['Auto (Default)', 'Voice 1'],
            'ro': ['Auto (Default)', 'Voice 1'],
            'ru': ['Auto (Default)', 'Voice 1'],
            'zh': ['Auto (Default)', 'Voice 1'],
            'ja': ['Auto (Default)', 'Voice 1'],
            'ko': ['Auto (Default)', 'Voice 1'],
        }
        
        # Voice ID mappings (extend this as you discover more voice IDs from GenAI Pro)
        self.voice_id_map = {
            'Auto (Default)': 'auto',
            'Voice 1': 'uju3wxzG5OhpWcoi3SMy',
            'Voice 2': 'uju3wxzG5OhpWcoi3SMy',  # Replace with actual voice IDs
            'Voice 3': 'uju3wxzG5OhpWcoi3SMy',  # Replace with actual voice IDs
        }
        
        self.tts_voice_var = tk.StringVar(value='Auto (Default)')
        current_voices = self.voice_options.get(TTS_LANGUAGE, ['Auto (Default)'])
        self.tts_voice_combo = ttk.Combobox(left_frame, textvariable=self.tts_voice_var, 
                                            values=current_voices, state='readonly', width=15)
        self.tts_voice_combo.grid(row=row, column=1, sticky="w", padx=(6,0))
        self.tts_voice_combo.bind('<<ComboboxSelected>>', self.on_voice_selected)
        ttk.Label(left_frame, text="(select voice)").grid(row=row, column=2, sticky='w', padx=(4,0))
        self._register_lockable("tts_voice", _lbl_tts_voice, self.tts_voice_combo, is_combo_readonly=True)
        
        # Load custom voices on startup
        self.update_voice_dropdown(TTS_LANGUAGE)
        
        row += 1

        # Fallback Voice dropdown - used when primary voice fails (policy violation, crash, etc.)
        _lbl_fallback_voice = ttk.Label(left_frame, text="Fallback Voice:")
        _lbl_fallback_voice.grid(row=row, column=0, sticky="e")
        self.tts_fallback_voice_var = tk.StringVar(value='Auto (Default)')
        self.tts_fallback_voice_combo = ttk.Combobox(left_frame, textvariable=self.tts_fallback_voice_var,
                                                     values=current_voices, state='readonly', width=15)
        self.tts_fallback_voice_combo.grid(row=row, column=1, sticky="w", padx=(6,0))
        self.tts_fallback_voice_combo.bind('<<ComboboxSelected>>', self.on_fallback_voice_selected)
        ttk.Label(left_frame, text="(backup if primary fails)").grid(row=row, column=2, sticky='w', padx=(4,0))
        self._register_lockable("tts_fallback_voice", _lbl_fallback_voice, self.tts_fallback_voice_combo, is_combo_readonly=True)
        row += 1

        # API Key input for premium TTS services
        ttk.Label(left_frame, text="TTS API Key:").grid(row=row, column=0, sticky="e")
        
        # Load saved API key
        saved_api_key = ""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "tts_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    saved_api_key = config.get("api_key", "")
        except Exception:
            pass
        
        self.tts_api_key_var = tk.StringVar(value=saved_api_key)
        self.tts_api_key_entry = ttk.Entry(left_frame, textvariable=self.tts_api_key_var, show="*", width=30)
        self.tts_api_key_entry.grid(row=row, column=1, columnspan=2, sticky="we", padx=(6,0))
        row += 1
        
        # Button to save API key
        ttk.Button(left_frame, text="Save API Key", style='Accent.TButton', command=self.on_save_api_key).grid(row=row, column=1, sticky="w", padx=(6,0))
        row += 1
        
        # Custom Voice Management Section
        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=6)
        row += 1
        
        ttk.Label(left_frame, text="Custom Voice ID:").grid(row=row, column=0, sticky="e")
        self.custom_voice_id_var = tk.StringVar()
        self.custom_voice_id_entry = ttk.Entry(left_frame, textvariable=self.custom_voice_id_var, width=30)
        self.custom_voice_id_entry.grid(row=row, column=1, columnspan=2, sticky="we", padx=(6,0))
        row += 1
        
        ttk.Label(left_frame, text="Voice Name:").grid(row=row, column=0, sticky="e")
        self.custom_voice_name_var = tk.StringVar()
        self.custom_voice_name_entry = ttk.Entry(left_frame, textvariable=self.custom_voice_name_var, width=30)
        self.custom_voice_name_entry.grid(row=row, column=1, columnspan=2, sticky="we", padx=(6,0))
        row += 1
        
        _lbl_voice_lang = ttk.Label(left_frame, text="Language Category:")
        _lbl_voice_lang.grid(row=row, column=0, sticky="e")
        self.custom_voice_lang_var = tk.StringVar(value="en")
        lang_options = ["en", "es", "fr", "de", "it", "pt", "ro", "ru", "zh", "ja", "ko", "ar"]
        self.custom_voice_lang_combo = ttk.Combobox(left_frame, textvariable=self.custom_voice_lang_var, values=lang_options, state="readonly", width=10)
        self.custom_voice_lang_combo.grid(row=row, column=1, sticky="w", padx=(6,0))
        self._register_lockable("custom_voice_lang", _lbl_voice_lang, self.custom_voice_lang_combo, is_combo_readonly=True)
        row += 1
        
        ttk.Button(left_frame, text="Save Custom Voice", style='Accent.TButton', command=self.on_save_custom_voice).grid(row=row, column=1, sticky="w", padx=(6,0))
        row += 1
        
        # Silence Removal Threshold Section
        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=6)
        row += 1
        
        _lbl_silence = ttk.Label(left_frame, text="Silence Threshold (ms):")
        _lbl_silence.grid(row=row, column=0, sticky="e")
        self.silence_threshold_var = tk.IntVar(value=300)
        silence_threshold_spinbox = ttk.Spinbox(
            left_frame, 
            from_=100, 
            to=2000, 
            increment=50,
            textvariable=self.silence_threshold_var,
            width=10
        )
        silence_threshold_spinbox.grid(row=row, column=1, sticky="w", padx=(6,0))
        ttk.Label(left_frame, text="(Gaps to remove from AI voice)").grid(row=row, column=2, sticky="w", padx=(3,0))
        self._register_lockable("silence_threshold", _lbl_silence, silence_threshold_spinbox)
        row += 1
        
        # Words per caption control (CapCut-style)
        _lbl_wpc = ttk.Label(left_frame, text="Words per caption:")
        _lbl_wpc.grid(row=row, column=0, sticky="e")
        self.words_per_caption_var = tk.IntVar(value=2)
        words_per_caption_spinbox = ttk.Spinbox(
            left_frame, 
            from_=1, 
            to=3, 
            increment=1,
            textvariable=self.words_per_caption_var,
            width=10,
            command=self.on_words_per_caption_changed
        )
        words_per_caption_spinbox.grid(row=row, column=1, sticky="w", padx=(6,0))
        # Also bind to var changes for direct typing
        self.words_per_caption_var.trace_add('write', self.on_words_per_caption_changed)
        ttk.Label(left_frame, text="(1=single word, 2-3=groups)").grid(row=row, column=2, sticky="w", padx=(3,0))
        self._register_lockable("words_per_caption", _lbl_wpc, words_per_caption_spinbox)
        row += 1

        # --- TTS Voice Speed Control ---
        _lbl_speed = ttk.Label(left_frame, text="TTS speed:")
        _lbl_speed.grid(row=row, column=0, sticky="e")
        self.tts_speed_var = tk.DoubleVar(value=TTS_SPEED)
        self.tts_speed_scale = tk.Scale(left_frame, from_=0.5, to=2.0, resolution=0.05, orient='horizontal', length=200, showvalue=0, variable=self.tts_speed_var, command=self.on_tts_speed_changed)
        self.tts_speed_scale.grid(row=row, column=1, padx=(6,0))
        self.tts_speed_label = ttk.Label(left_frame, text=f"{TTS_SPEED:.2f}x", width=22)
        self.tts_speed_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self._register_lockable("tts_speed", _lbl_speed, self.tts_speed_scale)
        row += 1

        # Caption case toggle buttons (UPPER / lower) — mutually exclusive
        ttk.Label(left_frame, text="Caption case:").grid(row=row, column=0, sticky="e")
        self.caption_case_var = tk.StringVar(value="none")
        _case_frame = ttk.Frame(left_frame)
        _case_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=(6,0))
        self._btn_upper = ttk.Button(_case_frame, text="UPPER", width=7, style='Bordered.TButton',
                                     command=lambda: self._set_caption_case("upper"))
        self._btn_upper.pack(side="left", padx=(0, 4))
        self._btn_lower = ttk.Button(_case_frame, text="lower", width=7, style='Bordered.TButton',
                                     command=lambda: self._set_caption_case("lower"))
        self._btn_lower.pack(side="left", padx=(0, 4))
        self._btn_case_off = ttk.Button(_case_frame, text="Off", width=5, style='Bordered.TButton',
                                        command=lambda: self._set_caption_case("none"))
        self._btn_case_off.pack(side="left")
        row += 1

        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        # ══════════ VIDEO EFFECTS ══════════
        ttk.Label(left_frame, text="✨  Video Effects", style='SectionHeader.TLabel').grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        # Sharpness/Resilience effect
        self.effect_sharpness_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Resilience (Sharpness)", variable=self.effect_sharpness_var,
                       command=self._mini_update_worker_async).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Label(left_frame, text="💎").grid(row=row, column=2, sticky="w")
        row += 1
        
        _lbl_sharp = ttk.Label(left_frame, text="Intensity:")
        _lbl_sharp.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.effect_sharpness_intensity_var = tk.DoubleVar(value=1.5)
        self.sharpness_scale = tk.Scale(left_frame, from_=0.5, to=3.0, resolution=0.1, orient='horizontal', 
                                   length=120, showvalue=0, variable=self.effect_sharpness_intensity_var,
                                   command=lambda v: self._mini_update_worker_async())
        self.sharpness_scale.grid(row=row, column=1, padx=(6,0))
        self.sharpness_label = ttk.Label(left_frame, text=f"{self.effect_sharpness_intensity_var.get():.1f}x")
        self.sharpness_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.effect_sharpness_intensity_var.trace('w', lambda *args: self.sharpness_label.config(text=f"{self.effect_sharpness_intensity_var.get():.1f}x"))
        self._register_lockable("sharpness", _lbl_sharp, self.sharpness_scale)
        row += 1

        # Saturation boost
        self.effect_saturation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Vibrance (Saturation)", variable=self.effect_saturation_var,
                       command=self._mini_update_worker_async).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Label(left_frame, text="🌈").grid(row=row, column=2, sticky="w")
        row += 1
        
        _lbl_sat = ttk.Label(left_frame, text="Intensity:")
        _lbl_sat.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.effect_saturation_intensity_var = tk.DoubleVar(value=1.3)
        self.saturation_scale = tk.Scale(left_frame, from_=0.5, to=2.0, resolution=0.1, orient='horizontal', 
                                    length=120, showvalue=0, variable=self.effect_saturation_intensity_var,
                                    command=lambda v: self._mini_update_worker_async())
        self.saturation_scale.grid(row=row, column=1, padx=(6,0))
        self.saturation_label = ttk.Label(left_frame, text=f"{self.effect_saturation_intensity_var.get():.1f}x")
        self.saturation_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.effect_saturation_intensity_var.trace('w', lambda *args: self.saturation_label.config(text=f"{self.effect_saturation_intensity_var.get():.1f}x"))
        self._register_lockable("saturation", _lbl_sat, self.saturation_scale)
        row += 1

        # Contrast enhancement
        self.effect_contrast_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="HDR (Contrast)", variable=self.effect_contrast_var,
                       command=self._mini_update_worker_async).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Label(left_frame, text="⚡").grid(row=row, column=2, sticky="w")
        row += 1
        
        _lbl_con = ttk.Label(left_frame, text="Intensity:")
        _lbl_con.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.effect_contrast_intensity_var = tk.DoubleVar(value=1.2)
        self.contrast_scale = tk.Scale(left_frame, from_=0.5, to=2.0, resolution=0.1, orient='horizontal', 
                                 length=120, showvalue=0, variable=self.effect_contrast_intensity_var,
                                 command=lambda v: self._mini_update_worker_async())
        self.contrast_scale.grid(row=row, column=1, padx=(6,0))
        self.contrast_label = ttk.Label(left_frame, text=f"{self.effect_contrast_intensity_var.get():.1f}x")
        self.contrast_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.effect_contrast_intensity_var.trace('w', lambda *args: self.contrast_label.config(text=f"{self.effect_contrast_intensity_var.get():.1f}x"))
        self._register_lockable("contrast", _lbl_con, self.contrast_scale)
        row += 1

        # Brightness adjustment
        self.effect_brightness_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Brightness Boost", variable=self.effect_brightness_var,
                       command=self._mini_update_worker_async).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Label(left_frame, text="☀️").grid(row=row, column=2, sticky="w")
        row += 1
        
        _lbl_bri = ttk.Label(left_frame, text="Intensity:")
        _lbl_bri.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.effect_brightness_intensity_var = tk.DoubleVar(value=1.15)
        self.brightness_scale = tk.Scale(left_frame, from_=0.5, to=2.0, resolution=0.05, orient='horizontal', 
                                    length=120, showvalue=0, variable=self.effect_brightness_intensity_var,
                                    command=lambda v: self._mini_update_worker_async())
        self.brightness_scale.grid(row=row, column=1, padx=(6,0))
        self.brightness_label = ttk.Label(left_frame, text=f"{self.effect_brightness_intensity_var.get():.2f}x")
        self.brightness_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.effect_brightness_intensity_var.trace('w', lambda *args: self.brightness_label.config(text=f"{self.effect_brightness_intensity_var.get():.2f}x"))
        self._register_lockable("brightness", _lbl_bri, self.brightness_scale)
        row += 1

        # Film grain / Vintage
        self.effect_vintage_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Vintage (Film Grain)", variable=self.effect_vintage_var,
                       command=self._mini_update_worker_async).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Label(left_frame, text="📽️").grid(row=row, column=2, sticky="w")
        row += 1
        
        _lbl_vin = ttk.Label(left_frame, text="Grain:")
        _lbl_vin.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.effect_vintage_intensity_var = tk.DoubleVar(value=0.3)
        self.vintage_scale = tk.Scale(left_frame, from_=0.1, to=1.0, resolution=0.05, orient='horizontal', 
                                length=120, showvalue=0, variable=self.effect_vintage_intensity_var,
                                command=lambda v: self._mini_update_worker_async())
        self.vintage_scale.grid(row=row, column=1, padx=(6,0))
        self.vintage_label = ttk.Label(left_frame, text=f"{self.effect_vintage_intensity_var.get():.2f}")
        self.vintage_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.effect_vintage_intensity_var.trace('w', lambda *args: self.vintage_label.config(text=f"{self.effect_vintage_intensity_var.get():.2f}"))
        self._register_lockable("vintage", _lbl_vin, self.vintage_scale)
        row += 1

        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        # ══════════ BLUR OVERLAY ══════════
        ttk.Label(left_frame, text="🔲  Blur Overlay", style='SectionHeader.TLabel').grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        self.blur_overlay_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text="Enable Blur Overlay", variable=self.blur_overlay_enabled_var,
                       command=self._mini_update_worker_async).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Label(left_frame, text="🔲").grid(row=row, column=2, sticky="w")
        row += 1

        _lbl_bx = ttk.Label(left_frame, text="X (%):")
        _lbl_bx.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.blur_overlay_x_var = tk.DoubleVar(value=10.0)
        self.blur_x_scale = tk.Scale(left_frame, from_=0, to=100, resolution=0.5, orient='horizontal',
                                length=120, showvalue=0, variable=self.blur_overlay_x_var,
                                command=lambda v: self._mini_update_worker_async())
        self.blur_x_scale.grid(row=row, column=1, padx=(6,0))
        self.blur_overlay_x_label = ttk.Label(left_frame, text=f"{self.blur_overlay_x_var.get():.1f}%")
        self.blur_overlay_x_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.blur_overlay_x_var.trace('w', lambda *args: self.blur_overlay_x_label.config(text=f"{self.blur_overlay_x_var.get():.1f}%"))
        self._register_lockable("blur_x", _lbl_bx, self.blur_x_scale)
        row += 1

        _lbl_by = ttk.Label(left_frame, text="Y (%):")
        _lbl_by.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.blur_overlay_y_var = tk.DoubleVar(value=10.0)
        self.blur_y_scale = tk.Scale(left_frame, from_=0, to=100, resolution=0.5, orient='horizontal',
                                length=120, showvalue=0, variable=self.blur_overlay_y_var,
                                command=lambda v: self._mini_update_worker_async())
        self.blur_y_scale.grid(row=row, column=1, padx=(6,0))
        self.blur_overlay_y_label = ttk.Label(left_frame, text=f"{self.blur_overlay_y_var.get():.1f}%")
        self.blur_overlay_y_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.blur_overlay_y_var.trace('w', lambda *args: self.blur_overlay_y_label.config(text=f"{self.blur_overlay_y_var.get():.1f}%"))
        self._register_lockable("blur_y", _lbl_by, self.blur_y_scale)
        row += 1

        _lbl_bw = ttk.Label(left_frame, text="W (%):")
        _lbl_bw.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.blur_overlay_w_var = tk.DoubleVar(value=20.0)
        self.blur_w_scale = tk.Scale(left_frame, from_=1, to=100, resolution=0.5, orient='horizontal',
                                length=120, showvalue=0, variable=self.blur_overlay_w_var,
                                command=lambda v: self._mini_update_worker_async())
        self.blur_w_scale.grid(row=row, column=1, padx=(6,0))
        self.blur_overlay_w_label = ttk.Label(left_frame, text=f"{self.blur_overlay_w_var.get():.1f}%")
        self.blur_overlay_w_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.blur_overlay_w_var.trace('w', lambda *args: self.blur_overlay_w_label.config(text=f"{self.blur_overlay_w_var.get():.1f}%"))
        self._register_lockable("blur_w", _lbl_bw, self.blur_w_scale)
        row += 1

        _lbl_bh = ttk.Label(left_frame, text="H (%):")
        _lbl_bh.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.blur_overlay_h_var = tk.DoubleVar(value=15.0)
        self.blur_h_scale = tk.Scale(left_frame, from_=1, to=100, resolution=0.5, orient='horizontal',
                                length=120, showvalue=0, variable=self.blur_overlay_h_var,
                                command=lambda v: self._mini_update_worker_async())
        self.blur_h_scale.grid(row=row, column=1, padx=(6,0))
        self.blur_overlay_h_label = ttk.Label(left_frame, text=f"{self.blur_overlay_h_var.get():.1f}%")
        self.blur_overlay_h_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.blur_overlay_h_var.trace('w', lambda *args: self.blur_overlay_h_label.config(text=f"{self.blur_overlay_h_var.get():.1f}%"))
        self._register_lockable("blur_h", _lbl_bh, self.blur_h_scale)
        row += 1

        _lbl_bi = ttk.Label(left_frame, text="Blur:")
        _lbl_bi.grid(row=row, column=0, sticky="e", padx=(20,0))
        self.blur_overlay_intensity_var = tk.IntVar(value=20)
        self.blur_intensity_scale = tk.Scale(left_frame, from_=2, to=80, resolution=1, orient='horizontal',
                                        length=120, showvalue=0, variable=self.blur_overlay_intensity_var,
                                        command=lambda v: self._mini_update_worker_async())
        self.blur_intensity_scale.grid(row=row, column=1, padx=(6,0))
        self.blur_overlay_intensity_label = ttk.Label(left_frame, text=f"{self.blur_overlay_intensity_var.get()}")
        self.blur_overlay_intensity_label.grid(row=row, column=2, sticky='w', padx=(4,0))
        self.blur_overlay_intensity_var.trace('w', lambda *args: self.blur_overlay_intensity_label.config(text=f"{self.blur_overlay_intensity_var.get()}"))
        self._register_lockable("blur_intensity", _lbl_bi, self.blur_intensity_scale)
        row += 1

        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        # ══════════ CROP SETTINGS ══════════
        ttk.Label(left_frame, text="✂️  Crop Settings", style='SectionHeader.TLabel').grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        ttk.Label(left_frame, text="Top:").grid(row=row, column=0, sticky="e")
        self.top_percent_var = tk.DoubleVar(value=CROP_TOP_RATIO*100)
        self.top_label = ttk.Label(left_frame, text=f"{self.top_percent_var.get():.1f}%")
        self.top_label.grid(row=row, column=1, sticky="w", padx=(6,0))
        row += 1

        ttk.Label(left_frame, text="Bottom:").grid(row=row, column=0, sticky="e")
        self.bottom_percent_var = tk.DoubleVar(value=CROP_BOTTOM_RATIO*100)
        self.bottom_label = ttk.Label(left_frame, text=f"{self.bottom_percent_var.get():.1f}%")
        self.bottom_label.grid(row=row, column=1, sticky="w", padx=(6,0))
        row += 1

        btns = ttk.Frame(left_frame)
        try:
            btns.configure(style='TFrame')
            btns.config(bg=self.BG_DARK)
        except Exception:
            pass
        btns.grid(row=row, column=0, columnspan=3, sticky="w", pady=(6,0))
        ttk.Button(btns, text="Save crop settings", style='Accent.TButton', command=self.on_save_crop).pack(side="left", padx=4)
        ttk.Button(btns, text="Load crop settings", style='Accent.TButton', command=self.on_load_crop).pack(side="left", padx=4)
        row += 1

        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        # ══════════ PRESETS ══════════
        ttk.Label(left_frame, text="💾  Settings Presets", style='SectionHeader.TLabel').grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1
        preset_btns = ttk.Frame(left_frame)
        try:
            preset_btns.configure(style='TFrame')
            preset_btns.config(bg=self.BG_DARK)
        except Exception:
            pass
        preset_btns.grid(row=row, column=0, columnspan=3, sticky="w", pady=(4,0))
        ttk.Button(preset_btns, text="Save Preset", style='Accent.TButton', command=self.save_preset).pack(side="left", padx=4)
        ttk.Button(preset_btns, text="Load Preset", style='Accent.TButton', command=self.load_preset).pack(side="left", padx=4)
        ttk.Button(preset_btns, text="Reset to Defaults", style='Warning.TButton', command=self.reset_to_defaults).pack(side="left", padx=4)
        row += 1

        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        # ══════════ JOB QUEUE ══════════
        ttk.Label(left_frame, text="📋  Job Queue", style='SectionHeader.TLabel').grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1
        # Container so the scrollbar sits beside the listbox without disturbing the grid layout
        job_lb_frame = tk.Frame(left_frame, bg=self.BG_SECONDARY)
        job_lb_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        job_lb_frame.columnconfigure(0, weight=1)
        job_lb_frame.rowconfigure(0, weight=1)

        job_lb_scroll = tk.Scrollbar(job_lb_frame, orient="vertical")
        job_lb_scroll.grid(row=0, column=1, sticky="ns")

        self.job_listbox = tk.Listbox(job_lb_frame, height=10, selectmode=tk.EXTENDED,
                                      bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY,
                                      font=('Segoe UI', 9),
                                      yscrollcommand=job_lb_scroll.set)
        self.job_listbox.grid(row=0, column=0, sticky="nsew")
        job_lb_scroll.config(command=self.job_listbox.yview)
        try:
            self.job_listbox.config(selectbackground=self.SELECT_BG, selectforeground='#ffffff',
                                   highlightthickness=1, highlightbackground=self.BORDER_COLOR,
                                   highlightcolor=self.ACCENT_BLUE,
                                   bd=1, relief='flat')
        except Exception:
            pass
        # Mouse-wheel scrolling (Windows/Mac/Linux)
        def _jlb_scroll(event):
            try:
                if event.num == 4:
                    self.job_listbox.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.job_listbox.yview_scroll(1, "units")
                else:
                    self.job_listbox.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass
        self.job_listbox.bind('<MouseWheel>', _jlb_scroll)
        self.job_listbox.bind('<Button-4>', _jlb_scroll)
        self.job_listbox.bind('<Button-5>', _jlb_scroll)
        # Add double-click binding to load job settings
        self.job_listbox.bind('<Double-Button-1>', self.on_job_double_click)
        left_frame.rowconfigure(row, weight=1)
        row += 1

        job_btns = ttk.Frame(left_frame)
        try:
            job_btns.configure(style='TFrame')
            job_btns.config(bg=self.BG_DARK)
        except Exception:
            pass
        job_btns.grid(row=row, column=0, columnspan=3, sticky="w", pady=(6,0))
        ttk.Button(job_btns, text="➕ Add Job", style='Accent.TButton', command=self.add_job).pack(side="left", padx=4)
        ttk.Button(job_btns, text="➕ Add Multiple", style='Accent.TButton', command=self.add_multiple_jobs).pack(side="left", padx=4)
        self.edit_job_btn = ttk.Button(job_btns, text="✏️ Edit Job", style='Info.TButton', command=self.on_edit_job)
        self.edit_job_btn.pack(side="left", padx=4)
        self.update_job_btn = ttk.Button(job_btns, text="💾 Update Job", style='Accent.TButton', command=self.update_job, state="disabled")
        self.update_job_btn.pack(side="left", padx=4)
        ttk.Button(job_btns, text="Remove Selected", style='Danger.TButton', command=self.remove_job).pack(side="left", padx=4)
        self.run_queue_btn = ttk.Button(job_btns, text="▶ Run Queue", style='Success.TButton', command=self.run_queue)
        self.run_queue_btn.pack(side="left", padx=4)
        self._editing_job_idx = None  # index of job currently being edited
        row += 1

        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        # ══════════ TIKTOK DOWNLOADER ══════════
        ttk.Label(left_frame, text="📥  TikTok Downloader", style='SectionHeader.TLabel').grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1

        ttk.Label(left_frame, text="Account URL / @user:").grid(row=row, column=0, sticky="w")
        self.tt_dl_url_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.tt_dl_url_var).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
        row += 1

        ttk.Label(left_frame, text="Output folder:").grid(row=row, column=0, sticky="w")
        self.tt_dl_folder_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.tt_dl_folder_var).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(left_frame, text="Browse…", style='Bordered.TButton',
                   command=lambda: self.tt_dl_folder_var.set(
                       filedialog.askdirectory(title="Select output folder") or self.tt_dl_folder_var.get()
                   )).grid(row=row, column=2, padx=6)
        row += 1

        ttk.Label(left_frame, text="Start numbering from:").grid(row=row, column=0, sticky="w")
        self.tt_dl_start_var = tk.IntVar(value=1)
        ttk.Spinbox(left_frame, from_=1, to=9999, textvariable=self.tt_dl_start_var, width=8).grid(row=row, column=1, sticky="w", padx=6)
        ttk.Button(left_frame, text="🔍 Detect", style='Bordered.TButton',
                   command=self._tt_dl_detect_start).grid(row=row, column=2, padx=6)
        row += 1

        ttk.Label(left_frame, text="Max videos to download:").grid(row=row, column=0, sticky="w")
        self.tt_dl_max_var = tk.IntVar(value=10)
        self.tt_dl_max_spin = ttk.Spinbox(left_frame, from_=1, to=9999, textvariable=self.tt_dl_max_var, width=8)
        self.tt_dl_max_spin.grid(row=row, column=1, sticky="w", padx=6)
        self.tt_dl_all_var = tk.BooleanVar(value=False)
        def _toggle_all_videos():
            if self.tt_dl_all_var.get():
                self.tt_dl_max_spin.config(state="disabled")
            else:
                self.tt_dl_max_spin.config(state="normal")
        ttk.Checkbutton(left_frame, text="Descarcă TOATE", variable=self.tt_dl_all_var,
                        command=_toggle_all_videos).grid(row=row, column=2, sticky="w", padx=6)
        row += 1

        ttk.Label(left_frame, text="Sortare videouri:").grid(row=row, column=0, sticky="w")
        self.tt_dl_order_var = tk.StringVar(value="Cele mai noi")
        tt_dl_order_cb = ttk.Combobox(left_frame, textvariable=self.tt_dl_order_var, width=22,
                                      values=["Cele mai noi", "Cele mai vechi", "Populare"],
                                      state="readonly")
        tt_dl_order_cb.grid(row=row, column=1, columnspan=2, sticky="w", padx=6)
        row += 1

        ttk.Label(left_frame, text="Începe de la video URL:").grid(row=row, column=0, sticky="w")
        self.tt_dl_from_url_var = tk.StringVar()
        ttk.Entry(left_frame, textvariable=self.tt_dl_from_url_var).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
        row += 1

        ttk.Label(left_frame, textvariable=tk.StringVar(
            value="(opțional) Dacă ai mai descărcat, lipește link-ul ultimului video și descărcarea continuă de la următorul."
        ), style='TLabel', foreground=self.TEXT_MUTED, wraplength=340).grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))
        row += 1

        tt_dl_btns = ttk.Frame(left_frame)
        try:
            tt_dl_btns.configure(style='TFrame')
            tt_dl_btns.config(bg=self.BG_DARK)
        except Exception:
            pass
        tt_dl_btns.grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.tt_dl_btn = ttk.Button(tt_dl_btns, text="⬇ Download", style='Accent.TButton', command=self._tt_dl_start)
        self.tt_dl_btn.pack(side="left", padx=4)
        self.tt_dl_cancel_btn = ttk.Button(tt_dl_btns, text="⏹ Cancel", style='Danger.TButton',
                                           command=self._tt_dl_cancel, state="disabled")
        self.tt_dl_cancel_btn.pack(side="left", padx=4)
        row += 1

        self.tt_dl_status_var = tk.StringVar(value="")
        self.tt_dl_status_label = ttk.Label(left_frame, textvariable=self.tt_dl_status_var,
                                            style='TLabel', foreground=self.TEXT_SECONDARY, wraplength=340)
        self.tt_dl_status_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self._tt_dl_stop = False
        row += 1

        ttk.Separator(left_frame).grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        # ══════════ ACTIONS ══════════
        bottom_controls = ttk.Frame(left_frame)
        try:
            bottom_controls.configure(style='TFrame')
            bottom_controls.config(bg=self.BG_DARK)
        except Exception:
            pass
        bottom_controls.grid(row=row, column=0, columnspan=3, sticky="we")
        self.run_single_btn = ttk.Button(bottom_controls, text="▶ Run (Single)", style='Success.TButton', command=self.on_run_single)
        self.run_single_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(bottom_controls, text="⏹ Stop", style='Danger.TButton', command=self._on_stop)
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(bottom_controls, text="Toggle Fullscreen", style='Info.TButton', command=self.toggle_fullscreen).pack(side="left", padx=4)
        ttk.Button(bottom_controls, text="Quit", style='Danger.TButton', command=root.quit).pack(side="left", padx=4)

        # preview
        ttk.Label(preview_frame, text="🖼️  Mini Preview — drag lines to adjust crop",
                 style='SectionHeader.TLabel').pack(anchor="w")
        self.mini_canvas_container = ttk.Frame(preview_frame)
        self.mini_canvas_container.pack(side='left', pady=(6,4), fill="both", expand=False)
        try:
            self.mini_canvas_container.config(style='MiniCanvas.TFrame')
        except Exception:
            pass
        # create a visible border with accent color
        self.mini_canvas_border = tk.Frame(self.mini_canvas_container, bg=self.BORDER_COLOR, bd=2, relief='solid')
        self.mini_canvas_border.pack(fill='both', expand=True, padx=0, pady=0)
        self.mini_canvas = tk.Canvas(self.mini_canvas_border, width=360, height=640, bg=self.BG_SECONDARY, highlightthickness=0)
        self.mini_canvas.pack(fill='both', expand=True)
        
        self.mini_image_ref = None
        # font selection panel to the right of the mini preview
        self.font_panel = ttk.Frame(preview_frame)
        self.font_panel.pack(side='left', fill='y', padx=(8,0), pady=(6,4))
        try:
            self.font_panel.configure(style='TFrame')
        except Exception:
            pass
        ttk.Label(self.font_panel, text='🔤 Fonts:').pack(anchor='nw')
        self.font_listbox = tk.Listbox(self.font_panel, height=12, exportselection=False)
        self.font_listbox.pack(fill='y', expand=True)
        # preview sample
        self.font_preview = ttk.Label(self.font_panel, text='Sample: Hello!', anchor='center')
        self.font_preview.pack(fill='x', pady=6)
        # --- Color selectors for caption text and stroke (inserted) ---
        try:
            color_frame = ttk.Frame(self.font_panel)
            color_frame.pack(fill='x', pady=(6,4))
            ttk.Label(color_frame, text='Text color:').grid(row=0, column=0, sticky='w')
            self.text_color_canvas = tk.Canvas(color_frame, width=28, height=28, bg=self.BORDER_COLOR, highlightthickness=1, highlightbackground=self.BORDER_COLOR, bd=0)
            self.text_color_canvas.grid(row=0, column=1, padx=(6,12))
            ttk.Button(color_frame, text='Custom...', style='Bordered.TButton', command=self.on_pick_text_color).grid(row=0, column=2, padx=(0,6))
            ttk.Label(color_frame, text='Stroke color:').grid(row=1, column=0, sticky='w', pady=(6,0))
            self.stroke_color_canvas = tk.Canvas(color_frame, width=28, height=28, bg=self.BORDER_COLOR, highlightthickness=1, highlightbackground=self.BORDER_COLOR, bd=0)
            self.stroke_color_canvas.grid(row=1, column=1, padx=(6,12), pady=(6,0))
            ttk.Button(color_frame, text='Custom...', style='Bordered.TButton', command=self.on_pick_stroke_color).grid(row=1, column=2, padx=(0,6), pady=(6,0))
            # quick preset swatches below
            swatch_frame = ttk.Frame(self.font_panel)
            swatch_frame.pack(fill='x', pady=(4,6))
            PRESET_SWATCHES = ['#FFFFFF', '#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']
            ttk.Label(swatch_frame, text='Presets:').pack(anchor='w')
            sf = tk.Frame(swatch_frame, bg=self.BG_DARK)
            sf.pack(fill='x', pady=(4,0))

            # --- Stroke width control ---
            try:
                sw_frame = ttk.Frame(self.font_panel)
                sw_frame.pack(fill='x', pady=(6,4))
                _lbl_sw = ttk.Label(sw_frame, text='Stroke width:')
                _lbl_sw.grid(row=0, column=0, sticky='w')
                max_w = max(1, int(CAPTION_FONT_SIZE * 0.5))
                self.stroke_width_var = tk.DoubleVar(value=float(globals().get('CAPTION_STROKE_WIDTH', max(1, int(CAPTION_FONT_SIZE * 0.05)))))
                self.stroke_width_scale = tk.Scale(sw_frame, from_=0, to=max_w, orient='horizontal', length=140, showvalue=0, variable=self.stroke_width_var, command=self.on_stroke_width_changed)
                self.stroke_width_scale.grid(row=0, column=1, padx=(6,8))
                self.stroke_width_label = ttk.Label(sw_frame, text=str(int(self.stroke_width_var.get())))
                self.stroke_width_label.grid(row=0, column=2, sticky='w')
                self._register_lockable("stroke_width", _lbl_sw, self.stroke_width_scale)
            except Exception:
                pass

            # --- Caption vertical position control (NEW) ---
            try:
                pos_frame = ttk.Frame(self.font_panel)
                pos_frame.pack(fill='x', pady=(6,4))
                _lbl_yoff = ttk.Label(pos_frame, text='Caption Y offset:')
                _lbl_yoff.grid(row=0, column=0, sticky='w')
                # Offset from bottom in pixels (0 = at bottom, negative = move up, positive = move down)
                self.caption_y_offset_var = tk.IntVar(value=0)
                self.caption_y_offset_scale = tk.Scale(pos_frame, from_=-1080, to=200, orient='horizontal', length=140, showvalue=0, resolution=1, variable=self.caption_y_offset_var, command=self.on_caption_position_changed)
                self.caption_y_offset_scale.grid(row=0, column=1, padx=(6,8))
                # Spinbox for direct numeric input (precise positioning)
                self.caption_y_offset_spinbox = ttk.Spinbox(pos_frame, from_=-1920, to=200, textvariable=self.caption_y_offset_var, width=6, command=self._on_caption_y_spinbox_changed)
                self.caption_y_offset_spinbox.grid(row=0, column=2, sticky='w')
                self.caption_y_offset_spinbox.bind('<Return>', lambda e: self._on_caption_y_spinbox_changed())
                self.caption_y_offset_spinbox.bind('<FocusOut>', lambda e: self._on_caption_y_spinbox_changed())
                # Keep the label reference for backward compatibility (preset load/save uses it)
                self.caption_y_offset_label = ttk.Label(pos_frame, text="px")
                self.caption_y_offset_label.grid(row=0, column=3, sticky='w', padx=(2,0))
                self._register_lockable("caption_y_offset", _lbl_yoff, self.caption_y_offset_scale)
                
                # --- Font Size slider (row 1) ---
                _lbl_fs = ttk.Label(pos_frame, text='Font Size:')
                _lbl_fs.grid(row=1, column=0, sticky='w', pady=(4,0))
                # Font size in pixels (20-120 range, default 56)
                self.caption_font_size_var = tk.IntVar(value=globals().get('CAPTION_FONT_SIZE', 56))
                self.caption_font_size_scale = tk.Scale(pos_frame, from_=20, to=120, orient='horizontal', length=140, showvalue=0, variable=self.caption_font_size_var, command=self.on_caption_font_size_changed)
                self.caption_font_size_scale.grid(row=1, column=1, padx=(6,8), pady=(4,0))
                self.caption_font_size_label = ttk.Label(pos_frame, text=f"{self.caption_font_size_var.get()}px")
                self.caption_font_size_label.grid(row=1, column=2, sticky='w', pady=(4,0))
                self._register_lockable("caption_font_size", _lbl_fs, self.caption_font_size_scale)
            except Exception:
                pass

            for c in PRESET_SWATCHES:
                b = tk.Canvas(sf, width=22, height=22, bg=c, highlightthickness=1, bd=0)
                b.pack(side='left', padx=4)
                b.bind('<Button-1>', lambda e, col=c: self._set_text_color_hex(col))
                b.bind('<Button-3>', lambda e, col=c: self._set_stroke_color_hex(col))
        except Exception:
            pass

        # --- Caption template selector + preview ---
        try:
            self.template_var = tk.StringVar(value='2 words')
            _lbl_tpl = ttk.Label(self.font_panel, text='Caption template:')
            _lbl_tpl.pack(anchor='nw', pady=(6,0))
            self.template_cb = ttk.Combobox(self.font_panel, values=['1 word', '2 words', '3 words'], textvariable=self.template_var, state='readonly', width=20)
            self.template_cb.pack(fill='x', pady=(2,4))
            self.template_cb.bind('<<ComboboxSelected>>', self.on_template_selected)
            self._register_lockable("template", _lbl_tpl, self.template_cb, is_combo_readonly=True)
            # Preview area for generated caption image
            self.caption_preview_canvas = tk.Canvas(self.font_panel, width=320, height=120,
                                                    bg=self.BG_SECONDARY, highlightthickness=0)
            self.caption_preview_canvas.pack(pady=(4,6))
            ttk.Button(self.font_panel, text='Preview template', style='Accent.TButton', command=self.on_preview_template).pack(pady=(0,6))
        except Exception:
            pass

        
# load only fonts from the C:	iktok folder (only .ttf/.otf)
        try:
            self._tiktok_font_paths = {}
            fams = []
            search_root = os.path.normpath("C:\\tiktok")
            for root, dirs, files in os.walk(search_root):
                for fn in files:
                    if fn.lower().endswith((".ttf", ".otf")):
                        name = os.path.splitext(fn)[0]
                        display = name
                        path = os.path.join(root, fn)
                        if display not in self._tiktok_font_paths:
                            self._tiktok_font_paths[display] = path
                            fams.append(display)
            fams = sorted(fams, key=lambda s: s.lower())
        except Exception:
            fams = ['Arial', 'Helvetica', 'Times']
        for f in fams:
            try:
                self.font_listbox.insert('end', f)
            except Exception:
                pass
        self.font_listbox.bind('<<ListboxSelect>>', self.on_font_selected)
        if not hasattr(self, 'saved_fonts'):
            self.saved_fonts = {}

        # save/load font buttons
        btn_frame = ttk.Frame(self.font_panel)
        try:
            btn_frame.configure(style='TFrame')
        except Exception:
            pass
        btn_frame.pack(fill='x', pady=(4,0))
        self.save_font_btn = ttk.Button(btn_frame, text='Save font for video', style='Accent.TButton', command=self.on_save_font_clicked)
        self.save_font_btn.pack(side='left', padx=(0,4))
        self.load_font_btn = ttk.Button(btn_frame, text='Load saved font', style='Accent.TButton', command=self.on_load_font_clicked)
        self.load_font_btn.pack(side='left')
        self.saved_font_label = ttk.Label(self.font_panel, text='Saved: None', anchor='w')
        self.saved_font_label.pack(fill='x', pady=(4,0))
        self.selected_font = None

        self.selected_font_path = None
        self.mini_worker_thread = None

        tl_frame = ttk.Frame(preview_frame)
        try:
            tl_frame.configure(style='TFrame')
            tl_frame.config(bg=self.BG_DARK)
        except Exception:
            pass
        tl_frame.pack(fill="x", pady=(4,0))
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_scale = CanvasSlider(tl_frame, from_=0.0, to=1.0, orient="horizontal", length=360, resolution=0.1, variable=self.time_var, command=self.on_time_changed)
        self.time_scale.pack(side="left", fill="x", expand=True)
        self.time_label = ttk.Label(tl_frame, text="00:00")
        self.time_label.pack(side="left", padx=6)
        
        # --- Zoom Control Slider ---
        zoom_frame = ttk.Frame(preview_frame)
        try:
            zoom_frame.configure(style='TFrame')
            zoom_frame.config(bg=self.BG_DARK)
        except Exception:
            pass
        zoom_frame.pack(fill="x", pady=(4,0))
        ttk.Label(zoom_frame, text="🔍 Zoom:").pack(side="left")
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_scale = CanvasSlider(zoom_frame, from_=0.5, to=2.0, orient="horizontal", length=300, resolution=0.01, variable=self.zoom_var, command=self.on_zoom_changed)
        self.zoom_scale.pack(side="left", fill="x", expand=True, padx=4)
        self.zoom_label = ttk.Label(zoom_frame, text="1.00x")
        self.zoom_label.pack(side="left", padx=6)
        
        ttk.Button(preview_frame, text="🔄 Refresh Preview", style='Accent.TButton',
                  command=self.on_mini_refresh_clicked).pack(pady=(6,0))
        
        # --- TikTok Format Preview (9:16 aspect ratio) ---
        tiktok_preview_frame = ttk.Frame(preview_frame)
        try:
            tiktok_preview_frame.configure(style='TFrame')
            tiktok_preview_frame.config(bg=self.BG_DARK)
        except Exception:
            pass
        tiktok_preview_frame.pack(fill='both', expand=False, pady=(6,0))
        ttk.Label(tiktok_preview_frame, text="📱  TikTok Preview (9:16):",
                 style='SectionHeader.TLabel').pack(anchor="w")
        self.tiktok_preview_border = tk.Frame(tiktok_preview_frame, bg=self.BORDER_COLOR, bd=2, relief='solid')
        self.tiktok_preview_border.pack(fill='both', expand=True, padx=0, pady=(4,0))
        # 180x320 = 9:16 aspect ratio
        self.tiktok_preview_canvas = tk.Canvas(self.tiktok_preview_border, width=180, height=320,
                                               bg=self.BG_SECONDARY, highlightthickness=0)
        self.tiktok_preview_canvas.pack(fill='both', expand=True)
        self.tiktok_preview_image_ref = None
        
        ttk.Button(preview_frame, text="🔄 Refresh TikTok Preview", style='Accent.TButton',
                  command=self.on_tiktok_preview_refresh).pack(pady=(6,0))

        self.mini_canvas.bind("<ButtonPress-1>", self._mini_on_mouse_down)
        self.mini_canvas.bind("<B1-Motion>", self._mini_on_mouse_move)
        self.mini_canvas.bind("<ButtonRelease-1>", self._mini_on_mouse_up)
        self._mini_blur_drag_start_y = None
        self._mini_blur_drag_start_val = None

        self.mini_base_img = None
        self.mini_scale = 1.0
        self.dragging = None
        self.drag_threshold_px = 9

        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)

        log_label = ttk.Label(bottom_frame, text="📋  Output Log",
                             style='SectionHeader.TLabel')
        log_label.grid(row=0, column=0, sticky="w")
        self.log_widget = ScrolledText(bottom_frame, width=80, height=12, state="disabled", wrap="word",
                                       font=('Consolas', 9))
        self.log_widget.grid(row=1, column=0, sticky="nsew", padx=(0,8))
        try:
            self.log_widget.config(bg=self.BG_SECONDARY, fg=self.TEXT_PRIMARY,
                                  insertbackground=self.TEXT_PRIMARY,
                                  selectbackground=self.SELECT_BG, selectforeground='#ffffff',
                                  relief='flat', bd=1)
        except Exception:
            pass

        right_status = ttk.Frame(bottom_frame)
        try:
            right_status.configure(style='TFrame')
            right_status.config(bg=self.BG_DARK)
        except Exception:
            pass
        right_status.grid(row=1, column=1, sticky="nsew")
        right_status.columnconfigure(0, weight=1)
        ttk.Label(right_status, text="ℹ️  Status", style='SectionHeader.TLabel').grid(row=0, column=0, sticky="w")
        self.preview_info = ttk.Label(right_status, text="No preview", style='SubHeader.TLabel')
        self.preview_info.grid(row=1, column=0, sticky="nw")
        self.font_info = ttk.Label(right_status, text="", style='SubHeader.TLabel')
        self.font_info.grid(row=2, column=0, sticky="nw", pady=(6,0))

        # internal job list
        self.jobs = []

        self.root.after(200, self.poll_queue)
        try:
            self._update_color_canvases()
        except Exception:
            pass


        data = load_crop_settings()
        if data:
            try:
                self.top_percent_var.set(float(data.get("top_pct", CROP_TOP_RATIO*100)))
                self.bottom_percent_var.set(float(data.get("bottom_pct", CROP_BOTTOM_RATIO*100)))
                self.top_label.config(text=f"{self.top_percent_var.get():.1f}%")
                self.bottom_label.config(text=f"{self.bottom_percent_var.get():.1f}%")
                self.use_custom_crop_var.set(True)
            except Exception:
                pass

        self.time_update_after_id = None
        # last time we triggered a preview update (used for simple throttling)
        self._last_time_update_ts = 0.0
        # cached preview clip for faster scrubbing
        self._preview_clip = None
        # ensure we cleanup moviepy processes on window close
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception:
            pass

        # persistent mini-preview request handling
        self._mini_request_time = None
        try:
            self._mini_request_event = threading.Event()
        except Exception:
            self._mini_request_event = None
        self._mini_persistent_thread = None
        self._mini_request_stop = False
        self._mini_update_worker_async()

    def _rgba_from_hex(self, hx):
        try:
            if hx.startswith('#'): hx = hx[1:]
            if len(hx) == 6:
                r = int(hx[0:2],16); g = int(hx[2:4],16); b = int(hx[4:6],16)
                return (r,g,b,255)
            if len(hx) == 8:
                r = int(hx[0:2],16); g = int(hx[2:4],16); b = int(hx[4:6],16); a = int(hx[6:8],16)
                return (r,g,b,a)
        except Exception:
            pass
        return (255,255,255,255)

    def _update_color_canvases(self):
        try:
            try:
                txt = CAPTION_TEXT_COLOR
                col = '#%02x%02x%02x' % (txt[0], txt[1], txt[2])
                if hasattr(self, 'text_color_canvas') and self.text_color_canvas:
                    self.text_color_canvas.delete('all')
                    self.text_color_canvas.create_oval(2,2,26,26, fill=col, outline='white')
            except Exception:
                pass
            try:
                st = CAPTION_STROKE_COLOR
                col2 = '#%02x%02x%02x' % (st[0], st[1], st[2])
                if hasattr(self, 'stroke_color_canvas') and self.stroke_color_canvas:
                    self.stroke_color_canvas.delete('all')
                    self.stroke_color_canvas.create_oval(2,2,26,26, fill=col2, outline='white')
            except Exception:
                pass
        except Exception:
            pass

    def on_pick_text_color(self):
        try:
            from tkinter import colorchooser
            col = colorchooser.askcolor(title='Choose caption text color')
            if col and col[1]:
                rgba = self._rgba_from_hex(col[1])
                globals()['CAPTION_TEXT_COLOR'] = rgba
                self._update_color_canvases()
                # Trigger live preview update
                self._mini_update_worker_async()
        except Exception as e:
            try:
                self.log_widget.config(state='normal'); self.log_widget.insert('end', f"[COLOR-ERR] {e}\n"); self.log_widget.config(state='disabled')
            except Exception:
                pass

    def on_pick_stroke_color(self):
        try:
            from tkinter import colorchooser
            col = colorchooser.askcolor(title='Choose caption stroke color')
            if col and col[1]:
                rgba = self._rgba_from_hex(col[1])
                if len(rgba) == 4 and rgba[3] == 255:
                    rgba = (rgba[0], rgba[1], rgba[2], 150)
                globals()['CAPTION_STROKE_COLOR'] = rgba
                self._update_color_canvases()
                # Trigger live preview update
                self._mini_update_worker_async()
        except Exception as e:
            try:
                self.log_widget.config(state='normal'); self.log_widget.insert('end', f"[COLOR-ERR] {e}\n"); self.log_widget.config(state='disabled')
            except Exception:
                pass

    def _set_text_color_hex(self, hx):
        try:
            rgba = self._rgba_from_hex(hx)
            globals()['CAPTION_TEXT_COLOR'] = rgba
            self._update_color_canvases()
            # Trigger live preview update
            self._mini_update_worker_async()
        except Exception:
            pass

    def _set_stroke_color_hex(self, hx):
        try:
            rgba = self._rgba_from_hex(hx)
            rgba = (rgba[0], rgba[1], rgba[2], 150)
            globals()['CAPTION_STROKE_COLOR'] = rgba
            self._update_color_canvases()
            # Trigger live preview update
            self._mini_update_worker_async()
        except Exception:
            pass


    


    def on_stroke_width_changed(self, val):
        try:
            # val comes as string; set global and update label
            try:
                w = int(float(val))
            except Exception:
                try:
                    w = int(val)
                except Exception:
                    w = max(1, int(CAPTION_FONT_SIZE * 0.05))
            globals()['CAPTION_STROKE_WIDTH'] = max(0, w)
            try:
                if hasattr(self, 'stroke_width_label') and self.stroke_width_label:
                    self.stroke_width_label.config(text=str(int(globals().get('CAPTION_STROKE_WIDTH', w))))
            except Exception:
                pass
        except Exception as e:
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[STROKE-ERR] {e}\n")
                self.log_widget.config(state='disabled')
            except Exception:
                pass

    def on_voice_gain_changed(self, val):
        """Callback when voice volume slider changes — displays percentage + dB"""
        try:
            gain = float(val)
            globals()['VOICE_GAIN'] = gain
            if hasattr(self, 'voice_gain_label') and self.voice_gain_label:
                self.voice_gain_label.config(text=_gain_to_display(gain))
        except Exception:
            pass

    def on_music_gain_changed(self, val):
        """Callback when music volume slider changes — displays percentage + dB"""
        try:
            gain = float(val)
            globals()['MUSIC_GAIN'] = gain
            if hasattr(self, 'music_gain_label') and self.music_gain_label:
                self.music_gain_label.config(text=_gain_to_display(gain))
        except Exception:
            pass

    def on_tts_speed_changed(self, val):
        """Callback when TTS speed slider changes — displays speed multiplier"""
        try:
            speed = float(val)
            globals()['TTS_SPEED'] = speed
            if hasattr(self, 'tts_speed_label') and self.tts_speed_label:
                self.tts_speed_label.config(text=f"{speed:.2f}x")
        except Exception:
            pass

    # ── Lock / scroll-protection helpers ──────────────────────────────
    def _register_lockable(self, lock_key, label_widget, target_widget, is_combo_readonly=False):
        """Register a widget as lockable.  Adds 🔓 prefix + click handler to the label."""
        self._lock_vars[lock_key] = tk.BooleanVar(value=False)
        self._lock_labels[lock_key] = label_widget
        self._lock_target_widgets[lock_key] = (target_widget, is_combo_readonly)
        # Make the label a clickable lock toggle
        try:
            label_widget.config(cursor="hand2")
        except Exception:
            pass
        label_widget.bind("<Button-1>", lambda e, k=lock_key: self._toggle_lock(k))
        old_text = label_widget.cget("text")
        label_widget.config(text=f"🔓{old_text}")
        # Block scroll wheel on the target widget
        self._block_widget_scroll(target_widget)

    def _toggle_lock(self, lock_key):
        """Toggle lock state for a registered widget."""
        if lock_key not in self._lock_vars:
            return
        var = self._lock_vars[lock_key]
        new_state = not var.get()
        var.set(new_state)
        self._apply_lock(lock_key, new_state)

    def _apply_lock(self, lock_key, locked):
        """Apply lock/unlock visual + state to a widget."""
        if lock_key not in self._lock_labels:
            return
        lbl = self._lock_labels[lock_key]
        widget, is_combo_readonly = self._lock_target_widgets[lock_key]
        old_text = lbl.cget("text")
        if locked:
            lbl.config(text=old_text.replace("🔓", "🔒"))
            try:
                widget.config(state='disabled')
            except Exception:
                pass
        else:
            lbl.config(text=old_text.replace("🔒", "🔓"))
            try:
                if is_combo_readonly:
                    widget.config(state='readonly')
                else:
                    widget.config(state='normal')
            except Exception:
                pass

    def _block_widget_scroll(self, widget):
        """Prevent scroll wheel from changing a Scale/Combobox/Spinbox value.
        Redirects the event to scroll the left canvas instead."""
        def _redirect(event):
            if self._left_canvas:
                self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"
        widget.bind("<MouseWheel>", _redirect)

    def _setup_scroll_protection(self):
        """Block scroll wheel on ALL Scale/Combobox/Spinbox widgets in the left panel
        that haven't been explicitly registered as lockable (those are already blocked)."""
        already_blocked = set()
        for _k, (w, _c) in self._lock_target_widgets.items():
            already_blocked.add(str(w))
        def _recurse(parent):
            for child in parent.winfo_children():
                if isinstance(child, (tk.Scale, ttk.Combobox, ttk.Spinbox)):
                    if str(child) not in already_blocked:
                        self._block_widget_scroll(child)
                _recurse(child)
        if self._left_canvas:
            # The left_frame is the first (and only) window inside the canvas
            try:
                frame_ids = self._left_canvas.find_all()
                if frame_ids:
                    frame_widget = self._left_canvas.nametowidget(
                        self._left_canvas.itemcget(frame_ids[0], 'window'))
                    _recurse(frame_widget)
            except Exception:
                pass
    # ── end lock helpers ──────────────────────────────────────────────

    def _preview_audio_mix(self):
        """Preview voice + music mix at current volume levels using FFmpeg/ffplay."""
        try:
            voice_path = self.voice_var.get().strip() if hasattr(self, 'voice_var') else ""
            music_path = self.music_var.get().strip() if hasattr(self, 'music_var') else ""

            if not voice_path and not music_path:
                messagebox.showwarning("No Audio", "Please select at least a voice or music file to preview.")
                return

            for p, label in [(voice_path, "Voice"), (music_path, "Music")]:
                if p and not os.path.isfile(p):
                    messagebox.showerror("File Not Found", f"{label} file not found:\n{p}")
                    return

            voice_gain = self.voice_gain_var.get()
            music_gain = self.music_gain_var.get()

            if hasattr(self, 'log_to_console'):
                self.log_to_console(f"[PREVIEW] Generating audio preview (voice={voice_gain:.2f}x, music={music_gain:.2f}x)...")

            # Build FFmpeg command to mix audio and pipe to ffplay
            # Use first 15 seconds for quick preview
            preview_duration = 15
            tmp_dir = tempfile.gettempdir()
            preview_path = os.path.join(tmp_dir, "_tiktok_preview_mix.wav")

            # Clean up any old preview file
            if os.path.exists(preview_path):
                try:
                    os.remove(preview_path)
                except Exception:
                    pass

            if voice_path and music_path:
                # Mix both: voice at voice_gain, music at music_gain
                cmd = [
                    "ffmpeg", "-y",
                    "-i", voice_path,
                    "-i", music_path,
                    "-t", str(preview_duration),
                    "-filter_complex",
                    f"[0:a]volume={voice_gain}[v];[1:a]volume={music_gain}[m];[v][m]amix=inputs=2:duration=shortest:normalize=0,alimiter=limit=0.95",
                    "-ac", "2", "-ar", "44100",
                    preview_path
                ]
            elif voice_path:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", voice_path,
                    "-t", str(preview_duration),
                    "-af", f"volume={voice_gain}",
                    "-ac", "2", "-ar", "44100",
                    preview_path
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", music_path,
                    "-t", str(preview_duration),
                    "-af", f"volume={music_gain}",
                    "-ac", "2", "-ar", "44100",
                    preview_path
                ]

            def _run_preview():
                try:
                    # Generate mixed audio file
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode != 0 or not os.path.isfile(preview_path):
                        self.root.after(0, lambda: messagebox.showerror("Preview Error",
                            f"FFmpeg failed to generate preview.\n\n{result.stderr[:500] if result.stderr else 'Unknown error'}"))
                        return

                    # Get actual duration of generated file
                    dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "default=noprint_wrappers=1:nokey=1", preview_path]
                    dur_result = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=10)
                    try:
                        actual_duration = float(dur_result.stdout.strip())
                    except (ValueError, AttributeError):
                        actual_duration = preview_duration

                    if hasattr(self, 'log_to_console'):
                        self.root.after(0, lambda: self.log_to_console(f"[PREVIEW] Playing {actual_duration:.1f}s audio preview..."))

                    # Open preview playback window on main thread
                    self.root.after(0, lambda: self._open_preview_player(preview_path, actual_duration))

                except subprocess.TimeoutExpired:
                    self.root.after(0, lambda: messagebox.showerror("Preview Error", "FFmpeg timed out generating the audio preview."))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Preview Error", f"Error: {e}"))

            threading.Thread(target=_run_preview, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Preview Error", f"Failed to start preview:\n{e}")

    def _open_preview_player(self, audio_path, duration):
        """Open a Toplevel window with a progress bar and stop button for audio preview."""
        try:
            # Kill any previous preview
            if hasattr(self, '_preview_process') and self._preview_process:
                try:
                    self._preview_process.terminate()
                    self._preview_process.wait(timeout=2)
                except Exception:
                    pass
                self._preview_process = None
            if hasattr(self, '_preview_window') and self._preview_window:
                try:
                    self._preview_window.destroy()
                except Exception:
                    pass

            # Create playback window
            win = tk.Toplevel(self.root)
            win.title("🔊 Audio Preview")
            win.geometry("420x140")
            win.resizable(False, False)
            win.attributes('-topmost', True)
            self._preview_window = win

            # Volume info
            voice_gain = self.voice_gain_var.get()
            music_gain = self.music_gain_var.get()
            info_text = f"Voice: {_gain_to_display(voice_gain)}   |   Music: {_gain_to_display(music_gain)}"
            ttk.Label(win, text=info_text, font=('Segoe UI', 9)).pack(pady=(10, 4))

            # Progress bar
            progress_var = tk.DoubleVar(value=0)
            progress_bar = ttk.Progressbar(win, variable=progress_var, maximum=duration, length=380, mode='determinate')
            progress_bar.pack(padx=20, pady=(4, 4))

            # Time label
            time_label = ttk.Label(win, text=f"0:00 / {int(duration)//60}:{int(duration)%60:02d}", font=('Segoe UI', 9))
            time_label.pack()

            # Stop button
            btn_frame = ttk.Frame(win)
            btn_frame.pack(pady=(6, 8))
            stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=lambda: self._stop_preview(win))
            stop_btn.pack()

            # Start ffplay process
            self._preview_uses_timer_only = False
            play_cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]
            try:
                self._preview_process = subprocess.Popen(play_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                # ffplay not available, try system default player
                if hasattr(self, 'log_to_console'):
                    self.log_to_console("[PREVIEW] ffplay not found, trying system player...")
                try:
                    if sys.platform == "win32":
                        os.startfile(audio_path)
                        self._preview_process = None
                        self._preview_uses_timer_only = True
                    elif sys.platform == "darwin":
                        self._preview_process = subprocess.Popen(["afplay", audio_path])
                    else:
                        self._preview_process = subprocess.Popen(["xdg-open", audio_path])
                except Exception as e:
                    messagebox.showerror("Preview Error", f"No audio player available.\nInstall ffplay (part of FFmpeg) to use audio preview.\n\n{e}")
                    win.destroy()
                    return

            self._preview_start_time = time.time()
            self._preview_duration = duration

            def _update_progress():
                try:
                    if not win.winfo_exists():
                        return
                    elapsed = time.time() - self._preview_start_time
                    if elapsed > duration:
                        elapsed = duration
                    progress_var.set(elapsed)
                    mins = int(elapsed) // 60
                    secs = int(elapsed) % 60
                    total_mins = int(duration) // 60
                    total_secs = int(duration) % 60
                    time_label.config(text=f"{mins}:{secs:02d} / {total_mins}:{total_secs:02d}")

                    # Check if process still running
                    proc = getattr(self, '_preview_process', None)
                    timer_only = getattr(self, '_preview_uses_timer_only', False)
                    if timer_only:
                        # No process to poll — use elapsed time
                        if elapsed >= duration:
                            progress_var.set(duration)
                            time_label.config(text=f"{total_mins}:{total_secs:02d} / {total_mins}:{total_secs:02d}")
                            stop_btn.config(text="✓ Done")
                            if hasattr(self, 'log_to_console'):
                                self.log_to_console("[PREVIEW] Playback finished.")
                            return
                    elif proc and proc.poll() is not None:
                        # Playback finished
                        progress_var.set(duration)
                        time_label.config(text=f"{total_mins}:{total_secs:02d} / {total_mins}:{total_secs:02d}")
                        stop_btn.config(text="✓ Done")
                        if hasattr(self, 'log_to_console'):
                            self.log_to_console("[PREVIEW] Playback finished.")
                        return

                    win.after(200, _update_progress)
                except Exception:
                    pass

            win.after(200, _update_progress)

            # Handle window close
            def _on_close():
                self._stop_preview(win)
            win.protocol("WM_DELETE_WINDOW", _on_close)

        except Exception as e:
            if hasattr(self, 'log_to_console'):
                self.log_to_console(f"[PREVIEW] Error: {e}")

    def _stop_preview(self, win=None):
        """Stop audio preview playback and close the window."""
        try:
            proc = getattr(self, '_preview_process', None)
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
            self._preview_process = None
            if win:
                try:
                    win.destroy()
                except Exception:
                    pass
            self._preview_window = None
            if hasattr(self, 'log_to_console'):
                self.log_to_console("[PREVIEW] Stopped.")
        except Exception:
            pass

    def on_translation_toggle(self):
        """Callback when translation checkbox is toggled"""
        try:
            enabled = self.translation_enabled_var.get()
            globals()['TRANSLATION_ENABLED'] = enabled
            if enabled:
                # Check if any translation backend is available
                has_openai = bool(globals().get('OPENAI_API_KEY')) and REQUESTS_AVAILABLE
                has_googletrans = TRANSLATION_AVAILABLE
                if not has_openai and not has_googletrans:
                    messagebox.showwarning(
                        "Translation Unavailable",
                        "No translation backend available.\n\n"
                        "Option 1: Set an OpenAI API key (recommended)\n"
                        "Option 2: Install googletrans: pip install googletrans==4.0.0rc1"
                    )
                    self.translation_enabled_var.set(False)
                    globals()['TRANSLATION_ENABLED'] = False
                elif has_openai and not has_googletrans:
                    self.log_to_console("[Translation] ✓ Using OpenAI for translation (googletrans not installed)")
                elif has_openai:
                    self.log_to_console("[Translation] ✓ Using OpenAI for translation (primary)")
        except Exception as e:
            print(f"Translation toggle error: {e}")
    
    def _apply_openai_key(self):
        """Apply the OpenAI API key from the GUI entry field."""
        try:
            key = self.openai_key_var.get().strip()
            if key:
                globals()['OPENAI_API_KEY'] = key
                if hasattr(self, 'log_to_console'):
                    self.log_to_console("[OpenAI] API key set - contextual translations enabled")
                    self.log_to_console("[OpenAI] Use 'Verify' button to test your key")
                    self.log_to_console("[OpenAI] API usage visible at: https://platform.openai.com/usage")
            else:
                globals()['OPENAI_API_KEY'] = None
                if hasattr(self, 'log_to_console'):
                    self.log_to_console("[OpenAI] API key cleared - using googletrans")
        except Exception as e:
            print(f"OpenAI key apply error: {e}")

    def _save_openai_key(self):
        """Save the OpenAI API key to a config file and apply it."""
        try:
            key = self.openai_key_var.get().strip()
            if not key:
                messagebox.showwarning("No API Key", "Please enter an OpenAI API key.")
                return
            # Apply the key first
            globals()['OPENAI_API_KEY'] = key
            # Save to config file (preserve existing model setting)
            config_path = os.path.join(os.path.dirname(__file__), "openai_config.json")
            config = {}
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
            except Exception:
                pass
            config["openai_api_key"] = key
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f)
                # Restrict file permissions to owner only (API key security)
                try:
                    os.chmod(config_path, 0o600)
                except Exception:
                    pass  # Windows doesn't support chmod the same way
                if hasattr(self, 'log_to_console'):
                    self.log_to_console("[OpenAI] API key saved and activated")
                messagebox.showinfo("API Key Saved", "OpenAI API key saved successfully!")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save OpenAI API key: {e}")
        except Exception as e:
            print(f"Save OpenAI key error: {e}")

    def _apply_openai_model(self):
        """Apply the OpenAI model selection from the GUI combo box."""
        try:
            model = self.openai_model_var.get().strip()
            if model:
                globals()['OPENAI_MODEL'] = model
                if hasattr(self, 'log_to_console'):
                    self.log_to_console(f"[OpenAI] Model set to: {model}")
            else:
                globals()['OPENAI_MODEL'] = 'gpt-4o-mini'
                if hasattr(self, 'log_to_console'):
                    self.log_to_console("[OpenAI] Model reset to default: gpt-4o-mini")
        except Exception as e:
            print(f"OpenAI model apply error: {e}")

    def _save_openai_model(self):
        """Save the OpenAI model selection to the config file and apply it."""
        try:
            model = self.openai_model_var.get().strip()
            if not model:
                model = 'gpt-4o-mini'
            globals()['OPENAI_MODEL'] = model
            # Save to config file (preserve existing key setting)
            config_path = os.path.join(os.path.dirname(__file__), "openai_config.json")
            config = {}
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
            except Exception:
                pass
            config["openai_model"] = model
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f)
                try:
                    os.chmod(config_path, 0o600)
                except Exception:
                    pass
                if hasattr(self, 'log_to_console'):
                    self.log_to_console(f"[OpenAI] Model saved: {model}")
                messagebox.showinfo("Model Saved", f"OpenAI model saved: {model}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save OpenAI model: {e}")
        except Exception as e:
            print(f"Save OpenAI model error: {e}")

    def _verify_openai_key(self):
        """Verify the OpenAI API key works by making a small test API call."""
        try:
            key = self.openai_key_var.get().strip()
            if not key:
                messagebox.showwarning("No API Key", "Please enter an OpenAI API key first.")
                return
            if not REQUESTS_AVAILABLE:
                messagebox.showerror("Missing Library", "The 'requests' library is required. Install with: pip install requests")
                return
            import requests as _requests
            if hasattr(self, 'log_to_console'):
                self.log_to_console("[OpenAI] Verifying API key...")
            response = _requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-4o-mini',
                    'temperature': 0,
                    'max_tokens': 10,
                    'messages': [
                        {'role': 'user', 'content': 'Say OK'}
                    ]
                },
                timeout=15
            )
            if response.status_code == 200:
                globals()['OPENAI_API_KEY'] = key
                if hasattr(self, 'log_to_console'):
                    self.log_to_console(f"[OpenAI] ✓ API key is VALID - translations will use {globals().get('OPENAI_MODEL', 'gpt-4o-mini')}")
                    self.log_to_console("[OpenAI] NOTE: OpenAI is used ONLY for translation, NOT for voice generation (TTS)")
                    self.log_to_console("[OpenAI] NOTE: API usage is visible at https://platform.openai.com/usage")
                    self.log_to_console("[OpenAI] API calls do NOT appear on chat.openai.com (that is a different product)")
                messagebox.showinfo("API Key Valid",
                    f"✓ Your OpenAI API key is working!\n\n"
                    f"• Translations will use {globals().get('OPENAI_MODEL', 'gpt-4o-mini')}\n"
                    "• Voice generation uses GenAI Pro (NOT OpenAI)\n\n"
                    "IMPORTANT: API calls do NOT appear on chat.openai.com.\n"
                    "Check your API usage at:\nhttps://platform.openai.com/usage")
            elif response.status_code == 401:
                if hasattr(self, 'log_to_console'):
                    self.log_to_console("[OpenAI] ✗ API key is INVALID or expired")
                messagebox.showerror("Invalid API Key",
                    "✗ Your OpenAI API key is invalid or expired.\n\n"
                    "Please check your key at:\nhttps://platform.openai.com/api-keys")
            elif response.status_code == 429:
                if hasattr(self, 'log_to_console'):
                    self.log_to_console("[OpenAI] ✗ Rate limit or quota exceeded (HTTP 429)")
                    self.log_to_console("[OpenAI] FIX: Go to https://platform.openai.com/settings/organization/billing")
                    self.log_to_console("[OpenAI] and add at least $5 prepaid credit. Free-tier keys have no quota.")
                messagebox.showwarning("Billing Credit Required",
                    "Your API key is valid but has no billing credit.\n\n"
                    "Even brand-new keys need prepaid credit to work.\n"
                    "OpenAI free-tier keys have $0 quota by default.\n\n"
                    "FIX: Add at least $5 credit at:\n"
                    "https://platform.openai.com/settings/organization/billing\n\n"
                    "After adding credit, click 'Verify' again.")
            else:
                err_text = response.text[:200] + ('...' if len(response.text) > 200 else '')
                if hasattr(self, 'log_to_console'):
                    self.log_to_console(f"[OpenAI] ✗ API returned status {response.status_code}: {err_text}")
                messagebox.showerror("API Error",
                    f"API returned status {response.status_code}.\n\n{err_text}")
        except Exception as e:
            if hasattr(self, 'log_to_console'):
                self.log_to_console(f"[OpenAI] ✗ Verification failed: {e}")
            messagebox.showerror("Connection Error",
                f"Could not connect to OpenAI API:\n{e}")

    def _check_genaipro_api(self):
        """Silently verify the GenAI Pro API key on startup by calling GET /v1/labs/task?limit=1.
        Runs in a background thread so the UI is never blocked."""
        import threading

        def _worker():
            try:
                api_key = _get_genaipro_api_key()
                if not api_key:
                    return  # No key configured — nothing to check
                if not REQUESTS_AVAILABLE:
                    self.root.after(0, lambda: self.log_to_console(
                        "[GenAI Pro] ⚠ 'requests' library not available — cannot verify API key"))
                    return
                import requests as _requests
                session = _genaipro_session(_requests)
                response = session.get(
                    f'{GENAIPRO_API_BASE}/v1/labs/task?limit=1',
                    headers={
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json',
                    },
                    timeout=15,
                )
                if response.status_code == 200:
                    self.root.after(0, lambda: self.log_to_console(
                        "[GenAI Pro] ✓ API key is VALID — voice generation is ready"))
                elif response.status_code == 401:
                    self.root.after(0, lambda: self.log_to_console(
                        "[GenAI Pro] ✗ API key is INVALID or expired — please update tts_config.json"))
                else:
                    self.root.after(0, lambda: self.log_to_console(
                        f"[GenAI Pro] ⚠ API check returned HTTP {response.status_code} — key may still work"))
            except Exception as e:
                self.root.after(0, lambda: self.log_to_console(
                    f"[GenAI Pro] ⚠ Could not reach API on startup: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_translation_prompt(self):
        """Apply the custom translation prompt from the GUI entry field."""
        try:
            prompt = self.translation_prompt_var.get().strip()
            globals()['TRANSLATION_CUSTOM_PROMPT'] = prompt
            if hasattr(self, 'log_to_console'):
                if prompt:
                    self.log_to_console(f"[Translation] Custom prompt set: {prompt[:60]}...")
                else:
                    self.log_to_console("[Translation] Custom prompt cleared - using default")
        except Exception as e:
            print(f"Translation prompt apply error: {e}")
    
    def on_language_selected(self, event=None):
        """Callback when target language is selected. Auto-syncs TTS language when TTS is enabled."""
        try:
            lang = self.target_language_var.get()
            globals()['TARGET_LANGUAGE'] = lang
            # Auto-sync TTS language with translation target when TTS is enabled
            if hasattr(self, 'use_ai_voice_var') and self.use_ai_voice_var.get() and lang != 'none':
                tts_lang = TRANS_TO_TTS_LANG.get(lang, lang)
                if hasattr(self, 'tts_language_var'):
                    self.tts_language_var.set(tts_lang)
                    globals()['TTS_LANGUAGE'] = tts_lang
        except Exception as e:
            print(f"Language selection error: {e}")
    
    def on_ai_voice_toggle(self):
        """Callback when AI voice replacement checkbox is toggled"""
        try:
            enabled = self.use_ai_voice_var.get()
            globals()['USE_AI_VOICE_REPLACEMENT'] = enabled
            if enabled:
                if not _get_genaipro_api_key():
                    messagebox.showwarning(
                        "No TTS Engine Available",
                        "GenAI Pro API key not found.\n"
                        "Please configure tts_config.json with a valid API key."
                    )
                    self.use_ai_voice_var.set(False)
                    globals()['USE_AI_VOICE_REPLACEMENT'] = False
        except Exception as e:
            print(f"AI voice toggle error: {e}")
    
    def on_tts_language_selected(self, event=None):
        """Callback when TTS language is selected"""
        try:
            lang = self.tts_language_var.get()
            globals()['TTS_LANGUAGE'] = lang
            
            # Update voice dropdown based on selected language (including custom voices)
            self.update_voice_dropdown(lang)
        except Exception as e:
            print(f"TTS language selection error: {e}")
    
    def load_custom_voices(self):
        """Load custom voices from config file"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "tts_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get("custom_voices", {})
        except Exception as e:
            print(f"Error loading custom voices: {e}")
        return {}
    
    def update_voice_dropdown(self, language=None):
        """Update voice dropdown with default and custom voices for the selected language"""
        try:
            if language is None:
                language = self.tts_language_var.get()
            
            # Start with default voices
            voices = self.voice_options.get(language, ['Auto (Default)']).copy()
            
            # Load and add custom voices
            custom_voices = self.load_custom_voices()
            if language in custom_voices:
                for voice_data in custom_voices[language]:
                    if isinstance(voice_data, dict):
                        voice_name = voice_data.get("name")
                        voice_id = voice_data.get("id")
                        if voice_name and voice_id:
                            # Add to voices list if not already there
                            if voice_name not in voices:
                                voices.append(voice_name)
                            # Update the voice ID map
                            self.voice_id_map[voice_name] = voice_id
            
            # Update the combobox – preserve current selection if still valid
            self.tts_voice_combo['values'] = voices
            current = self.tts_voice_var.get()
            if current not in voices:
                self.tts_voice_var.set('Auto (Default)')
                globals()['TTS_VOICE_ID'] = 'auto'
            else:
                # Re-sync the global in case it was stale
                globals()['TTS_VOICE_ID'] = self.voice_id_map.get(current, 'auto')
            
            # Also update the fallback voice dropdown with the same voice list
            if hasattr(self, 'tts_fallback_voice_combo'):
                self.tts_fallback_voice_combo['values'] = voices
                fb_current = self.tts_fallback_voice_var.get()
                if fb_current not in voices:
                    self.tts_fallback_voice_var.set('Auto (Default)')
                    globals()['TTS_FALLBACK_VOICE_ID'] = 'auto'
                else:
                    globals()['TTS_FALLBACK_VOICE_ID'] = self.voice_id_map.get(fb_current, 'auto')
            
            print(f"[Voice Dropdown] Updated for {language}: {len(voices)} voices available")
        except Exception as e:
            print(f"Update voice dropdown error: {e}")
    
    def on_voice_selected(self, event=None):
        """Callback when a specific voice is selected"""
        try:
            voice_name = self.tts_voice_var.get()
            voice_id = self.voice_id_map.get(voice_name, 'auto')
            globals()['TTS_VOICE_ID'] = voice_id
            print(f"[Voice Selection] Selected: {voice_name} (ID: {voice_id})")
        except Exception as e:
            print(f"Voice selection error: {e}")
    
    def on_fallback_voice_selected(self, event=None):
        """Callback when a fallback voice is selected"""
        try:
            voice_name = self.tts_fallback_voice_var.get()
            voice_id = self.voice_id_map.get(voice_name, 'auto')
            globals()['TTS_FALLBACK_VOICE_ID'] = voice_id
            print(f"[Fallback Voice] Selected: {voice_name} (ID: {voice_id})")
        except Exception as e:
            print(f"Fallback voice selection error: {e}")
    
    def on_save_api_key(self):
        """Callback when Save API Key button is clicked"""
        try:
            api_key = self.tts_api_key_var.get().strip()
            if not api_key:
                messagebox.showwarning("No API Key", "Please enter an API key.")
                return
            
            api_key = _normalize_genaipro_api_key(api_key)
            
            # Save to config file — read existing config first to preserve custom_voices and other data
            config_path = os.path.join(os.path.dirname(__file__), "tts_config.json")
            config = {}
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
            except Exception:
                pass
            config["api_key"] = api_key
            
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                preview = api_key[:8] + "..." if len(api_key) > 8 else api_key
                messagebox.showinfo("API Key Saved", f"Your API key has been saved successfully!\n\nToken starts with: {preview}\nConfig: {config_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save API key: {e}")
        except Exception as e:
            print(f"Save API key error: {e}")
    
    def on_save_custom_voice(self):
        """Callback when Save Custom Voice button is clicked"""
        try:
            voice_id = self.custom_voice_id_var.get().strip()
            voice_name = self.custom_voice_name_var.get().strip()
            language = self.custom_voice_lang_var.get()
            
            if not voice_id:
                messagebox.showwarning("No Voice ID", "Please enter a voice ID.")
                return
            
            if not voice_name:
                messagebox.showwarning("No Voice Name", "Please enter a name for this voice.")
                return
            
            # Load existing config
            config_path = os.path.join(os.path.dirname(__file__), "tts_config.json")
            config = {}
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
            except Exception:
                pass
            
            # Add custom voices section if not exists
            if "custom_voices" not in config:
                config["custom_voices"] = {}
            
            # Add the custom voice
            if language not in config["custom_voices"]:
                config["custom_voices"][language] = []
            
            # Check if voice name already exists for this language
            existing_voices = config["custom_voices"][language]
            voice_exists = any(v["name"] == voice_name for v in existing_voices if isinstance(v, dict))
            
            if voice_exists:
                response = messagebox.askyesno(
                    "Voice Exists", 
                    f"A voice named '{voice_name}' already exists for {language}. Do you want to update it?"
                )
                if not response:
                    return
                # Remove the old one
                config["custom_voices"][language] = [v for v in existing_voices if not (isinstance(v, dict) and v.get("name") == voice_name)]
            
            # Add the new voice
            config["custom_voices"][language].append({
                "name": voice_name,
                "id": voice_id
            })
            
            # Save config
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Custom Voice Saved", f"Voice '{voice_name}' has been saved for {language.upper()}!")
                
                # Clear the input fields
                self.custom_voice_id_var.set("")
                self.custom_voice_name_var.set("")
                
                # Reload voices in the dropdown
                self.update_voice_dropdown()
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save custom voice: {e}")
        except Exception as e:
            print(f"Save custom voice error: {e}")
    
    def on_4k_toggle(self):
        """Toggle between HD (1080x1920) and 4K (2160x3840) resolution"""
        try:
            use_4k = self.use_4k_var.get()
            if use_4k:
                # 4K is 2x HD resolution
                globals()['WIDTH'] = 2160
                globals()['HEIGHT'] = 3840
                # Scale caption font size for 4K (2x)
                globals()['CAPTION_FONT_SIZE'] = 112  # 56 * 2
                # Update stroke width based on new font size
                globals()['CAPTION_STROKE_WIDTH'] = max(1, int(112 * 0.05))
                # Scale background blur radius for 4K (2x)
                globals()['STATIC_BG_BLUR_RADIUS'] = 50  # 25 * 2
                # Scale background zoom factor for 4K
                globals()['BG_SCALE_EXTRA'] = 1.08  # Keep same for both (proportional to resolution)
                # Mark that we're in 4K mode for scaling calculations
                globals()['IS_4K_MODE'] = True
                # Sync caption font size slider and label with 4K value
                if hasattr(self, 'caption_font_size_scale'):
                    self.caption_font_size_scale.config(from_=40, to=800)
                if hasattr(self, 'caption_font_size_var'):
                    self.caption_font_size_var.set(112)
                if hasattr(self, 'caption_font_size_label') and self.caption_font_size_label:
                    self.caption_font_size_label.config(text="112px")
                # Update caption Y offset slider range for 4K (-3840 to +200)
                if hasattr(self, 'caption_y_offset_scale'):
                    self.caption_y_offset_scale.config(from_=-3840, to=200)
                    # Scale current offset value (multiply by 2)
                    current_offset = self.caption_y_offset_var.get()
                    self.caption_y_offset_var.set(current_offset * 2)
                    globals()['CAPTION_Y_OFFSET'] = current_offset * 2
                if hasattr(self, 'caption_y_offset_spinbox'):
                    self.caption_y_offset_spinbox.config(from_=-3840, to=200)
                # Log resolution change
                try:
                    self.log_widget.config(state='normal')
                    self.log_widget.insert('end', "Resolution set to 4K: 2160x3840\n")
                    self.log_widget.insert('end', "Caption font size scaled to: 112px\n")
                    self.log_widget.insert('end', f"Background blur scaled to: {globals()['STATIC_BG_BLUR_RADIUS']}\n")
                    self.log_widget.insert('end', "Zoom/scale factors adjusted for 4K\n")
                    self.log_widget.see('end')
                    self.log_widget.config(state='disabled')
                except Exception:
                    pass
            else:
                # HD resolution
                globals()['WIDTH'] = 1080
                globals()['HEIGHT'] = 1920
                # Reset caption font size to default HD (56)
                globals()['CAPTION_FONT_SIZE'] = 56
                # Update stroke width based on default font size
                globals()['CAPTION_STROKE_WIDTH'] = max(1, int(56 * 0.05))
                # Reset background blur radius for HD
                globals()['STATIC_BG_BLUR_RADIUS'] = 25
                # Reset background zoom factor for HD
                globals()['BG_SCALE_EXTRA'] = 1.08
                # Mark that we're in HD mode
                globals()['IS_4K_MODE'] = False
                # Sync caption font size slider and label with HD value
                if hasattr(self, 'caption_font_size_scale'):
                    self.caption_font_size_scale.config(from_=20, to=120)
                if hasattr(self, 'caption_font_size_var'):
                    self.caption_font_size_var.set(56)
                if hasattr(self, 'caption_font_size_label') and self.caption_font_size_label:
                    self.caption_font_size_label.config(text="56px")
                # Reset caption Y offset slider range for HD (-1080 to +200)
                if hasattr(self, 'caption_y_offset_scale'):
                    self.caption_y_offset_scale.config(from_=-1920, to=200)
                    # Scale current offset value back (divide by 2)
                    current_offset = self.caption_y_offset_var.get()
                    self.caption_y_offset_var.set(current_offset // 2)
                    globals()['CAPTION_Y_OFFSET'] = current_offset // 2
                if hasattr(self, 'caption_y_offset_spinbox'):
                    self.caption_y_offset_spinbox.config(from_=-1920, to=200)
                # Log resolution change
                try:
                    self.log_widget.config(state='normal')
                    self.log_widget.insert('end', "Resolution set to HD: 1080x1920\n")
                    self.log_widget.insert('end', "Caption font size reset to: 56px\n")
                    self.log_widget.insert('end', f"Background blur reset to: {globals()['STATIC_BG_BLUR_RADIUS']}\n")
                    self.log_widget.insert('end', "Zoom/scale factors reset to HD\n")
                    self.log_widget.see('end')
                    self.log_widget.config(state='disabled')
                except Exception:
                    pass
        except Exception as e:
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[4K-TOGGLE-ERR] {e}\n")
                self.log_widget.see('end')
                self.log_widget.config(state='disabled')
            except Exception:
                pass

    def _draw_caption_indicator_on_preview(self, composed, h, top_y, bottom_y, offset):
        """Draw the caption position indicator and SAMPLE TEXT on the mini preview canvas.
        
        Args:
            composed: The composed PIL image
            h: Canvas height
            top_y: Top crop line Y position
            bottom_y: Bottom crop line Y position
            offset: Caption Y offset value
        """
        try:
            # DELETE old caption indicator items first to prevent stacking
            self.mini_canvas.delete("caption_line")
            self.mini_canvas.delete("caption_box")
            self.mini_canvas.delete("caption_label")
            self.mini_canvas.delete("caption_sample")
            self.mini_canvas.delete("caption_sample_stroke")
            
            # Calculate where caption will appear on the preview
            # offset: negative = move up, positive = move down
            # In actual video: y = HEIGHT - caption_height + offset
            # In preview: caption_y = h - (scaled distance from bottom)
            # distance_from_bottom in video = HEIGHT - (HEIGHT - caption_height + offset) = caption_height - offset
            # For indicator baseline (bottom of caption): distance_from_bottom = -offset (approximately, ignoring caption height)
            preview_ratio = h / HEIGHT if HEIGHT > 0 else 1.0
            caption_baseline_from_bottom = int(-offset * preview_ratio)  # Negative offset means higher (less from bottom)
            caption_y = h - caption_baseline_from_bottom
            
            # DO NOT clamp to crop lines - show actual caption position even if outside crop area
            # Only clamp to canvas boundaries (0 to h)
            caption_y = max(5, min(h - 5, caption_y))
            
            # Draw green dashed line showing caption baseline - PERMANENT
            self.mini_canvas.create_line(0, caption_y, composed.width, caption_y, 
                                        fill="#00FF00", dash=(6, 4), width=2, tags="caption_line")
            
            # Get current caption settings for the SAMPLE TEXT preview
            font_size = globals().get('CAPTION_FONT_SIZE', 56)
            text_color = globals().get('CAPTION_TEXT_COLOR', (255, 255, 255, 255))
            stroke_color = globals().get('CAPTION_STROKE_COLOR', (0, 0, 0, 255))
            stroke_width = globals().get('CAPTION_STROKE_WIDTH', 3)  # Get actual stroke width
            words_per_caption = globals().get('WORDS_PER_GROUP', 2)  # Get words per caption
            font_family = globals().get('LOADED_FONT_FAMILY', None) or globals().get('CAPTION_FONT_PREFERRED', 'Arial')
            
            # Scale font size for mini preview (mini canvas is much smaller than 1920px)
            # Mini canvas height is about 380px vs 1920px actual
            scaled_font_size = max(10, int(font_size * preview_ratio))
            
            # Scale stroke width for mini preview (proportional to font size scaling)
            scaled_stroke_width = max(1, int(stroke_width * preview_ratio))
            
            # Convert RGB tuple to hex color
            try:
                text_hex = '#%02x%02x%02x' % (text_color[0], text_color[1], text_color[2])
            except Exception:
                text_hex = '#FFFFFF'  # Default white
            
            try:
                stroke_hex = '#%02x%02x%02x' % (stroke_color[0], stroke_color[1], stroke_color[2])
            except Exception:
                stroke_hex = '#000000'  # Default black
            
            # Create font tuple for tkinter canvas
            # Use the loaded font family name, tkinter handles unknown fonts gracefully
            canvas_font = (font_family if font_family else "Arial", scaled_font_size, "bold")
            
            # Generate sample text based on words per caption setting (max 3 words)
            word_list = ["WORD", "ONE", "TWO"]
            sample_words = word_list[:min(words_per_caption, len(word_list))]
            sample_text = " ".join(sample_words)
            
            text_x = composed.width // 2
            text_y = caption_y - scaled_font_size // 2
            
            # Draw stroke (outline) effect by drawing text at the outer boundary only
            # This is more efficient than drawing at every pixel of the stroke width
            stroke_offsets = [
                (-scaled_stroke_width, -scaled_stroke_width), (-scaled_stroke_width, scaled_stroke_width),
                (scaled_stroke_width, -scaled_stroke_width), (scaled_stroke_width, scaled_stroke_width),
                (-scaled_stroke_width, 0), (scaled_stroke_width, 0),
                (0, -scaled_stroke_width), (0, scaled_stroke_width)
            ]
            
            for dx, dy in stroke_offsets:
                self.mini_canvas.create_text(text_x + dx, text_y + dy, 
                                            text=sample_text, 
                                            fill=stroke_hex, font=canvas_font, 
                                            tags="caption_sample_stroke")
            
            # Draw main caption text
            self.mini_canvas.create_text(text_x, text_y, 
                                        text=sample_text, 
                                        fill=text_hex, font=canvas_font, 
                                        tags="caption_sample")
            
            # Draw small info label below the sample text
            info_text = f"Y: {offset}px | Size: {font_size}px | Words: {words_per_caption}"
            self.mini_canvas.create_text(text_x, caption_y + 10, 
                                        text=info_text, 
                                        fill="#00FF00", font=("Arial", 8), 
                                        tags="caption_label")
                
        except Exception as e:
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[CAPTION-INDICATOR-ERR] {e}\n")
                self.log_widget.config(state='disabled')
            except Exception:
                pass

    def _redraw_mini_canvas_with_caption_indicator(self, composed, top_pct, bottom_pct):
        """Helper to redraw the entire mini canvas with crop lines and caption indicator.
        
        This ensures the caption indicator is ALWAYS drawn on every mini preview update.
        """
        try:
            photo = ImageTk.PhotoImage(composed)
            h = composed.height
            top_y = int(round(h * top_pct))
            bottom_y = int(round(h * (1.0 - bottom_pct)))
            
            # Clear and redraw canvas
            self.mini_canvas.delete("all")
            self.mini_canvas.create_image(0, 0, anchor="nw", image=photo, tags="base_img")
            self.mini_image_ref = photo  # Keep reference
            
            # Draw crop lines
            self.mini_canvas.create_rectangle(0, top_y-4, composed.width, top_y+4, 
                                            fill="#000000", stipple="gray50", tags="line_top")
            self.mini_canvas.create_rectangle(0, bottom_y-4, composed.width, bottom_y+4, 
                                            fill="#000000", stipple="gray50", tags="line_bottom")
            self.mini_canvas.create_text(6, max(6, top_y-18), anchor="nw", 
                                        text=f"Top {int(top_pct*100)}% ({top_y}px)", 
                                        fill="#fff", font=("Arial",9), tags="label_top")
            self.mini_canvas.create_text(6, min(h-18, bottom_y+6), anchor="nw", 
                                        text=f"Bottom {int(bottom_pct*100)}% ({h - bottom_y}px)", 
                                        fill="#fff", font=("Arial",9), tags="label_bottom")
            
            # Draw blur box outline on mini_canvas (before caption so captions stay on top)
            try:
                if getattr(self, 'blur_overlay_enabled_var', None) and self.blur_overlay_enabled_var.get():
                    bx = self.blur_overlay_x_var.get() / 100.0
                    by = self.blur_overlay_y_var.get() / 100.0
                    bw = self.blur_overlay_w_var.get() / 100.0
                    bh_frac = self.blur_overlay_h_var.get() / 100.0
                    bx1 = bx * composed.width
                    by1 = by * h
                    bx2 = (bx + bw) * composed.width
                    by2 = (by + bh_frac) * h
                    self.mini_canvas.create_rectangle(
                        bx1, by1, bx2, by2,
                        outline="#00CFFF", width=2, dash=(5, 3), tags="blur_box"
                    )
                    self.mini_canvas.create_text(
                        (bx1 + bx2) / 2, max(8, by1 - 4),
                        text="⇕ blur", fill="#00CFFF", font=("Arial", 8),
                        anchor="s", tags="blur_box_label"
                    )
            except Exception:
                pass

            # ALWAYS draw caption position indicator
            y_offset = globals().get('CAPTION_Y_OFFSET', 0)
            self._draw_caption_indicator_on_preview(composed, h, top_y, bottom_y, y_offset)
                
        except Exception as e:
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[MINI-REDRAW-ERR] {e}\n")
                self.log_widget.config(state='disabled')
            except Exception:
                pass

    def on_caption_position_changed(self, val):
        """Callback when caption vertical position slider changes (debounced)."""
        try:
            # val comes as string; set global and update label
            try:
                offset = int(float(val))
            except Exception:
                try:
                    offset = int(val)
                except Exception:
                    offset = 0
            globals()['CAPTION_Y_OFFSET'] = offset

            # Cancel any pending debounced preview update
            pending = getattr(self, '_caption_pos_debounce_id', None)
            if pending is not None:
                self.root.after_cancel(pending)
                self._caption_pos_debounce_id = None

            # Schedule the heavy preview update after a short delay (debounce)
            self._caption_pos_debounce_id = self.root.after(80, self._do_caption_position_update)
        except Exception:
            pass

    def _do_caption_position_update(self):
        """Deferred caption position preview update (called after debounce delay)."""
        self._caption_pos_debounce_id = None
        try:
            offset = globals().get('CAPTION_Y_OFFSET', 0)
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[CAPTION-POS-CHANGE] Y offset changed to: {offset}px\n")
                self.log_widget.config(state='disabled')
                self.log_widget.see('end')
            except Exception:
                pass

            # Update mini preview to show new caption position
            try:
                if hasattr(self, 'mini_base_img') and self.mini_base_img is not None:
                    top_pct = float(self.top_percent_var.get())/100.0
                    bottom_pct = float(self.bottom_percent_var.get())/100.0
                    composed = overlay_crop_on_image(self.mini_base_img, top_pct, bottom_pct)
                    self._redraw_mini_canvas_with_caption_indicator(composed, top_pct, bottom_pct)
            except Exception as e:
                try:
                    self.log_widget.config(state='normal')
                    self.log_widget.insert('end', f"[CAPTION-UPDATE-ERR] {e}\n")
                    self.log_widget.config(state='disabled')
                except Exception:
                    pass

            # Refresh TikTok preview in background thread to avoid blocking the UI.
            try:
                self._refresh_tiktok_preview_async()
            except Exception:
                pass
        except Exception as e:
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[CAPTION-POS-ERR] {e}\n")
                self.log_widget.config(state='disabled')
            except Exception:
                pass
    
    def _on_caption_y_spinbox_changed(self):
        """Callback when caption Y offset spinbox value changes (typed or incremented)."""
        try:
            offset = self.caption_y_offset_var.get()
            # Clamp to valid range (adapts to current resolution mode)
            min_val = -3840 if globals().get('IS_4K_MODE', False) else -1920
            offset = max(min_val, min(200, offset))
            self.caption_y_offset_var.set(offset)
            globals()['CAPTION_Y_OFFSET'] = offset

            # Cancel any pending debounced preview update (shared with slider)
            pending = getattr(self, '_caption_pos_debounce_id', None)
            if pending is not None:
                self.root.after_cancel(pending)
                self._caption_pos_debounce_id = None

            # Schedule the heavy preview update after a short delay (debounce)
            self._caption_pos_debounce_id = self.root.after(80, self._do_caption_position_update)
        except Exception:
            pass

    def on_caption_font_size_changed(self, val):
        """Callback when caption font size slider changes (debounced)."""
        try:
            # val comes as string; set global and update label immediately (lightweight)
            try:
                size = int(float(val))
            except Exception:
                try:
                    size = int(val)
                except Exception:
                    size = 56
            # Clamp to valid range (scaled for 4K)
            max_size = 800 if globals().get('IS_4K_MODE', False) else 120
            min_size = 40 if globals().get('IS_4K_MODE', False) else 20
            size = max(min_size, min(max_size, size))
            globals()['CAPTION_FONT_SIZE'] = size
            try:
                if hasattr(self, 'caption_font_size_label') and self.caption_font_size_label:
                    self.caption_font_size_label.config(text=f"{size}px")
            except Exception:
                pass

            # Cancel any pending debounced preview update
            pending = getattr(self, '_font_size_debounce_id', None)
            if pending is not None:
                self.root.after_cancel(pending)
                self._font_size_debounce_id = None

            # Schedule the heavy preview update after a short delay (debounce)
            self._font_size_debounce_id = self.root.after(80, self._do_font_size_preview_update)
        except Exception as e:
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[FONT-SIZE-ERR] {e}\n")
                self.log_widget.config(state='disabled')
            except Exception:
                pass

    def _do_font_size_preview_update(self):
        """Deferred font size preview update (called after debounce delay)."""
        self._font_size_debounce_id = None
        try:
            size = globals().get('CAPTION_FONT_SIZE', 56)
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[FONT-SIZE-CHANGE] Font size changed to: {size}px\n")
                self.log_widget.config(state='disabled')
                self.log_widget.see('end')
            except Exception:
                pass

            # Update mini preview to show new font size
            try:
                if hasattr(self, 'mini_base_img') and self.mini_base_img is not None:
                    top_pct = float(self.top_percent_var.get())/100.0
                    bottom_pct = float(self.bottom_percent_var.get())/100.0
                    composed = overlay_crop_on_image(self.mini_base_img, top_pct, bottom_pct)
                    self._redraw_mini_canvas_with_caption_indicator(composed, top_pct, bottom_pct)
            except Exception as e:
                try:
                    self.log_widget.config(state='normal')
                    self.log_widget.insert('end', f"[FONT-SIZE-UPDATE-ERR] {e}\n")
                    self.log_widget.config(state='disabled')
                except Exception:
                    pass

            # Refresh TikTok preview in background thread to avoid blocking the UI.
            # Opening MoviePy VideoFileClip is heavy; running it on the main thread
            # would freeze the slider after 1-2 changes.
            try:
                self._refresh_tiktok_preview_async()
            except Exception:
                pass
        except Exception:
            pass

    def on_words_per_caption_changed(self, *args):
        """Callback when words per caption spinbox changes."""
        try:
            # Get the new value from spinbox
            try:
                words = self.words_per_caption_var.get()
            except Exception:
                words = 2
            # Clamp to valid range
            words = max(1, min(3, words))
            
            # Update the global WORDS_PER_GROUP
            globals()['WORDS_PER_GROUP'] = words
            
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[WORDS-PER-CAPTION] Changed to: {words} words\n")
                self.log_widget.config(state='disabled')
                self.log_widget.see('end')
            except Exception:
                pass
            
            # Update mini preview to show new sample text - FORCE REDRAW
            try:
                if hasattr(self, 'mini_base_img') and self.mini_base_img is not None:
                    # Redraw the mini preview with updated caption
                    top_pct = float(self.top_percent_var.get())/100.0
                    bottom_pct = float(self.bottom_percent_var.get())/100.0
                    composed = overlay_crop_on_image(self.mini_base_img, top_pct, bottom_pct)
                    # Use centralized redraw method that includes caption indicator
                    self._redraw_mini_canvas_with_caption_indicator(composed, top_pct, bottom_pct)
                    # Force canvas update
                    self.mini_canvas.update_idletasks()
                    
            except Exception as e:
                try:
                    self.log_widget.config(state='normal')
                    self.log_widget.insert('end', f"[WORDS-UPDATE-ERR] {e}\n")
                    self.log_widget.config(state='disabled')
                except Exception:
                    pass
        except Exception as e:
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[WORDS-ERR] {e}\n")
                self.log_widget.config(state='disabled')
            except Exception:
                pass

    def _set_caption_case(self, mode, force=False):
        """Set caption case mode: 'upper', 'lower', or 'none'. Mutually exclusive.
        If force=True, skip the toggle-off logic (used when restoring from preset)."""
        if not force and mode == self.caption_case_var.get():
            mode = "none"  # clicking the active button turns it off
        self.caption_case_var.set(mode)
        globals()['CAPTION_CASE'] = mode
        # Visual feedback – highlight active button
        for btn, m in [(self._btn_upper, "upper"), (self._btn_lower, "lower"), (self._btn_case_off, "none")]:
            try:
                btn.state(['pressed'] if m == mode else ['!pressed'])
            except Exception:
                pass
        try:
            self.log_to_console(f"[CAPTION CASE] Set to: {mode}")
        except Exception:
            pass

    def on_template_selected(self, event=None):
        try:
            sel = (self.template_var.get() if hasattr(self, 'template_var') else '2 words').strip()
            num = 2
            try:
                if sel and sel[0].isdigit():
                    num = int(sel[0])
            except Exception:
                num = 2
            globals()['CAPTION_TEMPLATE'] = num
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[TEMPLATE] Selected: {sel} -> CAPTION_TEMPLATE={globals().get('CAPTION_TEMPLATE')}\n")
                self.log_widget.see('end')
                self.log_widget.config(state='disabled')
            except Exception:
                pass
        except Exception as e:
            print('Template select error:', e)


    def on_preview_template(self):
        # Generate a sample caption image using the selected template and show only ONE group (1/2/3 words)
        try:
            sample_text = "THIS IS A SAMPLE PREVIEW\nHow will it look?"
            sel = self.template_var.get() if hasattr(self, 'template_var') else '2 words'
            try:
                val = int(sel[0]) if sel and sel[0].isdigit() else 2
            except Exception:
                val = 2
            if val <= 0:
                val = 2
            # split into words and take the first 'val' words as a single caption
            words = sample_text.split()
            first_group = ' '.join(words[0:val]) if words else sample_text
            preview_text = first_group

                        # determine preferred font: prefer selected_font_path (ttf path) else selected_font family name
            pf = None
            try:
                selected_font_name = getattr(self, 'selected_font', None)
                selected_font_path = getattr(self, 'selected_font_path', None)
                pf = get_validated_font(selected_font_name, selected_font_path)
            except Exception:
                pf = None
            img_obj = generate_caption_image(preview_text, preferred_font=pf, log=lambda s: None)
            from PIL import Image
            # Accept either PIL.Image or numpy.ndarray (compatibility)
            if isinstance(img_obj, Image.Image):
                im = img_obj.convert('RGBA')
            else:
                im = Image.fromarray(img_obj).convert('RGBA')
            # resize to fit canvas width if necessary
            max_w = int(self.caption_preview_canvas['width']); max_h = int(self.caption_preview_canvas['height'])
            w, h = im.size
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            im_tk = ImageTk.PhotoImage(im)
            # store ref to avoid GC
            self._caption_preview_im = im_tk
            self.caption_preview_canvas.delete('all')
            
            # Get caption Y offset for position visualization
            y_offset = globals().get('CAPTION_Y_OFFSET', 0)
            cw = int(self.caption_preview_canvas['width']); ch = int(self.caption_preview_canvas['height'])
            
            # Simulate caption position in preview
            # In actual video (FFmpeg drawtext): y = h - th + y_offset
            # y_offset < 0 moves text UP from bottom, y_offset = 0 means text at bottom
            # In preview: use proportional FFmpeg text height to position the PIL image
            # so the visible text inside it matches the actual export position.
            #
            # The PIL image has top padding (4px gap + 24px padding_y = 28px) before the
            # text starts. We position the image so its text aligns with where FFmpeg
            # would place it, using the height ratio for Y mapping.
            preview_ratio = ch / HEIGHT if HEIGHT > 0 else 1.0
            font_size_cur = globals().get('CAPTION_FONT_SIZE', 56)
            image_scale = cw / WIDTH if WIDTH > 0 else 1.0
            # FFmpeg text height scaled to preview (th ≈ font_size * 0.85)
            ffmpeg_th_preview = font_size_cur * 0.85 * preview_ratio
            # PIL image top padding (4px gap + 24px padding_y) scaled to preview image
            top_pad_preview = int(28 * image_scale)
            # Position image so text inside it aligns with FFmpeg text_y
            simulated_y_pos = int(ch - ffmpeg_th_preview + y_offset * preview_ratio - top_pad_preview)
            
            # Clamp: keep at least a sliver visible, allow extending below canvas
            simulated_y_pos = max(0, simulated_y_pos)
            
            # Center image horizontally, position vertically based on offset
            x = (cw - im.width) // 2
            y = simulated_y_pos
            
            # Draw position indicator line at bottom to show baseline
            baseline_y = ch + int(y_offset * preview_ratio)
            self.caption_preview_canvas.create_line(0, baseline_y, cw, baseline_y, fill='#00FF00', dash=(4, 4), width=1)
            self.caption_preview_canvas.create_text(5, baseline_y - 10, text=f'Y offset: {y_offset}px', anchor='w', fill='#00FF00', font=('Arial', 8))
            
            self.caption_preview_canvas.create_image(x, y, anchor='nw', image=im_tk)
        except Exception as e:
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[PREVIEW-ERR] {e}\n")
                self.log_widget.see('end')
                self.log_widget.config(state='disabled')
            except Exception:
                pass

    def on_save_font_clicked(self):
        try:
            font = self.selected_font or (self.font_listbox.get(self.font_listbox.curselection()) if self.font_listbox.curselection() else None)
        except Exception:
            font = None
        video = self.video_var.get().strip() or self.output_var.get().strip()
        if not video:
            messagebox.showwarning("No video", "Select a video or set output filename before saving a font.")
            return
        if font:
            try:
                self.saved_fonts[video] = font
                self.saved_font_label.config(text=f"Saved: {font}")
                messagebox.showinfo("Saved", f"Saved font '{font}' for this video.")
            except Exception:
                pass
        else:
            messagebox.showwarning("No font", "Select a font first.")

    def on_load_font_clicked(self):
        video = self.video_var.get().strip() or self.output_var.get().strip()
        if not video:
            messagebox.showwarning("No video", "Select a video or set output filename before loading a font.")
            return
        font = self.saved_fonts.get(video)
        if not font:
            messagebox.showinfo("No saved font", "No font saved for this video.")
            return
        # select in listbox if available
        try:
            idxs = self.font_listbox.get(0, 'end')
            if font in idxs:
                i = idxs.index(font)
                self.font_listbox.selection_clear(0, 'end')
                self.font_listbox.selection_set(i)
                self.font_listbox.see(i)
                self.on_font_selected()
                messagebox.showinfo("Loaded", f"Loaded font '{font}' for this video.")
                self.saved_font_label.config(text=f"Saved: {font}")
        except Exception:
            pass
# ---------- UI callbacks & helpers ----------
    def toggle_fullscreen(self):
        try:
            if sys.platform == "win32":
                if self.root.state() == "normal":
                    self.root.state('zoomed')
                else:
                    self.root.state('normal')
            else:
                cur = bool(self.root.attributes("-fullscreen"))
                self.root.attributes("-fullscreen", not cur)
        except Exception:
            pass

    def browse_video(self):
        path = filedialog.askopenfilename(title="Select video", filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")])
        if path:
            self.video_var.set(path)
            try:
                # close previous preview clip if present
                try:
                    if hasattr(self, "_preview_clip") and self._preview_clip:
                        try:
                            self.safe_close_clip(self._preview_clip)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    self._preview_clip = VideoFileClip(path)
                    dur = self._preview_clip.duration
                except Exception:
                    # fallback if caching fails
                    clip = VideoFileClip(path)
                    dur = clip.duration
                    clip.close()
                try:
                    self.time_scale.to = max(0.1, dur)
                    self.time_scale.set(min(self.time_var.get(), self.time_scale.to))
                except Exception:
                    try:
                        self.time_scale.config(to=max(0.1, dur))
                    except Exception:
                        pass
                try:
                    self.time_var.set(min(self.time_var.get(), dur))
                    self.time_label.config(text=seconds_to_hms(self.time_var.get()))
                except Exception:
                    pass
            except Exception:
                try:
                    self.time_scale.to = 1.0
                    self.time_scale.set(min(self.time_var.get(), 1.0))
                except Exception:
                    try:
                        self.time_scale.config(to=1.0)
                    except Exception:
                        pass
            self._mini_update_worker_async()

    def _rgba_from_hex(self, hx):
        try:
            if hx.startswith('#'): hx = hx[1:]
            if len(hx) == 6:
                r = int(hx[0:2],16); g = int(hx[2:4],16); b = int(hx[4:6],16)
                return (r,g,b,255)
            if len(hx) == 8:
                r = int(hx[0:2],16); g = int(hx[2:4],16); b = int(hx[4:6],16); a = int(hx[6:8],16)
                return (r,g,b,a)
        except Exception:
            pass
        return (255,255,255,255)

    def _update_color_canvases(self):
        try:
            try:
                txt = CAPTION_TEXT_COLOR
                col = '#%02x%02x%02x' % (txt[0], txt[1], txt[2])
                if hasattr(self, 'text_color_canvas') and self.text_color_canvas:
                    self.text_color_canvas.delete('all')
                    self.text_color_canvas.create_oval(2,2,26,26, fill=col, outline='white')
            except Exception:
                pass
            try:
                st = CAPTION_STROKE_COLOR
                col2 = '#%02x%02x%02x' % (st[0], st[1], st[2])
                if hasattr(self, 'stroke_color_canvas') and self.stroke_color_canvas:
                    self.stroke_color_canvas.delete('all')
                    self.stroke_color_canvas.create_oval(2,2,26,26, fill=col2, outline='white')
            except Exception:
                pass
        except Exception:
            pass

    def on_pick_text_color(self):
        try:
            from tkinter import colorchooser
            col = colorchooser.askcolor(title='Choose caption text color')
            if col and col[1]:
                rgba = self._rgba_from_hex(col[1])
                globals()['CAPTION_TEXT_COLOR'] = rgba
                self._update_color_canvases()
                # Trigger live preview update
                self._mini_update_worker_async()
        except Exception as e:
            try:
                self.log_widget.config(state='normal'); self.log_widget.insert('end', f"[COLOR-ERR] {e}\n"); self.log_widget.config(state='disabled')
            except Exception:
                pass

    def on_pick_stroke_color(self):
        try:
            from tkinter import colorchooser
            col = colorchooser.askcolor(title='Choose caption stroke color')
            if col and col[1]:
                rgba = self._rgba_from_hex(col[1])
                if len(rgba) == 4 and rgba[3] == 255:
                    rgba = (rgba[0], rgba[1], rgba[2], 150)
                globals()['CAPTION_STROKE_COLOR'] = rgba
                self._update_color_canvases()
                # Trigger live preview update
                self._mini_update_worker_async()
        except Exception as e:
            try:
                self.log_widget.config(state='normal'); self.log_widget.insert('end', f"[COLOR-ERR] {e}\n"); self.log_widget.config(state='disabled')
            except Exception:
                pass

    def _set_text_color_hex(self, hx):
        try:
            rgba = self._rgba_from_hex(hx)
            globals()['CAPTION_TEXT_COLOR'] = rgba
            self._update_color_canvases()
            # Trigger live preview update
            self._mini_update_worker_async()
        except Exception:
            pass

    def _set_stroke_color_hex(self, hx):
        try:
            rgba = self._rgba_from_hex(hx)
            rgba = (rgba[0], rgba[1], rgba[2], 150)
            globals()['CAPTION_STROKE_COLOR'] = rgba
            self._update_color_canvases()
            # Trigger live preview update
            self._mini_update_worker_async()
        except Exception:
            pass


    def safe_close_clip(self, clip):
        """Închide în siguranță un VideoFileClip/reader fără a arunca excepții."""
        if not clip:
            return
        try:
            # If there's a reader with a proc, try to terminate it safely
            reader = getattr(clip, "reader", None)
            if reader is not None:
                proc = getattr(reader, "proc", None)
                if proc is not None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                try:
                    reader.close()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            clip.close()
        except Exception:
            pass

    def on_close(self):
        try:
            self._stop_mini_persistent_worker()
        except Exception:
            pass

        """Cleanup la închiderea aplicației: închide clipurile din cache și distruge fereastra."""
        try:
            if hasattr(self, "_preview_clip") and self._preview_clip:
                try:
                    self.safe_close_clip(self._preview_clip)
                except Exception:
                    pass
                self._preview_clip = None
        except Exception:
            pass

        # (optional) close any other cached clips here if you added them

        try:
            self.root.destroy()
        except Exception:
            try:
                import sys
                sys.exit(0)
            except Exception:
                pass


    def browse_voice(self):
        path = filedialog.askopenfilename(title="Select voice audio", filetypes=[("Audio files", "*.wav *.mp3 *.m4a"), ("All files", "*.*")])
        if path: self.voice_var.set(path)

    def browse_music(self):
        path = filedialog.askopenfilename(title="Select music", filetypes=[("Audio files", "*.mp3 *.wav *.m4a *.ogg"), ("All files", "*.*")])
        if path: self.music_var.set(path)

    def choose_output(self):
        path = filedialog.asksaveasfilename(title="Output file", defaultextension=".mp4", filetypes=[("MP4 file", "*.mp4")])
        if path: self.output_var.set(path)

    def log_to_console(self, msg):
        """Append a message to the log widget."""
        try:
            self.log_widget.config(state='normal')
            self.log_widget.insert('end', str(msg) + "\n")
            self.log_widget.see('end')
            self.log_widget.config(state='disabled')
        except Exception:
            pass

    def _format_job_info(self, job):
        """Format job information for display, showing all relevant settings."""
        info_parts = []
        
        # Resolution and basic settings
        if job.get("use_4k"):
            info_parts.append("4K")
        if job.get("mirror_video"):
            info_parts.append("Mirror")
        if job.get("words_per_caption", 2) != 2:
            info_parts.append(f"{job.get('words_per_caption')} words/cap")
        
        # AI and translation settings
        if job.get("use_ai_voice"):
            info_parts.append("AI-TTS")
        if job.get("translation_enabled"):
            lang = job.get("target_language", "?")
            info_parts.append(f"Translate:{lang}")
        
        # Silence threshold (only show if AI voice is enabled and not default)
        if job.get("use_ai_voice") and job.get("silence_threshold_ms", 300) != 300:
            info_parts.append(f"silence:{job.get('silence_threshold_ms')}ms")
        
        # Font color (show if not default white)
        text_color = job.get("caption_text_color", (255, 255, 255, 255))
        if text_color != (255, 255, 255, 255):
            info_parts.append(f"color:RGB({text_color[0]},{text_color[1]},{text_color[2]})")
        
        # Font size (show if not default 56)
        font_size = job.get("caption_font_size", 56)
        if font_size != 56:
            info_parts.append(f"size:{font_size}px")
        
        # Stroke/border settings (show if not default)
        stroke_color = job.get("caption_stroke_color", (0, 0, 0, 150))
        stroke_width = job.get("caption_stroke_width", 3)
        if stroke_color != (0, 0, 0, 150) or stroke_width != 3:
            info_parts.append(f"border:w={stroke_width},RGB({stroke_color[0]},{stroke_color[1]},{stroke_color[2]})")
        
        # Add effects info if different from defaults
        blur = job.get("blur_radius", 25)
        bg_scale = job.get("bg_scale_extra", 1.08)
        dim = job.get("dim_factor", 1.0)
        if blur != 25 or bg_scale != 1.08 or dim != 1.0:
            info_parts.append(f"effects:blur={blur}/scale={bg_scale:.2f}/dim={dim:.2f}")
        
        # Video effects (CapCut-style)
        effects_active = []
        if job.get("effect_sharpness"):
            effects_active.append("Resilience")
        if job.get("effect_saturation"):
            effects_active.append("Vibrance")
        if job.get("effect_contrast"):
            effects_active.append("HDR")
        if job.get("effect_brightness"):
            effects_active.append("Brightness")
        if job.get("effect_vintage"):
            effects_active.append("Vintage")
        if effects_active:
            info_parts.append(f"fx:{','.join(effects_active)}")
        
        # Blur overlay
        if job.get("blur_overlay_enabled"):
            bx = job.get("blur_overlay_x", 10)
            by = job.get("blur_overlay_y", 10)
            bw = job.get("blur_overlay_w", 20)
            bh = job.get("blur_overlay_h", 15)
            bi = job.get("blur_overlay_intensity", 20)
            info_parts.append(f"blur_box:({bx:.0f}%,{by:.0f}%) {bw:.0f}x{bh:.0f}% intensity={bi}")
        
        return " [" + ", ".join(info_parts) + "]" if info_parts else ""

    def add_job(self):
        try:
            video = self.video_var.get().strip()
            voice = self.voice_var.get().strip()
            music = self.music_var.get().strip()
            output = self.output_var.get().strip() or "final_tiktok.mp4"
            if not video or not music:
                messagebox.showwarning("Missing fields", "Please select VIDEO and MUSIC before adding a job.")
                return
            # preferred font: prefer selected_font_path (.ttf) else selected_font family name
            try:
                selected_font_name = getattr(self, 'selected_font', None)
                selected_font_path = getattr(self, 'selected_font_path', None)
                pref_font = get_validated_font(selected_font_name, selected_font_path)
            except Exception:
                pref_font = None
            
            # Log crop and font settings for debugging
            use_custom = self.use_custom_crop_var.get()
            top_val = self.top_percent_var.get()
            bottom_val = self.bottom_percent_var.get()
            self.log_to_console(f"\n[DEBUG JOB] Creating job with:")
            self.log_to_console(f"[DEBUG JOB] Selected font name: {selected_font_name}")
            self.log_to_console(f"[DEBUG JOB] Selected font path: {selected_font_path}")
            self.log_to_console(f"[DEBUG JOB] Validated font (will use): {pref_font}")
            self.log_to_console(f"[DEBUG JOB] Use custom crop checkbox: {use_custom}")
            self.log_to_console(f"[DEBUG JOB] Top percent: {top_val}")
            self.log_to_console(f"[DEBUG JOB] Bottom percent: {bottom_val}")
            self.log_to_console(f"[DEBUG JOB] Will send custom_top_ratio: {(top_val/100.0) if use_custom else None}")
            self.log_to_console(f"[DEBUG JOB] Will send custom_bottom_ratio: {(bottom_val/100.0) if use_custom else None}")
            
            job = {
                "video": video,
                "voice": voice,
                "music": music,
                "output": output,
                "font": pref_font,
                "custom_top_ratio": (self.top_percent_var.get()/100.0) if self.use_custom_crop_var.get() else None,
                "custom_bottom_ratio": (self.bottom_percent_var.get()/100.0) if self.use_custom_crop_var.get() else None,
                "mirror_video": self.mirror_video_var.get(),
                "words_per_caption": self.words_per_caption_var.get(),
                "use_4k": self.use_4k_var.get(),
                "blur_radius": globals().get('STATIC_BG_BLUR_RADIUS', 25),
                "bg_scale_extra": globals().get('BG_SCALE_EXTRA', 1.08),
                "dim_factor": globals().get('DIM_FACTOR', 1.0),
                # AI and caption settings
                "use_ai_voice": self.use_ai_voice_var.get(),
                "translation_enabled": self.translation_enabled_var.get(),
                "target_language": self.target_language_var.get() if hasattr(self, 'target_language_var') else 'none',
                "tts_language": self.tts_language_var.get() if hasattr(self, 'tts_language_var') else 'en',
                "silence_threshold_ms": self.silence_threshold_var.get(),
                # Font and border settings (per-job caption styling)
                "caption_text_color": globals().get('CAPTION_TEXT_COLOR', (255, 255, 255, 255)),
                "caption_stroke_color": globals().get('CAPTION_STROKE_COLOR', (0, 0, 0, 150)),
                "caption_stroke_width": globals().get('CAPTION_STROKE_WIDTH', 3),
                "caption_font_size": globals().get('CAPTION_FONT_SIZE', 56),
                "caption_y_offset": globals().get('CAPTION_Y_OFFSET', 0),
                # Video effects (CapCut-style)
                "effect_sharpness": self.effect_sharpness_var.get(),
                "effect_sharpness_intensity": self.effect_sharpness_intensity_var.get(),
                "effect_saturation": self.effect_saturation_var.get(),
                "effect_saturation_intensity": self.effect_saturation_intensity_var.get(),
                "effect_contrast": self.effect_contrast_var.get(),
                "effect_contrast_intensity": self.effect_contrast_intensity_var.get(),
                "effect_brightness": self.effect_brightness_var.get(),
                "effect_brightness_intensity": self.effect_brightness_intensity_var.get(),
                "effect_vintage": self.effect_vintage_var.get(),
                "effect_vintage_intensity": self.effect_vintage_intensity_var.get(),
                # Blur overlay (cover-up region)
                "blur_overlay_enabled": self.blur_overlay_enabled_var.get(),
                "blur_overlay_x": self.blur_overlay_x_var.get(),
                "blur_overlay_y": self.blur_overlay_y_var.get(),
                "blur_overlay_w": self.blur_overlay_w_var.get(),
                "blur_overlay_h": self.blur_overlay_h_var.get(),
                "blur_overlay_intensity": self.blur_overlay_intensity_var.get(),
                # Audio volume settings (captured at job creation time)
                "voice_gain": self.voice_gain_var.get(),
                "music_gain": self.music_gain_var.get()
            }
            self.jobs.append(job)
            # Show complete job info in the display using helper
            info = self._format_job_info(job)
            display = f"{Path(video).name} | {Path(voice).name} | {Path(music).name} -> {Path(output).name} (font={pref_font or 'default'}){info}"
            self.job_listbox.insert('end', display)
        except Exception as e:
            messagebox.showerror("Error adding job", str(e))

    def add_multiple_jobs(self):
        """Open a multi-file dialog, then add one job per selected video using the current preset."""
        try:
            music = self.music_var.get().strip()
            if not music:
                messagebox.showwarning("Missing Music", "Please select a MUSIC file before adding multiple jobs.")
                return
            video_paths = filedialog.askopenfilenames(
                title="Select multiple video files",
                filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.flv"), ("All files", "*.*")]
            )
            if not video_paths:
                return

            # Snapshot all current preset settings once
            voice = self.voice_var.get().strip()
            try:
                selected_font_name = getattr(self, 'selected_font', None)
                selected_font_path = getattr(self, 'selected_font_path', None)
                pref_font = get_validated_font(selected_font_name, selected_font_path)
            except Exception:
                pref_font = None

            use_custom = self.use_custom_crop_var.get()

            preset = {
                "voice": voice,
                "music": music,
                "font": pref_font,
                "custom_top_ratio": (self.top_percent_var.get() / 100.0) if use_custom else None,
                "custom_bottom_ratio": (self.bottom_percent_var.get() / 100.0) if use_custom else None,
                "mirror_video": self.mirror_video_var.get(),
                "words_per_caption": self.words_per_caption_var.get(),
                "use_4k": self.use_4k_var.get(),
                "blur_radius": globals().get('STATIC_BG_BLUR_RADIUS', 25),
                "bg_scale_extra": globals().get('BG_SCALE_EXTRA', 1.08),
                "dim_factor": globals().get('DIM_FACTOR', 1.0),
                "use_ai_voice": self.use_ai_voice_var.get(),
                "translation_enabled": self.translation_enabled_var.get(),
                "target_language": self.target_language_var.get() if hasattr(self, 'target_language_var') else 'none',
                "tts_language": self.tts_language_var.get() if hasattr(self, 'tts_language_var') else 'en',
                "silence_threshold_ms": self.silence_threshold_var.get(),
                "caption_text_color": globals().get('CAPTION_TEXT_COLOR', (255, 255, 255, 255)),
                "caption_stroke_color": globals().get('CAPTION_STROKE_COLOR', (0, 0, 0, 150)),
                "caption_stroke_width": globals().get('CAPTION_STROKE_WIDTH', 3),
                "caption_font_size": globals().get('CAPTION_FONT_SIZE', 56),
                "caption_y_offset": globals().get('CAPTION_Y_OFFSET', 0),
                "effect_sharpness": self.effect_sharpness_var.get(),
                "effect_sharpness_intensity": self.effect_sharpness_intensity_var.get(),
                "effect_saturation": self.effect_saturation_var.get(),
                "effect_saturation_intensity": self.effect_saturation_intensity_var.get(),
                "effect_contrast": self.effect_contrast_var.get(),
                "effect_contrast_intensity": self.effect_contrast_intensity_var.get(),
                "effect_brightness": self.effect_brightness_var.get(),
                "effect_brightness_intensity": self.effect_brightness_intensity_var.get(),
                "effect_vintage": self.effect_vintage_var.get(),
                "effect_vintage_intensity": self.effect_vintage_intensity_var.get(),
                "blur_overlay_enabled": self.blur_overlay_enabled_var.get(),
                "blur_overlay_x": self.blur_overlay_x_var.get(),
                "blur_overlay_y": self.blur_overlay_y_var.get(),
                "blur_overlay_w": self.blur_overlay_w_var.get(),
                "blur_overlay_h": self.blur_overlay_h_var.get(),
                "blur_overlay_intensity": self.blur_overlay_intensity_var.get(),
                "voice_gain": self.voice_gain_var.get(),
                "music_gain": self.music_gain_var.get(),
            }

            added = 0
            for video_path in video_paths:
                video_stem = Path(video_path).stem
                output = f"{video_stem}_tiktok.mp4"
                job = dict(preset)
                job["video"] = video_path
                job["output"] = output
                self.jobs.append(job)
                info = self._format_job_info(job)
                voice_name = Path(voice).name if voice else "(none)"
                display = (f"{Path(video_path).name} | {voice_name} | {Path(music).name}"
                           f" -> {output} (font={pref_font or 'default'}){info}")
                self.job_listbox.insert('end', display)
                added += 1

            self.log_to_console(f"[Jobs] ✓ Added {added} job(s) with current preset.")
        except Exception as e:
            messagebox.showerror("Error adding jobs", str(e))

    def _refresh_job_listbox(self):
        self.job_listbox.delete(0, tk.END)
        for i, j in enumerate(self.jobs, start=1):
            tag = ""
            if j.get("custom_top_ratio") is not None:
                tag = f" (crop {int(j['custom_top_ratio']*100)}%/{int(j['custom_bottom_ratio']*100)}%)"
            
            # Use helper to format job info consistently
            info = self._format_job_info(j)
            
            display = f"{i}. {os.path.basename(j['video'])} | {os.path.basename(j['voice'])} | {os.path.basename(j['music'])}{tag}{info}"
            self.job_listbox.insert(tk.END, display)

    def remove_job(self):
        sel = self.job_listbox.curselection()
        if not sel:
            return
        selected_indices = sorted(sel, reverse=True)
        for idx in selected_indices:
            try:
                self.jobs.pop(idx)
            except Exception:
                pass
        self._refresh_job_listbox()

    def on_job_double_click(self, event):
        """Load job settings when double-clicking a job in the list."""
        try:
            # Get selected job index
            sel = self.job_listbox.curselection()
            if not sel:
                return
            
            job_idx = sel[0]
            if job_idx >= len(self.jobs):
                return
            
            job = self.jobs[job_idx]
            
            # Load file paths
            self.video_var.set(job.get("video", ""))
            self.voice_var.set(job.get("voice", ""))
            self.music_var.set(job.get("music", ""))
            self.output_var.set(job.get("output", ""))
            
            # Load basic settings
            self.mirror_video_var.set(job.get("mirror_video", False))
            self.words_per_caption_var.set(job.get("words_per_caption", 2))
            self.use_4k_var.set(job.get("use_4k", False))
            
            # Load crop settings
            if job.get("custom_top_ratio") is not None:
                self.use_custom_crop_var.set(True)
                self.top_percent_var.set(job.get("custom_top_ratio", 0.0) * 100.0)
                self.bottom_percent_var.set(job.get("custom_bottom_ratio", 0.0) * 100.0)
            else:
                self.use_custom_crop_var.set(False)
            
            # Load effects settings
            globals()['STATIC_BG_BLUR_RADIUS'] = job.get("blur_radius", 25)
            globals()['BG_SCALE_EXTRA'] = job.get("bg_scale_extra", 1.08)
            # DIM_FACTOR: no longer loaded from saved jobs (always use global default 1.0)
            
            # Load AI and translation settings
            self.use_ai_voice_var.set(job.get("use_ai_voice", False))
            self.translation_enabled_var.set(job.get("translation_enabled", False))
            if hasattr(self, 'target_language_var'):
                self.target_language_var.set(job.get("target_language", "none"))
            if hasattr(self, 'tts_language_var'):
                self.tts_language_var.set(job.get("tts_language", "en"))
            self.silence_threshold_var.set(job.get("silence_threshold_ms", 300))
            
            # Load video effects (CapCut-style)
            if hasattr(self, 'effect_sharpness_var'):
                self.effect_sharpness_var.set(job.get("effect_sharpness", False))
                self.effect_sharpness_intensity_var.set(job.get("effect_sharpness_intensity", 1.5))
            if hasattr(self, 'effect_saturation_var'):
                self.effect_saturation_var.set(job.get("effect_saturation", False))
                self.effect_saturation_intensity_var.set(job.get("effect_saturation_intensity", 1.3))
            if hasattr(self, 'effect_contrast_var'):
                self.effect_contrast_var.set(job.get("effect_contrast", False))
                self.effect_contrast_intensity_var.set(job.get("effect_contrast_intensity", 1.2))
            if hasattr(self, 'effect_brightness_var'):
                self.effect_brightness_var.set(job.get("effect_brightness", False))
                self.effect_brightness_intensity_var.set(job.get("effect_brightness_intensity", 1.15))
            if hasattr(self, 'effect_vintage_var'):
                self.effect_vintage_var.set(job.get("effect_vintage", False))
                self.effect_vintage_intensity_var.set(job.get("effect_vintage_intensity", 0.3))
            
            # Load blur overlay settings
            if hasattr(self, 'blur_overlay_enabled_var'):
                self.blur_overlay_enabled_var.set(job.get("blur_overlay_enabled", False))
                self.blur_overlay_x_var.set(job.get("blur_overlay_x", 10.0))
                self.blur_overlay_y_var.set(job.get("blur_overlay_y", 10.0))
                self.blur_overlay_w_var.set(job.get("blur_overlay_w", 20.0))
                self.blur_overlay_h_var.set(job.get("blur_overlay_h", 15.0))
                self.blur_overlay_intensity_var.set(job.get("blur_overlay_intensity", 20))
            
            # Load font and border settings
            text_color = job.get("caption_text_color", (255, 255, 255, 255))
            stroke_color = job.get("caption_stroke_color", (0, 0, 0, 150))
            stroke_width = job.get("caption_stroke_width", 3)
            
            globals()['CAPTION_TEXT_COLOR'] = text_color
            globals()['CAPTION_STROKE_COLOR'] = stroke_color
            globals()['CAPTION_STROKE_WIDTH'] = stroke_width
            
            # Load caption Y offset
            caption_y_off = job.get("caption_y_offset", 0)
            globals()['CAPTION_Y_OFFSET'] = caption_y_off
            if hasattr(self, 'caption_y_offset_var'):
                self.caption_y_offset_var.set(caption_y_off)
            
            # Load caption font size
            font_sz = job.get("caption_font_size", 56)
            globals()['CAPTION_FONT_SIZE'] = font_sz
            if hasattr(self, 'caption_font_size_var'):
                self.caption_font_size_var.set(font_sz)
                if hasattr(self, 'caption_font_size_label'):
                    self.caption_font_size_label.config(text=f"{font_sz}px")
            
            # Update UI elements for colors if they exist
            try:
                if hasattr(self, 'text_color_canvas') and self.text_color_canvas:
                    col = f'#{text_color[0]:02x}{text_color[1]:02x}{text_color[2]:02x}'
                    self.text_color_canvas.delete('all')
                    self.text_color_canvas.create_oval(2,2,26,26, fill=col, outline='white')
            except Exception:
                pass
            
            try:
                if hasattr(self, 'stroke_color_canvas') and self.stroke_color_canvas:
                    col = f'#{stroke_color[0]:02x}{stroke_color[1]:02x}{stroke_color[2]:02x}'
                    self.stroke_color_canvas.delete('all')
                    self.stroke_color_canvas.create_oval(2,2,26,26, fill=col, outline='white')
            except Exception:
                pass
            
            try:
                if hasattr(self, 'stroke_width_var'):
                    self.stroke_width_var.set(stroke_width)
                if hasattr(self, 'stroke_width_label') and self.stroke_width_label:
                    self.stroke_width_label.config(text=str(int(stroke_width)))
            except Exception:
                pass
            
            # Load font if specified
            font_path = job.get("font")
            if font_path:
                try:
                    self.selected_font_path = font_path
                    self.selected_font = os.path.basename(font_path) if font_path else None
                except Exception:
                    pass
            
            # Load audio gain settings
            if hasattr(self, 'voice_gain_var'):
                self.voice_gain_var.set(job.get("voice_gain", 1.0))
            if hasattr(self, 'music_gain_var'):
                self.music_gain_var.set(job.get("music_gain", 0.5))

            # If this was a plain double-click (not via Edit Job button), reset edit mode
            if event is not None:
                self._editing_job_idx = None
                try:
                    self.update_job_btn.config(state="disabled")
                    self.edit_job_btn.config(text="✏️ Edit Job")
                except Exception:
                    pass

            # Update video preview if video file exists
            video_path = job.get("video", "")
            if video_path and os.path.exists(video_path):
                try:
                    # Trigger mini preview update
                    self._mini_update_worker_async()
                except Exception as e:
                    print(f"Could not update preview: {e}")
            
            # Log the load action
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"\n[LOAD] Loaded settings from job #{job_idx + 1}\n")
                self.log_widget.see('end')
                self.log_widget.config(state='disabled')
            except Exception:
                pass
                
        except Exception as e:
            messagebox.showerror("Error loading job", str(e))

    def on_edit_job(self):
        """Load selected job into UI and activate Update Job mode."""
        try:
            sel = self.job_listbox.curselection()
            if not sel:
                messagebox.showwarning("No selection", "Selectează un job din listă pentru a-l edita.")
                return
            job_idx = sel[0]
            if job_idx >= len(self.jobs):
                return
            self._editing_job_idx = job_idx
            # Re-use the existing double-click loader
            self.on_job_double_click(None)
            # Enable Update button, keep track visually
            self.update_job_btn.config(state="normal")
            self.edit_job_btn.config(text=f"✏️ Editing #{job_idx + 1}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_job(self):
        """Save current UI settings back into the job being edited."""
        try:
            if self._editing_job_idx is None:
                messagebox.showwarning("Nothing to update", "Apasă mai întâi '✏️ Edit Job' pe un job selectat.")
                return
            idx = self._editing_job_idx
            if idx >= len(self.jobs):
                messagebox.showerror("Error", "Job-ul nu mai există în listă.")
                self._editing_job_idx = None
                self.update_job_btn.config(state="disabled")
                self.edit_job_btn.config(text="✏️ Edit Job")
                return

            video = self.video_var.get().strip()
            voice = self.voice_var.get().strip()
            music = self.music_var.get().strip()
            output = self.output_var.get().strip() or "final_tiktok.mp4"
            if not video or not music:
                messagebox.showwarning("Missing fields", "VIDEO și MUSIC sunt obligatorii.")
                return

            try:
                selected_font_name = getattr(self, 'selected_font', None)
                selected_font_path = getattr(self, 'selected_font_path', None)
                pref_font = get_validated_font(selected_font_name, selected_font_path)
            except Exception:
                pref_font = self.jobs[idx].get("font")

            job = {
                "video": video,
                "voice": voice,
                "music": music,
                "output": output,
                "font": pref_font,
                "custom_top_ratio": (self.top_percent_var.get() / 100.0) if self.use_custom_crop_var.get() else None,
                "custom_bottom_ratio": (self.bottom_percent_var.get() / 100.0) if self.use_custom_crop_var.get() else None,
                "mirror_video": self.mirror_video_var.get(),
                "words_per_caption": self.words_per_caption_var.get(),
                "use_4k": self.use_4k_var.get(),
                "blur_radius": globals().get('STATIC_BG_BLUR_RADIUS', 25),
                "bg_scale_extra": globals().get('BG_SCALE_EXTRA', 1.08),
                "dim_factor": globals().get('DIM_FACTOR', 1.0),
                "use_ai_voice": self.use_ai_voice_var.get(),
                "translation_enabled": self.translation_enabled_var.get(),
                "target_language": self.target_language_var.get() if hasattr(self, 'target_language_var') else 'none',
                "tts_language": self.tts_language_var.get() if hasattr(self, 'tts_language_var') else 'en',
                "silence_threshold_ms": self.silence_threshold_var.get(),
                "caption_text_color": globals().get('CAPTION_TEXT_COLOR', (255, 255, 255, 255)),
                "caption_stroke_color": globals().get('CAPTION_STROKE_COLOR', (0, 0, 0, 150)),
                "caption_stroke_width": globals().get('CAPTION_STROKE_WIDTH', 3),
                "caption_font_size": globals().get('CAPTION_FONT_SIZE', 56),
                "caption_y_offset": globals().get('CAPTION_Y_OFFSET', 0),
                "effect_sharpness": self.effect_sharpness_var.get() if hasattr(self, 'effect_sharpness_var') else False,
                "effect_sharpness_intensity": self.effect_sharpness_intensity_var.get() if hasattr(self, 'effect_sharpness_intensity_var') else 1.5,
                "effect_saturation": self.effect_saturation_var.get() if hasattr(self, 'effect_saturation_var') else False,
                "effect_saturation_intensity": self.effect_saturation_intensity_var.get() if hasattr(self, 'effect_saturation_intensity_var') else 1.3,
                "effect_contrast": self.effect_contrast_var.get() if hasattr(self, 'effect_contrast_var') else False,
                "effect_contrast_intensity": self.effect_contrast_intensity_var.get() if hasattr(self, 'effect_contrast_intensity_var') else 1.2,
                "effect_brightness": self.effect_brightness_var.get() if hasattr(self, 'effect_brightness_var') else False,
                "effect_brightness_intensity": self.effect_brightness_intensity_var.get() if hasattr(self, 'effect_brightness_intensity_var') else 1.15,
                "effect_vintage": self.effect_vintage_var.get() if hasattr(self, 'effect_vintage_var') else False,
                "effect_vintage_intensity": self.effect_vintage_intensity_var.get() if hasattr(self, 'effect_vintage_intensity_var') else 0.3,
                "blur_overlay_enabled": self.blur_overlay_enabled_var.get() if hasattr(self, 'blur_overlay_enabled_var') else False,
                "blur_overlay_x": self.blur_overlay_x_var.get() if hasattr(self, 'blur_overlay_x_var') else 10.0,
                "blur_overlay_y": self.blur_overlay_y_var.get() if hasattr(self, 'blur_overlay_y_var') else 10.0,
                "blur_overlay_w": self.blur_overlay_w_var.get() if hasattr(self, 'blur_overlay_w_var') else 20.0,
                "blur_overlay_h": self.blur_overlay_h_var.get() if hasattr(self, 'blur_overlay_h_var') else 15.0,
                "blur_overlay_intensity": self.blur_overlay_intensity_var.get() if hasattr(self, 'blur_overlay_intensity_var') else 20,
                "voice_gain": self.voice_gain_var.get(),
                "music_gain": self.music_gain_var.get(),
            }
            self.jobs[idx] = job
            self._refresh_job_listbox()
            # Reset edit mode
            self._editing_job_idx = None
            self.update_job_btn.config(state="disabled")
            self.edit_job_btn.config(text="✏️ Edit Job")
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"\n[UPDATE] Job #{idx + 1} actualizat cu succes.\n")
                self.log_widget.see('end')
                self.log_widget.config(state='disabled')
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error updating job", str(e))

    # ──────────────────────────────────────────────────────────────
    # TikTok Downloader helpers
    # ──────────────────────────────────────────────────────────────

    def _tt_dl_detect_start(self):
        """Scan the output folder for existing numbered .mp4 files and set start to max+1."""
        folder = self.tt_dl_folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No folder", "Selectează mai întâi un folder de output.")
            return
        import re
        max_num = 0
        for fname in os.listdir(folder):
            m = re.match(r'^(\d+)\.(mp4|mov|avi|mkv)$', fname, re.IGNORECASE)
            if m:
                max_num = max(max_num, int(m.group(1)))
        self.tt_dl_start_var.set(max_num + 1)
        self.tt_dl_status_var.set(f"Am găsit {max_num} fișier(e). Numerotarea va începe de la {max_num + 1}.")

    def _tt_dl_cancel(self):
        self._tt_dl_stop = True
        self.tt_dl_status_var.set("Se oprește…")

    def _tt_dl_set_ui(self, running: bool):
        state_btn = "disabled" if running else "normal"
        state_cancel = "normal" if running else "disabled"
        try:
            self.tt_dl_btn.config(state=state_btn)
            self.tt_dl_cancel_btn.config(state=state_cancel)
        except Exception:
            pass

    def _tt_dl_start(self):
        url = self.tt_dl_url_var.get().strip()
        if not url:
            messagebox.showwarning("URL lipsă", "Introdu URL-ul sau @username-ul contului TikTok.")
            return
        folder = self.tt_dl_folder_var.get().strip()
        if not folder:
            messagebox.showwarning("Folder lipsă", "Selectează un folder de output.")
            return
        os.makedirs(folder, exist_ok=True)

        # Normalise: accept @user, user, or full URL
        if not url.startswith("http"):
            username = url.lstrip("@")
            url = f"https://www.tiktok.com/@{username}"

        max_videos = self.tt_dl_max_var.get()
        download_all = self.tt_dl_all_var.get()
        start_num = self.tt_dl_start_var.get()
        sort_order = self.tt_dl_order_var.get()
        from_video_url = self.tt_dl_from_url_var.get().strip()

        self._tt_dl_stop = False
        self._tt_dl_set_ui(True)
        self.tt_dl_status_var.set("Se pornește descărcarea…")

        def _run():
            try:
                # Check yt-dlp availability
                try:
                    import yt_dlp
                except ImportError:
                    self.root.after(0, lambda: self._tt_dl_finish(
                        "yt-dlp nu este instalat. Rulează: pip install yt-dlp și repornește aplicația."))
                    return

                # Collect video URLs first (playlist / user page)
                self.root.after(0, lambda: self.tt_dl_status_var.set("Se colectează lista de videouri…"))

                # For Popular or Oldest we need the full list to sort/reverse properly.
                # yt-dlp ignores sort_type URL params and always returns newest-first from
                # TikTok's API, so we handle ordering ourselves after collection.
                need_full_list = download_all or (sort_order in ("Populare", "Cele mai vechi")) or bool(from_video_url)
                ydl_opts_list = {
                    'quiet': True,
                    'no_warnings': True,
                    'ignoreerrors': True,
                }
                if download_all:
                    # Do NOT use extract_flat when downloading all: TikTok's extractor
                    # stops paginating after the first page (~20-30 videos) when flat
                    # extraction is active. Without extract_flat, yt-dlp follows all
                    # API cursor pages and collects the full playlist.
                    pass
                else:
                    ydl_opts_list['extract_flat'] = True
                    if not need_full_list:
                        ydl_opts_list['playlistend'] = max_videos

                with yt_dlp.YoutubeDL(ydl_opts_list) as ydl:
                    info = ydl.extract_info(url, download=False)

                if not info:
                    self.root.after(0, lambda: self._tt_dl_finish("Nu s-au putut colecta videouri. Verifică URL-ul."))
                    return

                raw_entries = info.get('entries') or [info]
                # When download_all without extract_flat, entries may be fully-resolved
                # info dicts or nested playlists; flatten one level and keep only video entries.
                entries = []
                for e in raw_entries:
                    if not e:
                        continue
                    if e.get('_type') == 'playlist' or (e.get('entries') and not e.get('url')):
                        # Nested playlist — flatten its children
                        for sub in (e.get('entries') or []):
                            if sub:
                                entries.append(sub)
                    else:
                        entries.append(e)

                if not entries:
                    self.root.after(0, lambda: self._tt_dl_finish("Nicio înregistrare găsită."))
                    return

                # Apply sort order client-side (yt-dlp always returns newest-first from TikTok API)
                if sort_order == "Cele mai vechi":
                    entries = list(reversed(entries))
                elif sort_order == "Populare":
                    # Sort descending by view_count; entries without a count go last
                    entries = sorted(entries, key=lambda e: e.get('view_count') or 0, reverse=True)

                # Find start position from specific video URL
                if from_video_url:
                    # Extract the video ID from the given URL
                    import re as _re
                    m = _re.search(r'/video/(\d+)', from_video_url)
                    target_id = m.group(1) if m else None
                    start_idx = 0
                    if target_id:
                        for idx, e in enumerate(entries):
                            entry_id = str(e.get('id', '') or e.get('url', ''))
                            if target_id in entry_id:
                                # Start from the video AFTER the found one
                                start_idx = idx + 1
                                break
                    if start_idx >= len(entries):
                        self.root.after(0, lambda: self._tt_dl_finish(
                            "Videoul specificat a fost găsit ca ultimul din listă. Nu mai sunt videouri de descărcat."))
                        return
                    entries = entries[start_idx:]
                    skipped_msg = f"Am sărit {start_idx} videouri. Descărc de la poziția {start_idx + 1}…"
                    self.root.after(0, lambda m=skipped_msg: self.tt_dl_status_var.set(m))

                # Limit to max_videos (unless "download all" is checked)
                if not download_all:
                    entries = entries[:max_videos]
                total = len(entries)

                if total == 0:
                    self.root.after(0, lambda: self._tt_dl_finish("Nicio înregistrare găsită după filtrare."))
                    return

                downloaded = 0
                last_video_url = None
                for i, entry in enumerate(entries):
                    if self._tt_dl_stop:
                        break
                    current_num = start_num + i
                    out_path = os.path.join(folder, f"{current_num}.mp4")
                    # Prefer webpage_url (stable TikTok watch URL) over raw stream URL
                    video_url = entry.get('webpage_url') or entry.get('url') or entry.get('id')
                    if not video_url:
                        continue

                    status_msg = f"Descarcă {i + 1}/{total} → {current_num}.mp4"
                    self.root.after(0, lambda msg=status_msg: self.tt_dl_status_var.set(msg))

                    # Download options: best quality, no watermark (via aweme_id trick / native formats)
                    ydl_opts_dl = {
                        'quiet': True,
                        'no_warnings': True,
                        'outtmpl': out_path,
                        # TikTok serves combined (pre-merged) audio+video streams; using bestvideo+bestaudio
                        # selects video-only streams and requires ffmpeg merging which often silently drops audio.
                        # Use 'b' (best combined) filtered by acodec!=none to guarantee audio is present.
                        'format': 'b[ext=mp4][vcodec!*=h265][acodec!=none]/b[ext=mp4][acodec!=none]/b[acodec!=none]/b',
                        'merge_output_format': 'mp4',
                        # TikTok-specific: prefer download URL (no watermark) over play URL
                        'extractor_args': {'tiktok': {'webpage_download': True}},
                        'overwrites': True,
                        'retries': 3,
                        'fragment_retries': 3,
                        'http_headers': {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        },
                    }
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
                            ydl.download([video_url])
                        downloaded += 1
                        # Track the webpage URL (not the raw stream URL) for the log file
                        last_video_url = entry.get('webpage_url') or video_url
                    except Exception as dl_err:
                        err_msg = f"Eroare la {current_num}.mp4: {dl_err}"
                        self.root.after(0, lambda m=err_msg: self.tt_dl_status_var.set(m))
                        import time as _t; _t.sleep(0.5)

                # --- Save progress log file ---
                import re as _re2, datetime as _dt
                _m = _re2.search(r'@([\w.]+)', url)
                account_name = f"@{_m.group(1)}" if _m else url
                safe_name = _re2.sub(r'[^\w.-]', '_', account_name.lstrip('@'))
                log_path = os.path.join(folder, f"{safe_name}_progress.txt")
                try:
                    with open(log_path, 'w', encoding='utf-8') as _lf:
                        _lf.write(f"Cont: {account_name}\n")
                        _lf.write(f"URL cont: {url}\n")
                        _lf.write(f"Categorie folosita: {sort_order}\n")
                        _lf.write(f"Videouri descarcate in aceasta sesiune: {downloaded}\n")
                        _lf.write(f"Numerotare: {start_num}–{start_num + downloaded - 1}\n")
                        if last_video_url:
                            _lf.write(f"\nUltimul video descarcat:\n{last_video_url}\n")
                            _lf.write("\n(Lipeste URL-ul de mai sus in campul 'Incepe de la video URL' ca sa continui de unde ai ramas)\n")
                        _lf.write(f"\nData: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                except Exception:
                    pass

                if self._tt_dl_stop:
                    finish_msg = f"Oprit. Descărcat {downloaded} din {total} fișiere în {folder}."
                else:
                    finish_msg = f"✅ Descărcat {downloaded}/{total} fișiere. Numerotare: {start_num}–{start_num + downloaded - 1} în {folder}"
                self.root.after(0, lambda msg=finish_msg: self._tt_dl_finish(msg))

            except Exception as e:
                err = str(e)
                self.root.after(0, lambda m=err: self._tt_dl_finish(f"❌ Eroare: {m}"))

        threading.Thread(target=_run, daemon=True).start()

    def _tt_dl_finish(self, msg: str):
        self.tt_dl_status_var.set(msg)
        self._tt_dl_set_ui(False)

    def _on_stop(self):
        """Stop the current processing job."""
        globals()['STOP_REQUESTED'] = True
        try:
            self.log_widget.config(state='normal')
            self.log_widget.insert('end', "[STOP] Stop requested — finishing current step…\n")
            self.log_widget.see('end')
            self.log_widget.config(state='disabled')
        except Exception:
            pass

    def on_run_single(self):
        try:
            globals()['STOP_REQUESTED'] = False
            video = self.video_var.get().strip()
            voice = self.voice_var.get().strip()
            music = self.music_var.get().strip()
            output = self.output_var.get().strip() or "final_tiktok.mp4"
            if not video or not music:
                messagebox.showwarning("Missing fields", "Please select VIDEO and MUSIC before running.")
                return
            try:
                selected_font_name = getattr(self, 'selected_font', None)
                selected_font_path = getattr(self, 'selected_font_path', None)
                pref_font = get_validated_font(selected_font_name, selected_font_path)
                # Log font being used
                try:
                    self.log_widget.config(state='normal')
                    self.log_widget.insert('end', f"[FONT DEBUG] selected_font={selected_font_name}, selected_font_path={selected_font_path}, using={pref_font}\n")
                    self.log_widget.see('end')
                    self.log_widget.config(state='disabled')
                except Exception:
                    pass
            except Exception:
                pref_font = None
            # create a queue for logs and start the job in a worker thread
            job = {"video": video, "voice": voice, "music": music, "output": output, "font": pref_font,
                   "custom_top_ratio": (self.top_percent_var.get()/100.0) if self.use_custom_crop_var.get() else None,
                   "custom_bottom_ratio": (self.bottom_percent_var.get()/100.0) if self.use_custom_crop_var.get() else None,
                   "mirror_video": self.mirror_video_var.get(),
                   "words_per_caption": self.words_per_caption_var.get(),
                   "use_4k": self.use_4k_var.get(),
                   "blur_radius": globals().get('STATIC_BG_BLUR_RADIUS', 25),
                   "bg_scale_extra": globals().get('BG_SCALE_EXTRA', 1.08),
                   "dim_factor": globals().get('DIM_FACTOR', 1.0),
                   # AI and caption settings
                   "use_ai_voice": self.use_ai_voice_var.get(),
                   "translation_enabled": self.translation_enabled_var.get(),
                   "target_language": self.target_language_var.get() if hasattr(self, 'target_language_var') else 'none',
                   "tts_language": self.tts_language_var.get() if hasattr(self, 'tts_language_var') else 'en',
                   "silence_threshold_ms": self.silence_threshold_var.get(),
                   # Font and border settings
                   "caption_text_color": globals().get('CAPTION_TEXT_COLOR', (255, 255, 255, 255)),
                   "caption_stroke_color": globals().get('CAPTION_STROKE_COLOR', (0, 0, 0, 150)),
                   "caption_stroke_width": globals().get('CAPTION_STROKE_WIDTH', 3),
                   "caption_font_size": globals().get('CAPTION_FONT_SIZE', 56),
                   "caption_y_offset": globals().get('CAPTION_Y_OFFSET', 0),
                   # Video effects (CapCut-style)
                   "effect_sharpness": self.effect_sharpness_var.get(),
                   "effect_sharpness_intensity": self.effect_sharpness_intensity_var.get(),
                   "effect_saturation": self.effect_saturation_var.get(),
                   "effect_saturation_intensity": self.effect_saturation_intensity_var.get(),
                   "effect_contrast": self.effect_contrast_var.get(),
                   "effect_contrast_intensity": self.effect_contrast_intensity_var.get(),
                   "effect_brightness": self.effect_brightness_var.get(),
                   "effect_brightness_intensity": self.effect_brightness_intensity_var.get(),
                   "effect_vintage": self.effect_vintage_var.get(),
                   "effect_vintage_intensity": self.effect_vintage_intensity_var.get(),
                   # Blur overlay (cover-up region)
                   "blur_overlay_enabled": self.blur_overlay_enabled_var.get(),
                   "blur_overlay_x": self.blur_overlay_x_var.get(),
                   "blur_overlay_y": self.blur_overlay_y_var.get(),
                   "blur_overlay_w": self.blur_overlay_w_var.get(),
                   "blur_overlay_h": self.blur_overlay_h_var.get(),
                   "blur_overlay_intensity": self.blur_overlay_intensity_var.get(),
                   # Audio volume settings (captured at job creation time)
                   "voice_gain": self.voice_gain_var.get(),
                   "music_gain": self.music_gain_var.get()}
            q = self.q
            # Extract effect settings
            effect_settings = {
                'effect_sharpness': job.get("effect_sharpness", False),
                'effect_sharpness_intensity': job.get("effect_sharpness_intensity", 1.5),
                'effect_saturation': job.get("effect_saturation", False),
                'effect_saturation_intensity': job.get("effect_saturation_intensity", 1.3),
                'effect_contrast': job.get("effect_contrast", False),
                'effect_contrast_intensity': job.get("effect_contrast_intensity", 1.2),
                'effect_brightness': job.get("effect_brightness", False),
                'effect_brightness_intensity': job.get("effect_brightness_intensity", 1.15),
                'effect_vintage': job.get("effect_vintage", False),
                'effect_vintage_intensity': job.get("effect_vintage_intensity", 0.3),
                'blur_overlay_enabled': job.get("blur_overlay_enabled", False),
                'blur_overlay_x': job.get("blur_overlay_x", 10.0),
                'blur_overlay_y': job.get("blur_overlay_y", 10.0),
                'blur_overlay_w': job.get("blur_overlay_w", 20.0),
                'blur_overlay_h': job.get("blur_overlay_h", 15.0),
                'blur_overlay_intensity': job.get("blur_overlay_intensity", 20)
            }
            # Run in background thread so GUI remains responsive
            _job_kwargs = {"custom_top_ratio": job.get("custom_top_ratio"), "custom_bottom_ratio": job.get("custom_bottom_ratio"), "mirror_video": job.get("mirror_video", False), "words_per_caption": job.get("words_per_caption", 2), "use_4k": job.get("use_4k", False), "blur_radius": job.get("blur_radius"), "bg_scale_extra": job.get("bg_scale_extra"), "dim_factor": job.get("dim_factor"), "effect_settings": effect_settings, "use_ai_voice": job.get("use_ai_voice", False), "target_language": job.get("target_language", 'none'), "translation_enabled": job.get("translation_enabled", False), "tts_language": job.get("tts_language", 'en'), "caption_text_color": job.get("caption_text_color"), "caption_stroke_color": job.get("caption_stroke_color"), "caption_stroke_width": job.get("caption_stroke_width"), "caption_font_size": job.get("caption_font_size"), "caption_y_offset": job.get("caption_y_offset"), "voice_gain": job.get("voice_gain"), "music_gain": job.get("music_gain")}
            def _run_and_signal(video=job["video"], voice=job["voice"], music=job["music"], out=job["output"], font=job.get("font"), kw=_job_kwargs, _q=q):
                try:
                    process_single_job(video, voice, music, out, _q, font, **kw)
                finally:
                    _q.put("[SINGLE_DONE]")
            t = threading.Thread(target=_run_and_signal, daemon=True)
            t.start()
            try:
                self.log_widget.config(state='normal')
                self.log_widget.insert('end', f"[RUN] Started single job -> {Path(output).name}\n")
                self.log_widget.see('end')
                self.log_widget.config(state='disabled')
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Run error", str(e))

    def _run_single_thread(self, video, voice, music, output, top_ratio, bottom_ratio):
        words_per_caption = self.words_per_caption_var.get()
        use_ai_voice = self.use_ai_voice_var.get()
        target_language = self.target_language_var.get()
        translation_enabled = self.translation_enabled_var.get()
        tts_language = self.tts_language_var.get()
        silence_threshold_ms = self.silence_threshold_var.get()
        voice_gain_val = self.voice_gain_var.get()
        music_gain_val = self.music_gain_var.get()
        process_single_job(video, voice, music, output, self.q, custom_top_ratio=top_ratio, custom_bottom_ratio=bottom_ratio, words_per_caption=words_per_caption, use_ai_voice=use_ai_voice, target_language=target_language, translation_enabled=translation_enabled, tts_language=tts_language, silence_threshold_ms=silence_threshold_ms, voice_gain=voice_gain_val, music_gain=music_gain_val)
        self.q.put("[SINGLE_DONE]")

    def run_queue(self):
        if not self.jobs:
            messagebox.showerror("Empty queue", "Nu ai niciun job în listă")
            return
        self.run_queue_btn.config(state="disabled")
        self.run_single_btn.config(state="disabled")
        self.log_widget.config(state="normal")
        self.log_widget.delete("1.0", tk.END)
        self.log_widget.config(state="disabled")
        t = threading.Thread(target=queue_worker, args=(list(self.jobs), self.q), daemon=True)
        t.start()

    def on_mini_refresh_clicked(self):
        self._mini_update_worker_async()

    def _rgba_from_hex(self, hx):
        try:
            if hx.startswith('#'): hx = hx[1:]
            if len(hx) == 6:
                r = int(hx[0:2],16); g = int(hx[2:4],16); b = int(hx[4:6],16)
                return (r,g,b,255)
            if len(hx) == 8:
                r = int(hx[0:2],16); g = int(hx[2:4],16); b = int(hx[4:6],16); a = int(hx[6:8],16)
                return (r,g,b,a)
        except Exception:
            pass
        return (255,255,255,255)

    def _update_color_canvases(self):
        try:
            try:
                txt = CAPTION_TEXT_COLOR
                col = '#%02x%02x%02x' % (txt[0], txt[1], txt[2])
                if hasattr(self, 'text_color_canvas') and self.text_color_canvas:
                    self.text_color_canvas.delete('all')
                    self.text_color_canvas.create_oval(2,2,26,26, fill=col, outline='white')
            except Exception:
                pass
            try:
                st = CAPTION_STROKE_COLOR
                col2 = '#%02x%02x%02x' % (st[0], st[1], st[2])
                if hasattr(self, 'stroke_color_canvas') and self.stroke_color_canvas:
                    self.stroke_color_canvas.delete('all')
                    self.stroke_color_canvas.create_oval(2,2,26,26, fill=col2, outline='white')
            except Exception:
                pass
        except Exception:
            pass

    def on_pick_text_color(self):
        try:
            from tkinter import colorchooser
            col = colorchooser.askcolor(title='Choose caption text color')
            if col and col[1]:
                rgba = self._rgba_from_hex(col[1])
                globals()['CAPTION_TEXT_COLOR'] = rgba
                self._update_color_canvases()
                # Trigger live preview update
                self._mini_update_worker_async()
        except Exception as e:
            try:
                self.log_widget.config(state='normal'); self.log_widget.insert('end', f"[COLOR-ERR] {e}\n"); self.log_widget.config(state='disabled')
            except Exception:
                pass

    def on_pick_stroke_color(self):
        try:
            from tkinter import colorchooser
            col = colorchooser.askcolor(title='Choose caption stroke color')
            if col and col[1]:
                rgba = self._rgba_from_hex(col[1])
                if len(rgba) == 4 and rgba[3] == 255:
                    rgba = (rgba[0], rgba[1], rgba[2], 150)
                globals()['CAPTION_STROKE_COLOR'] = rgba
                self._update_color_canvases()
                # Trigger live preview update
                self._mini_update_worker_async()
        except Exception as e:
            try:
                self.log_widget.config(state='normal'); self.log_widget.insert('end', f"[COLOR-ERR] {e}\n"); self.log_widget.config(state='disabled')
            except Exception:
                pass

    def _set_text_color_hex(self, hx):
        try:
            rgba = self._rgba_from_hex(hx)
            globals()['CAPTION_TEXT_COLOR'] = rgba
            self._update_color_canvases()
            # Trigger live preview update
            self._mini_update_worker_async()
        except Exception:
            pass

    def _set_stroke_color_hex(self, hx):
        try:
            rgba = self._rgba_from_hex(hx)
            rgba = (rgba[0], rgba[1], rgba[2], 150)
            globals()['CAPTION_STROKE_COLOR'] = rgba
            self._update_color_canvases()
            # Trigger live preview update
            self._mini_update_worker_async()
        except Exception:
            pass


    def _get_current_effect_settings(self):
        """Get current effect settings from UI for preview."""
        return {
            'effect_sharpness': self.effect_sharpness_var.get(),
            'effect_sharpness_intensity': self.effect_sharpness_intensity_var.get(),
            'effect_saturation': self.effect_saturation_var.get(),
            'effect_saturation_intensity': self.effect_saturation_intensity_var.get(),
            'effect_contrast': self.effect_contrast_var.get(),
            'effect_contrast_intensity': self.effect_contrast_intensity_var.get(),
            'effect_brightness': self.effect_brightness_var.get(),
            'effect_brightness_intensity': self.effect_brightness_intensity_var.get(),
            'effect_vintage': self.effect_vintage_var.get(),
            'effect_vintage_intensity': self.effect_vintage_intensity_var.get(),
            'blur_overlay_enabled': self.blur_overlay_enabled_var.get(),
            'blur_overlay_x': self.blur_overlay_x_var.get(),
            'blur_overlay_y': self.blur_overlay_y_var.get(),
            'blur_overlay_w': self.blur_overlay_w_var.get(),
            'blur_overlay_h': self.blur_overlay_h_var.get(),
            'blur_overlay_intensity': self.blur_overlay_intensity_var.get()
        }

    def _mini_update_worker_async(self):
        video = self.video_var.get().strip()
        if not video or not os.path.isfile(video):
            self._set_mini_canvas_placeholder("No video selected")
            return
        if self.mini_worker_thread and self.mini_worker_thread.is_alive(): return
        tval = float(self.time_var.get())
        # decide whether to pass cached clip or path
        clip_or_path = video
        try:
            if hasattr(self, '_preview_clip') and self._preview_clip and getattr(self._preview_clip, 'filename', None):
                try:
                    if os.path.abspath(video) == os.path.abspath(self._preview_clip.filename):
                        clip_or_path = self._preview_clip
                except Exception:
                    clip_or_path = video
        except Exception:
            clip_or_path = video
        self.mini_worker_thread = threading.Thread(target=self._mini_extract_and_update, args=(clip_or_path, tval), daemon=True)
        self.mini_worker_thread.start()

    def _mini_extract_and_update(self, video_path, time_val):
        try:
            img, scale = extract_and_scale_frame(video_path, time_sec=time_val, desired_width=360)
            
            # Apply video effects to preview
            effect_settings = self._get_current_effect_settings()
            if any([effect_settings.get('effect_sharpness', False),
                    effect_settings.get('effect_saturation', False),
                    effect_settings.get('effect_contrast', False),
                    effect_settings.get('effect_brightness', False),
                    effect_settings.get('effect_vintage', False)]):
                img = apply_video_effects(img, effect_settings)
            
            # Apply blur overlay to preview (real-time visualization)
            if effect_settings.get('blur_overlay_enabled', False):
                img = apply_blur_overlay_to_preview(img, effect_settings)
            
            self.mini_base_img = img
            self.mini_scale = scale
            top_pct = float(self.top_percent_var.get())/100.0
            bottom_pct = float(self.bottom_percent_var.get())/100.0
            composed = overlay_crop_on_image(img, top_pct, bottom_pct)
            photo = ImageTk.PhotoImage(composed)
            def apply():
                self.mini_canvas.config(width=composed.width, height=composed.height)
                # Use centralized redraw method that ALWAYS includes caption indicator
                self._redraw_mini_canvas_with_caption_indicator(composed, top_pct, bottom_pct)
                self.preview_info.config(text=f"Preview at {seconds_to_hms(time_val)} | scale={self.mini_scale:.2f}")
                self.font_info.config(text=f"Font: {LOADED_FONT_PATH or 'not loaded'}")
            self.root.after(0, apply)
        except Exception as e:
            self.root.after(0, lambda: self._set_mini_canvas_placeholder("Preview error"))
            self.q.put(f"Mini-preview error: {e}")

    def _start_mini_persistent_worker(self):
        # Start a background thread that listens for slider requests and updates preview in-place.
        if self._mini_persistent_thread and self._mini_persistent_thread.is_alive():
            return
        self._mini_request_stop = False
        self._mini_persistent_thread = threading.Thread(target=self._mini_persistent_worker, daemon=True)
        self._mini_persistent_thread.start()

    def _stop_mini_persistent_worker(self):
        try:
            self._mini_request_stop = True
            if self._mini_request_event:
                self._mini_request_event.set()
        except Exception:
            pass

    def _mini_persistent_worker(self):
        # Background loop that processes the latest requested time and updates the mini preview.
        # It uses the cached VideoFileClip if available for faster seeks.
        min_interval = 0.03  # aim for up to ~30 FPS responsiveness during scrubbing
        last_ts = 0.0
        cap = None
        while not getattr(self, '_mini_request_stop', False):
            try:
                self._mini_request_event.wait(timeout=min_interval)
            except Exception:
                pass
            if getattr(self, '_mini_request_stop', False):
                break
            req_time = None
            try:
                req_time = self._mini_request_time
            except Exception:
                req_time = None
            if req_time is None:
                self._mini_request_event.clear()
                continue
            now = time.time()
            if now - last_ts < min_interval:
                time.sleep(max(0.0, min_interval - (now - last_ts)))
            last_ts = time.time()
            video_path = self.video_var.get().strip()
            clip_to_use = None
            created_local_clip = False
            try:
                if hasattr(self, '_preview_clip') and self._preview_clip and getattr(self._preview_clip, 'filename', None):
                    try:
                        if os.path.abspath(video_path) == os.path.abspath(self._preview_clip.filename):
                            clip_to_use = self._preview_clip
                    except Exception:
                        clip_to_use = None
                if clip_to_use is None:
                    clip_to_use = VideoFileClip(video_path)
                    created_local_clip = True
            except Exception:
                clip_to_use = None
                created_local_clip = False
                try:
                    import cv2
                    if cap is None:
                        cap = cv2.VideoCapture(video_path)
                except Exception:
                    cap = None
            try:
                if clip_to_use is not None:
                    t = float(max(0.0, min(req_time, max(0.001, clip_to_use.duration - 0.001))))
                    frame = clip_to_use.get_frame(t)
                    img = Image.fromarray(frame).convert('RGB')
                elif cap is not None and cap.isOpened():
                    import cv2
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    frame_num = int(round(req_time * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, fr = cap.read()
                    if ret:
                        img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).convert('RGB')
                    else:
                        img = None
                else:
                    img = None
                if img is not None:
                    w, h = img.size
                    desired_width = 360
                    scale = desired_width / float(w) if w else 1.0
                    new_w = desired_width
                    new_h = int(round(h * scale))
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                    
                    # Apply video effects to preview
                    effect_settings = self._get_current_effect_settings()
                    if any([effect_settings.get('effect_sharpness', False),
                            effect_settings.get('effect_saturation', False),
                            effect_settings.get('effect_contrast', False),
                            effect_settings.get('effect_brightness', False),
                            effect_settings.get('effect_vintage', False)]):
                        img = apply_video_effects(img, effect_settings)
                    
                    # Apply blur overlay to preview (real-time visualization)
                    if effect_settings.get('blur_overlay_enabled', False):
                        img = apply_blur_overlay_to_preview(img, effect_settings)
                    
                    top_pct = float(self.top_percent_var.get())/100.0
                    bottom_pct = float(self.bottom_percent_var.get())/100.0
                    composed = overlay_crop_on_image(img, top_pct, bottom_pct)
                    try:
                        def _update_with_caption():
                            try:
                                # Use centralized redraw method that ALWAYS includes caption indicator
                                self._redraw_mini_canvas_with_caption_indicator(composed, top_pct, bottom_pct)
                            except Exception:
                                pass
                        self.root.after(0, _update_with_caption)
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                try:
                    if clip_to_use is not None and created_local_clip:
                        try:
                            clip_to_use.close()
                        except Exception:
                            pass
                except Exception:
                    pass
            try:
                self._mini_request_event.clear()
            except Exception:
                pass
        try:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception:
            pass


    def _set_mini_canvas_placeholder(self, text):
        self.mini_canvas.delete("all")
        w = int(self.mini_canvas.cget("width"))
        h = int(self.mini_canvas.cget("height"))
        self.mini_canvas.create_rectangle(0,0,w,h, fill="#222222", outline="")
        self.mini_canvas.create_text(w//2, h//2, text=text, fill="#ddd", font=("Arial", 12))

    def update_mini_preview_immediate(self):
        if self.mini_base_img is None:
            self._mini_update_worker_async()
        else:
            # Redraw with current crop settings and caption indicator
            try:
                top_pct = float(self.top_percent_var.get())/100.0
                bottom_pct = float(self.bottom_percent_var.get())/100.0
                composed = overlay_crop_on_image(self.mini_base_img, top_pct, bottom_pct)
                self._redraw_mini_canvas_with_caption_indicator(composed, top_pct, bottom_pct)
            except Exception:
                pass

    def _rgba_from_hex(self, hx):
        try:
            if hx.startswith('#'): hx = hx[1:]
            if len(hx) == 6:
                r = int(hx[0:2],16); g = int(hx[2:4],16); b = int(hx[4:6],16)
                return (r,g,b,255)
            if len(hx) == 8:
                r = int(hx[0:2],16); g = int(hx[2:4],16); b = int(hx[4:6],16); a = int(hx[6:8],16)
                return (r,g,b,a)
        except Exception:
            pass
        return (255,255,255,255)

    def _update_color_canvases(self):
        try:
            try:
                txt = CAPTION_TEXT_COLOR
                col = '#%02x%02x%02x' % (txt[0], txt[1], txt[2])
                if hasattr(self, 'text_color_canvas') and self.text_color_canvas:
                    self.text_color_canvas.delete('all')
                    self.text_color_canvas.create_oval(2,2,26,26, fill=col, outline='white')
            except Exception:
                pass
            try:
                st = CAPTION_STROKE_COLOR
                col2 = '#%02x%02x%02x' % (st[0], st[1], st[2])
                if hasattr(self, 'stroke_color_canvas') and self.stroke_color_canvas:
                    self.stroke_color_canvas.delete('all')
                    self.stroke_color_canvas.create_oval(2,2,26,26, fill=col2, outline='white')
            except Exception:
                pass
        except Exception:
            pass

    def on_pick_text_color(self):
        try:
            from tkinter import colorchooser
            col = colorchooser.askcolor(title='Choose caption text color')
            if col and col[1]:
                rgba = self._rgba_from_hex(col[1])
                globals()['CAPTION_TEXT_COLOR'] = rgba
                self._update_color_canvases()
                # Trigger live preview update
                self._mini_update_worker_async()
        except Exception as e:
            try:
                self.log_widget.config(state='normal'); self.log_widget.insert('end', f"[COLOR-ERR] {e}\n"); self.log_widget.config(state='disabled')
            except Exception:
                pass

    def on_pick_stroke_color(self):
        try:
            from tkinter import colorchooser
            col = colorchooser.askcolor(title='Choose caption stroke color')
            if col and col[1]:
                rgba = self._rgba_from_hex(col[1])
                if len(rgba) == 4 and rgba[3] == 255:
                    rgba = (rgba[0], rgba[1], rgba[2], 150)
                globals()['CAPTION_STROKE_COLOR'] = rgba
                self._update_color_canvases()
                # Trigger live preview update
                self._mini_update_worker_async()
        except Exception as e:
            try:
                self.log_widget.config(state='normal'); self.log_widget.insert('end', f"[COLOR-ERR] {e}\n"); self.log_widget.config(state='disabled')
            except Exception:
                pass

    def _set_text_color_hex(self, hx):
        try:
            rgba = self._rgba_from_hex(hx)
            globals()['CAPTION_TEXT_COLOR'] = rgba
            self._update_color_canvases()
            # Trigger live preview update
            self._mini_update_worker_async()
        except Exception:
            pass

    def _set_stroke_color_hex(self, hx):
        try:
            rgba = self._rgba_from_hex(hx)
            rgba = (rgba[0], rgba[1], rgba[2], 150)
            globals()['CAPTION_STROKE_COLOR'] = rgba
            self._update_color_canvases()
            # Trigger live preview update
            self._mini_update_worker_async()
        except Exception:
            pass

            return
        top_pct = float(self.top_percent_var.get())/100.0
        bottom_pct = float(self.bottom_percent_var.get())/100.0
        if top_pct + bottom_pct > 0.95:
            bottom_pct = max(0.0, 0.95 - top_pct)
            self.bottom_percent_var.set(bottom_pct*100)
        composed = overlay_crop_on_image(self.mini_base_img.copy(), top_pct, bottom_pct)
        photo = ImageTk.PhotoImage(composed)
        self.mini_canvas.config(width=composed.width, height=composed.height)
        self.mini_canvas.delete("all")
        self.mini_canvas.create_image(0, 0, anchor="nw", image=photo, tags="base_img")
        self.mini_image_ref = photo
        h = composed.height
        top_y = int(round(h * top_pct))
        bottom_y = int(round(h * (1.0 - bottom_pct)))
        self.mini_canvas.create_rectangle(0, top_y-4, composed.width, top_y+4, fill="#000000", stipple="gray50", tags="line_top")
        self.mini_canvas.create_rectangle(0, bottom_y-4, composed.width, bottom_y+4, fill="#000000", stipple="gray50", tags="line_bottom")
        self.mini_canvas.create_text(6, max(6, top_y-18), anchor="nw", text=f"Top {int(top_pct*100)}% ({top_y}px)", fill="#fff", font=("Arial",9), tags="label_top")
        self.mini_canvas.create_text(6, min(h-18, bottom_y+6), anchor="nw", text=f"Bottom {int(bottom_pct*100)}% ({h - bottom_y}px)", fill="#fff", font=("Arial",9), tags="label_bottom")
        self.top_label.config(text=f"{self.top_percent_var.get():.1f}%")
        self.bottom_label.config(text=f"{self.bottom_percent_var.get():.1f}%")

    def on_time_changed(self, _=None):
        # update the time label immediately
        t = float(self.time_var.get())
        self.time_label.config(text=seconds_to_hms(t))

        # live scrubbing: set desired preview time and notify persistent worker
        try:
            self._mini_request_time = t
            try:
                if not (self._mini_persistent_thread and self._mini_persistent_thread.is_alive()):
                    self._start_mini_persistent_worker()
            except Exception:
                pass
            try:
                self._mini_request_event.set()
            except Exception:
                pass
        except Exception:
            pass
    
    def on_zoom_changed(self, _=None):
        """Called when zoom slider is moved"""
        global VIDEO_ZOOM_SCALE
        zoom = float(self.zoom_var.get())
        VIDEO_ZOOM_SCALE = zoom
        self.zoom_label.config(text=f"{zoom:.2f}x")
        # Update TikTok preview when zoom changes
        self.on_tiktok_preview_refresh()
    
    def on_tiktok_preview_refresh(self):
        """Refresh the TikTok format preview with current settings.
        
        Safe to call from any thread: heavy work (MoviePy frame extraction,
        PIL rendering) runs inline, then the lightweight Tkinter canvas update
        is scheduled on the main thread via root.after().
        """
        try:
            if not self.video_var.get() or not os.path.isfile(self.video_var.get()):
                return
            
            # Get current video path and time
            video_path = self.video_var.get()
            current_time = float(self.time_var.get())
            
            # Load video clip at current time
            from moviepy.editor import VideoFileClip
            video_clip = None
            try:
                video_clip = VideoFileClip(video_path)
            except Exception as ve:
                print(f"TikTok preview: Cannot open video file — {ve}")
                return
            
            try:
                # Get frame at current time
                if current_time > video_clip.duration:
                    current_time = 0.0
                
                frame = video_clip.get_frame(current_time)
            except Exception as fe:
                print(f"TikTok preview: Cannot read frame at {current_time:.2f}s — {fe}")
                video_clip.close()
                return
            video_clip.close()
            
            # Get crop settings
            crop_top_ratio = float(self.top_percent_var.get()) / 100.0
            crop_bottom_ratio = float(self.bottom_percent_var.get()) / 100.0
            
            # Calculate crop coordinates
            img_h, img_w = frame.shape[:2]
            crop_y = int(img_h * crop_top_ratio)
            crop_h = int(img_h * (1.0 - crop_top_ratio - crop_bottom_ratio))
            crop_x = 0
            crop_w = img_w
            
            # Crop the frame
            cropped_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # Apply zoom scaling — MUST match export scaling (processing_job)
            zoom = float(self.zoom_var.get())
            width_scale = WIDTH / crop_w
            base_scale_factor = 1.03
            
            # Same MIN_FG_HEIGHT_RATIO logic as export (processing_job lines 3699-3707)
            min_fg_height = HEIGHT * MIN_FG_HEIGHT_RATIO
            estimated_height = crop_h * width_scale * base_scale_factor
            if estimated_height < min_fg_height and crop_h > 0:
                height_scale = min_fg_height / crop_h
                fg_base_scale = max(width_scale, height_scale)
            else:
                fg_base_scale = width_scale
            
            fg_scale = fg_base_scale * base_scale_factor * zoom
            
            # Scale the frame
            scaled_h = max(2, int(crop_h * fg_scale))
            scaled_w = max(2, int(crop_w * fg_scale))
            
            from PIL import Image, ImageFilter, ImageEnhance
            import numpy as np
            
            pil_img = Image.fromarray(cropped_frame)
            pil_img = pil_img.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            
            # Create blurred background from the CROPPED frame (matches export)
            # Background uses same crop as foreground, then zooms to fill canvas
            bg_img = Image.fromarray(cropped_frame)
            # Scale to FILL canvas (cover entire area, may crop edges)
            bg_ar = bg_img.width / bg_img.height
            canvas_ar = WIDTH / HEIGHT
            if bg_ar > canvas_ar:
                # Wider than canvas: scale to height, crop width
                fill_h = HEIGHT
                fill_w = int(HEIGHT * bg_ar)
            else:
                # Taller than canvas: scale to width, crop height
                fill_w = WIDTH
                fill_h = int(WIDTH / bg_ar)
            bg_img = bg_img.resize((fill_w, fill_h), Image.Resampling.LANCZOS)
            # Center-crop to exact canvas size
            cx = (fill_w - WIDTH) // 2
            cy = (fill_h - HEIGHT) // 2
            bg_img = bg_img.crop((cx, cy, cx + WIDTH, cy + HEIGHT))
            bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=20))
            # Apply brightness: match export's NV12 brightness compensation
            effective_brightness = DIM_FACTOR * (1.0 + BG_BRIGHTNESS_BOOST)
            if abs(effective_brightness - 1.0) > 0.01:
                bg_img = ImageEnhance.Brightness(bg_img).enhance(effective_brightness)
            canvas = bg_img.copy()
            
            # Center the scaled foreground on canvas (clips if wider/taller)
            paste_x = (WIDTH - scaled_w) // 2
            paste_y = (HEIGHT - scaled_h) // 2
            canvas.paste(pil_img, (paste_x, paste_y))
            
            # Overlay caption on the TikTok preview at exact FFmpeg position
            # FFmpeg drawtext: y = h - th + y_offset
            # This shows exactly where the caption will appear in the final export
            try:
                y_off = globals().get('CAPTION_Y_OFFSET', 0)
                pf = None
                try:
                    selected_font_name = getattr(self, 'selected_font', None)
                    selected_font_path = getattr(self, 'selected_font_path', None)
                    pf = get_validated_font(selected_font_name, selected_font_path)
                except Exception:
                    pf = None
                # Generate sample caption text
                wpc = globals().get('WORDS_PER_GROUP', 2)
                word_list = ["WORD", "ONE", "TWO"]
                sample_words = word_list[:min(wpc, len(word_list))]
                sample_text = " ".join(sample_words)
                cap_img = generate_caption_image(sample_text, preferred_font=pf, log=lambda s: None)
                if not isinstance(cap_img, Image.Image):
                    cap_img = Image.fromarray(cap_img)
                cap_img = cap_img.convert('RGBA')
                # Position: FFmpeg uses y = h - th + y_offset for text top
                # PIL image includes padding above and below text that FFmpeg's th doesn't.
                # Below the text: padding_y(24) + 2*extra_bottom_margin + 4px of padding.
                # We shift the image down by this amount so the visible text aligns with FFmpeg.
                font_size = globals().get('CAPTION_FONT_SIZE', 56)
                extra_bottom_margin = int(font_size * 0.35)
                bottom_pad = 24 + 2 * extra_bottom_margin + 4  # padding below text in PIL image
                cap_x = (WIDTH - cap_img.width) // 2
                cap_y = HEIGHT - cap_img.height + y_off + bottom_pad
                cap_y = max(0, cap_y)
                canvas_rgba = canvas.convert('RGBA')
                canvas_rgba.paste(cap_img, (cap_x, cap_y), cap_img)
                canvas = canvas_rgba.convert('RGB')
            except Exception as cap_err:
                print(f"[TIKTOK-PREVIEW] Caption overlay skipped: {cap_err}")
            
            # Scale down to preview size (180x320)
            preview_canvas = canvas.resize((180, 320), Image.Resampling.LANCZOS)

            # Display in TikTok preview canvas — must run on main thread.
            # Use root.after(0, ...) so this is safe from both main and worker threads.
            photo = ImageTk.PhotoImage(preview_canvas)
            def _apply_preview(photo_image=photo):
                try:
                    self.tiktok_preview_canvas.delete("all")
                    self.tiktok_preview_canvas.create_image(90, 160, image=photo_image)
                    self.tiktok_preview_image_ref = photo_image  # Keep reference
                except Exception:
                    pass
            self.root.after(0, _apply_preview)
            
        except Exception as e:
            print(f"TikTok preview error: {e}")
            import traceback
            traceback.print_exc()

    def _refresh_tiktok_preview_async(self):
        """Run TikTok preview refresh in a background thread to avoid blocking UI.
        
        on_tiktok_preview_refresh() opens a MoviePy VideoFileClip each call,
        which is heavy and blocks the main thread. Running it synchronously
        from slider handlers (font size, Y offset) causes the slider to freeze
        after 1-2 changes. This method spawns a daemon thread and skips if a
        previous refresh is still running.
        
        Note: This is always called from the Tkinter main thread (via after()
        debounce), so the is_alive() check-and-set is not racy in practice.
        Same pattern as _mini_update_worker_async.
        """
        t = getattr(self, '_tiktok_preview_thread', None)
        if t is not None and t.is_alive():
            return  # previous refresh still running, skip
        def _worker():
            try:
                self.on_tiktok_preview_refresh()
            except Exception:
                pass
        self._tiktok_preview_thread = threading.Thread(target=_worker, daemon=True)
        self._tiktok_preview_thread.start()

    def on_font_selected(self, event=None):
        try:
            sel = None
            w = self.font_listbox
            if w.curselection():
                sel = w.get(w.curselection()[0])
        
            if sel:
                # IMMEDIATELY clear ALL old font data to prevent stale values
                self.selected_font_path = None
                self.selected_font = sel
                # Clear global font cache so export uses the new font
                clear_font_cache()
                
                try:
                    fnt = tkfont.Font(family=sel, size=16)
                    self.font_preview.config(font=fnt)
                except Exception:
                    try:
                        self.font_preview.config(font=(sel, 14))
                    except Exception:
                        pass
                
                # try to find a matching .ttf in C:\tiktok and store path
                try:
                    found = find_ttf_in_tiktok(sel, search_root=os.path.normpath("C:\\tiktok"))
                    if found:
                        self.selected_font_path = found
                        # Also update the global LOADED_FONT_PATH to match
                        globals()['LOADED_FONT_PATH'] = found
                        try:
                            self.saved_font_label.config(text=f"Found TTF: {os.path.basename(found)}")
                        except Exception:
                            pass
                    else:
                        # Font not found in C:\tiktok - use font name for downstream search
                        self.selected_font_path = None
                        try:
                            self.saved_font_label.config(text=f"Will search for: {sel}")
                        except Exception:
                            pass
                except Exception:
                    self.selected_font_path = None

        except Exception:
            pass

    def _mini_on_mouse_down(self, event):
        if self.mini_base_img is None: return
        canvas_h = int(self.mini_canvas.cget("height"))
        y = event.y
        top_y = int(round(canvas_h * (float(self.top_percent_var.get())/100.0)))
        bottom_y = int(round(canvas_h * (1.0 - float(self.bottom_percent_var.get())/100.0)))
        if abs(y - top_y) <= max(self.drag_threshold_px, 6):
            self.dragging = "top"
        elif abs(y - bottom_y) <= max(self.drag_threshold_px, 6):
            self.dragging = "bottom"
        else:
            # Check if click is inside the blur box
            if getattr(self, 'blur_overlay_enabled_var', None) and self.blur_overlay_enabled_var.get():
                canvas_w = int(self.mini_canvas.cget("width"))
                bx = self.blur_overlay_x_var.get() / 100.0
                by = self.blur_overlay_y_var.get() / 100.0
                bw = self.blur_overlay_w_var.get() / 100.0
                bh_frac = self.blur_overlay_h_var.get() / 100.0
                bx1 = bx * canvas_w; bx2 = (bx + bw) * canvas_w
                by1 = by * canvas_h; by2 = (by + bh_frac) * canvas_h
                tolerance = 12
                ex = event.x
                if (bx1 - tolerance <= ex <= bx2 + tolerance) and (by1 - tolerance <= y <= by2 + tolerance):
                    self.dragging = "blur_y"
                    self._mini_blur_drag_start_y = y
                    self._mini_blur_drag_start_val = self.blur_overlay_y_var.get()
                    return
            self.dragging = None

    def _mini_on_mouse_move(self, event):
        if self.dragging is None: return
        canvas_h = int(self.mini_canvas.cget("height"))
        y = max(0, min(canvas_h-1, event.y))
        if self.dragging == "top":
            top_ratio = y / float(canvas_h)
            bottom_ratio = float(self.bottom_percent_var.get())/100.0
            if top_ratio + bottom_ratio > 0.95:
                top_ratio = max(0.0, 0.95 - bottom_ratio)
            self.top_percent_var.set(round(top_ratio*100, 2))
        elif self.dragging == "bottom":
            bottom_ratio = 1.0 - (y / float(canvas_h))
            top_ratio = float(self.top_percent_var.get())/100.0
            if top_ratio + bottom_ratio > 0.95:
                bottom_ratio = max(0.0, 0.95 - top_ratio)
            self.bottom_percent_var.set(round(bottom_ratio*100, 2))
        elif self.dragging == "blur_y":
            delta_px = event.y - self._mini_blur_drag_start_y
            delta_pct = (delta_px / float(canvas_h)) * 100.0
            new_y = self._mini_blur_drag_start_val + delta_pct
            bh = self.blur_overlay_h_var.get()
            new_y = max(0.0, min(100.0 - bh, new_y))
            self.blur_overlay_y_var.set(round(new_y, 1))
            self.update_mini_preview_immediate()
            return
        self.top_label.config(text=f"{self.top_percent_var.get():.1f}%")
        self.bottom_label.config(text=f"{self.bottom_percent_var.get():.1f}%")
        self.update_mini_preview_immediate()
        self.use_custom_crop_var.set(True)

    def _mini_on_mouse_up(self, event):
        self.dragging = None

    def on_save_crop(self):
        top = float(self.top_percent_var.get())
        bottom = float(self.bottom_percent_var.get())
        ok, info = save_crop_settings(top, bottom)
        if ok:
            messagebox.showinfo("Saved", f"Crop settings saved to {info}")
        else:
            messagebox.showerror("Save failed", f"Could not save crop settings: {info}")

    def on_load_crop(self):
        data = load_crop_settings()
        if not data:
            messagebox.showwarning("Not found", f"No saved settings at '{CROP_SETTINGS_FILE}'")
            return
        try:
            self.top_percent_var.set(float(data.get("top_pct", CROP_TOP_RATIO*100)))
            self.bottom_percent_var.set(float(data.get("bottom_pct", CROP_BOTTOM_RATIO*100)))
            self.top_label.config(text=f"{self.top_percent_var.get():.1f}%")
            self.bottom_label.config(text=f"{self.bottom_percent_var.get():.1f}%")
            self.use_custom_crop_var.set(True)
            self.update_mini_preview_immediate()
            messagebox.showinfo("Loaded", "Crop settings loaded and applied to UI (Use custom crop enabled).")
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not load crop settings: {e}")

    def save_preset(self):
        """Save all current settings to a preset file."""
        try:
            preset_data = {
                # File paths (optional - user might not want to save these)
                "video_path": self.video_var.get(),
                "voice_path": self.voice_var.get(),
                "music_path": self.music_var.get(),
                "output_path": self.output_var.get(),
                
                # Video settings
                "mirror_video": self.mirror_video_var.get(),
                "use_4k": self.use_4k_var.get(),
                "use_custom_crop": self.use_custom_crop_var.get(),
                "top_percent": self.top_percent_var.get(),
                "bottom_percent": self.bottom_percent_var.get(),
                
                # Audio settings
                "voice_gain": self.voice_gain_var.get(),
                "music_gain": self.music_gain_var.get(),
                
                # Translation/TTS settings
                "translation_enabled": self.translation_enabled_var.get(),
                "target_language": self.target_language_var.get(),
                "translation_custom_prompt": self.translation_prompt_var.get() if hasattr(self, 'translation_prompt_var') else "",
                "openai_api_key": self.openai_key_var.get().strip() if hasattr(self, 'openai_key_var') else "",
                "use_ai_voice": self.use_ai_voice_var.get(),
                "tts_language": self.tts_language_var.get(),
                "tts_voice": self.tts_voice_var.get(),
                "tts_fallback_voice": self.tts_fallback_voice_var.get() if hasattr(self, 'tts_fallback_voice_var') else "Auto (Default)",
                "silence_threshold": self.silence_threshold_var.get(),
                "tts_speed": self.tts_speed_var.get() if hasattr(self, 'tts_speed_var') else 1.0,
                
                # Caption settings
                "words_per_caption": self.words_per_caption_var.get(),
                "caption_case": self.caption_case_var.get() if hasattr(self, 'caption_case_var') else "none",
                "caption_text_color": list(globals()['CAPTION_TEXT_COLOR']),
                "caption_stroke_color": list(globals()['CAPTION_STROKE_COLOR']),
                "caption_stroke_width": self.stroke_width_var.get(),
                "caption_y_offset": self.caption_y_offset_var.get(),
                "caption_font_size": self.caption_font_size_var.get() if hasattr(self, 'caption_font_size_var') else globals().get('CAPTION_FONT_SIZE', 56),
                
                # Video effects
                "effect_sharpness": self.effect_sharpness_var.get(),
                "effect_sharpness_intensity": self.effect_sharpness_intensity_var.get(),
                "effect_saturation": self.effect_saturation_var.get(),
                "effect_saturation_intensity": self.effect_saturation_intensity_var.get(),
                "effect_contrast": self.effect_contrast_var.get(),
                "effect_contrast_intensity": self.effect_contrast_intensity_var.get(),
                "effect_brightness": self.effect_brightness_var.get(),
                "effect_brightness_intensity": self.effect_brightness_intensity_var.get(),
                "effect_vintage": self.effect_vintage_var.get(),
                "effect_vintage_intensity": self.effect_vintage_intensity_var.get(),
                
                # Blur overlay
                "blur_overlay_enabled": self.blur_overlay_enabled_var.get(),
                "blur_overlay_x": self.blur_overlay_x_var.get(),
                "blur_overlay_y": self.blur_overlay_y_var.get(),
                "blur_overlay_w": self.blur_overlay_w_var.get(),
                "blur_overlay_h": self.blur_overlay_h_var.get(),
                "blur_overlay_intensity": self.blur_overlay_intensity_var.get(),
                
                # Background effects
                "blur_radius": globals().get('STATIC_BG_BLUR_RADIUS', 25),
                "bg_scale_extra": globals().get('BG_SCALE_EXTRA', 1.08),
                "dim_factor": globals().get('DIM_FACTOR', 1.0),
                
                # Lock states (scroll protection)
                "lock_states": {k: v.get() for k, v in self._lock_vars.items()},
            }
            
            # Save to file
            preset_file = Path.home() / ".tiktok_preset.json"
            with open(preset_file, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2)
            
            self.log_widget.config(state="normal")
            self.log_widget.insert(tk.END, f"\n💾 Settings preset saved to: {preset_file}\n")
            self.log_widget.config(state="disabled")
            self.log_widget.see(tk.END)
            
            messagebox.showinfo("Preset Saved", f"All settings saved successfully!\n\nLocation: {preset_file}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save preset: {e}")

    def load_preset(self):
        """Load settings from preset file and apply them to the UI."""
        preset_file = Path.home() / ".tiktok_preset.json"
        
        if not preset_file.exists():
            messagebox.showwarning("No Preset", f"No preset file found at:\n{preset_file}\n\nSave a preset first!")
            return
        
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            
            # Apply file paths (optional)
            if preset_data.get("video_path"):
                self.video_var.set(preset_data["video_path"])
            if preset_data.get("voice_path"):
                self.voice_var.set(preset_data["voice_path"])
            if preset_data.get("music_path"):
                self.music_var.set(preset_data["music_path"])
            if preset_data.get("output_path"):
                self.output_var.set(preset_data["output_path"])
            
            # Apply video settings
            self.mirror_video_var.set(preset_data.get("mirror_video", False))
            self.use_4k_var.set(preset_data.get("use_4k", False))
            self.top_percent_var.set(preset_data.get("top_percent", CROP_TOP_RATIO*100))
            self.bottom_percent_var.set(preset_data.get("bottom_percent", CROP_BOTTOM_RATIO*100))
            self.top_label.config(text=f"{self.top_percent_var.get():.1f}%")
            self.bottom_label.config(text=f"{self.bottom_percent_var.get():.1f}%")
            # Auto-enable custom crop if preset has non-default crop values
            top_val = preset_data.get("top_percent", CROP_TOP_RATIO*100)
            bottom_val = preset_data.get("bottom_percent", CROP_BOTTOM_RATIO*100)
            if abs(top_val - CROP_TOP_RATIO*100) > 0.1 or abs(bottom_val - CROP_BOTTOM_RATIO*100) > 0.1:
                self.use_custom_crop_var.set(True)
            else:
                self.use_custom_crop_var.set(preset_data.get("use_custom_crop", False))
            
            # Apply audio settings
            self.voice_gain_var.set(preset_data.get("voice_gain", VOICE_GAIN))
            self.music_gain_var.set(preset_data.get("music_gain", MUSIC_GAIN))
            # var.set() doesn't trigger Scale callbacks, so sync globals + dB labels manually
            self.on_voice_gain_changed(str(self.voice_gain_var.get()))
            self.on_music_gain_changed(str(self.music_gain_var.get()))
            
            # Apply translation/TTS settings
            self.translation_enabled_var.set(preset_data.get("translation_enabled", TRANSLATION_ENABLED))
            self.target_language_var.set(preset_data.get("target_language", TARGET_LANGUAGE))
            if hasattr(self, 'translation_prompt_var'):
                saved_prompt = preset_data.get("translation_custom_prompt", "")
                self.translation_prompt_var.set(saved_prompt)
                globals()['TRANSLATION_CUSTOM_PROMPT'] = saved_prompt
            if hasattr(self, 'openai_key_var'):
                saved_key = preset_data.get("openai_api_key", "")
                if saved_key:
                    self.openai_key_var.set(saved_key)
                    globals()['OPENAI_API_KEY'] = saved_key
            self.use_ai_voice_var.set(preset_data.get("use_ai_voice", USE_AI_VOICE_REPLACEMENT))
            self.tts_language_var.set(preset_data.get("tts_language", TTS_LANGUAGE))
            # Rebuild dropdown for language first, then restore selected voice + sync global
            self.update_voice_dropdown(preset_data.get("tts_language", TTS_LANGUAGE))
            saved_voice = preset_data.get("tts_voice", 'Auto (Default)')
            self.tts_voice_var.set(saved_voice)
            globals()['TTS_VOICE_ID'] = self.voice_id_map.get(saved_voice, 'auto')
            if hasattr(self, 'tts_fallback_voice_var'):
                saved_fb = preset_data.get("tts_fallback_voice", 'Auto (Default)')
                self.tts_fallback_voice_var.set(saved_fb)
                globals()['TTS_FALLBACK_VOICE_ID'] = self.voice_id_map.get(saved_fb, 'auto')
            self.silence_threshold_var.set(preset_data.get("silence_threshold", 300))
            if hasattr(self, 'tts_speed_var'):
                self.tts_speed_var.set(preset_data.get("tts_speed", TTS_SPEED))
                self.on_tts_speed_changed(str(self.tts_speed_var.get()))
            
            # Apply caption settings
            self.words_per_caption_var.set(preset_data.get("words_per_caption", 2))
            if hasattr(self, 'caption_case_var'):
                _cc = preset_data.get("caption_case", "none")
                self.caption_case_var.set(_cc)
                globals()['CAPTION_CASE'] = _cc
                self._set_caption_case(_cc, force=True)
            if "caption_text_color" in preset_data:
                globals()['CAPTION_TEXT_COLOR'] = tuple(preset_data["caption_text_color"])
            if "caption_stroke_color" in preset_data:
                globals()['CAPTION_STROKE_COLOR'] = tuple(preset_data["caption_stroke_color"])
            self.stroke_width_var.set(preset_data.get("caption_stroke_width", max(1, int(CAPTION_FONT_SIZE * 0.05))))
            self.caption_y_offset_var.set(preset_data.get("caption_y_offset", 0))
            globals()['CAPTION_Y_OFFSET'] = preset_data.get("caption_y_offset", 0)
            # Load font size (apply block 1)
            if hasattr(self, 'caption_font_size_var'):
                font_size = preset_data.get("caption_font_size", 56)
                self.caption_font_size_var.set(font_size)
                globals()["CAPTION_FONT_SIZE"] = font_size
                if hasattr(self, 'caption_font_size_label'):
                    self.caption_font_size_label.config(text=f"{font_size}px")
            
            # Apply video effects
            self.effect_sharpness_var.set(preset_data.get("effect_sharpness", False))
            self.effect_sharpness_intensity_var.set(preset_data.get("effect_sharpness_intensity", 1.5))
            self.effect_saturation_var.set(preset_data.get("effect_saturation", False))
            self.effect_saturation_intensity_var.set(preset_data.get("effect_saturation_intensity", 1.3))
            self.effect_contrast_var.set(preset_data.get("effect_contrast", False))
            self.effect_contrast_intensity_var.set(preset_data.get("effect_contrast_intensity", 1.2))
            self.effect_brightness_var.set(preset_data.get("effect_brightness", False))
            self.effect_brightness_intensity_var.set(preset_data.get("effect_brightness_intensity", 1.15))
            self.effect_vintage_var.set(preset_data.get("effect_vintage", False))
            self.effect_vintage_intensity_var.set(preset_data.get("effect_vintage_intensity", 0.3))
            
            # Apply blur overlay settings
            self.blur_overlay_enabled_var.set(preset_data.get("blur_overlay_enabled", False))
            self.blur_overlay_x_var.set(preset_data.get("blur_overlay_x", 10.0))
            self.blur_overlay_y_var.set(preset_data.get("blur_overlay_y", 10.0))
            self.blur_overlay_w_var.set(preset_data.get("blur_overlay_w", 20.0))
            self.blur_overlay_h_var.set(preset_data.get("blur_overlay_h", 15.0))
            self.blur_overlay_intensity_var.set(preset_data.get("blur_overlay_intensity", 20))
            
            # Apply background effects
            globals()['STATIC_BG_BLUR_RADIUS'] = preset_data.get("blur_radius", 25)
            globals()['BG_SCALE_EXTRA'] = preset_data.get("bg_scale_extra", 1.08)
            # DIM_FACTOR: no longer loaded from presets (always use global default 1.0)
            
            # Restore lock states
            saved_locks = preset_data.get("lock_states", {})
            for lock_key, locked in saved_locks.items():
                if lock_key in self._lock_vars:
                    self._lock_vars[lock_key].set(locked)
                    self._apply_lock(lock_key, locked)
            
            # Update UI elements that show values
            self._update_color_canvases()
            self.update_mini_preview_immediate()
            
            self.log_widget.config(state="normal")
            self.log_widget.insert(tk.END, f"\n📂 Settings preset loaded from: {preset_file}\n")
            self.log_widget.config(state="disabled")
            self.log_widget.see(tk.END)
            
            messagebox.showinfo("Preset Loaded", "All settings loaded and applied successfully!")
        except Exception as e:
            messagebox.showerror("Load Failed", f"Could not load preset: {e}")

    def load_preset_silent(self):
        """Load settings from preset file silently (no popup messages) - used for auto-load on startup."""
        preset_file = Path.home() / ".tiktok_preset.json"
        
        if not preset_file.exists():
            return
        
        try:
            with open(preset_file, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            
            # Apply file paths (optional)
            if preset_data.get("video_path"):
                self.video_var.set(preset_data["video_path"])
            if preset_data.get("voice_path"):
                self.voice_var.set(preset_data["voice_path"])
            if preset_data.get("music_path"):
                self.music_var.set(preset_data["music_path"])
            if preset_data.get("output_path"):
                self.output_var.set(preset_data["output_path"])
            
            # Apply video settings
            self.mirror_video_var.set(preset_data.get("mirror_video", False))
            self.use_4k_var.set(preset_data.get("use_4k", False))
            self.top_percent_var.set(preset_data.get("top_percent", CROP_TOP_RATIO*100))
            self.bottom_percent_var.set(preset_data.get("bottom_percent", CROP_BOTTOM_RATIO*100))
            self.top_label.config(text=f"{self.top_percent_var.get():.1f}%")
            self.bottom_label.config(text=f"{self.bottom_percent_var.get():.1f}%")
            # Auto-enable custom crop if preset has non-default crop values
            top_val = preset_data.get("top_percent", CROP_TOP_RATIO*100)
            bottom_val = preset_data.get("bottom_percent", CROP_BOTTOM_RATIO*100)
            if abs(top_val - CROP_TOP_RATIO*100) > 0.1 or abs(bottom_val - CROP_BOTTOM_RATIO*100) > 0.1:
                self.use_custom_crop_var.set(True)
            else:
                self.use_custom_crop_var.set(preset_data.get("use_custom_crop", False))
            
            # Apply audio settings
            self.voice_gain_var.set(preset_data.get("voice_gain", VOICE_GAIN))
            self.music_gain_var.set(preset_data.get("music_gain", MUSIC_GAIN))
            # var.set() doesn't trigger Scale callbacks, so sync globals + dB labels manually
            self.on_voice_gain_changed(str(self.voice_gain_var.get()))
            self.on_music_gain_changed(str(self.music_gain_var.get()))
            
            # Apply translation/TTS settings
            self.translation_enabled_var.set(preset_data.get("translation_enabled", TRANSLATION_ENABLED))
            self.target_language_var.set(preset_data.get("target_language", TARGET_LANGUAGE))
            if hasattr(self, 'translation_prompt_var'):
                saved_prompt = preset_data.get("translation_custom_prompt", "")
                self.translation_prompt_var.set(saved_prompt)
                globals()['TRANSLATION_CUSTOM_PROMPT'] = saved_prompt
            if hasattr(self, 'openai_key_var'):
                saved_key = preset_data.get("openai_api_key", "")
                if saved_key:
                    self.openai_key_var.set(saved_key)
                    globals()['OPENAI_API_KEY'] = saved_key
            self.use_ai_voice_var.set(preset_data.get("use_ai_voice", USE_AI_VOICE_REPLACEMENT))
            self.tts_language_var.set(preset_data.get("tts_language", TTS_LANGUAGE))
            # Rebuild dropdown for language first, then restore selected voice + sync global
            self.update_voice_dropdown(preset_data.get("tts_language", TTS_LANGUAGE))
            saved_voice = preset_data.get("tts_voice", 'Auto (Default)')
            self.tts_voice_var.set(saved_voice)
            globals()['TTS_VOICE_ID'] = self.voice_id_map.get(saved_voice, 'auto')
            if hasattr(self, 'tts_fallback_voice_var'):
                saved_fb = preset_data.get("tts_fallback_voice", 'Auto (Default)')
                self.tts_fallback_voice_var.set(saved_fb)
                globals()['TTS_FALLBACK_VOICE_ID'] = self.voice_id_map.get(saved_fb, 'auto')
            self.silence_threshold_var.set(preset_data.get("silence_threshold", 300))
            if hasattr(self, 'tts_speed_var'):
                self.tts_speed_var.set(preset_data.get("tts_speed", TTS_SPEED))
                self.on_tts_speed_changed(str(self.tts_speed_var.get()))
            
            # Apply caption settings
            self.words_per_caption_var.set(preset_data.get("words_per_caption", 2))
            if hasattr(self, 'caption_case_var'):
                _cc = preset_data.get("caption_case", "none")
                self.caption_case_var.set(_cc)
                globals()['CAPTION_CASE'] = _cc
                self._set_caption_case(_cc, force=True)
            if "caption_text_color" in preset_data:
                globals()['CAPTION_TEXT_COLOR'] = tuple(preset_data["caption_text_color"])
            if "caption_stroke_color" in preset_data:
                globals()['CAPTION_STROKE_COLOR'] = tuple(preset_data["caption_stroke_color"])
            self.stroke_width_var.set(preset_data.get("caption_stroke_width", max(1, int(CAPTION_FONT_SIZE * 0.05))))
            self.caption_y_offset_var.set(preset_data.get("caption_y_offset", 0))
            globals()['CAPTION_Y_OFFSET'] = preset_data.get("caption_y_offset", 0)
            # Load font size (apply block 2)
            if hasattr(self, 'caption_font_size_var'):
                font_size = preset_data.get("caption_font_size", 56)
                self.caption_font_size_var.set(font_size)
                globals()["CAPTION_FONT_SIZE"] = font_size
                if hasattr(self, 'caption_font_size_label'):
                    self.caption_font_size_label.config(text=f"{font_size}px")
            
            # Apply video effects
            self.effect_sharpness_var.set(preset_data.get("effect_sharpness", False))
            self.effect_sharpness_intensity_var.set(preset_data.get("effect_sharpness_intensity", 1.5))
            self.effect_saturation_var.set(preset_data.get("effect_saturation", False))
            self.effect_saturation_intensity_var.set(preset_data.get("effect_saturation_intensity", 1.3))
            self.effect_contrast_var.set(preset_data.get("effect_contrast", False))
            self.effect_contrast_intensity_var.set(preset_data.get("effect_contrast_intensity", 1.2))
            self.effect_brightness_var.set(preset_data.get("effect_brightness", False))
            self.effect_brightness_intensity_var.set(preset_data.get("effect_brightness_intensity", 1.15))
            self.effect_vintage_var.set(preset_data.get("effect_vintage", False))
            self.effect_vintage_intensity_var.set(preset_data.get("effect_vintage_intensity", 0.3))
            
            # Apply blur overlay settings
            self.blur_overlay_enabled_var.set(preset_data.get("blur_overlay_enabled", False))
            self.blur_overlay_x_var.set(preset_data.get("blur_overlay_x", 10.0))
            self.blur_overlay_y_var.set(preset_data.get("blur_overlay_y", 10.0))
            self.blur_overlay_w_var.set(preset_data.get("blur_overlay_w", 20.0))
            self.blur_overlay_h_var.set(preset_data.get("blur_overlay_h", 15.0))
            self.blur_overlay_intensity_var.set(preset_data.get("blur_overlay_intensity", 20))
            
            # Apply background effects
            globals()['STATIC_BG_BLUR_RADIUS'] = preset_data.get("blur_radius", 25)
            globals()['BG_SCALE_EXTRA'] = preset_data.get("bg_scale_extra", 1.08)
            # DIM_FACTOR: no longer loaded from presets (always use global default 1.0)
            
            # Restore lock states
            saved_locks = preset_data.get("lock_states", {})
            for lock_key, locked in saved_locks.items():
                if lock_key in self._lock_vars:
                    self._lock_vars[lock_key].set(locked)
                    self._apply_lock(lock_key, locked)
            
            # Update UI elements that show values
            self._update_color_canvases()
            
            self.log_widget.config(state="normal")
            self.log_widget.insert(tk.END, f"\n📂 Auto-loaded preset from: {preset_file}\n")
            self.log_widget.config(state="disabled")
            self.log_widget.see(tk.END)
        except Exception as e:
            # Silently fail on auto-load
            self.log_widget.config(state="normal")
            self.log_widget.insert(tk.END, f"\n⚠️ Could not auto-load preset: {e}\n")
            self.log_widget.config(state="disabled")
            self.log_widget.see(tk.END)

    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        result = messagebox.askyesno("Reset to Defaults", 
                                      "This will reset ALL settings to their default values.\n\nAre you sure?")
        if not result:
            return
        
        try:
            # Reset file paths
            self.video_var.set("")
            self.voice_var.set("")
            self.music_var.set("")
            self.output_var.set("final_tiktok.mp4")
            
            # Reset video settings
            self.mirror_video_var.set(False)
            self.use_4k_var.set(False)
            self.use_custom_crop_var.set(False)
            self.top_percent_var.set(CROP_TOP_RATIO*100)
            self.bottom_percent_var.set(CROP_BOTTOM_RATIO*100)
            self.top_label.config(text=f"{self.top_percent_var.get():.1f}%")
            self.bottom_label.config(text=f"{self.bottom_percent_var.get():.1f}%")
            
            # Reset audio settings
            self.voice_gain_var.set(VOICE_GAIN)
            self.music_gain_var.set(MUSIC_GAIN)
            # var.set() doesn't trigger Scale callbacks, so sync globals + dB labels manually
            self.on_voice_gain_changed(str(self.voice_gain_var.get()))
            self.on_music_gain_changed(str(self.music_gain_var.get()))
            
            # Reset translation/TTS settings
            self.translation_enabled_var.set(TRANSLATION_ENABLED)
            self.target_language_var.set(TARGET_LANGUAGE)
            self.use_ai_voice_var.set(USE_AI_VOICE_REPLACEMENT)
            self.tts_language_var.set(TTS_LANGUAGE)
            self.tts_voice_var.set('Auto (Default)')
            self.silence_threshold_var.set(300)
            if hasattr(self, 'tts_speed_var'):
                self.tts_speed_var.set(TTS_SPEED)
                self.on_tts_speed_changed(str(TTS_SPEED))
            
            # Reset caption settings
            self.words_per_caption_var.set(2)
            if hasattr(self, 'caption_case_var'):
                self._set_caption_case("none", force=True)
            globals()['CAPTION_TEXT_COLOR'] = (255, 255, 255, 255)
            globals()['CAPTION_STROKE_COLOR'] = (0, 0, 0, 150)
            self.stroke_width_var.set(max(1, int(CAPTION_FONT_SIZE * 0.05)))
            self.caption_y_offset_var.set(0)
            globals()['CAPTION_Y_OFFSET'] = 0
            # Reset font size
            if hasattr(self, 'caption_font_size_var'):
                self.caption_font_size_var.set(56)
                globals()["CAPTION_FONT_SIZE"] = 56
                if hasattr(self, 'caption_font_size_label'):
                    self.caption_font_size_label.config(text="56px")
            
            # Reset video effects
            self.effect_sharpness_var.set(False)
            self.effect_sharpness_intensity_var.set(1.5)
            self.effect_saturation_var.set(False)
            self.effect_saturation_intensity_var.set(1.3)
            self.effect_contrast_var.set(False)
            self.effect_contrast_intensity_var.set(1.2)
            self.effect_brightness_var.set(False)
            self.effect_brightness_intensity_var.set(1.15)
            self.effect_vintage_var.set(False)
            self.effect_vintage_intensity_var.set(0.3)
            
            # Reset blur overlay
            self.blur_overlay_enabled_var.set(False)
            self.blur_overlay_x_var.set(10.0)
            self.blur_overlay_y_var.set(10.0)
            self.blur_overlay_w_var.set(20.0)
            self.blur_overlay_h_var.set(15.0)
            self.blur_overlay_intensity_var.set(20)
            
            # Reset background effects
            globals()['STATIC_BG_BLUR_RADIUS'] = 25
            globals()['BG_SCALE_EXTRA'] = 1.08
            globals()['DIM_FACTOR'] = 1.0
            
            # Unlock all locked widgets
            for lock_key in list(self._lock_vars.keys()):
                self._lock_vars[lock_key].set(False)
                self._apply_lock(lock_key, False)
            
            # Update UI
            self._update_color_canvases()
            self.update_mini_preview_immediate()
            
            self.log_widget.config(state="normal")
            self.log_widget.insert(tk.END, "\n🔄 All settings reset to defaults\n")
            self.log_widget.config(state="disabled")
            self.log_widget.see(tk.END)
            
            messagebox.showinfo("Reset Complete", "All settings have been reset to their default values.")
        except Exception as e:
            messagebox.showerror("Reset Failed", f"Could not reset settings: {e}")

    def poll_queue(self):
        try:
            processed = 0
            log_batch = []
            while processed < 80:
                msg = self.q.get_nowait()
                processed += 1
                if msg == "[SINGLE_DONE]":
                    if log_batch:
                        self.log_widget.config(state="normal")
                        self.log_widget.insert(tk.END, "".join(log_batch))
                        self.log_widget.config(state="disabled")
                        log_batch = []
                    self.run_single_btn.config(state="normal")
                    self.run_queue_btn.config(state="normal")
                    self.log_widget.config(state="normal")
                    self.log_widget.insert(tk.END, "\n[Single job finished]\n")
                    self.log_widget.config(state="disabled")
                    self.log_widget.see(tk.END)
                    continue
                if msg == "[QUEUE_DONE]":
                    if log_batch:
                        self.log_widget.config(state="normal")
                        self.log_widget.insert(tk.END, "".join(log_batch))
                        self.log_widget.config(state="disabled")
                        log_batch = []
                    self.run_queue_btn.config(state="normal")
                    self.run_single_btn.config(state="normal")
                    self.log_widget.config(state="normal")
                    self.log_widget.insert(tk.END, "\n✔ ALL JOBS FINISHED\n")
                    self.log_widget.config(state="disabled")
                    self.log_widget.see(tk.END)
                    continue
                line = str(msg)
                if not line.endswith("\n"):
                    line += "\n"
                log_batch.append(line)
            # Flush remaining log lines in one batch (3 widget ops instead of 4 per message)
            if log_batch:
                self.log_widget.config(state="normal")
                self.log_widget.insert(tk.END, "".join(log_batch))
                self.log_widget.config(state="disabled")
                self.log_widget.see(tk.END)
        except queue.Empty:
            if log_batch:
                self.log_widget.config(state="normal")
                self.log_widget.insert(tk.END, "".join(log_batch))
                self.log_widget.config(state="disabled")
                self.log_widget.see(tk.END)
        self.root.after(200, self.poll_queue)
        try:
            self._update_color_canvases()
        except Exception:
            pass



# ----------------- entrypoint -----------------
def main():
    root = tk.Tk()
    app = App(root)
    
    # Auto-load preset if it exists
    preset_file = Path.home() / ".tiktok_preset.json"
    if preset_file.exists():
        try:
            # Use root.after to load preset after UI is fully initialized
            root.after(100, app.load_preset_silent)
        except Exception:
            pass  # Silently ignore errors during auto-load
    
    # Catch any remaining Scale/Combobox/Spinbox widgets not explicitly registered
    root.after(200, app._setup_scroll_protection)
    
    # Verify GenAI Pro API key on startup (runs in background thread, logs result)
    root.after(500, app._check_genaipro_api)
    
    root.mainloop()

if __name__ == "__main__":
    main()
# Could not insert color UI automatically; please add manually.
