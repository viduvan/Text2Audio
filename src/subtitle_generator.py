"""Subtitle Generator - Create SRT subtitles from word timestamps."""
import os
import logging
from typing import List, Optional
from src.tts_engine import WordTimestamp

logger = logging.getLogger(__name__)


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(
    timestamps: List[WordTimestamp],
    output_path: str,
    words_per_line: int = 8,
    max_line_chars: int = 60,
) -> str:
    """Generate SRT subtitle file from word timestamps.

    Groups words into subtitle lines of appropriate length.
    """
    if not timestamps:
        logger.warning("No timestamps provided for subtitle generation")
        return output_path

    # Group words into subtitle lines
    lines = []
    current_words = []
    current_chars = 0

    for ts in timestamps:
        current_words.append(ts)
        current_chars += len(ts.text) + 1  # +1 for space

        if len(current_words) >= words_per_line or current_chars >= max_line_chars:
            lines.append(current_words[:])
            current_words = []
            current_chars = 0

    if current_words:
        lines.append(current_words)

    # Write SRT
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, word_group in enumerate(lines, 1):
            start_time = word_group[0].offset
            end_time = word_group[-1].offset + word_group[-1].duration
            text = " ".join(w.text for w in word_group)

            f.write(f"{i}\n")
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            f.write(f"{text}\n\n")

    logger.info(f"Generated SRT with {len(lines)} entries: {output_path}")
    return output_path


def generate_srt_from_text(
    text: str,
    audio_duration: float,
    output_path: str,
    chars_per_second: float = 5.0,
) -> str:
    """Generate approximate SRT from text and audio duration (no timestamps).

    Uses character count to estimate timing.
    """
    import re
    # Split into sentences
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    total_chars = sum(len(s) for s in sentences)
    if total_chars == 0:
        return output_path

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        current_time = 0.0
        for i, sentence in enumerate(sentences, 1):
            # Estimate duration proportional to character count
            duration = (len(sentence) / total_chars) * audio_duration
            start_time = current_time
            end_time = current_time + duration

            f.write(f"{i}\n")
            f.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            f.write(f"{sentence}\n\n")

            current_time = end_time

    logger.info(f"Generated approximate SRT: {output_path}")
    return output_path
