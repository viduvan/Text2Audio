"""TTS Engine - Convert Vietnamese text to speech audio."""
import os
import asyncio
import logging
import json
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """Word-level timestamp for subtitle sync."""
    text: str
    offset: float  # seconds
    duration: float  # seconds


class EdgeTTSEngine:
    """Text-to-Speech engine using Microsoft Edge TTS (free, high quality)."""

    VOICES = {
        "female": "vi-VN-HoaiMyNeural",
        "male": "vi-VN-NamMinhNeural",
    }

    def __init__(self, voice: str = "vi-VN-HoaiMyNeural",
                 rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume

    async def _generate_async(self, text: str, output_path: str,
                              collect_timestamps: bool = False) -> Tuple[str, List[WordTimestamp]]:
        """Generate audio asynchronously using edge-tts."""
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume,
        )

        timestamps = []
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if collect_timestamps:
            # Collect word boundaries for subtitle sync
            with open(output_path, "wb") as f:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        ts = WordTimestamp(
                            text=chunk["text"],
                            offset=chunk["offset"] / 10_000_000,  # Convert from 100ns to seconds
                            duration=chunk["duration"] / 10_000_000,
                        )
                        timestamps.append(ts)
        else:
            await communicate.save(output_path)

        return output_path, timestamps

    def generate(self, text: str, output_path: str,
                 collect_timestamps: bool = False) -> Tuple[str, List[WordTimestamp]]:
        """Generate audio synchronously."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context (e.g., Gradio)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self._generate_async(text, output_path, collect_timestamps)
                )
                return future.result()
        else:
            return asyncio.run(
                self._generate_async(text, output_path, collect_timestamps)
            )

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file in seconds."""
        try:
            from moviepy import AudioFileClip
            clip = AudioFileClip(audio_path)
            duration = clip.duration
            clip.close()
            return duration
        except Exception:
            # Fallback: estimate from file size (rough)
            import struct
            size = os.path.getsize(audio_path)
            # Rough MP3 estimate: ~16KB per second at 128kbps
            return size / 16000


def create_tts_engine(engine_type: str = "edge-tts", **kwargs) -> EdgeTTSEngine:
    """Factory function to create TTS engine."""
    if engine_type == "edge-tts":
        return EdgeTTSEngine(**kwargs)
    else:
        raise ValueError(f"Unknown TTS engine: {engine_type}. Supported: edge-tts")
