"""TTS Engine - Convert Vietnamese text to speech audio.

Supports two engines:
- edge-tts: Microsoft Edge cloud TTS (2 voices, free, needs internet)
- vieneu: VieNeu-TTS v3 Turbo from HuggingFace Hub (10 voices, voice cloning, offline after first download)
"""
import os
import asyncio
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """Word-level timestamp for subtitle sync."""
    text: str
    offset: float  # seconds
    duration: float  # seconds


# ─────────────────────────────────────────────
# Edge-TTS Engine (Cloud)
# ─────────────────────────────────────────────

class EdgeTTSEngine:
    """Text-to-Speech engine using Microsoft Edge TTS (free, high quality)."""

    VOICES = {
        "female": "vi-VN-HoaiMyNeural",
        "male": "vi-VN-NamMinhNeural",
    }

    def __init__(self, voice: str = "vi-VN-HoaiMyNeural",
                 rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%",
                 **kwargs):
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
            size = os.path.getsize(audio_path)
            # Rough MP3 estimate: ~16KB per second at 128kbps
            return size / 16000


# ─────────────────────────────────────────────
# VieNeu-TTS Engine (Local)
# ─────────────────────────────────────────────

class VieNeuTTSEngine:
    """Text-to-Speech engine using VieNeu-TTS (local, multi-voice, voice cloning).

    Features:
    - 7+ preset Vietnamese voices (male/female, Northern/Southern)
    - Voice cloning from 3-5 seconds of reference audio
    - Storytelling & Natural emotion modes
    - Offline operation (no internet needed)
    - 24kHz audio quality
    """

    def __init__(
        self,
        model_id: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        mode: str = "turbo",
        emotion: str = "natural",
        voice_id: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs,
    ):
        self.model_id = model_id  # HuggingFace Hub model ID
        self.mode = mode          # "turbo" (v3 only supports turbo)
        self.emotion = emotion    # "natural" or "storytelling"
        self.voice_id = voice_id  # Preset voice name (e.g. "Ngọc Lan", "Xuân Vĩnh")
        self.ref_audio = ref_audio  # Path to reference audio for cloning
        self.ref_text = ref_text    # Reference text for cloning (unused in v3)
        self.rate = kwargs.get("rate", "+0%")
        self.tts = None
        self._voice_data = None

    def _init(self):
        """Lazy-load VieNeu-TTS model from HuggingFace Hub."""
        if self.tts is not None:
            return

        try:
            from vieneu import Vieneu
        except ImportError:
            raise ImportError(
                "VieNeu-TTS not installed. Install with: pip install vieneu"
            )

        logger.info(f"Loading VieNeu-TTS v3 Turbo from HF Hub: {self.model_id}")
        self.tts = Vieneu(model_id=self.model_id)

        # Pre-load voice data
        if self.ref_audio and os.path.exists(self.ref_audio):
            logger.info(f"Encoding reference voice: {self.ref_audio}")
            self._voice_data = self.tts.encode_reference(self.ref_audio)
        elif self.voice_id:
            logger.info(f"Using preset voice: {self.voice_id}")
            try:
                self._voice_data = self.voice_id  # v3: pass name string directly
            except Exception as e:
                logger.warning(f"Could not set preset voice '{self.voice_id}': {e}")
                self._voice_data = None

        logger.info("VieNeu-TTS v3 Turbo loaded successfully")

    def list_voices(self) -> List[Tuple[str, str]]:
        """List available preset voices.

        Returns:
            List of (description, voice_id) tuples.
        """
        self._init()
        try:
            return self.tts.list_preset_voices()
        except Exception:
            # Fallback: return known v3 Turbo voices
            return [
                ("Ngọc Lan (Nữ, Nhẹ nhàng)", "Ngọc Lan"),
                ("Ngọc Linh (Nữ, Trong sáng)", "Ngọc Linh"),
                ("Trúc Ly (Nữ, Trẻ trung)", "Trúc Ly"),
                ("Mỹ Duyên (Nữ, Mượt mà)", "Mỹ Duyên"),
                ("Xuân Vĩnh (Nam, Năng động)", "Xuân Vĩnh"),
                ("Thái Sơn (Nam, Quyết đoán)", "Thái Sơn"),
                ("Gia Bảo (Nam, Mượt mà)", "Gia Bảo"),
                ("Đức Trí (Nam, Rõ ràng)", "Đức Trí"),
                ("Trọng Hữu (Nam, Chính trực)", "Trọng Hữu"),
                ("Bình An (Nam, Nhẹ nhàng)", "Bình An"),
            ]

    def set_voice(self, voice_id: str):
        """Change the active preset voice."""
        self._init()
        self.voice_id = voice_id
        self._voice_data = voice_id  # v3: name passed directly to infer()
        logger.info(f"Switched to voice: {voice_id}")

    def set_ref_audio(self, ref_audio_path: str, ref_text: Optional[str] = None):
        """Set reference audio for voice cloning."""
        self._init()
        self.ref_audio = ref_audio_path
        self.ref_text = ref_text
        if os.path.exists(ref_audio_path):
            self._voice_data = self.tts.encode_reference(ref_audio_path)
            logger.info(f"Encoded reference voice from: {ref_audio_path}")

    def generate(
        self,
        text: str,
        output_path: str,
        collect_timestamps: bool = False,
    ) -> Tuple[str, List[WordTimestamp]]:
        """Generate audio from text using VieNeu-TTS v3 Turbo.

        Args:
            text: Vietnamese text to synthesize
            output_path: Path to save audio file (.wav, 48kHz)
            collect_timestamps: Not supported yet (ignored)

        Returns:
            Tuple of (output_path, timestamps_list)
        """
        self._init()

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # Ensure .wav extension for VieNeu output
        if not output_path.lower().endswith(".wav"):
            wav_path = os.path.splitext(output_path)[0] + ".wav"
        else:
            wav_path = output_path

        # Generate audio — v3 API: voice is passed as a string name
        voice_arg = self._voice_data if self._voice_data else "Ngọc Lan"
        audio = self.tts.infer(text=text, voice=voice_arg)

        # Apply speed adjustment if needed
        if self.rate and self.rate != "+0%":
            try:
                import librosa
                import numpy as np
                rate_val = 1.0 + float(self.rate.strip('%').strip('+')) / 100.0
                if abs(rate_val - 1.0) > 0.01:
                    # Convert to float32 for librosa
                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                        # Normalize if it's 16-bit PCM
                        if np.max(np.abs(audio)) > 1.0:
                            audio = audio / 32768.0
                    audio = librosa.effects.time_stretch(audio, rate=rate_val)
                    logger.info(f"Stretched VieNeu audio by {rate_val}x")
            except Exception as e:
                logger.warning(f"Could not apply speed to VieNeu audio: {e}")

        self.tts.save(audio, wav_path)

        logger.info(f"VieNeu-TTS v3 audio saved (48kHz): {wav_path}")
        return wav_path, []

    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file in seconds."""
        try:
            import soundfile as sf
            data, sr = sf.read(audio_path)
            return len(data) / sr
        except Exception:
            pass

        try:
            from moviepy import AudioFileClip
            clip = AudioFileClip(audio_path)
            duration = clip.duration
            clip.close()
            return duration
        except Exception:
            size = os.path.getsize(audio_path)
            return size / 96000  # WAV 48kHz mono 16-bit ≈ 96KB/s


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def create_tts_engine(engine_type: str = "vieneu", **kwargs):
    """Factory function to create TTS engine.

    Args:
        engine_type: "vieneu" (default, HF Hub) or "edge-tts" (cloud)
        **kwargs: Engine-specific parameters

    Returns:
        TTS engine instance
    """
    if engine_type == "edge-tts":
        return EdgeTTSEngine(**kwargs)
    elif engine_type == "vieneu":
        return VieNeuTTSEngine(**kwargs)
    else:
        raise ValueError(
            f"Unknown TTS engine: {engine_type}. Supported: vieneu, edge-tts"
        )


def list_available_engines() -> List[dict]:
    """List available TTS engines with their descriptions."""
    engines = [
        {
            "id": "vieneu",
            "name": "🇻🇳 VieNeu-TTS v3 Turbo (HuggingFace Hub)",
            "description": "10 giọng Việt 48kHz, clone giọng nói, tải tự động từ HF Hub",
            "voices": "Ngọc Lan, Ngọc Linh, Trúc Ly, Mỹ Duyên, Xuân Vĩnh, Thái Sơn, Gia Bảo, Đức Trí, Trọng Hữu, Bình An",
        },
        {
            "id": "edge-tts",
            "name": "☁️ Edge-TTS (Cloud)",
            "description": "2 giọng Microsoft, cần internet, nhẹ",
            "voices": "HoaiMy (nữ), NamMinh (nam)",
        },
    ]
    return engines
