"""Configuration loader for Text2Audio pipeline."""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(config: dict):
    """Create output directories if they don't exist."""
    output_dir = config.get("paths", {}).get("output_dir", "./output")
    subdirs = ["audio", "images", "videos", "scenes", "final", "subtitles"]
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    # Also create stories dir
    stories_dir = config.get("paths", {}).get("stories_dir", "./stories")
    os.makedirs(stories_dir, exist_ok=True)
    # Models dir
    models_dir = config.get("paths", {}).get("models_dir", "./models")
    os.makedirs(models_dir, exist_ok=True)


@dataclass
class TTSConfig:
    engine: str = "vieneu"
    voice: str = "vi-VN-HoaiMyNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    output_format: str = "mp3"
    # VieNeu-TTS specific
    vieneu_model_id: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo"  # HF Hub model ID
    vieneu_mode: str = "turbo"       # v3 chỉ hỗ trợ turbo
    vieneu_emotion: str = "natural"  # "natural" hoặc "storytelling"
    vieneu_voice_id: Optional[str] = "Ngọc Lan"  # v3 default voice
    vieneu_ref_audio: Optional[str] = None
    vieneu_ref_text: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TTSConfig":
        # Extract VieNeu nested config
        vieneu = d.get("vieneu", {})
        return cls(
            engine=d.get("engine", "vieneu"),
            voice=d.get("voice", "vi-VN-HoaiMyNeural"),
            rate=d.get("rate", "+0%"),
            pitch=d.get("pitch", "+0Hz"),
            volume=d.get("volume", "+0%"),
            output_format=d.get("output_format", "mp3"),
            vieneu_model_id=vieneu.get("model_id", "pnnbao-ump/VieNeu-TTS-v3-Turbo"),
            vieneu_mode=vieneu.get("mode", "turbo"),
            vieneu_emotion=vieneu.get("emotion", "natural"),
            vieneu_voice_id=vieneu.get("voice_id", "Ngọc Lan"),
            vieneu_ref_audio=vieneu.get("ref_audio"),
            vieneu_ref_text=vieneu.get("ref_text"),
        )


@dataclass
class ImageConfig:
    model_id: str = "cagliostrolab/animagine-xl-3.1"
    width: int = 1280
    height: int = 720
    num_inference_steps: int = 25
    guidance_scale: float = 7.0
    style_prefix: str = "anime style, detailed anime illustration, vibrant colors, masterpiece, best quality"
    negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, blurry"
    seed: int = -1
    enable_cpu_offload: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ImageConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class VideoConfig:
    ken_burns: dict = field(default_factory=lambda: {
        "effect": "random", "zoom_ratio": 1.2, "fps": 24
    })

    @classmethod
    def from_dict(cls, d: dict) -> "VideoConfig":
        return cls(
            ken_burns=d.get("ken_burns", {}),
        )


@dataclass
class MergerConfig:
    output_width: int = 1280
    output_height: int = 720
    fps: int = 24
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    transition_duration: float = 0.5
    transition_type: str = "crossfade"

    @classmethod
    def from_dict(cls, d: dict) -> "MergerConfig":
        res = d.get("output_resolution", {})
        return cls(
            output_width=res.get("width", 1280),
            output_height=res.get("height", 720),
            fps=d.get("fps", 24),
            video_codec=d.get("video_codec", "libx264"),
            audio_codec=d.get("audio_codec", "aac"),
            transition_duration=d.get("transition_duration", 0.5),
            transition_type=d.get("transition_type", "crossfade"),
        )


@dataclass 
class PipelineConfig:
    """Aggregated pipeline configuration."""
    tts: TTSConfig = field(default_factory=TTSConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    merger: MergerConfig = field(default_factory=MergerConfig)
    output_dir: str = "./output"
    stories_dir: str = "./stories"
    models_dir: str = "./models"

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "PipelineConfig":
        raw = load_config(config_path)
        paths = raw.get("paths", {})
        return cls(
            tts=TTSConfig.from_dict(raw.get("tts", {})),
            image=ImageConfig.from_dict(raw.get("image", {})),
            video=VideoConfig.from_dict(raw.get("video", {})),
            merger=MergerConfig.from_dict(raw.get("merger", {})),
            output_dir=paths.get("output_dir", "./output"),
            stories_dir=paths.get("stories_dir", "./stories"),
            models_dir=paths.get("models_dir", "./models"),
        )
