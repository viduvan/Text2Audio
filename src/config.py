"""Configuration loader for Text2Audio pipeline."""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict


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


# ── Rewriter Config ──

@dataclass
class RewriterConfig:
    """Configuration for text rewriter (N1)."""
    enabled: bool = True
    provider: str = "gemini"            # "gemini" or "ollama"
    style: str = "narrator"             # "narrator", "summary", "creative"
    batch_size: int = 5
    # Gemini settings
    gemini_model: str = "gemini-2.0-flash"  # API key read from env GEMINI_API_KEY
    # Ollama settings
    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_dict(cls, d: dict) -> "RewriterConfig":
        gemini = d.get("gemini", {})
        ollama = d.get("ollama", {})
        return cls(
            enabled=d.get("enabled", True),
            provider=d.get("provider", "gemini"),
            style=d.get("style", "narrator"),
            batch_size=d.get("batch_size", 5),
            gemini_model=gemini.get("model", "gemini-2.0-flash"),
            ollama_model=ollama.get("model", "qwen2.5:7b"),
            ollama_base_url=ollama.get("base_url", "http://localhost:11434"),
        )


# ── Watermark Config ──

@dataclass
class WatermarkConfig:
    """Configuration for audio watermark (anti-theft)."""
    enabled: bool = True
    channel_name: str = "Kênh Truyện Audio"
    interval_scenes: int = 25           # Insert watermark every N scenes (~15 min)
    templates: Dict[str, str] = field(default_factory=lambda: {
        "intro": "Bạn đang nghe {channel_name}, kênh audio truyện chất lượng cao. Nếu thích, hãy đăng ký kênh nhé.",
        "middle": "Nội dung thuộc bản quyền kênh {channel_name}.",
        "part_end": "{channel_name}. Cảm ơn bạn đã theo dõi, phần tiếp theo sẽ bắt đầu ngay.",
        "outro": "Cảm ơn bạn đã nghe hết truyện trên kênh {channel_name}. Đăng ký kênh để không bỏ lỡ truyện hay tiếp theo nhé!",
    })

    @classmethod
    def from_dict(cls, d: dict) -> "WatermarkConfig":
        return cls(
            enabled=d.get("enabled", True),
            channel_name=d.get("channel_name", "Kênh Truyện Audio"),
            interval_scenes=d.get("interval_scenes", 25),
            templates=d.get("templates", cls.__dataclass_fields__["templates"].default_factory()),
        )

    def render_template(self, template_key: str) -> str:
        """Render a template with channel_name substitution."""
        template = self.templates.get(template_key, "")
        return template.format(channel_name=self.channel_name)


# ── TTS Config ──

@dataclass
class TTSConfig:
    engine: str = "vieneu"
    voice: str = "vi-VN-HoaiMyNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"
    output_format: str = "wav"
    # VieNeu-TTS specific
    vieneu_model_id: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
    vieneu_mode: str = "turbo"
    vieneu_emotion: str = "natural"
    vieneu_voice_id: Optional[str] = "Ngọc Lan"
    vieneu_ref_audio: Optional[str] = None
    vieneu_ref_text: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TTSConfig":
        vieneu = d.get("vieneu", {})
        return cls(
            engine=d.get("engine", "vieneu"),
            voice=d.get("voice", "vi-VN-HoaiMyNeural"),
            rate=d.get("rate", "+0%"),
            pitch=d.get("pitch", "+0Hz"),
            volume=d.get("volume", "+0%"),
            output_format=d.get("output_format", "wav"),
            vieneu_model_id=vieneu.get("model_id", "pnnbao-ump/VieNeu-TTS-v3-Turbo"),
            vieneu_mode=vieneu.get("mode", "turbo"),
            vieneu_emotion=vieneu.get("emotion", "natural"),
            vieneu_voice_id=vieneu.get("voice_id", "Ngọc Lan"),
            vieneu_ref_audio=vieneu.get("ref_audio"),
            vieneu_ref_text=vieneu.get("ref_text"),
        )


# ── Image Config ──

@dataclass
class ImageConfig:
    model_id: str = "cagliostrolab/animagine-xl-3.1"
    width: int = 1280
    height: int = 720
    num_inference_steps: int = 8
    guidance_scale: float = 2.0
    style_prefix: str = "anime style, detailed anime illustration, vibrant colors, masterpiece, best quality"
    negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, blurry"
    seed: int = -1
    enable_cpu_offload: bool = True
    # LCM LoRA settings
    use_lcm: bool = True
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdxl"

    @classmethod
    def from_dict(cls, d: dict) -> "ImageConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Video Config ──

@dataclass
class VideoConfig:
    ken_burns: dict = field(default_factory=lambda: {
        "effect": "random", "zoom_ratio": 1.2, "fps": 24
    })
    use_ffmpeg: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "VideoConfig":
        return cls(
            ken_burns=d.get("ken_burns", {}),
            use_ffmpeg=d.get("use_ffmpeg", True),
        )


# ── Merger Config ──

@dataclass
class MergerConfig:
    output_width: int = 1280
    output_height: int = 720
    fps: int = 24
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    transition_duration: float = 0.5
    transition_type: str = "crossfade"
    parts_per_video: int = 10

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
            parts_per_video=d.get("parts_per_video", 10),
        )


# ── Post-process Config ──

@dataclass
class PostProcessConfig:
    """Configuration for post-processing (N8)."""
    burn_subtitles: bool = True
    add_intro: bool = False
    add_outro: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "PostProcessConfig":
        return cls(
            burn_subtitles=d.get("burn_subtitles", True),
            add_intro=d.get("add_intro", False),
            add_outro=d.get("add_outro", False),
        )


# ── Aggregated Pipeline Config ──

@dataclass
class PipelineConfig:
    """Aggregated pipeline configuration."""
    rewriter: RewriterConfig = field(default_factory=RewriterConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    merger: MergerConfig = field(default_factory=MergerConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)
    output_dir: str = "./output"
    stories_dir: str = "./stories"
    models_dir: str = "./models"

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "PipelineConfig":
        raw = load_config(config_path)
        paths = raw.get("paths", {})
        return cls(
            rewriter=RewriterConfig.from_dict(raw.get("rewriter", {})),
            watermark=WatermarkConfig.from_dict(raw.get("watermark", {})),
            tts=TTSConfig.from_dict(raw.get("tts", {})),
            image=ImageConfig.from_dict(raw.get("image", {})),
            video=VideoConfig.from_dict(raw.get("video", {})),
            merger=MergerConfig.from_dict(raw.get("merger", {})),
            post_process=PostProcessConfig.from_dict(raw.get("post_process", {})),
            output_dir=paths.get("output_dir", "./output"),
            stories_dir=paths.get("stories_dir", "./stories"),
            models_dir=paths.get("models_dir", "./models"),
        )
