"""Text Processor - Split stories into scenes and generate prompts."""
import re
import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """Represents a single scene in the story."""
    scene_id: int
    text: str  # Vietnamese text for TTS
    image_prompt: str = ""  # English prompt for SDXL
    video_prompt: str = ""  # English prompt for Wan 2.1
    audio_path: Optional[str] = None
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_duration: float = 0.0
    status: str = "pending"  # pending, processing, done, error

    def to_dict(self) -> dict:
        return asdict(self)


# Vietnamese → English keyword mapping for anime scene generation
SCENE_KEYWORDS = {
    "rừng": "forest, trees, nature",
    "biển": "ocean, sea, beach, waves",
    "núi": "mountain, peaks, clouds",
    "sông": "river, water, flowing",
    "hồ": "lake, calm water, reflection",
    "làng": "village, countryside, rural",
    "thành phố": "city, urban, buildings",
    "nhà": "house, home, interior",
    "vườn": "garden, flowers, plants",
    "trường": "school, classroom",
    "chợ": "market, stalls, busy",
    "đêm": "night, moonlight, stars, dark sky",
    "bình minh": "sunrise, dawn, morning light, golden hour",
    "hoàng hôn": "sunset, dusk, evening sky, orange sky",
    "mưa": "rain, rainy, wet, umbrella",
    "tuyết": "snow, winter, cold",
    "hoa": "flowers, blooming, petals",
    "bướm": "butterfly, colorful wings",
    "chiến đấu": "battle, fighting, action",
    "phép thuật": "magic, spell, glowing, mystical",
    "ngọc": "jewel, gem, glowing, precious",
    "cây cổ thụ": "ancient tree, giant tree, mystical",
    "bà lão": "elderly woman, wise, mystical figure",
    "cô bé": "young girl, cute, innocent",
    "chàng trai": "young man, handsome, brave",
    "công chúa": "princess, royal, elegant",
    "rồng": "dragon, mythical, powerful",
    "kiếm": "sword, weapon, warrior",
}

# ─── Image Style Presets ───
# Each style has: prefix (main style tags), quality (boosters), negative_extra (extra negatives)
IMAGE_STYLES = {
    "anime": {
        "label": "🎌 Anime",
        "prefix": "anime style, detailed anime illustration, vibrant colors, masterpiece, best quality",
        "quality": "cinematic lighting, highly detailed, 4k",
        "negative_extra": "",
    },
    "anime_dark": {
        "label": "🌙 Anime Dark / Gothic",
        "prefix": "dark anime style, gothic anime, dramatic shadows, detailed illustration, masterpiece, best quality",
        "quality": "dark atmosphere, moody lighting, highly detailed, 4k",
        "negative_extra": "bright, cheerful, pastel colors",
    },
    "anime_chibi": {
        "label": "🧸 Anime Chibi",
        "prefix": "chibi anime style, cute chibi character, kawaii, super deformed, colorful, masterpiece, best quality",
        "quality": "soft lighting, pastel colors, highly detailed, 4k",
        "negative_extra": "realistic, dark, scary",
    },
    "anime_watercolor": {
        "label": "🎨 Anime Watercolor",
        "prefix": "watercolor anime style, soft watercolor painting, delicate brush strokes, pastel tones, masterpiece, best quality",
        "quality": "dreamy atmosphere, soft lighting, artistic, highly detailed, 4k",
        "negative_extra": "sharp lines, digital art look, photorealistic",
    },
    "fantasy_art": {
        "label": "⚔️ Fantasy Art",
        "prefix": "fantasy art style, epic fantasy illustration, magical, ethereal glow, masterpiece, best quality",
        "quality": "dramatic lighting, volumetric light, highly detailed, 4k",
        "negative_extra": "",
    },
    "realistic_anime": {
        "label": "📸 Semi-Realistic Anime",
        "prefix": "semi-realistic anime style, detailed realistic anime, photorealistic shading, masterpiece, best quality",
        "quality": "ray tracing, cinematic lighting, highly detailed, 4k, 8k",
        "negative_extra": "flat colors, simple, cartoonish",
    },
    "ink_wash": {
        "label": "🖌️ Ink Wash / Thủy Mặc",
        "prefix": "chinese ink wash painting style, sumi-e, traditional asian art, monochrome with subtle color, masterpiece, best quality",
        "quality": "atmospheric, misty, elegant composition, highly detailed",
        "negative_extra": "colorful, modern, digital art look",
    },
    "pixel_art": {
        "label": "👾 Pixel Art",
        "prefix": "pixel art style, retro game art, 16-bit style, colorful pixels, masterpiece, best quality",
        "quality": "clean pixels, vibrant palette, detailed sprite art",
        "negative_extra": "blurry, smooth, photorealistic",
    },
    "oil_painting": {
        "label": "🖼️ Oil Painting",
        "prefix": "oil painting style, classical oil painting, rich texture, thick brush strokes, masterpiece, best quality",
        "quality": "dramatic chiaroscuro, golden hour lighting, museum quality, highly detailed",
        "negative_extra": "flat, digital, anime",
    },
    "comic_book": {
        "label": "💥 Comic / Manga",
        "prefix": "manga style, comic book art, bold lines, high contrast, dynamic composition, masterpiece, best quality",
        "quality": "screentone, dramatic angles, highly detailed, 4k",
        "negative_extra": "photorealistic, soft, watercolor",
    },
}


def get_style_choices() -> List[tuple]:
    """Get style choices for UI dropdown."""
    return [(v["label"], k) for k, v in IMAGE_STYLES.items()]


def get_style_prefix(style_key: str) -> str:
    """Get the full style prefix for a given style key."""
    style = IMAGE_STYLES.get(style_key, IMAGE_STYLES["anime"])
    return style["prefix"]


def get_style_negative(style_key: str) -> str:
    """Get extra negative prompt for a given style key."""
    style = IMAGE_STYLES.get(style_key, IMAGE_STYLES["anime"])
    return style.get("negative_extra", "")


def extract_scene_keywords(text: str) -> List[str]:
    """Extract English keywords from Vietnamese text for image prompts."""
    keywords = []
    text_lower = text.lower()
    for vn_word, en_keywords in SCENE_KEYWORDS.items():
        if vn_word in text_lower:
            keywords.append(en_keywords)
    return keywords


def generate_image_prompt(text: str, scene_id: int, style_prefix: str = "") -> str:
    """Generate an English image prompt from Vietnamese scene text."""
    keywords = extract_scene_keywords(text)

    if not keywords:
        base = "a beautiful scene, landscape, atmospheric lighting"
    else:
        base = ", ".join(keywords)

    # Use provided style_prefix or fall back to default anime
    if style_prefix:
        prompt = f"{style_prefix}, {base}"
    else:
        prompt = f"{IMAGE_STYLES['anime']['prefix']}, {base}"

    # Add quality boosters from style if matching, otherwise generic
    style_match = next((s for s in IMAGE_STYLES.values() if s["prefix"] == style_prefix), None)
    if style_match:
        prompt += f", {style_match['quality']}"
    else:
        prompt += ", cinematic lighting, highly detailed, 4k"

    return prompt


def split_text_to_scenes(
    text: str,
    min_chars: int = 100,
    max_chars: int = 1000,
    style_prefix: str = "",
) -> List[Scene]:
    """Split story text into scenes.

    Strategy:
    1. Split by double newline (paragraphs)
    2. Merge short paragraphs together
    3. Split overly long paragraphs
    """
    # Normalize whitespace
    text = text.strip()
    text = re.sub(r'\r\n', '\n', text)

    # Split by double newline
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Merge short paragraphs
    merged = []
    buffer = ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) > max_chars:
            merged.append(buffer.strip())
            buffer = para
        elif len(buffer) + len(para) < min_chars:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = para

    if buffer:
        merged.append(buffer.strip())

    # Split overly long paragraphs by sentences
    final_scenes = []
    for text_block in merged:
        if len(text_block) > max_chars * 1.5:
            sentences = re.split(r'(?<=[.!?。])\s+', text_block)
            chunk = ""
            for sent in sentences:
                if len(chunk) + len(sent) > max_chars:
                    if chunk:
                        final_scenes.append(chunk.strip())
                    chunk = sent
                else:
                    chunk = f"{chunk} {sent}" if chunk else sent
            if chunk:
                final_scenes.append(chunk.strip())
        else:
            final_scenes.append(text_block)

    # Create Scene objects
    scenes = []
    for i, scene_text in enumerate(final_scenes):
        img_prompt = generate_image_prompt(scene_text, i, style_prefix)
        scene = Scene(
            scene_id=i,
            text=scene_text,
            image_prompt=img_prompt,
            video_prompt=img_prompt,  # Reuse image prompt for video
        )
        scenes.append(scene)

    logger.info(f"Split text into {len(scenes)} scenes")
    return scenes


def save_scenes(scenes: List[Scene], output_path: str):
    """Save scenes to JSON file."""
    data = [s.to_dict() for s in scenes]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(scenes)} scenes to {output_path}")


def load_scenes(input_path: str) -> List[Scene]:
    """Load scenes from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = []
    for d in data:
        scene = Scene(
            scene_id=d["scene_id"],
            text=d["text"],
            image_prompt=d.get("image_prompt", ""),
            video_prompt=d.get("video_prompt", ""),
            audio_path=d.get("audio_path"),
            image_path=d.get("image_path"),
            video_path=d.get("video_path"),
            audio_duration=d.get("audio_duration", 0.0),
            status=d.get("status", "pending"),
        )
        scenes.append(scene)
    return scenes
