"""Video Generator - Create video clips using Ken Burns effect on images.

Supports two backends:
- FFmpeg (fast, recommended): Uses zoompan filter
- MoviePy (fallback): Python-based, slower but more flexible
"""
import os
import logging
import random
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class KenBurnsGenerator:
    """Create video clips with Ken Burns effect (pan/zoom) on static images."""

    def __init__(self, fps: int = 24, zoom_ratio: float = 1.2, use_ffmpeg: bool = True):
        self.fps = fps
        self.zoom_ratio = zoom_ratio
        self.use_ffmpeg = use_ffmpeg

    def generate(
        self,
        image_path: str,
        output_path: str,
        duration: float,
        effect: str = "random",
        output_width: int = 1280,
        output_height: int = 720,
    ) -> str:
        """Generate a video clip with Ken Burns effect.

        Uses FFmpeg by default for speed, falls back to MoviePy.
        """
        if effect == "random":
            effect = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])

        if self.use_ffmpeg:
            try:
                return self._generate_ffmpeg(
                    image_path, output_path, duration, effect,
                    output_width, output_height,
                )
            except Exception as e:
                logger.warning(f"FFmpeg Ken Burns failed, falling back to MoviePy: {e}")

        return self._generate_moviepy(
            image_path, output_path, duration, effect,
            output_width, output_height,
        )

    def _generate_ffmpeg(
        self,
        image_path: str,
        output_path: str,
        duration: float,
        effect: str,
        output_width: int,
        output_height: int,
    ) -> str:
        """Generate Ken Burns effect using FFmpeg zoompan filter.

        Much faster than MoviePy — no Python frame-by-frame processing.
        """
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        total_frames = int(duration * self.fps)
        # zoompan 'd' is total frames for the animation
        # 'z' is zoom level expression, 's' is output size

        if effect == "zoom_in":
            # Start at zoom=1, end at zoom=zoom_ratio
            zoom_expr = f"min(zoom+{(self.zoom_ratio - 1.0) / total_frames:.6f},{{zr}})"
            zoom_expr = zoom_expr.format(zr=self.zoom_ratio)
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif effect == "zoom_out":
            # Start at zoom=zoom_ratio, end at zoom=1
            zoom_expr = f"if(eq(on,1),{self.zoom_ratio},max(zoom-{(self.zoom_ratio - 1.0) / total_frames:.6f},1.0))"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif effect == "pan_left":
            # Pan from right to left
            zoom_expr = "1.1"  # Slight zoom to avoid black edges
            x_expr = f"iw*(1-1/zoom)*(1-on/{total_frames})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif effect == "pan_right":
            # Pan from left to right
            zoom_expr = "1.1"
            x_expr = f"iw*(1-1/zoom)*(on/{total_frames})"
            y_expr = "ih/2-(ih/zoom/2)"
        else:
            zoom_expr = "1"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"

        zoompan_filter = (
            f"zoompan=z='{zoom_expr}'"
            f":x='{x_expr}'"
            f":y='{y_expr}'"
            f":d={total_frames}"
            f":s={output_width}x{output_height}"
            f":fps={self.fps}"
        )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", zoompan_filter,
            "-t", str(duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-an",  # No audio
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=max(duration * 3, 30),
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg error: {stderr[-500:]}")

            logger.info(f"Ken Burns video (FFmpeg, {effect}, {duration:.1f}s): {output_path}")
            return output_path

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FFmpeg timeout for {output_path}")

    def _generate_moviepy(
        self,
        image_path: str,
        output_path: str,
        duration: float,
        effect: str,
        output_width: int,
        output_height: int,
    ) -> str:
        """Generate Ken Burns effect using MoviePy (fallback)."""
        import numpy as np
        from PIL import Image
        from moviepy import VideoClip

        # Load and prepare image
        img = Image.open(image_path).convert("RGB")
        scale_factor = self.zoom_ratio + 0.1
        new_w = int(output_width * scale_factor)
        new_h = int(output_height * scale_factor)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img_array = np.array(img)

        def make_frame(t):
            """Generate frame at time t with Ken Burns effect."""
            progress = t / max(duration, 0.001)
            progress = min(progress, 1.0)

            if effect == "zoom_in":
                scale = 1.0 + (self.zoom_ratio - 1.0) * progress
                crop_w = int(output_width * scale_factor / scale)
                crop_h = int(output_height * scale_factor / scale)
                x = (new_w - crop_w) // 2
                y = (new_h - crop_h) // 2
            elif effect == "zoom_out":
                scale = self.zoom_ratio - (self.zoom_ratio - 1.0) * progress
                crop_w = int(output_width * scale_factor / scale)
                crop_h = int(output_height * scale_factor / scale)
                x = (new_w - crop_w) // 2
                y = (new_h - crop_h) // 2
            elif effect == "pan_left":
                crop_w = output_width
                crop_h = output_height
                max_pan = new_w - crop_w
                x = int(max_pan * (1.0 - progress))
                y = (new_h - crop_h) // 2
            elif effect == "pan_right":
                crop_w = output_width
                crop_h = output_height
                max_pan = new_w - crop_w
                x = int(max_pan * progress)
                y = (new_h - crop_h) // 2
            else:
                crop_w = output_width
                crop_h = output_height
                x = (new_w - crop_w) // 2
                y = (new_h - crop_h) // 2

            # Ensure bounds
            crop_w = min(crop_w, new_w)
            crop_h = min(crop_h, new_h)
            x = max(0, min(x, new_w - crop_w))
            y = max(0, min(y, new_h - crop_h))

            # Crop and resize
            cropped = img_array[y:y+crop_h, x:x+crop_w]
            from PIL import Image as PILImage
            frame = PILImage.fromarray(cropped).resize(
                (output_width, output_height), PILImage.LANCZOS
            )
            return np.array(frame)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        clip = VideoClip(make_frame, duration=duration)
        clip.write_videofile(
            output_path,
            fps=self.fps,
            codec="libx264",
            audio=False,
            logger=None,
        )
        clip.close()

        logger.info(f"Ken Burns video (MoviePy, {effect}, {duration:.1f}s): {output_path}")
        return output_path


def create_video_generator(
    fps: int = 24,
    zoom_ratio: float = 1.2,
    use_ffmpeg: bool = True,
    **kwargs,
) -> KenBurnsGenerator:
    """Factory function to create a Ken Burns video generator."""
    return KenBurnsGenerator(fps=fps, zoom_ratio=zoom_ratio, use_ffmpeg=use_ffmpeg)
