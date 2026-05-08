"""Video Generator - Create video clips using Ken Burns effect on images."""
import os
import logging
import random
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class KenBurnsGenerator:
    """Create video clips with Ken Burns effect (pan/zoom) on static images."""

    def __init__(self, fps: int = 24, zoom_ratio: float = 1.2):
        self.fps = fps
        self.zoom_ratio = zoom_ratio

    def generate(
        self,
        image_path: str,
        output_path: str,
        duration: float,
        effect: str = "random",
        output_width: int = 1280,
        output_height: int = 720,
    ) -> str:
        """Generate a video clip with Ken Burns effect on a static image.

        Args:
            image_path: Path to the source image
            output_path: Path to save the video
            duration: Duration in seconds
            effect: "zoom_in", "zoom_out", "pan_left", "pan_right", "random"
            output_width: Output video width
            output_height: Output video height

        Returns:
            Path to the generated video
        """
        from PIL import Image
        from moviepy import ImageClip, VideoClip

        if effect == "random":
            effect = random.choice(["zoom_in", "zoom_out", "pan_left", "pan_right"])

        # Load and prepare image
        img = Image.open(image_path).convert("RGB")
        # Scale image up for zoom headroom
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
                # Start wide, end zoomed in
                scale = 1.0 + (self.zoom_ratio - 1.0) * progress
                crop_w = int(output_width * scale_factor / scale)
                crop_h = int(output_height * scale_factor / scale)
                x = (new_w - crop_w) // 2
                y = (new_h - crop_h) // 2

            elif effect == "zoom_out":
                # Start zoomed in, end wide
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

            # Resize to output dimensions
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

        logger.info(f"Ken Burns video saved ({effect}, {duration:.1f}s): {output_path}")
        return output_path


def create_video_generator(fps: int = 24, zoom_ratio: float = 1.2, **kwargs) -> KenBurnsGenerator:
    """Factory function to create a Ken Burns video generator."""
    return KenBurnsGenerator(fps=fps, zoom_ratio=zoom_ratio)
