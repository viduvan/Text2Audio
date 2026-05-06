"""Video Generator - Create video clips from images/prompts."""
import os
import gc
import logging
import random
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def _cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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


class Wan21Generator:
    """Generate AI video using Wan 2.1 T2V 1.3B model."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        enable_cpu_offload: bool = True,
        enable_vae_tiling: bool = True,
    ):
        self.model_id = model_id
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_vae_tiling = enable_vae_tiling
        self.pipe = None

    def load_model(self):
        """Load the Wan 2.1 model into GPU."""
        if self.pipe is not None:
            return

        import torch
        from diffusers import WanPipeline

        logger.info(f"Loading Wan 2.1 model: {self.model_id}")

        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )

        if self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()

        if self.enable_vae_tiling:
            try:
                self.pipe.vae.enable_tiling()
            except Exception:
                pass

        logger.info("Wan 2.1 model loaded successfully")

    def unload_model(self):
        """Unload model from GPU."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            _cleanup_gpu()
            logger.info("Wan 2.1 model unloaded")

    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 832,
        height: int = 480,
        num_frames: int = 81,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        fps: int = 16,
        seed: int = -1,
    ) -> str:
        """Generate a video clip from text prompt.

        Args:
            prompt: English text prompt
            output_path: Path to save the video
            width: Video width (recommend 832)
            height: Video height (recommend 480)
            num_frames: Number of frames (81 = ~5 seconds at 16fps)
            num_inference_steps: Denoising steps
            guidance_scale: Prompt adherence
            fps: Frames per second
            seed: Random seed (-1 for random)

        Returns:
            Path to the generated video
        """
        import torch
        from diffusers.utils import export_to_video

        self.load_model()

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"Generating video (seed={seed}): {prompt[:80]}...")

        video = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        export_to_video(video, output_path, fps=fps)

        logger.info(f"AI video saved ({num_frames} frames): {output_path}")
        return output_path


def create_video_generator(mode: str = "ken_burns", **kwargs):
    """Factory function to create a video generator.

    Args:
        mode: "ken_burns" or "wan21"
    """
    if mode == "ken_burns":
        return KenBurnsGenerator(
            fps=kwargs.get("fps", 24),
            zoom_ratio=kwargs.get("zoom_ratio", 1.2),
        )
    elif mode == "wan21":
        return Wan21Generator(
            model_id=kwargs.get("model_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            enable_cpu_offload=kwargs.get("enable_cpu_offload", True),
            enable_vae_tiling=kwargs.get("enable_vae_tiling", True),
        )
    else:
        raise ValueError(f"Unknown video mode: {mode}. Supported: ken_burns, wan21")
