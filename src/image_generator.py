"""Image Generator - Generate anime illustrations using SDXL."""
import os
import gc
import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_pipe = None


def _cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class SDXLGenerator:
    """Generate anime-style images using Stable Diffusion XL."""

    def __init__(
        self,
        model_id: str = "cagliostrolab/animagine-xl-3.1",
        enable_cpu_offload: bool = True,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.enable_cpu_offload = enable_cpu_offload
        self.device = device
        self.pipe = None

    def load_model(self):
        """Load the SDXL model into memory."""
        if self.pipe is not None:
            return

        import torch
        from diffusers import StableDiffusionXLPipeline

        logger.info(f"Loading SDXL model: {self.model_id}")

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        if self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload")
        else:
            self.pipe = self.pipe.to(self.device)

        # Enable optimizations
        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass

        logger.info("SDXL model loaded successfully")

    def unload_model(self):
        """Unload model from memory to free GPU."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            _cleanup_gpu()
            logger.info("SDXL model unloaded")

    def generate(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = "lowres, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, blurry",
        width: int = 1280,
        height: int = 720,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        seed: int = -1,
    ) -> str:
        """Generate a single image.

        Args:
            prompt: Text prompt for image generation
            output_path: Path to save the output image
            negative_prompt: Things to avoid in generation
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed (-1 for random)

        Returns:
            Path to the generated image
        """
        import torch

        self.load_model()

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(f"Generating image (seed={seed}): {prompt[:80]}...")

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        image.save(output_path)
        logger.info(f"Image saved: {output_path}")

        return output_path


def create_image_generator(
    model_id: str = "cagliostrolab/animagine-xl-3.1",
    enable_cpu_offload: bool = True,
) -> SDXLGenerator:
    """Factory function to create an image generator."""
    return SDXLGenerator(model_id=model_id, enable_cpu_offload=enable_cpu_offload)
