"""Pipeline - Orchestrate the full story-to-video pipeline."""
import os
import time
import logging
from typing import List, Optional, Callable
from dataclasses import dataclass

from src.config import PipelineConfig
from src.text_processor import Scene, split_text_to_scenes, save_scenes, load_scenes
from src.tts_engine import create_tts_engine
from src.subtitle_generator import generate_srt, generate_srt_from_text
from src.image_generator import create_image_generator
from src.video_generator import create_video_generator
from src.video_merger import VideoMerger

logger = logging.getLogger(__name__)

# Type alias for progress callbacks: (current_step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


class StoryPipeline:
    """Full pipeline: Story Text → Audio → Images → Video → Final YouTube Video."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tts_engine = None
        self.image_generator = None
        self.video_generator = None
        self.video_merger = None

    def _init_tts(self):
        if self.tts_engine is None:
            cfg = self.config.tts
            if cfg.engine == "vieneu":
                self.tts_engine = create_tts_engine(
                    engine_type="vieneu",
                    model_id=cfg.vieneu_model_id,
                    mode=cfg.vieneu_mode,
                    emotion=cfg.vieneu_emotion,
                    voice_id=cfg.vieneu_voice_id,
                    ref_audio=cfg.vieneu_ref_audio,
                    ref_text=cfg.vieneu_ref_text,
                    rate=cfg.rate,
                )
            else:
                self.tts_engine = create_tts_engine(
                    engine_type=cfg.engine,
                    voice=cfg.voice,
                    rate=cfg.rate,
                    pitch=cfg.pitch,
                    volume=cfg.volume,
                )

    def _init_image(self):
        if self.image_generator is None:
            cfg = self.config.image
            self.image_generator = create_image_generator(
                model_id=cfg.model_id,
                enable_cpu_offload=cfg.enable_cpu_offload,
            )

    def _init_video(self):
        if self.video_generator is None:
            kb_cfg = self.config.video.ken_burns
            self.video_generator = create_video_generator(
                fps=kb_cfg.get("fps", 24),
                zoom_ratio=kb_cfg.get("zoom_ratio", 1.2),
            )

    def _init_merger(self):
        if self.video_merger is None:
            cfg = self.config.merger
            self.video_merger = VideoMerger(
                output_width=cfg.output_width,
                output_height=cfg.output_height,
                fps=cfg.fps,
                video_codec=cfg.video_codec,
                audio_codec=cfg.audio_codec,
                transition_duration=cfg.transition_duration,
                transition_type=cfg.transition_type,
            )

    def step1_process_text(
        self,
        story_text: str,
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Scene]:
        """Step 1: Split story into scenes."""
        if progress:
            progress(0, 1, "📝 Đang chia đoạn truyện...")

        scenes = split_text_to_scenes(
            text=story_text,
            min_chars=100,
            max_chars=1000,
            style_prefix=self.config.image.style_prefix,
        )

        scenes_path = os.path.join(project_dir, "scenes.json")
        save_scenes(scenes, scenes_path)

        if progress:
            progress(1, 1, f"✅ Đã chia thành {len(scenes)} đoạn")

        return scenes

    def step2_generate_audio(
        self,
        scenes: List[Scene],
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Scene]:
        """Step 2: Generate audio for each scene using TTS."""
        self._init_tts()
        audio_dir = os.path.join(project_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        total = len(scenes)
        for i, scene in enumerate(scenes):
            if progress:
                progress(i, total, f"🔊 Tạo audio scene {i+1}/{total}...")

            if scene.status == "done" and scene.audio_path and os.path.exists(scene.audio_path):
                logger.info(f"Skipping scene {i} (already done)")
                continue

            # VieNeu outputs WAV, edge-tts outputs MP3
            audio_ext = ".wav" if self.config.tts.engine == "vieneu" else ".mp3"
            output_path = os.path.join(audio_dir, f"scene_{scene.scene_id:04d}{audio_ext}")

            try:
                path, timestamps = self.tts_engine.generate(
                    text=scene.text,
                    output_path=output_path,
                    collect_timestamps=True,
                )
                scene.audio_path = path
                scene.audio_duration = self.tts_engine.get_audio_duration(path)

                # Generate subtitle for this scene
                if timestamps:
                    srt_dir = os.path.join(project_dir, "subtitles")
                    os.makedirs(srt_dir, exist_ok=True)
                    srt_path = os.path.join(srt_dir, f"scene_{scene.scene_id:04d}.srt")
                    generate_srt(timestamps, srt_path)

            except Exception as e:
                logger.error(f"TTS error scene {i}: {e}")
                scene.status = "error"
                continue

        # Save progress
        save_scenes(scenes, os.path.join(project_dir, "scenes.json"))

        if progress:
            progress(total, total, f"✅ Audio hoàn tất ({total} files)")

        return scenes

    def step3_generate_images(
        self,
        scenes: List[Scene],
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Scene]:
        """Step 3: Generate anime images for each scene."""
        self._init_image()
        image_dir = os.path.join(project_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        cfg = self.config.image
        total = len(scenes)

        for i, scene in enumerate(scenes):
            if progress:
                progress(i, total, f"🎨 Tạo hình scene {i+1}/{total}...")

            if scene.image_path and os.path.exists(scene.image_path):
                logger.info(f"Skipping image scene {i} (exists)")
                continue

            output_path = os.path.join(image_dir, f"scene_{scene.scene_id:04d}.png")

            try:
                path = self.image_generator.generate(
                    prompt=scene.image_prompt,
                    output_path=output_path,
                    negative_prompt=cfg.negative_prompt,
                    width=cfg.width,
                    height=cfg.height,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    seed=cfg.seed,
                )
                scene.image_path = path

            except Exception as e:
                logger.error(f"Image generation error scene {i}: {e}")
                scene.status = "error"
                continue

        # Unload image model to free VRAM for video generation
        self.image_generator.unload_model()

        save_scenes(scenes, os.path.join(project_dir, "scenes.json"))

        if progress:
            progress(total, total, f"✅ Hình ảnh hoàn tất ({total} images)")

        return scenes

    def step4_generate_videos(
        self,
        scenes: List[Scene],
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Scene]:
        """Step 4: Generate Ken Burns video clips for each scene."""
        self._init_video()
        video_dir = os.path.join(project_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        total = len(scenes)
        cfg = self.config.merger

        for i, scene in enumerate(scenes):
            if progress:
                progress(i, total, f"🎬 Tạo video scene {i+1}/{total}...")

            if scene.video_path and os.path.exists(scene.video_path):
                logger.info(f"Skipping video scene {i} (exists)")
                continue

            if not scene.image_path:
                logger.warning(f"Scene {i}: No image, skipping video generation")
                continue

            output_path = os.path.join(video_dir, f"scene_{scene.scene_id:04d}.mp4")

            try:
                path = self.video_generator.generate(
                    image_path=scene.image_path,
                    output_path=output_path,
                    duration=max(scene.audio_duration, 3.0),
                    output_width=cfg.output_width,
                    output_height=cfg.output_height,
                )
                scene.video_path = path
                scene.status = "done"

            except Exception as e:
                logger.error(f"Video generation error scene {i}: {e}")
                scene.status = "error"
                continue

        save_scenes(scenes, os.path.join(project_dir, "scenes.json"))

        if progress:
            progress(total, total, f"✅ Video hoàn tất ({total} clips)")

        return scenes

    def step5_merge_final(
        self,
        scenes: List[Scene],
        project_dir: str,
        output_filename: str = "final_video.mp4",
        progress: Optional[ProgressCallback] = None,
    ) -> str:
        """Step 5: Merge all scene clips into final video."""
        self._init_merger()

        if progress:
            progress(0, 1, "🎞️ Đang ghép video cuối cùng...")

        # Collect scene clips
        scene_clips = []
        for scene in scenes:
            if scene.video_path and os.path.exists(scene.video_path):
                audio = scene.audio_path if scene.audio_path and os.path.exists(scene.audio_path) else None
                scene_clips.append((scene.video_path, audio))

        if not scene_clips:
            raise ValueError("No valid scene clips found for merging")

        final_dir = os.path.join(project_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        output_path = os.path.join(final_dir, output_filename)

        self.video_merger.concatenate_scenes(
            scene_clips=scene_clips,
            output_path=output_path,
            progress_callback=progress,
        )

        if progress:
            progress(1, 1, f"✅ Video hoàn tất: {output_path}")

        return output_path

    def run(
        self,
        story_text: str,
        project_name: str = "my_story",
        progress: Optional[ProgressCallback] = None,
    ) -> str:
        """Run the full pipeline from text to final video.

        Args:
            story_text: The Vietnamese story text
            project_name: Name for the project folder
            progress: Progress callback function

        Returns:
            Path to the final video file
        """
        project_dir = os.path.join(self.config.output_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)

        start_time = time.time()
        logger.info(f"Starting pipeline for project: {project_name}")

        # Step 1: Process text
        scenes = self.step1_process_text(story_text, project_dir, progress)

        # Step 2: Generate audio
        scenes = self.step2_generate_audio(scenes, project_dir, progress)

        # Step 3: Generate images
        scenes = self.step3_generate_images(scenes, project_dir, progress)

        # Step 4: Generate videos (Ken Burns)
        scenes = self.step4_generate_videos(scenes, project_dir, progress)

        # Step 5: Merge final
        final_path = self.step5_merge_final(scenes, project_dir, progress=progress)

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.1f}s: {final_path}")

        return final_path
