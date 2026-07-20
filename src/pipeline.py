"""Pipeline - Orchestrate the full story-to-video pipeline (8 nodes).

Pipeline flow:
  N1: Text Rewrite (Gemini/Ollama — anti-copyright)
  N2: Scene Split + Watermark Injection
  N3: TTS Audio Generation (VieNeu v3 Turbo / Edge-TTS)
  N4: Image Generation (SDXL + LCM LoRA)
  N5: Ken Burns Video (FFmpeg zoompan)
  N6: Subtitle Generation (SRT)
  N7: Part Merge (scenes → parts with checkpoints)
  N8: Final Assembly (burn subs + merge parts → 1 file)
"""
import os
import time
import math
import logging
from typing import List, Optional, Callable
from collections import defaultdict

from src.config import PipelineConfig
from src.text_processor import (
    Scene, split_text_to_scenes, save_scenes, load_scenes,
    inject_watermarks,
)
from src.tts_engine import create_tts_engine
from src.subtitle_generator import generate_srt, generate_srt_from_text
from src.image_generator import create_image_generator
from src.video_generator import create_video_generator
from src.video_merger import VideoMerger
from src.post_processor import PostProcessor

logger = logging.getLogger(__name__)

# Type alias for progress callbacks: (current_step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


class StoryPipeline:
    """Full pipeline: Story Text → Audio → Images → Video → Final YouTube Video.

    8-node pipeline with parts-based checkpointing for reliability.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tts_engine = None
        self.image_generator = None
        self.video_generator = None
        self.video_merger = None
        self.text_rewriter = None
        self.post_processor = None

    # ─── Lazy Initialization ───

    def _init_rewriter(self):
        if self.text_rewriter is None and self.config.rewriter.enabled:
            from src.text_rewriter import TextRewriter
            self.text_rewriter = TextRewriter(config=self.config.rewriter)
            logger.info(f"Text rewriter initialized (provider={self.config.rewriter.provider})")

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
                use_lcm=cfg.use_lcm,
                lcm_lora_id=cfg.lcm_lora_id,
            )

    def _init_video(self):
        if self.video_generator is None:
            kb_cfg = self.config.video.ken_burns
            self.video_generator = create_video_generator(
                fps=kb_cfg.get("fps", 24),
                zoom_ratio=kb_cfg.get("zoom_ratio", 1.2),
                use_ffmpeg=self.config.video.use_ffmpeg,
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

    def _init_post_processor(self):
        if self.post_processor is None:
            cfg = self.config.post_process
            self.post_processor = PostProcessor(
                burn_subtitles=cfg.burn_subtitles,
            )

    # ─── N1: Text Rewriter ───

    def step0_rewrite_text(
        self,
        story_text: str,
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> str:
        """N1: Rewrite story text to avoid copyright issues.

        Saves both original and rewritten versions for comparison.
        """
        if not self.config.rewriter.enabled:
            if progress:
                progress(1, 1, "⏭️ Text rewrite disabled, skipping...")
            return story_text

        self._init_rewriter()

        if progress:
            progress(0, 1, "🔄 Đang viết lại truyện (chống bản quyền)...")

        # Save original
        original_path = os.path.join(project_dir, "original_text.txt")
        with open(original_path, "w", encoding="utf-8") as f:
            f.write(story_text)

        # Rewrite
        rewritten = self.text_rewriter.rewrite_story(
            story_text,
            progress_callback=progress,
        )

        # Save rewritten
        rewritten_path = os.path.join(project_dir, "rewritten_text.txt")
        with open(rewritten_path, "w", encoding="utf-8") as f:
            f.write(rewritten)

        if progress:
            progress(1, 1, f"✅ Viết lại hoàn tất ({len(rewritten)} chars)")

        return rewritten

    # ─── N2: Scene Splitter + Watermark ───

    def step1_process_text(
        self,
        story_text: str,
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Scene]:
        """N2: Split story into scenes and inject watermarks."""
        if progress:
            progress(0, 1, "📝 Đang chia đoạn truyện...")

        scenes = split_text_to_scenes(
            text=story_text,
            min_chars=100,
            max_chars=1000,
            style_prefix=self.config.image.style_prefix,
        )

        content_count = len(scenes)

        # Inject watermarks
        scenes = inject_watermarks(
            scenes=scenes,
            watermark_config=self.config.watermark,
            parts_per_video=self.config.merger.parts_per_video,
        )

        scenes_path = os.path.join(project_dir, "scenes.json")
        save_scenes(scenes, scenes_path)

        wm_count = sum(1 for s in scenes if s.is_watermark)
        if progress:
            progress(
                1, 1,
                f"✅ {content_count} đoạn + {wm_count} watermarks = {len(scenes)} scenes"
            )

        return scenes

    # ─── N3: TTS Audio ───

    def step2_generate_audio(
        self,
        scenes: List[Scene],
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Scene]:
        """N3: Generate audio for each scene using TTS."""
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

    # ─── N4: Image Generation ───

    def step3_generate_images(
        self,
        scenes: List[Scene],
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Scene]:
        """N4: Generate anime images for each content scene.

        Watermark scenes skip image generation and reuse the previous scene's image.
        """
        self._init_image()
        image_dir = os.path.join(project_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        cfg = self.config.image
        total = len(scenes)
        last_image_path = None

        for i, scene in enumerate(scenes):
            if progress:
                progress(i, total, f"🎨 Tạo hình scene {i+1}/{total}...")

            # Watermark scenes reuse previous image
            if scene.is_watermark:
                scene.image_path = last_image_path
                continue

            if scene.image_path and os.path.exists(scene.image_path):
                last_image_path = scene.image_path
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
                last_image_path = path

            except Exception as e:
                logger.error(f"Image generation error scene {i}: {e}")
                scene.status = "error"
                continue

        # Unload image model to free VRAM
        self.image_generator.unload_model()

        save_scenes(scenes, os.path.join(project_dir, "scenes.json"))

        if progress:
            content_images = sum(1 for s in scenes if not s.is_watermark and s.image_path)
            progress(total, total, f"✅ Hình ảnh hoàn tất ({content_images} images)")

        return scenes

    # ─── N5: Ken Burns Video ───

    def step4_generate_videos(
        self,
        scenes: List[Scene],
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> List[Scene]:
        """N5: Generate Ken Burns video clips for each scene."""
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

            if not scene.image_path or not os.path.exists(scene.image_path):
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

    # ─── N6: Subtitle Generation ───

    def step5_generate_subtitles(
        self,
        scenes: List[Scene],
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> str:
        """N6: Generate master SRT subtitle file from all scenes.

        Returns path to the master SRT file.
        """
        self._init_post_processor()

        if progress:
            progress(0, 1, "📝 Đang tạo phụ đề...")

        # Collect scene timing information
        scene_texts = []
        current_time = 0.0

        for scene in scenes:
            if scene.audio_duration > 0:
                scene_texts.append((
                    scene.text,
                    current_time,
                    scene.audio_duration,
                ))
                current_time += scene.audio_duration

        # Create master SRT
        srt_path = os.path.join(project_dir, "subtitles", "master.srt")
        self.post_processor.create_srt_from_scenes(scene_texts, srt_path)

        if progress:
            progress(1, 1, f"✅ Phụ đề hoàn tất ({len(scene_texts)} entries)")

        return srt_path

    # ─── N7: Part Merge ───

    def step6_merge_parts(
        self,
        scenes: List[Scene],
        project_dir: str,
        progress: Optional[ProgressCallback] = None,
    ) -> List[str]:
        """N7: Merge scenes into parts with checkpoints.

        Groups scenes by part_id, merges each part separately.
        Each part is a checkpoint — if pipeline crashes, completed parts are preserved.

        Returns list of part file paths.
        """
        self._init_merger()

        parts_dir = os.path.join(project_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)

        # Group scenes by part_id
        parts = defaultdict(list)
        for scene in scenes:
            if scene.video_path and os.path.exists(scene.video_path):
                parts[scene.part_id].append(scene)

        if not parts:
            raise ValueError("No valid scenes found for merging")

        sorted_part_ids = sorted(parts.keys())
        total_parts = len(sorted_part_ids)
        part_paths = []

        for idx, part_id in enumerate(sorted_part_ids):
            part_scenes = parts[part_id]
            part_path = os.path.join(parts_dir, f"part_{part_id:03d}.mp4")

            # Skip if part already exists (checkpoint)
            if os.path.exists(part_path):
                logger.info(f"Part {part_id} already exists, skipping")
                part_paths.append(part_path)
                continue

            if progress:
                progress(
                    idx, total_parts,
                    f"🎞️ Ghép part {idx+1}/{total_parts} ({len(part_scenes)} scenes)..."
                )

            # Prepare scene clips for merging
            scene_clips = []
            for scene in part_scenes:
                audio = (
                    scene.audio_path
                    if scene.audio_path and os.path.exists(scene.audio_path)
                    else None
                )
                scene_clips.append((scene.video_path, audio))

            try:
                self.video_merger.concatenate_part_ffmpeg(
                    scene_clips=scene_clips,
                    output_path=part_path,
                )
                part_paths.append(part_path)
                logger.info(f"Part {part_id} saved: {part_path}")
            except Exception as e:
                logger.error(f"Part {part_id} merge failed: {e}")
                continue

        if progress:
            progress(
                total_parts, total_parts,
                f"✅ {len(part_paths)}/{total_parts} parts hoàn tất"
            )

        return part_paths

    # ─── N8: Final Assembly ───

    def step7_final_assembly(
        self,
        part_paths: List[str],
        srt_path: str,
        project_dir: str,
        output_filename: str = "final_video.mp4",
        progress: Optional[ProgressCallback] = None,
    ) -> str:
        """N8: Final assembly — merge parts and burn subtitles.

        1. Merge all parts into one video
        2. Burn subtitles (if enabled)

        Returns path to the final video.
        """
        self._init_post_processor()

        final_dir = os.path.join(project_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        if progress:
            progress(0, 2, "📦 Ghép tất cả parts thành video cuối...")

        # Step 1: Merge parts
        if self.config.post_process.burn_subtitles and os.path.exists(srt_path):
            # Merge to temp, then burn subs
            temp_path = os.path.join(final_dir, "_temp_merged.mp4")
            merged_path = self.post_processor.merge_parts_to_final(
                part_paths, temp_path,
            )

            if progress:
                progress(1, 2, "📝 Đang ghi phụ đề vào video...")

            # Step 2: Burn subtitles
            output_path = os.path.join(final_dir, output_filename)
            final_path = self.post_processor.burn_subtitles_ffmpeg(
                merged_path, srt_path, output_path,
            )

            # Cleanup temp
            try:
                if os.path.exists(temp_path) and final_path != temp_path:
                    os.remove(temp_path)
            except Exception:
                pass
        else:
            # No subtitles — merge directly to final
            output_path = os.path.join(final_dir, output_filename)
            final_path = self.post_processor.merge_parts_to_final(
                part_paths, output_path,
            )

        if progress:
            progress(2, 2, f"✅ Video hoàn tất: {final_path}")

        return final_path

    # ─── Full Pipeline Run ───

    def run(
        self,
        story_text: str,
        project_name: str = "my_story",
        progress: Optional[ProgressCallback] = None,
    ) -> str:
        """Run the full 8-node pipeline from text to final video.

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

        # N1: Text Rewrite (anti-copyright)
        story_text = self.step0_rewrite_text(story_text, project_dir, progress)

        # N2: Scene Split + Watermark
        scenes = self.step1_process_text(story_text, project_dir, progress)

        # N3: TTS Audio
        scenes = self.step2_generate_audio(scenes, project_dir, progress)

        # N4: Image Generation (SDXL + LCM LoRA)
        scenes = self.step3_generate_images(scenes, project_dir, progress)

        # N5: Ken Burns Video
        scenes = self.step4_generate_videos(scenes, project_dir, progress)

        # N6: Subtitle Generation
        srt_path = self.step5_generate_subtitles(scenes, project_dir, progress)

        # N7: Part Merge (with checkpoints)
        part_paths = self.step6_merge_parts(scenes, project_dir, progress)

        # N8: Final Assembly (merge parts + burn subs)
        final_path = self.step7_final_assembly(
            part_paths, srt_path, project_dir, progress=progress,
        )

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.1f}s: {final_path}")

        return final_path
