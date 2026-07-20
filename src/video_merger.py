"""Video Merger - Combine scene clips and audio into final YouTube video.

Supports two backends:
- FFmpeg (recommended for long videos, stream-based, low memory)
- MoviePy (fallback, loads into memory — NOT suitable for 10h+ videos)
"""
import os
import logging
import subprocess
import tempfile
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class VideoMerger:
    """Merge video clips with audio into a final YouTube-ready video."""

    def __init__(
        self,
        output_width: int = 1280,
        output_height: int = 720,
        fps: int = 24,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        transition_duration: float = 0.5,
        transition_type: str = "crossfade",
    ):
        self.output_width = output_width
        self.output_height = output_height
        self.fps = fps
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.transition_duration = transition_duration
        self.transition_type = transition_type

    # ─── FFmpeg Methods (Primary — fast, low memory) ───

    def merge_scene_ffmpeg(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Merge a single video + audio into one file using FFmpeg."""
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", self.audio_codec,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg merge error: {stderr[-500:]}")

        logger.debug(f"Merged scene: {output_path}")
        return output_path

    def concatenate_part_ffmpeg(
        self,
        scene_clips: List[Tuple[str, Optional[str]]],
        output_path: str,
        progress_callback=None,
    ) -> str:
        """Concatenate scene clips into one part using FFmpeg concat.

        Args:
            scene_clips: List of (video_path, audio_path) tuples
            output_path: Path for the part output
            progress_callback: Optional callback(current, total, message)

        Returns:
            Path to the merged part video
        """
        if not scene_clips:
            raise ValueError("No scene clips provided")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # Step 1: Merge each scene (video + audio) into temp files
        temp_dir = os.path.join(
            os.path.dirname(output_path), "_temp_merge"
        )
        os.makedirs(temp_dir, exist_ok=True)

        merged_files = []
        total = len(scene_clips)

        for i, (video_path, audio_path) in enumerate(scene_clips):
            if progress_callback:
                progress_callback(i, total, f"Merging scene {i+1}/{total}")

            if audio_path and os.path.exists(audio_path):
                merged_path = os.path.join(temp_dir, f"merged_{i:04d}.mp4")
                try:
                    self.merge_scene_ffmpeg(video_path, audio_path, merged_path)
                    merged_files.append(merged_path)
                except Exception as e:
                    logger.error(f"Failed to merge scene {i}: {e}")
                    continue
            else:
                # Video-only scene (e.g., watermark with no separate audio)
                merged_files.append(video_path)

        if not merged_files:
            raise ValueError("No valid merged clips")

        # Step 2: Create concat file
        concat_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for path in merged_files:
                # FFmpeg concat requires forward slashes and escaping
                safe_path = os.path.abspath(path).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        # Step 3: Concatenate all into one part
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_path,
            "-c:v", self.video_codec,
            "-c:a", self.audio_codec,
            "-preset", "fast",
            "-movflags", "+faststart",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg concat error: {stderr[-500:]}")

        # Cleanup temp files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        if progress_callback:
            progress_callback(total, total, "Part merge complete")

        logger.info(f"Part merged ({len(scene_clips)} scenes): {output_path}")
        return output_path

    def concatenate_parts_ffmpeg(
        self,
        part_paths: List[str],
        output_path: str,
    ) -> str:
        """Concatenate multiple part videos into the final video using FFmpeg.

        Uses stream copy (no re-encoding) for speed since parts share the same codec.
        """
        if not part_paths:
            raise ValueError("No parts to concatenate")

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # Create concat file
        concat_dir = os.path.dirname(output_path)
        concat_path = os.path.join(concat_dir, "_parts_concat.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for path in part_paths:
                safe_path = os.path.abspath(path).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        # Use stream copy for fast concat (same codec/resolution assumed)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True)

        # Cleanup
        try:
            os.remove(concat_path)
        except Exception:
            pass

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg parts concat error: {stderr[-500:]}")

        logger.info(f"Final video ({len(part_paths)} parts): {output_path}")
        return output_path

    # ─── MoviePy Methods (Fallback) ───

    def merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Merge a single audio file with a single video file (MoviePy fallback)."""
        from moviepy import VideoFileClip, AudioFileClip

        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Match video duration to audio
        if video.duration < audio.duration:
            from moviepy import vfx
            video = video.with_effects([vfx.Loop(duration=audio.duration)])

        video = video.with_audio(audio)
        video = video.subclipped(0, audio.duration)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        video.write_videofile(
            output_path,
            codec=self.video_codec,
            audio_codec=self.audio_codec,
            fps=self.fps,
            logger=None,
        )

        video.close()
        audio.close()

        logger.info(f"Merged audio+video: {output_path}")
        return output_path

    def concatenate_scenes(
        self,
        scene_clips: List[Tuple[str, str]],
        output_path: str,
        progress_callback=None,
    ) -> str:
        """Concatenate multiple scene clips into one final video (MoviePy fallback).

        WARNING: Not suitable for very long videos (10h+) due to memory usage.
        """
        from moviepy import (
            VideoFileClip, AudioFileClip,
            concatenate_videoclips, vfx
        )

        if not scene_clips:
            raise ValueError("No scene clips provided")

        final_clips = []
        total = len(scene_clips)

        for i, (video_path, audio_path) in enumerate(scene_clips):
            if progress_callback:
                progress_callback(i, total, f"Processing scene {i+1}/{total}")

            try:
                video = VideoFileClip(video_path)
                video = video.resized((self.output_width, self.output_height))

                if audio_path and os.path.exists(audio_path):
                    audio = AudioFileClip(audio_path)

                    if video.duration < audio.duration:
                        video = video.with_effects([vfx.Loop(duration=audio.duration)])

                    video = video.subclipped(0, audio.duration)
                    video = video.with_audio(audio)

                final_clips.append(video)
            except Exception as e:
                logger.error(f"Error processing scene {i}: {e}")
                continue

        if not final_clips:
            raise ValueError("No valid scene clips were processed")

        # Concatenate with transitions
        if self.transition_type == "crossfade" and self.transition_duration > 0:
            for i in range(len(final_clips)):
                if i > 0:
                    final_clips[i] = final_clips[i].with_effects([
                        vfx.CrossFadeIn(self.transition_duration)
                    ])
                if i < len(final_clips) - 1:
                    final_clips[i] = final_clips[i].with_effects([
                        vfx.CrossFadeOut(self.transition_duration)
                    ])

        final_video = concatenate_videoclips(final_clips, method="compose")

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        final_video.write_videofile(
            output_path,
            codec=self.video_codec,
            audio_codec=self.audio_codec,
            fps=self.fps,
            logger=None,
        )

        # Cleanup
        for clip in final_clips:
            clip.close()
        final_video.close()

        if progress_callback:
            progress_callback(total, total, "Done!")

        logger.info(f"Final video saved ({len(scene_clips)} scenes): {output_path}")
        return output_path

    # ─── Subtitle Burning ───

    def add_subtitles_ffmpeg(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
        font_size: int = 36,
        font_color: str = "white",
        stroke_color: str = "black",
        stroke_width: int = 2,
    ) -> str:
        """Burn subtitles into video using FFmpeg (more reliable than MoviePy)."""
        style = (
            f"FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"Outline={stroke_width},"
            f"Alignment=2,"
            f"MarginV=30"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles={srt_path}:force_style='{style}'",
            "-c:a", "copy",
            output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Subtitles burned: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg subtitle error: {e.stderr.decode()}")
            return video_path  # Return original if subtitle burn fails
