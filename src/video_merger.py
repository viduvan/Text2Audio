"""Video Merger - Combine scene clips and audio into final YouTube video."""
import os
import logging
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

    def merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Merge a single audio file with a single video file."""
        from moviepy import VideoFileClip, AudioFileClip

        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)

        # Match video duration to audio
        if video.duration < audio.duration:
            # Loop video to match audio length
            from moviepy import vfx
            loops = int(audio.duration / video.duration) + 1
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
        scene_clips: List[Tuple[str, str]],  # List of (video_path, audio_path)
        output_path: str,
        progress_callback=None,
    ) -> str:
        """Concatenate multiple scene clips into one final video.

        Args:
            scene_clips: List of (video_path, audio_path) tuples
            output_path: Path for the final output video
            progress_callback: Optional callback(current, total, message)

        Returns:
            Path to the final video
        """
        from moviepy import (
            VideoFileClip, AudioFileClip,
            concatenate_videoclips, CompositeAudioClip, vfx
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

                    # Adjust video duration to match audio
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
            # Apply crossfade between clips
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
        import subprocess

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
