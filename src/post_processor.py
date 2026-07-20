"""Post-processor - Burn subtitles and merge parts into final video.

This is the final node (N8) in the pipeline:
- Merge per-scene SRT files into one master SRT with correct time offsets
- Burn subtitles into the final video
- Merge parts into a single output file
"""
import os
import logging
import subprocess
from typing import List, Tuple

logger = logging.getLogger(__name__)


class PostProcessor:
    """Post-processing: subtitles, part merging, final assembly."""

    def __init__(
        self,
        burn_subtitles: bool = True,
        subtitle_font_size: int = 36,
        subtitle_stroke_width: int = 2,
    ):
        self.burn_subtitles = burn_subtitles
        self.subtitle_font_size = subtitle_font_size
        self.subtitle_stroke_width = subtitle_stroke_width

    def merge_srt_files(
        self,
        srt_entries: List[Tuple[str, float]],
        output_path: str,
    ) -> str:
        """Merge multiple per-scene SRT files into a single master SRT.

        Args:
            srt_entries: List of (srt_file_path, start_time_offset) tuples
            output_path: Path for the merged SRT file

        Returns:
            Path to the merged SRT file
        """
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        global_index = 1
        with open(output_path, "w", encoding="utf-8") as out_f:
            for srt_path, time_offset in srt_entries:
                if not os.path.exists(srt_path):
                    continue

                with open(srt_path, "r", encoding="utf-8") as in_f:
                    content = in_f.read().strip()

                if not content:
                    continue

                # Parse SRT entries
                blocks = content.split("\n\n")
                for block in blocks:
                    lines = block.strip().split("\n")
                    if len(lines) < 3:
                        continue

                    # Parse timestamp line (format: HH:MM:SS,mmm --> HH:MM:SS,mmm)
                    timestamp_line = lines[1]
                    try:
                        start_str, end_str = timestamp_line.split(" --> ")
                        start_sec = _parse_srt_time(start_str.strip()) + time_offset
                        end_sec = _parse_srt_time(end_str.strip()) + time_offset
                    except Exception:
                        continue

                    text = "\n".join(lines[2:])

                    out_f.write(f"{global_index}\n")
                    out_f.write(
                        f"{_format_srt_time(start_sec)} --> "
                        f"{_format_srt_time(end_sec)}\n"
                    )
                    out_f.write(f"{text}\n\n")
                    global_index += 1

        logger.info(f"Merged SRT ({global_index - 1} entries): {output_path}")
        return output_path

    def create_srt_from_scenes(
        self,
        scene_texts: List[Tuple[str, float, float]],
        output_path: str,
    ) -> str:
        """Create a master SRT from scene texts and their timing.

        Args:
            scene_texts: List of (text, start_time, duration) tuples
            output_path: Path for the output SRT file

        Returns:
            Path to the SRT file
        """
        import re

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        index = 1
        with open(output_path, "w", encoding="utf-8") as f:
            for text, start_time, duration in scene_texts:
                if not text.strip():
                    continue

                # Split text into subtitle lines (~60 chars each)
                sentences = re.split(r'(?<=[.!?。])\s+', text.strip())
                total_chars = sum(len(s) for s in sentences)
                if total_chars == 0:
                    continue

                current_time = start_time
                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    # Duration proportional to character count
                    sent_duration = (len(sentence) / max(total_chars, 1)) * duration
                    end_time = current_time + sent_duration

                    # Split long sentences into 2 lines
                    if len(sentence) > 60:
                        mid = len(sentence) // 2
                        # Find nearest space to midpoint
                        space_pos = sentence.find(" ", mid)
                        if space_pos == -1 or space_pos > mid + 15:
                            space_pos = sentence.rfind(" ", 0, mid)
                        if space_pos > 0:
                            sentence = sentence[:space_pos] + "\n" + sentence[space_pos + 1:]

                    f.write(f"{index}\n")
                    f.write(
                        f"{_format_srt_time(current_time)} --> "
                        f"{_format_srt_time(end_time)}\n"
                    )
                    f.write(f"{sentence}\n\n")
                    index += 1
                    current_time = end_time

        logger.info(f"Created SRT ({index - 1} entries): {output_path}")
        return output_path

    def burn_subtitles_ffmpeg(
        self,
        video_path: str,
        srt_path: str,
        output_path: str,
    ) -> str:
        """Burn SRT subtitles into video using FFmpeg.

        Args:
            video_path: Input video path
            srt_path: SRT subtitle file
            output_path: Output video with burned subtitles

        Returns:
            Path to the output video (or original if burn fails)
        """
        if not os.path.exists(srt_path):
            logger.warning(f"SRT file not found: {srt_path}")
            return video_path

        # Skip if SRT is empty (0 entries)
        if os.path.getsize(srt_path) == 0:
            logger.warning("SRT file is empty (0 bytes), skipping subtitle burn")
            return video_path

        # Also check content to ensure it has valid entries
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read().strip()
        if not srt_content:
            logger.warning("SRT file has no content, skipping subtitle burn")
            return video_path

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        style = (
            f"FontSize={self.subtitle_font_size},"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"Outline={self.subtitle_stroke_width},"
            f"Alignment=2,"
            f"MarginV=30"
        )

        # Escape the SRT path for FFmpeg filter
        srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
            "-c:a", "copy",
            "-preset", "fast",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                logger.error(f"FFmpeg subtitle error: {stderr[-500:]}")
                return video_path

            logger.info(f"Subtitles burned: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Subtitle burn failed: {e}")
            return video_path

    def merge_parts_to_final(
        self,
        part_paths: List[str],
        output_path: str,
    ) -> str:
        """Merge part videos into a single final video using FFmpeg concat.

        Uses stream copy (no re-encoding) since all parts share codec/resolution.
        """
        if not part_paths:
            raise ValueError("No parts to merge")

        if len(part_paths) == 1:
            # Only one part — just copy/rename
            import shutil
            os.makedirs(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                exist_ok=True,
            )
            shutil.copy2(part_paths[0], output_path)
            return output_path

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )

        # Create concat list
        concat_dir = os.path.dirname(output_path)
        concat_path = os.path.join(concat_dir, "_final_concat.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for path in part_paths:
                safe_path = os.path.abspath(path).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg final concat error: {stderr[-500:]}")

            logger.info(f"Final video ({len(part_paths)} parts): {output_path}")
            return output_path
        finally:
            try:
                os.remove(concat_path)
            except Exception:
                pass


# ─── Helper Functions ───

def _parse_srt_time(time_str: str) -> float:
    """Parse SRT time format (HH:MM:SS,mmm) to seconds."""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
