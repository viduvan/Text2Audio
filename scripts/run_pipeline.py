#!/usr/bin/env python
"""Run the full pipeline from command line."""
import os
import sys
import argparse
import logging

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import PipelineConfig, ensure_directories
from src.pipeline import StoryPipeline


def main():
    parser = argparse.ArgumentParser(description="Text2Audio - Story to Video Pipeline")
    parser.add_argument("--story", required=True, help="Path to story text file")
    parser.add_argument("--project", default="cli_project", help="Project name")
    parser.add_argument("--voice", default="vi-VN-HoaiMyNeural",
                        choices=["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
                        help="TTS voice")
    parser.add_argument("--rate", default="+0%", help="TTS speed rate")
    parser.add_argument("--video-mode", default="ken_burns",
                        choices=["ken_burns", "wan21"], help="Video generation mode")
    parser.add_argument("--config", default="config.yaml", help="Config file path")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Read story
    with open(args.story, "r", encoding="utf-8") as f:
        story_text = f.read()

    print(f"\n📄 Story: {args.story} ({len(story_text)} chars)")
    print(f"🔊 Voice: {args.voice}")
    print(f"🎬 Video: {args.video_mode}")
    print(f"📁 Project: {args.project}\n")

    # Load config and apply CLI overrides
    config = PipelineConfig.from_yaml(args.config)
    config.tts.voice = args.voice
    config.tts.rate = args.rate
    config.video.default_mode = args.video_mode
    ensure_directories(config.__dict__)

    # Run pipeline
    pipeline = StoryPipeline(config)

    def progress_cb(current, total, message):
        if total > 0:
            pct = int(current / total * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}% - {message}", end="", flush=True)
            if current == total:
                print()

    final_path = pipeline.run(
        story_text=story_text,
        project_name=args.project,
        video_mode=args.video_mode,
        progress=progress_cb,
    )

    print(f"\n🎉 Done! Final video: {final_path}")


if __name__ == "__main__":
    main()
