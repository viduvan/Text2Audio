"""Text2Audio - Web UI (Gradio)

A premium dark-themed Web UI for the Story-to-YouTube-Video pipeline.
"""
import os
import sys
import time
import logging
import gradio as gr
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import PipelineConfig, ensure_directories
from src.text_processor import split_text_to_scenes, Scene, get_style_choices, get_style_prefix, get_style_negative, IMAGE_STYLES
from src.tts_engine import create_tts_engine, list_available_engines
from src.pipeline import StoryPipeline


# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
pipeline_instance: StoryPipeline = None


def get_pipeline() -> StoryPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        config = PipelineConfig.from_yaml(CONFIG_PATH)
        ensure_directories(config.__dict__)
        pipeline_instance = StoryPipeline(config)
    return pipeline_instance


# ─────────────────────────────────────────────
# CSS Theme
# ─────────────────────────────────────────────
CUSTOM_CSS = """
/* ═══ Root Theme ═══ */
:root {
    --primary: #8b5cf6;
    --primary-hover: #7c3aed;
    --accent: #ec4899;
    --bg-dark: #0a0a1a;
    --bg-card: #111127;
    --bg-input: #1a1a3e;
    --border: #2d2d5e;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
}

/* ═══ Global ═══ */
.gradio-container {
    background: var(--bg-dark) !important;
    max-width: 1400px !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ═══ Header ═══ */
.app-header {
    text-align: center;
    padding: 24px 0 16px 0;
    background: linear-gradient(135deg, #1a1a3e 0%, #0f0f2e 100%);
    border-radius: 16px;
    margin-bottom: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 24px rgba(139, 92, 246, 0.1);
}

.app-header h1 {
    font-size: 2rem !important;
    background: linear-gradient(135deg, #8b5cf6, #ec4899, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

.app-header p {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
    margin-top: 4px !important;
}

/* ═══ Tabs ═══ */
.tab-nav button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
    color: white !important;
    border-color: var(--primary) !important;
}

/* ═══ Buttons ═══ */
.primary-btn {
    background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(139, 92, 246, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(139, 92, 246, 0.5) !important;
}

/* ═══ Inputs ═══ */
textarea, input[type="text"], select {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
}

/* ═══ Cards ═══ */
.gr-panel, .gr-box, .gr-group {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ═══ Log Box ═══ */
.log-box textarea {
    background: #050510 !important;
    color: #10b981 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.85rem !important;
    border: 1px solid #1a3a2a !important;
}

/* ═══ Status badges ═══ */
.status-running {
    color: var(--warning);
    font-weight: bold;
}
.status-done {
    color: var(--success);
    font-weight: bold;
}
.status-error {
    color: var(--error);
    font-weight: bold;
}
"""


# ─────────────────────────────────────────────
# Pipeline Functions
# ─────────────────────────────────────────────

def load_story_file(file):
    """Load story text from uploaded file."""
    if file is None:
        return ""
    with open(file.name, "r", encoding="utf-8") as f:
        return f.read()


def preview_scenes(story_text, min_chars, max_chars, image_style):
    """Preview how the story will be split into scenes."""
    if not story_text.strip():
        return "⚠️ Vui lòng nhập nội dung truyện!", ""

    style_prefix = get_style_prefix(image_style) if image_style else ""
    scenes = split_text_to_scenes(
        text=story_text,
        min_chars=int(min_chars),
        max_chars=int(max_chars),
        style_prefix=style_prefix,
    )

    preview = f"📊 **Tổng cộng: {len(scenes)} scene** | Style: **{IMAGE_STYLES.get(image_style, {}).get('label', image_style)}**\n\n"
    for s in scenes:
        text_preview = s.text[:150] + "..." if len(s.text) > 150 else s.text
        preview += f"---\n**Scene {s.scene_id + 1}** ({len(s.text)} ký tự)\n\n"
        preview += f"📝 {text_preview}\n\n"
        preview += f"🎨 *Prompt:* {s.image_prompt[:120]}...\n\n"

    return preview, f"✅ {len(scenes)} scenes"


def run_tts_only(text, engine_type, voice_or_id, rate, emotion):
    """Run TTS on a single text input."""
    if not text.strip():
        return None, "⚠️ Vui lòng nhập text!"

    output_dir = os.path.join(PROJECT_ROOT, "output", "tts_preview")
    os.makedirs(output_dir, exist_ok=True)

    try:
        if engine_type == "vieneu":
            engine = create_tts_engine(
                engine_type="vieneu",
                emotion=emotion or "storytelling",
                voice_id=voice_or_id if voice_or_id else None,
            )
            ext = "wav"
        else:
            engine = create_tts_engine(
                engine_type="edge-tts",
                voice=voice_or_id or "vi-VN-HoaiMyNeural",
                rate=rate or "+0%",
            )
            ext = "mp3"

        output_path = os.path.join(output_dir, f"preview_{int(time.time())}.{ext}")
        path, _ = engine.generate(text, output_path)
        duration = engine.get_audio_duration(path)
        return path, f"✅ Audio tạo thành công! ({duration:.1f}s) [{engine_type}]"
    except Exception as e:
        logger.exception("TTS error")
        return None, f"❌ Lỗi: {str(e)}"


def run_full_pipeline(
    story_text, project_name, tts_engine, voice_or_id, rate, emotion,
    video_mode, image_style, progress=gr.Progress(track_tqdm=True)
):
    """Run the full story-to-video pipeline."""
    if not story_text.strip():
        yield "⚠️ Vui lòng nhập nội dung truyện!", None, None, None, []

    if not project_name.strip():
        project_name = f"project_{int(time.time())}"

    # Clean project name
    project_name = project_name.strip().replace(" ", "_").lower()

    log_messages = []

    def log(msg):
        log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        return "\n".join(log_messages)

    def progress_cb(current, total, message):
        if total > 0:
            progress((current / total), desc=message)
        log(message)

    try:
        # Initialize pipeline
        config = PipelineConfig.from_yaml(CONFIG_PATH)
        config.tts.engine = tts_engine
        if tts_engine == "vieneu":
            config.tts.vieneu_emotion = emotion or "storytelling"
            config.tts.vieneu_voice_id = voice_or_id if voice_or_id else "Ly"
        else:
            config.tts.voice = voice_or_id or "vi-VN-HoaiMyNeural"
            config.tts.rate = rate or "+0%"
        config.video.default_mode = video_mode

        # Apply image style
        style_prefix = get_style_prefix(image_style) if image_style else ""
        style_negative = get_style_negative(image_style) if image_style else ""
        config.image.style_prefix = style_prefix
        if style_negative:
            config.image.negative_prompt += f", {style_negative}"

        ensure_directories(config.__dict__)

        pipe = StoryPipeline(config)
        project_dir = os.path.join(config.output_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)

        # Step 1: Process text
        yield log("📝 Bước 1: Đang chia đoạn truyện..."), None, None, None, []
        scenes = pipe.step1_process_text(story_text, project_dir, progress_cb)
        yield log(f"✅ Bước 1 hoàn tất: {len(scenes)} scenes"), None, None, None, []

        # Step 2: Generate audio
        yield log("🔊 Bước 2: Đang tạo audio..."), None, None, None, []
        scenes = pipe.step2_generate_audio(scenes, project_dir, progress_cb)
        total_duration = sum(s.audio_duration for s in scenes)
        first_audio = next((s.audio_path for s in scenes if s.audio_path), None)
        yield log(f"✅ Bước 2 hoàn tất: {total_duration:.1f}s audio"), first_audio, None, None, []

        # Step 3: Generate images
        yield log("🎨 Bước 3: Đang tạo hình ảnh anime..."), first_audio, None, None, []
        scenes = pipe.step3_generate_images(scenes, project_dir, progress_cb)
        images = [s.image_path for s in scenes if s.image_path and os.path.exists(s.image_path)]
        yield log(f"✅ Bước 3 hoàn tất: {len(images)} images"), first_audio, None, None, images

        # Step 4: Generate videos
        yield log(f"🎬 Bước 4: Đang tạo video ({video_mode})..."), first_audio, None, None, images
        scenes = pipe.step4_generate_videos(scenes, project_dir, video_mode, progress_cb)
        yield log("✅ Bước 4 hoàn tất"), first_audio, None, None, images

        # Step 5: Merge final
        yield log("🎞️ Bước 5: Đang ghép video cuối cùng..."), first_audio, None, None, images
        final_path = pipe.step5_merge_final(scenes, project_dir, progress=progress_cb)
        yield (
            log(f"🎉 HOÀN TẤT! Video: {final_path}"),
            first_audio,
            final_path,
            final_path,
            images,
        )

    except Exception as e:
        logger.exception("Pipeline error")
        yield log(f"❌ LỖI: {str(e)}"), None, None, None, []


# ─────────────────────────────────────────────
# Gradio UI Layout
# ─────────────────────────────────────────────

def create_ui():
    """Build the Gradio web interface."""

    theme = gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="pink",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Text2Audio - Story to Video") as app:

        # ═══ HEADER ═══
        gr.HTML("""
        <div class="app-header">
            <h1>✨ Text2Audio Studio</h1>
            <p>Chuyển đổi truyện thành video YouTube với AI | Vietnamese TTS + Anime Art + AI Video</p>
        </div>
        """)

        with gr.Tabs():
            # ═══════════════════════════════
            # TAB 1: FULL PIPELINE
            # ═══════════════════════════════
            with gr.Tab("🚀 Pipeline", id="pipeline"):
                with gr.Row():
                    # ── Left: Input ──
                    with gr.Column(scale=3):
                        gr.Markdown("### 📄 Nội dung truyện")
                        story_input = gr.Textbox(
                            label="Nhập hoặc dán truyện tiếng Việt",
                            placeholder="Ngày xưa, ở một ngôi làng nhỏ...",
                            lines=12,
                            max_lines=30,
                        )
                        story_file = gr.File(
                            label="📂 Hoặc tải file .txt",
                            file_types=[".txt"],
                        )
                        story_file.change(
                            fn=load_story_file,
                            inputs=[story_file],
                            outputs=[story_input],
                        )

                        gr.Markdown("### ⚙️ Cài đặt")
                        with gr.Row():
                            project_name = gr.Textbox(
                                label="Tên dự án",
                                value="my_story",
                                scale=2,
                            )
                            tts_engine_select = gr.Dropdown(
                                label="🔊 Engine TTS",
                                choices=[
                                    ("🇻🇳 VieNeu-TTS (Local, 7+ giọng)", "vieneu"),
                                    ("☁️ Edge-TTS (Cloud, 2 giọng)", "edge-tts"),
                                ],
                                value="vieneu",
                                scale=2,
                            )

                        with gr.Row():
                            voice_select = gr.Dropdown(
                                label="🎙️ Giọng đọc",
                                choices=[
                                    ("🎤 Trúc Ly (nữ Bắc)", "Ly"),
                                    ("🎤 Bích Ngọc (nữ Bắc)", "Ngoc"),
                                    ("🎤 Thục Đoan (nữ Nam)", "Doan"),
                                    ("🎤 Thanh Bình (nam Bắc)", "Binh"),
                                    ("🎤 Phạm Tuyên (nam Bắc)", "Tuyen"),
                                    ("🎤 Xuân Vĩnh (nam Nam)", "Vinh"),
                                    ("🎤 Thái Sơn (nam Nam)", "Sơn"),
                                ],
                                value="Ly",
                                scale=2,
                            )
                            emotion_select = gr.Dropdown(
                                label="🎭 Phong cách đọc",
                                choices=[
                                    ("📖 Kể chuyện (Storytelling)", "storytelling"),
                                    ("💬 Tự nhiên (Natural)", "natural"),
                                ],
                                value="storytelling",
                                scale=2,
                            )

                        with gr.Row():
                            rate_select = gr.Dropdown(
                                label="⏩ Tốc độ (Edge-TTS)",
                                choices=[
                                    ("Rất chậm (-30%)", "-30%"),
                                    ("Chậm (-15%)", "-15%"),
                                    ("Bình thường", "+0%"),
                                    ("Nhanh (+15%)", "+15%"),
                                    ("Rất nhanh (+30%)", "+30%"),
                                ],
                                value="+0%",
                                scale=2,
                            )
                            video_mode_select = gr.Dropdown(
                                label="🎬 Chế độ video",
                                choices=[
                                    ("🖼️ Ken Burns (ảnh + zoom/pan)", "ken_burns"),
                                    ("🤖 Wan 2.1 AI Video", "wan21"),
                                ],
                                value="ken_burns",
                                scale=2,
                            )

                        with gr.Row():
                            image_style_select = gr.Dropdown(
                                label="🎨 Phong cách hình ảnh",
                                choices=get_style_choices(),
                                value="anime",
                                scale=4,
                            )

                        # Dynamic UI: switch voice choices based on engine
                        def update_voice_choices(engine):
                            if engine == "vieneu":
                                return gr.update(
                                    choices=[
                                        ("🎤 Trúc Ly (nữ Bắc)", "Ly"),
                                        ("🎤 Bích Ngọc (nữ Bắc)", "Ngoc"),
                                        ("🎤 Thục Đoan (nữ Nam)", "Doan"),
                                        ("🎤 Thanh Bình (nam Bắc)", "Binh"),
                                        ("🎤 Phạm Tuyên (nam Bắc)", "Tuyen"),
                                        ("🎤 Xuân Vĩnh (nam Nam)", "Vinh"),
                                        ("🎤 Thái Sơn (nam Nam)", "Sơn"),
                                    ],
                                    value="Ly",
                                )
                            else:
                                return gr.update(
                                    choices=[
                                        ("👩 Nữ (HoaiMy)", "vi-VN-HoaiMyNeural"),
                                        ("👨 Nam (NamMinh)", "vi-VN-NamMinhNeural"),
                                    ],
                                    value="vi-VN-HoaiMyNeural",
                                )

                        tts_engine_select.change(
                            fn=update_voice_choices,
                            inputs=[tts_engine_select],
                            outputs=[voice_select],
                        )

                        # Preview scenes
                        with gr.Accordion("👀 Xem trước phân đoạn", open=False):
                            with gr.Row():
                                min_chars = gr.Number(label="Min chars/scene", value=100, scale=1)
                                max_chars = gr.Number(label="Max chars/scene", value=1000, scale=1)
                                preview_btn = gr.Button("Xem trước", scale=1)
                            scene_preview = gr.Markdown(label="Preview")
                            scene_count = gr.Textbox(label="Tổng scene", interactive=False)

                            preview_btn.click(
                                fn=preview_scenes,
                                inputs=[story_input, min_chars, max_chars, image_style_select],
                                outputs=[scene_preview, scene_count],
                            )

                        # RUN button
                        run_btn = gr.Button(
                            "🚀 Bắt Đầu Tạo Video",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"],
                        )

                    # ── Right: Output ──
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Tiến trình")
                        log_output = gr.Textbox(
                            label="Log",
                            lines=15,
                            max_lines=30,
                            interactive=False,
                            elem_classes=["log-box"],
                        )

                        gr.Markdown("### 🔊 Audio Preview")
                        audio_preview = gr.Audio(label="Audio scene đầu tiên", type="filepath")

                        gr.Markdown("### 📺 Video kết quả")
                        video_output = gr.Video(label="Video hoàn chỉnh")
                        download_btn = gr.File(label="📥 Tải video")

                        gr.Markdown("### 🖼️ Hình ảnh minh họa")
                        gallery = gr.Gallery(
                            label="Anime illustrations",
                            columns=3,
                            rows=2,
                            height="auto",
                        )

                # Connect run button
                run_btn.click(
                    fn=run_full_pipeline,
                    inputs=[
                        story_input, project_name, tts_engine_select,
                        voice_select, rate_select, emotion_select,
                        video_mode_select, image_style_select,
                    ],
                    outputs=[log_output, audio_preview, video_output, download_btn, gallery],
                )

            # ═══════════════════════════════
            # TAB 2: TTS ONLY
            # ═══════════════════════════════
            with gr.Tab("🔊 TTS", id="tts"):
                gr.Markdown("### 🔊 Text-to-Speech (Chuyển văn bản → giọng nói)")
                gr.Markdown("Tạo audio từ text tiếng Việt — hỗ trợ VieNeu-TTS (7+ giọng) và Edge-TTS")

                with gr.Row():
                    with gr.Column(scale=3):
                        tts_text = gr.Textbox(
                            label="Nhập text tiếng Việt",
                            placeholder="Xin chào, đây là một ví dụ...",
                            lines=6,
                        )
                        with gr.Row():
                            tts_engine = gr.Dropdown(
                                label="🔊 Engine",
                                choices=[
                                    ("🇻🇳 VieNeu-TTS", "vieneu"),
                                    ("☁️ Edge-TTS", "edge-tts"),
                                ],
                                value="vieneu",
                            )
                            tts_voice = gr.Dropdown(
                                label="🎙️ Giọng đọc",
                                choices=[
                                    ("🎤 Trúc Ly (nữ Bắc)", "Ly"),
                                    ("🎤 Bích Ngọc (nữ Bắc)", "Ngoc"),
                                    ("🎤 Thục Đoan (nữ Nam)", "Doan"),
                                    ("🎤 Thanh Bình (nam Bắc)", "Binh"),
                                    ("🎤 Phạm Tuyên (nam Bắc)", "Tuyen"),
                                    ("🎤 Xuân Vĩnh (nam Nam)", "Vinh"),
                                    ("🎤 Thái Sơn (nam Nam)", "Sơn"),
                                ],
                                value="Ly",
                            )
                        with gr.Row():
                            tts_emotion = gr.Dropdown(
                                label="🎭 Phong cách",
                                choices=[
                                    ("📖 Kể chuyện", "storytelling"),
                                    ("💬 Tự nhiên", "natural"),
                                ],
                                value="storytelling",
                            )
                            tts_rate = gr.Dropdown(
                                label="⏩ Tốc độ (Edge-TTS)",
                                choices=[
                                    ("Chậm (-15%)", "-15%"),
                                    ("Bình thường", "+0%"),
                                    ("Nhanh (+15%)", "+15%"),
                                ],
                                value="+0%",
                            )

                        # Dynamic voice switching for TTS tab
                        def update_tts_voice(engine):
                            if engine == "vieneu":
                                return gr.update(
                                    choices=[
                                        ("🎤 Trúc Ly (nữ Bắc)", "Ly"),
                                        ("🎤 Bích Ngọc (nữ Bắc)", "Ngoc"),
                                        ("🎤 Thục Đoan (nữ Nam)", "Doan"),
                                        ("🎤 Thanh Bình (nam Bắc)", "Binh"),
                                        ("🎤 Phạm Tuyên (nam Bắc)", "Tuyen"),
                                        ("🎤 Xuân Vĩnh (nam Nam)", "Vinh"),
                                        ("🎤 Thái Sơn (nam Nam)", "Sơn"),
                                    ],
                                    value="Ly",
                                )
                            else:
                                return gr.update(
                                    choices=[
                                        ("👩 Nữ (HoaiMy)", "vi-VN-HoaiMyNeural"),
                                        ("👨 Nam (NamMinh)", "vi-VN-NamMinhNeural"),
                                    ],
                                    value="vi-VN-HoaiMyNeural",
                                )

                        tts_engine.change(
                            fn=update_tts_voice,
                            inputs=[tts_engine],
                            outputs=[tts_voice],
                        )

                        tts_btn = gr.Button(
                            "🎵 Tạo Audio",
                            variant="primary",
                            elem_classes=["primary-btn"],
                        )

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Kết quả", type="filepath")
                        tts_status = gr.Textbox(label="Trạng thái", interactive=False)

                tts_btn.click(
                    fn=run_tts_only,
                    inputs=[tts_text, tts_engine, tts_voice, tts_rate, tts_emotion],
                    outputs=[tts_audio_out, tts_status],
                )

            # ═══════════════════════════════
            # TAB 3: SETTINGS
            # ═══════════════════════════════
            with gr.Tab("⚙️ Cài đặt", id="settings"):
                gr.Markdown("### ⚙️ Cấu hình Pipeline")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🎨 Hình ảnh (SDXL Anime)")
                        img_model = gr.Textbox(
                            label="Model ID",
                            value="cagliostrolab/animagine-xl-3.1",
                        )
                        img_steps = gr.Slider(10, 50, value=25, step=1, label="Inference Steps")
                        img_guidance = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="Guidance Scale")
                        with gr.Row():
                            img_width = gr.Number(label="Width", value=1280)
                            img_height = gr.Number(label="Height", value=720)

                    with gr.Column():
                        gr.Markdown("#### 🎬 Video (Wan 2.1)")
                        wan_model = gr.Textbox(
                            label="Model ID",
                            value="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                        )
                        wan_steps = gr.Slider(10, 50, value=30, step=1, label="Inference Steps")
                        wan_guidance = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="Guidance Scale")
                        wan_frames = gr.Slider(17, 161, value=81, step=8, label="Num Frames")

                gr.Markdown("""
                ---
                #### 💡 GPU Info

                Dự án này được tối ưu cho **NVIDIA A5000 16GB VRAM**:

                | Model | VRAM sử dụng | Ghi chú |
                |-------|-------------|---------|
                | edge-tts | 0 GB | Cloud API miễn phí |
                | SDXL Anime | ~8-10 GB | CPU offload enabled |
                | Wan 2.1 1.3B | ~8-10 GB | CPU offload + VAE tiling |

                > ⚡ Các model AI được load/unload tuần tự để không vượt quá 16GB VRAM.
                """)

            # ═══════════════════════════════
            # TAB 4: ABOUT
            # ═══════════════════════════════
            with gr.Tab("ℹ️ Hướng dẫn", id="about"):
                gr.Markdown("""
                ## 📖 Text2Audio Studio - Hướng dẫn sử dụng

                ### Pipeline hoạt động như thế nào?

                ```
                📄 Truyện (Tiếng Việt)
                    ↓
                🔪 Chia thành các đoạn (scenes)
                    ↓
                🔊 Tạo audio cho mỗi đoạn (edge-tts)
                    ↓
                🎨 Tạo hình minh họa anime (SDXL)
                    ↓
                🎬 Tạo video từ hình (Ken Burns / Wan 2.1)
                    ↓
                🎞️ Ghép tất cả → Video YouTube
                ```

                ### Chế độ Video

                | Chế độ | Mô tả | Thời gian | VRAM |
                |--------|-------|-----------|------|
                | **Ken Burns** | Hiệu ứng zoom/pan trên ảnh tĩnh | Nhanh | 0 GB |
                | **Wan 2.1** | AI tạo video chuyển động | Chậm | ~10 GB |

                ### Tips

                1. **Bắt đầu với Ken Burns** để test pipeline nhanh
                2. **Mỗi scene** sẽ tạo 1 hình + 1 audio + 1 video clip
                3. **Video dài** (5 tiếng): Chia truyện thành nhiều phần, render từng phần
                4. Có thể **chỉnh sửa prompt** hình ảnh cho từng scene trong file `scenes.json`
                5. SDXL và Wan 2.1 được **load/unload tuần tự** để tiết kiệm VRAM

                ### Cấu trúc output
                ```
                output/
                └── {project_name}/
                    ├── scenes.json      # Metadata các scene
                    ├── audio/           # File audio từng scene
                    ├── images/          # Hình minh họa anime
                    ├── videos/          # Video clip từng scene
                    ├── subtitles/       # Phụ đề SRT
                    └── final/           # Video hoàn chỉnh
                ```
                """)

    return app


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = create_ui()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
