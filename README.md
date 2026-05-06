# Text2Audio 🎬✨

**Chuyển đổi truyện tiếng Việt thành video YouTube tự động**

Pipeline: Text → Audio (TTS) → Anime Images (SDXL) → Video → YouTube

## Tính năng

- 🔊 **Vietnamese TTS**: edge-tts với giọng đọc tự nhiên (HoaiMy/NamMinh)
- 🎨 **Anime Illustrations**: SDXL (animagine-xl-3.1) tạo hình minh họa anime
- 🎬 **Video Generation**: Ken Burns effect + Wan 2.1 AI video
- 🎞️ **Auto Merge**: Tự động ghép audio + video + subtitle
- 🌐 **Web UI**: Giao diện Gradio dark theme, dễ sử dụng
- ⚡ **GPU Optimized**: Tối ưu cho NVIDIA A5000 16GB VRAM

## Cài đặt

### 1. Tạo conda environment
```bash
conda create -n text2audio python=3.10 -y
conda activate text2audio
```

### 2. Cài PyTorch (CUDA)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Cài dependencies
```bash
pip install -r requirements.txt
```

### 4. Cài FFmpeg
```bash
sudo apt install ffmpeg
```

## Sử dụng

### Web UI (khuyên dùng)
```bash
python app.py
# Mở http://localhost:7860
```

### Command Line
```bash
python scripts/run_pipeline.py \
  --story stories/sample_story.txt \
  --project my_first_video \
  --voice vi-VN-HoaiMyNeural \
  --video-mode ken_burns
```

## Cấu trúc

```
Text2Audio/
├── app.py                 # Web UI (Gradio)
├── config.yaml            # Cấu hình
├── requirements.txt       # Dependencies
├── src/
│   ├── config.py          # Config loader
│   ├── text_processor.py  # Chia đoạn truyện
│   ├── tts_engine.py      # Text-to-Speech
│   ├── image_generator.py # SDXL anime images
│   ├── video_generator.py # Ken Burns + Wan 2.1
│   ├── video_merger.py    # Merge audio + video
│   ├── subtitle_generator.py
│   └── pipeline.py        # Pipeline orchestrator
├── stories/               # Input text files
├── output/                # Generated output
└── scripts/               # CLI scripts
```

## Models

| Component | Model | VRAM |
|-----------|-------|------|
| TTS | edge-tts (cloud) | 0 GB |
| Images | animagine-xl-3.1 | ~8-10 GB |
| Video | Wan 2.1 T2V 1.3B | ~8-10 GB |

## License

Apache License 2.0
