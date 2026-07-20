"""Text Rewriter - Paraphrase stories to avoid YouTube copyright.

Supports two LLM providers:
- Gemini API (Google, recommended for speed)
- Ollama (local, uses Qwen2.5:7b for Vietnamese)

IMPORTANT: The rewriter must keep ALL original content, details, and dialogue.
It only changes sentence structure and word choice. It does NOT summarize.
"""
import os
import re
import logging
from typing import List, Optional

from src.config import RewriterConfig

logger = logging.getLogger(__name__)

# System prompt — strict rules to prevent summarization
REWRITE_SYSTEM_PROMPT = """Bạn là một biên tập viên văn học. Nhiệm vụ của bạn là VIẾT LẠI đoạn văn sau \
bằng cách diễn đạt KHÁC nhưng giữ NGUYÊN TOÀN BỘ nội dung, chi tiết, hội thoại và ý nghĩa.

QUY TẮC BẮT BUỘC:
1. KHÔNG được tóm tắt, lược bỏ hay rút gọn bất kỳ chi tiết nào
2. KHÔNG được thêm thông tin mới không có trong bản gốc
3. Giữ nguyên toàn bộ lời thoại nhân vật (có thể đổi từ "nói" sang "thốt lên", "lên tiếng", v.v.)
4. Thay đổi cấu trúc câu: đảo vị trí mệnh đề, đổi câu chủ động ↔ bị động
5. Sử dụng từ đồng nghĩa thay thế khi có thể
6. Có thể thêm 1 câu dẫn dắt ngắn ở đầu đoạn (giọng người kể chuyện)
7. Độ dài bản viết lại phải TƯƠNG ĐƯƠNG bản gốc (chênh lệch ≤ 15%)
8. Giữ nguyên ngôn ngữ tiếng Việt

Chỉ trả về đoạn văn đã viết lại, không giải thích."""


def _split_for_rewrite(text: str, max_chars: int = 2000) -> List[str]:
    """Split text into chunks suitable for LLM rewriting.

    Splits on paragraph boundaries, keeping chunks under max_chars.
    """
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if current_chunk and len(current_chunk) + len(para) + 2 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


class TextRewriter:
    """Rewrite story text to avoid copyright issues using LLM."""

    def __init__(self, config: Optional[RewriterConfig] = None):
        """Initialize rewriter.

        Args:
            config: RewriterConfig instance. If None, uses defaults.
        """
        self.config = config or RewriterConfig()
        self.provider = self.config.provider
        self._gemini_client = None
        self._gemini_model = None

    def _init_gemini(self):
        """Lazy-init Gemini API client."""
        if self._gemini_client is not None:
            return

        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package required for Gemini. "
                "Install: pip install google-genai"
            )

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Please set the environment variable: export GEMINI_API_KEY=\"your_key\""
            )

        self._gemini_client = genai.Client(api_key=api_key)
        self._gemini_model = self.config.gemini_model
        logger.info(f"Initialized Gemini client (model={self._gemini_model})")

    def _call_gemini(self, user_prompt: str) -> str:
        """Call Gemini API for text rewriting."""
        self._init_gemini()

        try:
            response = self._gemini_client.models.generate_content(
                model=self._gemini_model,
                contents=user_prompt,
                config={
                    "system_instruction": REWRITE_SYSTEM_PROMPT,
                    "temperature": 0.7,
                    "max_output_tokens": 4096,
                },
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _call_ollama(self, user_prompt: str) -> str:
        """Call Ollama local API for text rewriting."""
        import json
        import urllib.request
        import urllib.error

        url = f"{self.config.ollama_base_url}/api/generate"
        payload = {
            "model": self.config.ollama_model,
            "system": REWRITE_SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 4096,
            },
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            logger.error(f"Ollama connection error: {e}")
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.ollama_base_url}. "
                f"Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def rewrite_chunk(self, text: str) -> str:
        """Rewrite a single text chunk using the configured LLM.

        Args:
            text: Vietnamese text to rewrite

        Returns:
            Rewritten text with same meaning but different wording
        """
        user_prompt = f"Viết lại đoạn văn sau:\n\n{text}"

        if self.provider == "gemini":
            result = self._call_gemini(user_prompt)
        elif self.provider == "ollama":
            result = self._call_ollama(user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Validate: rewritten text should be within 15% of original length
        original_len = len(text)
        rewritten_len = len(result)
        ratio = rewritten_len / max(original_len, 1)

        if ratio < 0.5:
            logger.warning(
                f"Rewrite too short ({rewritten_len} vs {original_len} chars, "
                f"ratio={ratio:.2f}). Using original text."
            )
            return text

        if ratio < 0.85 or ratio > 1.3:
            logger.warning(
                f"Rewrite length differs significantly "
                f"({rewritten_len} vs {original_len} chars, ratio={ratio:.2f})"
            )

        return result

    def rewrite_story(
        self,
        full_text: str,
        progress_callback=None,
    ) -> str:
        """Rewrite entire story in chunks, preserving paragraph structure.

        Args:
            full_text: Full story text in Vietnamese
            progress_callback: Optional callback(current, total, message)

        Returns:
            Fully rewritten story text
        """
        chunks = _split_for_rewrite(full_text, max_chars=2000)
        total = len(chunks)

        logger.info(
            f"Rewriting story: {len(full_text)} chars, "
            f"{total} chunks (provider={self.provider})"
        )

        rewritten_chunks = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(
                    i, total,
                    f"🔄 Đang viết lại đoạn {i+1}/{total}..."
                )

            try:
                rewritten = self.rewrite_chunk(chunk)
                rewritten_chunks.append(rewritten)
                logger.debug(
                    f"Chunk {i+1}/{total}: "
                    f"{len(chunk)} → {len(rewritten)} chars"
                )
            except Exception as e:
                logger.error(f"Failed to rewrite chunk {i+1}: {e}")
                # Fall back to original text for this chunk
                rewritten_chunks.append(chunk)

        result = "\n\n".join(rewritten_chunks)

        if progress_callback:
            progress_callback(total, total, "✅ Viết lại hoàn tất")

        # Log summary
        original_len = len(full_text)
        rewritten_len = len(result)
        logger.info(
            f"Rewrite complete: {original_len} → {rewritten_len} chars "
            f"(ratio={rewritten_len/max(original_len,1):.2f})"
        )

        return result
