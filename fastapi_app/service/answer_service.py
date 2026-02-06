import base64
import json
import os
from typing import Optional

from ..config import get_app_config
from ..prompts import get_prompts
from ..utils import get_logger
from .ocr_service import OCRService

logger = get_logger(__name__)


class AnswerService:
    """
    Abstraction for the final answer generator.
    Default backend uses OCR2; can be extended with other backends.
    """

    def __init__(self, ocr: OCRService):
        self.cfg = get_app_config().answer_generator
        self.ocr = ocr

    def generate_from_image(
        self,
        *,
        image_path: str,
        question: Optional[str],
        context_text: Optional[str],
    ) -> str:
        backend = (getattr(self.cfg, "backend", "ocr2") or "ocr2").lower().strip()
        if backend in ("ocr2", "ocr", "deepseek_ocr2"):
            return self.ocr.run(image_path, question=question, context_text=context_text)
        if backend in ("openai_compat", "openai", "gpt"):
            return self._openai_compat(image_path=image_path, question=question, context_text=context_text)
        if backend in ("none", "disabled"):
            return ""
        if backend in ("local_vl", "qwen_vl"):
            return self._local_vl(image_path=image_path, question=question, context_text=context_text)

        logger.warning(f"Unknown answer_generator.backend={backend}; falling back to ocr2")
        return self.ocr.run(image_path, question=question, context_text=context_text)

    def _local_vl(self, *, image_path: str, question: Optional[str], context_text: Optional[str]) -> str:
        """
        Call a local VLM served by an OpenAI-compatible server (e.g. vLLM `serve`).
        Config keys are under `answer_generator.local_vl` in config.yaml.
        """
        api_key = (getattr(self.cfg, "local_vl_api_key", None) or os.environ.get("OPENAI_API_KEY") or "").strip()
        base_url = (getattr(self.cfg, "local_vl_base_url", None) or "").strip()
        model = (getattr(self.cfg, "local_vl_model", None) or "").strip()
        if not base_url or not model:
            logger.warning("local_vl missing base_url/model; returning empty")
            return ""

        # Read image as base64 data URL
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        prompts = get_prompts()
        system = (
            (prompts.get("answer_generator", {}) or {}).get("openai_compat", {}) or {}
        ).get("system") or (
            "You are a careful document QA assistant.\n"
            "Answer the question accurately using the provided context and image.\n"
            "Rules:\n"
            "- Do NOT output your reasoning steps.\n"
            "- Output ONLY the final answer.\n"
            "- Be concise: 1 sentence (or a single number + unit).\n"
            "- No headings, no markdown, no citations, no extra context.\n"
        )

        user_parts = []
        if context_text:
            user_parts.append({"type": "text", "text": f"Context:\n{context_text}"})
        user_parts.append({"type": "text", "text": f"Question: {question or ''}"})
        user_parts.append({"type": "image_url", "image_url": {"url": data_url}})

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_parts},
            ],
            "max_tokens": int(getattr(self.cfg, "local_vl_max_tokens", 256)),
            "temperature": float(getattr(self.cfg, "local_vl_temperature", 0.0)),
        }

        try:
            import urllib.request
            import urllib.error

            url = base_url.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=int(getattr(self.cfg, "local_vl_timeout", 60))) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            content = (((data.get("choices") or [])[0] or {}).get("message") or {}).get("content")
            return (content or "").strip()
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            logger.error(f"local_vl HTTPError {e.code}: {body[:1000]}")
            return ""
        except Exception:
            logger.exception("local_vl answer generation failed")
            return ""

    def _openai_compat(self, *, image_path: str, question: Optional[str], context_text: Optional[str]) -> str:
        api_key = (self.cfg.openai_api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        base_url = (self.cfg.openai_base_url or os.environ.get("OPENAI_BASE_URL") or "").strip()
        if not api_key or not base_url:
            logger.warning("openai_compat backend missing api_key/base_url; returning empty")
            return ""

        # Read image as base64 data URL
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"

        prompts = get_prompts()
        system = (
            (prompts.get("answer_generator", {}) or {}).get("openai_compat", {}) or {}
        ).get("system") or (
            "You are a careful document QA assistant.\n"
            "Answer the question accurately using the provided context and image.\n"
            "Rules:\n"
            "- Do NOT output your reasoning steps.\n"
            "- Output ONLY the final answer.\n"
            "- Be concise: 1 sentence (or a single number + unit).\n"
            "- No headings, no markdown, no citations, no extra context.\n"
        )

        user_parts = []
        if context_text:
            user_parts.append({"type": "text", "text": f"Context:\n{context_text}"})
        user_parts.append({"type": "text", "text": f"Question: {question or ''}"})
        user_parts.append({"type": "image_url", "image_url": {"url": data_url}})

        payload = {
            "model": self.cfg.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_parts},
            ],
            "max_tokens": int(self.cfg.openai_max_tokens),
            "temperature": float(self.cfg.openai_temperature),
        }

        try:
            import urllib.request
            import urllib.error

            url = base_url.rstrip("/") + "/chat/completions"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=int(self.cfg.openai_timeout)) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("error"):
                logger.error(f"openai_compat upstream error: {str(data.get('error'))[:1000]}")
                return ""
            choices = (data.get("choices") or []) if isinstance(data, dict) else []
            if not choices:
                logger.error(f"openai_compat returned empty choices: {raw[:1000]}")
                return ""
            content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
            if not content:
                logger.error(f"openai_compat returned empty content: {raw[:1000]}")
                return ""
            return content
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            logger.error(f"openai_compat HTTPError {e.code}: {body[:1000]}")
            return ""
        except Exception:
            logger.exception("openai_compat answer generation failed")
            return ""
