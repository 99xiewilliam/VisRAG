import os
from typing import Any, Dict

import yaml

from .config import get_app_config
from .utils import get_logger

logger = get_logger(__name__)

_PROMPTS_CACHE: Dict[str, Any] | None = None


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _default_prompts() -> Dict[str, Any]:
    return {
        "query": {
            "text_only": {
                "system": (
                    "You are a careful document QA assistant.\n"
                    "Your job is to answer the question accurately using ONLY the provided context.\n"
                    "Rules:\n"
                    "- Do NOT output your reasoning steps.\n"
                    "- Output ONLY the final answer.\n"
                    "- Use ONLY the provided context.\n"
                    "- Be concise: 1 sentence (or a single number + unit).\n"
                    "- No headings, no markdown, no citations, no extra context.\n"
                ),
                "user": (
                    "Answer the question using ONLY the context.\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n"
                    "Final answer (short):"
                ),
            },
            "vision": {
                "system": (
                    "You are a careful document QA assistant.\n"
                    "Your job is to answer the question accurately using the provided context and image.\n"
                    "Rules:\n"
                    "- Do NOT output your reasoning steps.\n"
                    "- Output ONLY the final answer.\n"
                    "- Be concise: 1 sentence (or a single number + unit).\n"
                    "- No headings, no markdown, no citations, no extra context.\n"
                ),
                "user": (
                    "Answer the question using the given context and the image.\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n"
                    "Final answer (short):"
                ),
            },
            "vision_only": {
                "system": (
                    "You are a careful document QA assistant.\n"
                    "Your job is to answer the question accurately using ONLY the image.\n"
                    "Rules:\n"
                    "- Do NOT output your reasoning steps.\n"
                    "- Output ONLY the final answer.\n"
                    "- Be concise: 1 sentence (or a single number + unit).\n"
                    "- No headings, no markdown, no citations, no extra context.\n"
                ),
                "user": (
                    "Answer the question using ONLY the image.\n"
                    "Question: {question}\n"
                    "Final answer (short):"
                ),
            },
        },
        "answer_generator": {
            "openai_compat": {
                "system": (
                    "You are a careful document QA assistant.\n"
                    "Answer the question accurately using the provided context and image.\n"
                    "Rules:\n"
                    "- Do NOT output your reasoning steps.\n"
                    "- Output ONLY the final answer.\n"
                    "- Be concise: 1 sentence (or a single number + unit).\n"
                    "- No headings, no markdown, no citations, no extra context.\n"
                )
            }
        },
        "ocr": {
            "prompt_template": (
                "<image>\n"
                "<|grounding|>\n"
                "You are a helpful document QA assistant. "
                "Use the image and the provided context to answer the question. "
                "If the context is insufficient, answer based on the image.\n"
                "Question: {question}\n"
                "Context:\n{context}\n"
                "Answer:"
            )
        },
    }


def get_prompts() -> Dict[str, Any]:
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE

    data = _default_prompts()
    cfg = get_app_config()
    path = cfg.prompts_path or os.path.join(_project_root(), "prompts.yaml")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                data = _deep_merge(data, loaded)
        except Exception:
            logger.exception("Failed to load prompts file; using defaults")
    _PROMPTS_CACHE = data
    return data
