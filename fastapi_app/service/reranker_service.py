import importlib.util
import json
import os
from typing import List, Optional, Any, Dict

from PIL import Image

from ..config import get_app_config
from ..utils import get_logger

logger = get_logger(__name__)


def _load_qwen3_vl_reranker_from_checkpoint(model_path: str):
    """
    Load reranker from the checkpoint's bundled script:
    <model_path>/scripts/qwen3_vl_reranker.py::Qwen3VLReranker
    """
    script_path = os.path.join(model_path, "scripts", "qwen3_vl_reranker.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Missing reranker script: {script_path}")
    spec = importlib.util.spec_from_file_location("qwen3_vl_reranker_local", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Qwen3VLReranker"):
        raise RuntimeError("Qwen3VLReranker not found in checkpoint script")
    return mod.Qwen3VLReranker


class Qwen3VLRerankerService:
    """
    Thin wrapper around Qwen3-VL-Reranker-2B for (query_text, doc_image) scoring.
    """

    def __init__(self):
        self.cfg = get_app_config().reranker
        self._model = None

    def _lazy_init(self):
        if self._model is not None:
            return
        # If remote service is configured, do not init local model
        if getattr(self.cfg, "service_url", None):
            return
        model_path = getattr(self.cfg, "model_path", None)
        if not model_path:
            raise ValueError("reranker.model_path is required")

        kwargs: Dict[str, Any] = {}
        torch_dtype = (getattr(self.cfg, "torch_dtype", None) or "").strip()
        if torch_dtype:
            # pass through; script uses transformers.from_pretrained(**kwargs)
            import torch

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if torch_dtype.lower() in dtype_map:
                kwargs["torch_dtype"] = dtype_map[torch_dtype.lower()]

        attn_impl = getattr(self.cfg, "attn_implementation", None)
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        RerankerCls = _load_qwen3_vl_reranker_from_checkpoint(model_path)
        self._model = RerankerCls(model_name_or_path=model_path, **kwargs)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "enabled", False))

    def score_images(self, *, query_text: str, images: List[Image.Image]) -> List[float]:
        """
        Returns a list of relevance scores, higher is better.
        """
        if not images:
            return []
        if not query_text:
            query_text = ""
        service_url = getattr(self.cfg, "service_url", None)
        if service_url:
            return self._score_images_remote(query_text=query_text, images=images)

        self._lazy_init()
        assert self._model is not None

        docs = [{"image": img} for img in images]
        inputs = {
            "instruction": getattr(self.cfg, "instruction", None) or "Retrieve the most relevant candidates.",
            "query": {"text": query_text},
            "documents": docs,
            "fps": 1.0,
        }
        scores = self._model.process(inputs)  # type: ignore[attr-defined]
        return [float(s) for s in (scores or [])]

    def _score_images_remote(self, *, query_text: str, images: List[Image.Image]) -> List[float]:
        """
        Call remote reranker service. Note: remote service expects base64-encoded images.
        """
        import base64
        import io
        import urllib.request

        url = str(getattr(self.cfg, "service_url")).rstrip("/") + "/score_images"
        payload = {
            "instruction": getattr(self.cfg, "instruction", None),
            "query_text": query_text,
            "images_b64": [],
        }
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            payload["images_b64"].append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = int(getattr(self.cfg, "service_timeout", 120))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw) if raw else {}
        scores = data.get("scores") or []
        return [float(s) for s in scores]


_reranker_singleton: Optional[Qwen3VLRerankerService] = None


def get_reranker() -> Qwen3VLRerankerService:
    global _reranker_singleton
    if _reranker_singleton is None:
        _reranker_singleton = Qwen3VLRerankerService()
    return _reranker_singleton

