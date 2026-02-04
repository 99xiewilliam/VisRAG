import base64
import io
import json
import os
import urllib.request
from typing import List, Optional, Sequence, Any, Dict

import numpy as np
from PIL import Image

from ..config import get_app_config
from ..utils import get_logger

logger = get_logger(__name__)


def _l2_normalize(vecs: List[List[float]]) -> List[List[float]]:
    if not vecs:
        return []
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arr = arr / norms
    return arr.tolist()


def _qwen3_vl_prompt(*, text: str, include_image: bool, instruction: str = "Represent the user's input.") -> str:
    """
    vLLM 官方示例对 Qwen3-VL-Embedding 的 prompt 需要用特殊 token 格式（而不是 chat_template）。
    """
    image_placeholder = "<<|vision_start|>><<|image_pad|>><<|vision_end|>>" if include_image else ""
    user_content = f"{image_placeholder}{text or ''}"
    return (
        f"<<|im_start|>>system\n{instruction}<<|im_end|>>\n"
        f"<<|im_start|>>user\n{user_content}<<|im_end|>>\n"
        f"<<|im_start|>>assistant\n"
    )


class _VllmQwenVLEmbedder:
    def __init__(self):
        cfg = get_app_config().vl_embedding
        self.model_path = cfg.model_path
        self.batch_size = cfg.batch_size
        self.max_length = cfg.max_length
        self.embedding_dim = getattr(cfg, "embedding_dim", None)
        self.dtype = getattr(cfg, "dtype", "bfloat16")
        self._llm = None
        self._pooling_params = None
        self._dim: Optional[int] = None

    def _lazy_init(self):
        if self._llm is not None:
            return
        from vllm import LLM, EngineArgs, PoolingParams

        # vLLM 0.8.x 没有 runner="pooling"，使用 task="embed" 来启用 embedding 接口
        engine_args = EngineArgs(
            model=self.model_path,
            task="embed",
            dtype=self.dtype,
            trust_remote_code=True,
            max_model_len=int(self.max_length),
            limit_mm_per_prompt={"image": 1},
        )
        self._llm = LLM(**vars(engine_args))
        if self.embedding_dim:
            self._pooling_params = PoolingParams(dimensions=int(self.embedding_dim))

    @property
    def dim(self) -> int:
        if self._dim is None:
            vec = self.embed_texts(["probe"])[0]
            self._dim = len(vec)
        return int(self._dim)

    def _embed_requests(self, requests: List[Any]) -> List[List[float]]:
        self._lazy_init()
        assert self._llm is not None
        kwargs: Dict[str, Any] = {"use_tqdm": False}
        if self._pooling_params is not None:
            kwargs["pooling_params"] = self._pooling_params
        outputs = self._llm.embed(requests, **kwargs)
        vecs = [out.outputs.embedding for out in outputs]
        return _l2_normalize(vecs)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            prompts = [_qwen3_vl_prompt(text=t, include_image=False) for t in batch]
            out.extend(self._embed_requests(prompts))
        return out

    def embed_images(self, images: Sequence[Image.Image]) -> List[List[float]]:
        if not images:
            return []
        out: List[List[float]] = []
        for i in range(0, len(images), self.batch_size):
            batch = list(images[i : i + self.batch_size])
            reqs = []
            for img in batch:
                reqs.append(
                    {
                        "prompt": _qwen3_vl_prompt(text="", include_image=True),
                        "multi_modal_data": {"image": img},
                    }
                )
            out.extend(self._embed_requests(reqs))
        return out

    def embed_query(self, text: Optional[str] = None, image: Optional[Image.Image] = None) -> List[float]:
        if image is not None and text is not None:
            req = {
                "prompt": _qwen3_vl_prompt(text=text, include_image=True),
                "multi_modal_data": {"image": image},
            }
            return self._embed_requests([req])[0]
        if image is not None:
            return self.embed_images([image])[0]
        if text is not None:
            return self.embed_texts([text])[0]
        raise ValueError("Query requires text and/or image")


class _HfScriptQwenVLEmbedder:
    """
    Fallback：使用 checkpoint 自带的 `scripts/qwen3_vl_embedding.py::Qwen3VLEmbedder`。
    这条链路与模型仓库 README 的用法一致（pooling_last + normalize + 多模态）。
    """

    def __init__(self):
        import torch

        cfg = get_app_config().vl_embedding
        self.model_path = cfg.model_path
        self.batch_size = cfg.batch_size
        self.max_length = cfg.max_length
        self.embedding_dim = getattr(cfg, "embedding_dim", None)
        self._torch = torch
        self._embedder = None
        self._dim: Optional[int] = None

    def _lazy_init(self):
        if self._embedder is not None:
            return
        import importlib.util

        script_path = os.path.join(self.model_path, "scripts", "qwen3_vl_embedding.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Missing embedding script: {script_path}")
        spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_local", script_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "Qwen3VLEmbedder"):
            raise RuntimeError("Qwen3VLEmbedder not found in checkpoint script")
        self._embedder = mod.Qwen3VLEmbedder(model_name_or_path=self.model_path, max_length=int(self.max_length))

    @property
    def dim(self) -> int:
        if self._dim is None:
            vec = self.embed_texts(["probe"])[0]
            self._dim = len(vec)
        return int(self._dim)

    def _post(self, emb) -> List[List[float]]:
        torch = self._torch
        if isinstance(emb, torch.Tensor):
            vecs = emb.detach().cpu().float()
        else:
            vecs = torch.tensor(emb, dtype=torch.float32)
        if self.embedding_dim:
            vecs = vecs[:, : int(self.embedding_dim)]
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
        return vecs.tolist()

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        self._lazy_init()
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            emb = self._embedder.process([{"text": t} for t in batch], normalize=True)
            out.extend(self._post(emb))
        return out

    def embed_images(self, images: Sequence[Image.Image]) -> List[List[float]]:
        if not images:
            return []
        self._lazy_init()
        out: List[List[float]] = []
        for i in range(0, len(images), self.batch_size):
            batch = list(images[i : i + self.batch_size])
            emb = self._embedder.process([{"image": img} for img in batch], normalize=True)
            out.extend(self._post(emb))
        return out

    def embed_query(self, text: Optional[str] = None, image: Optional[Image.Image] = None) -> List[float]:
        self._lazy_init()
        if image is not None and text is not None:
            emb = self._embedder.process([{"text": text, "image": image}], normalize=True)
            return self._post(emb)[0]
        if image is not None:
            return self.embed_images([image])[0]
        if text is not None:
            return self.embed_texts([text])[0]
        raise ValueError("Query requires text and/or image")


class _RemoteQwenVLEmbedder:
    def __init__(self):
        cfg = get_app_config().vl_embedding
        self.service_url = (cfg.service_url or "").rstrip("/")
        self.timeout = int(getattr(cfg, "service_timeout", 60))
        self.batch_size = int(cfg.batch_size)
        self.embedding_dim = getattr(cfg, "embedding_dim", None)
        self._dim: Optional[int] = None

    @staticmethod
    def _encode_image(image: Image.Image) -> Dict[str, str]:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"data": payload, "mime": "image/png"}

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.service_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)

    @property
    def dim(self) -> int:
        if self._dim is None:
            vec = self.embed_texts(["probe"])[0]
            self._dim = len(vec)
        return int(self._dim)

    def _postprocess(self, vecs: List[List[float]]) -> List[List[float]]:
        if self.embedding_dim:
            vecs = [v[: int(self.embedding_dim)] for v in vecs]
        return _l2_normalize(vecs)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            resp = self._post_json("/embed_texts", {"texts": batch})
            out.extend(resp.get("embeddings", []))
        return self._postprocess(out)

    def embed_images(self, images: Sequence[Image.Image]) -> List[List[float]]:
        if not images:
            return []
        out: List[List[float]] = []
        for i in range(0, len(images), self.batch_size):
            batch = list(images[i : i + self.batch_size])
            payload = {"images": [self._encode_image(img) for img in batch]}
            resp = self._post_json("/embed_images", payload)
            out.extend(resp.get("embeddings", []))
        return self._postprocess(out)

    def embed_query(self, text: Optional[str] = None, image: Optional[Image.Image] = None) -> List[float]:
        if text is None and image is None:
            raise ValueError("Query requires text and/or image")
        payload: Dict[str, Any] = {}
        if text is not None:
            payload["text"] = text
        if image is not None:
            payload["image"] = self._encode_image(image)
        resp = self._post_json("/embed_query", payload)
        vec = resp.get("embedding", [])
        return self._postprocess([vec])[0]


class QwenVLEmbedder:
    """
    默认优先走 vLLM（llm.embed）；如果 vLLM 走不通/不支持多模态 embedding，则回退到 HF 官方脚本实现。
    """

    def __init__(self):
        self._primary = None
        self._fallback = _HfScriptQwenVLEmbedder()
        cfg = get_app_config().vl_embedding
        if cfg.service_url:
            self._primary = _RemoteQwenVLEmbedder()
        else:
            try:
                import vllm  # noqa: F401
                self._primary = _VllmQwenVLEmbedder()
            except Exception:
                logger.warning("vLLM not available, using HF-script fallback for embeddings")

    @property
    def dim(self) -> int:
        if self._primary is not None:
            try:
                return self._primary.dim
            except Exception:
                logger.exception("Primary embedder failed; falling back")
        return self._fallback.dim

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if self._primary is not None:
            try:
                return self._primary.embed_texts(texts)
            except Exception:
                logger.exception("Primary embed_texts failed; falling back")
        return self._fallback.embed_texts(texts)

    def embed_images(self, images: Sequence[Image.Image]) -> List[List[float]]:
        if self._primary is not None:
            try:
                return self._primary.embed_images(images)
            except Exception:
                logger.exception("Primary embed_images failed; falling back")
        return self._fallback.embed_images(images)

    def embed_query(self, text: Optional[str] = None, image: Optional[Image.Image] = None) -> List[float]:
        if self._primary is not None:
            try:
                return self._primary.embed_query(text=text, image=image)
            except Exception:
                logger.exception("Primary embed_query failed; falling back")
        return self._fallback.embed_query(text=text, image=image)
