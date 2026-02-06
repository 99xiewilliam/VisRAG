import base64
import importlib.util
import io
import os
import tempfile
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from .config import get_app_config


def _load_qwen3_vl_reranker_from_checkpoint(model_path: str):
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


def _get_model_path() -> str:
    return os.environ.get("RERANKER_MODEL_PATH") or get_app_config().reranker.model_path


def _get_torch_dtype_name() -> str:
    return os.environ.get("RERANKER_TORCH_DTYPE") or get_app_config().reranker.torch_dtype or "bfloat16"


def _get_attn_impl() -> Optional[str]:
    return os.environ.get("RERANKER_ATTN_IMPL") or get_app_config().reranker.attn_implementation or None


_model = None
_backend: Optional[str] = None
_vllm_llm = None
_vllm_chat_template: Optional[str] = None


def _get_backend() -> str:
    # auto | transformers | vllm
    return (os.environ.get("RERANKER_BACKEND") or get_app_config().reranker.backend or "auto").strip().lower()


def _get_vllm_dtype() -> str:
    return (os.environ.get("RERANKER_VLLM_DTYPE") or get_app_config().reranker.vllm_dtype or _get_torch_dtype_name() or "bfloat16").strip()


def _get_vllm_max_model_len() -> int:
    """
    有些 checkpoint 的 config 会带超大 max seq len（例如 262144），
    vLLM 会据此预留巨大的 KV cache，容易 OOM。对 rerank 没必要这么大，
    默认限制到 36248（你也可用环境变量覆盖）。
    """
    raw = (os.environ.get("RERANKER_VLLM_MAX_MODEL_LEN") or "").strip()
    if raw:
        try:
            return int(raw)
        except Exception:
            pass
    return int(get_app_config().reranker.vllm_max_model_len or 36248)


def _get_vllm_gpu_memory_utilization() -> float:
    raw = (os.environ.get("RERANKER_VLLM_GPU_MEMORY_UTILIZATION") or "").strip()
    if raw:
        try:
            return float(raw)
        except Exception:
            pass
    return float(get_app_config().reranker.vllm_gpu_memory_utilization or 0.5)


def _get_vllm_tmp_dir() -> Optional[str]:
    return os.environ.get("RERANKER_TMP_DIR") or get_app_config().reranker.vllm_tmp_dir


def _get_vllm_allowed_local_media_path() -> str:
    """
    vLLM 默认禁止读取 file:// 本地多模态文件；需显式开启允许目录。
    我们只允许临时图片落盘目录，避免放开整个文件系统。
    """
    cand = _get_vllm_tmp_dir()
    if cand and os.path.isdir(cand):
        return os.path.abspath(cand)
    if os.path.isdir("/dev/shm"):
        return "/dev/shm"
    return "/tmp"


def _get_template_path(model_path: str) -> Optional[str]:
    # vLLM score API needs a score-specific template (NOT the checkpoint chat_template.jinja).
    p = os.environ.get("RERANKER_TEMPLATE_PATH") or get_app_config().reranker.vllm_template_path
    if p and os.path.exists(p):
        return p
    # default: use bundled vLLM score template
    bundled = os.path.join(os.path.dirname(__file__), "templates", "qwen3_vl_reranker.jinja")
    if os.path.exists(bundled):
        return bundled
    return None


def _lazy_init():
    global _model, _backend, _vllm_llm, _vllm_chat_template
    if _model is not None or _vllm_llm is not None:
        return
    model_path = _get_model_path()
    backend = _get_backend()

    # Prefer vLLM if available and requested/auto
    if backend in ("auto", "vllm"):
        try:
            from vllm import LLM, EngineArgs  # type: ignore

            engine_args = EngineArgs(
                model=model_path,
                runner="pooling",
                dtype=_get_vllm_dtype(),
                trust_remote_code=True,
                gpu_memory_utilization=_get_vllm_gpu_memory_utilization(),
                max_model_len=int(_get_vllm_max_model_len()),
                allowed_local_media_path=_get_vllm_allowed_local_media_path(),
                hf_overrides={
                    "architectures": ["Qwen3VLForSequenceClassification"],
                    "classifier_from_token": ["no", "yes"],
                    "is_original_qwen3_reranker": True,
                },
            )
            _vllm_llm = LLM(**vars(engine_args))
            tp = _get_template_path(model_path)
            _vllm_chat_template = open(tp, "r", encoding="utf-8").read() if tp else None
            _backend = "vllm"
            return
        except Exception:
            _vllm_llm = None
            _vllm_chat_template = None

    # transformers fallback
    kwargs: Dict[str, Any] = {}
    try:
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        dtype_name = _get_torch_dtype_name().lower().strip()
        if dtype_name in dtype_map:
            kwargs["torch_dtype"] = dtype_map[dtype_name]
    except Exception:
        pass

    attn_impl = _get_attn_impl()
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    RerankerCls = _load_qwen3_vl_reranker_from_checkpoint(model_path)
    _model = RerankerCls(model_name_or_path=model_path, **kwargs)
    _backend = "transformers"


class ScoreImagesRequest(BaseModel):
    query_text: str
    images_b64: List[str]
    instruction: Optional[str] = None


class ScoreImagesResponse(BaseModel):
    scores: List[float]


class HealthResponse(BaseModel):
    ok: bool
    backend: str


app = FastAPI(title="VisRAG Reranker Service")


@app.on_event("startup")
def _startup_preload():
    _lazy_init()


@app.get("/health", response_model=HealthResponse)
def health():
    _lazy_init()
    return HealthResponse(ok=True, backend=_backend or "unknown")


def _format_document_to_score_param(doc_dict: Dict[str, Any]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    text = doc_dict.get("text")
    image = doc_dict.get("image")

    if text:
        content.append({"type": "text", "text": text})

    if image:
        image_url = image
        if isinstance(image, str) and not image.startswith(("http", "https", "oss", "file://")):
            abs_image_path = os.path.abspath(image)
            image_url = "file://" + abs_image_path
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    if not content:
        content.append({"type": "text", "text": ""})

    return {"content": content}


@app.post("/score_images", response_model=ScoreImagesResponse)
def score_images(req: ScoreImagesRequest):
    _lazy_init()
    images: List[Image.Image] = []
    for b64 in req.images_b64:
        raw = base64.b64decode(b64.encode("utf-8"))
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        images.append(img)

    # vLLM backend
    if _backend == "vllm" and _vllm_llm is not None:
        tmp_dir = _get_vllm_allowed_local_media_path()
        tmp_files: List[str] = []
        try:
            for img in images:
                fd, path = tempfile.mkstemp(prefix="visrag_rerank_", suffix=".png", dir=tmp_dir)
                os.close(fd)
                img.save(path, format="PNG")
                tmp_files.append(path)

            scores: List[float] = []
            for path in tmp_files:
                doc_param = _format_document_to_score_param({"image": path})
                outputs = _vllm_llm.score(req.query_text, doc_param, chat_template=_vllm_chat_template)
                score = float(outputs[0].outputs.score)
                scores.append(score)
            return ScoreImagesResponse(scores=scores)
        finally:
            for p in tmp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass

    # transformers backend
    assert _model is not None
    docs = [{"image": img} for img in images]
    inputs = {
        "instruction": req.instruction or "Retrieve the most relevant candidates.",
        "query": {"text": req.query_text},
        "documents": docs,
        "fps": 1.0,
    }
    scores = _model.process(inputs)
    return ScoreImagesResponse(scores=[float(s) for s in (scores or [])])
