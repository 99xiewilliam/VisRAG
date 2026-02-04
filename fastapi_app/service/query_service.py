import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from ..config import get_app_config
from ..dao import ChromaDAO
from ..utils import ensure_dir, get_logger
from .embedding_service import QwenVLEmbedder
from .ocr_service import OCRService

# Split encoder/decoder QA (vision_tokens -> decoder-only)
_ds_encoder = None


def _get_ds_encoder():
    global _ds_encoder
    if _ds_encoder is None:
        # Lazy import to avoid GPU init on startup
        from src.vision import DeepseekEncoder

        _ds_encoder = DeepseekEncoder()
    return _ds_encoder


def _vision_tokens_cache_path(image_path: str) -> str:
    cfg = get_app_config()
    out_dir = os.path.join(cfg.indexing.assets_dir, "vision_tokens_cache")
    ensure_dir(out_dir)
    st = os.stat(image_path)
    key = f"{image_path}:{st.st_mtime_ns}:{st.st_size}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()
    return os.path.join(out_dir, f"{digest}.pt")


def _encode_image_to_vision_tokens_pt(image_path: str) -> str:
    out_path = _vision_tokens_cache_path(image_path)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    img = Image.open(image_path).convert("RGB")
    feats = _get_ds_encoder().encode_image(img)
    # Keep file small + portable
    torch.save(feats.detach().cpu(), out_path)
    return out_path


def _answer_with_decoder_only(*, question: str, context_text: str, vision_tokens_pt: str) -> str:
    from src.generator import get_generator

    gen = get_generator()
    system = (
        "You are a document QA assistant.\n"
        "Rules:\n"
        "- Output ONLY the final answer.\n"
        "- Be concise: 1 sentence (or a single number + unit).\n"
        "- No headings, no markdown, no citations, no extra context.\n"
    )
    user = "Answer the question using the given context and the image.\n"
    if context_text:
        user += f"Context:\n{context_text}\n\n"
    user += f"Question: {question}\nFinal answer (short):"
    raw = (gen.generate(system, user, vision_tokens=[vision_tokens_pt]) or "").strip()
    if not raw:
        return ""
    # keep it short even if the model starts dumping page text
    first = raw.splitlines()[0].strip()
    return first or raw[:200].strip()


def _maybe_enhance_query(query_text: Optional[str]) -> Optional[str]:
    """Enhance query for retrieval only (English prompt), optional via config."""
    if not query_text:
        return query_text
    cfg = get_app_config().query_enhance
    if not getattr(cfg, "enabled", False):
        return query_text

    oai = cfg.openai
    api_key = (oai.api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    base_url = (oai.base_url or os.environ.get("OPENAI_BASE_URL") or "").strip()
    if not api_key or not base_url:
        return query_text

    url = base_url.rstrip("/") + "/chat/completions"
    system = (
        "You rewrite user questions for semantic retrieval.\n"
        "Return ONE single-line enhanced query. Do NOT answer the question.\n"
        "Rules:\n"
        "- Keep <= 30 words.\n"
        "- Extract the most discriminative keywords/phrases and repeat them 3 times.\n"
        "- Preserve any paper title if present.\n"
        "- Preserve numbers/units if present.\n"
        "- No markdown, no quotes, no bullets."
    )
    user = f"Original query:\n{query_text}\n\nEnhanced query:"

    payload = {
        "model": oai.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(getattr(cfg, "temperature", 0.0)),
        "max_tokens": int(getattr(cfg, "max_tokens", 80)),
    }

    try:
        # no extra deps: urllib
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=int(getattr(oai, "timeout", 20))) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw)
        content = (((data.get("choices") or [])[0] or {}).get("message") or {}).get("content")
        if not content:
            return query_text
        enhanced = str(content).strip().replace("\n", " ")
        return enhanced or query_text
    except Exception:
        logger.exception("query_enhance failed; using original query")
        return query_text

logger = get_logger(__name__)


class QueryService:
    def __init__(self, chroma: ChromaDAO, embedder: QwenVLEmbedder, ocr: OCRService):
        self.chroma = chroma
        self.embedder = embedder
        self.ocr = ocr
        self.cfg = get_app_config()

    def query(
        self,
        *,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        image_collection: Optional[str] = None,
        image_top_k: Optional[int] = None,
        text_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        多模态检索流程（支持仅文本 / 仅图像 / 文本+图像；视频在 controller 层已抽帧为图像）：
        1) 用用户输入得到查询向量，在 image_collection 检索最相关的 vision（页面图像）；
        2) 根据命中图像的元数据定位到具体 PDF 的某一页，对应该页的 text collection（如 pdf_xxx_1）；
        3) 用同一查询向量在该页的 text collection 里检索最相关的文字块；
        4) 将命中页图像 + 用户问题 + 检索到的文字上下文一起喂给 DeepSeek-OCR，得到最终答案。
        """
        if not query_text and not query_image_path:
            raise ValueError("Query requires text and/or image")

        # Enhance query (retrieval only); keep original question for QA
        original_query_text = query_text
        retrieval_query_text = _maybe_enhance_query(query_text)

        if query_image_path:
            image = Image.open(query_image_path).convert("RGB")
        else:
            image = None

        # 多模态 embedding：仅文本 / 仅图像 / 文本+图像 都会得到同一空间下的一个向量
        query_vec = self.embedder.embed_query(text=retrieval_query_text, image=image)
        image_collection = image_collection or self.cfg.indexing.default_image_collection
        image_top_k = image_top_k or self.cfg.retrieval.image_top_k
        text_top_k = text_top_k or self.cfg.retrieval.text_top_k

        # 1) 用查询向量在 vision collection 里检索最相关的页面图像
        logger.info(f"Query image collection: {image_collection}, top_k={image_top_k}")
        image_res = self.chroma.query(image_collection, query_vec, image_top_k, dim=self.embedder.dim)
        image_ids = (image_res.get("ids") or [[]])[0] or []
        image_metas = (image_res.get("metadatas") or [[]])[0] or []

        if not image_ids:
            return {"image_results": [], "text_results": [], "ocr_text": ""}

        # 2) 根据命中图像元数据定位到该页对应的 text collection（某 PDF 某页）
        top_image_meta = image_metas[0] if image_metas else {}
        pdf_name = top_image_meta.get("pdf_name")
        page = top_image_meta.get("page")
        text_collection = top_image_meta.get("text_collection")
        if not text_collection and pdf_name and page is not None:
            prefix = self.cfg.indexing.text_collection_prefix or "pdf"
            text_collection = f"{prefix}_{pdf_name}_{page}"

        # 3) 在同一页的 text collection 里检索最相关的文字内容
        text_results: List[Dict[str, Any]] = []
        context_texts: List[str] = []
        if text_collection:
            logger.info(f"Query text collection: {text_collection}, top_k={text_top_k}")
            text_res = self.chroma.query(text_collection, query_vec, text_top_k, dim=self.embedder.dim)
            text_ids = (text_res.get("ids") or [[]])[0] or []
            text_metas = (text_res.get("metadatas") or [[]])[0] or []
            for tid, md in zip(text_ids, text_metas):
                text_results.append({"id": tid, "metadata": md})
                if isinstance(md, dict) and md.get("text"):
                    context_texts.append(md["text"])

        # 4) 命中页图像 + 问题 + 检索到的文字上下文 -> DeepSeek-OCR2（直接回答 / OCR）
        image_path = top_image_meta.get("image_path")
        ocr_text = ""
        answer = ""
        if image_path:
            # NOTE: 先改回“直接用 OCR2 模型”方便你 debug（定位 prompt/processor 问题）。
            # 如果后续要恢复 decoder-only QA，把下面这段注释解除即可。
            #
            # try:
            #     logger.info("Running decoder-only QA on retrieved page (vision_tokens)")
            #     tokens_pt = _encode_image_to_vision_tokens_pt(image_path)
            #     answer = _answer_with_decoder_only(
            #         question=original_query_text or retrieval_query_text or "",
            #         context_text="\n".join(context_texts),
            #         vision_tokens_pt=tokens_pt,
            #     )
            # except Exception:
            #     logger.exception("Decoder-only QA failed; falling back to OCR output")
            #     try:
            #         ocr_text = self.ocr.run(
            #             image_path,
            #             question=query_text,
            #             context_text="\n".join(context_texts),
            #         )
            #     except Exception:
            #         logger.exception("OCR fallback failed")

            try:
                logger.info("Running OCR2 on retrieved page (direct)")
                ocr_text = self.ocr.run(
                    image_path,
                    question=original_query_text or retrieval_query_text,
                    context_text="\n".join(context_texts),
                )
            except Exception:
                logger.exception("OCR2 failed")

        logger.info("Query completed")
        image_results = [
            {"id": pid, "metadata": md}
            for pid, md in zip(image_ids, image_metas)
        ]
        return {
            "image_results": image_results,
            "text_results": text_results,
            "answer": answer,
            "ocr_text": ocr_text,
        }
