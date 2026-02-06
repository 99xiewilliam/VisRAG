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
from .answer_service import AnswerService
from ..prompts import get_prompts
from .reranker_service import get_reranker

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
    prompts = get_prompts().get("query", {}).get("vision", {})
    system = prompts.get("system", "")
    tmpl = prompts.get("user", "Question: {question}\nAnswer:")
    user = tmpl.format(context=context_text or "", question=question or "")
    raw = (gen.generate(system, user, vision_tokens=[vision_tokens_pt]) or "").strip()
    if not raw:
        return ""
    # keep it short even if the model starts dumping page text
    first = raw.splitlines()[0].strip()
    return first or raw[:200].strip()


def _answer_with_text_only(*, question: str, context_text: str) -> str:
    """
    用纯文本 LLM 基于 context 做答案抽取/归纳（比让 OCR 模型“读整页”稳定）。
    """
    if not question or not context_text:
        return ""
    from src.generator import get_generator

    gen = get_generator()
    prompts = get_prompts().get("query", {}).get("text_only", {})
    system = prompts.get("system", "")
    tmpl = prompts.get("user", "Question: {question}\nAnswer:")
    user = tmpl.format(context=context_text or "", question=question or "")
    raw = (gen.generate(system, user) or "").strip()
    if not raw:
        return ""
    first = raw.splitlines()[0].strip()
    return first or raw[:200].strip()

def _render_text_to_image(text: str, *, width: int = 512, font_size: int = 24) -> Image.Image:
    """把纯文本渲染成一张 PIL 图（白底黑字），用于「问题转图片」检索。"""
    from PIL import ImageDraw, ImageFont

    if not text or not text.strip():
        # 返回一张小空白图，避免 embed 报错
        return Image.new("RGB", (max(width, 64), 64), (255, 255, 255))

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # 先估算行高与换行
    margin = 20
    max_line_w = width - 2 * margin
    words = text.strip().split()
    lines: List[str] = []
    current = []
    current_w = 0
    for w in words:
        if hasattr(font, "getbbox"):
            b = font.getbbox(w)
            w_w = b[2] - b[0]
        else:
            w_w = (font.getsize(w)[0] if hasattr(font, "getsize") else len(w) * font_size // 2)
        if current and current_w + w_w > max_line_w:
            lines.append(" ".join(current))
            current = [w]
            current_w = w_w
        else:
            current.append(w)
            current_w += w_w + (font_size // 4)
    if current:
        lines.append(" ".join(current))

    line_height = int(font_size * 1.4)
    img_h = 2 * margin + len(lines) * line_height
    img = Image.new("RGB", (width, max(img_h, 64)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = margin + i * line_height
        draw.text((margin, y), line, fill=(0, 0, 0), font=font)
    return img


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
        "You rewrite the user's question into a single line optimized for finding the right paragraph/page in documents (e.g. PDFs).\n"
        "Do NOT answer the question. Output ONLY the enhanced query line.\n\n"
        "Rules:\n"
        "- Expand with synonyms and phrases that actually appear in papers: e.g. 'how much better' -> 'improvement over, gain, margin, percentage, accuracy, F1'.\n"
        "- Include the exact paper title or method name if the user mentioned it (e.g. 'Enriching BERT with Knowledge Graph Embeddings').\n"
        "- Add related terms a document would use: model names (BERT, baseline), task names (document classification), and metrics (%, accuracy, improvement).\n"
        "- Rephrase the question as key phrases that might appear in a section heading or results sentence (e.g. 'improvement over standard BERT in document classification').\n"
        "- Keep 1 line, 25–50 words. No markdown, no quotes, no bullets."
    )
    user = f"Original query:\n{query_text}\n\nEnhanced query (one line for retrieval):"

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
        with urllib.request.urlopen(req, timeout=int(getattr(oai, "timeout", 60))) as resp:
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
    def __init__(self, chroma: ChromaDAO, embedder: QwenVLEmbedder, ocr: OCRService, answer_service: AnswerService):
        self.chroma = chroma
        self.embedder = embedder
        self.ocr = ocr
        self.answer_service = answer_service
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
        logger.info(retrieval_query_text)

        if query_image_path:
            image = Image.open(query_image_path).convert("RGB")
        else:
            image = None

        # 可选：将纯文本问题转成图片再做检索（由 config retrieval.query_render_to_image 控制）
        if image is None and self.cfg.retrieval.query_render_to_image and retrieval_query_text:
            image = _render_text_to_image(retrieval_query_text)
            query_vec = self.embedder.embed_query(text=None, image=image)
        else:
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
            return {"image_results": [], "text_results": [], "answer": ""}

        # Optional: rerank retrieved page images with Qwen3-VL-Reranker
        try:
            reranker = get_reranker()
            if reranker.enabled and retrieval_query_text:
                # Only rerank candidates that have image_path
                candidate_imgs: List[Image.Image] = []
                candidate_indices: List[int] = []
                for i, md in enumerate(image_metas):
                    if isinstance(md, dict) and md.get("image_path") and os.path.exists(md["image_path"]):
                        try:
                            candidate_imgs.append(Image.open(md["image_path"]).convert("RGB"))
                            candidate_indices.append(i)
                        except Exception:
                            logger.exception("Failed to open candidate image for rerank")

                if candidate_imgs:
                    scores = reranker.score_images(query_text=retrieval_query_text, images=candidate_imgs)
                    if scores and len(scores) == len(candidate_indices):
                        ranked = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)
                        new_ids: List[str] = []
                        new_metas: List[Dict[str, Any]] = []
                        used = set()
                        for idx, sc in ranked:
                            used.add(idx)
                            new_ids.append(image_ids[idx])
                            md = image_metas[idx] if idx < len(image_metas) else {}
                            md2 = dict(md) if isinstance(md, dict) else {"_raw": md}
                            md2["rerank_score"] = float(sc)
                            new_metas.append(md2)
                        # Append any non-reranked items (no image_path) in original order
                        for i, pid in enumerate(image_ids):
                            if i in used:
                                continue
                            new_ids.append(pid)
                            md = image_metas[i] if i < len(image_metas) else {}
                            new_metas.append(dict(md) if isinstance(md, dict) else {"_raw": md})
                        image_ids, image_metas = new_ids, new_metas
                        logger.info("Rerank applied to image candidates")
        except Exception:
            logger.exception("Image rerank failed; using original order")

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
                backend = getattr(getattr(self.answer_service, "cfg", None), "backend", "ocr2")
                logger.info(f"Running answer_generator on retrieved page (backend={backend})")
                ocr_text = self.answer_service.generate_from_image(
                    image_path=image_path,
                    question=original_query_text or retrieval_query_text,
                    context_text="\n".join(context_texts),
                )
            except Exception:
                logger.exception("answer_generator failed")

        logger.info("Query completed")
        image_results = [
            {"id": pid, "metadata": md}
            for pid, md in zip(image_ids, image_metas)
        ]
        answer = ocr_text or ""
        return {
            "image_results": image_results,
            "text_results": text_results,
            "answer": answer,
        }

    def query_text_only(
        self,
        *,
        query_text: str,
        text_collection: Optional[str] = None,
        text_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        A: Text-only baseline. Search over global text chunk collection.
        """
        if not query_text:
            raise ValueError("Query requires text")

        original_query_text = query_text
        retrieval_query_text = _maybe_enhance_query(query_text)
        query_vec = self.embedder.embed_query(text=retrieval_query_text, image=None)

        cfg = self.cfg.indexing
        text_collection = text_collection or getattr(cfg, "global_text_collection", None) or cfg.default_text_collection
        text_top_k = text_top_k or self.cfg.retrieval.text_top_k

        text_results: List[Dict[str, Any]] = []
        context_texts: List[str] = []
        logger.info(f"Query text-only collection: {text_collection}, top_k={text_top_k}")
        text_res = self.chroma.query(text_collection, query_vec, text_top_k, dim=self.embedder.dim)
        text_ids = (text_res.get("ids") or [[]])[0] or []
        text_metas = (text_res.get("metadatas") or [[]])[0] or []
        for tid, md in zip(text_ids, text_metas):
            text_results.append({"id": tid, "metadata": md})
            if isinstance(md, dict) and md.get("text"):
                context_texts.append(md["text"])

        answer = ""
        if original_query_text and context_texts:
            try:
                answer = _answer_with_text_only(
                    question=original_query_text,
                    context_text="\n".join(context_texts),
                )
            except Exception:
                logger.exception("text-only answering failed")

        return {
            "image_results": [],
            "text_results": text_results,
            "answer": answer,
        }

    def query_vision_only(
        self,
        *,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        image_collection: Optional[str] = None,
        image_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        B: Vision-only baseline. Search images only, no text chunk retrieval.
        """
        if not query_text and not query_image_path:
            raise ValueError("Query requires text and/or image")

        original_query_text = query_text
        retrieval_query_text = _maybe_enhance_query(query_text)

        image = Image.open(query_image_path).convert("RGB") if query_image_path else None
        if image is None and self.cfg.retrieval.query_render_to_image and retrieval_query_text:
            image = _render_text_to_image(retrieval_query_text)
            query_vec = self.embedder.embed_query(text=None, image=image)
        else:
            query_vec = self.embedder.embed_query(text=retrieval_query_text, image=image)

        image_collection = image_collection or self.cfg.indexing.default_image_collection
        image_top_k = image_top_k or self.cfg.retrieval.image_top_k
        logger.info(f"Query image collection (vision-only): {image_collection}, top_k={image_top_k}")
        image_res = self.chroma.query(image_collection, query_vec, image_top_k, dim=self.embedder.dim)
        image_ids = (image_res.get("ids") or [[]])[0] or []
        image_metas = (image_res.get("metadatas") or [[]])[0] or []

        if not image_ids:
            return {"image_results": [], "text_results": [], "answer": ""}

        # Optional rerank
        try:
            reranker = get_reranker()
            if reranker.enabled and retrieval_query_text:
                candidate_imgs: List[Image.Image] = []
                candidate_indices: List[int] = []
                for i, md in enumerate(image_metas):
                    if isinstance(md, dict) and md.get("image_path") and os.path.exists(md["image_path"]):
                        try:
                            candidate_imgs.append(Image.open(md["image_path"]).convert("RGB"))
                            candidate_indices.append(i)
                        except Exception:
                            logger.exception("Failed to open candidate image for rerank")

                if candidate_imgs:
                    scores = reranker.score_images(query_text=retrieval_query_text, images=candidate_imgs)
                    if scores and len(scores) == len(candidate_indices):
                        ranked = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)
                        new_ids: List[str] = []
                        new_metas: List[Dict[str, Any]] = []
                        used = set()
                        for idx, sc in ranked:
                            used.add(idx)
                            new_ids.append(image_ids[idx])
                            md = image_metas[idx] if idx < len(image_metas) else {}
                            md2 = dict(md) if isinstance(md, dict) else {"_raw": md}
                            md2["rerank_score"] = float(sc)
                            new_metas.append(md2)
                        for i, pid in enumerate(image_ids):
                            if i in used:
                                continue
                            new_ids.append(pid)
                            md = image_metas[i] if i < len(image_metas) else {}
                            new_metas.append(dict(md) if isinstance(md, dict) else {"_raw": md})
                        image_ids, image_metas = new_ids, new_metas
        except Exception:
            logger.exception("Image rerank failed; using original order")

        # Use OCR QA on top image only (vision-only)
        top_image_meta = image_metas[0] if image_metas else {}
        image_path = top_image_meta.get("image_path")
        ocr_text = ""
        answer = ""
        if image_path:
            try:
                backend = getattr(getattr(self.answer_service, "cfg", None), "backend", "ocr2")
                logger.info(f"Running answer_generator on retrieved page (backend={backend})")
                ocr_text = self.answer_service.generate_from_image(
                    image_path=image_path,
                    question=original_query_text or retrieval_query_text,
                    context_text=None,
                )
                answer = ocr_text or ""
            except Exception:
                logger.exception("answer_generator failed")

        image_results = [{"id": pid, "metadata": md} for pid, md in zip(image_ids, image_metas)]
        return {
            "image_results": image_results,
            "text_results": [],
            "answer": answer,
        }
