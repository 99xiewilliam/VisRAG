import os
import uuid
from typing import Any, Dict, List, Optional, Sequence
from PIL import Image

from ..config import get_app_config
from ..dao import ChromaDAO
from ..utils import get_logger, ensure_dir, slugify
from .embedding_service import QwenVLEmbedder
from src.text import extract_pdf_text_by_page
from src.vision import pdf_to_images

logger = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


class IndexService:
    def __init__(self, chroma: ChromaDAO, embedder: QwenVLEmbedder):
        self.chroma = chroma
        self.embedder = embedder
        self.cfg = get_app_config()

    def index_items(
        self,
        *,
        texts: Optional[Sequence[str]] = None,
        image_paths: Optional[Sequence[str]] = None,
        collection: Optional[str] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        pdf_name: Optional[str] = None,
        page: Optional[int] = None,
    ) -> Dict[str, Any]:
        texts = list(texts or [])
        image_paths = list(image_paths or [])
        metadatas = list(metadatas or [])

        if not texts and not image_paths:
            raise ValueError("No texts or images to index")

        collection_name = collection or self.cfg.indexing.default_text_collection
        logger.info(f"Indexing items into collection: {collection_name}")

        ids: List[str] = []
        vectors: List[List[float]] = []
        metas: List[Dict[str, Any]] = []

        if texts:
            logger.info(f"Embedding {len(texts)} texts")
            text_vecs = self.embedder.embed_texts(texts)
            for idx, (text, vec) in enumerate(zip(texts, text_vecs)):
                item_id = f"txt_{uuid.uuid4().hex}"
                md = {"text": text, "type": "text"}
                if pdf_name:
                    md["pdf_name"] = pdf_name
                if page is not None:
                    md["page"] = page
                if idx < len(metadatas) and isinstance(metadatas[idx], dict):
                    md.update(metadatas[idx])
                ids.append(item_id)
                vectors.append(vec)
                metas.append(md)

        if image_paths:
            logger.info(f"Embedding {len(image_paths)} images")
            images = [Image.open(p).convert("RGB") for p in image_paths]
            image_vecs = self.embedder.embed_images(images)
            for idx, (path, vec) in enumerate(zip(image_paths, image_vecs)):
                item_id = f"img_{uuid.uuid4().hex}"
                md = {"image_path": path, "type": "image"}
                if pdf_name:
                    md["pdf_name"] = pdf_name
                if page is not None:
                    md["page"] = page
                if idx < len(metadatas) and isinstance(metadatas[idx], dict):
                    md.update(metadatas[idx])
                ids.append(item_id)
                vectors.append(vec)
                metas.append(md)

        self.chroma.add(collection_name, ids, vectors, metas, dim=self.embedder.dim)
        return {"collection": collection_name, "count": len(ids), "ids": ids}

    def index_pdf(self, pdf_path: str, pdf_name: Optional[str] = None) -> Dict[str, Any]:
        pdf_base = slugify(pdf_name or os.path.splitext(os.path.basename(pdf_path))[0])
        logger.info(f"Indexing PDF: {pdf_base} -> {pdf_path}")

        assets_dir = ensure_dir(os.path.join(self.cfg.indexing.assets_dir, "pdf_images", pdf_base))
        image_collection = self.cfg.indexing.default_image_collection

        pages = extract_pdf_text_by_page(pdf_path)
        images = pdf_to_images(pdf_path)
        logger.info(f"PDF pages: {len(pages)}; images: {len(images)}")

        image_ids: List[str] = []
        image_vecs: List[List[float]] = []
        image_metas: List[Dict[str, Any]] = []
        text_collections: List[str] = []
        total_text_chunks = 0
        global_text_ids: List[str] = []
        global_text_vecs: List[List[float]] = []
        global_text_metas: List[Dict[str, Any]] = []
        use_global_text = bool(getattr(self.cfg.indexing, "enable_global_text_collection", True))
        global_text_collection = getattr(self.cfg.indexing, "global_text_collection", "global_text_chunks")

        for idx, (page, image) in enumerate(zip(pages, images)):
            page_num = int(page["page"])
            image_path = os.path.join(assets_dir, f"page_{page_num}.png")
            image.save(image_path)
            vec = self.embedder.embed_images([image])[0]
            item_id = f"{pdf_base}_p{page_num}"
            prefix = self.cfg.indexing.text_collection_prefix or "pdf"
            text_collection = f"{prefix}_{pdf_base}_{page_num}"
            image_ids.append(item_id)
            image_vecs.append(vec)
            image_metas.append(
                {
                    "pdf_name": pdf_base,
                    "page": page_num,
                    "image_path": image_path,
                    "text_collection": text_collection,
                }
            )

            chunks = _chunk_text(page.get("text", ""), self.cfg.indexing.chunk_size, self.cfg.indexing.chunk_overlap)
            if chunks:
                text_ids: List[str] = []
                text_vecs = self.embedder.embed_texts(chunks)
                text_metas: List[Dict[str, Any]] = []
                for c_idx, (chunk, c_vec) in enumerate(zip(chunks, text_vecs)):
                    text_ids.append(f"{item_id}_c{c_idx}")
                    text_vecs[c_idx] = c_vec
                    text_metas.append(
                        {
                            "pdf_name": pdf_base,
                            "page": page_num,
                            "chunk_id": c_idx,
                            "text": chunk,
                        }
                    )
                self.chroma.add(text_collection, text_ids, text_vecs, text_metas, dim=self.embedder.dim)
                text_collections.append(text_collection)
                total_text_chunks += len(chunks)
                if use_global_text:
                    # Reuse same vectors for global text collection (A: text-only baseline)
                    for gid, gvec, gmd in zip(text_ids, text_vecs, text_metas):
                        global_text_ids.append(gid)
                        global_text_vecs.append(gvec)
                        md = dict(gmd)
                        md["text_collection"] = text_collection
                        global_text_metas.append(md)

        if image_ids:
            self.chroma.add(image_collection, image_ids, image_vecs, image_metas, dim=self.embedder.dim)
        if use_global_text and global_text_ids:
            self.chroma.add(global_text_collection, global_text_ids, global_text_vecs, global_text_metas, dim=self.embedder.dim)

        return {
            "pdf_name": pdf_base,
            "image_collection": image_collection,
            "images_indexed": len(image_ids),
            "text_collections": text_collections,
            "text_chunks_indexed": total_text_chunks,
            "global_text_collection": global_text_collection if use_global_text else None,
            "global_text_chunks_indexed": len(global_text_ids) if use_global_text else 0,
            "pages": len(pages),
        }

    def index_pdf_text_global(self, pdf_path: str, pdf_name: Optional[str] = None) -> Dict[str, Any]:
        """
        仅构建全量文本 chunk collection（A: text-only baseline）。
        不写入 page-level text collection，也不写入图片 collection。
        """
        pdf_base = slugify(pdf_name or os.path.splitext(os.path.basename(pdf_path))[0])
        logger.info(f"Indexing PDF text-only (global): {pdf_base} -> {pdf_path}")

        if not bool(getattr(self.cfg.indexing, "enable_global_text_collection", True)):
            return {"pdf_name": pdf_base, "global_text_collection": None, "global_text_chunks_indexed": 0, "pages": 0}

        global_text_collection = getattr(self.cfg.indexing, "global_text_collection", "global_text_chunks")
        pages = extract_pdf_text_by_page(pdf_path)
        logger.info(f"PDF pages: {len(pages)} (text-only global)")

        global_text_ids: List[str] = []
        global_text_vecs: List[List[float]] = []
        global_text_metas: List[Dict[str, Any]] = []
        total_text_chunks = 0

        for idx, page in enumerate(pages):
            page_num = int(page.get("page", idx + 1))
            chunks = _chunk_text(page.get("text", ""), self.cfg.indexing.chunk_size, self.cfg.indexing.chunk_overlap)
            if not chunks:
                continue
            text_vecs = self.embedder.embed_texts(chunks)
            for c_idx, (chunk, vec) in enumerate(zip(chunks, text_vecs)):
                chunk_id = f"{pdf_base}_p{page_num}_c{c_idx}"
                md = {
                    "pdf_name": pdf_base,
                    "page": page_num,
                    "chunk_id": c_idx,
                    "text": chunk,
                }
                global_text_ids.append(chunk_id)
                global_text_vecs.append(vec)
                global_text_metas.append(md)
            total_text_chunks += len(chunks)

        if global_text_ids:
            self.chroma.add(global_text_collection, global_text_ids, global_text_vecs, global_text_metas, dim=self.embedder.dim)

        return {
            "pdf_name": pdf_base,
            "global_text_collection": global_text_collection,
            "global_text_chunks_indexed": total_text_chunks,
            "pages": len(pages),
        }
