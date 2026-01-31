import os
from typing import List, Dict, Any
from .store import ChromaStore
from .text import extract_pdf_text_by_page
from .embedder import create_embedder, get_text_dim
from .vision import extract_pdf_vision_tokens, decode_tokens_list
from .utils import get_logger

logger = get_logger(__name__)

class VisRAGPipeline:
    def __init__(self, persist_dir: str):
        self.store = ChromaStore(persist_dir)
        self.embedder = create_embedder()
        self.text_dim = get_text_dim()
        self.vision_dim = 1280
        self.text_collection = "text_pages"
        self.vision_collection = "vision_pages"
        logger.info(f"VisRAGPipeline 初始化完成，数据目录: {persist_dir}, 文本维度: {self.text_dim}")

    def index_pdf(
        self,
        doc_id: str,
        pdf_path: str,
        tokens_dir: str,
        max_pages: int | None = None,
    ):
        logger.info(f"开始索引 PDF: {doc_id} -> {pdf_path}")

        pages = extract_pdf_text_by_page(pdf_path)
        if max_pages is not None:
            pages = pages[:max_pages]
        text_ids: List[str] = []
        text_vecs: List[List[float]] = []
        text_meta: List[Dict[str, Any]] = []
        # 批量编码文本
        texts = [page["text"] for page in pages]
        text_vectors = self.embedder.embed(texts)
        
        for page, vec in zip(pages, text_vectors):
            page_id = f"{doc_id}_p{page['page']}"
            text_ids.append(page_id)
            text_vecs.append(vec)
            text_meta.append({
                "doc_id": doc_id,
                "page": page["page"],
                "text": page["text"],
                "pdf_path": pdf_path,
            })

        self.store.add(self.text_collection, text_ids, text_vecs, text_meta)
        logger.info(f"文本索引完成: {len(pages)} 页")

        # 索引视觉
        vision_pages = extract_pdf_vision_tokens(pdf_path, tokens_dir, max_pages=max_pages)
        vision_ids: List[str] = []
        vision_vecs: List[List[float]] = []
        vision_meta: List[Dict[str, Any]] = []
        for page in vision_pages:
            page_id = f"{doc_id}_p{page['page']}"
            vision_ids.append(page_id)
            vision_vecs.append(page["vector"])
            vision_meta.append({
                "doc_id": doc_id,
                "page": page["page"],
                "tokens_path": page["tokens_path"],
                "pdf_path": pdf_path,
            })

        self.store.add(self.vision_collection, vision_ids, vision_vecs, vision_meta)
        logger.info(f"视觉索引完成: {len(vision_pages)} 页")

    def query_text(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        logger.debug(f"文本查询: '{query[:50]}...' top_k={top_k}")
        qvec = self.embedder.embed_single(query)
        return self.store.query(self.text_collection, qvec, top_k)

    def query_vision(
        self,
        query: str,
        top_k: int = 5,
        text_top_k: int | None = None,
    ) -> Dict[str, Any]:
        return self._query_vision_locked_by_text_pages(query, top_k=top_k, text_top_k=text_top_k)

    def _query_vision_locked_by_text_pages(
        self,
        query: str,
        top_k: int = 5,
        text_top_k: int | None = None,
    ) -> Dict[str, Any]:
        pages_k = text_top_k or top_k
        text_res = self.query_text(query, top_k=pages_k)
        page_ids = (text_res.get("ids") or [[]])[0] or []
        text_distances = (text_res.get("distances") or [[]])[0] or []
        if not page_ids:
            return {"ids": [[]], "metadatas": [[]], "distances": [[]]}

        vis_res = self.store.get(self.vision_collection, page_ids, include=["metadatas"])
        vis_ids = vis_res.get("ids") or []
        vis_metas = vis_res.get("metadatas") or []
        meta_by_id: Dict[str, Any] = {}
        for pid, md in zip(vis_ids, vis_metas):
            meta_by_id[pid] = md

        out_ids: List[str] = []
        out_metas: List[Dict[str, Any]] = []
        out_distances: List[float] = []
        for idx, pid in enumerate(page_ids):
            md = meta_by_id.get(pid)
            if not isinstance(md, dict):
                continue
            out_ids.append(pid)
            out_metas.append(md)
            if idx < len(text_distances):
                out_distances.append(text_distances[idx])
            else:
                out_distances.append(1.0)
        return {"ids": [out_ids], "metadatas": [out_metas], "distances": [out_distances]}

    def decode_vision_tokens(self, tokens_paths: List[str], prompt: str | None = None) -> List[str]:
        return decode_tokens_list(tokens_paths, prompt=prompt)
