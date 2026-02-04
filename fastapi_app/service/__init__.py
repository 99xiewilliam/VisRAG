from typing import Optional

from ..config import get_app_config
from ..dao import ChromaDAO
from .embedding_service import QwenVLEmbedder
from .index_service import IndexService
from .query_service import QueryService
from .ocr_service import OCRService


_chroma: Optional[ChromaDAO] = None
_embedder: Optional[QwenVLEmbedder] = None
_ocr: Optional[OCRService] = None
_index_service: Optional[IndexService] = None
_query_service: Optional[QueryService] = None


def _get_chroma() -> ChromaDAO:
    global _chroma
    if _chroma is None:
        cfg = get_app_config()
        _chroma = ChromaDAO(cfg.indexing.persist_dir)
    return _chroma


def _get_embedder() -> QwenVLEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = QwenVLEmbedder()
    return _embedder


def _get_ocr() -> OCRService:
    global _ocr
    if _ocr is None:
        _ocr = OCRService()
    return _ocr


def get_index_service() -> IndexService:
    global _index_service
    if _index_service is None:
        _index_service = IndexService(_get_chroma(), _get_embedder())
    return _index_service


def get_query_service() -> QueryService:
    global _query_service
    if _query_service is None:
        _query_service = QueryService(_get_chroma(), _get_embedder(), _get_ocr())
    return _query_service
