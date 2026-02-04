from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TextQueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    image_collection: Optional[str] = None
    image_top_k: Optional[int] = None
    text_top_k: Optional[int] = None


class QueryResponse(BaseModel):
    image_results: List[Dict[str, Any]]
    text_results: List[Dict[str, Any]]
    answer: Optional[str] = None
    ocr_text: str


class IndexTextRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    collection: Optional[str] = None
    pdf_name: Optional[str] = None
    page: Optional[int] = None


class IndexResponse(BaseModel):
    collection: str
    count: int
    ids: List[str]
