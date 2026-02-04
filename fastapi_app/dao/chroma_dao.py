import os
from typing import Any, Dict, List, Optional
import chromadb

from ..utils import get_logger

logger = get_logger(__name__)


class ChromaDAO:
    def __init__(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info(f"ChromaDAO initialized: {persist_dir}")

    def get_or_create_collection(self, name: str, dim: int):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "dim": dim},
        )

    def add(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        dim: int,
    ) -> None:
        collection = self.get_or_create_collection(collection_name, dim=dim)
        if hasattr(collection, "upsert"):
            collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas)
        else:
            collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)
        logger.debug(f"Added {len(ids)} items to {collection_name}")

    def query(self, collection_name: str, vector: List[float], top_k: int, dim: int):
        collection = self.get_or_create_collection(collection_name, dim=dim)
        return collection.query(query_embeddings=[vector], n_results=top_k)

    def get(self, collection_name: str, ids: List[str], dim: int, include: Optional[List[str]] = None):
        collection = self.get_or_create_collection(collection_name, dim=dim)
        if include is None:
            include = ["metadatas"]
        return collection.get(ids=ids, include=include)
