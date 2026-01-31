import os
import chromadb
from typing import List, Dict, Any, Optional
from .utils import get_logger

logger = get_logger(__name__)


class ChromaStore:
    def __init__(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info(f"ChromaStore 初始化完成: {persist_dir}")

    def get_collection(self, name: str, dim: int):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "dim": dim},
        )

    def add(self, collection_name: str, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        collection = self.client.get_or_create_collection(name=collection_name)
        if hasattr(collection, "upsert"):
            collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas)
        else:
            collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)
        logger.debug(f"向 {collection_name} 添加 {len(ids)} 条记录")

    def query(self, collection_name: str, vector: List[float], top_k: int):
        collection = self.client.get_or_create_collection(name=collection_name)
        return collection.query(query_embeddings=[vector], n_results=top_k)

    def get(self, collection_name: str, ids: List[str], include: Optional[List[str]] = None):
        collection = self.client.get_or_create_collection(name=collection_name)
        if include is None:
            include = ["metadatas"]
        return collection.get(ids=ids, include=include)

    def get_metadatas(self, collection_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        res = self.get(collection_name, ids, include=["metadatas"])
        metas = res.get("metadatas") or []
        out: List[Dict[str, Any]] = []
        for md in metas:
            if isinstance(md, dict):
                out.append(md)
            else:
                out.append({})
        return out
