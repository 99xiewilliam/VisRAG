"""文本 Embedding 抽象层，支持 hash、本地模型和 OpenAI API 三种模式

使用方式:
    >>> from src.embedder import create_embedder, get_embedder
    >>> embedder = create_embedder()  # 自动读取 config.yaml
    >>> vectors = embedder.embed(["文本1", "文本2"])
    >>> # 或单个文本
    >>> vector = embedder.embed_single("单个文本")
"""

import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

from .config import EmbeddingConfig, get_config
from .utils import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Embedding 抽象基类"""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本列表编码为向量列表
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表，每个向量是 float 列表
        """
        pass
    
    def embed_single(self, text: str) -> List[float]:
        """编码单个文本"""
        return self.embed([text])[0]
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """返回 embedding 维度"""
        pass


class HashEmbedder(BaseEmbedder):
    """
    基于 MD5 Hash 的轻量级 Embedding
    
    特点:
    - 无需额外模型或依赖
    - 确定性输出（相同文本总是产生相同向量）
    - 适用于快速原型和对比实验
    - 语义表达能力有限
    """
    
    def __init__(self, dim: int = 256):
        self._dim = dim
        logger.info(f"初始化 HashEmbedder: dim={dim}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """使用 MD5 hash 生成向量"""
        results = []
        for text in texts:
            vec = [0.0] * self._dim
            if text:
                for token in text.split():
                    h = hashlib.md5(token.encode("utf-8")).digest()
                    for i in range(self._dim):
                        vec[i] += h[i % len(h)] / 255.0
                norm = sum(v * v for v in vec) ** 0.5
                if norm > 0:
                    vec = [v / norm for v in vec]
            results.append(vec)
        return results
    
    @property
    def dim(self) -> int:
        return self._dim


class LocalEmbedder(BaseEmbedder):
    """
    本地 HuggingFace Embedding 模型
    
    支持的模型:
    - /data/xwh/models/Qwen3-Embedding-0.6B
    - sentence-transformers 系列模型
    - 其他 HuggingFace embedding 模型
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        self.cfg = config or get_config().embedding
        self.local_cfg = self.cfg.local
        
        # 确定设备
        if self.local_cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.local_cfg.device)
        
        self.dtype = torch.float16 if self.local_cfg.use_fp16 and self.device.type == "cuda" else torch.float32
        
        logger.info(f"初始化 LocalEmbedder: model={self.local_cfg.model_path}")
        logger.info(f"设备: {self.device}, 精度: {self.dtype}")
        
        # 加载模型和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_cfg.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.local_cfg.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        
        # 获取实际维度
        with torch.no_grad():
            sample_inputs = self.tokenizer(
                "sample",
                return_tensors="pt",
                truncation=True,
                max_length=self.local_cfg.max_length,
            ).to(self.device)
            sample_output = self.model(**sample_inputs)
            self._dim = sample_output.last_hidden_state.shape[-1]
        
        logger.info(f"模型加载完成，维度: {self._dim}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """使用本地模型编码文本"""
        import torch
        import torch.nn.functional as F
        
        results = []
        batch_size = self.local_cfg.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.local_cfg.max_length,
                return_tensors="pt",
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用 mean pooling
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                # 归一化
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            results.extend(embeddings.cpu().float().numpy().tolist())
        
        return results
    
    @property
    def dim(self) -> int:
        return self._dim


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI Embedding API
    
    支持的模型:
    - text-embedding-3-small (推荐，性价比高)
    - text-embedding-3-large (质量最高)
    - text-embedding-ada-002 (旧版)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "使用 OpenAI Embedding 需要安装 openai 包: "
                "pip install openai"
            )
        
        self.cfg = config or get_config().embedding
        self.oai_cfg = self.cfg.openai
        
        # 检查 API Key
        if not self.oai_cfg.api_key:
            raise ValueError(
                "OpenAI API Key 未设置。请通过以下方式之一配置：\n"
                "1. 在 config.yaml 中设置 embedding.openai.api_key\n"
                "2. 设置环境变量 OPENAI_API_KEY"
            )
        
        # 初始化 OpenAI 客户端
        client_kwargs = {
            "api_key": self.oai_cfg.api_key,
            "timeout": self.oai_cfg.timeout,
            "max_retries": self.oai_cfg.max_retries,
        }
        if self.oai_cfg.base_url:
            client_kwargs["base_url"] = self.oai_cfg.base_url
        
        self.client = OpenAI(**client_kwargs)
        self.model = self.oai_cfg.model
        
        # 根据模型确定维度
        self._dim = self._get_model_dim()
        
        logger.info(f"初始化 OpenAIEmbedder: model={self.model}, dim={self._dim}")
        if self.oai_cfg.base_url:
            logger.info(f"API Base: {self.oai_cfg.base_url}")
    
    def _get_model_dim(self) -> int:
        """获取模型的输出维度"""
        if self.oai_cfg.dimensions:
            return self.oai_cfg.dimensions
        
        # 默认维度
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dims.get(self.model, 1536)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """使用 OpenAI API 编码文本"""
        kwargs = {
            "model": self.model,
            "input": texts,
        }
        
        if self.oai_cfg.dimensions:
            kwargs["dimensions"] = self.oai_cfg.dimensions
        
        response = self.client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
    
    @property
    def dim(self) -> int:
        return self._dim


def create_embedder(config: Optional[EmbeddingConfig] = None) -> BaseEmbedder:
    """
    工厂函数：根据配置创建对应的 Embedder
    
    Usage:
        >>> from src.embedder import create_embedder
        >>> embedder = create_embedder()  # 自动读取 config.yaml
        >>> vectors = embedder.embed(["文本1", "文本2"])
    """
    cfg = config or get_config().embedding
    
    if cfg.backend == "hash":
        return HashEmbedder(cfg.dim)
    elif cfg.backend == "local":
        return LocalEmbedder(cfg)
    elif cfg.backend == "openai":
        return OpenAIEmbedder(cfg)
    else:
        raise ValueError(f"不支持的 embedding backend 类型: {cfg.backend}，可选: hash, local, openai")


# 全局 Embedder 实例（单例模式）
_embedder_instance: Optional[BaseEmbedder] = None


def get_embedder() -> BaseEmbedder:
    """获取全局 Embedder 实例"""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = create_embedder()
    return _embedder_instance


def reset_embedder():
    """重置 Embedder 实例（用于重新加载配置后）"""
    global _embedder_instance
    _embedder_instance = None


def get_text_dim() -> int:
    """获取当前配置下的文本 embedding 维度"""
    return get_embedder().dim
