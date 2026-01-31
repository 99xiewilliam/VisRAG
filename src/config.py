"""VisRAG 配置管理模块"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class LocalModelConfig:
    model_path: str = "/data/xwh/models/Qwen3-1.7B"
    max_new_tokens: int = 768
    load_in_4bit: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    device: str = "auto"


@dataclass
class OpenAIConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    max_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 1.0
    timeout: int = 60
    max_retries: int = 3


@dataclass
class GeneratorConfig:
    backend: str = "local"  # "local" | "openai"
    local: LocalModelConfig = field(default_factory=LocalModelConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)


@dataclass
class VisionConfig:
    model_path: str = "/data/xwh/models/DeepSeek-OCR-2"
    device: str = "auto"


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    file_output: bool = False
    log_dir: str = "./logs"
    log_filename: Optional[str] = None
    show_location: bool = False


@dataclass
class LocalEmbeddingConfig:
    """本地 Embedding 模型配置"""
    model_path: str = "/data/xwh/models/Qwen3-Embedding-0.6B"
    device: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    use_fp16: bool = True


@dataclass
class OpenAIEmbeddingConfig:
    """OpenAI Embedding API 配置"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None  # null 表示使用模型默认维度
    timeout: int = 60
    max_retries: int = 3


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    backend: str = "hash"  # "hash" | "local" | "openai"
    dim: int = 256
    local: LocalEmbeddingConfig = field(default_factory=LocalEmbeddingConfig)
    openai: OpenAIEmbeddingConfig = field(default_factory=OpenAIEmbeddingConfig)


@dataclass
class VisRAGConfig:
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    logging: LogConfig = field(default_factory=LogConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


def load_config(config_path: str = "config.yaml") -> VisRAGConfig:
    """从 YAML 文件加载配置"""
    if not os.path.exists(config_path):
        # 尝试从项目根目录查找
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.yaml")
    
    if not os.path.exists(config_path):
        # 延迟导入 logger 避免循环依赖
        try:
            from .utils import get_logger
            _logger = get_logger(__name__)
            _logger.warning(f"未找到配置文件 {config_path}，使用默认配置")
        except:
            pass  # 日志尚未初始化，静默处理
        return VisRAGConfig()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    # 解析 generator 配置
    gen_data = data.get('generator', {})
    local_data = gen_data.get('local', {})
    openai_data = gen_data.get('openai', {})
    
    # 处理 OpenAI API Key（优先从环境变量读取）
    openai_api_key = openai_data.get('api_key')
    if openai_api_key is None:
        openai_api_key = os.environ.get('OPENAI_API_KEY')
    
    # 处理 OpenAI Base URL（优先从环境变量读取）
    openai_base_url = openai_data.get('base_url')
    if openai_base_url is None:
        openai_base_url = os.environ.get('OPENAI_BASE_URL')
    
    local_config = LocalModelConfig(
        model_path=local_data.get('model_path', "/data/xwh/models/Qwen3-1.7B"),
        max_new_tokens=local_data.get('max_new_tokens', 768),
        load_in_4bit=local_data.get('load_in_4bit', False),
        temperature=local_data.get('temperature', 0.0),
        top_p=local_data.get('top_p', 1.0),
        device=local_data.get('device', 'auto'),
    )
    
    openai_config = OpenAIConfig(
        api_key=openai_api_key,
        base_url=openai_base_url,
        model=openai_data.get('model', 'gpt-4o-mini'),
        max_tokens=openai_data.get('max_tokens', 768),
        temperature=openai_data.get('temperature', 0.0),
        top_p=openai_data.get('top_p', 1.0),
        timeout=openai_data.get('timeout', 60),
        max_retries=openai_data.get('max_retries', 3),
    )
    
    generator_config = GeneratorConfig(
        backend=gen_data.get('backend', 'local'),
        local=local_config,
        openai=openai_config,
    )
    
    # 解析 vision 配置
    vision_data = data.get('vision', {})
    vision_config = VisionConfig(
        model_path=vision_data.get('model_path', '/data/xwh/models/DeepSeek-OCR-2'),
        device=vision_data.get('device', 'auto'),
    )
    
    # 解析日志配置
    log_data = data.get('logging', {})
    log_config = LogConfig(
        level=log_data.get('level', 'INFO'),
        file_output=log_data.get('file_output', False),
        log_dir=log_data.get('log_dir', './logs'),
        log_filename=log_data.get('log_filename', None),
        show_location=log_data.get('show_location', False),
    )
    
    # 解析 embedding 配置
    embed_data = data.get('embedding', {})
    embed_local_data = embed_data.get('local', {})
    embed_openai_data = embed_data.get('openai', {})
    
    # 处理 OpenAI API Key（优先从环境变量读取）
    embed_api_key = embed_openai_data.get('api_key')
    if embed_api_key is None:
        embed_api_key = os.environ.get('OPENAI_API_KEY')
    
    # 处理 OpenAI Base URL（优先从环境变量读取）
    embed_base_url = embed_openai_data.get('base_url')
    if embed_base_url is None:
        embed_base_url = os.environ.get('OPENAI_BASE_URL')
    
    local_embed_config = LocalEmbeddingConfig(
        model_path=embed_local_data.get('model_path', '/data/xwh/models/Qwen3-Embedding-0.6B'),
        device=embed_local_data.get('device', 'auto'),
        batch_size=embed_local_data.get('batch_size', 32),
        max_length=embed_local_data.get('max_length', 512),
        use_fp16=embed_local_data.get('use_fp16', True),
    )
    
    openai_embed_config = OpenAIEmbeddingConfig(
        api_key=embed_api_key,
        base_url=embed_base_url,
        model=embed_openai_data.get('model', 'text-embedding-3-small'),
        dimensions=embed_openai_data.get('dimensions', None),
        timeout=embed_openai_data.get('timeout', 60),
        max_retries=embed_openai_data.get('max_retries', 3),
    )
    
    embedding_config = EmbeddingConfig(
        backend=embed_data.get('backend', 'hash'),
        dim=embed_data.get('dim', 256),
        local=local_embed_config,
        openai=openai_embed_config,
    )
    
    return VisRAGConfig(
        generator=generator_config,
        vision=vision_config,
        logging=log_config,
        embedding=embedding_config,
    )


# 全局配置实例
_config: Optional[VisRAGConfig] = None


def get_config(config_path: Optional[str] = None) -> VisRAGConfig:
    """获取全局配置（单例模式）"""
    global _config
    if _config is None or config_path is not None:
        _config = load_config(config_path) if config_path else load_config()
    return _config


def reload_config(config_path: Optional[str] = None):
    """重新加载配置"""
    global _config
    _config = load_config(config_path) if config_path else load_config()
    return _config
