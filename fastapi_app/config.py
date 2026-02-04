import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))

def _resolve_path(path: str) -> str:
    """Resolve relative paths against project root."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_project_root(), path))


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    log_requests: bool = True


@dataclass
class RetrievalConfig:
    image_top_k: int = 1
    text_top_k: int = 3


@dataclass
class IndexingConfig:
    default_image_collection: str = "vision_pages"
    default_text_collection: str = "default_texts"
    text_collection_prefix: str = "pdf"
    chunk_size: int = 400
    chunk_overlap: int = 50
    persist_dir: str = field(default_factory=lambda: os.path.join(_project_root(), "output", "chroma_db"))
    assets_dir: str = field(default_factory=lambda: os.path.join(_project_root(), "output", "api_assets"))


@dataclass
class QwenVLEmbeddingConfig:
    model_path: str = "/home/xwh/models/Qwen3-VL-Embedding-2B"
    device: str = "auto"
    dtype: str = "bfloat16"  # vLLM dtype, e.g. "bfloat16" / "float16"
    batch_size: int = 8
    max_length: int = 32768  # Qwen3-VL-Embedding-2B supports up to 32k context length
    embedding_dim: Optional[int] = None  # null = use model default (up to 2048), or set 64-2048 for custom dimension
    service_url: Optional[str] = None  # Optional remote embedding service endpoint
    service_timeout: int = 60


@dataclass
class OCRConfig:
    model_path: str = "/home/xwh/models/DeepSeek-OCR-2"
    vllm_code_dir: Optional[str] = None  # Path to DeepSeek-OCR2-vllm (for deepseek_ocr2 + process imports)
    prompt_template: Optional[str] = None  # Optional base prompt template. If None, uses default English QA template
    # HF infer() options (preferred over vLLM for QA-style prompting)
    base_size: int = 1024
    image_size: int = 768
    crop_mode: bool = True
    save_results: bool = False
    output_dir: Optional[str] = None  # null -> use indexing.assets_dir/ocr_infer


@dataclass
class OpenAICompatConfig:
    """OpenAI-compatible Chat Completions config (e.g. api.zhizengzeng.com)."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    timeout: int = 20


@dataclass
class QueryEnhanceConfig:
    enabled: bool = False
    max_tokens: int = 80
    temperature: float = 0.0
    openai: OpenAICompatConfig = field(default_factory=OpenAICompatConfig)


@dataclass
class LoggingConfig:
    enabled: bool = True
    level: str = "INFO"
    file_output: bool = False
    log_dir: str = "./logs"
    log_filename: Optional[str] = None
    show_location: bool = False


@dataclass
class AppConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    vl_embedding: QwenVLEmbeddingConfig = field(default_factory=QwenVLEmbeddingConfig)
    query_enhance: QueryEnhanceConfig = field(default_factory=QueryEnhanceConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    if not config_path:
        config_path = os.path.join(_project_root(), "config.yaml")
    if not os.path.exists(config_path):
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    api_data = data.get("api", {})
    retrieval_data = data.get("retrieval", {})
    indexing_data = data.get("indexing", {})
    vl_data = data.get("vl_embedding", {})
    qe_data = data.get("query_enhance", {})
    ocr_data = data.get("ocr", {})
    log_data = data.get("logging", {})

    api_cfg = ApiConfig(
        host=api_data.get("host", "0.0.0.0"),
        port=int(api_data.get("port", 8000)),
        cors_origins=api_data.get("cors_origins", ["*"]),
        log_requests=bool(api_data.get("log_requests", True)),
    )

    retrieval_cfg = RetrievalConfig(
        image_top_k=int(retrieval_data.get("image_top_k", 1)),
        text_top_k=int(retrieval_data.get("text_top_k", 3)),
    )

    indexing_cfg = IndexingConfig(
        default_image_collection=indexing_data.get("default_image_collection", "vision_pages"),
        default_text_collection=indexing_data.get("default_text_collection", "default_texts"),
        text_collection_prefix=indexing_data.get("text_collection_prefix", "pdf"),
        chunk_size=int(indexing_data.get("chunk_size", 400)),
        chunk_overlap=int(indexing_data.get("chunk_overlap", 50)),
        persist_dir=_resolve_path(indexing_data.get("persist_dir", os.path.join("output", "chroma_db"))),
        assets_dir=_resolve_path(indexing_data.get("assets_dir", os.path.join("output", "api_assets"))),
    )

    embedding_dim_raw = vl_data.get("embedding_dim")
    embedding_dim = None if embedding_dim_raw is None else int(embedding_dim_raw)
    vl_cfg = QwenVLEmbeddingConfig(
        model_path=vl_data.get("model_path", "/home/xwh/models/Qwen3-VL-Embedding-2B"),
        device=vl_data.get("device", "auto"),
        dtype=vl_data.get("dtype", "bfloat16"),
        batch_size=int(vl_data.get("batch_size", 8)),
        max_length=int(vl_data.get("max_length", 32768)),
        embedding_dim=embedding_dim,
        service_url=vl_data.get("service_url", None),
        service_timeout=int(vl_data.get("service_timeout", 60)),
    )

    qe_openai = qe_data.get("openai", {}) if isinstance(qe_data, dict) else {}
    qe_cfg = QueryEnhanceConfig(
        enabled=bool(qe_data.get("enabled", False)) if isinstance(qe_data, dict) else False,
        max_tokens=int(qe_data.get("max_tokens", 80)) if isinstance(qe_data, dict) else 80,
        temperature=float(qe_data.get("temperature", 0.0)) if isinstance(qe_data, dict) else 0.0,
        openai=OpenAICompatConfig(
            api_key=qe_openai.get("api_key", None),
            base_url=qe_openai.get("base_url", None),
            model=qe_openai.get("model", "gpt-4o-mini"),
            timeout=int(qe_openai.get("timeout", 20)),
        ),
    )

    ocr_cfg = OCRConfig(
        model_path=ocr_data.get("model_path", "/home/xwh/models/DeepSeek-OCR-2"),
        vllm_code_dir=ocr_data.get("vllm_code_dir", None),
        prompt_template=ocr_data.get("prompt_template", None),
        base_size=int(ocr_data.get("base_size", 1024)),
        image_size=int(ocr_data.get("image_size", 768)),
        crop_mode=bool(ocr_data.get("crop_mode", True)),
        save_results=bool(ocr_data.get("save_results", False)),
        output_dir=ocr_data.get("output_dir", None),
    )

    log_cfg = LoggingConfig(
        enabled=bool(log_data.get("enabled", True)),
        level=log_data.get("level", "INFO"),
        file_output=bool(log_data.get("file_output", False)),
        log_dir=_resolve_path(log_data.get("log_dir", "./logs")),
        log_filename=log_data.get("log_filename", None),
        show_location=bool(log_data.get("show_location", False)),
    )

    return AppConfig(
        api=api_cfg,
        retrieval=retrieval_cfg,
        indexing=indexing_cfg,
        vl_embedding=vl_cfg,
        query_enhance=qe_cfg,
        ocr=ocr_cfg,
        logging=log_cfg,
    )


_app_config: Optional[AppConfig] = None


def get_app_config(config_path: Optional[str] = None) -> AppConfig:
    global _app_config
    if _app_config is None or config_path is not None:
        _app_config = load_config(config_path)
    return _app_config
