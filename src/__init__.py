"""VisRAG Source Package"""

from .config import (
    VisRAGConfig,
    GeneratorConfig,
    LocalModelConfig,
    OpenAIConfig,
    VisionConfig,
    LogConfig,
    EmbeddingConfig,
    LocalEmbeddingConfig,
    OpenAIEmbeddingConfig,
    get_config,
    reload_config,
)

from .generator import (
    BaseGenerator,
    LocalGenerator,
    OpenAIGenerator,
    create_generator,
    get_generator,
)

from .embedder import (
    BaseEmbedder,
    HashEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    create_embedder,
    get_embedder,
    get_text_dim,
)

from .utils import (
    get_logger,
    setup_logging,
    LogConfig as LoggerConfig,
)

__all__ = [
    # Config
    "VisRAGConfig",
    "GeneratorConfig",
    "LocalModelConfig",
    "OpenAIConfig",
    "VisionConfig",
    "LogConfig",
    "EmbeddingConfig",
    "LocalEmbeddingConfig",
    "OpenAIEmbeddingConfig",
    "get_config",
    "reload_config",
    # Generator
    "BaseGenerator",
    "LocalGenerator",
    "OpenAIGenerator",
    "create_generator",
    "get_generator",
    # Embedder
    "BaseEmbedder",
    "HashEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "create_embedder",
    "get_embedder",
    "get_text_dim",
    # Utils
    "get_logger",
    "setup_logging",
    "LoggerConfig",
]
