from .health_controller import router as health_router
from .index_controller import router as index_router
from .query_controller import router as query_router

__all__ = ["health_router", "index_router", "query_router"]
