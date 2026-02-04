import time
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Handle both relative and absolute imports
try:
    from .config import get_app_config
    from .controller import health_router, index_router, query_router
    from .utils import get_logger, init_logging
except ImportError:
    # If running directly, add parent directory to path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from fastapi_app.config import get_app_config
    from fastapi_app.controller import health_router, index_router, query_router
    from fastapi_app.utils import get_logger, init_logging

logger = get_logger(__name__)


def create_app() -> FastAPI:
    init_logging()
    cfg = get_app_config()
    app = FastAPI(title="VisRAG API", version="1.0.0")

    if cfg.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    if cfg.api.log_requests:
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            start = time.time()
            response = await call_next(request)
            elapsed = (time.time() - start) * 1000
            logger.info(f"{request.method} {request.url.path} {response.status_code} {elapsed:.1f}ms")
            return response

    app.include_router(health_router)
    app.include_router(index_router)
    app.include_router(query_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_app_config()
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)
