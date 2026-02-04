import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import get_app_config, LoggingConfig


def _build_format(cfg: LoggingConfig) -> str:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if cfg.show_location:
        fmt = fmt.replace("%(name)s", "%(name)s:%(funcName)s:%(lineno)d")
    return fmt


def init_logging(config_path: Optional[str] = None) -> None:
    cfg = get_app_config(config_path).logging
    if not cfg.enabled:
        logging.getLogger().handlers = []
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    root_logger.handlers = []

    fmt = _build_format(cfg)
    date_fmt = "%Y-%m-%d %H:%M:%S"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root_logger.addHandler(console_handler)

    if cfg.file_output:
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if cfg.log_filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            cfg.log_filename = f"visrag_api_{date_str}.log"
        log_path = log_dir / cfg.log_filename
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, cfg.level.upper(), logging.INFO))
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        init_logging()
    return logging.getLogger(name)
