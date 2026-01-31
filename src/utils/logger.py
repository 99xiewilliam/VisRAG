"""VisRAG 统一日志模块

使用方式:
    >>> from src.utils import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("这是一条信息日志")
    >>> logger.debug("调试信息: %s", some_var)
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class LogConfig:
    """日志配置"""
    # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
    level: str = "INFO"
    
    # 日志格式
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # 控制台输出
    console_output: bool = True
    
    # 文件输出
    file_output: bool = False
    log_dir: str = "./logs"
    log_filename: Optional[str] = None  # 默认为 visrag_YYYYMMDD.log
    
    # 是否显示函数名和行号（调试用）
    show_location: bool = False


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（仅在终端支持颜色时生效）"""
    
    # ANSI 颜色码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m',       # 重置
    }
    
    def __init__(self, fmt: str, datefmt: str = None, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(config: Optional[LogConfig] = None) -> None:
    """
    设置全局日志配置
    
    Args:
        config: 日志配置，如果为 None 则使用默认配置
    """
    cfg = config or LogConfig()
    
    # 构建格式字符串
    fmt = cfg.format
    if cfg.show_location:
        fmt = fmt.replace(
            "%(name)s",
            "%(name)s:%(funcName)s:%(lineno)d"
        )
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, cfg.level.upper()))
    
    # 清除现有处理器
    root_logger.handlers = []
    
    # 控制台处理器
    if cfg.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, cfg.level.upper()))
        console_formatter = ColoredFormatter(
            fmt,
            datefmt=cfg.date_format,
            use_color=True
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # 文件处理器
    if cfg.file_output:
        log_dir = Path(cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if cfg.log_filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            cfg.log_filename = f"visrag_{date_str}.log"
        
        log_path = log_dir / cfg.log_filename
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, cfg.level.upper()))
        file_formatter = logging.Formatter(fmt, datefmt=cfg.date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"日志文件: {log_path}")


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 通常是 __name__，建议使用模块的完整路径
        
    Returns:
        logging.Logger: 配置好的日志记录器
        
    Example:
        >>> from src.utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("模块已加载")
    """
    # 确保日志系统已初始化
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()
    
    return logging.getLogger(name)


# 全局日志配置实例
_log_config: Optional[LogConfig] = None


def set_log_config(config: LogConfig) -> None:
    """设置全局日志配置"""
    global _log_config
    _log_config = config
    setup_logging(config)


def get_log_config() -> LogConfig:
    """获取当前日志配置"""
    global _log_config
    if _log_config is None:
        _log_config = LogConfig()
    return _log_config


def init_logging_from_config(config_path: Optional[str] = None) -> None:
    """
    从 YAML 配置文件初始化日志设置
    
    在 config.yaml 中添加:
    ```yaml
    logging:
      level: "INFO"
      file_output: true
      log_dir: "./logs"
    ```
    """
    try:
        import yaml
        from src.config import get_config
        
        cfg = get_config(config_path)
        
        # 尝试从配置中获取日志设置
        if hasattr(cfg, 'logging') and cfg.logging:
            # cfg.logging 是一个 LogConfig dataclass 对象
            log_cfg = cfg.logging
            log_config = LogConfig(
                level=log_cfg.level if hasattr(log_cfg, 'level') else 'INFO',
                file_output=log_cfg.file_output if hasattr(log_cfg, 'file_output') else False,
                log_dir=log_cfg.log_dir if hasattr(log_cfg, 'log_dir') else './logs',
                log_filename=log_cfg.log_filename if hasattr(log_cfg, 'log_filename') else None,
                show_location=log_cfg.show_location if hasattr(log_cfg, 'show_location') else False,
            )
        else:
            # 从全局配置字典中读取（如果 config.yaml 中有 logging 字段）
            config_file = config_path or "config.yaml"
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                log_cfg_dict = data.get('logging', {})
                log_config = LogConfig(
                    level=log_cfg_dict.get('level', 'INFO'),
                    file_output=log_cfg_dict.get('file_output', False),
                    log_dir=log_cfg_dict.get('log_dir', './logs'),
                    log_filename=log_cfg_dict.get('log_filename', None),
                    show_location=log_cfg_dict.get('show_location', False),
                )
            else:
                log_config = LogConfig()
        
        set_log_config(log_config)
        
    except Exception as e:
        # 配置加载失败时使用默认配置
        print(f"[Logger] 从配置文件初始化日志失败: {e}")
        print("[Logger] 使用默认日志配置")
        setup_logging()


class LoggerMixin:
    """日志混入类，方便类中使用日志
    
    Example:
        >>> class MyClass(LoggerMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.logger.info("MyClass 初始化")
    """
    
    def __init__(self):
        self._logger: Optional[logging.Logger] = None
    
    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)
        return self._logger
