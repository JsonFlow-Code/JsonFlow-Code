import logging
from config.config import Config

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    config = Config()
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = config.get("logging.level", "INFO")
        log_file = config.get("logging.file", "jsonflow.log")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.addHandler(handler)
    return logger
