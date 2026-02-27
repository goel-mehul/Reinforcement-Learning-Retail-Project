import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: Path = None, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger that writes to both console and a log file.

    Usage:
        logger = get_logger(__name__)
        logger.info("Training started")
        logger.warning("Low inventory detected")
    """
    logger = logging.getLogger(name)

    # avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger