# logger_setup.py
import logging
from logging.handlers import RotatingFileHandler
from logging import FileHandler
import os

def setup_logger(name: str, log_file: str = "logs/app.log", level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def attach_conversation_log(logger, convo_log_path: str, level=logging.INFO):
    """
    Add a dedicated FileHandler for a single conversation run.
    Returns the handler so you can remove/close it when done.
    """
    os.makedirs(os.path.dirname(convo_log_path), exist_ok=True)

    # ✅ Use the SAME clean formatter here too (no INFO | name |)
    formatter = logging.Formatter(
        fmt="%(asctime)s\n%(message)s\n",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = FileHandler(convo_log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return fh