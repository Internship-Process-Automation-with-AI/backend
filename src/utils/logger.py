"""
Reusable logger utility for the backend application.
"""

import logging
import sys


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger with the specified name, configured for the application.
    Args:
        name: Logger name (usually __name__)
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# Example usage:
if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Logger is working!")
    log.warning("This is a warning.")
    log.error("This is an error.")
