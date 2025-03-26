import sys

from loguru import logger

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<level>{level}</level> | <cyan>{message}</cyan>",
    level="TRACE",
)

logger.info("Logger initialised")


def get_logger():
    """Returns the configured loguru logger."""
    return logger
