"""
logger.py — File-based logging for TeaVision AI backend
========================================================
Logs all API events, agent steps, and errors to timestamped log files.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Create logs directory
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Log filename with date
LOG_FILENAME = LOGS_DIR / f"teavision_{datetime.now().strftime('%Y%m%d')}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILENAME, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Create named loggers for different components
api_logger = logging.getLogger("API")
agent_logger = logging.getLogger("AGENT")
model_logger = logging.getLogger("MODEL")
error_logger = logging.getLogger("ERROR")

def log_api(message: str):
    """Log API-level events."""
    api_logger.info(message)

def log_agent(message: str):
    """Log agent pipeline events."""
    agent_logger.info(message)

def log_model(message: str):
    """Log model inference events."""
    model_logger.info(message)

def log_error(message: str, exc: Exception = None):
    """Log errors with optional exception."""
    if exc:
        error_logger.error(f"{message}: {exc}", exc_info=True)
    else:
        error_logger.error(message)

def log_warning(message: str):
    """Log warnings."""
    error_logger.warning(message)

# Startup log
log_api(f"Logger initialized. Writing to: {LOG_FILENAME}")
