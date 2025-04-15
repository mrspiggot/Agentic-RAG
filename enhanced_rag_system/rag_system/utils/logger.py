# Logging setup functions
# File: rag_system/utils/logger.py
# Instruction: Create this new file with the following content.

import logging
import sys

DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
DEFAULT_LOG_LEVEL = logging.INFO

configured_loggers = set()

def setup_logger(name: str = None, level: int = DEFAULT_LOG_LEVEL, log_format: str = DEFAULT_LOG_FORMAT) -> logging.Logger:
    """
    Sets up and configures a logger or the root logger.

    Avoids adding multiple handlers to the same logger if called multiple times.

    Args:
        name: Name of the specific logger. If None, configures the root logger.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_format: Format string for log messages.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger(name) # Get specific logger or root if name is None

    # Check if this logger (or root if name is None) has already been configured by this function
    logger_key = name if name else "root"
    if logger_key in configured_loggers and logger.hasHandlers():
         logger.debug(f"Logger '{logger_key}' already configured. Skipping handler setup.")
         logger.setLevel(level) # Still allow level update
         return logger

    # Set level
    logger.setLevel(level)

    # Create handler (console handler)
    # Check if a handler of the same type already exists (more robust check)
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    if not has_stream_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(handler)
        logger.debug(f"Configured new StreamHandler for logger '{logger_key}'.")
    else:
         logger.debug(f"StreamHandler already exists for logger '{logger_key}'.")
         # Optionally update level of existing handlers if needed
         for h in logger.handlers:
              if isinstance(h, logging.StreamHandler): h.setLevel(level)


    # Prevent propagation to root logger if configuring a specific logger
    if name:
        logger.propagate = False

    configured_loggers.add(logger_key)
    return logger