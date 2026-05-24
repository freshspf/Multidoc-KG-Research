"""
Centralized logging system for Multi-Paper Knowledge Graph Construction Framework.

Provides structured logging with:
- Console output: INFO level (clean and minimal)
- File output: DEBUG level (detailed for debugging)
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


# Global logger instance cache
_loggers: dict = {}


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    This is a singleton pattern - same name returns the same logger instance.
    
    Args:
        name: Logger name (typically __name__ or module name)
        log_dir: Directory to store log files (default: "logs")
        
    Returns:
        Configured logger instance
    """
    # Return existing logger if already created
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers filter
    
    # Avoid duplicate handlers if logger already has them
    if logger.handlers:
        return logger
    
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create file handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"run_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Create console handler (minimal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(name)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Cache logger
    _loggers[name] = logger
    
    # Log initialization
    logger.debug(f"Logger initialized: {name}, log file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger instance.
    
    Convenience function that calls setup_logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name)


# Default logger for the application
logger = setup_logger("Multidoc-KG")
