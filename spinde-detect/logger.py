#!/usr/bin/env python3
"""
Logger Module
=============

Handles logging configuration for the spindle analysis pipeline.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_main_logger(log_level: str = "INFO", project_dir: str = None) -> logging.Logger:
    """
    Setup main pipeline logging configuration.
    
    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    project_dir : str, optional
        Project directory to create logs subdirectory
    
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Create logs directory if project_dir is provided
    if project_dir:
        logs_dir = Path(project_dir) / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / 'spindle_analysis_pipeline.log'
    else:
        log_file = 'spindle_analysis.log'
    
    # Clear any existing handlers to avoid duplication
    logging.getLogger().handlers.clear()
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Main pipeline logging initialized - log file: {log_file}")
    return logger

def setup_subject_logger(subject_id: str, log_dir: Path, global_logger: logging.Logger) -> logging.Logger:
    """
    Setup subject-specific logger.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    log_dir : Path
        Directory to store subject logs
    global_logger : logging.Logger
        Main pipeline logger
    
    Returns
    -------
    logging.Logger
        Subject-specific logger
    """
    # Create logs directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup subject-specific logger
    subject_logger = logging.getLogger(f"subject_{subject_id}")
    subject_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    subject_logger.handlers.clear()
    
    # Create file handler for subject-specific logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{subject_id}_processing_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to subject logger
    subject_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplication
    subject_logger.propagate = False
    
    global_logger.info(f"[{subject_id}] Subject logger initialized - log file: {log_file}")
    return subject_logger

def log_info(logger: logging.Logger, subject_id: str, message: str):
    """Log info message to both subject and global loggers."""
    logger.info(f"[{subject_id}] {message}")

def log_warning(logger: logging.Logger, subject_id: str, message: str):
    """Log warning message to both subject and global loggers."""
    logger.warning(f"[{subject_id}] {message}")

def log_error(logger: logging.Logger, subject_id: str, message: str):
    """Log error message to both subject and global loggers."""
    logger.error(f"[{subject_id}] {message}")

def log_debug(logger: logging.Logger, subject_id: str, message: str):
    """Log debug message to subject logger only."""
    logger.debug(f"[{subject_id}] {message}")

def cleanup_logger(logger: logging.Logger):
    """Clean up logger handlers."""
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear() 