#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper utilities for the multi-agent knowledge graph pipeline
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # Replace environment variables
    config_content = os.path.expandvars(config_content)
    
    config = yaml.safe_load(config_content)
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("kg_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Create necessary directories based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of created directory paths
    """
    paths = config['paths']
    directories = {}
    
    base_output = Path(paths['output_dir'])
    directories['output'] = base_output
    directories['extraction'] = base_output / paths['extraction_dir']
    directories['evaluation'] = base_output / paths['evaluation_dir']
    directories['qa'] = base_output / paths['qa_dir']
    directories['logs'] = base_output / 'logs'
    
    # Create all directories
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
    
    return directories


def get_input_papers(config: Dict[str, Any]) -> List[Path]:
    """
    Get list of input paper files
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of paper file paths
    """
    input_dir = Path(config['paths']['input_dir'])
    pattern = config['file_patterns']['input_papers']
    
    paper_files = list(input_dir.glob(pattern))
    
    # 自定义排序：按数字顺序排序paper_文件
    def paper_sort_key(file_path):
        name = file_path.name
        # 提取paper_后面的数字
        if name.startswith('paper_'):
            try:
                num_str = name.replace('paper_', '').replace('.json', '')
                return int(num_str)
            except ValueError:
                return float('inf')  # 非数字文件排在最后
        return name  # 非paper_文件按名称排序
    
    paper_files.sort(key=paper_sort_key)
    
    return paper_files


def generate_filename(template: str, paper_name: str, timestamp: Optional[str] = None) -> str:
    """
    Generate filename based on template
    
    Args:
        template: Filename template with placeholders
        paper_name: Paper name
        timestamp: Optional timestamp
        
    Returns:
        Generated filename
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Remove .json extension from paper name if present
    clean_paper_name = paper_name.replace('.json', '')
    
    return template.format(
        paper_name=clean_paper_name,
        timestamp=timestamp
    )


def save_json(data: Any, file_path: Path, ensure_ascii: bool = False, indent: int = 2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Output file path
        ensure_ascii: Whether to ensure ASCII encoding
        indent: JSON indentation
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def load_json(file_path: Path) -> Any:
    """
    Load data from JSON file
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_environment(config: Dict = None) -> Dict[str, Dict[str, Any]]:
    """
    Validate API configuration from both config and environment variables

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of validation results with detailed information
    """
    validation = {}

    # Load configuration
    config = config or {}
    api_config = config.get('api', {})

    # Check API key (can be from config or environment)
    api_key_from_config = api_config.get('openai_api_key')
    api_key_from_env = os.getenv('OPENAI_API_KEY')
    api_key_available = bool(api_key_from_config or api_key_from_env)

    validation['OPENAI_API_KEY'] = {
        'required': True,
        'present': api_key_available,
        'config_value': bool(api_key_from_config),
        'env_value': bool(api_key_from_env),
        'source': 'config' if api_key_from_config else 'environment' if api_key_from_env else 'missing'
    }

    # Check base URL (optional)
    base_url_from_config = api_config.get('openai_base_url')
    base_url_from_env = os.getenv('OPENAI_BASE_URL')
    base_url_available = bool(base_url_from_config or base_url_from_env)

    validation['OPENAI_BASE_URL'] = {
        'required': False,
        'present': base_url_available,
        'config_value': bool(base_url_from_config),
        'env_value': bool(base_url_from_env),
        'source': 'config' if base_url_from_config else 'environment' if base_url_from_env else 'default'
    }

    return validation


def get_file_stats(file_path: Path) -> Dict[str, Any]:
    """
    Get file statistics
    
    Args:
        file_path: File path
        
    Returns:
        File statistics dictionary
    """
    if not file_path.exists():
        return {'exists': False}
    
    stat = file_path.stat()
    return {
        'exists': True,
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir()
    }


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
    
    def get_duration(self) -> float:
        """Get duration in seconds"""
        return self.duration if self.duration else 0.0
    
    def get_duration_str(self) -> str:
        """Get formatted duration string"""
        if not self.duration:
            return "0.0s"
        
        if self.duration < 60:
            return f"{self.duration:.1f}s"
        elif self.duration < 3600:
            minutes = int(self.duration // 60)
            seconds = self.duration % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(self.duration // 3600)
            minutes = int((self.duration % 3600) // 60)
            seconds = self.duration % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"
