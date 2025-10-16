"""
Configuration management for the Graph + Vector RAG system.

This module provides centralized configuration loading and validation
using Pydantic models for type safety and validation.
"""

import os
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback to simple classes if Pydantic not available
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

    def Field(default=None, **kwargs):
        return default

    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration settings."""
    chromadb: Dict[str, Any] = Field(default_factory=dict)
    path: str = "./data/chromadb"
    vector_dim: int = 384
    max_connections: int = 100


class DocumentsConfig(BaseModel):
    """Document processing configuration."""
    input_dir: str = "./documents"
    supported_formats: List[str] = Field(default_factory=lambda: [
        ".pdf", ".md", ".txt", ".docx", ".xlsx", ".pptx"
    ])
    chunk_size: int = 1000
    chunk_overlap: int = 200


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"
    normalize: bool = True


class LLMConfig(BaseModel):
    """LLM configuration for Ollama."""
    base_url: str = "http://localhost:11434"
    model_name: str = "llama2:7b"
    entity_model: str = "llama2:7b"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 60


class EntitiesConfig(BaseModel):
    """Entity extraction configuration."""
    enabled: bool = True
    max_entities_per_chunk: int = 10
    min_confidence: float = 0.7
    relationship_types: List[str] = Field(default_factory=lambda: [
        "related_to", "part_of", "instance_of", "subclass_of",
        "located_in", "works_at", "created_by"
    ])


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1


class RetrievalConfig(BaseModel):
    """Information retrieval configuration."""
    vector_top_k: int = 5
    graph_depth: int = 2
    hybrid_alpha: float = 0.7
    rerank_top_k: int = 10


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file: str = "./data/rag_system.log"
    max_size: str = "10 MB"
    backup_count: int = 5


class PerformanceConfig(BaseModel):
    """Performance optimization settings."""
    cache_embeddings: bool = True
    cache_size: int = 1000
    parallel_processing: bool = True
    max_workers: int = 4


class Config(BaseModel):
    """Main configuration class that holds all settings."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    entities: EntitiesConfig = Field(default_factory=EntitiesConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    @validator('*', pre=True)
    def convert_nested_dicts(cls, v):
        """Convert nested dictionaries to appropriate config objects."""
        if isinstance(v, dict):
            # Convert nested dicts to their respective config classes
            if 'chromadb' in v and isinstance(v.get('chromadb'), dict):
                v['database'] = DatabaseConfig(**v)
                if 'chromadb' in v:
                    v['database'].chromadb = v.pop('chromadb')
            elif any(key in v for key in ['input_dir', 'supported_formats', 'chunk_size']):
                v['documents'] = DocumentsConfig(**v)
            elif any(key in v for key in ['model_name', 'batch_size', 'device']):
                v['embeddings'] = EmbeddingsConfig(**v)
            elif any(key in v for key in ['base_url', 'model_name', 'temperature']):
                v['llm'] = LLMConfig(**v)
            elif any(key in v for key in ['max_entities_per_chunk', 'min_confidence']):
                v['entities'] = EntitiesConfig(**v)
            elif any(key in v for key in ['host', 'port', 'reload']):
                v['api'] = APIConfig(**v)
            elif any(key in v for key in ['vector_top_k', 'graph_depth', 'hybrid_alpha']):
                v['retrieval'] = RetrievalConfig(**v)
            elif any(key in v for key in ['level', 'file', 'max_size']):
                v['logging'] = LoggingConfig(**v)
            elif any(key in v for key in ['cache_embeddings', 'cache_size', 'parallel_processing']):
                v['performance'] = PerformanceConfig(**v)

        return v


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with all settings loaded
    """
    if config_path is None:
        # Try to find config file in standard locations
        possible_paths = [
            "config.yaml",
            "config.yml",
            "./config.yaml",
            "./config.yml",
            "~/config.yaml",
            "~/config.yml"
        ]

        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                config_path = expanded_path
                break

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            if config_dict:
                return Config(**config_dict)
            else:
                logger.warning(f"Empty or invalid config file: {config_path}")

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")

    # Return default configuration
    logger.info("Using default configuration")
    return Config()


def save_config(config: Config, config_path: str = "config.yaml") -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        config_path: Path where to save the configuration
    """
    try:
        # Convert to dictionary
        config_dict = config.dict() if hasattr(config, 'dict') else vars(config)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, allow_unicode=True)

        logger.info(f"Configuration saved to: {config_path}")

    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        raise


def create_default_config(config_path: str = "config.yaml") -> Config:
    """
    Create a default configuration file.

    Args:
        config_path: Path where to create the configuration file

    Returns:
        Config object with default settings
    """
    config = Config()
    save_config(config, config_path)
    return config


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(f"Database path: {config.database.path}")
    print(f"Embedding model: {config.embeddings.model_name}")
    print(f"LLM model: {config.llm.model_name}")
    print(f"API port: {config.api.port}")
