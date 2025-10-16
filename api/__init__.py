"""
API module for the Graph + Vector RAG system.

This module provides FastAPI endpoints for querying the RAG system
and managing the knowledge base.
"""

from .main import app

__all__ = ["app"]
