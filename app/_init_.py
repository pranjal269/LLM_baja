# app/__init__.py
"""
LLM Document Processing System

A comprehensive system for processing natural language queries 
against unstructured documents using AI and vector search.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# app/services/__init__.py
"""
Services module containing core business logic components.
"""

from .document_loader import DocumentLoader
from .chunker import DocumentChunker
from .embeddings import EmbeddingService
from .semantic_search import SemanticSearch
from .query_parser import QueryParser
from .answer_generator import AnswerGenerator
from .auth import AuthService

__all__ = [
    "DocumentLoader",
    "DocumentChunker", 
    "EmbeddingService",
    "SemanticSearch",
    "QueryParser",
    "AnswerGenerator",
    "AuthService"
]