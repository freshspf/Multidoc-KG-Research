"""
Core module for external service interfaces.
"""
from .llm_client import LLMClient, MockLLMClient
from .graph_store import MockGraphStore

__all__ = ["LLMClient", "MockLLMClient", "MockGraphStore"]
