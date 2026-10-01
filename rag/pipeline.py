"""Compatibility wrapper. New code should use rag_strategies.naive.NaiveRAG."""

from rag_strategies.naive.pipeline import NaiveRAG as RAGPipeline

__all__ = ["RAGPipeline"]
