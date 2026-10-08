"""
Unit tests for RAG knowledge retrieval system.
"""

import pytest
from src.rag import chunk_text, load_knowledge_documents, retrieve_context


def test_chunk_text():
    sample_text = "Paragraph one.\n\nParagraph two with more text.\n\nParagraph three."
    chunks = chunk_text(sample_text, chunk_size=20)
    assert len(chunks) >= 3


def test_load_knowledge_documents():
    chunks = load_knowledge_documents("data/knowledge")
    assert len(chunks) > 0
    assert any("eGFR" in c for c in chunks)


def test_retrieve_context():
    retrieved = retrieve_context("eGFR slope and kidney function", k=2)
    assert len(retrieved) <= 2
    assert len(retrieved) > 0
    assert isinstance(retrieved[0], str)
