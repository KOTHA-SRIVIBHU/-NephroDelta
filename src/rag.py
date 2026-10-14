"""
Simple Local RAG System for NephroDelta.

Chunks educational knowledge files, generates embeddings using SentenceTransformers,
indexes them with FAISS, and provides vector similarity retrieval for AI explanations.
"""

import os
import glob
from typing import List, Tuple, Any
import numpy as np

# Global variables for caching vector store components
_VECTOR_STORE_CACHE = None


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Splits text into overlapping chunks based on paragraph or character counts.

    Args:
        text: Raw text string.
        chunk_size: Approximate maximum character size per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    # An overlap >= chunk_size would give a non-positive step and loop forever.
    step = chunk_size - overlap if 0 <= overlap < chunk_size else chunk_size

    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunk = para[start:end]
                chunks.append(chunk.strip())
                start += step

    return chunks


def load_knowledge_documents(knowledge_dir: str = "data/knowledge") -> List[str]:
    """
    Loads all text documents from the knowledge directory and chunks them.

    Args:
        knowledge_dir: Path to knowledge directory containing .txt files.

    Returns:
        List of all document text chunks.
    """
    if not os.path.exists(knowledge_dir):
        return [
            "Kidneys filter blood to remove waste products and maintain electrolyte balance.",
            "eGFR (Estimated Glomerular Filtration Rate) measures overall renal filtration capacity.",
            "UACR (Urine Albumin-to-Creatinine Ratio) detects early protein leakage in urine."
        ]

    txt_files = glob.glob(os.path.join(knowledge_dir, "*.txt"))
    all_chunks = []

    for filepath in txt_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                file_chunks = chunk_text(content)
                all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"Warning: Could not read knowledge file {filepath}: {e}")

    if not all_chunks:
        all_chunks = [
            "eGFR represents estimated glomerular filtration rate in mL/min/1.73m^2.",
            "UACR measures urine albumin-to-creatinine ratio in mg/g.",
            "Longitudinal delta changes indicate trends over observation periods."
        ]

    return all_chunks


def build_vector_store(knowledge_dir: str = "data/knowledge") -> Tuple[Any, Any, List[str]]:
    """
    Builds or returns cached FAISS vector store and SentenceTransformer model.

    Args:
        knowledge_dir: Directory containing knowledge files.

    Returns:
        Tuple of (index, model, text_chunks).
    """
    global _VECTOR_STORE_CACHE
    if _VECTOR_STORE_CACHE is not None:
        return _VECTOR_STORE_CACHE

    chunks = load_knowledge_documents(knowledge_dir)

    try:
        from sentence_transformers import SentenceTransformer
        import faiss

        # Load small fast model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))

        _VECTOR_STORE_CACHE = (index, model, chunks)
        return _VECTOR_STORE_CACHE

    except Exception as e:
        print(f"Warning: SentenceTransformer or FAISS failed to initialize: {e}")
        # Return fallback mock tuple if sentence-transformers/faiss are missing or offline
        _VECTOR_STORE_CACHE = (None, None, chunks)
        return _VECTOR_STORE_CACHE


def retrieve_context(query: str, k: int = 3, knowledge_dir: str = "data/knowledge") -> List[str]:
    """
    Retrieves the top-k most relevant text chunks for a given query string.

    Args:
        query: User query or analytics context query string.
        k: Number of relevant chunks to retrieve (default: 3).
        knowledge_dir: Path to knowledge directory.

    Returns:
        List of top-k retrieved text chunk strings.
    """
    if not query or not query.strip():
        return []

    index, model, chunks = build_vector_store(knowledge_dir)

    if index is not None and model is not None:
        try:
            import faiss
            query_vector = model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_vector)

            top_k = min(k, len(chunks))
            distances, indices = index.search(query_vector.astype(np.float32), top_k)

            retrieved = []
            for idx in indices[0]:
                if 0 <= idx < len(chunks):
                    retrieved.append(chunks[idx])
            return retrieved
        except Exception as e:
            print(f"Warning: Vector search error: {e}")

    # Keyword fallback retrieval
    query_lower = query.lower()
    matches = [c for c in chunks if any(term in c.lower() for term in query_lower.split())]
    if not matches:
        return chunks[:k]
    return matches[:k]
