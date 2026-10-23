"""core/rag.py – ChromaDB-backed RAG over guideline documents.

Index:  seed.py calls build_index() once.
Query:  retrieve_guidelines(query, n=3) → list of {text, source, score}.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import chromadb
import yaml

from core.llm import embed

_CFG = yaml.safe_load(Path("config.yaml").read_text())["chroma"]
_PERSIST_DIR: str = _CFG["persist_dir"]
_COLLECTION: str = _CFG["collection_name"]
_GUIDELINES_DIR = Path("data/guidelines")

_CHUNK_SIZE = 400   # characters
_CHUNK_OVERLAP = 80


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=_PERSIST_DIR)
    return client.get_or_create_collection(
        name=_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def _doc_id(source: str, idx: int) -> str:
    raw = f"{source}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def build_index(force: bool = False) -> int:
    """
    Index all .md files in data/guidelines/.
    Returns number of chunks indexed.
    Skips if already indexed (unless force=True).
    """
    col = _get_collection()
    if col.count() > 0 and not force:
        return col.count()

    chunks_added = 0
    for md_file in sorted(_GUIDELINES_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        # Strip markdown headers for cleaner chunking
        clean = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
        chunks = _chunk_text(clean)
        for i, chunk in enumerate(chunks):
            doc_id = _doc_id(md_file.name, i)
            embedding = embed(chunk)
            col.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": md_file.name, "chunk_idx": i}],
            )
            chunks_added += 1

    return chunks_added


def retrieve_guidelines(query: str, n: int = 3) -> list[dict]:
    """
    Retrieve the top-n guideline chunks most relevant to `query`.
    Returns list of {text, source, score}.
    """
    col = _get_collection()
    if col.count() == 0:
        return []

    query_embedding = embed(query)
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(n, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict] = []
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    for doc, meta, dist in zip(docs, metas, dists):
        output.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": round(1.0 - dist, 4),  # cosine similarity
            }
        )
    return output
