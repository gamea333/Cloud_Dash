"""Knowledge base ingestion: chunk and store articles in ChromaDB."""

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
COLLECTION_NAME = "clouddash_kb"


def get_embedding_function() -> embedding_functions.DefaultEmbeddingFunction:
    """ChromaDB default: all-MiniLM-L6-v2 via ONNX (lower memory than sentence-transformers)."""
    return embedding_functions.DefaultEmbeddingFunction()


def get_collection(client: chromadb.PersistentClient, name: str = COLLECTION_NAME):
    return client.get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_articles(articles_dir: Path) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    for path in sorted(articles_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            articles.append(json.load(f))
    return articles


def ingest(
    articles_dir: str | Path | None = None,
    persist_dir: str | None = None,
    reset: bool = False,
) -> int:
    """
    Load JSON articles, chunk content, and persist to ChromaDB.
    Embeddings are computed by ChromaDB's DefaultEmbeddingFunction.
    Returns number of chunks indexed.
    """
    base = Path(__file__).resolve().parent.parent
    articles_path = Path(articles_dir) if articles_dir else base / "knowledge_base" / "articles"
    chroma_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    articles = load_articles(articles_path)
    if not articles:
        raise FileNotFoundError(f"No JSON articles found in {articles_path}")

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass

    collection = get_collection(client)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for article in articles:
        article_id = article["id"]
        title = article["title"]
        category = article["category"]
        content = article["content"]
        chunks = _chunk_text(content)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{article_id}_chunk_{idx}"
            enriched = f"[{title}] {chunk}"

            ids.append(chunk_id)
            documents.append(enriched)
            metadatas.append(
                {
                    "article_id": article_id,
                    "title": title,
                    "category": category,
                    "chunk_index": idx,
                    "tags": ",".join(article.get("tags", [])),
                }
            )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    return len(ids)


if __name__ == "__main__":
    count = ingest(reset=True)
    print(f"Ingested {count} chunks into ChromaDB")
