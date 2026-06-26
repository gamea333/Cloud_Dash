"""Knowledge base ingestion: chunk, embed, and store articles in ChromaDB."""

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "clouddash_kb"


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
    Load JSON articles, chunk content, embed with sentence-transformers,
    and persist to ChromaDB. Returns number of chunks indexed.
    """
    base = Path(__file__).resolve().parent.parent
    articles_path = Path(articles_dir) if articles_dir else base / "knowledge_base" / "articles"
    chroma_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    articles = load_articles(articles_path)
    if not articles:
        raise FileNotFoundError(f"No JSON articles found in {articles_path}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    for article in articles:
        article_id = article["id"]
        title = article["title"]
        category = article["category"]
        content = article["content"]
        chunks = _chunk_text(content)

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{article_id}_chunk_{idx}"
            enriched = f"[{title}] {chunk}"
            embedding = model.encode(enriched, normalize_embeddings=True).tolist()

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
            embeddings.append(embedding)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(ids)


if __name__ == "__main__":
    count = ingest(reset=True)
    print(f"Ingested {count} chunks into ChromaDB")
