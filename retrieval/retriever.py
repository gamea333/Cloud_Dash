"""RAG retriever with query rewriting and citation formatting."""

import os
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings

from models import Message
from retrieval.ingest import COLLECTION_NAME, get_collection
from utils.groq_client import FAST_MODEL, GroqClient
from utils.logger import SupportLogger

REWRITE_PROMPT = """Rewrite the user's latest question into a standalone search query for a knowledge base.
Use the conversation history for context. Output ONLY the rewritten query, nothing else.

Conversation history:
{history}

Latest user message: {query}

Rewritten search query:"""


class KnowledgeRetriever:
    """Retrieve relevant KB chunks with optional query rewriting via Groq."""

    def __init__(
        self,
        persist_dir: str | None = None,
        groq_client: Optional[GroqClient] = None,
        logger: Optional[SupportLogger] = None,
    ):
        chroma_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = get_collection(self.client, name=COLLECTION_NAME)
        self.groq = groq_client
        self.logger = logger

    def query_rewriter(
        self,
        query: str,
        conversation_history: list[Message],
    ) -> str:
        recent = conversation_history[-3:] if conversation_history else []
        if not recent or self.groq is None:
            return query

        history_text = "\n".join(
            f"{msg.role.value}: {msg.content}" for msg in recent
        )
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)
        rewritten = self.groq.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=FAST_MODEL,
            temperature=0.0,
            max_tokens=256,
        )
        return rewritten.strip() or query

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        conversation_history: Optional[list[Message]] = None,
        rewrite: bool = True,
    ) -> list[dict[str, Any]]:
        search_query = query
        if rewrite and conversation_history is not None:
            search_query = self.query_rewriter(query, conversation_history)

        results = self.collection.query(
            query_texts=[search_query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return chunks

        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "article_id": metadata.get("article_id", "unknown"),
                    "title": metadata.get("title", "Unknown"),
                    "category": metadata.get("category", ""),
                    "distance": results["distances"][0][i] if results["distances"] else None,
                }
            )

        if self.logger:
            self.logger.kb_retrieved(
                query=search_query,
                sources=[c["article_id"] for c in chunks],
            )
        return chunks

    @staticmethod
    def format_citations(chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return ""
        seen: set[str] = set()
        citations: list[str] = []
        for chunk in chunks:
            article_id = chunk.get("article_id", "unknown")
            if article_id in seen:
                continue
            seen.add(article_id)
            title = chunk.get("title", "Unknown")
            citations.append(f"Source: {article_id} | {title}")
        return "\n".join(citations)

    def ensure_indexed(self, articles_dir: str | Path | None = None) -> None:
        """Ingest KB if collection is empty."""
        if self.collection.count() == 0:
            from retrieval.ingest import ingest

            ingest(articles_dir=articles_dir)
