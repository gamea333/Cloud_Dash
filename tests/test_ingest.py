"""Tests for knowledge base ingestion and chunking."""

import json
import tempfile
from pathlib import Path

from retrieval.ingest import _chunk_text, load_articles


def test_chunk_text_short():
    text = "Short text"
    chunks = _chunk_text(text, chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long():
    text = "A" * 1000
    chunks = _chunk_text(text, chunk_size=512, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 512 for c in chunks)


def test_load_articles():
    articles_dir = Path(__file__).resolve().parent.parent / "knowledge_base" / "articles"
    articles = load_articles(articles_dir)
    assert len(articles) == 20
    for article in articles:
        assert "id" in article
        assert "title" in article
        assert "category" in article
        assert "content" in article
        assert len(article["content"]) > 100


def test_article_schema():
    articles_dir = Path(__file__).resolve().parent.parent / "knowledge_base" / "articles"
    for path in sorted(articles_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            article = json.load(f)
        required = {"id", "title", "category", "tags", "content", "last_updated", "applies_to"}
        assert required.issubset(article.keys())
