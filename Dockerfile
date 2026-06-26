FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python retrieval/ingest.py

EXPOSE 8000

ENV CHROMA_PERSIST_DIR=/app/chroma_db
ENV LOG_LEVEL=INFO

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
