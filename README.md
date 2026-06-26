# CloudDash Multi-Agent Customer Support

## Overview

CloudDash Support is a multi-agent AI customer service system for **CloudDash**, a fictional cloud infrastructure monitoring SaaS. It routes customer messages through a Triage agent to specialist agents (Technical Support, Billing, Escalation), retrieves answers from a local knowledge base via RAG, and coordinates agent handovers with structured logging and guardrails.

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              Knowledge Base (JSON)           │
                         │         knowledge_base/articles/           │
                         └─────────────────────┬───────────────────────┘
                                               │ ingest + embed
                                               ▼
┌──────────┐    ┌──────────┐    ┌─────────────────────┐    ┌──────────────┐
│   User   │───▶│ FastAPI  │───▶│    Orchestrator     │───▶│    Triage    │
└──────────┘    │  /api    │    │  (route + guardrails)│    └──────┬───────┘
                └──────────┘    └──────────┬──────────┘           │
                                           │                        │ handover
                                           │              ┌─────────┼─────────┐
                                           │              ▼         ▼         ▼
                                           │      Technical   Billing   Escalation
                                           │         Support
                                           ▼
                                  ┌────────────────┐
                                  │ Handover Manager│◀──── entity + context transfer
                                  └────────────────┘
                                           │
                                           ▼
                                  ┌────────────────┐
                                  │  RAG Pipeline  │──── ChromaDB (local vectors)
                                  │   retriever    │──── sentence-transformers
                                  └────────────────┘
                                           │
                                           ▼
                                  ┌────────────────┐
                                  │   Groq LLM     │──── llama-3.1-8b-instant
                                  └────────────────┘
```

## Setup Instructions

### 1. Clone and create a virtual environment

```powershell
git clone <your-repo-url>
cd CloudDash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

```powershell
copy .env.example .env
```

Edit `.env` and set your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
CHROMA_PERSIST_DIR=./chroma_db
LOG_LEVEL=INFO
```

Get a free API key at [https://console.groq.com](https://console.groq.com).

### 4. Ingest the knowledge base

```powershell
python retrieval/ingest.py
```

This chunks 20 KB articles, embeds them with `all-MiniLM-L6-v2`, and stores vectors in ChromaDB.

### 5. Run the API server

```powershell
python -m uvicorn api.main:app --reload
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 6. Run tests

```powershell
python -m pytest tests/ -v
```

## Running the 4 Test Scenarios

Start the server (`uvicorn api.main:app --reload`), then run these PowerShell commands.

### Scenario 1 — Technical Support (alerts not firing)

```powershell
$conv = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/conversations"
$convId = $conv.conversation_id

Invoke-RestMethod -Method POST `
  -Uri "http://localhost:8000/conversations/$convId/messages" `
  -ContentType "application/json" `
  -Body '{"content": "My alerts stopped firing on AWS CloudWatch"}'
```

**Expected:** Triage routes to Technical Support; response includes numbered troubleshooting steps and KB citations (e.g. KB-005).

### Scenario 2 — Billing (invoice question)

```powershell
$conv = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/conversations"
$convId = $conv.conversation_id

Invoke-RestMethod -Method POST `
  -Uri "http://localhost:8000/conversations/$convId/messages" `
  -ContentType "application/json" `
  -Body '{"content": "I have a question about my invoice and Pro plan charges"}'
```

**Expected:** Triage routes to Billing; response references account/invoice details and billing KB policies.

### Scenario 3 — Escalation (manager request)

```powershell
$conv = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/conversations"
$convId = $conv.conversation_id

Invoke-RestMethod -Method POST `
  -Uri "http://localhost:8000/conversations/$convId/messages" `
  -ContentType "application/json" `
  -Body '{"content": "This is unacceptable. I need to speak to a manager about a $600 refund."}'
```

**Expected:** Routes to Billing then Escalation; response includes `HUMAN OPERATOR ALERT` block and priority assignment.

### Scenario 4 — Guardrail rejection (off-topic)

```powershell
$conv = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/conversations"
$convId = $conv.conversation_id

Invoke-RestMethod -Method POST `
  -Uri "http://localhost:8000/conversations/$convId/messages" `
  -ContentType "application/json" `
  -Body '{"content": "Who won the cricket match yesterday?"}'
```

**Expected:** Input guardrail blocks the message; polite rejection explaining CloudDash-only scope.

### Bonus — View conversation history and handover log

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/conversations/$convId/history"
Invoke-RestMethod -Uri "http://localhost:8000/conversations/$convId/handover-log"
```

## Design Decisions

| Choice | Rationale |
|--------|-----------|
| **ChromaDB** | Lightweight embedded vector store; no external DB service; persists to disk; ideal for demos and Render free tier |
| **FastAPI** | Async-ready, automatic OpenAPI docs, Pydantic validation, minimal boilerplate |
| **State machine over graph framework** | Four agents with explicit handover rules are simple enough for a hand-rolled orchestrator; avoids LangGraph/LangChain complexity for a focused support flow |
| **sentence-transformers (`all-MiniLM-L6-v2`)** | Runs locally, no API key, fast embeddings, good quality for short KB articles |
| **Groq** | Very fast inference, generous free tier, supports JSON-mode classification and agent responses via `llama-3.1-8b-instant` |

## Known Limitations

- **In-memory conversation state** — conversations are lost on server restart (no PostgreSQL/Redis persistence)
- **Mock billing data** — account lookups generate deterministic fake data from `customer_id`; not connected to a real billing system
- **Regex guardrails** — prompt injection and off-topic detection use patterns, not a dedicated safety model; edge cases may slip through
- **Groq rate limits** — free tier has request/token limits; production would need paid tier and caching
- **Ephemeral ChromaDB on Render free tier** — filesystem may reset between deploys; rebuild runs `ingest.py` on each deploy

## Deployment

### Render.com

1. Push repo to GitHub
2. Create a new **Web Service** on Render and connect the repo
3. Render reads `render.yaml` automatically
4. Set `GROQ_API_KEY` in the Render dashboard (Environment)
5. Deploy

**Live URL:** Deployed at: https://YOUR-APP.onrender.com

### Docker

```powershell
docker build -t clouddash-support .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key clouddash-support
```

## Project Structure

```
CloudDash/
├── agents/           # Triage, Technical, Billing, Escalation + Orchestrator
├── api/              # FastAPI app and conversation service
├── config/           # agents.yaml, routing_rules.yaml
├── handover/         # Handover protocol manager
├── knowledge_base/   # 20 JSON support articles
├── retrieval/        # ChromaDB ingest + RAG retriever
├── tests/            # pytest suite (mocked LLM calls)
├── utils/            # Groq client, guardrails, structlog logger
├── models.py         # Pydantic models
├── render.yaml       # Render.com deployment config
├── Dockerfile
└── requirements.txt
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/conversations` | Create conversation; returns `conversation_id` + `trace_id` |
| `POST` | `/conversations/{id}/messages` | Send message; returns `AgentResponse` |
| `GET` | `/conversations/{id}/history` | Full message history |
| `GET` | `/conversations/{id}/handover-log` | Handover events for conversation |
| `GET` | `/health` | Health check |

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component and data-flow documentation.
