# CloudDash Support — Architecture

## Component Breakdown

### `api/main.py` + `api/service.py`
FastAPI HTTP layer and conversation lifecycle manager. `ConversationService` holds in-memory `ConversationState` objects and delegates each user message to the `Orchestrator`. Exposes REST endpoints for creating conversations, sending messages, and retrieving history/handover logs.

### `agents/orchestrator.py`
Central routing coordinator. Runs input guardrails, appends user messages, invokes Triage or the current specialist agent, chains handovers up to three hops, applies output guardrails, and records assistant replies. Bootstraps all agents and the `HandoverManager` via `Orchestrator.create()`.

### `agents/triage_agent.py`
First-line classifier. Uses Groq JSON output (`llama-3.1-8b-instant`) to extract intent, entities, and target agent. Never answers product questions directly — only routes. Low confidence (&lt; 0.6) forces Escalation.

### `agents/technical_agent.py`
KB-grounded troubleshooting agent. Retrieves top-3 chunks, filters by cosine similarity threshold, returns numbered steps with code examples, and offers escalation when no verified article matches.

### `agents/billing_agent.py`
Handles invoices, plans, and payments. Generates deterministic mock account data from `customer_id`, cites billing KB articles, simulates plan changes, and escalates when refund authority is exceeded or a manager is requested.

### `agents/escalation_agent.py`
Packages the full conversation into an `EscalationPackage` with priority, sentiment, and recommended action. Outputs a formatted `HUMAN OPERATOR ALERT` block and marks the conversation status as `escalated`.

### `handover/handover_manager.py`
Executes agent-to-agent transfers. Validates target agent exists, merges extracted entities into conversation state, writes immutable `HandoverLog` entries with context snapshots, and falls back to Triage on failure.

### `retrieval/ingest.py` + `retrieval/retriever.py`
RAG pipeline. `ingest.py` chunks articles (512 tokens, 50 overlap), stores in ChromaDB with `DefaultEmbeddingFunction` (all-MiniLM-L6-v2 via ONNX). `retriever.py` rewrites queries with conversation context via Groq, retrieves top-k chunks, and formats citations.

### `utils/groq_client.py`
Groq SDK wrapper with exponential backoff on rate limits, JSON parsing, and per-call latency/token logging.

### `utils/guardrails.py`
Fast regex/keyword input and output safety checks. Blocks prompt injection and off-topic queries; redacts PII; blocks unverified product claims when KB retrieval returned no chunks.

### `utils/logger.py`
structlog JSON logger emitting `trace_id`, `conversation_id`, `event_type`, and `agent_name` on every event (AGENT_INVOKED, KB_RETRIEVED, HANDOVER_*, ESCALATION_TRIGGERED, GUARDRAIL_TRIGGERED).

### `models.py`
Pydantic models: `Message`, `ConversationState`, `AgentResponse`, `HandoverPayload`, `HandoverLog`, `EscalationPackage`, `TriageResult`, guardrail result types.

### `config/agents.yaml` + `config/routing_rules.yaml`
Per-agent system prompts, tool lists, escalation thresholds, model selection, and intent-to-agent routing map.

---

## Data Flow — Scenario 1: Technical Support

**User message:** "My alerts stopped firing on AWS CloudWatch"

1. `POST /conversations/{id}/messages` → `ConversationService.process_message()`
2. `Orchestrator.route()` runs **input guardrails** → message allowed (contains "alert", "AWS")
3. User message appended to `ConversationState.messages`
4. `current_agent == "Triage"` → **TriageAgent.run()**
5. Groq fast model returns: `{intent: "technical", target_agent: "TechnicalSupport", confidence: 0.88}`
6. Triage returns `AgentResponse(requires_handover=True, handover_target="TechnicalSupport")`
7. **HandoverManager.execute_handover()** — updates `current_agent`, preserves entities, logs `HandoverLog`
8. **TechnicalSupportAgent.run()** — `retriever.retrieve("alerts not firing")` → KB-005 chunks
9. Groq generates numbered troubleshooting steps with citations
10. **Output guardrails** — KB chunks present → response allowed; PII redaction if needed
11. Assistant message appended; `AgentResponse` returned to client

---

## Data Flow — Scenario 2: Billing

**User message:** "I have a question about my invoice and Pro plan charges"

1. API → Orchestrator → input guardrails pass ("invoice", "plan")
2. **TriageAgent** classifies: `{intent: "billing", target_agent: "Billing", confidence: 0.92}`
3. Handover Triage → Billing; `customer_id` entity extracted if present
4. **BillingAgent.run()** — `_mock_account_lookup(customer_id)` generates plan/invoice data
5. KB retrieval fetches KB-009 (plan comparison), KB-011 (invoice explanation)
6. Groq explains charges citing KB policy + mock invoice line items
7. Output guardrails pass; response returned

---

## Data Flow — Scenario 3: Escalation

**User message:** "I need to speak to a manager about a $600 refund"

1. Input guardrails pass ("refund" is a domain keyword)
2. Triage may route to **Billing** first (billing intent detected)
3. **BillingAgent** detects manager request + refund &gt; $500 authority limit
4. Returns `requires_handover=True, handover_target="Escalation"` with enriched entities (`invoice_details`, `sentiment`, `priority: high`)
5. Handover Billing → Escalation; handover log records full context snapshot
6. **EscalationAgent** builds `EscalationPackage`:
   - `priority: critical` (manager + high refund keywords)
   - `sentiment: frustrated/angry`
   - `recommended_action: Immediate callback...`
7. Formats `HUMAN OPERATOR ALERT` ASCII block
8. Sets `ConversationState.status = ESCALATED`
9. Logs `ESCALATION_TRIGGERED`; response returned

---

## Data Flow — Scenario 4: Guardrail Rejection

**User message:** "Who won the cricket match yesterday?"

1. API → Orchestrator → **input guardrails**
2. `_is_off_topic()` matches cricket/sports pattern
3. `_has_domain_overlap()` returns False (no CloudDash keywords)
4. `check_input()` returns `allowed=False`
5. Orchestrator returns polite rejection **without** invoking any agent or LLM
6. Logs `GUARDRAIL_TRIGGERED` (off_topic, block)
7. No handover, no KB retrieval, no Groq call

---

## Handover Protocol

```
┌─────────────┐     requires_handover=True      ┌──────────────────┐
│ Source Agent│ ──────────────────────────────▶ │ HandoverManager  │
│ (e.g.Triage)│     handover_target="Billing"   │ execute_handover │
└─────────────┘                                 └────────┬─────────┘
                                                         │
                    ┌────────────────────────────────────┼────────────────────────┐
                    │                                    │                        │
                    ▼                                    ▼                        ▼
           Validate target                      Merge entities            Write HandoverLog
           agent exists                         into state              + context snapshot
                    │                                    │                        │
                    ▼                                    ▼                        ▼
           Update current_agent              status = HANDOVER → ACTIVE    Log HANDOVER_COMPLETED
                    │
         ┌──────────┴──────────┐
         │ failure?          │
         ▼                   ▼
  current_agent="Triage"   Target agent.run()
  HANDOVER_FAILED          (specialist handles message)
```

**Handover payload includes:**
- `source_agent`, `target_agent`, `reason`
- `conversation_summary` (last agent work)
- `extracted_entities` (customer_id, plan_type, invoice_details, etc.)
- `priority` (normal / high for billing escalations)
- Full `context_snapshot` in `HandoverLog`

---

## Guardrail Decision Tree

```
                    User Input
                        │
                        ▼
              ┌─────────────────────┐
              │ Prompt injection    │──match──▶ BLOCK (allowed=False)
              │ patterns?           │
              └─────────┬───────────┘
                        │ no
                        ▼
              ┌─────────────────────┐
              │ Off-topic pattern   │──yes──▶ Has CloudDash keyword?──no──▶ BLOCK
              │ (sports/weather/    │                    │
              │  politics/etc.)?    │                    yes
              └─────────┬───────────┘                    │
                        │ no                             ▼
                        ▼                              ALLOW
                     ALLOW
                        │
                        ▼
              ┌─────────────────┐
              │  Agent Response  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │ PII in output?      │──yes──▶ REDACT (mask card/SSN)
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ KB chunks empty AND │──yes──▶ BLOCK → replace with
              │ specific product/   │         "don't have verified information"
              │ pricing claims?     │
              └─────────┬───────────┘
                        │ no
                        ▼
                     ALLOW → return to client
```

---

## Logging Events

| Event | When |
|-------|------|
| `CONVERSATION_CREATED` | New conversation started |
| `AGENT_INVOKED` | Any agent `run()` called; includes intent for Triage |
| `KB_RETRIEVED` | Retriever returns chunks |
| `HANDOVER_INITIATED` | Handover begins |
| `HANDOVER_COMPLETED` | Handover succeeds |
| `HANDOVER_FAILED` | Unknown agent or exception; fallback to Triage |
| `ESCALATION_TRIGGERED` | Escalation agent packages case |
| `GUARDRAIL_TRIGGERED` | Input blocked or output redacted/flagged |
| `LLM_CALL` | Every Groq request with model, tokens, latency |

All log entries include: `trace_id`, `timestamp`, `event_type`, `conversation_id`, `agent_name`.
