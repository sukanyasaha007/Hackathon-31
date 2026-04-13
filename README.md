# Agentica — Agentic CRM for Global Trade

An AI-powered agent that classifies products into HS/HTS tariff codes, calculates landed costs, generates compliance documents, and orchestrates enterprise integrations — all from a single chat interface. Built for customs brokers, trade compliance teams, and importers navigating the 2025-2026 tariff landscape.

## Problem

Tariff classification is a **$2T+ trade compliance bottleneck**. Companies manually classify thousands of products against 18,000+ HTS line items, risking penalties of up to 40% of shipment value for misclassification. Professional customs brokers charge $50-150 per classification.

But classification is just the start. After getting the HTS code, importers must:
- Calculate landed costs (duties + freight + insurance + fees)
- Check Section 301, IEEPA, and reciprocal tariff surcharges by country of origin
- Generate customs invoices, supplier cost-sharing letters, and surcharge notices
- Screen products against export controls (BIS EAR, denied party lists)
- Sync classification data to SAP, Zoho, QuickBooks, and other enterprise systems
- File exemption requests with USTR when applicable

Each step involves a different system, a different team, and a different manual workflow. Agentica collapses this into a single conversational agent.

## Solution

Agentica is an **agentic CRM for global trade** — a chat-based AI agent that classifies products, then orchestrates a pipeline of 18 MCP-style tool calls across documents, communications, regulatory, enterprise integrations, and analytics.

### Core Pipeline

```
User → "Lithium-ion battery 48V for e-bike from China"
                    │
        ┌───────────┴───────────┐
        │   5-Stage Pipeline    │
        │                       │
        │  1. RETRIEVE          │  Hybrid vector + BM25 search over 2,186 chunks
        │  2. CLASSIFY          │  LLM applies GRI Rules 1-6 with retrieved context
        │  3. ENRICH            │  Add duty rates, CROSS rulings, alternative codes
        │  4. GUARDRAILS        │  Section 301/IEEPA/reciprocal tariff warnings
        │  5. CLARIFY           │  Human-in-the-loop questions if confidence < 70%
        │                       │
        └───────────┬───────────┘
                    │
        HTS 8507.60.0090 — 0% MFN + 25% Section 301
        │
        ├── "run full pipeline"
        │   ├── Landed Cost Calculator
        │   ├── Customs Invoice Generator
        │   └── Zoho CRM Sync
        │
        ├── "draft supplier letter" → cost-sharing negotiation
        ├── "send via gmail" → email delivery
        ├── "push to SAP" → material master update
        └── "export controls check" → BIS EAR screening
```

### CRM Deal Pipeline

Every classification creates a **deal** that progresses through 5 stages:

```
New Deal → Classified → Costed → Reviewed → CRM Synced
```

The "run full pipeline" command auto-executes 4 tool calls in sequence (landed cost → invoice → compliance review → CRM sync), advancing the deal through all stages in one command.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   Streamlit Chat UI                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Chat Panel   │  │ CRM Pipeline │  │ Tool Showcase   │  │
│  │ st.chat_msg  │  │ 5-stage bar  │  │ 18 MCP tools    │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
├───────────────────────────────────────────────────────────┤
│              TariffAgent Orchestrator                      │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────┐ ┌──────┐ │
│  │ RETRIEVE │→│ CLASSIFY │→│ ENRICH │→│GUARD │→│CLARFY│ │
│  │ (local)  │ │  (LLM)   │ │(local) │ │RAILS │ │(HITL)│ │
│  └──────────┘ └──────────┘ └────────┘ └──────┘ └──────┘ │
├───────────────────────────────────────────────────────────┤
│  Hybrid RAG Search         │  LLM (OpenAI-compatible)     │
│  • Vector (MiniLM-L6-v2)  │  • Llama 3.3 70B (Groq)     │
│  • BM25 full-text search   │  • Llama 3.1 8B (fallback)  │
│  • Cross-encoder re-rank   │  • Auto-switchover on quota  │
│  • RRF fusion (k=60)      │                               │
├───────────────────────────────────────────────────────────┤
│  LanceDB (2,186 chunks)   │  USITC Live API              │
│  HTS Schedule + GRI +     │  hts.usitc.gov/reststop      │
│  283 CBP CROSS Rulings    │  Real-time duty rate lookup   │
└───────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hybrid search (vector + BM25)** | HTS codes are structured data — BM25 catches exact tariff numbers that vector search misses. RRF fusion combines both signals. |
| **Cross-encoder re-ranking** | ms-marco-MiniLM-L-6-v2 re-ranks top candidates after fusion. Cheap CPU inference, significant accuracy gain. |
| **Duty rate validation** | LLMs hallucinate duty rates. We cross-check against (1) retrieved HTS text, then (2) live USITC API. Unverified rates are flagged. |
| **Auto model fallback** | Groq's 70B model has daily token limits. When exhausted, the client auto-switches to 8B with zero user-facing errors. |
| **Country-aware guardrails** | Static rule engine (no LLM call) applies Section 301, IEEPA, reciprocal tariff warnings based on country of origin. Fast and deterministic. |

## MCP Tools (18)

Every post-classification action is a discrete, auditable MCP-style tool call with structured input/output and timing metrics.

### Documents
| Tool | Description |
|------|-------------|
| `generate_invoice` | U.S. customs import invoice with duty estimates, MPF, harbor fees |
| `landed_cost_calculator` | Full CIF breakdown: FOB + freight + insurance + MFN + Section 301 |

### Communications
| Tool | Description |
|------|-------------|
| `draft_supplier_letter` | LLM-generated cost-sharing negotiation letter referencing specific HTS/duties |
| `draft_surcharge_notice` | Customer tariff surcharge announcement (5-15% pass-through) |
| `send_email` | Send classification report via Gmail or Outlook MCP integration |
| `slack_notify` | Post formatted tariff alert to Slack channel |
| `teams_notify` | Microsoft Teams adaptive card notification |

### Regulatory
| Tool | Description |
|------|-------------|
| `draft_exemption_request` | USTR Section 301 / IEEPA exclusion request template |
| `export_controls_check` | BIS EAR CCL + denied party list screening with risk assessment |
| `hts_lookup` | Live USITC API query — returns MFN, special/FTA, and Column 2 rates |

### Enterprise Integrations
| Tool | Description |
|------|-------------|
| `zoho_create_invoice` | Draft invoice in Zoho Books with HTS code + duty line items |
| `zoho_create_deal` | Create opportunity in Zoho CRM for tariff mitigation tracking |
| `wave_create_invoice` | Wave Accounting invoice with customs duty lines |
| `quickbooks_invoice` | QuickBooks Online invoice with tariff surcharge line |
| `sap_update_material` | SAP S/4HANA material master update (HTS code, commodity code, duty rate, CoO) |
| `erp_update_po` | Purchase order update for SAP/Oracle/NetSuite with tariff-adjusted pricing |

### Analytics
| Tool | Description |
|------|-------------|
| `tariff_impact_report` | Executive-level tariff impact analysis with mitigation strategies |
| `batch_classify` | Classify up to 20 products from CSV — portfolio-level tariff analysis |

## Copilot Skills

The project includes a `.github/skills/tariff/` skill for VS Code Copilot agent integration. This enables natural language commands for tariff operations:

```
/tariff sync latest CROSS rulings
/tariff classify lithium battery from China
/tariff rebuild vector index
/tariff check duty rates for chapter 85
```

The skill covers: classification, CROSS sync, HTS data download, index rebuild, duty/guardrail lookups, MCP tool calls, benchmarks, vector search, and UI restart procedures.

## Memory & Context Management

### Short-Term Memory (Session State)
- **Chat history**: Full conversation stored in `st.session_state.messages` — persists across Streamlit reruns within a session
- **Classification history**: Every classification result cached in `st.session_state.history` with timestamps, enabling "refine" and "compare" follow-ups
- **CRM deal state**: Active deal object (`st.session_state.deal`) tracks pipeline stage, tool calls used, and classification data
- **Uploaded products**: CSV batch results held in `st.session_state.uploaded_products` for batch operations

### Long-Term Memory (Persistent Storage)
- **Vector store**: LanceDB on disk (`data/indexes/`) — 2,186 embedded chunks from HTS chapters and CROSS rulings. Survives restarts.
- **Sync state**: `data/sync_state.json` tracks last CROSS sync timestamp and ruling counts — enables incremental updates without re-downloading
- **HTS/CROSS PDFs**: Raw source documents persisted in `data/hts/` and `data/cross/` — can be re-parsed and re-embedded at any time

### Context Window Management
- **Retrieval budget**: Hybrid search returns top 4 HTS chunks + top 3 CROSS rulings (7 chunks total) — fits within a single LLM context window
- **Progressive disclosure**: UI uses collapsible `<details>` elements for reasoning, rulings, and tool outputs — keeps the chat compact while preserving full audit trail
- **Query expansion**: The orchestrator generates multiple search queries from a single product description to improve recall across different HTS chapter structures

### Data Freshness
- **CROSS rulings sync**: `TariffSync.run()` queries `rulings.cbp.gov/api/stat/lastupdate`, downloads only new PDFs, embeds and appends to the vector store — no full re-index needed
- **Live USITC API**: The `hts_lookup` tool and duty rate validator query `hts.usitc.gov/reststop/search` in real-time for current MFN/special/Column 2 rates

## Example Prompts

### Basic Classification
```
Lithium-ion battery pack 48V for electric bicycle from China
```
→ Returns HTS 8507.60.0090, 0% MFN duty, Section 301 + IEEPA warnings, 90% confidence

### Full Pipeline (Auto-Workflow)
```
run full pipeline
```
→ After a classification, auto-executes: Landed Cost Calculator → Invoice Generator → Zoho CRM Sync. Deal advances through all 5 pipeline stages.

### Country Comparison
```
compare countries
```
→ Classifies across China, Vietnam, Mexico, India, Japan — shows duty differences and advisory counts

### Supplier Negotiation
```
draft supplier cost-sharing letter
```
→ LLM-generated letter referencing the specific HTS code, duty rate, and tariff warnings

### Export Controls
```
export controls check for end user Huawei in China
```
→ Screens product against BIS EAR controlled items list and denied party lists

### Live Duty Rate Lookup
```
hts lookup 8507.60
```
→ Queries USITC API for real-time MFN, special/FTA, and Column 2 rates

### Batch Classification (CSV Upload)
Upload a CSV with columns: `product`, `country`, `quantity`, `value` — the agent classifies all rows and returns a portfolio summary table.

### Enterprise Integrations
```
push to SAP material master
send report via gmail
create Zoho CRM deal
```

## Benchmark Results

10-case benchmark across easy, medium, and hard tariff classifications:

| Metric | RAG Agent (8B fallback) | RAG Agent (70B) | No RAG Baseline |
|--------|-------------------------|-----------------|-----------------|
| Chapter (2-digit) | 70% | 80% | 80% |
| Heading (4-digit) | 50% | 70% | 60% |
| Subheading (6-digit) | 20% | 40% | 30% |

| Difficulty | 8B Fallback | 70B | Baseline |
|------------|-------------|-----|----------|
| Easy (3 cases) | **100%** | **100%** | 100% |
| Medium (4 cases) | 50% | 50% | 50% |
| Hard (3 cases) | 0% | **67%** | 33% |

Hard cases (ambiguous products with competing headings) benefit significantly from larger models. The auto-fallback ensures availability when the 70B quota is exhausted.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama 3.3 70B via Groq (auto-fallback to 8B) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local, CPU, 384-dim) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (local, CPU) |
| Vector Store | LanceDB (hybrid BM25 + vector, 2,186 chunks) |
| Live Data API | USITC HTS REST API (hts.usitc.gov) |
| Data Sources | US HTS tariff schedule (15 chapters) + CBP CROSS rulings (283 indexed) |
| Frontend | Streamlit |
| All open source | No proprietary dependencies |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your GROQ_API_KEY
```

## Run

```bash
source venv/bin/activate
streamlit run src/ui/app.py --server.port 8501
```

## Data Pipeline

```bash
# Download HTS chapters + CROSS rulings from USITC/CBP
python3 -m src.data.download

# Parse PDFs, embed, build LanceDB index
python3 -m src.data.ingest

# Incremental CROSS sync (check for new rulings, download, embed)
python3 -m src.data.sync
```

## Evaluate

```bash
# Full RAG pipeline benchmark (10 cases)
python3 -m src.eval.run_rag_benchmark

# Baseline (LLM only, no RAG)
python3 -m src.eval.run_benchmark
```

## Project Structure

```
agentica/
├── src/
│   ├── agent/
│   │   ├── orchestrator.py   # 5-stage classification pipeline
│   │   └── tools.py          # 18 MCP-style tools
│   ├── rag/
│   │   └── search.py         # Hybrid vector+BM25 search, cross-encoder re-ranking
│   ├── data/
│   │   ├── download.py       # HTS + CROSS data acquisition
│   │   ├── ingest.py         # PDF parsing, embedding, LanceDB index build
│   │   ├── parsers.py        # HTS/CROSS PDF extraction
│   │   └── sync.py           # Incremental CROSS sync via CBP API
│   ├── eval/
│   │   ├── run_benchmark.py  # No-RAG baseline evaluation
│   │   └── run_rag_benchmark.py  # Full pipeline evaluation
│   ├── ui/
│   │   └── app.py            # Chat-based Streamlit UI with CRM pipeline
│   ├── llm.py                # Groq client with auto-fallback (70B → 8B)
│   └── config.py             # Environment config and paths
├── data/
│   ├── hts/                  # HTS chapter PDFs
│   ├── cross/                # CROSS ruling PDFs (ny/, hq/)
│   ├── indexes/              # LanceDB vector store
│   └── sync_state.json       # Sync state persistence
├── .github/
│   └── skills/tariff/        # Copilot agent skill for tariff operations
├── requirements.txt
└── README.md
```

## Data Sources

- **HTS Schedule**: Official Harmonized Tariff Schedule PDFs from [USITC](https://hts.usitc.gov/)
- **CROSS Rulings**: CBP Customs Rulings Online Search System via [CBP API](https://rulings.cbp.gov/)
- **Live Rates**: USITC HTS REST API (`hts.usitc.gov/reststop/search`)
- **GRI Rules**: General Rules of Interpretation (built into classification prompt)

## License

MIT
