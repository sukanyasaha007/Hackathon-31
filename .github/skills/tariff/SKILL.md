---
name: tariff
description: "Tariff classification, duty rates, CROSS rulings sync, HTS data management, and trade compliance operations. Use when: classifying products into HTS codes, syncing CROSS rulings from CBP, checking duty rates, downloading HTS chapter data, rebuilding the vector index, running benchmarks, inspecting guardrails, or troubleshooting the RAG pipeline."
argument-hint: "Describe the tariff operation: e.g. 'sync latest CROSS rulings', 'classify lithium battery from China', 'rebuild vector index', 'check duty rate for chapter 85'"
---

# Tariff Operations Skill

Covers all tariff classification, duty lookup, data pipeline, and trade compliance operations in the Agentica project.

## Project Layout

```
src/
├── agent/orchestrator.py   # 5-stage classification pipeline
├── agent/tools.py          # 17 MCP-style post-classification tools
├── rag/search.py           # Hybrid vector+BM25 search with cross-encoder
├── data/download.py        # Download HTS chapters + CROSS rulings from USITC/CBP
├── data/ingest.py          # Parse PDFs, embed chunks, build LanceDB index
├── data/parsers.py         # PDF extraction for HTS headings, notes, CROSS rulings
├── data/sync.py            # Incremental CROSS sync (check → download → ingest)
├── llm.py                  # Groq LLM client (70B primary, 8B fallback)
├── config.py               # Paths, API keys, model names
├── eval/run_rag_benchmark.py  # 10-case benchmark suite
└── ui/app.py               # Streamlit chat UI with CRM pipeline
data/
├── hts/                    # HTS chapter PDFs (chapters 84, 85, 87, 90, 91, 95, 96)
├── cross/ny/ cross/hq/     # CROSS ruling PDFs by collection
├── indexes/                # LanceDB vector store (table: tariff_chunks, 2186 rows)
└── sync_state.json         # Last sync timestamp and counts
```

## Environment Setup

Every command requires the virtual environment and libffi fix:

```bash
cd "/home/ukasahct/work/AI Agent hackathon/agentica"
source venv/bin/activate
export LD_LIBRARY_PATH="$HOME/lib:$LD_LIBRARY_PATH"
```

## Procedures

### 1. Classify a Product

Run a single classification through the 5-stage pipeline (Retrieve → Classify → Enrich → Guardrails → Clarify):

```python
from src.agent.orchestrator import TariffAgent
agent = TariffAgent()
result = agent.classify("product description here", country_of_origin="China")
```

Key fields on `ClassificationResult`: `hts_code`, `hts_heading`, `hts_subheading`, `duty_rate`, `confidence`, `reasoning`, `tariff_warnings`, `clarifying_questions`, `needs_expert_review`.

Confidence < 0.7 triggers expert review flag. Country of origin activates tariff warnings (Section 301, IEEPA, reciprocal tariffs).

### 2. Sync CROSS Rulings (Incremental)

Checks CBP API for new rulings, downloads only new PDFs, embeds and adds to vector store:

```bash
python3 -m src.data.sync
```

Or programmatically:
```python
from src.data.sync import TariffSync
sync = TariffSync()
summary = sync.run()  # Returns {new_rulings: int, new_chunks: int, last_update: str}
```

API endpoints used:
- `https://rulings.cbp.gov/api/stat/lastupdate` — check for updates
- `https://rulings.cbp.gov/api/downloadLatestDocuments` — download ZIP

State persisted at `data/sync_state.json`.

### 3. Download Full HTS + CROSS Data

Downloads HTS chapter PDFs from USITC and CROSS rulings from CBP:

```bash
python3 -m src.data.download
```

Sources:
- HTS chapters: `https://hts.usitc.gov/reststop/file` (chapters 84, 85, 87, 90, 91, 95, 96 + GRI + General Notes)
- CROSS rulings: `https://rulings.cbp.gov/api/downloadLatestDocuments` (NY and HQ collections)

### 4. Rebuild Vector Index

Re-parses all PDFs, generates embeddings, and creates fresh LanceDB table with FTS index:

```bash
python3 -m src.data.ingest
```

Pipeline: parse_all_hts() + parse_all_cross() → batch embed (MiniLM-L6-v2, 384-dim) → create LanceDB table → build FTS index on "text" column.

### 5. Check Duty Rates and Guardrails

The guardrails system in `orchestrator.py` applies country-aware duty rate warnings:

| Category | Countries | Treatment |
|----------|-----------|-----------|
| High tariff | China, Hong Kong, Russia, Belarus | Section 301 (up to 25%), IEEPA, Column 2 rates |
| Moderate | Vietnam (46%), India (26%), Thailand (36%), Indonesia (32%), Taiwan (32%), S. Korea (25%), Japan (24%) | Reciprocal tariff warnings |
| FTA | Canada, Mexico, Australia, Singapore, Chile, Peru, Colombia, Panama, Israel, Jordan, Bahrain, Oman, Morocco | Preferential/zero duty note |

At-risk HTS chapters for trade remedies: 84, 85, 87, 90, 73, 39, 40, 94.

Duty rate validation cross-checks LLM output against HTS source text via `_validate_duty_rate()`.

### 6. Use Post-Classification Tools

After classification, call any of the 17 MCP tools:

```python
from src.agent.tools import call_tool
result = call_tool("landed_cost_calculator", classification, qty=100, unit_value=50.0, freight=500, insurance=100)
```

Tool categories: documents (invoice, landed cost), communications (supplier letter, surcharge, email, slack, teams), regulatory (exemption, export controls), integrations (Zoho, Wave, QuickBooks, SAP, ERP), analytics (impact report, batch classify).

### 7. Run Benchmarks

10-case benchmark across easy/medium/hard tariff classifications:

```bash
python3 -m src.eval.run_rag_benchmark
```

### 8. Search the Vector Store Directly

```python
from src.rag.search import TariffSearcher
searcher = TariffSearcher()
results = searcher.search("electric motor", top_k=5, rerank=True)
# Filter by chunk type: "hts_heading", "hts_note", "cross_ruling"
results = searcher.search("electric motor", chunk_types=["cross_ruling"])
```

### 9. Restart the Streamlit UI

```bash
pkill -f "streamlit run" 2>/dev/null; sleep 2
streamlit run src/ui/app.py --server.port 8501 --server.headless true
```

## Troubleshooting

- **Groq 70B quota exhausted**: The LLM client auto-falls back to Llama 3.1 8B. No action needed.
- **libffi.so.6 not found**: Run `export LD_LIBRARY_PATH="$HOME/lib:$LD_LIBRARY_PATH"` before any Python command.
- **Empty search results**: Rebuild index with `python3 -m src.data.ingest`.
- **Stale CROSS rulings**: Run `python3 -m src.data.sync` to pull latest from CBP.
- **Port 8501 in use**: Kill with `pkill -9 -f "streamlit run"`, wait 2 seconds, restart.
