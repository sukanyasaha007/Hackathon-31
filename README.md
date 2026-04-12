# Agentica — AI Tariff Classification Agent

An AI-powered agent that classifies products into HS/HTS tariff codes using Retrieval-Augmented Generation (RAG) over official US trade data. Built for customs brokers, trade compliance teams, and importers who need fast, accurate tariff lookups with transparent reasoning.

## Problem

Tariff classification is a $2T+ trade compliance bottleneck. Companies manually classify thousands of products against 18,000+ HTS codes, risking costly penalties for misclassification. Professional customs brokers charge $50-150 per classification.

## Solution

Agentica uses a 3-stage agentic pipeline:
1. **RETRIEVE** — Hybrid search (vector + BM25 with reciprocal rank fusion) over HTS schedule entries and 220,000+ CBP CROSS rulings, re-ranked with a cross-encoder
2. **CLASSIFY** — LLM applies General Rules of Interpretation (GRI 1-6) with retrieved context to determine the correct HTS code
3. **ENRICH** — Adds duty rates, confidence scores, similar rulings, alternative codes, and expert review flags

### Key Features
- **Expert-in-the-loop**: Flags ambiguous classifications (confidence < 0.7) for human review
- **Transparent reasoning**: Shows GRI rules applied, supporting rulings, and alternative codes
- **Auto-update**: One-click sync with CBP CROSS database for latest rulings
- **Model-agnostic**: Swap LLM providers via environment variable (Groq, OpenAI, Gemini, etc.)

## Architecture

```
┌─────────────────────────────────────────────┐
│              Streamlit UI                    │
├─────────────────────────────────────────────┤
│           TariffAgent Orchestrator           │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ RETRIEVE  │→│ CLASSIFY  │→│  ENRICH   │ │
│  │ (local)   │  │  (LLM)   │  │ (local)   │ │
│  └──────────┘  └──────────┘  └───────────┘ │
├─────────────────────────────────────────────┤
│  Hybrid RAG Search    │  LLM (OpenAI-compat)│
│  • Vector (MiniLM-L6) │  • Llama 3.3 70B   │
│  • BM25 (FTS)         │  • via Groq API     │
│  • Cross-encoder      │                     │
│  • RRF fusion         │                     │
├─────────────────────────────────────────────┤
│            LanceDB (2,186 chunks)           │
│  HTS Chapters + GRI + CBP CROSS Rulings    │
└─────────────────────────────────────────────┘
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API key
```

## Run

```bash
source venv/bin/activate
streamlit run src/ui/app.py
```

## Evaluate

```bash
# Baseline (LLM only, no RAG)
python3 -m src.eval.run_benchmark

# Full RAG pipeline
python3 -m src.eval.run_rag_benchmark
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

The agent supports automatic model fallback: when the primary model's daily quota is exhausted, it seamlessly switches to a smaller model to maintain availability. Hard cases (ambiguous products with competing headings) benefit significantly from larger models.

## Tech Stack

- **LLM**: Llama 3.3 70B via Groq (model-agnostic, swappable via env var)
- **Embeddings**: sentence-transformers all-MiniLM-L6-v2 (local, CPU)
- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, CPU)
- **Vector Store**: LanceDB (hybrid BM25 + vector search, 2,186 chunks)
- **Data Sources**: US HTS tariff schedule (15 chapters) + CBP CROSS rulings (220k+ database, 283 indexed)
- **Frontend**: Streamlit
- **All open source** — no proprietary dependencies

## Data Sources

- **HTS Schedule**: Official Harmonized Tariff Schedule PDFs from [USITC](https://hts.usitc.gov/)
- **CROSS Rulings**: CBP Customs Rulings Online Search System via [CBP API](https://rulings.cbp.gov/)
- **GRI Rules**: General Rules of Interpretation (built into classification prompt)

## License

MIT
