# Agentica — Tariff Classification Agent

AI agent that classifies products into HS/HTS tariff codes using advanced RAG over official US trade data.

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
source venv/bin/activate
python3 -m src.eval.run_benchmark
```

## Architecture

- **LLM**: Gemini 2.5 Flash (model-agnostic, swappable via env var)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, local)
- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-6-v2 (local)
- **Vector Store**: LanceDB (hybrid BM25 + vector search)
- **Data**: US HTS tariff schedule + CBP CROSS rulings (220k+)
