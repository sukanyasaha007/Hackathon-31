"""
LanceDB ingestion pipeline.

Takes parsed Chunk objects, generates embeddings, and stores them in LanceDB
with full-text search (BM25) + vector search support.

Usage: python3 -m src.data.ingest
"""

import sys
import time
from pathlib import Path

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import HTS_DIR, CROSS_DIR, LANCEDB_DIR, EMBEDDING_MODEL
from src.data.parsers import parse_all_hts, parse_all_cross, Chunk


TABLE_NAME = "tariff_chunks"


def build_embeddings(chunks: list[Chunk], model: SentenceTransformer, batch_size: int = 64) -> list[list[float]]:
    """Generate embeddings for all chunks."""
    texts = [c.text for c in chunks]
    print(f"  Generating embeddings for {len(texts)} chunks (batch_size={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} chunks/sec)")
    return embeddings.tolist()


def create_table(db: lancedb.DBConnection, chunks: list[Chunk], embeddings: list[list[float]]):
    """Create or overwrite a LanceDB table with chunk data + embeddings."""
    import json

    records = []
    for chunk, emb in zip(chunks, embeddings):
        records.append({
            "text": chunk.text,
            "source": chunk.source,
            "chunk_type": chunk.chunk_type,
            "metadata_json": json.dumps(chunk.metadata),
            "vector": emb,
        })

    # Drop existing table if any
    try:
        db.drop_table(TABLE_NAME)
    except Exception:
        pass

    print(f"  Creating LanceDB table '{TABLE_NAME}' with {len(records)} records...")
    table = db.create_table(TABLE_NAME, data=records)

    # Create full-text search index for BM25 hybrid search
    print("  Building FTS index on 'text' column...")
    table.create_fts_index("text", replace=True)

    print(f"  Table created: {table.count_rows()} rows")
    return table


def ingest():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("TARIFF DATA INGESTION")
    print("=" * 60)

    # 1. Parse all data
    print("\n[1/3] Parsing data...")
    t0 = time.time()

    hts_chunks = parse_all_hts(HTS_DIR)
    print(f"  HTS total: {len(hts_chunks)} chunks")

    cross_chunks = parse_all_cross(CROSS_DIR)
    print(f"  CROSS total: {len(cross_chunks)} chunks")

    all_chunks = hts_chunks + cross_chunks
    print(f"  Combined: {len(all_chunks)} chunks ({time.time()-t0:.1f}s)")

    if not all_chunks:
        print("ERROR: No chunks parsed. Check data directory.")
        return

    # 2. Generate embeddings
    print(f"\n[2/3] Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = build_embeddings(all_chunks, model)

    # 3. Store in LanceDB
    print(f"\n[3/3] Ingesting into LanceDB at '{LANCEDB_DIR}'...")
    db = lancedb.connect(str(LANCEDB_DIR))
    table = create_table(db, all_chunks, embeddings)

    # Stats
    print("\n--- Ingestion Summary ---")
    print(f"Total chunks: {table.count_rows()}")
    print(f"HTS chunks: {len(hts_chunks)}")
    print(f"CROSS chunks: {len(cross_chunks)}")
    print(f"Embedding dim: {len(embeddings[0])}")
    print(f"LanceDB path: {LANCEDB_DIR}")

    # Quick sanity check: search for "electric motor"
    print("\n--- Sanity Check ---")
    q_emb = model.encode("electric motor tariff classification", normalize_embeddings=True).tolist()
    results = table.search(q_emb).limit(3).to_list()
    for i, r in enumerate(results):
        print(f"  [{i+1}] (dist={r['_distance']:.4f}) {r['source']}: {r['text'][:120]}...")

    print("\nDone.")


if __name__ == "__main__":
    ingest()
