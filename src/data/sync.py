"""
Auto-update sync pipeline for tariff data.

Checks for new CROSS rulings and HTS revisions, downloads new data,
and incrementally updates the LanceDB index.

Usage:
    from src.data.sync import TariffSync
    sync = TariffSync()
    sync.run()
"""

import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

import httpx
import lancedb
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import HTS_DIR, CROSS_DIR, LANCEDB_DIR, EMBEDDING_MODEL
from src.data.parsers import parse_cross_ruling, Chunk


SYNC_STATE_FILE = Path(__file__).parent.parent.parent / "data" / "sync_state.json"
TABLE_NAME = "tariff_chunks"


class TariffSync:
    """Incremental sync pipeline for tariff data."""

    def __init__(self):
        self._embedder = None
        self._db = None
        self._table = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    @property
    def table(self):
        if self._table is None:
            self._db = lancedb.connect(str(LANCEDB_DIR))
            self._table = self._db.open_table(TABLE_NAME)
        return self._table

    def load_state(self) -> dict:
        """Load sync state from disk."""
        if SYNC_STATE_FILE.exists():
            return json.loads(SYNC_STATE_FILE.read_text())
        return {
            "last_sync": None,
            "cross_last_update": None,
            "cross_ruling_count": 0,
            "hts_revision": None,
            "indexed_rulings": [],
        }

    def save_state(self, state: dict):
        """Save sync state to disk."""
        SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))

    def check_cross_updates(self) -> dict:
        """Check if new CROSS rulings are available."""
        try:
            r = httpx.get("https://rulings.cbp.gov/api/stat/lastupdate", timeout=15)
            r.raise_for_status()
            stats = r.json()
            return {
                "last_update": stats.get("lastUpdateDate", ""),
                "total_rulings": stats.get("totalSearchableRulingsCount", 0),
                "new_rulings": stats.get("numberOfNewRlingsAdded", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def download_new_rulings(self, collection: str = "NY") -> list[Path]:
        """Download latest rulings ZIP and extract new PDFs."""
        url = "https://rulings.cbp.gov/api/downloadLatestDocuments"
        dest_zip = CROSS_DIR / f"cross_{collection.lower()}_latest.zip"
        extracted_dir = CROSS_DIR / collection.lower()

        try:
            with httpx.stream("GET", url, params={"collection": collection},
                            follow_redirects=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest_zip, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=65536):
                        f.write(chunk)

            extracted_dir.mkdir(parents=True, exist_ok=True)
            new_files = []
            with zipfile.ZipFile(dest_zip, "r") as zf:
                for name in zf.namelist():
                    dest_path = extracted_dir / name
                    if not dest_path.exists():
                        zf.extract(name, extracted_dir)
                        new_files.append(dest_path)

            return new_files
        except Exception as e:
            print(f"  Error downloading {collection}: {e}")
            return []

    def ingest_new_rulings(self, pdf_paths: list[Path]) -> int:
        """Parse, embed, and add new rulings to LanceDB."""
        if not pdf_paths:
            return 0

        chunks = []
        for pdf_path in pdf_paths:
            try:
                parsed = parse_cross_ruling(pdf_path)
                chunks.extend(parsed)
            except Exception as e:
                print(f"  Error parsing {pdf_path.name}: {e}")

        if not chunks:
            return 0

        # Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True).tolist()

        # Add to table
        records = []
        for chunk, emb in zip(chunks, embeddings):
            records.append({
                "text": chunk.text,
                "source": chunk.source,
                "chunk_type": chunk.chunk_type,
                "metadata_json": json.dumps(chunk.metadata),
                "vector": emb,
            })

        self.table.add(records)

        # Rebuild FTS index
        self.table.create_fts_index("text", replace=True)

        return len(records)

    def run(self) -> dict:
        """Run the full sync pipeline. Returns a summary dict."""
        state = self.load_state()
        summary = {
            "started_at": datetime.now().isoformat(),
            "cross_status": None,
            "new_rulings_downloaded": 0,
            "new_chunks_indexed": 0,
            "errors": [],
        }

        # 1. Check for CROSS updates
        print("Checking for CROSS updates...")
        cross_status = self.check_cross_updates()
        summary["cross_status"] = cross_status

        if "error" in cross_status:
            summary["errors"].append(f"CROSS check failed: {cross_status['error']}")
            print(f"  Error: {cross_status['error']}")
        else:
            print(f"  CROSS last update: {cross_status['last_update']}")
            print(f"  Total rulings: {cross_status['total_rulings']}")
            print(f"  New rulings: {cross_status['new_rulings']}")

            needs_update = (
                cross_status["last_update"] != state.get("cross_last_update")
                or cross_status["new_rulings"] > 0
            )

            if needs_update:
                # 2. Download new rulings
                print("Downloading new rulings...")
                all_new = []
                for collection in ["NY", "HQ"]:
                    new_files = self.download_new_rulings(collection)
                    all_new.extend(new_files)
                    print(f"  {collection}: {len(new_files)} new files")

                summary["new_rulings_downloaded"] = len(all_new)

                # 3. Ingest new rulings
                if all_new:
                    print(f"Ingesting {len(all_new)} new rulings...")
                    n = self.ingest_new_rulings(all_new)
                    summary["new_chunks_indexed"] = n
                    print(f"  Indexed {n} new chunks")
                else:
                    print("  No new files to ingest")

                state["cross_last_update"] = cross_status["last_update"]
                state["cross_ruling_count"] = cross_status["total_rulings"]
            else:
                print("  No updates needed")

        # Update state
        state["last_sync"] = summary["started_at"]
        self.save_state(state)

        summary["finished_at"] = datetime.now().isoformat()
        print(f"\nSync complete: {summary['new_chunks_indexed']} new chunks indexed")
        return summary


def get_sync_status() -> dict:
    """Get current sync status for the UI."""
    if SYNC_STATE_FILE.exists():
        state = json.loads(SYNC_STATE_FILE.read_text())
        return {
            "last_sync": state.get("last_sync", "Never"),
            "cross_ruling_count": state.get("cross_ruling_count", 0),
            "cross_last_update": state.get("cross_last_update", "Unknown"),
        }
    return {
        "last_sync": "Never",
        "cross_ruling_count": 0,
        "cross_last_update": "Unknown",
    }


if __name__ == "__main__":
    sync = TariffSync()
    result = sync.run()
    print(json.dumps(result, indent=2))
