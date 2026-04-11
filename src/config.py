import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
HTS_DIR = DATA_DIR / "hts"
CROSS_DIR = DATA_DIR / "cross"
LANCEDB_DIR = Path(os.getenv("LANCEDB_DIR", DATA_DIR / "indexes"))

# LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

# Local models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Ensure directories exist
for d in [DATA_DIR, HTS_DIR, CROSS_DIR, LANCEDB_DIR]:
    d.mkdir(parents=True, exist_ok=True)
