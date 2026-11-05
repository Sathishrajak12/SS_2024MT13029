

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from src.scoring_weights import RISK_WEIGHTS

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
GRAPH_FILE = STORAGE_DIR / "memory_graph.json"
QUARANTINE_FILE = STORAGE_DIR / "quarantine.json"
PENDING_REVIEW_FILE = STORAGE_DIR / "pending_review.json"
SEED_FILE = DATA_DIR / "seed_memories.csv"

COLLECTION_NAME = "secure_memories"
# Imported datasets participate in memory retrieval alongside accepted memories.
DATASET_MEMORY_COLLECTIONS = (
    "benchmark_locomo", "benchmark_harmbench", "benchmark_powergridqa",
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:0.5b")

TOP_K = int(os.getenv("TOP_K", "5"))
RETRIEVAL_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.30"))
MAX_GRAPH_NEIGHBORS = int(os.getenv("MAX_GRAPH_NEIGHBORS", "10"))
SIMILARITY_EDGE_THRESHOLD = float(os.getenv("SIMILARITY_EDGE_THRESHOLD", "0.42"))
ACCEPT_THRESHOLD = float(os.getenv("ACCEPT_THRESHOLD", "0.40"))
QUARANTINE_THRESHOLD = float(os.getenv("QUARANTINE_THRESHOLD", "0.70"))
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() in {"1", "true", "yes"}
DEBUG = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}

# The LLM never sets coefficients. Versioned coefficients load from calibration/weights.json.

# A small, separately displayed boost represents evidence created by graph composition.
MAX_GRAPH_BOOST = 0.15

DECISION_COLORS = {
    "ACCEPT": "#2e7d32",
    "REVIEW": "#ed6c02",
    "QUARANTINE": "#c62828",
}


def ensure_directories() -> None:
    """Create local folders needed by the application."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def validate_config() -> None:
    total = sum(RISK_WEIGHTS.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"RISK_WEIGHTS must sum to 1.0; current total is {total:.4f}")
    if not 0 <= ACCEPT_THRESHOLD < QUARANTINE_THRESHOLD <= 1:
        raise ValueError("Thresholds must satisfy 0 <= ACCEPT < QUARANTINE <= 1")
    if not -1 <= RETRIEVAL_SIMILARITY_THRESHOLD <= 1:
        raise ValueError("RETRIEVAL_SIMILARITY_THRESHOLD must be between -1 and 1")
