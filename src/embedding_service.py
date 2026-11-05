"""Local SentenceTransformers embedding service."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from src.scoring_weights import SCORING_CALIBRATION


class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed in the active Python environment. "
                "Start the application with run_windows.bat to use the bundled environment."
            ) from exc
        self.model_name = model_name
        try:
            revision = (
                SCORING_CALIBRATION.get("embedding_revision")
                if model_name == SCORING_CALIBRATION.get("embedding_model") else None
            )
            self.model = SentenceTransformer(model_name, revision=revision)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load embedding model '{model_name}'. Check your internet connection "
                "on first run, then try again. Original error: {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        result = np.asarray(vector, dtype=float).tolist()
        if len(result) != EMBEDDING_DIMENSION:
            raise ValueError(f"Expected {EMBEDDING_DIMENSION} dimensions, received {len(result)}")
        return result

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=len(texts) > 20
        )
        result = np.asarray(vectors, dtype=float).tolist()
        if result and len(result[0]) != EMBEDDING_DIMENSION:
            raise ValueError(f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings")
        return result
