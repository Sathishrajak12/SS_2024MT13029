"""Accepted memory storage and retrieval across imported dataset memories."""

from __future__ import annotations

import csv
import json
from typing import Any

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATASET_MEMORY_COLLECTIONS,
    EMBEDDING_MODEL,
    RETRIEVAL_SIMILARITY_THRESHOLD,
    SEED_FILE,
    ensure_directories,
)
from src.utils import extract_entities, fingerprint_text, normalize_text, utc_timestamp
from src.chroma_client import create_chroma_client


class MemoryStore:
    def __init__(self) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed in the active environment. "
                "Start the application with run_windows.bat to use the bundled environment."
            ) from exc
        ensure_directories()
        self.client = create_chroma_client(CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Trusted active long-term memory"},
        )

    def count(self) -> int:
        """Count accepted memories only, for admission and seed initialization."""
        return self.collection.count()

    def _retrieval_collections(self, include_datasets: bool = True) -> list[tuple[str, Any]]:
        from chromadb.errors import NotFoundError

        collections = [(COLLECTION_NAME, self.collection)]
        if not include_datasets:
            return collections
        for name in DATASET_MEMORY_COLLECTIONS:
            try:
                collection = self.client.get_collection(name)
            except NotFoundError:
                continue
            model = (collection.metadata or {}).get("embedding_model")
            if model and model != EMBEDDING_MODEL:
                raise ValueError(f"Memory collection {name} uses a different embedding model: {model}")
            collections.append((name, collection))
        return collections

    def collection_counts(self) -> dict[str, int]:
        return {name: collection.count() for name, collection in self._retrieval_collections()}

    def retrieval_count(self, include_datasets: bool = True) -> int:
        return sum(
            collection.count()
            for _, collection in self._retrieval_collections(include_datasets=include_datasets)
        )

    @staticmethod
    def _chroma_metadata(record: dict[str, Any]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key, value in record.items():
            if key in {"text", "embedding"} or value is None:
                continue
            if isinstance(value, (list, dict)):
                metadata[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                metadata[key] = str(value)
        metadata["embedding_model"] = EMBEDDING_MODEL
        return metadata

    @staticmethod
    def _decode_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        result = dict(metadata or {})
        for key in ("entities", "tags", "related_memory_ids"):
            value = result.get(key)
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = []
        return result

    def add_memory(self, record: dict[str, Any], embedding: list[float]) -> None:
        self.collection.add(
            ids=[record["memory_id"]],
            documents=[record["text"]],
            embeddings=[embedding],
            metadatas=[self._chroma_metadata(record)],
        )

    def query(self, embedding: list[float], top_k: int,
              include_datasets: bool = True) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        candidates = []
        for name, collection in self._retrieval_collections(include_datasets=include_datasets):
            count = collection.count()
            if count == 0:
                continue
            result = collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances", "embeddings"],
            )
            for index, memory_id in enumerate(result["ids"][0]):
                distance = float(result["distances"][0][index])
                similarity = max(-1.0, min(1.0, 1.0 - distance))
                if similarity < RETRIEVAL_SIMILARITY_THRESHOLD:
                    continue
                candidates.append((distance, name, memory_id, index, result))

        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        rows = []
        for rank, (distance, name, memory_id, index, result) in enumerate(candidates[:top_k], 1):
            similarity = max(-1.0, min(1.0, 1.0 - distance))
            metadata = self._decode_metadata(result["metadatas"][0][index])
            rows.append({
                "rank": rank,
                "memory_id": memory_id,
                "text": result["documents"][0][index],
                "memory_collection": name,
                "similarity": round(similarity, 4),
                "distance": round(distance, 4),
                "status": metadata.get("status", "accepted"),
                "metadata": metadata,
                "embedding": result["embeddings"][0][index],
                "reason_selected": (
                    f"Ranked #{rank} across all memory sources from {name} "
                    f"with cosine similarity {similarity:.4f}, "
                    f"above the relevance threshold {RETRIEVAL_SIMILARITY_THRESHOLD:.2f}."
                ),
            })
        return rows

    def list_memories(self, limit: int | None = None, collection_name: str | None = None,
                      search: str = "", offset: int = 0) -> list[dict[str, Any]]:
        name = collection_name or COLLECTION_NAME
        collection = self.collection if name == COLLECTION_NAME else dict(self._retrieval_collections())[name]
        if collection.count() == 0:
            return []
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if limit is not None:
            kwargs["limit"] = limit
        if offset:
            kwargs["offset"] = offset
        if search:
            kwargs["where_document"] = {"$contains": search}
        result = collection.get(**kwargs)
        rows = []
        for index, memory_id in enumerate(result["ids"]):
            metadata = self._decode_metadata(result["metadatas"][index])
            rows.append({"memory_id": memory_id, "text": result["documents"][index],
                         **metadata, "memory_collection": name})
        return sorted(rows, key=lambda row: row["memory_id"])

    def get_memory(self, memory_id: str, include_embedding: bool = False) -> dict[str, Any] | None:
        includes = ["documents", "metadatas"]
        for name, collection in self._retrieval_collections():
            result = collection.get(ids=[memory_id], include=includes)
            if not result["ids"]:
                continue
            record = {
                "memory_id": memory_id,
                "text": result["documents"][0],
                **self._decode_metadata(result["metadatas"][0]),
                "memory_collection": name,
            }
            if include_embedding:
                vectors = collection.get(ids=[memory_id], include=["embeddings"])
                record["embedding"] = vectors["embeddings"][0]
            return record
        return None

    def initialize_from_seed(self, embedding_service) -> list[dict[str, Any]]:
        if self.count() > 0:
            return []
        if not SEED_FILE.exists():
            raise FileNotFoundError(f"Seed dataset not found: {SEED_FILE}")
        with SEED_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("The seed dataset is empty.")
        texts = [normalize_text(row["text"]) for row in rows]
        embeddings = embedding_service.embed_many(texts)
        records = []
        for row, text, embedding in zip(rows, texts, embeddings):
            record = {
                "memory_id": row["memory_id"],
                "text": text,
                "timestamp": utc_timestamp(),
                "source": row.get("source", "seed_dataset"),
                "status": "accepted",
                "fingerprint": fingerprint_text(text),
                "risk_score": float(row.get("risk_score") or 0.08),
                "isolation_score": float(row.get("isolation_score") or 0.08),
                "compositional_score": float(row.get("compositional_score") or 0.08),
                "entities": extract_entities(text),
                "tags": [row.get("category", "general")],
                "related_memory_ids": [],
                "created_session": "seed-initialization",
                "category": row.get("category", "general"),
                "scenario_id": row.get("scenario_id", "BENIGN"),
                "sequence_number": int(row.get("sequence_number") or 0),
                "expected_label": row.get("expected_label", "benign"),
            }
            self.add_memory(record, embedding)
            records.append(record)
        return records

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Trusted active long-term memory"},
        )
