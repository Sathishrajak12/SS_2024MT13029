"""Persistent NetworkX knowledge graph and small candidate subgraphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph

from config import GRAPH_FILE, MAX_GRAPH_NEIGHBORS, SIMILARITY_EDGE_THRESHOLD
from src.utils import important_terms


class GraphManager:
    def __init__(self, path: Path = GRAPH_FILE) -> None:
        self.path = path
        self.graph = self._load()
        if self._remove_content_from_nodes():
            self.save()

    @staticmethod
    def _node_metadata(record: dict[str, Any]) -> dict[str, Any]:
        """Return JSON-safe, non-content metadata for a graph node."""
        keys = (
            "status", "timestamp", "source", "fingerprint", "risk_score",
            "isolation_score", "compositional_score", "category", "scenario_id",
            "sequence_number", "expected_label", "entities", "tags",
            "created_session", "decision", "new_capabilities",
            "memory_collection", "dataset", "source_file",
        )
        metadata = {key: record[key] for key in keys if key in record}
        metadata.setdefault("status", "accepted")
        metadata.setdefault("risk_score", 0.08)
        return metadata

    def _remove_content_from_nodes(self) -> bool:
        changed = False
        for _, attributes in self.graph.nodes(data=True):
            for forbidden_key in ("text", "embedding", "document"):
                if forbidden_key in attributes:
                    del attributes[forbidden_key]
                    changed = True
        return changed

    def _load(self) -> nx.Graph:
        if not self.path.exists():
            return nx.Graph()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return json_graph.node_link_graph(data, edges="links")
        except (json.JSONDecodeError, OSError, nx.NetworkXError, KeyError):
            return nx.Graph()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(self.graph, edges="links")
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)

    def rebuild_from_records(self, records: list[dict[str, Any]]) -> None:
        graph = nx.Graph()
        for record in records:
            graph.add_node(record["memory_id"], **self._node_metadata(record))
        by_scenario: dict[str, list[dict]] = {}
        for record in records:
            scenario = record.get("scenario_id", "")
            if scenario and scenario not in {"BENIGN", "NONE"}:
                by_scenario.setdefault(scenario, []).append(record)
        for items in by_scenario.values():
            ordered = sorted(items, key=lambda row: int(row.get("sequence_number", 0)))
            for left, right in zip(ordered, ordered[1:]):
                graph.add_edge(
                    left["memory_id"], right["memory_id"], relation_type="SAME_SCENARIO",
                    weight=0.90, explanation="Seed memories belong to adjacent steps of the same labelled scenario.",
                )
        by_category: dict[str, list[dict]] = {}
        for record in records:
            by_category.setdefault(record.get("category", "general"), []).append(record)
        for items in by_category.values():
            for left, right in zip(items, items[1:]):
                shared = important_terms(left["text"]) & important_terms(right["text"])
                if shared:
                    graph.add_edge(
                        left["memory_id"], right["memory_id"], relation_type="SHARED_ENTITY",
                        weight=round(min(0.8, 0.3 + 0.1 * len(shared)), 2),
                        explanation="Shared important term(s): " + ", ".join(sorted(shared)[:4]),
                    )
        self.graph = graph
        self.save()

    def sync_metadata_from_records(self, records: list[dict[str, Any]]) -> None:
        """Refresh node metadata from Chroma without changing graph edges."""
        changed = False
        for record in records:
            memory_id = record["memory_id"]
            metadata = self._node_metadata(record)
            if memory_id not in self.graph:
                self.graph.add_node(memory_id, **metadata)
                changed = True
                continue
            current = dict(self.graph.nodes[memory_id])
            if current != metadata:
                self.graph.nodes[memory_id].clear()
                self.graph.nodes[memory_id].update(metadata)
                changed = True
        if self._remove_content_from_nodes():
            changed = True
        if changed:
            self.save()

    @staticmethod
    def _candidate_relation(candidate: str, memory: dict) -> tuple[str, str]:
        combined = (candidate + " " + memory["text"]).lower()
        if any(word in candidate.lower() for word in ("send", "forward", "grant", "bypass", "automatic")):
            if any(word in memory["text"].lower() for word in ("destination", "support", "report", "approval", "administrator")):
                return "ENABLES", "The candidate action combines with a resource, destination, or policy in this memory."
        shared = important_terms(candidate) & important_terms(memory["text"])
        if shared:
            return "SHARED_ENTITY", "Shared important term(s): " + ", ".join(sorted(shared)[:5])
        return "SEMANTIC_SIMILARITY", "Selected as a nearest neighbour in embedding space."

    def build_candidate_context(self, candidate: str, retrieved: list[dict],
                                include_datasets: bool = True) -> tuple[nx.Graph, list[dict]]:
        local = nx.Graph()
        local.add_node("CANDIDATE", status="candidate", risk_score=0.0)
        edge_rows: list[dict[str, Any]] = []
        retrieved_ids = {row["memory_id"] for row in retrieved}
        for row in retrieved:
            node_id = row["memory_id"]
            attrs = dict(self.graph.nodes.get(node_id, {}))
            attrs.update(status=row.get("status", "accepted"))
            attrs["memory_collection"] = row.get("memory_collection", "secure_memories")
            attrs.pop("text", None)
            attrs.pop("embedding", None)
            local.add_node(node_id, **attrs)
            relation, explanation = self._candidate_relation(candidate, row)
            weight = float(row["similarity"])
            local.add_edge("CANDIDATE", node_id, relation_type=relation, weight=weight, explanation=explanation)
            edge_rows.append({"from": "CANDIDATE", "relation": relation, "to": node_id, "weight": round(weight, 3), "explanation": explanation, "proposed": True})

        added = 0
        for node_id in list(retrieved_ids):
            if node_id not in self.graph:
                continue
            for neighbour in self.graph.neighbors(node_id):
                if added >= MAX_GRAPH_NEIGHBORS:
                    break
                attrs = dict(self.graph.nodes[neighbour])
                if not include_datasets and attrs.get("memory_collection", "secure_memories") != "secure_memories":
                    continue
                attrs.pop("text", None)
                attrs.pop("embedding", None)
                local.add_node(neighbour, **attrs)
                edge = dict(self.graph.edges[node_id, neighbour])
                local.add_edge(node_id, neighbour, **edge)
                edge_rows.append({
                    "from": node_id, "relation": edge.get("relation_type", "RELATED_TO"),
                    "to": neighbour, "weight": float(edge.get("weight", 0.5)),
                    "explanation": edge.get("explanation", "Existing trusted graph relationship."),
                    "proposed": False,
                })
                added += 1
        return local, edge_rows

    def commit_candidate(self, record: dict, retrieved: list[dict]) -> None:
        memory_id = record["memory_id"]
        self.graph.add_node(memory_id, **self._node_metadata(record))
        for row in retrieved:
            if row["similarity"] < SIMILARITY_EDGE_THRESHOLD:
                continue
            if row["memory_id"] not in self.graph:
                self.graph.add_node(row["memory_id"], **self._node_metadata({
                    **row.get("metadata", {}), "status": row.get("status", "accepted"),
                    "memory_collection": row.get("memory_collection", "secure_memories"),
                }))
            relation, explanation = self._candidate_relation(record["text"], row)
            self.graph.add_edge(
                memory_id, row["memory_id"], relation_type=relation,
                weight=float(row["similarity"]), explanation=explanation,
            )
        self.save()
