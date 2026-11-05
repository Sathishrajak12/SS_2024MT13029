"""End-to-end candidate-memory processing pipeline."""

from __future__ import annotations

import time
import uuid
from typing import Any

from config import PENDING_REVIEW_FILE, QUARANTINE_FILE, TOP_K
from src.capability_analyzer import analyze_composition
from src.graph_manager import GraphManager
from src.isolation_analyzer import analyze_isolation
from src.llm_service import LocalLLMService
from src.quarantine_store import AuditStore
from src.risk_engine import calculate_risk
from src.utils import extract_entities, fingerprint_text, make_memory_id, normalize_text, utc_timestamp


class MemoryValidator:
    def __init__(self, memory_store, embedding_service, graph_manager: GraphManager | None = None) -> None:
        self.memory_store = memory_store
        self.embedding_service = embedding_service
        self.graph_manager = graph_manager or GraphManager()
        self.quarantine_store = AuditStore(QUARANTINE_FILE)
        self.pending_store = AuditStore(PENDING_REVIEW_FILE)
        self.local_llm = LocalLLMService()

    def analyze(self, candidate_text: str, session_id: str, use_llm: bool = True,
                include_datasets: bool = True,
                scoring_config: dict[str, Any] | None = None) -> dict[str, Any]:
        timings: dict[str, float] = {}
        pipeline: list[str] = []
        scoring_config = scoring_config or {}

        def mark(name: str, started: float) -> None:
            timings[name] = round((time.perf_counter() - started) * 1000, 1)

        started = time.perf_counter()
        candidate = normalize_text(candidate_text)
        if not candidate:
            raise ValueError("Please enter a candidate memory before selecting Analyze.")
        if len(candidate) > 4000:
            raise ValueError("Candidate memory is too long. Please keep it below 4,000 characters.")
        pipeline.append("Candidate received and normalized")
        mark("normalization", started)

        started = time.perf_counter()
        candidate_embedding = self.embedding_service.embed(candidate)
        pipeline.append(f"Embedding generated ({len(candidate_embedding)} dimensions)")
        mark("embedding", started)

        started = time.perf_counter()
        trusted_before = self.memory_store.count()
        retrieval_pool_size = self.memory_store.retrieval_count(include_datasets=include_datasets)
        retrieved = self.memory_store.query(
            candidate_embedding, TOP_K, include_datasets=include_datasets
        )
        retrieval_scope = "trusted and dataset" if include_datasets else "trusted only"
        pipeline.append(
            f"Top-{len(retrieved)} memories retrieved from {retrieval_pool_size} "
            f"available memories ({retrieval_scope})"
        )
        mark("retrieval", started)

        started = time.perf_counter()
        local_graph, edge_rows = self.graph_manager.build_candidate_context(
            candidate, retrieved, include_datasets=include_datasets
        )
        retrieved_ids = {row["memory_id"] for row in retrieved}
        graph_context = []
        graph_node_details = {"CANDIDATE": {"text": candidate, "status": "candidate"}}
        for row in retrieved:
            graph_node_details[row["memory_id"]] = {
                "text": row["text"], "status": row.get("status", "accepted")
            }
        for node_id in local_graph.nodes:
            if node_id == "CANDIDATE" or node_id in retrieved_ids:
                continue
            memory = self.memory_store.get_memory(node_id)
            if memory:
                graph_context.append({
                    "memory_id": node_id,
                    "text": memory["text"],
                    "similarity": 0.0,
                })
                graph_node_details[node_id] = {
                    "text": memory["text"], "status": memory.get("status", "accepted")
                }
        pipeline.append(f"Graph context generated ({len(edge_rows)} relationships considered)")
        mark("graph_context", started)

        started = time.perf_counter()
        isolation = analyze_isolation(candidate)
        pipeline.append("Isolation baseline complete")
        mark("isolation", started)

        started = time.perf_counter()
        composition = analyze_composition(
            candidate, retrieved, edge_rows, graph_context,
            composition_weights=scoring_config.get("composition_weights"),
        )
        pipeline.append("Capability Difference and compositional analysis complete")
        mark("composition", started)

        features = {
            "isolation": isolation["score"],
            "compositional": composition["compositional_risk"],
            "permission": composition["permission_risk"],
            "external": composition["external_risk"],
            "manipulation": composition["manipulation_risk"],
            "contradiction": composition["contradiction_risk"],
        }

        started = time.perf_counter()
        scoring = calculate_risk(
            features,
            composition["graph_evidence"],
            risk_weights=scoring_config.get("risk_weights"),
            acceptance_threshold=scoring_config.get("acceptance_threshold"),
            quarantine_threshold=scoring_config.get("quarantine_threshold"),
            max_graph_boost=scoring_config.get("max_graph_boost"),
        )
        scoring["configuration_source"] = "CUSTOM" if scoring_config else "SYSTEM DEFAULTS"
        decision = scoring["decision"]
        pipeline.append(
            f"Risk score calculated with {scoring['configuration_source']}; candidate marked {decision}"
        )
        mark("scoring", started)

        llm_result = None
        llm_isolation = None
        llm_status = self.local_llm.health()
        if use_llm and llm_status["connected"] and llm_status["model_available"]:
            started = time.perf_counter()
            llm_isolation = self.local_llm.analyze_candidate_isolation(candidate, isolation)
            pipeline.append(
                "Local LLM candidate-only advisory complete"
                if llm_isolation
                else "Local LLM candidate-only response invalid; deterministic fallback retained"
            )
            mark("llm_isolation", started)
            started = time.perf_counter()
            llm_result = self.local_llm.explain(candidate, retrieved, composition)
            mark("llm_explanation", started)

        event_id = f"EVT-{uuid.uuid4().hex[:10].upper()}"
        related_ids = [row["memory_id"] for row in retrieved]
        base_record = {
            "candidate_id": "MEM-CANDIDATE",
            "event_id": event_id,
            "text": candidate,
            "timestamp": utc_timestamp(),
            "source": "user_input",
            "fingerprint": fingerprint_text(candidate),
            "risk_score": scoring["effective_score"],
            "weighted_risk_score": scoring["weighted_score"],
            "isolation_score": isolation["score"],
            "compositional_score": composition["compositional_risk"],
            "composition_method": composition["method_version"],
            "composition_evidence": composition["chain_evidence"],
            "entities": extract_entities(candidate),
            "tags": ["user_added"],
            "related_memory_ids": related_ids,
            "retrieved_memory_sources": {
                row["memory_id"]: row.get("memory_collection", "secure_memories") for row in retrieved
            },
            "created_session": session_id,
            "datasets_included": include_datasets,
            "decision": decision,
            "decision_basis": scoring["decision_basis"],
            "llm_isolation_advisory": llm_isolation or {},
            "risk_factors": features,
            "risk_weights": scoring["weights"],
            "composition_weights": composition["composition_weights"],
            "scoring_configuration": scoring["configuration_source"],
            "calibration_method": scoring["calibration_method"],
            "calibration_protocol_sha256": scoring["calibration_protocol_sha256"],
            "new_capabilities": composition["new_capabilities"],
            "reason": composition["explanation"],
        }

        stored_in_trusted = False
        if decision == "ACCEPT":
            permanent_id = make_memory_id(trusted_before)
            memory_record = {
                **base_record,
                "memory_id": permanent_id,
                "status": "accepted",
                "category": "user_added",
                "scenario_id": "USER",
                "sequence_number": 0,
                "expected_label": "unknown",
            }
            memory_record.pop("candidate_id", None)
            self.memory_store.add_memory(memory_record, candidate_embedding)
            self.graph_manager.commit_candidate(memory_record, retrieved)
            stored_in_trusted = True
            final_record = memory_record
        elif decision == "REVIEW":
            final_record = {**base_record, "status": "pending_review", "memory_id": event_id}
            self.pending_store.add(final_record)
        else:
            final_record = {**base_record, "status": "quarantined", "memory_id": event_id}
            self.quarantine_store.add(final_record)

        trusted_after = self.memory_store.count()
        llm_evidence = (
            f"Local LLM candidate-only advisory was {llm_isolation['risk_label']} "
            f"at {llm_isolation['risk_score']:.2f}: {llm_isolation['explanation']}"
            if llm_isolation
            else "Local LLM candidate-only advisory was unavailable; deterministic isolation remained active."
        )
        evidence_steps = [
            f"Candidate-only control check: {isolation['label']}; indicator={isolation['score']:.0f} (not a probability).",
            llm_evidence,
            "Top-K Retrieval selected: " + (", ".join(related_ids) if related_ids else "no existing memories"),
            composition["explanation"],
            "New capability: " + (", ".join(composition["new_capabilities"]) if composition["new_capabilities"] else "NONE"),
            f"Compositional risk was {composition['compositional_risk']:.2f}; weighted raw score was {scoring['raw_score']:.2f}.",
            f"Graph/capability boost was {scoring['graph_boost']:.2f}; weighted risk was {scoring['weighted_score']:.2f}.",
            f"Effective Final Risk after the isolation guardrail was {scoring['effective_score']:.2f}.",
            f"Weighted-score decision was {scoring['weighted_decision']}; Isolation guardrail decision was {scoring['isolation_decision']}.",
            f"Final policy decision is {decision}. {scoring['decision_basis']}",
            "Candidate was committed to Trusted Long-Term Memory." if stored_in_trusted else "Candidate was NOT committed to Trusted Long-Term Memory.",
        ]
        trace = {
            "candidate": candidate,
            "retrieved_memory_ids": related_ids,
            "retrieved_memory_sources": base_record["retrieved_memory_sources"],
            "graph_context_memory_ids": [item["memory_id"] for item in graph_context],
            "similarities": [row["similarity"] for row in retrieved],
            "graph_edges_considered": edge_rows,
            "isolation_score": isolation["score"],
            "llm_isolation_advisory": llm_isolation,
            "compositional_score": composition["compositional_risk"],
            "composition_method": composition["method_version"],
            "composition_evidence": composition["chain_evidence"],
            "new_capabilities": composition["new_capabilities"],
            "risk_components": scoring["features"],
            "risk_weights": scoring["weights"],
            "composition_weights": composition["composition_weights"],
            "calibration_method": scoring["calibration_method"],
            "calibration_protocol_sha256": scoring["calibration_protocol_sha256"],
            "weighted_raw_score": scoring["raw_score"],
            "graph_boost": scoring["graph_boost"],
            "weighted_score": scoring["weighted_score"],
            "final_score": scoring["effective_score"],
            "threshold": scoring["quarantine_threshold"],
            "decision": decision,
            "weighted_decision": scoring["weighted_decision"],
            "isolation_guardrail_decision": scoring["isolation_decision"],
            "decision_basis": scoring["decision_basis"],
            "explanation": composition["explanation"],
            "stored_in_trusted_memory": stored_in_trusted,
        }
        return {
            "candidate": candidate,
            "candidate_embedding": candidate_embedding,
            "retrieved": retrieved,
            "local_graph": local_graph,
            "graph_node_details": graph_node_details,
            "graph_edges": edge_rows,
            "isolation": isolation,
            "composition": composition,
            "scoring": scoring,
            "decision": decision,
            "stored_in_trusted": stored_in_trusted,
            "record": final_record,
            "trusted_before": trusted_before,
            "retrieval_pool_size": retrieval_pool_size,
            "trusted_after": trusted_after,
            "pipeline": pipeline,
            "timings_ms": timings,
            "evidence_steps": evidence_steps,
            "trace": trace,
            "llm_status": llm_status,
            "llm_isolation": llm_isolation,
            "llm_result": llm_result,
        }
