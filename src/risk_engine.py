"""Transparent weighted scoring and threshold decisions."""

from __future__ import annotations

from config import ACCEPT_THRESHOLD, MAX_GRAPH_BOOST, QUARANTINE_THRESHOLD, RISK_WEIGHTS
from src.utils import clamp
from src.scoring_weights import SCORING_CALIBRATION


FEATURE_LABELS = {
    "isolation": "Isolation control flag (0/1; project policy)",
    "compositional": "Compositional",
    "permission": "Permission escalation",
    "external": "External action",
    "manipulation": "Manipulation / unsafe automation",
    "contradiction": "Contradiction / suspicious context",
}


def calculate_risk(features: dict[str, float], graph_evidence: dict,
                   risk_weights: dict[str, float] | None = None,
                   acceptance_threshold: float | None = None,
                   quarantine_threshold: float | None = None,
                   max_graph_boost: float | None = None) -> dict:
    """Calculate a reproducible score from visible components."""
    weights = (risk_weights or RISK_WEIGHTS).copy()
    accept_at = ACCEPT_THRESHOLD if acceptance_threshold is None else acceptance_threshold
    quarantine_at = QUARANTINE_THRESHOLD if quarantine_threshold is None else quarantine_threshold
    graph_boost_limit = MAX_GRAPH_BOOST if max_graph_boost is None else max_graph_boost
    normalized = {name: clamp(features.get(name, 0.0)) for name in weights}
    contributions = {
        name: round(normalized[name] * weight, 4)
        for name, weight in weights.items()
    }
    raw_score = round(sum(contributions.values()), 4)

    density = clamp(graph_evidence.get("semantic_density", 0.0))
    supporting = min(int(graph_evidence.get("supporting_memories", 0)), 4) / 4
    new_capability = 1.0 if graph_evidence.get("new_capability_detected") else 0.0
    dangerous_relation = 1.0 if graph_evidence.get("dangerous_relationship_detected") else 0.0
    graph_strength = 0.20 * density + 0.20 * supporting + 0.35 * new_capability + 0.25 * dangerous_relation
    graph_boost = round(graph_boost_limit * graph_strength, 4)
    final_score = clamp(raw_score + graph_boost)
    effective_score = clamp(max(final_score, normalized["isolation"]))
    weighted_decision = decide(final_score, accept_at, quarantine_at)
    isolation_decision = decide(normalized["isolation"], accept_at, quarantine_at)
    severity = {"ACCEPT": 0, "REVIEW": 1, "QUARANTINE": 2}
    decision = max((weighted_decision, isolation_decision), key=severity.get)
    isolation_override = severity[isolation_decision] > severity[weighted_decision]
    decision_basis = (
        f"Isolation guardrail raised the decision from {weighted_decision} to {decision}."
        if isolation_override
        else "Weighted Final Risk Score produced the same or stricter decision."
    )
    return {
        "features": normalized,
        "weights": weights,
        "calibration_method": SCORING_CALIBRATION["method"],
        "calibration_status": SCORING_CALIBRATION.get("status", "provisional"),
        "calibration_evaluation": SCORING_CALIBRATION.get("heldout_evaluation", {}),
        "calibration_protocol_sha256": SCORING_CALIBRATION.get("protocol_sha256"),
        "contributions": contributions,
        "raw_score": raw_score,
        "graph_boost": graph_boost,
        "graph_strength": round(graph_strength, 4),
        "weighted_score": final_score,
        "final_score": final_score,
        "effective_score": effective_score,
        "decision": decision,
        "weighted_decision": weighted_decision,
        "isolation_decision": isolation_decision,
        "isolation_override": isolation_override,
        "decision_basis": decision_basis,
        "acceptance_threshold": accept_at,
        "quarantine_threshold": quarantine_at,
        "max_graph_boost": graph_boost_limit,
    }


def decide(score: float, acceptance_threshold: float = ACCEPT_THRESHOLD,
           quarantine_threshold: float = QUARANTINE_THRESHOLD) -> str:
    if score < acceptance_threshold:
        return "ACCEPT"
    if score < quarantine_threshold:
        return "REVIEW"
    return "QUARANTINE"


def score_rows(result: dict) -> list[dict]:
    return [
        {
            "Feature": FEATURE_LABELS[name],
            "Value": result["features"][name],
            "Weight": result["weights"][name],
            "Contribution": result["contributions"][name],
        }
        for name in result["weights"]
    ]
