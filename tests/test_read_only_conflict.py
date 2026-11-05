from src.capability_analyzer import analyze_composition
from src.risk_engine import calculate_risk


def test_read_only_context_conflicts_with_record_modification():
    context = [{
        "memory_id": "MEM-CONTEXT",
        "text": "Support staff may only view customer records using read-only access.",
        "similarity": 0.90,
    }]

    composition = analyze_composition(
        "Allow support staff to modify customer records.", context, []
    )

    assert composition["contradiction_risk"] == 0.95
    assert composition["permission_risk"] >= 0.90
    assert "DATA_MODIFICATION" in composition["new_capabilities"]
    assert composition["chain_evidence"][0]["rule"] == "READ_ONLY_MODIFICATION_CONFLICT"
    assert composition["chain_evidence"][0]["context_evidence"] == [{
        "memory_id": "MEM-CONTEXT",
        "text": "Support staff may only view customer records using read-only access.",
    }]

    scoring = calculate_risk({
        "isolation": 0.0,
        "compositional": composition["compositional_risk"],
        "permission": composition["permission_risk"],
        "external": composition["external_risk"],
        "manipulation": composition["manipulation_risk"],
        "contradiction": composition["contradiction_risk"],
    }, composition["graph_evidence"])
    assert scoring["decision"] != "ACCEPT"

