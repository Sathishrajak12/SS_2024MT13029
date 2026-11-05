from src.risk_engine import calculate_risk


FEATURES = {
    "isolation": 0.0,
    "compositional": 0.5,
    "permission": 0.0,
    "external": 0.0,
    "manipulation": 0.0,
    "contradiction": 0.5,
}
GRAPH = {
    "semantic_density": 0.0,
    "supporting_memories": 0,
    "new_capability_detected": False,
    "dangerous_relationship_detected": False,
}


def test_custom_weights_and_thresholds_change_scoring():
    custom_weights = {
        "isolation": 0.0,
        "compositional": 0.0,
        "permission": 0.0,
        "external": 0.0,
        "manipulation": 0.0,
        "contradiction": 1.0,
    }
    result = calculate_risk(
        FEATURES,
        GRAPH,
        risk_weights=custom_weights,
        acceptance_threshold=0.60,
        quarantine_threshold=0.80,
        max_graph_boost=0.0,
    )

    assert result["weighted_score"] == 0.5
    assert result["decision"] == "ACCEPT"
    assert result["weights"] == custom_weights
    assert result["acceptance_threshold"] == 0.60
    assert result["quarantine_threshold"] == 0.80
    assert result["max_graph_boost"] == 0.0


def test_system_defaults_are_restored_after_custom_calculation():
    baseline = calculate_risk(FEATURES, GRAPH)
    calculate_risk(
        FEATURES,
        GRAPH,
        risk_weights={
            "isolation": 0.0, "compositional": 0.0, "permission": 0.0,
            "external": 0.0, "manipulation": 0.0, "contradiction": 1.0,
        },
        acceptance_threshold=0.60,
        quarantine_threshold=0.80,
        max_graph_boost=0.0,
    )
    restored = calculate_risk(FEATURES, GRAPH)

    assert restored == baseline

