"""Load versioned fitted coefficients without changing either scoring formula."""

import json
import math
from pathlib import Path

ORIGINAL_COMPOSITION_WEIGHTS = {
    "capability": 0.50, "semantic_density": 0.20, "context_support": 0.20, "limit": 0.10,
}
ORIGINAL_RISK_WEIGHTS = {
    "isolation": 0.15, "compositional": 0.30, "permission": 0.15,
    "external": 0.20, "manipulation": 0.10, "contradiction": 0.10,
}
WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "calibration" / "weights.json"


def validate_weights(values: dict, expected: dict) -> dict:
    if set(values) != set(expected):
        raise ValueError("Calibration coefficient names do not match the formula")
    result = {name: float(values[name]) for name in expected}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in result.values()):
        raise ValueError("Calibration coefficients must be finite and within [0, 1]")
    if abs(sum(result.values()) - 1) > 1e-8:
        raise ValueError("Calibration coefficients must sum to one")
    return result


def load_weights(path: Path = WEIGHTS_PATH) -> dict:
    if not path.exists():
        return {"method": "original_provisional", "composition": ORIGINAL_COMPOSITION_WEIGHTS.copy(),
                "risk": ORIGINAL_RISK_WEIGHTS.copy()}
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["composition"] = validate_weights(artifact["composition"], ORIGINAL_COMPOSITION_WEIGHTS)
    artifact["risk"] = validate_weights(artifact["risk"], ORIGINAL_RISK_WEIGHTS)
    if artifact["method"] in {"mpbench-positive-regularized-v1", "mpbench-positive-pairwise-v1"}:
        if any(value <= 0 for values in (artifact["composition"], artifact["risk"]) for value in values.values()):
            raise ValueError("This model requires every scoring factor to retain a positive contribution")
    return artifact


SCORING_CALIBRATION = load_weights()
COMPOSITION_WEIGHTS = SCORING_CALIBRATION["composition"]
RISK_WEIGHTS = SCORING_CALIBRATION["risk"]
