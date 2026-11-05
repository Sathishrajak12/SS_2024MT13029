"""Explainable capability-difference and compositional analysis."""

from __future__ import annotations

import re

from src.isolation_analyzer import (
    approval_bypass_present,
    security_control_bypass_present,
    standalone_exfiltration_present,
)
from src.utils import clamp
from src.isolation_evidence import PROFILES
from src.scoring_weights import COMPOSITION_WEIGHTS, SCORING_CALIBRATION


CAPABILITY_RULES = {
    "READ_DIAGNOSTIC_INFORMATION": ("diagnostic", "logs", "telemetry"),
    "GENERATE_REPORTS": ("generate report", "diagnostic report", "report is created"),
    "EXTERNAL_COMMUNICATION": ("external", "@", "vendor", "support contact", "forward", "send", "share"),
    "AUTOMATED_ACTION": ("automatic", "automatically", "without manual", "scheduled action"),
    "ACCESS_EXPANSION": ("administrator", "privilege", "elevated access", "grant access"),
    "CONFIGURATION_CHANGE": ("configuration", "setting", "change control"),
    "DATA_MODIFICATION": ("modify", "edit", "write", "delete", "overwrite", "update record", "update records"),
}


def _capabilities(text: str) -> list[str]:
    lower = text.lower()
    capabilities = [name for name, terms in CAPABILITY_RULES.items() if any(term in lower for term in terms)]
    if approval_bypass_present(lower) and "APPROVAL_BYPASS" not in capabilities:
        capabilities.append("APPROVAL_BYPASS")
    if security_control_bypass_present(lower) and "SECURITY_CONTROL_BYPASS" not in capabilities:
        capabilities.append("SECURITY_CONTROL_BYPASS")
    return capabilities


def _has(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


COMPOSITION_METHOD = "weighted-composition-v2"
OWASP_REFERENCE = "https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html"


def analyze_composition(candidate: str, retrieved: list[dict], graph_edges: list[dict],
                        graph_context: list[dict] | None = None,
                        composition_weights: dict[str, float] | None = None) -> dict:
    """Record candidate/context premises for configured control-concern chains."""
    all_context = []
    seen = set()
    for item in [*retrieved, *(graph_context or [])]:
        if item["memory_id"] not in seen:
            all_context.append(item)
            seen.add(item["memory_id"])
    lower = candidate.lower()
    context = " ".join(item["text"] for item in all_context)
    before = _capabilities(context)
    after = _capabilities(context + " " + candidate)
    new_capabilities = sorted(set(after) - set(before))
    findings = []

    def rows(terms):
        return [item for item in all_context if _has(item["text"].lower(), terms)]

    def add(rule, profile, premises):
        unique = {item["memory_id"]: item for item in premises}
        if not unique:
            return
        source = PROFILES[profile]
        findings.append({
            "rule": rule, "candidate_text": candidate,
            "context_evidence": [{"memory_id": item["memory_id"], "text": item["text"]}
                                 for item in unique.values()],
            "controls": source["controls"],
            "references": [*source["references"], OWASP_REFERENCE],
        })

    data_terms = ("diagnostic", "report", "logs", "telemetry", "customer record", "sensitive", "confidential", "credentials", "password")
    destination_terms = ("external", "@", "vendor", "support contact", "third-party")
    data_rows = rows(data_terms)
    destination_rows = rows(destination_terms)
    transfer = bool(re.search(r"\b(?:send|share|forward|transmit|deliver|upload)(?:s|ed|ing)?\b", lower))
    negated_transfer = bool(re.search(
        r"\b(?:do not|must not|never|cannot|can't|should not)(?:\s+\w+){0,4}\s+(?:send|share|forward|transmit|deliver|upload)\b", lower))
    protected_transfer = _has(lower, ("redacted", "anonymized", "approved portal", "approved destination", "after approval", "approval required", "public report"))
    has_data = _has(lower, data_terms) or bool(data_rows)
    has_destination = _has(lower, destination_terms) or bool(destination_rows)
    missing_premises = ([] if _has(lower, data_terms) else data_rows) + ([] if _has(lower, destination_terms) else destination_rows)
    sensitive_terms = ("sensitive", "confidential", "credentials", "password", "customer record")
    if not _has(lower, sensitive_terms):
        missing_premises += rows(sensitive_terms)
    if transfer and not negated_transfer and not protected_transfer and has_data and has_destination and missing_premises:
        add("CONTEXTUAL_EXTERNAL_TRANSFER", "data_disclosure", [*data_rows, *destination_rows])

    bypass = approval_bypass_present(lower)
    permission_rows = rows(("administrator", "privilege", "approval", "authorization", "access role", "elevated access"))
    if bypass and permission_rows:
        add("CONTEXTUAL_APPROVAL_BYPASS", "approval_bypass", permission_rows)
    elif _has(lower, ("grant", "elevate", "expand access")) and permission_rows:
        add("CONTEXTUAL_PRIVILEGE_GRANT", "administrator_access", permission_rows)

    policy_rows = rows(("approval required", "must be approved", "require administrator approval", "requires administrator approval", "change control"))
    if bypass and policy_rows:
        add("APPROVAL_POLICY_CONFLICT", "approval_bypass", policy_rows)
    security_rows = rows(("security", "warning", "alert", "review", "policy"))
    if security_control_bypass_present(lower) and security_rows:
        add("SECURITY_POLICY_CONFLICT", "security_bypass", security_rows)

    read_only_rows = rows((
        "read-only", "read only", "view-only", "view only", "only view",
        "may only view", "cannot modify", "must not modify", "no write access",
    ))
    modification_requested = bool(re.search(
        r"\b(?:modify|edit|write|update|delete|overwrite|change)(?:s|d|ing)?\b",
        lower,
    ))
    read_only_conflict = modification_requested and bool(read_only_rows)
    if read_only_conflict:
        add("READ_ONLY_MODIFICATION_CONFLICT", "administrator_access", read_only_rows)

    combined = (context + " " + candidate).lower()
    context_lower = context.lower()
    data_present = _has(combined, ("diagnostic", "report", "logs", "telemetry", "customer record", "sensitive"))
    destination_present = _has(context_lower, ("external", "@", "vendor", "support contact", "third-party"))
    candidate_destination = _has(lower, ("external", "@", "email address", "third-party", "provided by the user", "user-provided"))
    transfer_candidate = _has(lower, ("send", "share", "forward", "transmit", "deliver"))
    automation_present = _has(lower, ("automatic", "automatically", "without manual"))
    permission_context = _has(combined, ("administrator", "privilege", "elevated access", "access role"))
    security_bypass = security_control_bypass_present(lower)
    safe_context = _has(combined, ("public report", "approved portal", "read-only", "anonymized", "redacted", "approval required"))
    external_risk = 0.08
    if standalone_exfiltration_present(lower):
        external_risk = 0.98
        if "EXTERNAL_COMMUNICATION" not in new_capabilities:
            new_capabilities.append("EXTERNAL_COMMUNICATION")
    if destination_present and transfer_candidate:
        external_risk = max(external_risk, 0.82 if data_present else 0.55)
    if automation_present and (destination_present or candidate_destination) and data_present:
        external_risk = max(external_risk, 0.96)
        if "EXTERNAL_COMMUNICATION" not in new_capabilities:
            new_capabilities.append("EXTERNAL_COMMUNICATION")
    if safe_context and not automation_present and not bypass:
        external_risk *= 0.40
    permission_risk = 0.08
    if permission_context and _has(lower, ("grant", "elevate", "administrator", "expand access")):
        permission_risk = 0.72
    if permission_context and bypass:
        permission_risk = 0.95
        if "APPROVAL_BYPASS" not in new_capabilities:
            new_capabilities.append("APPROVAL_BYPASS")
    manipulation_risk = 0.08
    if automation_present:
        manipulation_risk = 0.58
    if automation_present and (external_risk > 0.7 or permission_risk > 0.7):
        manipulation_risk = 0.90
    if security_bypass:
        manipulation_risk = max(manipulation_risk, 0.94)
    contradiction_risk = 0.08
    if read_only_conflict:
        contradiction_risk = 0.95
        permission_risk = max(permission_risk, 0.90)
        if "DATA_MODIFICATION" not in new_capabilities:
            new_capabilities.append("DATA_MODIFICATION")
    elif bypass and _has(context_lower, ("approval required", "must be approved", "change control")):
        contradiction_risk = 0.95
    elif security_bypass and _has(context_lower, ("security", "warning", "alert", "review", "must", "required")):
        contradiction_risk = 0.95
    elif security_bypass:
        contradiction_risk = 0.80
    elif _has(lower, ("ignore", "override")):
        contradiction_risk = 0.65
    contributors = []
    for item in all_context:
        item_lower = item["text"].lower()
        if ((destination_present and _has(item_lower, ("external", "@", "vendor", "support contact")))
            or (data_present and _has(item_lower, ("diagnostic", "report", "logs", "telemetry", "customer record")))
            or (permission_context and _has(item_lower, ("administrator", "privilege", "approval", "access")))
            or (read_only_conflict and item in read_only_rows)
            or (security_bypass and _has(item_lower, ("security", "warning", "alert", "review", "policy")))):
            contributors.append(item["memory_id"])
    similarities = [float(item.get("similarity", 0)) for item in retrieved]
    density = sum(similarities) / len(similarities) if similarities else 0.0
    capability_signal = max(external_risk, permission_risk, manipulation_risk, contradiction_risk)
    dangerous = capability_signal >= 0.75
    context_signal = min(1.0, len(contributors) / 3)
    components = {"capability": capability_signal, "semantic_density": density,
                  "context_support": context_signal, "limit": float(dangerous)}
    weights = (composition_weights or COMPOSITION_WEIGHTS).copy()
    compositional = clamp(sum(weights[name] * value for name, value in components.items()))
    concern = bool(findings)
    explanation = (
        "Context-supported control chains: " + ", ".join(item["rule"] for item in findings) + "."
        if concern else "No configured context-supported control chain was detected."
    )
    return {
        "method_version": COMPOSITION_METHOD,
        "label": "HIGH RISK" if compositional >= 0.70 else "RISK" if compositional >= 0.40 else "SAFE",
        "composition_components": components,
        "composition_weights": weights,
        "weight_status": SCORING_CALIBRATION["method"],
        "composition_concern_detected": concern,
        "chain_evidence": findings,
        "capabilities_before": before, "capabilities_after": sorted(set(after)),
        "new_capabilities": sorted(set(new_capabilities)),
        "permission_risk": clamp(permission_risk), "external_risk": clamp(external_risk),
        "manipulation_risk": clamp(manipulation_risk), "contradiction_risk": clamp(contradiction_risk),
        "compositional_risk": compositional,
        "contributors": contributors, "explanation": explanation,
        "graph_evidence": {
            "method_version": COMPOSITION_METHOD,
            "semantic_density": clamp(density),
            "supporting_memories": len(contributors),
            "new_capability_detected": bool(new_capabilities),
            "dangerous_relationship_detected": dangerous,
            "edge_count_considered": len(graph_edges),
        },
    }
