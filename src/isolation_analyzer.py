"""Candidate-only baseline analysis."""

from __future__ import annotations

import re

from src.isolation_evidence import (
    METHOD_VERSION, PHRASE_PROFILES, rule_evidence,
)


HIGH_RISK_PHRASES = tuple(PHRASE_PROFILES)


def approval_bypass_present(text: str) -> bool:
    """Detect active/passive approval bypass wording without flagging prohibitions."""
    lower = text.lower()
    approval_present = bool(re.search(r"\b(?:approval|authorization|verification|sign-off)\b", lower))
    bypass_pattern = r"\b(?:skip(?:s|ped|ping)?|bypass(?:es|ed|ing)?|override(?:s|n|ing)?|circumvent(?:s|ed|ing)?)\b"
    bypass_present = bool(re.search(bypass_pattern, lower))
    negated_bypass = bool(re.search(
        r"\b(?:cannot|can't|must\s+not|never|may\s+not|should\s+not|do\s+not|does\s+not)"
        r"(?:\s+\w+){0,4}\s+" + bypass_pattern,
        lower,
    )) or bool(re.search(r"\bnot\s+allowed\s+to\s+" + bypass_pattern, lower))
    return approval_present and bypass_present and not negated_bypass


def standalone_exfiltration_present(text: str) -> bool:
    """Detect a complete sensitive-data-to-external-destination instruction."""
    lower = text.lower()
    sensitive = any(term in lower for term in (
        "diagnostic log", "audit log", "credentials", "password", "customer record",
        "confidential", "sensitive data", "system information", "host identifier",
    ))
    transfer = bool(re.search(
        r"\b(?:send|forward|share|upload|email|transmit|deliver)(?:s|ed|ing)?\b", lower
    ))
    external = any(term in lower for term in (
        "external", "email address", "third-party", "outside", "vendor address",
        "provided by the user", "user-provided address", "@",
    ))
    unrestricted = any(term in lower for term in (
        "any external", "any email", "provided by the user", "user-provided",
        "arbitrary", "unverified",
    ))
    automated = any(term in lower for term in ("automatic", "automatically", "without review"))
    safe_controls = any(term in lower for term in (
        "anonymized", "redacted", "public report", "approved portal",
        "approved destination", "after approval", "approval required",
    ))
    negated_transfer = bool(re.search(
        r"\b(?:do\s+not|does\s+not|must\s+not|never|cannot|can't|should\s+not)"
        r"(?:\s+\w+){0,4}\s+"
        r"(?:send|forward|share|upload|email|transmit|deliver)(?:s|ed|ing)?\b",
        lower,
    ))
    dangerous_chain = sensitive and transfer and external and (automated or unrestricted)
    return dangerous_chain and not negated_transfer and not safe_controls


def security_control_bypass_present(text: str) -> bool:
    """Detect instructions allowing security controls or warnings to be disregarded."""
    lower = text.lower()
    security_control = bool(re.search(
        r"\b(?:security|safety|authentication|authorization|access)\s+"
        r"(?:warning|alert|check|control|policy|restriction|safeguard)s?\b",
        lower,
    ))
    bypass_pattern = (
        r"\b(?:ignore(?:s|d|ing)?|dismiss(?:es|ed|ing)?|skip(?:s|ped|ping)?|"
        r"override(?:s|n|ing)?|disable(?:s|d|ing)?|suppress(?:es|ed|ing)?|"
        r"bypass(?:es|ed|ing)?)\b"
    )
    bypass_action = bool(re.search(bypass_pattern, lower))
    negated_bypass = bool(re.search(
        r"\b(?:cannot|can't|must\s+not|never|may\s+not|should\s+not|"
        r"do\s+not|does\s+not|are\s+not|is\s+not)"
        r"(?:\s+\w+){0,4}\s+" + bypass_pattern,
        lower,
    )) or bool(re.search(r"\bnot\s+allowed\s+to\s+" + bypass_pattern, lower))
    return security_control and bypass_action and not negated_bypass


def analyze_isolation(candidate: str) -> dict:
    """Analyze only the candidate, deliberately excluding retrieved context."""
    lower = candidate.lower()
    matched = [phrase for phrase in HIGH_RISK_PHRASES if phrase in lower]
    if approval_bypass_present(lower) and not any("approval" in item for item in matched):
        matched.append("approval or authorization checks can be bypassed")
    if standalone_exfiltration_present(lower):
        matched.append("automatic or unrestricted external transfer of sensitive operational data")
    if security_control_bypass_present(lower):
        matched.append("security warnings or controls can be ignored or bypassed")
    evidence = [rule_evidence(rule) for rule in matched]
    score = float(bool(evidence))

    explanation = (
        "Potential security-control concern(s): " + ", ".join(matched) + ". Authorization review required."
        if matched else
        "No configured control rule matched; this does not establish safety or compliance."
    )
    label = "CONTROL CONCERN" if matched else "NO MATCH"
    return {
        "score": score, "label": label, "explanation": explanation,
        "control_concern_detected": bool(evidence),
        "matched_rules": matched, "evidence": evidence,
        "method_version": METHOD_VERSION,
        "score_kind": "binary_control_concern_indicator",
    }
