"""NIST-referenced control concerns; no numerical severity estimates."""

METHOD_VERSION = "nist80053-control-flags-v2"
SPEC_URL = "https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final"
CATALOG_URL = "https://csrc.nist.gov/CSRC/media/Projects/risk-management/800-53%20Downloads/800-53r5/SP_800-53_v5_1-derived-OSCAL.pdf"


def _profile(controls: list[str], rationale: str, page: int) -> dict:
    return {
        "controls": controls,
        "rationale": rationale,
        "references": [SPEC_URL, f"{CATALOG_URL}#page={page}"],
        "source_version": "NIST SP 800-53 Rev. 5, pinned 5.1 derivative catalog",
    }


PROFILES = {
    "approval_bypass": _profile(["AC-3 Access Enforcement"],
        "AC-3 requires enforcement of approved access authorizations. Bypass wording warrants authorization review.", 24),
    "security_bypass": _profile(["CM-5 Access Restrictions for Change", "AC-6(1) Authorize Access to Security Functions"],
        "Security-control changes require authorized access and change restrictions; disabling a control is not automatically unauthorized.", 106),
    "audit_deletion": _profile(["AU-9 Protection of Audit Information"],
        "AU-9 protects audit information and logging tools against unauthorized deletion or modification; approved retention operations may be legitimate.", 78),
    "data_disclosure": _profile(["AC-4 Information Flow Enforcement"],
        "AC-4 requires approved information-flow authorizations. Check the destination and release authorization for confidential transfers.", 30),
    "credential_disclosure": _profile(["IA-5 Authenticator Management"],
        "IA-5 includes protecting authenticator content against unauthorized disclosure or modification. Check any credential-sharing instruction.", 144),
    "administrator_access": _profile(["AC-6 Least Privilege", "AC-2 Account Management"],
        "Privileged access must be authorized and necessary for assigned tasks; an administrator grant needs contextual authorization review.", 39),
}

PHRASE_PROFILES = {
    "bypass approval": "approval_bypass",
    "without approval": "approval_bypass",
    "disable security": "security_bypass",
    "ignore access policy": "security_bypass",
    "send credentials": "credential_disclosure",
    "share password": "credential_disclosure",
    "delete audit": "audit_deletion",
    "upload confidential": "data_disclosure",
    "grant administrator": "administrator_access",
}
PATTERN_PROFILES = {
    "approval or authorization checks can be bypassed": "approval_bypass",
    "automatic or unrestricted external transfer of sensitive operational data": "data_disclosure",
    "security warnings or controls can be ignored or bypassed": "security_bypass",
}


def rule_evidence(rule: str) -> dict:
    profile_id = {**PHRASE_PROFILES, **PATTERN_PROFILES}[rule]
    return {"rule": rule, "profile": profile_id, **PROFILES[profile_id]}
