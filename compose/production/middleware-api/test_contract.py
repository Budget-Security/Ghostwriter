"""
Golden-file contract test for /getProject, the response the portal backend parses.

``contract/get_project.json`` is the exact response the backend is written against: the
Budget-Security/dashboard repo keeps a byte-identical copy in
``dashboard/tests/fixtures/ghostwriter_get_project.json`` and tests its parsing against it.

The GraphQL stub is derived from the golden file by undoing what the handler does (raw
extra fields instead of ``c4Model`` / ``mitigationPrompt`` / ``retest`` / ``details`` / ``facts``,
evidence ids instead of the downloaded blobs), the handler runs on it, and its output must
equal the golden file. The stub also carries the internal extra fields the instance holds
(``INTERNAL_*`` below): the golden file has none of them, so the comparison proves they are
dropped.

When this fails because the response changed on purpose: regenerate with
``UPDATE_GOLDEN=1 pytest test_contract.py``, copy the file to the dashboard fixture, and
adjust the backend in the same change.
"""
import copy
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("FLASK_API_KEY", "test-key")
os.environ.setdefault("GHOSTWRITER_API_TOKEN", "test-token")
os.environ.setdefault("GHOSTWRITER_GRAPHQL_URL", "https://ghostwriter.test/v1/graphql")

import app as app_module  # noqa: E402
from test_endpoints import HEADERS, GraphQLStub  # noqa: E402

GOLDEN = Path(__file__).parent / "contract" / "get_project.json"

# Internal extra fields as the instance holds them. None may reach the response.
INTERNAL_FINDING = {
    "oracle_result": "pass", "client_visible": True, "root_cause_id": "search-binding",
    "merged_from": [{"title": "Export reads other statements", "suggested_by": "reviewer"}],
    "prior_rating": {"severity": "High", "score": 7.1}, "quantity_provenance": "<p>wc -l raw/x.txt</p>",
    "cwe_status": "Allowed / Allowed", "figures": [{"name": "fig1.png", "sha256": "ab12"}],
    "feedback_refs": ["FB-1"], "penny_finding_id": "F-1001", "poc": "internal",
}
INTERNAL_REPORT = {"qa_signoff": "A. Reviewer 2026-11-20", "client_feedback": [{"item": "FB-1"}],
                   "counts": {"severity": {"Critical": 1}}}
INTERNAL_PROJECT = {"roe_brute_force": False, "contracted_method": "tool-assisted",
                    "test_accounts": [{"account": "participant-a", "password": "s3cret"}]}

FINDING_DETAIL_FIELDS = {  # details key -> raw extra field
    "confidence": "confidence", "oracleDescription": "oracle_description", "rootCause": "root_cause",
    "remediationDifficulty": "remediation_difficulty", "affectedComponent": "affected_component",
    "occurrences": "occurrences", "attackChain": "attack_chain", "claimBasis": "claim_basis",
    "ratingRationale": "rating_rationale", "principalsTested": "principals_tested",
    "measurementWindow": "measurement_window", "priorIdentifiers": "prior_identifiers",
    "relatedFindings": "related_findings", "evidenceContainsLiveSecrets": "evidence_contains_live_secrets",
}
CLASSIFICATION_FIELDS = {"cweId": "cwe_id", "owaspCategory": "owasp_category", "apiTop10": "api_top10",
                         "wstg": "wstg", "asvs": "asvs", "ncscId": "ncsc_id"}
REPORT_FACT_FIELDS = {
    "version": "version", "previousVersionDelivered": "previous_version_delivered",
    "executiveSummary": "ExecutiveSummary", "engagementCleanup": "EngagementCleanup",
    "coverageNotAssessed": "CoverageNotAssessed", "identifierMapping": "identifier_mapping",
    "testingWindow": "testing_window", "captureSessions": "capture_sessions", "overallRisk": "overall_risk",
    "limitations": "limitations", "notAssessed": "not_assessed", "toolingDisclosure": "tooling_disclosure",
    "testDataLog": "test_data_log",
}


def _raw_finding_extra(details):
    raw = {FINDING_DETAIL_FIELDS[k]: copy.deepcopy(v) for k, v in details.items() if k in FINDING_DETAIL_FIELDS}
    raw.update({CLASSIFICATION_FIELDS[k]: v for k, v in details["classification"].items()})
    # Sub-fields that are internal: a link's evidence path, an occurrence's raw request file.
    for link in raw["attack_chain"]:
        link["evidence"] = "raw/injection/search-001.txt"
    for occurrence in raw["occurrences"]:
        occurrence["request"] = "raw/injection/search-002.txt"
    return dict(INTERNAL_FINDING, **raw)


def _graphql_data_from(golden):
    """The GetProject / downloadEvidence data that produces ``golden`` through the handler."""
    raw = copy.deepcopy(golden)
    downloads = {}
    raw["project_by_pk"]["extraFields"] = dict(INTERNAL_PROJECT)
    raw["project_by_pk"]["client"]["extraFields"] = {"showResultsInDashboard": True}
    for report in raw["project_by_pk"]["reports"]:
        facts = report.pop("facts")
        report["extraFields"] = dict(INTERNAL_REPORT, **{REPORT_FACT_FIELDS[k]: copy.deepcopy(v)
                                                          for k, v in facts.items()})
        model = report.pop("c4Model", None)
        if model is not None:
            report["extraFields"][app_module.REPORT_C4_MODEL_FIELD] = model
        for ev in report["evidence"]:
            downloads[ev["evidenceId"]] = ev
        report["evidence"] = [{"id": ev["evidenceId"]} for ev in report["evidence"]]
        for finding in report["findings"]:
            finding["extraFields"] = _raw_finding_extra(finding.pop("details"))
            prompt = finding.pop("mitigationPrompt", None)
            if prompt is not None:
                finding["extraFields"][app_module.FINDING_MITIGATION_PROMPT_FIELD] = prompt
            retest = finding.pop("retest", None)
            if retest is not None:
                finding["extraFields"].update({
                    app_module.FINDING_SOURCE_FINDING_ID_FIELD: retest["sourceFindingId"],
                    app_module.FINDING_SOURCE_PROJECT_ID_FIELD: retest["sourceProjectId"],
                    app_module.FINDING_RETEST_VERDICT_FIELD: retest["verdict"],
                    app_module.FINDING_POSITIVE_CONTROL_FIELD: retest["positiveControl"],
                    app_module.FINDING_RETEST_CLIENT_STATUS_FIELD: retest["clientStatus"],
                    app_module.FINDING_RETEST_CLIENT_NOTE_FIELD: retest["clientNote"],
                })
    return raw, downloads


@pytest.fixture
def client():
    app = app_module.create_app()
    app.testing = True
    return app.test_client()


def test_get_project_matches_the_golden_response(client, monkeypatch):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    project, downloads = _graphql_data_from(golden)
    stub = GraphQLStub({
        "GetProject": project,
        "downloadEvidence": lambda variables: {"downloadEvidence": downloads[variables["evidenceId"]]},
    })
    monkeypatch.setattr(app_module.requests.Session, "post", stub)

    response = client.get("/getProject?projectId=55555", headers=HEADERS)
    assert response.status_code == 200
    body = response.get_json()

    if os.environ.get("UPDATE_GOLDEN"):
        # LF on every platform: the dashboard keeps a byte-identical copy.
        GOLDEN.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8", newline="\n")
        pytest.skip("golden file rewritten; copy it to the dashboard fixture")
    assert body == golden, (
        "The /getProject response changed. If on purpose: UPDATE_GOLDEN=1 pytest test_contract.py, "
        "then copy contract/get_project.json to dashboard/tests/fixtures/ghostwriter_get_project.json"
    )


def test_golden_covers_the_fields_the_backend_reads():
    """The golden file must exercise what the backend parses, or the contract proves little."""
    report = json.loads(GOLDEN.read_text(encoding="utf-8"))["project_by_pk"]["reports"][0]
    assert report["c4Model"] and report["evidence"]
    assert any(f["mitigationPrompt"] for f in report["findings"])
    assert any(f["mitigationPrompt"] is None for f in report["findings"])
    assert all("findingGuidance" not in f for f in report["findings"])
    assert any(f["retest"] is None for f in report["findings"])
    assert any(f["retest"] and f["retest"]["verdict"] for f in report["findings"])
    assert any(f["details"]["occurrences"] and f["details"]["attackChain"] for f in report["findings"])
    assert any(not f["details"]["occurrences"] for f in report["findings"])
    assert report["facts"]["testDataLog"] and report["facts"]["executiveSummary"]
    # Nothing raw: every extra field the backend may read is lifted, the rest is gone.
    text = GOLDEN.read_text(encoding="utf-8")
    assert '"extraFields"' not in text and "penny" not in text.lower()
