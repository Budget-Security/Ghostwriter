"""
Golden-file contract test for /getProject, the response the portal backend parses.

``contract/get_project.json`` is the exact response the backend is written against: the
Budget-Security/dashboard repo keeps a byte-identical copy in
``dashboard/tests/fixtures/ghostwriter_get_project.json`` and tests its parsing against it.

The GraphQL stub is derived from the golden file by undoing what the handler does (raw
extra fields instead of ``c4Model`` / ``mitigationPrompt``, evidence ids instead of the
downloaded blobs), the handler runs on it, and its output must equal the golden file.

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


def _graphql_data_from(golden):
    """The GetProject / downloadEvidence data that produces ``golden`` through the handler."""
    raw = copy.deepcopy(golden)
    downloads = {}
    for report in raw["project_by_pk"]["reports"]:
        model = report.pop("c4Model", None)
        if model is not None:
            report["extraFields"][app_module.REPORT_C4_MODEL_FIELD] = model
        for ev in report["evidence"]:
            downloads[ev["evidenceId"]] = ev
        report["evidence"] = [{"id": ev["evidenceId"]} for ev in report["evidence"]]
        for finding in report["findings"]:
            prompt = finding.pop("mitigationPrompt", None)
            if prompt is not None:
                finding["extraFields"][app_module.FINDING_MITIGATION_PROMPT_FIELD] = prompt
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
        GOLDEN.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
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
