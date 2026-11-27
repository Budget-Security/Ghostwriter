"""
Endpoint tests for the middleware API.

``test_app.py`` covers the ``/runPoc`` PoC-runner logic. This module covers the
request/response contract of every route: auth, input validation, and - most
importantly - the Python that reads GraphQL responses.

That last part is the gap the CI schema check cannot fill. ``validate_graphql.py``
proves each query still matches the Hasura schema, but it cannot prove the handler
still reads the right keys out of the result. Commit 145b5529 broke exactly that
way: evidence moved from finding-scoped to report-scoped, and because the handler
used ``.get(..., [])`` the mismatch produced empty data rather than an error.

Run from this directory:

    pip install -r requirements.txt pytest
    pytest
"""
import base64
import copy
import json
import os

import pytest


os.environ.setdefault("FLASK_API_KEY", "test-key")
os.environ.setdefault("GHOSTWRITER_API_TOKEN", "test-token")
os.environ.setdefault("GHOSTWRITER_GRAPHQL_URL", "https://ghostwriter.test/v1/graphql")

import app as app_module  # noqa: E402


HEADERS = {"X-API-Key": "test-key"}

# Every route, with the method and a minimal valid-shaped body. Used by the
# blanket auth test so a new route cannot be added without an auth check.
ROUTES = [
    ("post", "/createClient", {"name": "Acme"}),
    ("post", "/updateClient", {"clientId": 1, "name": "Acme"}),
    ("post", "/uploadClientLogo", {"clientId": 1, "fileBase64": "eA==", "fileName": "l.png"}),
    ("get", "/getClient?clientId=1", None),
    ("post", "/createProject", {"clientId": 1, "projectTypeId": 1, "startDate": "2026-01-01", "endDate": "2026-01-02"}),
    ("post", "/updateProject", {"projectId": 1, "codename": "x"}),
    ("get", "/getProjects?clientId=1", None),
    ("get", "/getProject?projectId=1", None),
    ("get", "/getFindingSummaries?projectIds=1", None),
    ("get", "/getEvidence?projectId=1&evidenceId=1", None),
    ("post", "/createRetest", {"projectId": 1}),
    ("get", "/getStatistics?clientId=1", None),
    ("post", "/generateReport", {"projectId": 1, "templateId": 1}),
    ("post", "/runPoc", {"finding_id": 1}),
]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class GraphQLStub:
    """
    Fake ``requests.Session.post`` that answers GraphQL calls from a routing table
    and records every (query, variables) pair for assertions.

    ``routes`` maps a substring of the query to either a ``data`` dict or a
    callable taking the variables and returning one.
    """

    def __init__(self, routes, default=None):
        self.routes = routes
        self.default = default
        self.calls = []

    # Assigned onto Session.post as a plain object, so there is no descriptor
    # binding and no implicit ``self`` for the session - ``url`` arrives first.
    def __call__(self, url, json=None, timeout=None, **kwargs):
        query = (json or {}).get("query", "")
        variables = (json or {}).get("variables", {})
        self.calls.append({"query": query, "variables": variables, "timeout": timeout})

        for needle, response in self.routes.items():
            if needle in query:
                if callable(response):
                    response = response(variables)
                if isinstance(response, FakeResponse):
                    return response
                return FakeResponse(payload={"data": response})

        if self.default is not None:
            return FakeResponse(payload={"data": self.default})
        raise AssertionError(f"unexpected graphql query: {query.strip()[:80]}")

    def queries_matching(self, needle):
        return [c for c in self.calls if needle in c["query"]]

    def variables_for(self, needle):
        matches = self.queries_matching(needle)
        assert matches, f"no graphql call matching {needle!r}"
        return matches[0]["variables"]


@pytest.fixture
def client():
    app = app_module.create_app()
    app.testing = True
    return app.test_client()


@pytest.fixture
def stub(monkeypatch):
    """Install a GraphQLStub; tests set ``.routes`` before calling the endpoint."""
    holder = GraphQLStub({})
    monkeypatch.setattr(app_module.requests.Session, "post", holder)
    return holder


def call(client, method, path, body=None):
    fn = getattr(client, method)
    if method == "get":
        return fn(path, headers=HEADERS)
    return fn(path, json=body, headers=HEADERS)


# --- Auth -------------------------------------------------------------------

@pytest.mark.parametrize("method,path,body", ROUTES, ids=[r[1].split("?")[0] for r in ROUTES])
def test_route_requires_api_key(client, method, path, body):
    fn = getattr(client, method)
    resp = fn(path) if method == "get" else fn(path, json=body)
    assert resp.status_code == 401
    assert resp.get_json() == {"error": "unauthorized"}


@pytest.mark.parametrize("method,path,body", ROUTES, ids=[r[1].split("?")[0] for r in ROUTES])
def test_route_rejects_wrong_api_key(client, method, path, body):
    headers = {"X-API-Key": "wrong"}
    fn = getattr(client, method)
    resp = fn(path, headers=headers) if method == "get" else fn(path, json=body, headers=headers)
    assert resp.status_code == 401


# --- /createClient ----------------------------------------------------------

def test_create_client_requires_name(client):
    resp = call(client, "post", "/createClient", {})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "name is required"}


def test_create_client_happy_path(client, stub):
    stub.routes = {"insert_client_one": {"insert_client_one": {"id": 7}}}
    resp = call(client, "post", "/createClient", {
        "name": "Acme", "shortName": "ACM", "contacts": {"data": [{"name": "Bob"}]},
    })
    assert resp.status_code == 200
    assert resp.get_json() == {"insert_client_one": {"id": 7}}
    # contacts is a valid nested insert on client_insert_input and must survive.
    assert stub.variables_for("insert_client_one")["object"] == {
        "name": "Acme", "shortName": "ACM", "contacts": {"data": [{"name": "Bob"}]},
    }


def test_create_client_drops_unknown_and_logo_fields(client, stub):
    stub.routes = {"insert_client_one": {"insert_client_one": {"id": 7}}}
    call(client, "post", "/createClient", {
        "name": "Acme", "logo_base64": "eA==", "nope": "x",
    })
    obj = stub.variables_for("insert_client_one")["object"]
    assert obj == {"name": "Acme"}, "logo_base64 goes via /uploadClientLogo, unknown keys are dropped"


def test_create_client_graphql_error_is_502(client, stub):
    stub.routes = {"insert_client_one": FakeResponse(payload={"errors": [{"message": "boom"}]})}
    resp = call(client, "post", "/createClient", {"name": "Acme"})
    assert resp.status_code == 502
    assert resp.get_json()["error"]["message"] == "graphql error"


def test_create_client_missing_id_is_502(client, stub):
    stub.routes = {"insert_client_one": {"insert_client_one": {}}}
    resp = call(client, "post", "/createClient", {"name": "Acme"})
    assert resp.status_code == 502
    assert resp.get_json() == {"error": "GraphQL did not return client id"}


# --- /updateClient ----------------------------------------------------------

def test_update_client_requires_client_id(client):
    resp = call(client, "post", "/updateClient", {"name": "Acme"})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "clientId is required"}


def test_update_client_requires_fields(client):
    resp = call(client, "post", "/updateClient", {"clientId": 1})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "no fields to update"}


def test_update_client_happy_path(client, stub):
    stub.routes = {"update_client_by_pk": {"update_client_by_pk": {"id": 1, "name": "N", "shortName": "S"}}}
    resp = call(client, "post", "/updateClient", {"clientId": 1, "name": "N"})
    assert resp.status_code == 200
    assert resp.get_json()["update_client_by_pk"]["name"] == "N"
    assert stub.variables_for("update_client_by_pk") == {"clientId": 1, "set": {"name": "N"}}


def test_update_client_cannot_set_show_results_in_dashboard(client, stub):
    """The dashboard visibility flag is operator-controlled; the API must not set it."""
    stub.routes = {"update_client_by_pk": {"update_client_by_pk": {"id": 1}}}
    call(client, "post", "/updateClient", {
        "clientId": 1,
        "extraFields": {"showResultsInDashboard": True, "other": "keep"},
    })
    assert stub.variables_for("update_client_by_pk")["set"]["extraFields"] == {"other": "keep"}


def test_update_client_coerces_string_client_id(client, stub):
    stub.routes = {"update_client_by_pk": {"update_client_by_pk": {"id": 1}}}
    call(client, "post", "/updateClient", {"clientId": "1", "name": "N"})
    assert stub.variables_for("update_client_by_pk")["clientId"] == 1


@pytest.mark.xfail(reason="KNOWN BUG: contacts is read but never applied - silent no-op", strict=True)
def test_update_client_applies_contacts(client, stub):
    stub.routes = {"contact": {"insert_contact": {"affected_rows": 1}}}
    resp = call(client, "post", "/updateClient", {"clientId": 1, "contacts": [{"name": "Bob"}]})
    assert resp.status_code == 200
    assert stub.queries_matching("contact"), "contacts were accepted but no mutation was sent"


# --- /uploadClientLogo ------------------------------------------------------

@pytest.mark.parametrize("missing,expected", [
    ("clientId", "clientId is required"),
    ("fileBase64", "fileBase64 is required"),
    ("fileName", "fileName is required"),
])
def test_upload_client_logo_requires_fields(client, missing, expected):
    body = {"clientId": 1, "fileBase64": "eA==", "fileName": "logo.png"}
    del body[missing]
    resp = call(client, "post", "/uploadClientLogo", body)
    assert resp.status_code == 400
    assert resp.get_json() == {"error": expected}


def test_upload_client_logo_rejects_non_integer_id(client):
    resp = call(client, "post", "/uploadClientLogo", {
        "clientId": "abc", "fileBase64": "eA==", "fileName": "logo.png",
    })
    assert resp.status_code == 400
    assert "clientId must be an integer" in str(resp.get_json())


def test_upload_client_logo_happy_path(client, stub):
    stub.routes = {"uploadClientLogo": {"uploadClientLogo": {"id": 3}}}
    resp = call(client, "post", "/uploadClientLogo", {
        "clientId": "1", "fileBase64": "eA==", "fileName": "logo.png",
    })
    assert resp.status_code == 200
    assert resp.get_json() == {"uploadClientLogo": {"id": 3}}
    # The action argument names differ from the REST body names.
    assert stub.variables_for("uploadClientLogo") == {
        "clientId": 1, "file_base64": "eA==", "filename": "logo.png",
    }


# --- /getClient -------------------------------------------------------------

def test_get_client_requires_client_id(client):
    resp = call(client, "get", "/getClient")
    assert resp.status_code == 400


def test_get_client_rejects_non_integer_id(client):
    resp = call(client, "get", "/getClient?clientId=abc")
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "clientId must be an integer"}


def test_get_client_happy_path(client, stub):
    stub.routes = {"client_by_pk": {"client_by_pk": {"id": 1, "name": "Acme", "extraFields": {}}}}
    resp = call(client, "get", "/getClient?clientId=1")
    assert resp.status_code == 200
    assert resp.get_json()["client_by_pk"]["name"] == "Acme"
    assert stub.variables_for("client_by_pk") == {"clientId": 1}


# --- /createProject ---------------------------------------------------------

def test_create_project_reports_all_missing_fields(client):
    resp = call(client, "post", "/createProject", {"clientId": 1})
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["error"] == "missing required fields"
    assert set(payload["fields"]) == {"projectTypeId", "startDate", "endDate"}


def test_create_project_happy_path(client, stub):
    stub.routes = {"insert_project_one": {"insert_project_one": {"id": 12}}}
    resp = call(client, "post", "/createProject", {
        "clientId": 1, "projectTypeId": 2, "startDate": "2026-01-01", "endDate": "2026-01-31",
        "codename": "BLUE", "nope": "dropped",
    })
    assert resp.status_code == 200
    assert resp.get_json() == {"insert_project_one": {"id": 12}}
    obj = stub.variables_for("insert_project_one")["object"]
    assert "nope" not in obj
    assert obj["codename"] == "BLUE"


# --- /updateProject ---------------------------------------------------------

def test_update_project_requires_project_id(client):
    resp = call(client, "post", "/updateProject", {"codename": "x"})
    assert resp.status_code == 400


def test_update_project_requires_fields(client):
    resp = call(client, "post", "/updateProject", {"projectId": 1})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "no fields to update"}


def test_update_project_happy_path(client, stub):
    stub.routes = {"update_project_by_pk": {"update_project_by_pk": {"id": 1}}}
    resp = call(client, "post", "/updateProject", {"projectId": "1", "codename": "RED", "nope": 1})
    assert resp.status_code == 200
    assert stub.variables_for("update_project_by_pk") == {"projectId": 1, "set": {"codename": "RED"}}


# --- /getProjects -----------------------------------------------------------

def test_get_projects_requires_client_id(client):
    resp = call(client, "get", "/getProjects")
    assert resp.status_code == 400


def test_get_projects_happy_path(client, stub):
    stub.routes = {"ProjectsByClient": {"project": [{"id": 1, "codename": "BLUE"}]}}
    resp = call(client, "get", "/getProjects?clientId=1")
    assert resp.status_code == 200
    assert resp.get_json()["project"][0]["codename"] == "BLUE"


# --- /getProject ------------------------------------------------------------

def _project_payload(extra_fields):
    """A /getProject response with one report carrying a finding and evidence."""
    return {
        "project_by_pk": {
            "id": 1,
            "codename": "BLUE",
            "description": "desc",
            "clientId": 9,
            "client": {"id": 9, "name": "Acme", "shortName": "ACM", "extraFields": extra_fields},
            "projectType": {"id": 2, "projectType": "Pentest"},
            "reports": [{
                "id": 5,
                "title": "Report A",
                "evidence": [{"id": 77}],
                "findings": [{"id": 31, "title": "SQLi", "description": "secret detail"}],
            }],
        }
    }


def _get_project_routes(extra_fields):
    return {
        "GetProject": _project_payload(extra_fields),
        "downloadEvidence": {"downloadEvidence": {
            "evidenceId": 77, "filename": "e.png", "friendlyName": "Shot", "fileBase64": "eA==",
        }},
    }


def test_get_project_requires_project_id(client):
    resp = call(client, "get", "/getProject")
    assert resp.status_code == 400


def test_get_project_rejects_non_integer_id(client):
    resp = call(client, "get", "/getProject?projectId=abc")
    assert resp.status_code == 400


def test_get_project_returns_findings_when_flag_true(client, stub):
    stub.routes = _get_project_routes({"showResultsInDashboard": True})
    resp = call(client, "get", "/getProject?projectId=1")
    assert resp.status_code == 200
    report = resp.get_json()["project_by_pk"]["reports"][0]
    assert [f["title"] for f in report["findings"]] == ["SQLi"]
    # Evidence ids are replaced by downloaded blobs.
    assert report["evidence"] == [{
        "evidenceId": 77, "filename": "e.png", "friendlyName": "Shot", "fileBase64": "eA==",
    }]
    assert len(stub.queries_matching("downloadEvidence")) == 1


@pytest.mark.parametrize("extra_fields", [
    {"showResultsInDashboard": False},
    {},
    None,
    {"showResultsInDashboard": "true"},  # JSONB string is not a boolean true
], ids=["false", "key-absent", "null-extrafields", "string-true"])
def test_get_project_hides_findings_when_flag_not_true(client, stub, extra_fields):
    stub.routes = _get_project_routes(extra_fields)
    resp = call(client, "get", "/getProject?projectId=1")
    assert resp.status_code == 200
    report = resp.get_json()["project_by_pk"]["reports"][0]
    assert report["findings"] == []
    assert report["evidence"] == []


def test_get_project_skips_evidence_downloads_when_hidden(client, stub):
    """Hiding findings must also skip the per-evidence download round trips."""
    stub.routes = _get_project_routes({"showResultsInDashboard": False})
    call(client, "get", "/getProject?projectId=1")
    assert stub.queries_matching("downloadEvidence") == []


@pytest.mark.parametrize("extra_fields", [{"showResultsInDashboard": True}, {"showResultsInDashboard": False}])
def test_get_project_always_returns_project_details(client, stub, extra_fields):
    """The flag gates finding detail only - project, client and type always return."""
    stub.routes = _get_project_routes(extra_fields)
    project = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]
    assert project["codename"] == "BLUE"
    assert project["description"] == "desc"
    assert project["client"]["name"] == "Acme"
    assert project["projectType"]["projectType"] == "Pentest"
    assert project["reports"][0]["title"] == "Report A"


def test_get_project_queries_report_scoped_evidence(client, stub):
    """
    Regression guard for 145b5529: evidence is report-scoped, not finding-scoped.
    The schema check catches a bad query; this catches the handler reading the
    wrong key and silently returning nothing.
    """
    stub.routes = _get_project_routes({"showResultsInDashboard": True})
    call(client, "get", "/getProject?projectId=1")
    query = stub.queries_matching("GetProject")[0]["query"]
    assert "evidences" not in query, "evidences was the pre-145b5529 finding-scoped field"
    findings_at = query.index("findings(")
    evidence_at = query.index("evidence {")
    assert evidence_at < findings_at, "evidence must be selected on the report, not the finding"


def test_get_project_does_not_select_finding_guidance(client, stub):
    """
    Finding guidance is the tester's note to the report writer (internal ids, evidence
    paths) and must never reach the portal. Removed in 07e4bb20; this keeps it out.
    """
    stub.routes = _get_project_routes({"showResultsInDashboard": True})
    call(client, "get", "/getProject?projectId=1")
    query = stub.queries_matching("GetProject")[0]["query"]
    assert "findingGuidance" not in query


C4_MODEL = {"generated": "2026-10-23T07:48:22Z", "model": {"views": {"index": {}}}, "elements": {}, "views": []}


def _project_with_extra_fields(client_extra, report_extra, finding_extra):
    payload = _project_payload(client_extra)
    report = payload["project_by_pk"]["reports"][0]
    report["extraFields"] = report_extra
    report["findings"][0]["extraFields"] = finding_extra
    return {"GetProject": payload, "downloadEvidence": _get_project_routes(client_extra)["downloadEvidence"]}


def test_get_project_lifts_mitigation_prompt_and_c4_model(client, stub):
    stub.routes = _project_with_extra_fields(
        {"showResultsInDashboard": True},
        {"c4_model": C4_MODEL, "other": "keep"},
        {"mitigation_prompt": "# Fix: SQLi\n\nUse bound parameters.", "poc": "keep"},
    )
    report = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]
    assert report["c4Model"] == C4_MODEL
    assert report["findings"][0]["mitigationPrompt"] == "# Fix: SQLi\n\nUse bound parameters."
    # Carried once: the raw copies leave extraFields, everything else stays.
    # No raw extra fields leave Ghostwriter: a field not on the allow-list is dropped.
    assert "extraFields" not in report
    assert "extraFields" not in report["findings"][0]


@pytest.mark.parametrize("report_extra,finding_extra", [
    (None, None),
    ({}, {}),
    ({"c4_model": None}, {"mitigation_prompt": None}),
    ({"c4_model": {}}, {"mitigation_prompt": ""}),  # saved empty
    ({"c4_model": "not json"}, {"mitigation_prompt": "   "}),
    ({"c4_model": ["a list"]}, {"mitigation_prompt": 42}),
], ids=["null-extrafields", "keys-absent", "null", "empty", "garbage", "wrong-type"])
def test_get_project_unset_extra_fields_are_null(client, stub, report_extra, finding_extra):
    """Every way of not having a value reads as null, never as {} or a blank string."""
    stub.routes = _project_with_extra_fields({"showResultsInDashboard": True}, report_extra, finding_extra)
    report = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]
    assert report["c4Model"] is None
    assert report["findings"][0]["mitigationPrompt"] is None


def test_get_project_parses_c4_model_written_as_json_text(client, stub):
    import json
    stub.routes = _project_with_extra_fields(
        {"showResultsInDashboard": True}, {"c4_model": json.dumps(C4_MODEL)}, {}
    )
    report = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]
    assert report["c4Model"] == C4_MODEL


def test_get_project_hides_c4_model_when_flag_not_true(client, stub):
    """The model maps systems to findings, so it is finding detail and follows the flag."""
    stub.routes = _project_with_extra_fields(
        {"showResultsInDashboard": False}, {"c4_model": C4_MODEL}, {"mitigation_prompt": "x"}
    )
    report = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]
    assert report["c4Model"] is None
    assert "extraFields" not in report and report["facts"] is None
    assert report["findings"] == []


RETEST_EXTRA_FIELDS = {
    "source_finding_id": 22,
    "source_project_id": 19,
    "retest_verdict": "still_present",
    "positive_control": "<p>The check still rejects the payload.</p>",
    "retest_client_status": "Fixed",
    "retest_client_note": "<p>Patched in 2.4</p>",
}


def test_get_project_lifts_the_retest_block(client, stub):
    stub.routes = _project_with_extra_fields(
        {"showResultsInDashboard": True}, {}, dict(RETEST_EXTRA_FIELDS, poc="keep"),
    )
    finding = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]["findings"][0]
    assert finding["retest"] == {
        "sourceFindingId": 22,
        "sourceProjectId": 19,
        "verdict": "still_present",
        "positiveControl": "<p>The check still rejects the payload.</p>",
        "clientStatus": "Fixed",
        "clientNote": "<p>Patched in 2.4</p>",
    }
    assert "extraFields" not in finding


def test_get_project_retest_block_defaults_unset_values(client, stub):
    """A copy whose verdict is not given yet: strings read as "", a missing project id as null."""
    stub.routes = _project_with_extra_fields(
        {"showResultsInDashboard": True}, {}, {"source_finding_id": "22", "retest_verdict": None},
    )
    finding = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]["findings"][0]
    assert finding["retest"] == {
        "sourceFindingId": 22, "sourceProjectId": None, "verdict": "", "positiveControl": "",
        "clientStatus": "", "clientNote": "",
    }
    assert "extraFields" not in finding


@pytest.mark.parametrize("finding_extra", [
    None,
    {},
    {"source_finding_id": None, "retest_verdict": "fixed"},
    {"source_finding_id": ""},
    {"source_finding_id": "abc"},
], ids=["null-extrafields", "keys-absent", "null-id", "blank-id", "garbage-id"])
def test_get_project_retest_is_null_without_a_source_finding(client, stub, finding_extra):
    stub.routes = _project_with_extra_fields({"showResultsInDashboard": True}, {}, finding_extra)
    finding = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]["findings"][0]
    assert finding["retest"] is None
    assert "extraFields" not in finding


def test_get_project_retest_block_survives_an_edited_copy(client, stub):
    """
    Ghostwriter's editor keeps only extra fields that have a spec. A copy in a "Retest" report
    whose unspecified source_finding_id was dropped by an edit still carries its verdict.
    """
    routes = _project_with_extra_fields(
        {"showResultsInDashboard": True}, {},
        {"retest_verdict": "still-present", "positive_control": "<p>Still fires.</p>", "poc": "keep"},
    )
    routes["GetProject"]["project_by_pk"]["reports"][0]["title"] = "Retest"
    stub.routes = routes
    finding = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]["findings"][0]
    assert finding["retest"] == {
        "sourceFindingId": None, "sourceProjectId": None, "verdict": "still-present",
        "positiveControl": "<p>Still fires.</p>", "clientStatus": "", "clientNote": "",
    }
    assert "extraFields" not in finding


def test_get_project_retest_block_needs_a_source_outside_a_retest_report(client, stub):
    """Outside a "Retest" report a verdict alone does not make a finding a copy."""
    stub.routes = _project_with_extra_fields(
        {"showResultsInDashboard": True}, {}, {"retest_verdict": "fixed"},
    )
    finding = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]["findings"][0]
    assert finding["retest"] is None


def test_get_project_returns_the_retest_report(client, stub):
    """The query hides only the legacy "Original Findings" report, so "Retest" comes back."""
    stub.routes = _get_project_routes({"showResultsInDashboard": True})
    call(client, "get", "/getProject?projectId=1")
    query = stub.queries_matching("GetProject")[0]["query"]
    assert 'reports(where: {title: {_neq: "Original Findings"}})' in query
    assert '"Retest"' not in query


def test_get_project_hides_the_retest_block_when_flag_not_true(client, stub):
    stub.routes = _project_with_extra_fields({"showResultsInDashboard": False}, {}, dict(RETEST_EXTRA_FIELDS))
    report = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]
    assert report["findings"] == []


def test_get_project_graphql_error_is_502(client, stub):
    stub.routes = {"GetProject": FakeResponse(status_code=500, text="nope")}
    resp = call(client, "get", "/getProject?projectId=1")
    assert resp.status_code == 502


def test_get_project_evidence_none_lists_metadata_without_downloads(client, stub):
    routes = _get_project_routes({"showResultsInDashboard": True})
    routes["GetProject"]["project_by_pk"]["reports"][0]["evidence"] = [
        {"id": 77, "friendlyName": "Shot", "document": "evidence/5/shot.png"},
    ]
    stub.routes = routes
    resp = call(client, "get", "/getProject?projectId=1&evidence=none")
    assert resp.status_code == 200
    report = resp.get_json()["project_by_pk"]["reports"][0]
    assert report["evidence"] == [{"evidenceId": 77, "filename": "shot.png", "friendlyName": "Shot"}]
    assert [f["title"] for f in report["findings"]] == ["SQLi"]
    assert stub.queries_matching("downloadEvidence") == []


def test_get_project_evidence_none_still_hides_when_flag_not_true(client, stub):
    stub.routes = _get_project_routes({"showResultsInDashboard": False})
    report = call(client, "get", "/getProject?projectId=1&evidence=none").get_json()["project_by_pk"]["reports"][0]
    assert report["findings"] == []
    assert report["evidence"] == []


def test_get_project_rejects_unknown_evidence_mode(client):
    resp = call(client, "get", "/getProject?projectId=1&evidence=links")
    assert resp.status_code == 400


# --- /getFindingSummaries ---------------------------------------------------

def _summaries_payload(extra_fields=None):
    return {"project": [{
        "id": 1,
        "codename": "BLUE",
        "complete": False,
        "client": {"id": 9, "extraFields": extra_fields},
        "reports": [{
            "id": 5,
            "findings": [
                {
                    "id": 31, "title": "SQLi", "position": 1, "complete": True, "cvssScore": 9.1,
                    "description": "<p>The search   endpoint</p><div class=\"richtext-evidence\" data-evidence-id=\"77\"></div><p>concatenates input.</p>",
                    "severity": {"severity": "Critical", "weight": 1},
                },
                {
                    "id": 32, "title": "Banner", "position": 2, "complete": False, "cvssScore": None,
                    "description": None, "severity": None,
                },
            ],
        }],
    }]}


def test_get_finding_summaries_requires_project_ids(client):
    assert call(client, "get", "/getFindingSummaries").status_code == 400
    assert call(client, "get", "/getFindingSummaries?projectIds=").status_code == 400


def test_get_finding_summaries_rejects_non_integer_ids(client):
    assert call(client, "get", "/getFindingSummaries?projectIds=1,abc").status_code == 400


def test_get_finding_summaries_caps_the_number_of_projects(client):
    ids = ",".join(str(i) for i in range(1, 202))
    assert call(client, "get", f"/getFindingSummaries?projectIds={ids}").status_code == 400


def test_get_finding_summaries_returns_lean_findings(client, stub):
    stub.routes = {"FindingSummaries": _summaries_payload({"showResultsInDashboard": True})}
    resp = call(client, "get", "/getFindingSummaries?projectIds=1,1,2")
    assert resp.status_code == 200
    assert stub.variables_for("FindingSummaries") == {"projectIds": [1, 2]}
    project = resp.get_json()["projects"][0]
    assert project["id"] == 1 and project["codename"] == "BLUE" and project["showResults"] is True
    assert project["hasC4Model"] is False
    first, second = project["findings"]
    assert first == {
        "id": 31, "reportId": 5, "title": "SQLi", "summary": "The search endpoint concatenates input.",
        "position": 1, "complete": True, "cvssScore": 9.1, "severity": "Critical", "severityWeight": 1,
        "sourceFindingId": None, "retestVerdict": "", "confidence": "",
    }
    assert second["summary"] == "" and second["severity"] is None and second["complete"] is False


@pytest.mark.parametrize("source_finding_id, verdict, expected", [
    (22, "still_present", (22, "still_present")),
    ("22", None, (22, "")),  # integer field written as text; verdict not given yet
    (None, None, (None, "")),
    ("", 5, (None, "")),  # garbage reads as unset
], ids=["copy-with-verdict", "copy-without-verdict", "not-a-copy", "garbage"])
def test_get_finding_summaries_returns_retest_fields(client, stub, source_finding_id, verdict, expected):
    payload = _summaries_payload({"showResultsInDashboard": True})
    finding = payload["project"][0]["reports"][0]["findings"][0]
    finding["sourceFindingId"] = source_finding_id
    finding["retestVerdict"] = verdict
    stub.routes = {"FindingSummaries": payload}
    first = call(client, "get", "/getFindingSummaries?projectIds=1").get_json()["projects"][0]["findings"][0]
    assert (first["sourceFindingId"], first["retestVerdict"]) == expected


def test_get_finding_summaries_selects_only_the_retest_paths(client, stub):
    stub.routes = {"FindingSummaries": _summaries_payload({"showResultsInDashboard": True})}
    call(client, "get", "/getFindingSummaries?projectIds=1")
    query = stub.queries_matching("FindingSummaries")[0]["query"]
    assert 'sourceFindingId: extraFields(path: "source_finding_id")' in query
    assert 'retestVerdict: extraFields(path: "retest_verdict")' in query


def test_get_finding_summaries_selects_no_heavy_fields(client, stub):
    stub.routes = {"FindingSummaries": _summaries_payload({"showResultsInDashboard": True})}
    call(client, "get", "/getFindingSummaries?projectIds=1")
    query = stub.queries_matching("FindingSummaries")[0]["query"]
    for field in ("evidence", "mitigation", "replication_steps", "guidance", "impact", "affectedEntities"):
        assert field not in query
    assert stub.queries_matching("downloadEvidence") == []


def test_get_finding_summaries_cuts_long_descriptions(client, stub):
    payload = _summaries_payload({"showResultsInDashboard": True})
    payload["project"][0]["reports"][0]["findings"][0]["description"] = "<p>" + "word " * 200 + "</p>"
    stub.routes = {"FindingSummaries": payload}
    summary = call(client, "get", "/getFindingSummaries?projectIds=1").get_json()["projects"][0]["findings"][0]["summary"]
    assert len(summary) == app_module.FINDING_SUMMARY_CHARS
    assert summary.endswith("…")


@pytest.mark.parametrize("c4, expected", [
    ({"views": []}, True),
    ('{"views": []}', True),
    ({}, False),
    (None, False),
], ids=["object", "json-text", "empty", "unset"])
def test_get_finding_summaries_reports_whether_a_c4_model_exists(client, stub, c4, expected):
    payload = _summaries_payload({"showResultsInDashboard": True})
    payload["project"][0]["reports"][0]["c4Model"] = c4
    stub.routes = {"FindingSummaries": payload}
    project = call(client, "get", "/getFindingSummaries?projectIds=1").get_json()["projects"][0]
    assert project["hasC4Model"] is expected
    assert "c4Model" not in project and "views" not in str(project)


def test_get_finding_summaries_selects_only_the_c4_model_path(client, stub):
    stub.routes = {"FindingSummaries": _summaries_payload({"showResultsInDashboard": True})}
    call(client, "get", "/getFindingSummaries?projectIds=1")
    query = stub.queries_matching("FindingSummaries")[0]["query"]
    assert 'extraFields(path: "c4_model")' in query


@pytest.mark.parametrize("extra_fields", [{"showResultsInDashboard": False}, {}, None, {"showResultsInDashboard": "true"}])
def test_get_finding_summaries_hides_findings_when_flag_not_true(client, stub, extra_fields):
    stub.routes = {"FindingSummaries": _summaries_payload(extra_fields)}
    payload = _summaries_payload(extra_fields)
    payload["project"][0]["reports"][0]["c4Model"] = {"views": []}
    stub.routes = {"FindingSummaries": payload}
    project = call(client, "get", "/getFindingSummaries?projectIds=1").get_json()["projects"][0]
    assert project["showResults"] is False
    assert project["hasC4Model"] is False
    assert project["findings"] == []
    assert project["codename"] == "BLUE"


def test_get_finding_summaries_graphql_error_is_502(client, stub):
    stub.routes = {"FindingSummaries": FakeResponse(status_code=500, text="nope")}
    assert call(client, "get", "/getFindingSummaries?projectIds=1").status_code == 502


# --- /getEvidence -----------------------------------------------------------

def _evidence_routes(project_id=1, extra_fields=None, exists=True):
    owner = {"evidence_by_pk": {
        "id": 77,
        "report": {"projectId": project_id, "project": {"client": {"extraFields": extra_fields}}},
    } if exists else None}
    return {
        "EvidenceOwner": owner,
        "downloadEvidence": {"downloadEvidence": {
            "evidenceId": 77, "filename": "shot.png", "friendlyName": "Shot", "fileBase64": "eA==",
        }},
    }


def test_get_evidence_requires_both_ids(client):
    assert call(client, "get", "/getEvidence?projectId=1").status_code == 400
    assert call(client, "get", "/getEvidence?evidenceId=77").status_code == 400
    assert call(client, "get", "/getEvidence?projectId=1&evidenceId=x").status_code == 400


def test_get_evidence_downloads_one_file(client, stub):
    stub.routes = _evidence_routes(extra_fields={"showResultsInDashboard": True})
    resp = call(client, "get", "/getEvidence?projectId=1&evidenceId=77")
    assert resp.status_code == 200
    assert resp.get_json() == {"evidenceId": 77, "filename": "shot.png", "friendlyName": "Shot", "fileBase64": "eA=="}
    assert stub.variables_for("EvidenceOwner") == {"evidenceId": 77}
    assert len(stub.queries_matching("downloadEvidence")) == 1


@pytest.mark.parametrize("routes", [
    _evidence_routes(project_id=2, extra_fields={"showResultsInDashboard": True}),
    _evidence_routes(extra_fields={"showResultsInDashboard": False}),
    _evidence_routes(extra_fields=None),
    _evidence_routes(exists=False),
], ids=["other-project", "results-hidden", "flag-absent", "missing"])
def test_get_evidence_is_404_without_downloading(client, stub, routes):
    stub.routes = routes
    resp = call(client, "get", "/getEvidence?projectId=1&evidenceId=77")
    assert resp.status_code == 404
    assert stub.queries_matching("downloadEvidence") == []


def test_get_evidence_graphql_error_is_502(client, stub):
    stub.routes = {"EvidenceOwner": FakeResponse(status_code=500, text="nope")}
    assert call(client, "get", "/getEvidence?projectId=1&evidenceId=77").status_code == 502


# --- /createRetest ----------------------------------------------------------

ORIGINAL_PROJECT = {
    "project_by_pk": {
        "id": 1, "codename": "BLUE", "description": "d", "clientId": 9, "projectTypeId": 2,
        "startDate": "2026-01-01", "endDate": "2026-01-31", "timezone": "UTC",
        "reports": [{"id": 5, "title": "Web application test report", "findings": [
            {"id": 31, "title": "SQLi", "description": "orig", "severityId": 1,
             "extraFields": {"mitigation_prompt": "keep", "poc": "keep"}},
            {"id": 32, "title": "XSS", "description": "orig2", "severityId": 2},
        ]}],
    }
}


def _project_with_retest_reports():
    """ORIGINAL_PROJECT plus a "Retest" report (copies) and a legacy "Original Findings" one."""
    project = copy.deepcopy(ORIGINAL_PROJECT)
    project["project_by_pk"]["reports"] += [
        {"id": 6, "title": "Retest", "findings": [
            {"id": 41, "title": "SQLi", "description": "copy", "severityId": 1,
             "extraFields": {"source_finding_id": 11, "source_project_id": 0,
                             "retest_verdict": "still_present", "positive_control": "<p>x</p>",
                             "retest_client_status": "Fixed", "retest_client_note": "<p>old</p>"}},
        ]},
        {"id": 7, "title": "Original Findings", "findings": [
            {"id": 51, "title": "Old copy", "description": "legacy", "severityId": 1},
        ]},
    ]
    return project


def _retest_routes(project=None):
    return {
        "GetProjectForRetest": project if project is not None else ORIGINAL_PROJECT,
        "insert_project_one": {"insert_project_one": {"id": 100}},
        "insert_report_one": {"insert_report_one": {"id": 200}},
        "insert_reportedFinding": lambda variables: {"insert_reportedFinding": {
            "affected_rows": len(variables["objects"]),
            "returning": [{"id": 301 + i} for i in range(len(variables["objects"]))],
        }},
        "uploadEvidence": {"uploadEvidence": {"id": 400}},
    }


def _inserted_findings(stub):
    return stub.variables_for("insert_reportedFinding")["objects"]


def test_create_retest_requires_project_id(client):
    resp = call(client, "post", "/createRetest", {})
    assert resp.status_code == 400


def test_create_retest_rejects_non_integer_project_id(client, stub):
    resp = call(client, "post", "/createRetest", {"projectId": "abc"})
    assert resp.status_code == 400
    assert stub.calls == []


def test_create_retest_missing_project_is_404(client, stub):
    stub.routes = _retest_routes({"project_by_pk": None})
    resp = call(client, "post", "/createRetest", {"projectId": 1})
    assert resp.status_code == 404
    assert "not found" in resp.get_json()["error"]


def test_create_retest_happy_path(client, stub):
    stub.routes = _retest_routes()
    resp = call(client, "post", "/createRetest", {"projectId": 1})
    assert resp.status_code == 200
    assert resp.get_json() == {
        "projectId": 100,
        "reportId": 200,
        "findingsCount": 2,
        "findingIdMapping": {"31": 301, "32": 302},
        "evidenceErrors": [],
        # The pre-traceability names stay for old callers.
        "originalProjectId": 1,
        "newProjectId": 100,
        "newReportId": 200,
        "findingsCopied": 2,
    }


def test_create_retest_names_and_scopes_the_copy(client, stub):
    stub.routes = _retest_routes()
    call(client, "post", "/createRetest", {"projectId": 1})
    new_project = stub.variables_for("insert_project_one")["object"]
    assert new_project["codename"] == "Retest - BLUE"
    assert new_project["clientId"] == 9
    assert (new_project["startDate"], new_project["endDate"]) == ("2026-01-01", "2026-01-31")
    report = stub.variables_for("insert_report_one")["object"]
    assert report == {
        "projectId": 100, "title": "Retest",
        "creation": "2026-01-31", "last_update": "2026-01-31",
        "archived": False, "complete": False, "delivered": False,
        "include_bloodhound_data": False,
    }
    assert app_module.RETEST_REPORT_TITLE == "Retest"
    findings = _inserted_findings(stub)
    assert [f["reportId"] for f in findings] == [200, 200]
    assert "id" not in findings[0], "original primary keys must not be copied"


def test_create_retest_applies_codename_and_date_overrides(client, stub):
    stub.routes = _retest_routes()
    call(client, "post", "/createRetest", {
        "projectId": 1, "codename": "Retest - Web application test - 12 Jan 2027",
        "startDate": "2027-01-12", "endDate": "2027-01-13",
    })
    new_project = stub.variables_for("insert_project_one")["object"]
    assert new_project["codename"] == "Retest - Web application test - 12 Jan 2027"
    assert (new_project["startDate"], new_project["endDate"]) == ("2027-01-12", "2027-01-13")
    report = stub.variables_for("insert_report_one")["object"]
    assert report["creation"] == report["last_update"] == "2027-01-13"


@pytest.mark.parametrize("body", [
    {"codename": ""}, {"startDate": 20270112}, {"endDate": "   "},
], ids=["blank-codename", "numeric-date", "blank-date"])
def test_create_retest_rejects_bad_overrides_before_any_call(client, stub, body):
    resp = call(client, "post", "/createRetest", dict(body, projectId=1))
    assert resp.status_code == 400
    assert stub.calls == []


def test_create_retest_copies_only_the_requested_findings(client, stub):
    stub.routes = _retest_routes()
    resp = call(client, "post", "/createRetest", {"projectId": 1, "findingIds": [32, "32"]})
    assert resp.status_code == 200
    findings = _inserted_findings(stub)
    assert [f["title"] for f in findings] == ["XSS"], "duplicates are copied once"
    payload = resp.get_json()
    assert payload["findingIdMapping"] == {"32": 301}
    assert payload["findingsCount"] == 1


def test_create_retest_keeps_the_requested_order_in_the_mapping(client, stub):
    stub.routes = _retest_routes()
    resp = call(client, "post", "/createRetest", {"projectId": 1, "findingIds": [32, 31]})
    assert [f["title"] for f in _inserted_findings(stub)] == ["XSS", "SQLi"]
    assert resp.get_json()["findingIdMapping"] == {"32": 301, "31": 302}


def test_create_retest_unknown_finding_id_is_400_and_creates_nothing(client, stub):
    stub.routes = _retest_routes(_project_with_retest_reports())
    # 51 is in the legacy "Original Findings" report, 999 does not exist.
    resp = call(client, "post", "/createRetest", {"projectId": 1, "findingIds": [31, 999, 51]})
    assert resp.status_code == 400
    assert resp.get_json() == {"error": "unknown findingIds", "findingIds": [999, 51]}
    assert stub.queries_matching("insert_project_one") == []
    assert stub.queries_matching("insert_report_one") == []
    assert stub.queries_matching("insert_reportedFinding") == []


@pytest.mark.parametrize("finding_ids", [[], "31", [31, "x"], [True], {"a": 1}],
                         ids=["empty", "string", "non-integer", "bool", "object"])
def test_create_retest_rejects_malformed_finding_ids_before_any_call(client, stub, finding_ids):
    resp = call(client, "post", "/createRetest", {"projectId": 1, "findingIds": finding_ids})
    assert resp.status_code == 400
    assert stub.calls == []


def test_create_retest_without_finding_ids_skips_retest_reports(client, stub):
    """By default only the project's own findings are copied, not earlier copies."""
    stub.routes = _retest_routes(_project_with_retest_reports())
    resp = call(client, "post", "/createRetest", {"projectId": 1})
    assert [f["description"] for f in _inserted_findings(stub)] == ["orig", "orig2"]
    assert resp.get_json()["findingIdMapping"] == {"31": 301, "32": 302}
    query = stub.queries_matching("GetProjectForRetest")[0]["query"]
    assert "title" in query, "the report title decides which findings are copies"


def test_create_retest_of_a_retest_copies_a_copy_asked_for_by_id(client, stub):
    stub.routes = _retest_routes(_project_with_retest_reports())
    resp = call(client, "post", "/createRetest", {"projectId": 1, "findingIds": [41]})
    assert resp.status_code == 200
    (copy_,) = _inserted_findings(stub)
    # The new copy points at the copy it was made from, and starts without a verdict.
    assert copy_["extraFields"] == {
        "source_finding_id": 41, "source_project_id": 1,
        "retest_client_status": "", "retest_client_note": "",
        "retest_verdict": "", "positive_control": "",
    }
    assert resp.get_json()["findingIdMapping"] == {"41": 301}


def test_create_retest_sets_and_resets_the_extra_fields(client, stub):
    project = copy.deepcopy(ORIGINAL_PROJECT)
    project["project_by_pk"]["reports"][0]["findings"][0]["extraFields"].update({
        "retest_verdict": "fixed", "positive_control": "<p>old</p>", "source_finding_id": 7,
    })
    stub.routes = _retest_routes(project)
    call(client, "post", "/createRetest", {
        "projectId": 1,
        "comments": [{"findingId": 31, "status": "Fixed", "description": "Patched in 2.4 <b>&</b> deployed\nyesterday"}],
    })
    first, second = _inserted_findings(stub)
    assert first["extraFields"] == {
        "mitigation_prompt": "keep", "poc": "keep",  # the original's extra fields are kept
        "source_finding_id": 31,
        "source_project_id": 1,
        "retest_client_status": "Fixed",
        "retest_client_note": "<p>Patched in 2.4 &lt;b&gt;&amp;&lt;/b&gt; deployed<br>yesterday</p>",
        "retest_verdict": "",
        "positive_control": "",
    }
    assert second["extraFields"] == {
        "source_finding_id": 32, "source_project_id": 1,
        "retest_client_status": "", "retest_client_note": "",
        "retest_verdict": "", "positive_control": "",
    }
    # The source finding is not changed by building the copy.
    assert project["project_by_pk"]["reports"][0]["findings"][0]["extraFields"]["retest_verdict"] == "fixed"


def test_create_retest_leaves_title_and_description_unchanged(client, stub):
    """The customer's comment goes to extra fields; no title prefix, no appended text."""
    stub.routes = _retest_routes()
    call(client, "post", "/createRetest", {
        "projectId": 1,
        "comments": [{"findingId": 31, "status": "RESOLVED", "description": "<p>fixed</p>"}],
    })
    first, second = _inserted_findings(stub)
    assert (first["title"], first["description"]) == ("SQLi", "orig")
    assert (second["title"], second["description"]) == ("XSS", "orig2")
    assert first["extraFields"]["retest_client_status"] == "RESOLVED"
    assert first["extraFields"]["retest_client_note"] == "<p>&lt;p&gt;fixed&lt;/p&gt;</p>"


def test_create_retest_joins_several_comments_on_one_finding(client, stub):
    stub.routes = _retest_routes()
    call(client, "post", "/createRetest", {
        "projectId": 1,
        "comments": [
            {"findingId": 31, "status": "Mitigated", "description": "first"},
            {"findingId": "31", "status": "Fixed", "description": "second"},
            {"findingId": 31, "status": "", "description": ""},
        ],
    })
    extra = _inserted_findings(stub)[0]["extraFields"]
    assert extra["retest_client_status"] == "Fixed", "the last status given wins"
    assert extra["retest_client_note"] == "<p>first</p><p>second</p>"


def test_create_retest_uploads_evidence_to_the_report(client, stub):
    """
    Regression guard for 145b5529: uploadEvidence takes ``report``, not ``finding``.
    """
    stub.routes = _retest_routes()
    call(client, "post", "/createRetest", {
        "projectId": 1,
        "comments": [{"findingId": 31, "evidence_b64": "eA==", "evidence_filename": "a.png"}],
    })
    variables = stub.variables_for("uploadEvidence")
    assert variables["report"] == 200
    assert "finding" not in variables
    assert variables["filename"] == "a.png"


def test_create_retest_reports_unmatched_evidence(client, stub):
    stub.routes = _retest_routes()
    resp = call(client, "post", "/createRetest", {
        "projectId": 1,
        "comments": [{"findingId": 999, "evidence_b64": "eA=="}],
    })
    assert resp.status_code == 200
    assert resp.get_json()["evidenceErrors"][0]["originalFindingId"] == 999
    assert stub.queries_matching("uploadEvidence") == []


def test_create_retest_reports_evidence_for_a_finding_left_out(client, stub):
    """Evidence on a comment for a finding not in findingIds has no copy to go with."""
    stub.routes = _retest_routes()
    resp = call(client, "post", "/createRetest", {
        "projectId": 1, "findingIds": [32],
        "comments": [{"findingId": 31, "evidence_b64": "eA=="}],
    })
    assert resp.status_code == 200
    assert resp.get_json()["evidenceErrors"] == [
        {"originalFindingId": 31, "error": "no matching new finding found"},
    ]
    assert stub.queries_matching("uploadEvidence") == []


def test_create_retest_with_no_findings_skips_insert(client, stub):
    stub.routes = _retest_routes({"project_by_pk": dict(ORIGINAL_PROJECT["project_by_pk"], reports=[])})
    resp = call(client, "post", "/createRetest", {"projectId": 1})
    assert resp.status_code == 200
    assert resp.get_json()["findingsCopied"] == 0
    assert resp.get_json()["findingsCount"] == 0
    assert resp.get_json()["findingIdMapping"] == {}
    assert stub.queries_matching("insert_reportedFinding") == []


# --- /getStatistics ---------------------------------------------------------

def test_get_statistics_requires_client_id(client):
    resp = call(client, "get", "/getStatistics")
    assert resp.status_code == 400


def test_get_statistics_aggregates_totals(client, stub):
    stub.routes = {
        "ClientSeverityTotals": {"findingSeverity": [
            {"id": 1, "severity": "High", "reportedFindings_aggregate": {"aggregate": {"count": 3}}},
            {"id": 2, "severity": "Low", "reportedFindings_aggregate": {"aggregate": {"count": 1}}},
        ]},
        "ProjectsFindingsByClient": {"project": [{
            "id": 1, "codename": "BLUE",
            "reports": [{"findings": [
                {"severityId": 1, "severity": {"id": 1, "severity": "High"}},
                {"severityId": 1, "severity": {"id": 1, "severity": "High"}},
                {"severityId": 2, "severity": {"id": 2, "severity": "Low"}},
            ]}],
        }]},
    }
    resp = call(client, "get", "/getStatistics?clientId=1")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["clientTotals"] == [
        {"severityId": 1, "severity": "High", "count": 3},
        {"severityId": 2, "severity": "Low", "count": 1},
    ]
    assert payload["projects"] == [{
        "id": 1, "codename": "BLUE",
        "severityTotals": [
            {"severityId": 1, "severity": "High", "count": 2},
            {"severityId": 2, "severity": "Low", "count": 1},
        ],
    }]


def test_get_statistics_filters_on_dashboard_flag(client, stub):
    """Both statistics queries must be scoped by the client's visibility flag."""
    stub.routes = {
        "ClientSeverityTotals": {"findingSeverity": []},
        "ProjectsFindingsByClient": {"project": []},
    }
    call(client, "get", "/getStatistics?clientId=1")
    for needle in ("ClientSeverityTotals", "ProjectsFindingsByClient"):
        query = stub.queries_matching(needle)[0]["query"]
        assert "showResultsInDashboard: true" in query


# --- /generateReport --------------------------------------------------------

@pytest.fixture
def fake_soffice(monkeypatch):
    """Replace the LibreOffice call with one that writes a stub PDF."""
    class Result:
        returncode = 0
        stderr = b""

    def fake_run(args, **kwargs):
        outdir = args[args.index("--outdir") + 1]
        with open(os.path.join(outdir, "report.pdf"), "wb") as fh:
            fh.write(b"%PDF-1.4 stub")
        return Result()

    monkeypatch.setattr(app_module.subprocess, "run", fake_run)


@pytest.mark.parametrize("missing", ["projectId", "templateId"])
def test_generate_report_requires_fields(client, missing):
    body = {"projectId": 1, "templateId": 2}
    del body[missing]
    resp = call(client, "post", "/generateReport", body)
    assert resp.status_code == 400
    assert resp.get_json() == {"error": f"{missing} is required"}


def test_generate_report_rejects_non_integer_ids(client):
    resp = call(client, "post", "/generateReport", {"projectId": "abc", "templateId": 2})
    assert resp.status_code == 400


def test_generate_report_with_no_reports_returns_empty(client, stub):
    stub.routes = {"GetReports": {"report": []}}
    resp = call(client, "post", "/generateReport", {"projectId": 1, "templateId": 2})
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_generate_report_converts_docx_to_pdf(client, stub, fake_soffice):
    stub.routes = {
        "GetReports": {"report": [{"id": 5, "title": "Report A"}]},
        "generateDocReport": {"generateDocReport": {
            "docBase64": base64.b64encode(b"docx bytes").decode(),
            "fileName": "Report A.docx",
        }},
    }
    resp = call(client, "post", "/generateReport", {"projectId": 1, "templateId": 2})
    assert resp.status_code == 200
    results = resp.get_json()
    assert len(results) == 1
    assert results[0]["fileName"] == "Report A.pdf"
    assert base64.b64decode(results[0]["fileBase64"]) == b"%PDF-1.4 stub"


def test_generate_report_excludes_original_findings_report(client, stub, fake_soffice):
    stub.routes = {
        "GetReports": {"report": []},
    }
    call(client, "post", "/generateReport", {"projectId": 1, "templateId": 2})
    query = stub.queries_matching("GetReports")[0]["query"]
    assert '_neq: "Original Findings"' in query


def test_generate_report_uses_long_timeout(client, stub, fake_soffice):
    """Doc generation runs for minutes; it must not use the default query timeout."""
    stub.routes = {
        "GetReports": {"report": [{"id": 5, "title": "R"}]},
        "generateDocReport": {"generateDocReport": {
            "docBase64": base64.b64encode(b"x").decode(), "fileName": "R.docx",
        }},
    }
    call(client, "post", "/generateReport", {"projectId": 1, "templateId": 2})
    doc_call = stub.queries_matching("generateDocReport")[0]
    list_call = stub.queries_matching("GetReports")[0]
    assert doc_call["timeout"] > list_call["timeout"]


def test_generate_report_reports_per_report_errors(client, stub):
    stub.routes = {
        "GetReports": {"report": [{"id": 5, "title": "R"}]},
        "generateDocReport": FakeResponse(payload={"errors": [{"message": "template missing"}]}),
    }
    resp = call(client, "post", "/generateReport", {"projectId": 1, "templateId": 2})
    assert resp.status_code == 200
    assert resp.get_json()[0]["reportId"] == 5
    assert resp.get_json()[0]["error"]["message"] == "graphql error"


def test_generate_report_reports_conversion_failure(client, stub, monkeypatch):
    def boom(args, **kwargs):
        raise RuntimeError("soffice exploded")

    monkeypatch.setattr(app_module.subprocess, "run", boom)
    stub.routes = {
        "GetReports": {"report": [{"id": 5, "title": "R"}]},
        "generateDocReport": {"generateDocReport": {
            "docBase64": base64.b64encode(b"x").decode(), "fileName": "R.docx",
        }},
    }
    resp = call(client, "post", "/generateReport", {"projectId": 1, "templateId": 2})
    assert resp.status_code == 200
    assert resp.get_json()[0]["error"]["message"] == "PDF conversion failed"


# --- graphql_request transport ---------------------------------------------

def test_graphql_request_surfaces_transport_failure(client, monkeypatch):
    def boom(self, url, json=None, timeout=None, **kw):
        raise app_module.requests.RequestException("connection refused")

    monkeypatch.setattr(app_module.requests.Session, "post", boom)
    resp = call(client, "get", "/getClient?clientId=1")
    assert resp.status_code == 502
    assert resp.get_json()["error"]["message"] == "graphql request failed"


def test_graphql_request_surfaces_non_json_body(client, monkeypatch):
    monkeypatch.setattr(
        app_module.requests.Session, "post",
        lambda self, url, json=None, timeout=None, **kw: FakeResponse(status_code=200, payload=None),
    )
    resp = call(client, "get", "/getClient?clientId=1")
    assert resp.status_code == 502
    assert "decode failed" in resp.get_json()["error"]["message"]


# ---- Client-facing details and facts: an allow-list -----------------------------------------

def test_get_project_details_keep_only_allow_listed_keys_and_scalars(client, stub):
    stub.routes = _project_with_extra_fields({"showResultsInDashboard": True}, {}, {
        "confidence": " Confirmed ",
        "occurrences": [
            {"operation": "GET /a", "request": "raw/x.txt", "absorbed_id": "F-9", "nested": {"a": 1}},
            {"request": "raw/only-internal.txt"},  # nothing client-facing left: dropped
            "not an object",
        ],
        "attack_chain": '[{"step": 1, "claim": "c", "basis": "measured", "evidence": "raw/y.txt"}]',
        "wstg": ["WSTG-INPV-05", "", None, 4],
        "prior_identifiers": "NN-1",  # not a list
        "oracle_result": "pass", "quantity_provenance": "<p>wc -l</p>",
        "evidence_contains_live_secrets": "yes",  # only a real true counts
    })
    finding = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]["findings"][0]
    d = finding["details"]
    assert d["confidence"] == "Confirmed"
    assert d["occurrences"] == [{"operation": "GET /a"}]
    assert d["attackChain"] == [{"step": 1, "claim": "c", "basis": "measured"}]
    assert d["classification"]["wstg"] == ["WSTG-INPV-05", "4"]
    assert d["priorIdentifiers"] == [] and d["evidenceContainsLiveSecrets"] is False
    body = json.dumps(finding)
    assert "raw/" not in body and "oracle_result" not in body and "\"pass\"" not in body and "wc -l" not in body


def test_get_project_details_default_when_unset(client, stub):
    stub.routes = _project_with_extra_fields({"showResultsInDashboard": True}, None, None)
    report = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]
    d = report["findings"][0]["details"]
    assert d["confidence"] == "" and d["occurrences"] == [] and d["classification"]["cweId"] == ""
    assert report["facts"]["testDataLog"] == [] and report["facts"]["version"] == ""


def test_get_project_report_facts_drop_internal_fields(client, stub):
    stub.routes = _project_with_extra_fields({"showResultsInDashboard": True}, {
        "version": "1.2", "qa_signoff": "A. Reviewer", "client_feedback": [{"item": "x"}],
        "test_data_log": [{"object": "doc-1", "state": "present", "session": "testing", "account": "a"}],
        "limitations": ["brute force"],
    }, {})
    report = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]["reports"][0]
    assert report["facts"]["version"] == "1.2"
    assert report["facts"]["testDataLog"] == [{"object": "doc-1", "state": "present", "account": "a"}]
    assert report["facts"]["limitations"] == ["brute force"]
    assert "A. Reviewer" not in json.dumps(report) and "client_feedback" not in json.dumps(report)


def test_get_project_drops_project_and_client_extra_fields(client, stub):
    payload = _project_payload({"showResultsInDashboard": True, "internal": "x"})
    payload["project_by_pk"]["extraFields"] = {"test_accounts": [{"password": "s3cret"}]}
    stub.routes = {"GetProject": payload, "downloadEvidence": _get_project_routes({})["downloadEvidence"]}
    project = call(client, "get", "/getProject?projectId=1").get_json()["project_by_pk"]
    assert "extraFields" not in project and "extraFields" not in project["client"]
    assert project["reports"][0]["findings"], "the results flag is still read before it is dropped"


def test_get_finding_summaries_returns_confidence(client, stub):
    payload = _summaries_payload({"showResultsInDashboard": True})
    payload["project"][0]["reports"][0]["findings"][0]["confidence"] = "Tentative"
    stub.routes = {"FindingSummaries": payload}
    first = call(client, "get", "/getFindingSummaries?projectIds=1").get_json()["projects"][0]["findings"][0]
    assert first["confidence"] == "Tentative"
    query = stub.queries_matching("FindingSummaries")[0]["query"]
    assert 'confidence: extraFields(path: "confidence")' in query
