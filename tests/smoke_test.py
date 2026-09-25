"""
smoke_test.py - end-to-end check with NO ML dependencies (keyword engine, seed taxonomy, System 2 off).
Covers: pages, single flow, idempotent submit, batch with a bad file, retry, taxonomy versioning + hand-edit
detection, reprocess, restart-resume from checkpoints, System-2 fallback chain, suppliers and the error contract.

Run:  python -m tests.smoke_test      (or: pytest tests)
"""
import json
import os
import tempfile
import time

TMP = tempfile.mkdtemp(prefix="openjev_test_")
os.environ.update(OPENJEV_DATA_DIR=TMP, OPENJEV_CLASSIFIER_BACKEND="keyword", OPENJEV_SYSTEM2_ENABLED="false",
                  OPENJEV_COMMIT_EVERY_ROWS="3", OPENJEV_POLL_INTERVAL_MS="500")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.database import get_sf  # noqa: E402
from app.core.errors import TaxonomyError  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models.entities import FileRun, Job, RowResult  # noqa: E402
from app.services import taxonomy_service  # noqa: E402

GOOD = """Date,Narration,Amount
2024-01-05,AWS cloud hosting charges for Q1,"12,500.00"
2024-01-09,Annual maintenance contract renewal for core banking,"48,000.00"
2024-01-11,Oracle database license renewal,"30,250.50"
2024-01-15,Network security firewall subscription,"7,800.00"
2024-01-20,Consulting services for implementation project,"22,000.00"
2024-01-21,Consulting services for implementation project,"22,000.00"
2024-01-25,Staff training workshop on new platform,"3,100.00"
2024-02-01,Zzz qqq xyzzy,"120.00"
2024-02-03,,"99.00"
2024-02-05,Helpdesk support services monthly,"5,400.00"
"""
BAD = "Code,Value\n1,2\n3,4\n"
failures: list[str] = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def wait_job(c, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        d = c.get(f"/api/jobs/{job_id}/status").json()
        if d["finished"]:
            return d
        time.sleep(0.2)
    raise TimeoutError(f"job {job_id} did not finish")


def main():
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as c:
        print("pages")
        for p in ("/", "/single", "/batch", "/jobs", "/files", "/taxonomy", "/suppliers", "/diagnostics"):
            r = c.get(p)
            check(r.status_code == 200 and settings.app_name in r.text, f"GET {p} -> 200")
        check(c.get("/health").json()["status"] in ("ok", "warn"), "health endpoint")

        print("single-file flow")
        up = c.post("/api/single/upload", files={"file": ("ledger.csv", GOOD, "text/csv")}).json()
        g = up["guesses"]["erp_costs"]
        check(up["rows"] == 10 and g["text"] == "Narration" and g["amount"] == "Amount", f"upload preview + column guess {g}")
        body = dict(upload_id=up["upload_id"], profile="erp_costs", narration_column="Narration", amount_column="Amount",
                    supplier=dict(name="Acme Systems Ltd", code="AC-1", country="IN", contact_email="ops@acme.test"),
                    client_token="tok-1")
        r = c.post("/api/single/submit", json={**body, "supplier": {"name": ""}, "client_token": "tok-0"})
        check(r.status_code == 400 and "Supplier" in r.json()["error"]["message"], "supplier required in single mode")
        sub = c.post("/api/single/submit", json=body).json()
        check(c.post("/api/single/submit", json=body).json()["job_id"] == sub["job_id"], "idempotent submit (same token -> same job)")
        st = wait_job(c, sub["job_id"])
        f = st["files"][0]
        check(st["status"] == "completed" and f["processed_rows"] == 10, f"job completed, 10 rows ({st['status']})")
        check(f["error_rows"] == 1, "empty narration flagged as a row error, file still completes")
        page = c.get(f"/files/{sub['file_id']}")
        check(page.status_code == 200 and "Download JSON" in page.text and "Acme Systems Ltd" in page.text, "results page renders")
        check(c.get(f"/files/{sub['file_id']}?review=1&page=1").status_code == 200, "filtered rows page")
        out = json.loads(c.get(f"/files/{sub['file_id']}/download.json").text)
        check(len(out["results"]) == 10 and out["supplier"]["name"] == "Acme Systems Ltd" and out["file"]["taxonomy_version"] == 1,
              "JSON output: 10 rows + supplier + taxonomy version")
        check(out["summary"]["total_amount"] and abs(out["summary"]["total_amount"] - 151170.5) < 1, f"spend parsed ({out['summary']['total_amount']})")
        cats = {m["category"] for m in out["summary"]["by_category"]}
        check("Software Licensing" in cats and "Infrastructure & Hosting" in cats, f"sensible categories {sorted(cats)}")
        check(c.get(f"/files/{sub['file_id']}/download.csv").text.startswith("\ufeffrow,Date,Narration,Amount,main_category"), "CSV output")

        print("batch flow (one good, one unusable file)")
        (settings.inbox_dir / "good.csv").write_text(GOOD, encoding="utf-8")
        (settings.inbox_dir / "bad.csv").write_text(BAD, encoding="utf-8")
        sc = c.post("/api/batch/scan", json={"folder": "", "profile": "erp_costs"}).json()
        names = {x["name"]: x for x in sc["files"]}
        check(names["good.csv"]["text_column"] == "Narration" and names["bad.csv"]["text_column"] is None, "scan detects text columns")
        check(c.post("/api/batch/scan", json={"folder": "/etc"}).status_code == 400, "folder outside allowed roots rejected")
        bj = c.post("/api/batch/submit", json=dict(folder="", profile="erp_costs", client_token="b-1",
                    items=[{"name": "good.csv", "supplier": "Beta Corp"}, {"name": "bad.csv"}])).json()
        bs = wait_job(c, bj["job_id"])
        by = {x["filename"]: x for x in bs["files"]}
        check(bs["status"] == "completed_with_errors", f"batch status {bs['status']}")
        check(by["good.csv"]["status"] == "completed" and by["bad.csv"]["status"] == "failed", "good file done, bad file failed alone")
        check("text" in by["bad.csv"]["error_message"] and by["bad.csv"]["error_hint"] and by["bad.csv"]["error_trace"],
              "failure carries message + hint + trace")
        check(c.post(f"/api/jobs/{bj['job_id']}/retry").status_code == 200, "retry accepted")
        wait_job(c, bj["job_id"])
        check(c.post(f"/api/jobs/{bj['job_id']}/cancel").status_code == 409, "cancel on a finished job -> 409")
        check("Beta Corp" in c.get("/suppliers").text, "batch supplier persisted")
        rescan = {x["name"]: x for x in c.post("/api/batch/scan", json={"folder": "", "profile": "erp_costs"}).json()["files"]}
        check(rescan["good.csv"]["previous"] is not None, "re-scan flags 'processed before'")

        print("taxonomy")
        cur = c.get("/taxonomy?profile=erp_costs")
        check(cur.status_code == 200 and "Version 1" in cur.text, "seed taxonomy registered as v1")
        tax = json.loads((settings.taxonomy_dir / "erp_costs.json").read_text())
        tax["Software Licensing"].append("Robotic Process Automation")
        s = c.post("/api/taxonomy/save", json={"profile": "erp_costs", "taxonomy": tax, "note": "test"}).json()
        check(s["version"] == 2 and "Robotic Process Automation" in s["taxonomy"]["Software Licensing"], "save creates v2 with my addition")
        bad = c.post("/api/taxonomy/save", json={"profile": "erp_costs", "taxonomy": {"A": ["x"], "a": ["y"]}})
        check(bad.status_code == 422 and "twice" in bad.json()["error"]["message"], "duplicate category rejected")
        many = {f"Cat{i}": ["s"] for i in range(9)}
        w = c.post("/api/taxonomy/save", json={"profile": "generic", "taxonomy": many}).json()
        check(len(w["warnings"]) >= 1, "over-cap taxonomy saves with a warning")
        tax["Software Licensing"].append("Hand Edited Sub")
        (settings.taxonomy_dir / "erp_costs.json").write_text(json.dumps(tax), encoding="utf-8")
        check("Version 3" in c.get("/taxonomy?profile=erp_costs").text, "hand-edit of the file detected as v3")
        rid = c.get("/taxonomy?profile=erp_costs").text.split('data-restore="')[-1].split('"')[0]
        check(c.post("/api/taxonomy/restore", json={"profile": "erp_costs", "version_id": int(rid)}).json()["version"] == 4, "restore -> new version")
        rp = c.post(f"/api/files/{sub['file_id']}/reprocess").json()
        rs = wait_job(c, rp["job_id"])
        with get_sf()() as db:
            check(rs["status"] == "completed" and db.get(FileRun, rp["file_id"]).taxonomy_version == 4, "reprocess uses the current taxonomy (v4)")

        print("resume after restart (checkpointed rows are kept, not redone)")
        with get_sf()() as db:
            p = settings.inbox_dir / "good.csv"
            job = Job(kind="batch", profile="erp_costs", backend="keyword", status="running")
            fr = FileRun(job=job, filename="good.csv", source_path=str(p), file_size=p.stat().st_size, file_mtime=p.stat().st_mtime,
                         profile="erp_costs", backend="keyword", status="running", narration_column="Narration",
                         taxonomy_snapshot=json.dumps({"Software Licensing": ["Perpetual License"], "Others": []}), taxonomy_version=1)
            db.add(job)
            db.flush()
            for i in range(3):
                db.add(RowResult(file_run_id=fr.id, row_index=i, narration="PRE", main_category="Software Licensing",
                                 sub_category="Perpetual License", main_confidence=1.0, sub_confidence=1.0))
            db.commit()
            jid, fid = job.id, fr.id
        app.state.jobs._recover()
        rs = wait_job(c, jid)
        with get_sf()() as db:
            rows = db.query(RowResult).filter_by(file_run_id=fid).order_by(RowResult.row_index).all()
            check(rs["status"] == "completed" and len(rows) == 10, f"resumed file has all 10 rows once ({len(rows)})")
            check([r.narration for r in rows[:3]] == ["PRE"] * 3, "pre-existing checkpoint rows were skipped, not recomputed")
            check({r.main_category for r in rows[3:]} <= {"Software Licensing", "Others", "Unclassified"}, "resumed run kept its taxonomy snapshot")

        print("System-2 fallback chain")
        broken = settings.model_copy(update={"system2_enabled": True, "ollama_base_url": "http://127.0.0.1:9", "system2_timeout_sec": 2})
        with get_sf()() as db:
            tax_seed, _ = taxonomy_service.bootstrap(db, broken, "erp_costs", ["some text"])
            check("Software Licensing" in tax_seed, "Ollama down -> falls back to the seed taxonomy")
            try:
                taxonomy_service.bootstrap(db, broken, "support_tickets", ["printer broken"])
                check(False, "no seed + Ollama down should raise")
            except TaxonomyError as e:
                check(bool(e.hint) and "Cannot reach Ollama" in (e.detail or ""), "no seed + Ollama down -> actionable error")

        print("suppliers + error contract")
        r = c.post("/suppliers/save", data={"name": "Gamma Ltd", "contact_email": "not-an-email"}, follow_redirects=False)
        check(r.status_code == 303 and "err=" in r.headers["location"], "bad email bounces back with a message")
        r = c.post("/suppliers/save", data={"name": "Gamma Ltd", "country": "UK"}, follow_redirects=False)
        check(r.status_code == 303 and "Gamma" in c.get("/suppliers").text, "supplier created")
        r = c.post("/api/single/upload", files={"file": ("virus.exe", b"MZ", "application/octet-stream")})
        e = r.json()["error"]
        check(r.status_code == 400 and e["hint"] and e["request_id"], f"unsupported file -> JSON error with hint + reference ({e['message']})")
        r = c.get("/files/9999")
        check(r.status_code == 404 and "Reference" in r.text, "HTML 404 shows a reference id")
        check(c.get("/api/jobs/9999/status").json()["error"]["code"] == "not_found", "API 404 uses the same contract")
        check(c.get("/api/jobs/active").json()["count"] == 0, "no jobs left running")
        check("Diagnostics" in c.get("/diagnostics").text and "file_run:" in c.get("/diagnostics").text, "failed file recorded in Diagnostics")

    print("\nFAILED: " + "; ".join(failures) if failures else "\nALL CHECKS PASSED")
    return not failures


def test_smoke():
    assert main()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
