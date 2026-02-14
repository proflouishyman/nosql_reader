import json
from pathlib import Path
from main import app

client = app.test_client()
payload = {"strategy": "balanced", "total_budget": 5000}
resp = client.post("/api/rag/explore_corpus", json=payload)
print("explore status", resp.status_code)
if resp.status_code != 200:
    print(resp.data[:2000])
    raise SystemExit(1)
report = resp.get_json()
output_path = Path("/app/logs/tier0_run_5000_report_latest.json")
output_path.write_text(json.dumps(report, indent=2))
print("saved", output_path)
print("questions", len(report.get("questions") or []))
print("patterns", len(report.get("patterns") or []))
print("contradictions", len(report.get("contradictions") or []))
print("entities", len(report.get("entities") or []))
print("group_indicators", len(report.get("group_indicators") or []))
print("synthesis", bool(report.get("question_synthesis")))
