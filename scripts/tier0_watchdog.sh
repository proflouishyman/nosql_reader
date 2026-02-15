#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/louishyman/coding/nosql/nosql_reader"
LOG="$ROOT/app/logs/tier0_watchdog.log"
TIER0_DIR="$ROOT/app/logs/tier0"
NOW=$(date '+%Y-%m-%d %H:%M:%S')

LAST_LOG=$(ls -t "$TIER0_DIR"/tier0_*.log 2>/dev/null | head -n1 || true)
if [[ -z "${LAST_LOG}" ]]; then
  echo "[$NOW] status=no_log action=none" >> "$LOG"
  exit 0
fi

# macOS stat
LAST_MTIME=$(stat -f %m "$LAST_LOG")
NOW_EPOCH=$(date +%s)
AGE=$((NOW_EPOCH - LAST_MTIME))

STATUS="running"
ACTION="none"
if (( AGE > 900 )); then
  STATUS="stalled"
  ACTION="restart"
  echo "[$NOW] status=$STATUS action=$ACTION last_log=$(basename "$LAST_LOG") age_s=$AGE" >> "$LOG"

  docker compose -f "$ROOT/docker-compose.yml" exec -T app sh -lc "PYTHONPATH=/app python - <<'PY'
import json
from pathlib import Path
from main import app

client = app.test_client()
payload = {
    'strategy': 'balanced',
    'total_budget': 500,
    'save_notebook': True,
}
resp = client.post('/api/rag/explore_corpus', json=payload)
print('status', resp.status_code)
if resp.status_code != 200:
    print(resp.data[:1000])
    raise SystemExit(1)
report = resp.get_json()
output_path = Path('/app/logs/tier0_run_500_report_inductive.json')
output_path.write_text(json.dumps(report, indent=2))
print('saved', output_path)
print('questions', len(report.get('questions') or []))
print('patterns', len(report.get('patterns') or []))
print('contradictions', len(report.get('contradictions') or []))
print('entities', len(report.get('entities') or []))
print('notebook_path', report.get('notebook_path'))
PY" >> "$LOG" 2>&1
else
  echo "[$NOW] status=$STATUS action=$ACTION last_log=$(basename "$LAST_LOG") age_s=$AGE" >> "$LOG"
fi
