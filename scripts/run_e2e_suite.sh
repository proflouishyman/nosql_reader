#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Use the current canonical in-container utility path under app/util.
docker compose -p nosql_reader -f "$REPO_ROOT/docker-compose.yml" exec -T app python /app/util/e2e_cli_suite.py
