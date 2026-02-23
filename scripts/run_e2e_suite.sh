#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

docker compose -p nosql_reader -f "$REPO_ROOT/docker-compose.yml" exec -T app python /app/scripts/e2e_cli_suite.py
