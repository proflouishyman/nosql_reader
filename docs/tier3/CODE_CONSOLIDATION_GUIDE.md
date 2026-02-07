# Tier 0 Code Consolidation Guide

Tier 0 reuses shared infrastructure from the main codebase:
- `APP_CONFIG` for configuration (single source of truth)
- `LLMClient` (provider‑agnostic calls)
- `DocumentStore` (Mongo access)

## Consolidated Utilities
`app/historian_agent/tier0_utils.py` provides:
- `parse_llm_json()` (shared JSON cleaning)
- `save_with_timestamp()` (notebook persistence)
- `Tier0Logger` (debug log files)

## Configuration Source
All Tier 0 configuration lives in `.env` and is loaded once in `config.py` as `APP_CONFIG.tier0`.

See `docs/tier3/QUICKSTART.md` for the full key list.
