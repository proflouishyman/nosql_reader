# Tier 0 Branch Integration Postmortem (2026-02-22)

## Summary
The service was not failing due to model/runtime instability. It was failing due to partial branch integration: Tier 0 call paths were merged, but required supporting files/config and one tiered-agent bug fix were missing in this branch.

After applying the missing pieces, the full CLI matrix passed.

## What Went Wrong

### 1) Missing module import
- Symptom:
  - `POST /api/rag/explore_corpus` returned `500`.
  - Stack trace: `ModuleNotFoundError: No module named 'historian_agent.evidence_cluster'`.
- Cause:
  - `recursive_synthesis.py` imported `EvidenceClusterBuilder`, but `app/historian_agent/evidence_cluster.py` did not exist in this branch.

### 2) Missing Tier 0 config object
- Symptom:
  - After adding the module, `POST /api/rag/explore_corpus` still returned `500`.
  - Stack trace: `AttributeError: 'AppConfig' object has no attribute 'tier0'`.
- Cause:
  - This branch had an older `app/config.py` without `Tier0Config` and without `APP_CONFIG.tier0`.
  - Tier 0 components (`stratification.py`, `corpus_explorer.py`) require `APP_CONFIG.tier0`.

### 3) Tiered endpoint regression
- Symptom:
  - `POST /historian-agent/query-tiered` returned `500`.
  - Error body: `local variable 'docs' referenced before assignment`.
- Cause:
  - In `iterative_adversarial_agent.py`, the Tier 1 return path referenced `docs` that was never defined in that scope.
  - The same block also duplicated `sources_list.append(...)` logic.

## Why This Happened
- The branch contained a partial port from another working tree:
  - New Tier 0 synthesis code paths were present.
  - Supporting module + config and one route-path bug fix were not fully carried over.
- Result: runtime import/config errors appeared only when specific endpoints were exercised.

## Fixes Applied

### Added missing module
- Added:
  - `app/historian_agent/evidence_cluster.py`

### Synced Tier 0-capable config
- Replaced with Tier 0-aware config:
  - `app/config.py`
- Verified it now includes:
  - `Tier0Config` dataclass
  - `tier0` field on `AppConfig`
  - `APP_CONFIG.tier0` loading from env

### Patched tiered agent return path
- Updated:
  - `app/historian_agent/iterative_adversarial_agent.py`
- Changes:
  - Removed undefined `docs` usage.
  - Normalized sources from `tier1_metrics["sources"]`.
  - Removed duplicate source appends.

## Verification

After restart, the full CLI smoke matrix passed:

- `GET /` -> `200`
- `POST /historian-agent/query-basic` -> `200`
- `POST /historian-agent/query-adversarial` -> `200`
- `POST /historian-agent/query-tiered` -> `200`
- `POST /api/rag/explore_corpus` -> `200`
- `GET /api/rag/exploration_report` -> `200`

## Preventive Guardrails

1. Add a startup self-check that imports Tier 0 modules and asserts `hasattr(APP_CONFIG, "tier0")`.
2. Add a CI/API smoke test that hits all six endpoints above on each PR.
3. Keep Tier 0 integration changes atomic:
   - `recursive_synthesis.py` + `evidence_cluster.py` + `config.py` + route tests in one PR.
4. Add a lint rule/test for undefined locals in endpoint return paths (caught the `docs` issue).
