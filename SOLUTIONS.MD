# Solutions Log

## 2026-02-23 - Cleanup: Move dev/test scripts into `app/util` and setup script into `app/setup`

### Problem
Repository cleanup required relocating standalone and test-oriented scripts out of mixed directories so layout better reflects runtime vs tooling code.

### Solution
Moved files into target folders from the cleanup branch worktree:
- `app/entity_linking.py` -> `app/setup/entity_linking.py`
- `app/show_env.py` -> `app/util/show_env.py`
- `app/test_db_connection.py` -> `app/util/test_db_connection.py`
- `app/test_spacy.py` -> `app/util/test_spacy.py`
- `app/historian_agent/fix_imports.py` -> `app/util/fix_imports.py`
- `app/historian_agent/verify_db.py` -> `app/util/verify_db.py`
- `app/historian_agent/test_env.py` -> `app/util/test_env.py`
- `app/historian_agent/test_adversary_rag.py` -> `app/util/test_adversary_rag.py`
- `app/historian_agent/test_chromadb_health.py` -> `app/util/test_chromadb_health.py`
- `app/historian_agent/test_document_structure.py` -> `app/util/test_document_structure.py`
- `app/historian_agent/test_field_count.py` -> `app/util/test_field_count.py`
- `app/historian_agent/test_models.py` -> `app/util/test_models.py`
- `app/historian_agent/test_rag_system.py` -> `app/util/test_rag_system.py`
- `app/historian_agent/test_sources.py` -> `app/util/test_sources.py`
- `app/historian_agent/test_verify_dimensions.py` -> `app/util/test_verify_dimensions.py`
- `app/scripts/e2e_cli_suite.py` -> `app/util/e2e_cli_suite.py`

### Verification
- Confirmed moved files exist in destination paths.
- Import-scan check after move:
  - Relative import found: `app/util/test_sources.py` (`from .question_models import ...`).
  - Bare historian-agent imports found:
    - `app/util/test_rag_system.py:33` `from chunking import ...`
    - `app/util/test_rag_system.py:34` `from embeddings import ...`
    - `app/util/test_rag_system.py:35` `from vector_store import ...`

## 2026-02-23 - Consolidate diagnostics into one non-destructive master e2e suite

### Problem
The repository had many standalone utility scripts, including scripts that can delete or rewrite data/files. This made cleanup risky and scattered validation logic across redundant one-off files.

### Solution
- Deleted redundant and potentially destructive standalone util scripts.
- Kept only runtime-supporting utility code and the single master CLI suite:
  - Kept: `app/util/__init__.py`, `app/util/mounts.py`, `app/util/e2e_cli_suite.py`
- Expanded `app/util/e2e_cli_suite.py` to absorb useful non-destructive checks:
  - Environment contract validation
  - Python dependency import validation (`openai`, `spacy`, `chromadb`, `numpy`, `pymongo`)
  - spaCy model load + basic NER sanity check
  - Chroma vector-store contract check (collection + embedding dimension)
  - Historian-agent module import check
  - Document structure contract check (text fields + `sections` shape)

### Verification
- Confirmed `app/routes.py` still imports only `app/util/mounts.py` from util runtime code.
- Ran the master suite (`app/util/e2e_cli_suite.py`) after consolidation to validate end-to-end behavior.

## 2026-02-23 - Enforce single master `.env` source

### Problem
Multiple tracked `.env` files existed in subdirectories, which could conflict with the root environment configuration and violate a single source-of-truth setup.

### Solution
- Removed nested environment files:
  - `app/historian_agent/.env`
  - `app/util/.env`
- Retained root `/Users/louishyman/coding/nosql/nosql_reader_cleanup/.env` as the only active master `.env` file for the app runtime.

### Verification
- Confirmed nested `.env` files were removed from the cleanup branch workspace.

## 2026-02-23 - Merge removed `.env` configs into master and remove redundancy

### Problem
After removing nested `.env` files, some configuration keys existed only in those removed files, and the root `.env` still contained duplicate key definitions that could cause ambiguous overrides.

### Solution
- Compared key sets from removed files against root `.env`:
  - `app/historian_agent/.env`
  - `app/util/.env`
  - `app/CLI_ingestion_files_wip/example.env`
- Added missing keys to root `.env` exactly once:
  - `ENABLE_LLM`, `FUZZY_MATCH_THRESHOLD`, `BATCH_SIZE`, `FIELDS_TO_PROCESS`, `ENABLE_MULTIPROCESSING`, `LINK_WIKIDATA`
  - `OLLAMA_DEFAULT_MODEL`, `OPENAI_DEFAULT_MODEL`, `LOG_LEVEL`
  - `MONGODB_URI`, `MONGODB_DB_NAME`
- Removed duplicate key assignments in root `.env` by keeping a single canonical definition per key (for example `CHROMA_PERSIST_DIRECTORY`, `HISTORIAN_AGENT_TOP_K`, `SECRET_KEY`, `OPENAI_API_KEY`, `LLM_MODEL`, `LLM_TEMPERATURE`).

### Verification
- Confirmed no missing keys remain from removed env sources.
- Confirmed root `.env` has no duplicate key definitions.

## 2026-02-23 - Remove non-master env files and move examples into docs

### Problem
One non-master env file remained (`app/CLI_ingestion_files_wip/example.env`). The project requirement is a single master runtime `.env`.

### Solution
- Deleted the extra env example file:
  - `app/CLI_ingestion_files_wip/example.env`
- Moved the example content into documentation:
  - Added `docs/ENV_EXAMPLES.md`
- Updated CLI docs to point to the documented examples and master env workflow:
  - `app/CLI_ingestion_files_wip/QUICK_REFERENCE.md`
  - `app/CLI_ingestion_files_wip/CLI_README.md`

### Verification
- Confirmed only one env file remains in the repository:
  - `/Users/louishyman/coding/nosql/nosql_reader_cleanup/.env`

## 2026-02-23 - Fix RAG tier endpoint contract regressions

### Problem
Tier endpoint validation exposed two runtime regressions:
- `/historian-agent/query-adversarial` failed with `AttributeError: 'list' object has no attribute 'values'` due to source-shape mismatch.
- `/historian-agent/query-tiered` failed with `AttributeError: 'TieredHistorianAgent' object has no attribute 'run'` due to route/agent contract drift.

### Solution
- Updated `app/historian_agent/adversarial_rag.py` to normalize `metrics["sources"]` from either legacy dict or current list formats before extracting source IDs for context hydration.
- Updated `app/routes.py` tiered branch in `process_rag_query` to call `agent.investigate(question)` and adapt its tuple return into the standardized API metrics shape.

### Verification
- Re-ran tier probes for:
  - `/historian-agent/query-basic`
  - `/historian-agent/query-adversarial`
  - `/historian-agent/query-tiered`
  - `/historian-agent/query-tier0`
- All four returned HTTP 200 with non-empty answers.

## 2026-02-23 - Refresh core docs for current repository state

### Problem
Core documentation drifted from the live codebase. It referenced outdated environment setup (`.env.example`), old app port (`5000`), deleted utility scripts, and did not include the current consolidated testing/tier status.

### Solution
- Updated `readme.md` to reflect:
  - verified runtime status (`e2e_cli_suite.py` passing, RAG tier endpoints healthy),
  - single master `.env` policy,
  - `docs/ENV_EXAMPLES.md` as documentation-only examples.
- Updated `setup.md` to reflect:
  - one root `.env` workflow (no `.env.example` copy step),
  - correct app URL (`http://localhost:5001`),
  - current connectivity/smoke commands (`python -c ...`, `app/util/e2e_cli_suite.py`),
  - removal of references to deleted utility scripts.
- Updated `docs/february_how_this_works.md` to reflect:
  - current operational snapshot,
  - single `.env` source-of-truth policy,
  - tiered route contract (`investigate(...)`),
  - current command reference for master smoke and tier probing.

### Verification
- Reviewed all three documents for consistency against current routes, utilities, and runtime behavior.

## 2026-02-23 - Expand how-it-works with setup and file/data relationships

### Problem
`docs/february_how_this_works.md` did not explicitly describe setup steps in one place and lacked a direct map from key files/modules to the collections and caches they read/write.

### Solution
- Added a dedicated `Setup (Current)` section with exact startup and validation commands.
- Added a `File and Data Relationship Map` table linking runtime files to data sources/sinks and ownership boundaries.
- Added a concise data-flow summary from ingestion through graph/statistics/RAG usage.
- Cleaned heading text for readability where numbering had become inconsistent in the updated region.

### Verification
- Confirmed the updated setup commands align with current container workflow (`docker compose ...`, `app/util/e2e_cli_suite.py`).
- Reviewed the relationship table against active modules (`routes.py`, `network/*`, `historian_agent/*`, `util/e2e_cli_suite.py`) and current collection names.

## 2026-02-23 - Document usage workflows for search, historian, corpus, and network

### Problem
The README listed pages and endpoints but did not provide clear user-facing usage flows for database search, Historian Agent modes, Corpus Explorer runs, and network exploration workflow.

### Solution
- Updated `readme.md` with a `Feature Usage` section covering:
  - Database search workflow and related endpoints.
  - Historian Agent workflow and tier endpoint list.
  - Corpus Explorer (Tier 0) sync/async workflow and artifact endpoints.
  - Network Analysis workflow, filter usage, and viewer-to-document context flow.
- Added `/corpus-explorer` to the main pages list.

### Verification
- Cross-checked all documented routes against `app/routes.py` route declarations.
- Confirmed endpoint names and usage descriptions align with current runtime behavior.

## 2026-02-23 - Fix main-branch documentation/script path references

### Problem
Documentation and helper scripts still contained stale file paths from a cleanup workspace, which could mislead operators on `main` and break the E2E helper command.

### Solution
- Updated documentation paths to point to the active main repository root:
  - `readme.md`
  - `setup.md`
  - `docs/february_how_this_works.md`
  - `app/CLI_ingestion_files_wip/QUICK_REFERENCE.md`
  - `app/CLI_ingestion_files_wip/CLI_README.md`
- Fixed `scripts/run_e2e_suite.sh` to call the canonical in-container suite path:
  - from `/app/scripts/e2e_cli_suite.py`
  - to `/app/util/e2e_cli_suite.py`

### Verification
- Confirmed no remaining `nosql_reader_cleanup` path references in markdown files under the repo.
- Confirmed no remaining `/app/scripts/e2e_cli_suite.py` reference in shell scripts under the repo.

## 2026-02-23 - Harden Docker status and Chroma embedding checks in diagnostics

### Problem
Two diagnostics were brittle against runtime/tooling differences:
- `quick_check.sh` expected `docker compose ps` text to include `running`, but current Compose reports service status as `Up`, causing false negatives.
- `app/util/e2e_cli_suite.py` used `sample.get("embeddings") or []`, which triggers ambiguous truth-value evaluation when Chroma returns a NumPy ndarray.

### Solution
- Updated `quick_check.sh` to use Compose status-aware service detection:
  - `docker compose ps --status running --services | grep -qx "app"`
- Updated `app/util/e2e_cli_suite.py` Chroma contract check to avoid ndarray truthiness and normalize embedding containers:
  - check `None` explicitly,
  - convert ndarray via `.tolist()` when present,
  - otherwise coerce iterable to `list(...)`,
  - validate with length checks.

### Verification
- Re-ran `quick_check.sh` and confirmed it no longer fails at the container-status step.
- Re-ran `scripts/run_e2e_suite.sh` and confirmed the Chroma vector-store contract check passes.

## 2026-02-23 - Refresh settings build metadata at app boot via run script

### Problem
The settings page showed stale branch/commit metadata (for example `codex/main-tier0-integration`) because `app/build_info.json` was mounted from the host and not refreshed automatically when booting the stack.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/run_app.sh` to:
  - run `scripts/stamp_build_info.sh` before `docker compose up`,
  - enforce strict shell flags and add an explicit shebang.
- Kept metadata generation in the existing canonical script so data structure and contracts remain unchanged.

### Verification
- Ran `scripts/stamp_build_info.sh` and confirmed `/Users/louishyman/coding/nosql/nosql_reader/app/build_info.json` now reports:
  - `branch: main`
  - current commit SHA/date/subject
  - fresh UTC generation timestamp.

## 2026-02-23 - Restore unique terms explorer population

### Problem
`/search-terms` rendered but always returned empty results (`words=[]`, `phrases=[]`) for fields like `ocr_text` and `summary`.

Root causes:
- Legacy generator path `/app/setup/generate_unique_terms.py` used a stale import (`from app.database_setup ...`) and failed with `ModuleNotFoundError: No module named 'app'`.
- Bootstrap expected `/app/generate_unique_terms.py`, but no canonical script existed at that location.
- With `RUN_BOOTSTRAP` disabled by default, no fallback generation occurred at startup.

### Solution
- Added canonical generator:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/generate_unique_terms.py`
- Implemented deterministic full rebuild of `unique_terms` from `documents` while preserving existing data contract:
  - `term`, `field`, `type` (`word`/`phrase`), `frequency`
- Used batched aggregation plus MongoDB bulk upserts for performance.
- Added index assurance for:
  - `unique_term_field_type` (unique compound index)
  - `field_type_frequency_idx` (sort/support index)
- Replaced `/Users/louishyman/coding/nosql/nosql_reader/app/setup/generate_unique_terms.py` with a compatibility wrapper that forwards to the canonical script and ensures `/app` is on `sys.path`.

### Verification
- Ran: `docker exec nosql_reader_app sh -lc 'cd /app && python generate_unique_terms.py --batch-size 250'`
- Result: indexed `9,642` documents and stored `524,498` records in `unique_terms`.
- Endpoint checks now return populated data:
  - `/search-terms?field=ocr_text` shows non-zero `unique_words`/`unique_phrases` and populated top terms.
  - `/search-terms?field=summary` shows non-zero `unique_words`/`unique_phrases` and populated top terms.

## 2026-02-23 - Add unique terms contract smoke test and wire into master E2E suite

### Problem
After restoring unique terms generation, there was no dedicated automated contract test for:
- `unique_terms` schema/index invariants, and
- `/search-terms` AJAX response shape expected by the frontend.

This left a regression gap where future changes could silently break the explorer while unrelated E2E checks still passed.

### Solution
- Added standalone smoke test module:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/util/unique_terms_contract_smoke.py`
- The new script validates:
  - `unique_terms` non-empty state,
  - allowed `type` values (`word`/`phrase`),
  - no null/empty `term` or `field`,
  - `frequency` presence and non-negative values,
  - required indexes (`unique_term_field_type`, `field_type_frequency_idx`),
  - `/search-terms` AJAX payload contract (`words`, `phrases`, `unique_words`, `unique_phrases`, `total_records`) and row shapes.
- Integrated into the master E2E suite:
  - imported `run_unique_terms_contract_check` in `app/util/e2e_cli_suite.py`
  - added check label: `Unique terms explorer contract`

### Verification
- Standalone: `python /app/util/unique_terms_contract_smoke.py` -> PASS.
- Full suite: `./scripts/run_e2e_suite.sh` -> PASS, `Total checks: 19`, `Failed: 0`.

## 2026-02-23 - Auto-populate unique terms explorer on initial page load

### Problem
Users opening `/search-terms` could see an apparently empty explorer until manually selecting a field. This looked like data was missing even when `unique_terms` was populated.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search-terms.html` to:
  - guard `fetchData` when no field is selected and show a clear empty-state message,
  - auto-select a default field on load (priority: `ocr_text`, then `summary`, then `filename`, then first available),
  - automatically fetch and render the first page of terms at startup.

### Verification
- Confirmed rendered page includes `initializeExplorer()` and preferred default field logic.
- Confirmed AJAX endpoint still returns populated payload for `ocr_text` (`unique_words`, `unique_phrases`, and table rows).

## 2026-02-23 - Redesign unique terms explorer for intuitive workflow

### Problem
The prior `/search-terms` UI was functional but cognitively heavy:
- showed words and phrases simultaneously,
- required manual submit-style filtering,
- had no quick field picks,
- did not provide direct term-to-search workflow.

### Solution
- Reworked `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search-terms.html` into a guided flow:
  - step indicator (`field -> type -> explore`),
  - quick field chips (prefers `ocr_text`, `summary`, `filename`),
  - single active table with `Words | Phrases` tabs,
  - debounced live filtering + clear action,
  - per-row actions (`Search`, `Copy`),
  - frequency bar visualization for faster scanning.
- Replaced jQuery AJAX call with native `fetch` while keeping backend endpoint/response contracts unchanged.
- Added search prefill support in `/Users/louishyman/coding/nosql/nosql_reader/app/static/script.js`:
  - reads `prefill_field` and `prefill_term` query params,
  - sets first search row values,
  - auto-submits search once,
  - removes params from URL to avoid repeated auto-submit.
- Added matching styles in `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css` for new chips/tabs/table/action patterns and mobile behavior.

### Verification
- Ran full suite: `./scripts/run_e2e_suite.sh`.
- Result: `Total checks: 19`, `Failed: 0` (all pass).
