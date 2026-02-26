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

## 2026-02-23 - Add cross-tab disproportionate-word analysis for anchor terms

### Problem
Users needed a way to ask: “When this anchor word appears in OCR text, which other words are disproportionately present?”
The existing unique-terms explorer only showed marginal frequencies, not conditional associations.

### Solution
- Added backend endpoint:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py`
  - `GET /search-terms/cross-tab`
- Implemented document-level disproportionate analysis (`compute_disproportionate_terms`) with:
  - field-scoped scan,
  - anchor presence check (word or phrase),
  - document-frequency counters (`global_doc_frequency`, `co_doc_frequency`),
  - Laplace-smoothed lift score:
    - `lift = P(term | anchor) / P(term | not anchor)`
  - sortable outputs including `lift` and `log2_lift`.
- Added safeguards and contracts:
  - query arg parsing/clamping (`limit`, `min_doc_freq`, `min_co_docs`),
  - normalized string/list[str] field handling,
  - consistent JSON shape with summary stats and latency.
- Added UI panel on `/search-terms`:
  - anchor input + thresholds,
  - “Run analysis” action,
  - result table with term, lift, co-doc count, global doc count, and actions (`Search`, `Copy`).
- Added styling for new controls/table in:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`
- Extended smoke coverage:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/util/unique_terms_contract_smoke.py`
  - now validates `/search-terms/cross-tab` response contract.

### Verification
- Manual API check:
  - `GET /search-terms/cross-tab?field=ocr_text&anchor=relief&limit=5&min_doc_freq=5&min_co_docs=3`
  - returned valid payload with ranked terms and non-zero stats.
- Full regression suite:
  - `./scripts/run_e2e_suite.sh`
  - `Total checks: 19`, `Failed: 0`.

## 2026-02-23 - Simplify cross-tab result display and add header tooltips

### Problem
Cross-tab results were accurate but harder to scan:
- headers lacked inline definitions,
- `Lift` values showed trailing decimals,
- `log2` detail added visual noise for quick interpretation.

### Solution
- Updated cross-tab table headers in:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search-terms.html`
  - Added mouseover definitions via `title` attributes for:
    - `Term`
    - `Lift`
    - `With Anchor`
    - `Global Docs`
    - `Actions`
- Simplified lift display in the row renderer:
  - removed log2 display from UI,
  - rounded `lift` to whole numbers (no trailing decimals).
- Kept backend payload unchanged so API/data contracts remain stable.

### Verification
- Confirmed rendered HTML contains new tooltip titles and no visible `log2` label in the table.
- Re-ran full suite: `./scripts/run_e2e_suite.sh` -> `Total checks: 19`, `Failed: 0`.

## 2026-02-23 - Replace native title hover with explicit tooltip markers for cross-tab headers

### Problem
Native browser `title` tooltips were not consistently visible for header definitions in some environments.

### Solution
- Replaced native `title` attributes with explicit tooltip markers in:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search-terms.html`
  - Added `?` hint badges beside cross-tab headers using:
    - `.search-terms__tooltip`
    - `data-tip` content
    - keyboard focus support via `tabindex="0"`
- Added CSS tooltip rendering in:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`
  - Uses `::after` with `data-tip` for hover/focus popup.

### Verification
- Confirmed rendered page includes `.search-terms__tooltip` markers and `data-tip` content.
- Re-ran full suite: `./scripts/run_e2e_suite.sh` -> `Total checks: 19`, `Failed: 0`.

## 2026-02-23 - Align search-terms help behavior with network tooltip standard

### Problem
The cross-tab header help behavior diverged from the established network-analysis standard by using page-specific custom tooltip markup/CSS.

### Solution
- Standardized cross-tab header help to the network pattern:
  - use `data-help` attributes on header cells in:
    - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search-terms.html`
  - include shared tooltip runtime:
    - `{{ url_for('static', filename='js/help-tooltips.js') }}`
- Removed custom tooltip styling/markup remnants from:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`
  - removed `.search-terms__tooltip` and `.search-terms__th-label` rules.

### Verification
- Confirmed search-terms template now has `data-help` headers and no custom tooltip classes.
- Re-ran full suite: `./scripts/run_e2e_suite.sh` -> `Total checks: 19`, `Failed: 0`.

## 2026-02-23 - Enable tooltip standard globally across pages

### Problem
The tooltip runtime (`help-tooltips.js`) was only available on selected pages, so the `data-help` pattern was inconsistent across routes.

### Solution
- Added global tooltip runtime include in:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html`
  - this ensures all pages now load the same tooltip system.
- Removed redundant page-specific include from:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search-terms.html`
  - to avoid duplicate script loading.

### Verification
- Checked major routes (`/`, `/search`, `/search-terms`, `/historian-agent`, `/corpus-explorer`, `/network-analysis`, `/database-info`, `/settings`, `/help`) and confirmed `help-tooltips.js` is present in rendered HTML.
- Re-ran full suite: `./scripts/run_e2e_suite.sh` -> `Total checks: 19`, `Failed: 0`.

## 2026-02-23 - Enforce tooltip targets on all primary pages

### Problem
Even with global `help-tooltips.js` loading, several pages had no explicit `data-help` targets, so users would not see consistent delayed-help behavior across navigation flows.

### Solution
- Added `data-help` text to the shared site brand and each primary navigation link in:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html`
- Added a new contract check to the master E2E suite in:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/util/e2e_cli_suite.py`
- New check validates for key routes:
  - HTTP 200 response
  - `help-tooltips.js` include present
  - at least one `data-help` target rendered

### Verification
- Ran full suite: `./scripts/run_e2e_suite.sh`
- Result: `Total checks: 20`, `Failed: 0`
- New passing check: `Tooltip standard contract :: paths=9 total_data_help_targets=101`.

## 2026-02-23 - Add real-time SSE backend log streaming for Historian Agent and Corpus Explorer

### Problem
Python backend logs were only visible on stderr/CLI, while browser debug consoles (`historian_agent.js` and `corpus_explorer.js`) only showed frontend-generated messages. This caused delayed/no visibility into long-running backend steps during Historian and Tier 0 runs.

### Solution
- Added a global SSE log stream module at:
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/log_stream.py`
  - Includes:
    - shared in-memory queue
    - `push(...)` API for backend emitters
    - `stream_generator(...)` for Flask SSE streaming
    - `signal_done()` completion sentinel
    - `StderrTee` to mirror selected stderr lines into the stream
- Installed stderr tee at app startup in:
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/main.py`
- Registered SSE endpoint and completion signaling in:
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/routes.py`
  - Added `/api/log-stream`
  - Added `signal_done()` in long-running Historian routes (`query-basic`, `query-adversarial`, `query-tiered`, `query-tier0`) and Tier 0 sync/async exploration completion paths.
- Added additive backend log push hooks in:
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/rag_base.py` (`debug_print`)
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/historian_agent/adversarial_rag.py` (`debug_step`)
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/historian_agent/tier0_utils.py` (`Tier0Logger.log`)
- Frontend SSE integration:
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/static/historian_agent.js`
    - Added reusable `openLogStream()` and wired `submitQuestion()` to open stream before fetch and always close in `finally`.
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/static/corpus_explorer.js`
    - Replaced existing status-polling run flow with standardized SSE+single-fetch flow.
    - UI now opens `/api/log-stream` before calling `/api/rag/explore_corpus` and closes safely with timeout/finally.
- Added missing debug style level in:
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/static/style.css`
  - `.agent-debug-entry--debug`

### Verification
- Python syntax check:
  - `python -m py_compile app/log_stream.py app/main.py app/routes.py app/rag_base.py app/historian_agent/adversarial_rag.py app/historian_agent/tier0_utils.py`
- JavaScript syntax check:
  - `node --check app/static/historian_agent.js`
  - `node --check app/static/corpus_explorer.js`
- Result: all checks passed.

## 2026-02-25 - Standardize debug console title to "Backend Log"

### Problem
The UI used inconsistent debug-console titles ("Pipeline Debug Log" in Historian Agent and "Progress Log" in Corpus Explorer), plus a placeholder message referencing "Debug log" and "pipeline events." This made backend-streamed logs look inconsistent across pages.

### Solution
- Standardized console heading text to `Backend Log` in:
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/templates/historian_agent.html`
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/templates/corpus_explorer.html`
- Updated empty-state helper text to match backend streaming language in:
  - `/Users/louishyman/coding/nosql/nosql_reader_sse/app/static/style.css`
  - New text: `Backend log is empty. Start a run to see backend events.`

### Verification
- Searched for prior labels and confirmed no remaining matches for:
  - `Pipeline Debug Log`
  - `Progress Log`
- Verified all debug consoles now display consistent backend-oriented naming.

## 2026-02-25 - Keep document viewer controls visible in the right panel

### Problem
In the document detail view, the image consumed the full panel height while `.zoom-controls` and `.adjustment-controls` were rendered below it. On tall documents, controls were pushed out of view and could require page scrolling.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css` to make the right image panel a sticky, fixed-height flex column:
  - `.document-detail .image-panel` now uses `display: flex`, `flex-direction: column`, `height: calc(100vh - 40px)`, `position: sticky`, and `top: 0`.
- Converted `#imageContainer` into the scrollable flex child that fills remaining space above controls:
  - Uses `flex: 1`, `overflow: auto`, and `min-height: 0`.
  - Removed the previous fixed `max-height` behavior that could squeeze controls off-screen.
- Updated `.zoom-controls` and `.adjustment-controls` with `flex-shrink: 0` so controls remain visible and are not compressed out of the viewport.

### Verification
- Confirmed there is no inline `#imageContainer` style override in `/Users/louishyman/coding/nosql/nosql_reader/app/templates/document-detail.html`.
- Confirmed stylesheet now drives layout with image-container scrolling and non-shrinking controls.

## 2026-02-25 - Ensure full controls stack is reachable without page scroll

### Problem
After the first layout fix, the top of the controls became visible but the full `.adjustment-controls` stack could still be clipped at some viewport sizes because panel height did not account for fixed header space.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css` to reduce both panel viewport contracts below the fixed header:
  - `.document-detail .info-panel` `max-height` changed to `calc(100vh - 136px)`.
  - `.document-detail .image-panel` `height` changed to `calc(100vh - 136px)` and sticky offset adjusted to `top: 12px`.
- Added a controls-area fallback scroll container:
  - `.adjustment-controls` now uses `max-height: 42vh` and `overflow-y: auto` so all controls remain reachable even on shorter viewports.

### Verification
- Confirmed live stylesheet served from `http://localhost:5001/static/style.css` contains the updated panel height and adjustment-controls overflow rules.

## 2026-02-25 - Extend delayed-help tooltip standard to all primary pages

### Problem
The Network Analysis page had high-quality, explicit tooltip text, but most other pages relied on generic fallback help generated at runtime. This created uneven guidance quality and left many page-specific controls under-documented.

### Solution
- Added explicit `data-help` text to primary controls and sections across templates:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/home.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search-terms.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/historian_agent.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/corpus_explorer.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/settings.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/database-info.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/help.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/document-list.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/network-document-list.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/login.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/data_files.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/error.html`
- Added explicit tooltip attributes for dynamic controls created in JavaScript:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/static/script.js` (search result table controls)
  - `/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js` (source links)
  - `/Users/louishyman/coding/nosql/nosql_reader/app/static/corpus_explorer.js` (investigate links, recovery action)
- Strengthened tooltip contract tests in:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/util/e2e_cli_suite.py`
  - Updated contract now checks additional routes and requires page-specific tooltip targets beyond base-nav-only coverage.

### Verification
- Ran E2E suite in app container:
  - `docker compose -p nosql_reader -f docker-compose.yml exec -T app python /app/util/e2e_cli_suite.py`
- Result:
  - `Tooltip standard contract` passed with `paths=12 total_data_help_targets=228`
  - Overall suite passed: `Total checks: 20`, `Failed: 0`.

## 2026-02-25 - Add demographics database + dashboard with occupation normalization

### Problem
The application lacked a demographics reporting surface over `person_syntheses`, so researchers could not review corpus-wide workforce statistics, browse one-row-per-person records, or export demographics without writing SQL. It also lacked a normalization path for noisy `occupation_primary` values.

### Solution
- Added a standalone demographics builder script:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/setup/build_demographics_db.py`
  - Builds `/app/data/demographics.db` from Mongo `person_syntheses` using `person_folder` as the canonical unique key.
  - Extracts employment/injury/date fields from synthesis JSON plus LLM-assisted `gender/race/nationality` extraction.
  - Includes idempotent upsert behavior and migration-safe occupation-normalization columns/indexes:
    - `occupation_normalized`, `occupation_norm_score`, `occupation_norm_method`
    - `idx_occupation_norm`
- Added occupation fuzzy normalizer script:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/setup/normalize_occupations.py`
  - Uses `rapidfuzz` against curated canonical labels with manual overrides.
  - Supports `--preview`, `--show-unmatched`, `--force`, and threshold tuning.
- Added demographics routes and API endpoints:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py`
  - `GET /demographics`
  - `GET /api/demographics/summary`
  - `GET /api/demographics/people`
  - `GET /api/demographics/export`
  - Summary now prefers `COALESCE(occupation_normalized, occupation_primary)` for occupation rollups.
  - People endpoint includes normalized occupation fields and supports occupation filtering against raw or normalized values.
- Added full demographics dashboard UI:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/demographics.html`
  - Chart.js summary panels, confidence pills, sortable/filterable/paginated people table, CSV export actions.
  - Added table visibility for normalized occupation + score.
- Added nav entry and docs updates:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html`
  - `/Users/louishyman/coding/nosql/nosql_reader/readme.md`
  - `/Users/louishyman/coding/nosql/nosql_reader/setup.md`
- Added persistent DB ignore rules:
  - `/Users/louishyman/coding/nosql/nosql_reader/.gitignore`
  - `/Users/louishyman/coding/nosql/nosql_reader/app/data/.gitignore`

### Verification
- Syntax checks passed:
  - `python -m py_compile app/routes.py app/setup/build_demographics_db.py app/setup/normalize_occupations.py`
- Requested dry-run passed:
  - `docker compose exec app python /app/setup/build_demographics_db.py --limit 5 --dry-run`
- Live sample build passed:
  - `docker compose exec app python /app/setup/build_demographics_db.py --limit 5`
- Normalization preview and live runs passed:
  - `docker compose exec app python /app/setup/normalize_occupations.py --preview --show-unmatched`
  - `docker compose exec app python /app/setup/normalize_occupations.py`
- Endpoint checks passed:
  - `GET http://localhost:5001/demographics` (HTTP 200)
  - `GET http://localhost:5001/api/demographics/summary` (HTTP 200)
  - `GET http://localhost:5001/api/demographics/people?...` (HTTP 200)
  - `GET http://localhost:5001/api/demographics/export` (HTTP 200; CSV download headers present)
- Connection integrity check attempted via project helper:
  - `bash quick_check.sh` failed one check: app container could not reach Ollama (`Test 5`).
  - Other checks in that script passed (Ollama on host reachable, container running, mounts detected).

## 2026-02-25 - Convert demographics pie charts to horizontal bars

### Problem
The demographics dashboard used doughnut charts for gender and race. With few categories but strong count differences, those pie slices were harder to compare precisely than bars.

### Solution
- Updated chart rendering in `/Users/louishyman/coding/nosql/nosql_reader/app/templates/demographics.html`:
  - `buildGenderChart()` now renders `type: 'bar'` with `indexAxis: 'y'`.
  - `buildRaceChart()` now renders `type: 'bar'` with `indexAxis: 'y'`.
  - Hid legends for these two charts and enabled integer x-axis ticks.
  - Added inline comments documenting the reason for the chart-type change.

### Verification
- Loaded `http://localhost:5001/demographics` and confirmed dashboard script includes bar-chart config for both gender and race chart builders.
- Existing summary endpoint contract remained unchanged (`/api/demographics/summary`).

## 2026-02-25 - Add percentage labels on demographics horizontal bars

### Problem
After converting gender and race charts from pie to horizontal bars, the chart still required visual estimation to interpret each category's share of the total.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/demographics.html`:
  - Added a local Chart.js plugin (`percentageLabelPlugin`) that computes dataset totals and draws percentage text for each bar.
  - Applied the plugin to both `buildGenderChart()` and `buildRaceChart()`.
  - Kept labels constrained inside chart bounds to avoid clipping.

### Verification
- Confirmed template script now contains `percentageLabelPlugin` and chart configs include `plugins: [percentageLabelPlugin]` for both gender and race charts.
- Existing `/api/demographics/summary` payload contract remains unchanged.

## 2026-02-25 - Show percentages on all demographics bar charts

### Problem
Only gender and race charts showed percentage labels, while occupations, divisions, hire decades, and injury charts still required manual estimation of each category's share.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/demographics.html`:
  - Expanded `percentageLabelPlugin` to support both horizontal (`indexAxis: 'y'`) and vertical bar charts.
  - Applied the plugin to all remaining bar charts:
    - `buildOccupationChart()`
    - `buildDivisionChart()`
    - `buildDecadeChart()`
    - `buildInjuryOccChart()`
  - Added orientation-aware label placement comments for maintainability.

### Verification
- Confirmed each bar-chart constructor now includes `plugins: [percentageLabelPlugin]`.
- Confirmed plugin logic branches for horizontal and vertical label positioning in the template script.

## 2026-02-25 - Add Network-style delayed help tooltips to demographics page

### Problem
The demographics dashboard relied mostly on generic auto-generated tooltip text. This was less specific than Network Analysis, where controls have explicit research-facing guidance in `data-help` attributes.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/demographics.html` to add explicit `data-help` annotations across:
  - Header and CSV export actions
  - KPI bar and individual KPI cards
  - All chart panels
  - Filter controls and action buttons
  - Table container and sortable header columns
  - Pagination and page summary indicators
- Added an inline template comment noting that explicit help copy was added to mirror Network Analysis behavior.

### Verification
- Confirmed `37` explicit `data-help` attributes now exist in the demographics template.
- Loaded `http://localhost:5001/demographics` and verified key tooltip strings render in HTML for header, filters, sortable columns, and pagination elements.

## 2026-02-26 - UI overhaul branch implementation with contract-safe adjustments

### Problem
The UI refresh proposal was valid directionally, but direct copy/paste would have broken existing contracts in this codebase:
- Historian Agent metrics currently return `metrics.total_time`, `metrics.doc_count`, and tier-stage objects in `metrics.tiers` (not `duration_ms`, `docs_retrieved`, or first-class `confidence`).
- Current template and route naming uses mixed dash/underscore patterns (`network-analysis.html`, `database-info.html`) that required explicit mapping for nav active states.
- Existing SSE log stream writes to `#agentDebugOutput`; replacing debug UI had to preserve that DOM contract.

### Solution
Implemented the UI overhaul on branch `codex/ui-rethink` with contract-safe adaptations across templates, routes, JS, and CSS.

#### 1) `base.html` head + navigation refresh
File: `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html`
- Replaced the head block additions with:
  - viewport meta
  - inline SVG favicon
  - Google Fonts preconnect + stylesheet
- Reworked nav into:
  - primary tool links (`Home`, `Search`, `Historian Agent`, `Corpus Explorer`, `Network`)
  - `More` dropdown (`Demographics`, `Database Info`, `Settings`, `Help`)
- Added active-link class wiring via `active_page` template variable.
- Added in-template dropdown controller JS (`navMoreBtn` / `navMoreDropdown`) with:
  - click toggle
  - outside-click close
  - `Escape` close

Contract-safe note:
- Kept route endpoints unchanged (`url_for(...)` names all existing).

#### 2) Route-level active page injection via helper
File: `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py`
- Added helper:
  - `_render(template: str, **kwargs)`
  - injects `active_page` with a safe default derived from template name
- Updated key page routes to use `_render(...)` with explicit `page=` where needed:
  - `/` -> `home`
  - `/search` -> `search`
  - `/historian-agent` -> `historian_agent`
  - `/corpus-explorer` -> `corpus_explorer`
  - `/network-analysis` -> `network_analysis`
  - `/demographics` -> `demographics`
  - `/database-info` -> `database_info`
  - `/help` -> `help`
  - `/settings` -> `settings`

Contract-safe note:
- Preserved all existing response payload/data contexts passed to templates.
- Preserved template filenames exactly (`network-analysis.html`, `database-info.html`).

#### 3) Home page content wiring to existing card styles
File: `/Users/louishyman/coding/nosql/nosql_reader/app/templates/home.html`
- Replaced minimal hero-only body with:
  - richer archive-context hero copy
  - action buttons
  - fully populated `.home-sections` card grid using existing `.home-card` CSS
- Removed obsolete home tabs script block that was targeting elements not present in current DOM.

Contract-safe note:
- Kept all links on existing endpoints; no backend changes required.

#### 4) Historian Agent template update
File: `/Users/louishyman/coding/nosql/nosql_reader/app/templates/historian_agent.html`
- Updated method selector labels/hint text to Quick/Verified/Deep wording while preserving submitted values:
  - `basic`, `adversarial`, `tiered`
- Replaced terminal-style debug block with Activity Stream block:
  - indicator + label
  - show/hide log button
  - clear button
  - log container remains `id="agentDebugOutput"`
- Removed old bottom performance-metrics section.

Contract-safe note:
- Preserved IDs required by JS/SSE (`agentDebugOutput`, `agentDebugClear`).

#### 5) Historian Agent JS: activity stream + inline metrics without API shape changes
File: `/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`
- Updated method hint map text to match new selector labels.
- Added `ActivityStream` controller:
  - running/idle indicator states
  - auto-expand log during active run
  - auto-collapse shortly after completion
  - duration summary text
- Added `normalizeInlineMetrics(rawMetrics, method)`:
  - derives UI metrics from existing backend contract:
    - duration from `total_time` fallback
    - docs from `doc_count`
    - tier/confidence inferred from `metrics.tiers` (tiered mode) and verification `score`
- Added `renderInlineMetrics(...)` and attached pills under assistant answer bubbles.
- Updated history rendering so inline metrics are applied to the latest assistant message for each response render cycle.
- Removed old bottom metrics-grid rendering logic (`displayMetrics`, `addMetric`) to avoid stale/dead UI path.
- Preserved SSE event stream behavior and log wiring.

Contract-safe note:
- No backend API changes required.
- No changes to method endpoint URLs (`/historian-agent/query-basic|adversarial|tiered`).

#### 6) CSS tokenization + nav/activity/inline-metrics styling + legacy detail color cleanup
File: `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`
- Added top-level `:root` design tokens for colors/radii/shadows/transitions.
- Updated body/heading color usage to token variables.
- Updated nav styling for:
  - active link class
  - `More` dropdown button and menu
  - responsive wrap behavior with new `.primary-nav__list`
- Updated document-detail legacy hardcoded colors to tokenized values:
  - `.info-table` / `.nested-table` headers + cells
  - `.nested-item`, `.nested-block`
  - `.document-detail .info-section`
  - `.resizer`, `.linked-information`, `.entity-group`
- Added Activity Stream CSS (`.agent-activity*`).
- Added inline metrics CSS (`.agent-metrics-bar`, `.metric-pill*`).
- Added compatibility style for current message markup (`.agent-message__bubble`) alongside legacy `.agent-message__body`.

Contract-safe note:
- Kept existing class names used elsewhere while introducing new selectors incrementally.

### Verification
Executed checks:
- `python3 -m py_compile app/routes.py` -> passed
- `node --check app/static/historian_agent.js` -> passed
- `bash ./quick_check.sh` -> partially passed
  - Passed: host Ollama reachable, vision models detected, app container running, archives mounted
  - Failed: container-to-Ollama connectivity (`Test 5: Container CANNOT reach Ollama`)

Interpretation:
- The UI/code changes are syntactically valid.
- Existing environment/networking issue remains in Docker/Ollama connectivity and is unrelated to these UI edits.

## 2026-02-25 - Add Unique Terms Explorer to home cards and More menu

### Problem
The refreshed dashboard and header navigation did not expose the Unique Terms Explorer in the new IA:
- No front-page card linked to `/search-terms`.
- No utility-menu (`More`) entry for the terms explorer.
- `/search-terms` did not set `active_page`, so nav highlighting could not mark the new menu item as active.

### Solution
Updated templates and routing to surface Unique Terms Explorer in both requested locations while preserving route contracts.

#### 1) Add menu item under More
File: `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html`
- Added a new dropdown link in `primary-nav__dropdown`:
  - Label: `Unique Terms Explorer`
  - Target: `url_for('search_terms')`
- Extended the `More` button active-state condition to include `search_terms` so the utility group highlights when the terms page is open.
- Added inline template comment explaining why the link was introduced.

#### 2) Add dashboard card on the home page
File: `/Users/louishyman/coding/nosql/nosql_reader/app/templates/home.html`
- Added a new `.home-card` in the existing card grid:
  - Title: `Unique Terms Explorer`
  - Icon: `fa-hashtag`
  - Description focused on word/phrase frequency and cross-tab contrast workflow
  - CTA link: `url_for('search_terms')`
- Added inline template comment documenting that this is a first-class dashboard entry point.

#### 3) Make active nav highlighting work for `/search-terms`
File: `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py`
- Updated the HTML branch of `search_terms()` from `render_template(...)` to:
  - `_render('search-terms.html', page='search_terms', ...)`
- Kept AJAX behavior unchanged (`X-Requested-With == XMLHttpRequest` still returns JSON exactly as before).
- Added inline route comment for nav-highlighting intent.

### Verification
- Syntax check:
  - `python3 -m py_compile app/routes.py` -> passed
- Live UI verification against running container (`http://localhost:5001`):
  - Home page includes `Unique Terms Explorer` card + `Open Terms Explorer` link.
  - More menu contains `Unique Terms Explorer` link to `/search-terms`.
  - On `/search-terms`, both the More button and dropdown item show active-state classes.

## 2026-02-25 - Serif heading switch + complete palette token refactor

### Problem
After the UI overhaul, typography and color system were still partially mixed:
- Display typography used a sans-serif heading family (`Montserrat`) instead of a serif style aligned with the historical archive tone.
- `style.css` still contained many hardcoded color literals across components, despite introducing design tokens.

### Solution
Implemented a full typography/palette normalization pass.

#### 1) Switched heading/display font to serif
Files:
- `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html`
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`

Changes:
- Updated Google Fonts import from `Montserrat + Open Sans` to `Merriweather + Open Sans`.
- Added root font tokens:
  - `--font-body: 'Open Sans', sans-serif`
  - `--font-heading: 'Merriweather', Georgia, serif`
- Routed heading/display usages to tokens:
  - `h1, h2, h3`
  - `.site-brand`
  - `.entity-group h3`
  - `.agent-activity__label`
- Routed compact body text usage in pills to `--font-body`.

#### 2) Fully tokenized palette usage in stylesheet
File:
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`

Changes:
- Added `--color-surface-muted` token to remove fallback-only muted surface usage.
- Generated and inserted a complete set of `--palette-*` aliases in `:root` for every color literal used by component rules.
- Replaced all non-root hardcoded color literals (`#...`, `rgb(...)`, `rgba(...)`) with `var(--palette-...)` references.
- Kept literal values centralized inside `:root` only, making palette adjustments single-source.

Implementation note:
- The replacement was performed programmatically and then verified, ensuring no residual hardcoded colors remain outside token definitions.

### Verification
- `git diff --check` -> passed
- Live page check (`http://localhost:5001`) confirms updated Google Fonts link includes `Merriweather` + `Open Sans`.
- Static checks on `style.css`:
  - No hardcoded color literals remain outside `:root`.
  - No unresolved CSS custom properties are referenced (`var(--...)` all have definitions).
- Existing UI endpoints remained unchanged; running app container continued serving without restart failures.

## 2026-02-25 - Update Settings page to match refreshed IA and theme tokens

### Problem
After the IA, font, and palette refactor, the Settings page still lagged behind in two ways:
- It had no direct shortcut block for newly emphasized tools (including Unique Terms Explorer).
- Persisted defaults shown in the form still reflected old typography/color values from `app/config.json` (`Montserrat`, legacy primary/secondary/text/background colors).

### Solution
Updated settings template, styling, and persisted config defaults to align with the new UI system.

#### 1) Added tool shortcuts section in Settings
File:
- `/Users/louishyman/coding/nosql/nosql_reader/app/templates/settings.html`

Changes:
- Added a new `Tool shortcuts` settings section at the top with direct buttons to:
  - Search
  - Unique Terms Explorer
  - Historian Agent
  - Corpus Explorer
- Added explanatory inline comment and `data-help` coverage.

#### 2) Updated Settings form defaults to current theme contract
File:
- `/Users/louishyman/coding/nosql/nosql_reader/app/templates/settings.html`

Changes:
- Updated fallback defaults for font fields:
  - main: `Open Sans, sans-serif`
  - headers: `Merriweather, Georgia, serif`
- Updated fallback defaults for color fields:
  - primary: `#2563eb`
  - secondary: `#64748b`
  - background: `#ffffff`
  - text: `#1e293b`

#### 3) Added styles for settings shortcuts layout
File:
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`

Changes:
- Added `.settings-shortcuts` and `.settings-shortcuts .button` styles for responsive button grouping.
- Kept styling token-based (no hardcoded non-root literals introduced).

#### 4) Synced persisted config values to the new theme
File:
- `/Users/louishyman/coding/nosql/nosql_reader/app/config.json`

Changes:
- Updated both flat and nested config keys for consistency:
  - `fonts[headers]` / `fonts.headers` -> `'Merriweather', Georgia, serif`
  - `colors[primary]` / `colors.primary` -> `#2563eb`
  - `colors[secondary]` / `colors.secondary` -> `#64748b`
  - `colors[background]` / `colors.background` -> `#ffffff`
  - `colors[text]` / `colors.text` -> `#1e293b`

### Verification
- `python3 -m json.tool app/config.json` -> passed (valid JSON)
- `git diff --check` -> passed
- Live page check (`http://localhost:5001/settings`) confirms:
  - `Tool shortcuts` section renders
  - `Unique Terms Explorer` shortcut is present
  - font/color fields now display updated Merriweather + palette values from config.

## 2026-02-25 - Remove Tool shortcuts section from Settings

### Problem
The `Tool shortcuts` card introduced on the Settings page was not appropriate for the Settings information architecture, which should remain focused on configuration controls.

### Solution
Removed the shortcuts section and related styling.

Files changed:
- `/Users/louishyman/coding/nosql/nosql_reader/app/templates/settings.html`
  - Removed the entire `Tool shortcuts` section block (`#tool-shortcuts`).
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`
  - Removed now-unused `.settings-shortcuts` and `.settings-shortcuts .button` CSS rules.

### Verification
- Live check of `http://localhost:5001/settings` confirms the `Tool shortcuts` section no longer renders.
- `git diff --check` passed.

## 2026-02-26 - Historian UX implementation (all 7 features) with canonical search fields

### Problem
The historian workflow needed seven UI/backend improvements at once: entity action popovers, inline citation footnotes, structured answer sections, archival-first search filters, citation copy from document detail, research memo export, and person-first network entry. The search filter wiring also needed to use real comparable schema fields and avoid silent zero-result queries from wrong field paths.

### Solution
Implemented all seven features using existing contracts and existing schema fields, with additive changes only.

Files changed:
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/js/network.js`
- `/Users/louishyman/coding/nosql/nosql_reader/app/network/network_routes.py`
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`
- `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/rag_query_handler.py`
- `/Users/louishyman/coding/nosql/nosql_reader/app/templates/document-detail.html`
- `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search.html`
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/script.js`
- `/Users/louishyman/coding/nosql/nosql_reader/app/templates/corpus_explorer.html`
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/corpus_explorer.js`
- `/Users/louishyman/coding/nosql/nosql_reader/app/templates/network-analysis.html`
- `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py`

Key outcomes:
- Replaced legacy entity popup with historian action popover (View in Network, Search all records, Dismiss) and lazy mini-network preview.
- Added keyboard accessibility for entity actions and Escape-close behavior.
- Added person-first network search UI plus `/api/network/entity-search` endpoint.
- Added inline `[Source N]` footnote superscripts and archival citation footnote block in historian answers while preserving legacy sources panel.
- Added historian section styling for uncertainty/gaps/leads markers.
- Updated shared RAG generation prompt so Basic, Adversarial, and Tiered all produce symmetric historian sections.
- Added date range, record type, physical box filters to search UI while preserving existing JSON `/search` contract.
- Wired canonical comparable fields only: `year`, `document_type`, `archive_structure.physical_box`.
- Added explicit `missing` filter handling for `document_type` and `archive_structure.physical_box`.
- Added dynamic `document_type` options to `/api/dropdown-fields` from existing data.
- Added one-click archival citation copy on document detail using injected `window._documentMeta`.
- Added Corpus Explorer "Export as Memo" action and memo generation.
- Added fallback entity resolution by text when `entity_refs` are absent so popover actions still work where possible.

### Verification
- `python -m py_compile app/routes.py app/network/network_routes.py app/historian_agent/rag_query_handler.py` passed.
- `node --check app/static/js/network.js app/static/historian_agent.js app/static/corpus_explorer.js app/static/script.js` passed.
- `/api/dropdown-fields` now returns `document_type` values.
- `build_query(...)` in container confirms canonical-field queries and `missing` conditions are generated correctly.
- `quick_check.sh` was executed; DB/app checks ran, but it reported an existing environment connectivity issue: app container cannot reach Ollama.

## 2026-02-26 - Search UX rollback to field-row-only filters

### Problem
The new top-level archival filter panel (date range / record type / physical box) conflicted with the preferred search interaction model. The desired model is to keep filtering inside the existing row-based `Field + Operator + Search term` contract.

### Solution
Removed the top archival filter box and reverted search query handling to the original row-based contract while keeping `document_type` as a dropdown-backed field option inside existing field rows.

Files changed:
- `/Users/louishyman/coding/nosql/nosql_reader/app/templates/search.html`
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/script.js`
- `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py`
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`

Details:
- Removed `search-primary-filters` markup and associated JS wiring for top controls.
- Restored frontend validation to require first-row `field1 + searchTerm1` as before.
- Removed backend `date_from` / `date_to` / `record_type` / `physical_box` additive query block from `build_query`.
- Added `document_type` as a hardcoded dropdown-class field option in the existing field selector and excluded duplicates from dynamic loop.
- Kept `/api/dropdown-fields` support for `document_type` values so selecting `document_type` in any row uses a value dropdown.

### Verification
- `python -m py_compile app/routes.py app/network/network_routes.py` passed.
- `node --check app/static/script.js app/static/js/network.js` passed.
- Grep check confirms top-filter selectors/markup are removed.

### 2026-02-26 Follow-up
Adjusted row-based dropdown semantics so selecting `missing` for `document_type` or `archive_structure.physical_box` now maps to missing/null/empty-field queries in `build_query`, instead of a literal text regex. Also ensured `/api/dropdown-fields` returns `document_type` options from live data and includes a `missing` option for both dropdown-backed sparse fields.

## 2026-02-26 - Historian Agent Chicago-style inline footnotes + hover source previews

### Problem
- Historian Agent inline citations were rendered exactly where `[Source N]` appeared, which often placed superscripts before terminal punctuation instead of Chicago-style claim endings.
- Superscript hover text only showed the source label, not the actual citation.
- A legacy side source list could conflict with the new bottom footnote reading flow.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js` to:
  - build a source-reference map that links source labels to footnote numbers and citation tooltip text,
  - inject superscript links from `[Source N]` markers with source-aware hover text,
  - move superscripts after trailing punctuation (`. , ; : ! ?`) for Chicago-style placement,
  - keep citations in the per-answer bottom footnote block and stop repopulating the legacy side source panel.
- Added small helper utilities for source-label normalization and HTML-entity-safe tooltip text.

### Verification
- `node --check /Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`
- Confirmed no remaining code path unhides/repopulates the legacy side source list during answer rendering.

## 2026-02-26 - Historian Agent template/style cleanup for bottom-footnote citations

### Problem
- The Historian Agent template still included a legacy `agent-sources` side panel structure even after switching to inline superscripts + bottom footnotes.
- Keeping inactive legacy source-panel HTML/CSS made the page contract ambiguous.

### Solution
- Removed the legacy sources panel markup from `/Users/louishyman/coding/nosql/nosql_reader/app/templates/historian_agent.html`.
- Removed unused `.agent-sources`, `.agent-source__title`, and `.agent-source__snippet` styles from `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`.
- Kept the existing historian page layout and controls unchanged otherwise.

### Verification
- Confirmed no remaining `agent-sources` HTML block in the historian template.
- Confirmed legacy `agent-sources` CSS selectors are removed.

## 2026-02-26 - Historian Agent footnotes forced below answer + `(Source N)` superscript conversion

### Problem
- In live rendering, footnotes appeared as a right-side column beside the assistant answer instead of at the bottom.
- Some model outputs used `(Source N)` markers, which were not converted to inline superscripts because only `[Source N]` was parsed.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`:
  - append the footnote block inside the assistant bubble (`bubble.appendChild`) so citations render at the bottom of each answer,
  - expand citation marker parsing to handle both `[Source N]` and `(Source N)`.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`:
  - set `.agent-message--assistant` to a vertical stack (`flex-direction: column`) as a layout safeguard to keep answer + footnotes aligned bottom-first.

### Verification
- `node --check /Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`
- Quick regex sanity check confirmed `(Source 1)` and `[Source 7]` both transform to superscripts and land after punctuation.

### Follow-up cleanup
- Removed stale Historian Agent JS references to `agentSources`/`agentSourcesList` now that side-source markup is gone, preventing dead code paths and keeping the contract explicit.

## 2026-02-26 - Root-cause fix: historian citation parser contract drift

### Problem
- Citation rendering failed for real model output variants like `(Source 1: RDApp-225116Hupp003)` and `(Sources 2-10)`.
- The previous fallback extracted all digits from the marker text, so document IDs contaminated source-number resolution (for example, `Source 1: RDApp-225116...` became an invalid composite number), leaving citations unconverted.

### Solution
- Reworked citation parsing in `/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`:
  - `buildSourceReferenceMap` now stores references by label and by source number,
  - `resolveMarkerReferences` resolves exact labels first, then parsed numeric references,
  - `extractSourceNumbers` parses singular/plural/range/list forms and intentionally ignores suffix text after `:`/`;` to avoid document-ID digit bleed,
  - `injectFootnotes` now supports multi-source markers (comma-separated superscripts) and keeps Chicago punctuation placement.
- Added `.fn-link-sep` style in `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css` for multi-reference superscript separators.
- Tightened generation contract in `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/rag_query_handler.py` to enforce exact `[Source N]` markers and avoid non-canonical forms.

### Verification
- `node --check /Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`
- `python -m py_compile /Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/rag_query_handler.py`
- Manual parser sanity checks:
  - `Source 1: RDApp-225116Hupp003 => 1`
  - `Sources 2-10 => 2,3,4,5,6,7,8,9,10`
  - `Sources 1, 3, and 5 => 1,3,5`

## 2026-02-26 - Normalize `/historian-agent/query` sources at API boundary

### Problem
- `/historian-agent/query` returned legacy source lists (`reference/id/title/snippet`) directly from `agent.invoke`, while historian frontend citation rendering expects canonical source dict entries with `display_name` and `url`.
- This route-level shape drift risked `Unknown source` citations and inconsistent behavior compared with `/historian-agent/query-basic|adversarial|tiered`.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py`:
  - generalized `sources_list_to_dict` to normalize both modern (`label/id/display_name`) and legacy (`reference/id/title/snippet`) source shapes,
  - added tolerant pass-through for already-normalized dicts,
  - normalized `/historian-agent/query` output through `sources_list_to_dict` before returning,
  - added `search_id` + cached ordered IDs for normalized route parity,
  - persisted normalized sources in assistant history entries.

### Verification
- `python -m py_compile /Users/louishyman/coding/nosql/nosql_reader/app/routes.py`
- Confirmed `/historian-agent/query` response payload now emits canonical `sources` dict entries with `id`, `display_name`, and `url`.

## 2026-02-26 - Follow-up citation parser hardening for `Source [N: ...]` variants

### Problem
- Answers still showed literal markers like `Source [1: RDApp-...]` because citation replacement only matched bracket forms with numeric-only payloads, so bracket bodies containing `:` and IDs bypassed superscript conversion.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js` citation pattern to match:
  - `Source [1: ...]`
  - `Sources [2-4]`
  - bare `[1: ...]`
- Normalized matched bracket bodies back into `Source ...` markers before number extraction, so existing range/list parsing logic applies consistently.
- Added explicit assistant research-bubble width guard in `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css` to ensure full-width answer layout.

### Verification
- `node --check /Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`
- Regex sanity checks confirmed conversion for:
  - `Source [1: RDApp-...]`
  - `[1: RDApp-...]`
  - `Sources [2-4]`
- Verified live container files include the updated parser and width rule under `/app/static/`.

## 2026-02-26 - Semantic Search page + API endpoint

### Problem
- The app had vector retrieval infrastructure but no dedicated semantic-search API route or UI page for running meaning-based nearest-neighbor queries directly.
- Collection naming can differ by environment (`historian_documents` vs `historian_document_chunks`), so hardcoding one name risks returning empty results in some deployments.

### Solution
- Added semantic search routes in `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py`:
  - `GET /semantic-search` renders a new semantic-search page.
  - `POST /api/semantic-search` validates input, embeds query text, executes vector search, and returns scored/snippet results.
- Implemented collection fallback order in API route:
  - Use `APP_CONFIG.chroma.collection_name` first.
  - Fallback to `"historian_document_chunks"` when config is empty.
- Added new template `/Users/louishyman/coding/nosql/nosql_reader/app/templates/semantic_search.html`.
- Added frontend behavior in `/Users/louishyman/coding/nosql/nosql_reader/app/static/js/semantic_search.js`:
  - submit search query + `top_k`,
  - render ranked result cards with score/distance and document links,
  - clear/reset behavior.
- Added semantic-search styles in `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`.
- Added nav link in `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html`.

### Verification
- `python -m py_compile /Users/louishyman/coding/nosql/nosql_reader/app/routes.py`
- `node --check /Users/louishyman/coding/nosql/nosql_reader/app/static/js/semantic_search.js`

## 2026-02-26 - Semantic Search Meaning View + tooltip coverage

### Problem
- Nearest-neighbor rows were visible, but users could not quickly interpret what the neighbor set meant at a corpus level.
- Tooltip coverage on semantic search was incomplete, especially for dynamically rendered result elements.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/semantic_search.html`:
  - added a view toggle (`Neighbor view` vs `Meaning view`),
  - added a dedicated meaning panel container,
  - expanded `data-help` tooltip text across semantic-search controls and result framing.
- Reworked `/Users/louishyman/coding/nosql/nosql_reader/app/static/js/semantic_search.js`:
  - added `Meaning view` computation from returned neighbors (score bands, recurring terms, grouped document clusters, overlap cues),
  - added view-state toggling and mode-aware metadata text,
  - added `data-help` attributes for dynamic neighbor cards, score badges, meaning summaries, term chips, and document-cluster cards,
  - explicitly attached tooltips to dynamically inserted nodes using `window.AppHelpTooltips.attach(...)`.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css` with new semantic meaning-view styles (toggle pills, summary stats, term chips, cluster cards).

### Verification
- `node --check /Users/louishyman/coding/nosql/nosql_reader/app/static/js/semantic_search.js`

## 2026-02-26 - Semantic Meaning view not switching + tooltip discoverability fix

### Problem
- User reported that switching from Neighbor view to Meaning view showed no visible change.
- Tooltips were difficult to discover in practice (2s delay) and could appear absent during quick hover checks.
- Browser cache could keep an older `semantic_search.js` version after template/JS updates.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/js/semantic_search.js`:
  - switched to robust init (`run now if DOM already loaded`, otherwise `DOMContentLoaded`) so bindings are always attached,
  - added fallback mirroring of `data-help` to native `title` attributes for immediate tooltip availability if custom tooltip binding is delayed/missing.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/js/help-tooltips.js`:
  - reduced delayed-help trigger from 2000ms to 650ms for quicker visibility.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/semantic_search.html`:
  - added versioned semantic JS URL query param to force cache refresh of the updated script.

### Verification
- `node --check /Users/louishyman/coding/nosql/nosql_reader/app/static/js/semantic_search.js`
- `curl -sS http://localhost:5001/semantic-search` confirms updated markup and versioned semantic JS include.

## 2026-02-26 - Historian Agent refinement modal + persistent refinement toggle

### Problem
- Historian Agent submitted every query immediately, including short/vague prompts where users wanted a quick refinement checkpoint.
- Refinement preference needed to be user-discoverable on the Historian page and persist across sessions.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/historian_agent.html`:
  - added an in-page `Suggest refinements` checkbox near the query box,
  - added a refinement modal with suggestion text and `Search as typed` / `Search refined question` actions.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`:
  - added localStorage-backed preference key `refinement_enabled` (default enabled),
  - synchronized refinement toggles between in-page control and settings partial control when both exist,
  - added short-query interception (`< 5` words) that opens the refinement modal before search,
  - wired modal actions so users can run as typed or return to edit in place.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/partials/historian_agent_settings.html`:
  - added a matching refinement toggle control (local preference only, backend config contract unchanged).
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css` with modal/toggle styles.

### Verification
- `node --check /Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js`
- `curl -sS http://localhost:5001/historian-agent | rg "refinementToggleInline|refinementModal|vendor/marked.min.js"`

## 2026-02-26 - Self-hosted frontend vendor assets for offline runtime

### Problem
- Templates depended on external CDNs (Google Fonts, Font Awesome, Marked.js, Chart.js), which breaks in offline kiosk deployments.

### Solution
- Added local vendor assets under `/Users/louishyman/coding/nosql/nosql_reader/app/static/vendor`:
  - `font-awesome.min.css` + `webfonts/*`,
  - `marked.min.js`,
  - `chart.min.js`.
- Rebased Font Awesome CSS webfont paths to local `vendor/webfonts`.
- Updated template references:
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html` now uses local Font Awesome and removes Google Fonts links,
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/historian_agent.html` now uses local Marked,
  - `/Users/louishyman/coding/nosql/nosql_reader/app/templates/demographics.html` now uses local Chart.js.
- Kept typography intentional with offline-safe fallback stacks in `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`.

### Verification
- `curl -sS http://localhost:5001/semantic-search | rg "vendor/font-awesome.min.css"`
- `curl -sS http://localhost:5001/historian-agent | rg "vendor/marked.min.js"`
- `curl -sS http://localhost:5001/demographics | rg "vendor/chart.min.js"`

## 2026-02-26 - Demographics columns for ethnicity/national_origin with backward-compatible contract

### Problem
- Demographics people/export payloads did not expose `ethnicity` and `national_origin`.
- Some databases may not yet have those columns, so naive SELECTs can fail and break the page.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py`:
  - added `DEMOGRAPHIC_DISPLAY_FIELDS` shared contract,
  - added `_demographics_table_columns(...)` and `_demographics_optional_select_expr(...)` helpers,
  - updated `/api/demographics/people` to always return `ethnicity` and `national_origin` keys,
  - added safe fallback for missing `national_origin` column to `nationality`, otherwise empty string,
  - updated `/api/demographics/export` to include the same optional columns with fallback behavior.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/demographics.html`:
  - added sortable `Ethnicity` and `National Origin` table columns,
  - updated table rendering and loading/error `colspan` handling to match the expanded schema.

### Verification
- `python -m py_compile /Users/louishyman/coding/nosql/nosql_reader/app/routes.py`
- `curl -sS http://localhost:5001/demographics | rg "Ethnicity|National Origin"`

## 2026-02-26 - Demographics term clicks now open direct results list (no search UI)

### Problem
- Clicking a demographic name/term sent users to `/search` in interface mode, requiring extra interaction and not landing directly on a document list.

### Solution
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/demographics.html`:
  - replaced legacy name links with prefilled list-mode links:
    - `/search?prefill_field=<field>&prefill_term=<term>&list_only=1`
  - made key demographic terms clickable as well (name, ethnicity, national origin, occupation, normalized occupation, division),
  - kept missing/unknown values non-clickable,
  - added safe HTML escaping for dynamic term rendering.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/script.js`:
  - added `list_only` handling on search page,
  - hides search header/form and shows a compact “Document List” banner,
  - keeps existing prefill auto-submit behavior to load results immediately,
  - includes a “Show search controls” button to exit list-only view.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css`:
  - added styles for the list-only banner.

### Verification
- `node --check /Users/louishyman/coding/nosql/nosql_reader/app/static/script.js`
- Confirmed demographics link generation uses `list_only=1` and prefill query params.

## 2026-02-26 - Restore Open Sans + Merriweather as true defaults via local bundling

### Problem
- Typography defaults relied on fallback stacks only after CDN removal, so Open Sans/Merriweather could disappear when not installed locally.
- Requirement: keep Open Sans and Merriweather as the default look, with fallback only if those fonts cannot load.

### Solution
- Added local bundled font files in `/Users/louishyman/coding/nosql/nosql_reader/app/static/vendor/fonts/`:
  - `open-sans-400.ttf`, `open-sans-600.ttf`, `open-sans-400-italic.ttf`
  - `merriweather-600.ttf`, `merriweather-700.ttf`
- Added `/Users/louishyman/coding/nosql/nosql_reader/app/static/vendor/fonts.css` with `@font-face` declarations for `Open Sans` and `Merriweather`.
- Updated `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html` to load `vendor/fonts.css` globally.
- Restored font tokens in `/Users/louishyman/coding/nosql/nosql_reader/app/static/style.css` to:
  - `--font-body: 'Open Sans', sans-serif;`
  - `--font-heading: 'Merriweather', Georgia, serif;`

### Verification
- `curl -sS http://localhost:5001/search` confirms `vendor/fonts.css` is loaded.
- `curl -sSI http://localhost:5001/static/vendor/fonts.css` returns `200`.
- `curl -sSI http://localhost:5001/static/vendor/fonts/open-sans-400.ttf` returns `200`.
- `curl -sSI http://localhost:5001/static/vendor/fonts/merriweather-700.ttf` returns `200`.
