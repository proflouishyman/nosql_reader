# Tier 0 Status, Limitations, and What We Hoped Would Work

This document is a candid status report meant for an engineer friend who wants to understand the current Tier 0 system, what it delivers, and where the weak spots are.

## What We Hoped Would Work
- **Historian-style corpus exploration**: read thousands of documents and surface patterns, contradictions, and research questions before any user query (avoids top‑k myopia).
- **Notebook as a bounded workspace**: accumulate a structured research notebook that future investigation tiers can use as a closed world.
- **Question synthesis → essay synthesis**: turn patterns into higher‑order questions, then into a structured historical essay with evidence‑anchored paragraphs.
- **Low false positives**: prefer omission to hallucination; if evidence is thin, say so.

## What Works Today
- **Tier 0 exploration pipeline** reads corpora in structured strata (biographical, collection, etc.) and writes a persistent notebook.
- **Notebook data model** captures patterns, contradictions, entities, group indicators, and questions with evidence block IDs.
- **Inductive question filtering** now rejects individual‑factoid questions at notebook ingest.
- **Question synthesis** builds a hierarchy (grand narrative, theme questions, gaps, contradictions, group comparisons).
- **Recursive synthesis** produces a long‑form essay draft with evidence citations.
- **Notebook synthesis macros** provide theme‑level summaries from patterns/contradictions (new).
- **Evidence‑dense paragraphs** require multiple citations; weak leaves become explicit gaps (new).
- **Multi‑pass editor** (structure → evidence → style) uses a larger model (llama3.3) to improve prose (new).

## Current Limitations & Challenges
### 1) Question Quality Is Highly Sensitive to Batch Prompting
- Without strict inductive rules, batches generate “factoid” questions (single person, single attribute). 
- We now filter these out and also constrain the prompt, but notebooks created before this change are polluted.
- **Mitigation:** re-run Tier 0 with new prompts + filters.

### 2) Doc Cache Can Cause Silent Skips
- When cached doc blocks are empty, batches can become `cache_hit + no blocks`, skipping LLM analysis.
- This leads to runs that look "alive" but produce no patterns/questions.
- **Mitigation:** use `TIER0_DOC_CACHE_MODE=refresh` or clear cache collection.

### 3) Evidence Scarcity at Paragraph Level
- Leaf answers often return a single evidence item even when 10–20 doc IDs are available.
- This produced essays with weak support and generic claims.
- **Mitigation:** backfill evidence from doc IDs and enforce `TIER0_PARAGRAPH_MIN_EVIDENCE=2`.

### 4) Theme Names Can Be Generic
- If theme grouping is weak, the essay reads as “Theme 1/2/3.”
- We now inject notebook‑level theme macros and use theme names for section headers.
- Remaining issue: theme clustering quality still depends on the base question set.

### 5) LLM Variance + Validation Collapse
- The validator can collapse to uniform scores if the model is overloaded or prompts are too similar.
- This reduces the discriminating power of the pipeline.
- **Mitigation:** heuristic fallback scoring is in place; still needs more robust calibration.

### 6) Archive Metadata Gaps
- Many documents lack `year` or `document_type`, weakening temporal/genre stratification.
- **Mitigation:** strict date extraction helps, but temporal reasoning remains partial.

## What We’re Doing Now
- Running a **fresh 500‑doc Tier 0 pass** with inductive questions + cache refresh.
- Using the notebook synthesis macros + evidence briefs in the editor pipeline.
- Iterating the essay until it reads like a true historical synthesis.

## Practical Notes for Engineers
- Notebook artifacts live under: `app/logs/corpus_exploration/…/…_notebook.json`
- Tier 0 logs under: `app/logs/tier0/tier0_*.log`
- Essay synthesis outputs under: `app/logs/synthesis_matrix_*/synth_*.json`
- Force cache refresh: `TIER0_DOC_CACHE_MODE=refresh`
- Editor model: `TIER0_ESSAY_REVISE_MODEL=llama3.3:latest`

## Near‑Term Next Steps
1. Re-run a clean Tier 0 notebook with inductive batch prompts.
2. Verify that question counts are lower but higher quality (fewer factoids).
3. Validate that macro paragraphs reflect notebook patterns (not just question summaries).
4. Iterate editor prompts only after evidence density is consistently ≥2 citations.

---
If you want a single “demo‑quality” run, the current best path is: 
**fresh notebook (inductive + cache refresh) → question synthesis → recursive synthesis (3‑pass editor).**
