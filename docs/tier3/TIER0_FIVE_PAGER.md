# Tier 0: Notebook‑Driven Synthesis (Five‑Pager)

This five‑pager is intended for a software‑engineer peer. It summarizes what we set out to build, what we actually built, what it produces today (with concrete examples), and why the current outputs are falling short of the historical essay goal.

---

## 1) Goal (What We Hoped Would Work)

**Primary objective:**
Build a historian‑style, corpus‑first RAG system that reads thousands of archival documents, generates **research questions before a user query**, and produces a **synthetic historical essay** grounded in evidence — avoiding top‑k myopia and avoiding hallucinations.

**Operational goals:**
- **Systematic coverage** rather than top‑k retrieval.
- **Notebook as a closed world**: structured, persistent memory of patterns, contradictions, entities, and group indicators.
- **Inductive reasoning**: questions should reflect patterns across documents, not single‑person factoids.
- **Evidence density**: paragraphs should cite multiple documents, not single examples.
- **Historian‑style synthesis**: macro patterns → thematic paragraphs → long‑form narrative.

**Quality preference:**
Historians prefer **false negatives** (omission) over **false positives** (hallucinations). If evidence is thin, the system should skip a claim.

---

## 2) Algorithm (Structured Description)

This section describes the actual pipeline as it runs today, with **inputs, outputs, and data flow**. This is intended to be shareable and unambiguous.

### Key Terms (Defined)
- **Stratum / Strata**: A *slice* of the corpus created for systematic reading (e.g., all docs from one collection, or all docs for a specific person). Each stratum is split into smaller **batches** for LLM processing.
- **Batch**: A bounded set of documents (typically 10–50) read together in one LLM call.
- **Block**: A chunk of text with a stable `block_id` used for evidence citations.
- **Notebook**: The persistent JSON artifact that stores patterns, contradictions, entities, group indicators, and inductive questions.
- **Theme**: A higher‑level grouping of questions (e.g., medical surveillance, occupational mobility) derived during synthesis.

### A. Data Structures (Core Objects)
- **Document**: raw corpus document (MongoDB `documents` collection).
- **Block/Chunk**: text span created by `DocumentChunker` with a `block_id` that ties evidence to sources.
- **ResearchNotebook** (persistent JSON):
  - `patterns`: {pattern, type, confidence, evidence_block_ids}
  - `contradictions`: {claim_a, claim_b, source_a, source_b, context}
  - `entities`: {name, type, first_seen, context}
  - `group_indicators`: {group_type, label, evidence_blocks}
  - `questions`: inductive, cross‑document questions + evidence metadata
- **QuestionSynthesis**:
  - `themes`, `clusters`, `gaps`, `hierarchy`
  - `notebook_synthesis`: {theme_macros, evidence_briefs}
- **Essay** (recursive synthesis output):
  - `essay` text
  - `sections` per theme
  - `paragraph_gaps` (questions skipped due to low evidence)

### B. Pipeline Stages (Input → Output)

#### Stage 1 — Stratified Corpus Sampling
**Input:** full corpus  
**Output:** list of strata and batches  
**Method:** build stratification by available metadata:
- collection strata (always)
- biographical strata (top people by doc count)
- temporal + genre strata if metadata exists  

#### Stage 2 — Batch Analysis (Closed‑World)
**Input:** batch of documents + prior notebook summary  
**Output:** batch findings JSON  
**LLM Task:** extract patterns, contradictions, entities, group indicators, and inductive questions.  
**Constraints:** closed‑world rules + inductive question rules (no single‑person factoids).

#### Stage 3 — Notebook Integration
**Input:** batch findings  
**Output:** updated ResearchNotebook  
**Logic:** filter factoids, dedup questions, merge patterns/contradictions, update corpus stats.

#### Stage 4 — Question Synthesis
**Input:** ResearchNotebook  
**Output:** question hierarchy + themed agenda  
**Logic:**
1. Typed question generation (causal/comparative/etc.)
2. Adversarial validation (answerability/significance/specificity/evidence‑based)
3. Dedup + type diversity enforcement  

#### Stage 5 — Notebook‑Driven Theme Synthesis (New)
**Input:** notebook patterns/contradictions + themes  
**Output:** `notebook_synthesis`:
- **Theme macros**: 1–2 paragraphs that summarize each theme using notebook evidence
- **Evidence briefs**: patterns + contradictions + entities per theme for editor guidance

#### Stage 6 — Recursive Essay Synthesis
**Input:** question hierarchy + notebook synthesis + evidence store  
**Output:** structured essay  
**Logic:**
1. **Leaf answers**: each question answered from associated evidence docs
2. **Paragraph assembly**: enforce ≥2 citations per paragraph; insufficient evidence → gaps
3. **Editor passes** (3x): structure → evidence → style (llama3.3 default)

### C. Evidence Rules (Critical)
- **Paragraphs require ≥2 cited doc IDs**. If not, the paragraph is skipped and moved to gaps.
- Evidence is drawn from:
  - `evidence_doc_ids` attached to questions
  - DocumentStore snippets (backfill if needed)

### D. Outputs Produced
- **Notebook JSON** (archive‑level research notes)
- **Question synthesis JSON** (themes, clusters, gaps, hierarchy, macros)
- **Essay JSON** (sections + paragraph gaps + final essay text)

---

## 3) What It Produces Today (Examples)

### A. Notebook Output (from a 300‑doc run)

**Example pattern**
```
"pattern": "The Baltimore & Ohio Railroad Company has a Relief Department that handles
certificates for employees' disability and return to duty.",
"time_range": "1916-1922",
"type": "organizational",
"confidence": "high",
"evidence_count": 2
```

**Example contradiction**
```
"claim_a": "J. C. Duval reported sick on Dec 17, 1906.",
"claim_b": "James P. Duvall was able to return to duty on Sept 24, 1918.",
"context": "Different names (J. C. Duval vs James P. Duvall) are used for what might be the same person."
```

**Example question (problematic — too person‑specific)**
```
"How did the circumstances and medical protocols surrounding J. C. Duval’s reported sickness…",
"type": "comparative",
"score": 78
```

This shows the core issue: even when structured, many questions drift toward micro‑biography instead of cross‑document pattern analysis.

### B. Essay Output (Current State)

The essay **does assemble sections**, but often suffers from:
- **Sparse evidence per paragraph** (one citation despite multiple doc IDs available).
- **Generic macro claims** (“records are incomplete” repeated verbatim).
- **Topic drift** due to weak evidence‑question alignment.

Despite these flaws, the framework now has:
- **Notebook macro paragraphs** (theme‑level synthesis).
- **Evidence briefs** (editor guidance).
- **Paragraph evidence enforcement** (skip weak paragraphs → gaps section).

---

## 4) Why It’s Not Yet Producing True Historical Essays

### 1) Retrieval/Evidence Alignment Is the Bottleneck
Leaf questions often have **doc IDs but weak content** or **no blocks** after caching. The writer model can only cite what it sees. This produces generic paragraphs even when the corpus is large.

### 2) The Notebook Is Still Polluted
Earlier runs allowed individual‑factoid questions. Even with filters, **legacy notebooks remain polluted**. If you synthesize from old notebooks, you get weak themes.

### 3) Evidence Density Constraint Is Correct — But Exposes Weakness
Requiring ≥2 citations makes the essay more honest, but also exposes that many questions are not answerable with the retrieved evidence.

### 4) LLM Score Collapse
Validation scores have occasionally collapsed to uniform values (e.g., all 60), implying the validator is overloaded or the prompt is too repetitive. That reduces discrimination and allows mediocre questions through.

### 5) Metadata Gaps Limit Stratification
Without reliable `year` or `document_type`, stratification falls back to biographical/collection sampling. That can skew toward person‑level questions.

---

## 5) Current Fixes and Active Mitigations

- **Inductive batch prompt + filters**: rejects single‑person “factoid” questions.
- **Cache refresh mode**: `TIER0_DOC_CACHE_MODE=refresh` to avoid empty cached blocks.
- **Notebook macros + evidence briefs**: theme synthesis before question paragraphs.
- **Evidence backfill**: snippets pulled directly from DocumentStore to reach ≥2 citations.
- **Multi‑pass editor**: structure + evidence + style (llama3.3).

---

## 6) What We Need Next (Theories of Failure → Fixes)

### A. Evidence Retrieval Must Be Stronger Than Question Generation
**Problem:** We generate questions faster than we can answer them.

**Fix:** enforce evidence sufficiency before a question is admitted, or use **retrieval‑first question generation** (derive questions from evidence clusters only).

### B. “Notebook‑Only” Macro Essays Before Q&A
**Problem:** question‑level paragraphs create a brittle essay.

**Fix:** generate a **macro essay** from patterns/contradictions alone, then integrate micro‑questions as evidence expansion.

### C. Replace Single‑Paragraph Leaf Answers With Multi‑Evidence Summaries
**Problem:** leaf answers are one‑shot.

**Fix:** do multi‑doc summarization per question (cluster evidence chunks → summarize cluster → compose paragraph).

### D. Stronger Model for Essay Composition
Large essay synthesis likely needs a **long‑context model** with a historian‑style prompt. The current step is underpowered for the length and complexity.

---

## 7) Where to Inspect Artifacts

- **Notebook JSONs**: `app/logs/corpus_exploration/exploration_*/**_notebook.json`
- **Tier 0 logs**: `app/logs/tier0/tier0_*.log`
- **Essay outputs**: `app/logs/synthesis_matrix_*/synth_*.json`

---

## Bottom Line
We achieved a **novel corpus‑first architecture** with a working notebook, typed questions, and theme synthesis. What’s failing is **evidence alignment and synthesis quality**, not the overall pipeline structure. The path forward is to tighten evidence retrieval and shift more synthesis effort to the notebook‑level macro patterns before the essay stage.
