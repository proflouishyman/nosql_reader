# app/historian_agent/corpus_explorer.py
# Created: 2026-02-05
# Purpose: Tier 0 - Systematic corpus exploration with Notebook-LLM principles

"""
Corpus Explorer - Tier 0 of the historian agent system.

Notebook-LLM improvements:
- Closed-world batch analysis (no outside knowledge)
- Semantic chunking (document objects with logical blocks)
- Code-side batch statistics (LLM doesn't fabricate stats)
- Evidence validation against block IDs
"""

from __future__ import annotations

import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from rag_base import DocumentStore, MongoDBConnection, debug_print, count_tokens
from llm_abstraction import LLMClient
from config import APP_CONFIG

from historian_agent.research_notebook import ResearchNotebook
from historian_agent.stratification import CorpusStratifier, StratumReader, Stratum
from historian_agent.tier0_utils import Tier0Logger, Heartbeat, parse_llm_json, save_with_timestamp
from historian_agent.question_pipeline import QuestionGenerationPipeline
from historian_agent.question_synthesis import QuestionSynthesizer


# ============================================================================
# Prompts
# ============================================================================

# Tightened prompt instructions to prioritize cross-document inductive outputs over factoids.
BATCH_ANALYSIS_PROMPT = """You are a historian systematically reading archival documents.

CLOSED-WORLD RULES:
- Use ONLY the documents provided below.
- Do NOT use outside knowledge or assumptions.
- If a fact is not in the documents, do not include it.
- You may compare with PRIOR KNOWLEDGE to spot contradictions, but do NOT add facts that are not in the new documents.
- Historians prefer false negatives to false positives: only record group indicators when explicitly stated.
- Prioritize USER RESEARCH LENS topics when evidence exists, but do NOT infer missing groups or identities.

PRIOR KNOWLEDGE (summary from previous batches):
{prior_knowledge}

USER RESEARCH LENS (prioritized topics for this run):
{research_lens}

DOCUMENTS (closed-world JSON objects):
{documents_json}

TASK: Extract findings from this batch.

Return ONLY valid JSON with this schema:
{{
  "entities": [
    {{"name": str, "type": "person|organization|place", "first_seen": "block_id", "context": str}}
  ],
  "patterns": [
    {{"pattern": str, "type": str, "evidence_blocks": ["block_id"], "confidence": "low|medium|high", "time_range": str}}
  ],
  "contradictions": [
    {{"claim_a": str, "claim_b": str, "source_a": "block_id", "source_b": "block_id", "context": str}}
  ],
  "group_indicators": [
    {{"group_type": "race|gender|class|ethnicity|national_origin|occupation", "label": str, "evidence_blocks": ["block_id"], "confidence": "low|medium|high"}}
  ],
  "questions": [
    {{"question": str, "why_interesting": str, "evidence_needed": str, "related_entities": [], "time_window": str}}
  ],
  "temporal_events": {{"year": [str]}}
}}

PATTERN RULES:
A "pattern" is an ANALYTICAL OBSERVATION about what the documents reveal, not repeated text.
Do NOT record form headings, standard questions, or boilerplate text as patterns.
- BAD: "Have you ever been disabled through any other illness?" (form question)
- BAD: "striking by engine no 2056" (single event detail)
- BAD: "I desire to retain my membership for natural death benefits" (boilerplate)
- GOOD: "Multiple employees in the Maintenance of Way department report eye injuries from foreign matter, suggesting systematic lack of eye protection."
- GOOD: "Medical examiner reports consistently describe injuries in passive voice, avoiding attribution of fault to the company."
- GOOD: "Disability claims from the Cumberland Division show longer processing times than other divisions."

CONTRADICTION RULES:
A contradiction is when TWO SOURCES MAKE CONFLICTING FACTUAL CLAIMS about the same event, person, or situation.
Two different document types about different events are NOT contradictions.
- BAD: "Doc A is an injury report, Doc B is a death benefit application." (different forms)
- BAD: "Duval was sick Dec 1906; Duvall returned to duty Sept 1918." (12 years apart, not conflicting)
- GOOD: "Doc A says employee was injured on June 5, Doc B says June 12." (same event, conflicting dates)
- GOOD: "Surgeon reports employee fully recovered, but subsequent letter from employee states ongoing disability."
If uncertain, DO NOT include it. Prefer false negatives.

QUESTION RULES (CRITICAL):
Questions must reflect INDUCTIVE REASONING across multiple documents.
You are a social historian looking for PATTERNS, SYSTEMS, and STRUCTURES.

NEVER generate questions about a single individual's attributes.
NEVER generate "What is [person]'s [attribute]?" questions.
NEVER generate lookup/retrieval questions answerable from one document.

GOOD questions require evidence from MULTIPLE documents:
- "What types of injuries were most common among track laborers in this period?"
- "How did the Relief Department's benefit approval process vary by occupation?"
- "Were certain divisions disproportionately represented in disability claims?"
- "How did medical examiner reporting practices change over time?"
- "What role did the superintendent play in adjudicating contested claims?"

BAD questions (DO NOT GENERATE):
- "What is Thomas Freeland's disability?" (single person, single fact)
- "What is the date of birth of Edward Collins?" (lookup)
- "What is the occupation of James Duval?" (single entity attribute)
- "Who is W. J. Dudley?" (entity identification, not research)

If this batch reveals no cross-document patterns worth questioning, return EMPTY questions array: []
Better to return NO questions than individual-level factoid questions.

Evidence must reference ONLY valid block_id values from DOCUMENTS.
Focus on discoveries: new patterns, contradictions, and cross-document questions.
"""

INDUCTIVE_SYSTEM_MESSAGE = (
    "You are a social historian performing INDUCTIVE analysis of archival documents. "
    "Your job is to find patterns ACROSS documents, not to catalog facts about individuals. "
    "Think structurally: what do these documents reveal about SYSTEMS, INSTITUTIONS, and GROUP EXPERIENCES? "
    "Never generate questions about a single person's attributes — "
    "always reason at the level of occupations, departments, time periods, or policies."
)

BATCH_REPAIR_PROMPT = """You are auditing a batch analysis for evidence alignment.

CLOSED-WORLD RULES:
- Use ONLY the documents provided below.
- Every evidence reference must be a valid block_id in DOCUMENTS.
- Remove any items that cannot be supported by the documents.

DOCUMENTS (closed-world JSON objects):
{documents_json}

PREVIOUS OUTPUT (may contain invalid evidence IDs):
{previous_output}

TASK: Return a corrected JSON object with the same schema, using ONLY valid block_id references.
Return ONLY valid JSON.
"""

CORPUS_MAP_PROMPT = """You are a historian who has completed a systematic reading of an archival corpus.

READING SUMMARY:
{notebook_summary}

TASK: Write archive orientation notes.
Provide 5-7 observations about:
1. Scope (time period, activities documented)
2. Voices (whose perspectives appear or are missing)
3. Biases (selection gaps, archival imbalance)
4. Surprises (unexpected patterns or contradictions)
5. Research potential (what questions this archive supports)

Write 2-3 sentences per observation. Be specific.
"""

QUESTION_GENERATION_PROMPT = """You are a historian who has systematically read {docs_read} documents.

FINDINGS:
{notebook_summary}

USER RESEARCH LENS (prioritized topics for question generation):
{research_lens}

TASK: Generate research questions based on patterns and contradictions.
Return a JSON array of questions, each with:
- question
- why_interesting
- approach
- entities_involved
"""


# ============================================================================
# Corpus Explorer
# ============================================================================

class CorpusExplorer:
    """Tier 0: Systematic corpus exploration with cumulative synthesis."""

    def __init__(self) -> None:
        debug_print("Initializing CorpusExplorer (Tier 0)")

        self.doc_store = DocumentStore()
        self.llm = LLMClient()
        self.stratifier = CorpusStratifier()
        self.reader = StratumReader()
        self.notebook = ResearchNotebook()
        self.logger = Tier0Logger(Path(APP_CONFIG.tier0.tier0_log_dir), "tier0")
        self.question_pipeline = QuestionGenerationPipeline()
        self.question_synthesizer = QuestionSynthesizer()
        self._last_question_batch = None
        self._research_lens: List[str] = []  # Added run-scoped lens so prompts can prioritize historian-defined interests.
        self._doc_cache_indexed = False
        self._ensure_doc_cache_indexes()

    def explore(
        self,
        strategy: Optional[str] = None,
        total_budget: Optional[int] = None,
        year_range: Optional[Tuple[int, int]] = None,
        save_notebook: Optional[bool] = None,
        research_lens: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute systematic corpus exploration."""
        start_time = time.time()

        config = APP_CONFIG.tier0
        strategy = strategy or config.exploration_strategy
        total_budget = total_budget or config.exploration_budget
        save_notebook = config.notebook_auto_save if save_notebook is None else save_notebook
        self._research_lens = self._normalize_research_lens(research_lens)  # Added normalization to keep lens input predictable across API/UI variants.

        self.logger.log("start", f"strategy={strategy} budget={total_budget}")
        if self._research_lens:
            self.logger.log("lens", "; ".join(self._research_lens))

        strata = self._build_strata(strategy, total_budget, year_range)
        self.logger.log("stratification", f"{len(strata)} batches")

        for idx, stratum in enumerate(strata):
            self.logger.log("batch", f"{idx + 1}/{len(strata)} {stratum.label}")
            try:
                self._process_stratum(stratum)
            except Exception as exc:
                debug_print(f"Error processing {stratum.label}: {exc}")
                self.logger.log("batch_error", f"{stratum.label}: {exc}", level="WARN")
                continue

        corpus_map = self._generate_corpus_map()
        questions = self._generate_research_questions()
        question_synthesis = self._generate_question_synthesis()
        question_quality_gate = (
            self._last_question_batch.quality_gate
            if self._last_question_batch is not None
            else None
        )  # Added gate metadata passthrough so reports explain why synthesis may be skipped.

        report = {
            "corpus_map": corpus_map,
            "questions": questions,
            "question_synthesis": question_synthesis,
            "patterns": self._export_patterns(),
            "entities": self._export_entities(),
            "contradictions": self._export_contradictions(),
            "group_indicators": self._export_group_indicators(),
            "notebook_summary": self.notebook.get_summary(),
            "exploration_metadata": {
                "strategy": strategy,
                "total_budget": total_budget,
                "documents_read": self.notebook.corpus_map["total_documents_read"],
                "batches_processed": self.notebook.corpus_map["batches_processed"],
                "duration_seconds": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "research_lens": list(self._research_lens),  # Added to make lens usage explicit in persisted run metadata.
                "question_quality_gate": question_quality_gate,
            },
        }

        if save_notebook:
            notebook_path = save_with_timestamp(
                content=self.notebook.to_dict(),
                base_dir=Path(config.notebook_save_dir),
                filename_prefix="notebook",
                subdirectory=f"exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            report["notebook_path"] = str(notebook_path)
            self.logger.log("notebook", f"saved {notebook_path}")

        self._persist_run(report, strategy, total_budget, year_range)

        self.logger.log(
            "complete",
            f"docs={report['exploration_metadata']['documents_read']} questions={len(questions)}",
        )

        return report

    def _persist_run(
        self,
        report: Dict[str, Any],
        strategy: str,
        total_budget: int,
        year_range: Optional[Tuple[int, int]],
    ) -> None:
        try:
            mongo = MongoDBConnection()
            runs = mongo.get_collection(APP_CONFIG.tier0.runs_collection)

            tier0_cfg = APP_CONFIG.tier0
            profile_models = {
                name: profile.get("model")
                for name, profile in APP_CONFIG.llm_profiles.items()
            }

            run_doc: Dict[str, Any] = {
                "created_at": datetime.now().isoformat(),
                "strategy": strategy,
                "total_budget": total_budget,
                "year_range": year_range,
                "models": profile_models,
                "embedding": {
                    "provider": tier0_cfg.synthesis_embed_provider,
                    "model": tier0_cfg.synthesis_embed_model,
                },
                "results": report,
                "notebook_path": report.get("notebook_path"),
            }

            if tier0_cfg.runs_store_notebook:
                notebook_payload = self.notebook.to_dict()
                run_doc["notebook"] = notebook_payload
            else:
                run_doc["notebook"] = None

            runs.insert_one(run_doc)
            self.logger.log("run_persisted", f"saved to {tier0_cfg.runs_collection}")
        except Exception as exc:
            self.logger.log("run_persist_error", str(exc), level="WARN")

    def _build_strata(
        self,
        strategy: str,
        total_budget: int,
        year_range: Optional[Tuple[int, int]],
    ) -> List[Stratum]:
        if APP_CONFIG.tier0.full_corpus:
            return self.stratifier.full_corpus_stratification(total_budget=total_budget)
        if strategy == "temporal" and year_range:
            return self.stratifier.temporal_stratification(
                year_range=year_range,
                docs_per_year=min(50, total_budget // 20),
            )
        return self.stratifier.build_comprehensive_strategy(
            total_budget=total_budget,
            strategy=strategy,
        )

    def _normalize_research_lens(self, research_lens: Optional[Any]) -> List[str]:
        """Normalize optional lens input into a short deduplicated list of textual priorities."""
        if research_lens is None:
            return []
        if isinstance(research_lens, str):
            candidates = [p.strip() for p in research_lens.replace("\n", ",").split(",")]
        elif isinstance(research_lens, (list, tuple, set)):
            candidates = [str(item).strip() for item in research_lens]
        else:
            candidates = [str(research_lens).strip()]

        cleaned: List[str] = []
        seen = set()
        for candidate in candidates:
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(candidate)
        return cleaned[:10]

    def _format_research_lens_for_prompt(self) -> str:
        """Return lens priorities in prompt-ready bullet form."""
        if not self._research_lens:
            return "- (none)"
        return "\n".join(f"- {item}" for item in self._research_lens)

    def _process_stratum(self, stratum: Stratum) -> None:
        sub_batch_size = max(1, APP_CONFIG.tier0.sub_batch_docs)
        cache_mode = APP_CONFIG.tier0.doc_cache_mode.lower()

        if stratum.stream:
            batch_index = 0
            cache_batch_index = 0
            for sub_docs in self.reader.iter_stratum_batches(stratum, sub_batch_size):
                combined = list(sub_docs)
                cache_key = self._doc_cache_key()
                cached_map = self._fetch_doc_cache(
                    [str(doc.get("_id")) for doc in combined],
                    cache_key,
                )
                if cached_map:
                    cache_batch_index += 1
                    cache_label = f"{stratum.label} [cache {cache_batch_index}]"
                    self._integrate_cached_docs(combined, cached_map, cache_key, cache_label)

                if cache_mode == "rebuild":
                    continue

                uncached_docs = [
                    doc for doc in combined if str(doc.get("_id")) not in cached_map
                ] if self._doc_cache_should_use() else combined

                while uncached_docs:
                    batch_index += 1
                    doc_objects, consumed = self.reader.build_document_objects(
                        uncached_docs,
                        APP_CONFIG.tier0.batch_max_chars,
                    )
                    if consumed <= 0:
                        consumed = 1

                    if not doc_objects:
                        self.logger.log("batch_skip", f"{stratum.label} sub-batch {batch_index} no blocks")
                        uncached_docs = uncached_docs[consumed:]
                        continue

                    stats = self.reader.compute_object_stats(doc_objects)
                    findings = self._analyze_batch(doc_objects)
                    findings["stats"] = stats

                    batch_label = f"{stratum.label} [{batch_index}]"
                    self.notebook.integrate_batch_findings(findings, batch_label)
                    self.logger.log(
                        "batch_done",
                        f"{batch_label} entities={len(findings.get('entities', []))} patterns={len(findings.get('patterns', []))}",
                    )
                    self._persist_doc_cache(findings, doc_objects, cache_key)
                    uncached_docs = uncached_docs[consumed:]
            return

        docs = self.reader.read_stratum(stratum)
        if not docs:
            self.logger.log("batch_skip", f"{stratum.label} empty")
            return

        total_docs = len(docs)
        processed = 0
        batch_index = 0
        cache_batch_index = 0

        while processed < total_docs:
            batch_index += 1
            sub_docs = docs[processed: processed + sub_batch_size]
            cache_key = self._doc_cache_key()
            cached_map = self._fetch_doc_cache(
                [str(doc.get("_id")) for doc in sub_docs],
                cache_key,
            )
            if cached_map:
                cache_batch_index += 1
                cache_label = f"{stratum.label} [cache {cache_batch_index}]"
                self._integrate_cached_docs(sub_docs, cached_map, cache_key, cache_label)

            if cache_mode == "rebuild":
                processed += len(sub_docs)
                continue

            uncached_docs = [
                doc for doc in sub_docs if str(doc.get("_id")) not in cached_map
            ] if self._doc_cache_should_use() else sub_docs

            doc_objects, consumed = self.reader.build_document_objects(
                uncached_docs,
                APP_CONFIG.tier0.batch_max_chars,
            )
            if consumed <= 0:
                consumed = 1
            processed += len(sub_docs)

            if not doc_objects:
                self.logger.log("batch_skip", f"{stratum.label} sub-batch {batch_index} no blocks")
                continue

            stats = self.reader.compute_object_stats(doc_objects)
            findings = self._analyze_batch(doc_objects)
            findings["stats"] = stats

            batch_label = f"{stratum.label} [{batch_index}]"
            self.notebook.integrate_batch_findings(findings, batch_label)
            self.logger.log(
                "batch_done",
                f"{batch_label} entities={len(findings.get('entities', []))} patterns={len(findings.get('patterns', []))}",
            )
            self._persist_doc_cache(findings, doc_objects, cache_key)

    def _analyze_batch(self, doc_objects: List[Any]) -> Dict[str, Any]:
        prior_knowledge = self.notebook.format_for_llm_context()
        documents_json = self.reader.format_document_objects_for_llm(doc_objects)
        total_blocks = sum(len(doc.blocks) for doc in doc_objects)

        prompt = BATCH_ANALYSIS_PROMPT.format(
            prior_knowledge=prior_knowledge,
            research_lens=self._format_research_lens_for_prompt(),
            documents_json=documents_json,
        )

        self.logger.log("llm", f"batch tokens~{count_tokens(prompt)}")

        heartbeat_detail = f"batch docs={len(doc_objects)} blocks={total_blocks}"
        with Heartbeat(self.logger, "llm", heartbeat_detail, APP_CONFIG.tier0.heartbeat_seconds):
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": INDUCTIVE_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                profile=APP_CONFIG.tier0.batch_profile,
                temperature=0.2,
                timeout=APP_CONFIG.tier0.llm_timeout,
            )

        try:
            findings = parse_llm_json(response.content, default={}) if response.success else {}
            if not isinstance(findings, dict):
                findings = {}
            findings = self._normalize_findings(findings, doc_objects)
        except Exception as exc:
            self.logger.log("batch_parse_error", str(exc), level="WARN")
            findings = {
                "entities": [],
                "patterns": [],
                "contradictions": [],
                "questions": [],
                "temporal_events": {},
            }

        if self._needs_repair(findings, len(doc_objects)):
            attempts = 0
            while attempts < APP_CONFIG.tier0.repair_attempts:
                attempts += 1
                repair_prompt = BATCH_REPAIR_PROMPT.format(
                    documents_json=documents_json,
                    previous_output=json.dumps(findings, ensure_ascii=True),
                )
                repair_detail = f"repair attempt={attempts} docs={len(doc_objects)} blocks={total_blocks}"
                with Heartbeat(self.logger, "repair", repair_detail, APP_CONFIG.tier0.heartbeat_seconds):
                    repair_response = self.llm.generate(
                        messages=[
                            {"role": "system", "content": "You audit evidence references."},
                            {"role": "user", "content": repair_prompt},
                        ],
                        profile="verifier",
                        temperature=0.0,
                        timeout=APP_CONFIG.tier0.llm_timeout,
                    )
                repaired = parse_llm_json(repair_response.content, default={}) if repair_response.success else {}
                repaired = self._normalize_findings(repaired, doc_objects)
                if not self._needs_repair(repaired, len(doc_objects)):
                    findings = repaired
                    break

        return findings

    def _needs_repair(self, findings: Dict[str, Any], doc_count: int) -> bool:
        if not APP_CONFIG.tier0.strict_closed_world:
            return False
        if doc_count < APP_CONFIG.tier0.repair_min_docs:
            return False
        if len(findings.get("entities", [])) < APP_CONFIG.tier0.min_entities_per_batch:
            return True
        if len(findings.get("patterns", [])) < APP_CONFIG.tier0.min_patterns_per_batch:
            return True
        return False

    def _normalize_findings(self, findings: Dict[str, Any], doc_objects: List[Any]) -> Dict[str, Any]:
        clean = {
            "entities": [],
            "patterns": [],
            "contradictions": [],
            "group_indicators": [],
            "questions": [],
            "temporal_events": {},
        }

        doc_ids = {doc.doc_id for doc in doc_objects}
        block_to_doc = {}
        for doc in doc_objects:
            for block in doc.blocks:
                block_to_doc[block.block_id] = doc.doc_id

        valid_blocks = set(block_to_doc.keys())

        for entity in findings.get("entities", []):
            if not isinstance(entity, dict):
                continue
            name = entity.get("name")
            first_seen = entity.get("first_seen")
            if not name or not first_seen:
                continue

            if first_seen in valid_blocks:
                entity["first_seen_block"] = first_seen
                entity["first_seen"] = block_to_doc[first_seen]
            elif first_seen not in doc_ids:
                continue

            last_seen = entity.get("last_seen") or entity.get("first_seen")
            if last_seen in valid_blocks:
                entity["last_seen_block"] = last_seen
                entity["last_seen"] = block_to_doc[last_seen]
            entity["last_seen"] = entity.get("last_seen") or entity.get("first_seen")

            entity["type"] = entity.get("type") or entity.get("entity_type") or "unknown"
            if "context" not in entity:
                entity["context"] = ""
            clean["entities"].append(entity)

        for pattern in findings.get("patterns", []):
            if not isinstance(pattern, dict):
                continue
            if not pattern.get("pattern") and not pattern.get("pattern_text"):
                continue

            raw_blocks = pattern.get("evidence_blocks") or pattern.get("evidence") or []
            valid_block_ids = [b for b in raw_blocks if b in valid_blocks]
            if not valid_block_ids:
                continue

            pattern["evidence_blocks"] = valid_block_ids
            pattern["evidence"] = sorted({block_to_doc[b] for b in valid_block_ids})
            pattern["pattern"] = pattern.get("pattern") or pattern.get("pattern_text")
            pattern["type"] = pattern.get("type") or pattern.get("pattern_type") or "unknown"
            pattern["confidence"] = pattern.get("confidence") or "low"
            clean["patterns"].append(pattern)

        for contra in findings.get("contradictions", []):
            if not isinstance(contra, dict):
                continue
            source_a = contra.get("source_a")
            source_b = contra.get("source_b")
            if not source_a or not source_b:
                continue
            if source_a not in valid_blocks and source_a not in doc_ids:
                continue
            if source_b not in valid_blocks and source_b not in doc_ids:
                continue
            if not contra.get("claim_a") or not contra.get("claim_b"):
                continue
            contra.setdefault("context", "")
            clean["contradictions"].append(contra)

        for indicator in findings.get("group_indicators", []):
            if not isinstance(indicator, dict):
                continue
            group_type = indicator.get("group_type") or indicator.get("type")
            label = indicator.get("label") or indicator.get("value")
            if not group_type or not label:
                continue
            raw_blocks = indicator.get("evidence_blocks") or indicator.get("evidence") or []
            valid_block_ids = [b for b in raw_blocks if b in valid_blocks]
            if not valid_block_ids:
                continue
            indicator["group_type"] = str(group_type)
            indicator["label"] = str(label)
            indicator["evidence_blocks"] = valid_block_ids
            indicator.setdefault("confidence", "low")
            clean["group_indicators"].append(indicator)

        for question in findings.get("questions", []):
            if not isinstance(question, dict):
                continue
            if not question.get("question"):
                continue
            question.setdefault("why_interesting", "")
            question.setdefault("evidence_needed", "")
            question.setdefault("time_window", "")
            clean["questions"].append(question)

        temporal_events = findings.get("temporal_events") or {}
        if isinstance(temporal_events, dict):
            cleaned_temporal = {}
            for year, events in temporal_events.items():
                if not isinstance(events, list):
                    continue
                cleaned_temporal[str(year)] = [str(e) for e in events if e]
            clean["temporal_events"] = cleaned_temporal

        return clean

    def _doc_cache_should_use(self) -> bool:
        cfg = APP_CONFIG.tier0
        return cfg.doc_cache_enabled and cfg.doc_cache_mode.lower() in {"use", "rebuild"}

    def _doc_cache_is_rebuild(self) -> bool:
        cfg = APP_CONFIG.tier0
        return cfg.doc_cache_enabled and cfg.doc_cache_mode.lower() == "rebuild"

    def _ensure_doc_cache_indexes(self) -> None:
        cfg = APP_CONFIG.tier0
        if not cfg.doc_cache_enabled or self._doc_cache_indexed:
            return
        try:
            mongo = MongoDBConnection()
            coll = mongo.get_collection(cfg.doc_cache_collection)
            coll.create_index(
                [("doc_id", 1), ("cache_key", 1)],
                unique=True,
                name="doc_id_cache_key_unique",
            )
            coll.create_index(
                [("cache_key", 1), ("created_at", -1)],
                name="cache_key_created_at",
            )
            self._doc_cache_indexed = True
        except Exception as exc:
            self.logger.log("cache_index_error", str(exc), level="WARN")

    def _doc_cache_key(self) -> str:
        cfg = APP_CONFIG.tier0
        profile = cfg.batch_profile
        model = APP_CONFIG.llm_profiles.get(profile, {}).get("model", "unknown")
        chunk_cfg = f"{int(cfg.semantic_chunking)}:{cfg.block_max_chars}:{cfg.max_blocks_per_doc}"
        prompt_version = cfg.doc_cache_prompt_version
        lens_scope = ",".join(sorted(str(item).strip().lower() for item in self._research_lens if str(item).strip()))  # Added lens scope so cache reuse stays valid per research focus and avoids cross-lens contamination.
        raw = f"{profile}:{model}:{prompt_version}:{chunk_cfg}"
        if lens_scope:
            raw = f"{raw}:lens={lens_scope}"  # Preserve legacy no-lens cache keys while isolating focused-lens cache entries.
        return hashlib.sha256(raw.encode()).hexdigest()

    def _fetch_doc_cache(self, doc_ids: List[str], cache_key: str) -> Dict[str, Dict[str, Any]]:
        cfg = APP_CONFIG.tier0
        if not cfg.doc_cache_enabled:
            return {}
        if cfg.doc_cache_mode.lower() in {"refresh", "off"}:
            return {}

        if not doc_ids:
            return {}

        mongo = MongoDBConnection()
        coll = mongo.get_collection(cfg.doc_cache_collection)
        cursor = coll.find(
            {"doc_id": {"$in": doc_ids}, "cache_key": cache_key},
            {"doc_id": 1, "findings": 1, "cache_key": 1},
        )
        return {doc["doc_id"]: doc for doc in cursor if doc.get("doc_id")}

    def _integrate_cached_docs(
        self,
        docs: List[Dict[str, Any]],
        cached_map: Dict[str, Dict[str, Any]],
        cache_key: str,
        batch_label: str,
    ) -> None:
        if not self._doc_cache_should_use():
            return

        cached_docs = []
        for doc in docs:
            doc_id = str(doc.get("_id"))
            if doc_id in cached_map:
                cached_docs.append(doc)

        if not cached_docs:
            return

        stats = self.reader.compute_batch_stats(cached_docs)
        merged = {
            "entities": [],
            "patterns": [],
            "contradictions": [],
            "group_indicators": [],
            "questions": [],
            "temporal_events": {},
            "stats": stats,
        }

        for doc in cached_docs:
            doc_id = str(doc.get("_id"))
            cached = cached_map.get(doc_id) or {}
            findings = cached.get("findings") or {}
            merged["entities"].extend(findings.get("entities", []))
            merged["patterns"].extend(findings.get("patterns", []))
            merged["contradictions"].extend(findings.get("contradictions", []))
            merged["group_indicators"].extend(findings.get("group_indicators", []))

        self.notebook.integrate_batch_findings(merged, batch_label)

        missing = [str(doc.get("_id")) for doc in docs if str(doc.get("_id")) not in cached_map]
        if missing and self._doc_cache_is_rebuild():
            self.logger.log("cache_miss", f"{len(missing)} docs missing for {cache_key[:8]}")
        self.logger.log("cache_hit", f"{len(cached_docs)} docs {cache_key[:8]}")

    def _persist_doc_cache(self, findings: Dict[str, Any], doc_objects: List[Any], cache_key: str) -> None:
        cfg = APP_CONFIG.tier0
        if not cfg.doc_cache_enabled:
            return
        if cfg.doc_cache_mode.lower() in {"off", "rebuild"}:
            return

        per_doc = self._split_findings_by_doc(findings, doc_objects)
        if not per_doc:
            return

        mongo = MongoDBConnection()
        coll = mongo.get_collection(cfg.doc_cache_collection)
        profile = cfg.batch_profile
        model = APP_CONFIG.llm_profiles.get(profile, {}).get("model", "unknown")

        for doc_id, doc_findings in per_doc.items():
            payload = {
                "doc_id": doc_id,
                "cache_key": cache_key,
                "profile": profile,
                "model": model,
                "prompt_version": cfg.doc_cache_prompt_version,
                "created_at": datetime.now().isoformat(),
                "findings": doc_findings,
            }
            coll.update_one(
                {"doc_id": doc_id, "cache_key": cache_key},
                {"$set": payload},
                upsert=True,
            )

    def _split_findings_by_doc(
        self,
        findings: Dict[str, Any],
        doc_objects: List[Any],
    ) -> Dict[str, Dict[str, Any]]:
        block_to_doc = {}
        doc_ids = {doc.doc_id for doc in doc_objects}
        for doc in doc_objects:
            for block in doc.blocks:
                block_to_doc[block.block_id] = doc.doc_id

        per_doc: Dict[str, Dict[str, Any]] = {}
        for doc_id in doc_ids:
            per_doc[doc_id] = {
                "entities": [],
                "patterns": [],
                "contradictions": [],
                "group_indicators": [],
            }

        for entity in findings.get("entities", []):
            block_id = entity.get("first_seen_block")
            doc_id = block_to_doc.get(block_id) or entity.get("first_seen")
            if doc_id in per_doc:
                per_doc[doc_id]["entities"].append(entity)

        for pattern in findings.get("patterns", []):
            blocks = pattern.get("evidence_blocks") or []
            doc_blocks: Dict[str, List[str]] = {}
            for block_id in blocks:
                doc_id = block_to_doc.get(block_id)
                if not doc_id:
                    continue
                doc_blocks.setdefault(doc_id, []).append(block_id)
            for doc_id, block_ids in doc_blocks.items():
                if doc_id not in per_doc:
                    continue
                per_pattern = dict(pattern)
                per_pattern["evidence_blocks"] = block_ids
                per_pattern["evidence"] = [doc_id]
                per_doc[doc_id]["patterns"].append(per_pattern)

        for indicator in findings.get("group_indicators", []):
            blocks = indicator.get("evidence_blocks") or []
            doc_blocks: Dict[str, List[str]] = {}
            for block_id in blocks:
                doc_id = block_to_doc.get(block_id)
                if not doc_id:
                    continue
                doc_blocks.setdefault(doc_id, []).append(block_id)
            for doc_id, block_ids in doc_blocks.items():
                if doc_id not in per_doc:
                    continue
                per_indicator = dict(indicator)
                per_indicator["evidence_blocks"] = block_ids
                per_doc[doc_id]["group_indicators"].append(per_indicator)

        for contra in findings.get("contradictions", []):
            source_a = contra.get("source_a")
            source_b = contra.get("source_b")
            doc_id = block_to_doc.get(source_a) or (source_a if source_a in doc_ids else None)
            if not doc_id:
                doc_id = block_to_doc.get(source_b) or (source_b if source_b in doc_ids else None)
            if doc_id in per_doc:
                per_doc[doc_id]["contradictions"].append(contra)

        return per_doc

    def _generate_corpus_map(self) -> Dict[str, Any]:
        notebook_summary = json.dumps(self.notebook.get_summary(), indent=2)
        prompt = CORPUS_MAP_PROMPT.format(notebook_summary=notebook_summary)

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian writing archive notes."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.3,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        archive_notes = response.content if response.success else "Failed to generate archive notes."

        return {
            "statistics": self.notebook.corpus_map,
            "archive_notes": archive_notes,
        }

    def _generate_research_questions(self) -> List[Dict[str, Any]]:
        try:
            batch = self.question_pipeline.generate(
                self.notebook,
                research_lens=self._research_lens,
            )  # Added lens pass-through so pipeline ranking can prioritize user-selected historian themes.
            self._last_question_batch = batch
            return batch.to_list()
        except Exception as exc:
            debug_print(f"Question pipeline failed, falling back to simple prompt: {exc}")

        notebook_summary = json.dumps(self.notebook.get_summary(), indent=2)
        docs_read = self.notebook.corpus_map["total_documents_read"]

        prompt = QUESTION_GENERATION_PROMPT.format(
            docs_read=docs_read,
            notebook_summary=notebook_summary,
            research_lens=self._format_research_lens_for_prompt(),
        )

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian identifying research questions."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.4,
            max_tokens=3000,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if not response.success:
            return []

        parsed = parse_llm_json(response.content, default=[])
        if isinstance(parsed, dict) and "questions" in parsed:
            parsed = parsed.get("questions", [])
        if not isinstance(parsed, list):
            return []

        return [q for q in parsed if isinstance(q, dict) and q.get("question")]

    def _generate_question_synthesis(self) -> Dict[str, Any]:
        if not APP_CONFIG.tier0.synthesis_enabled:
            return {}

        if self._last_question_batch is None:
            return {}

        gate = self._last_question_batch.quality_gate or {}
        if gate.get("action") == "stop":
            reason = gate.get("reason") or "Question quality gate requested synthesis stop."
            self.logger.log("synthesis_skipped", reason, level="WARN")  # Added explicit skip logging so operators can see why essay generation did not run.
            return {
                "status": "skipped",
                "reason": "question_quality_gate",
                "diagnostics": gate,
            }

        return self.question_synthesizer.build_agenda(
            self.notebook,
            self._last_question_batch.questions,
        )

    def _export_patterns(self) -> List[Dict[str, Any]]:
        return [
            {
                "pattern": p.pattern_text,
                "type": p.pattern_type,
                "confidence": p.confidence,
                "evidence_count": len(p.evidence_doc_ids),
                "time_range": p.time_range,
                "first_noticed": p.first_noticed,
            }
            for p in self.notebook.patterns.values()
        ]

    def _export_entities(self) -> List[Dict[str, Any]]:
        sorted_entities = sorted(
            self.notebook.entities.values(),
            key=lambda e: e.document_count,
            reverse=True,
        )

        return [
            {
                "name": e.name,
                "type": e.entity_type,
                "document_count": e.document_count,
                "contexts": e.contexts[:3],
            }
            for e in sorted_entities[:100]
        ]

    def _export_contradictions(self) -> List[Dict[str, Any]]:
        return [
            {
                "claim_a": c.claim_a,
                "claim_b": c.claim_b,
                "source_a": c.source_a,
                "source_b": c.source_b,
                "context": c.context,
                "batch": c.noticed_in_batch,
                "contradiction_type": c.contradiction_type,
                "confidence": c.confidence,
            }
            for c in self.notebook.contradictions
        ]

    def _export_group_indicators(self) -> List[Dict[str, Any]]:
        indicators = list(self.notebook.group_indicators.values())
        return [
            {
                "group_type": g.group_type,
                "label": g.label,
                "evidence_count": len(g.evidence_doc_ids),
                "confidence": g.confidence,
                "first_noticed": g.first_noticed,
            }
            for g in indicators
        ]


# ============================================================================
# Convenience Function
# ============================================================================


def explore_corpus(
    strategy: Optional[str] = None,
    total_budget: Optional[int] = None,
    year_range: Optional[Tuple[int, int]] = None,
    research_lens: Optional[Any] = None,
) -> Dict[str, Any]:
    explorer = CorpusExplorer()
    return explorer.explore(
        strategy=strategy,
        total_budget=total_budget,
        year_range=year_range,
        save_notebook=True,
        research_lens=research_lens,
    )
