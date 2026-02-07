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
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from rag_base import DocumentStore, debug_print, count_tokens
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

BATCH_ANALYSIS_PROMPT = """You are a historian systematically reading archival documents.

CLOSED-WORLD RULES:
- Use ONLY the documents provided below.
- Do NOT use outside knowledge or assumptions.
- If a fact is not in the documents, do not include it.
- You may compare with PRIOR KNOWLEDGE to spot contradictions, but do NOT add facts that are not in the new documents.

PRIOR KNOWLEDGE (summary from previous batches):
{prior_knowledge}

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
  "questions": [
    {{"question": str, "why_interesting": str, "evidence_needed": str, "time_window": str}}
  ],
  "temporal_events": {{"year": [str]}}
}}

Evidence must reference ONLY valid block_id values from DOCUMENTS.
Focus on discoveries: new patterns, contradictions, and questions.
"""

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

    def explore(
        self,
        strategy: Optional[str] = None,
        total_budget: Optional[int] = None,
        year_range: Optional[Tuple[int, int]] = None,
        save_notebook: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute systematic corpus exploration."""
        start_time = time.time()

        config = APP_CONFIG.tier0
        strategy = strategy or config.exploration_strategy
        total_budget = total_budget or config.exploration_budget
        save_notebook = config.notebook_auto_save if save_notebook is None else save_notebook

        self.logger.log("start", f"strategy={strategy} budget={total_budget}")

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

        report = {
            "corpus_map": corpus_map,
            "questions": questions,
            "question_synthesis": question_synthesis,
            "patterns": self._export_patterns(),
            "entities": self._export_entities(),
            "contradictions": self._export_contradictions(),
            "notebook_summary": self.notebook.get_summary(),
            "exploration_metadata": {
                "strategy": strategy,
                "total_budget": total_budget,
                "documents_read": self.notebook.corpus_map["total_documents_read"],
                "batches_processed": self.notebook.corpus_map["batches_processed"],
                "duration_seconds": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
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

        self.logger.log(
            "complete",
            f"docs={report['exploration_metadata']['documents_read']} questions={len(questions)}",
        )

        return report

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

    def _process_stratum(self, stratum: Stratum) -> None:
        sub_batch_size = max(1, APP_CONFIG.tier0.sub_batch_docs)

        if stratum.stream:
            batch_index = 0
            for sub_docs in self.reader.iter_stratum_batches(stratum, sub_batch_size):
                batch_index += 1
                doc_objects = self.reader.build_document_objects(sub_docs, APP_CONFIG.tier0.batch_max_chars)
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
            return

        docs = self.reader.read_stratum(stratum)
        if not docs:
            self.logger.log("batch_skip", f"{stratum.label} empty")
            return

        total_docs = len(docs)
        processed = 0
        batch_index = 0

        while processed < total_docs:
            batch_index += 1
            sub_docs = docs[processed: processed + sub_batch_size]
            processed += len(sub_docs)

            doc_objects = self.reader.build_document_objects(sub_docs, APP_CONFIG.tier0.batch_max_chars)
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

    def _analyze_batch(self, doc_objects: List[Any]) -> Dict[str, Any]:
        prior_knowledge = self.notebook.format_for_llm_context()
        documents_json = self.reader.format_document_objects_for_llm(doc_objects)
        total_blocks = sum(len(doc.blocks) for doc in doc_objects)

        prompt = BATCH_ANALYSIS_PROMPT.format(
            prior_knowledge=prior_knowledge,
            documents_json=documents_json,
        )

        self.logger.log("llm", f"batch tokens~{count_tokens(prompt)}")

        heartbeat_detail = f"batch docs={len(doc_objects)} blocks={total_blocks}"
        with Heartbeat(self.logger, "llm", heartbeat_detail, APP_CONFIG.tier0.heartbeat_seconds):
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": "You are a careful historian."},
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

        if self._needs_repair(findings):
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
                if not self._needs_repair(repaired):
                    findings = repaired
                    break

        return findings

    def _needs_repair(self, findings: Dict[str, Any]) -> bool:
        if not APP_CONFIG.tier0.strict_closed_world:
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
            batch = self.question_pipeline.generate(self.notebook)
            self._last_question_batch = batch
            return batch.to_list()
        except Exception as exc:
            debug_print(f"Question pipeline failed, falling back to simple prompt: {exc}")

        notebook_summary = json.dumps(self.notebook.get_summary(), indent=2)
        docs_read = self.notebook.corpus_map["total_documents_read"]

        prompt = QUESTION_GENERATION_PROMPT.format(
            docs_read=docs_read,
            notebook_summary=notebook_summary,
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


# ============================================================================
# Convenience Function
# ============================================================================


def explore_corpus(
    strategy: Optional[str] = None,
    total_budget: Optional[int] = None,
    year_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    explorer = CorpusExplorer()
    return explorer.explore(
        strategy=strategy,
        total_budget=total_budget,
        year_range=year_range,
        save_notebook=True,
    )
