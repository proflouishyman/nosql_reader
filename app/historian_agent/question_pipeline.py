# app/historian_agent/question_pipeline.py
# Created: 2026-02-05
# Purpose: Question generation pipeline with typology and validation

"""
Question Generation Pipeline - Orchestrates typed generation and validation.

Layer 4: Pipeline orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from collections import defaultdict

from rag_base import debug_print
from config import APP_CONFIG

import re

from question_models import Question, QuestionBatch, QuestionType
from question_typology import TypedQuestionGenerator
from question_validator import QuestionValidator
from question_answerability import AnswerabilityChecker
from research_notebook import ResearchNotebook

PLACEHOLDER_RE = re.compile(r"\[[^\\]]+\\]|\\{[^\\}]+\\}")

CAUSAL_MARKERS = ("why", "how", "what explains", "what caused", "what led", "what contributed", "what resulted")
COMPARATIVE_MARKERS = ("between", "compare", "difference", "differ", "versus", "vs.")
DISTRIBUTIONAL_MARKERS = ("who", "which group", "differ", "difference", "distribution", "disproportion")
INSTITUTIONAL_MARKERS = ("criteria", "rule", "policy", "procedure", "eligibility", "requirement", "standard")
SCOPE_MARKERS = ("where", "when", "under what conditions", "scope", "limits", "apply")

@dataclass
class PipelineConfig:
    questions_per_type: int
    min_score_accept: int
    min_score_refine: int
    max_refinements: int
    target_questions: int
    min_questions: int
    enforce_type_diversity: bool
    min_types_represented: int

    @classmethod
    def from_app_config(cls) -> "PipelineConfig":
        cfg = APP_CONFIG.tier0
        return cls(
            questions_per_type=cfg.question_per_type,
            min_score_accept=cfg.question_min_score,
            min_score_refine=cfg.question_min_score_refine,
            max_refinements=cfg.question_max_refinements,
            target_questions=cfg.question_target_count,
            min_questions=cfg.question_min_count,
            enforce_type_diversity=cfg.question_enforce_type_diversity,
            min_types_represented=cfg.question_min_types,
        )


class QuestionGenerationPipeline:
    """Pipeline for generating, validating, and ranking questions."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig.from_app_config()
        self.generator = TypedQuestionGenerator()
        self.validator = QuestionValidator()
        self.answerability = AnswerabilityChecker()

    def generate(self, notebook: ResearchNotebook) -> QuestionBatch:
        debug_print("Starting question generation pipeline")

        candidates = self._generate_candidates(notebook)
        debug_print(f"Generated {len(candidates)} candidates")

        validated = self._validate_candidates(candidates, notebook)
        debug_print(f"Validated {len(validated)} candidates")

        validated = self._improve_quality(validated, notebook)
        debug_print(f"Quality pass kept {len(validated)} candidates")

        filtered = self._apply_answerability_precheck(validated)
        debug_print(f"Answerability precheck kept {len(filtered)} candidates")

        filtered = self._filter_by_score(filtered)
        debug_print(f"Filtered to {len(filtered)} (score >= {self.config.min_score_accept})")

        filtered = self._filter_by_evidence(filtered)
        debug_print(f"Evidence filter kept {len(filtered)}")

        sanitized = [self._sanitize_time_window(q, notebook) for q in filtered]

        deduped = self._deduplicate(sanitized)
        debug_print(f"Deduplicated to {len(deduped)}")

        if self.config.enforce_type_diversity:
            diverse = self._ensure_diversity(deduped)
        else:
            diverse = deduped

        final = self._rank_and_select(diverse)
        debug_print(f"Selected {len(final)} final questions")

        return QuestionBatch(
            questions=final,
            generation_strategy="typed_validated",
            total_candidates=len(candidates),
            total_validated=len(validated),
            total_accepted=len(filtered),
        )

    def _improve_quality(self, questions: List[Question], notebook: ResearchNotebook) -> List[Question]:
        improved: List[Question] = []
        for q in questions:
            critique = self._quality_critique(q)
            if not critique:
                improved.append(q)
                continue

            refined = self.validator.refine_with_critique(q, notebook, critique)
            # Re-validate to update scores if refinement occurred
            if refined.question_text != q.question_text:
                self.validator.validate(refined, notebook)

            # Drop if still low-quality (placeholders or missing intent markers)
            if self._quality_critique(refined):
                continue
            improved.append(refined)

        return improved

    def _quality_critique(self, question: Question) -> str | None:
        text = (question.question_text or "").strip().lower()
        if not text:
            return "Question is empty."

        if PLACEHOLDER_RE.search(text):
            return "Remove placeholders like [name] or {date} and replace with concrete entities from evidence."

        qtype = question.question_type
        if qtype == QuestionType.CAUSAL and not self._contains_any(text, CAUSAL_MARKERS):
            return "Make this a causal question by using why/how or an explicit mechanism."
        if qtype == QuestionType.COMPARATIVE and not self._contains_any(text, COMPARATIVE_MARKERS):
            return "Make this explicitly comparative (e.g., between X and Y)."
        if qtype == QuestionType.DISTRIBUTIONAL and not self._contains_any(text, DISTRIBUTIONAL_MARKERS):
            return "Make this distributional (who benefited or suffered, or which group differed)."
        if qtype == QuestionType.INSTITUTIONAL and not self._contains_any(text, INSTITUTIONAL_MARKERS):
            return "Make this institutional by naming criteria, rules, or procedures."
        if qtype == QuestionType.SCOPE_CONDITIONS and not self._contains_any(text, SCOPE_MARKERS):
            return "Make this a scope question (where/when/under what conditions a pattern applies)."

        if "pattern" in text and not question.pattern_source:
            return "Avoid vague references to 'pattern' and anchor the question in a specific observed claim."

        return None

    def _contains_any(self, text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)

    def _generate_candidates(self, notebook: ResearchNotebook) -> List[Question]:
        context = {
            "patterns": list(notebook.patterns.values()),
            "contradictions": notebook.contradictions,
            "temporal_map": dict(notebook.temporal_map),
            "entities": list(notebook.entities.values()),
        }

        candidates: List[Question] = []
        for qtype in QuestionType:
            typed = self.generator.generate_by_type(
                qtype,
                context,
                max_questions=self.config.questions_per_type,
            )
            candidates.extend(typed)

        return candidates

    def _validate_candidates(self, candidates: List[Question], notebook: ResearchNotebook) -> List[Question]:
        validated: List[Question] = []
        for question in candidates:
            validated.append(
                self.validator.validate_and_refine(
                    question,
                    notebook,
                    max_refinements=self.config.max_refinements,
                    min_score_refine=self.config.min_score_refine,
                    min_score_accept=self.config.min_score_accept,
                )
            )
        return validated

    def _filter_by_score(self, questions: List[Question]) -> List[Question]:
        return [
            q for q in questions
            if q.validation_score is not None
            and q.validation_score >= self.config.min_score_accept
        ]

    def _filter_by_evidence(self, questions: List[Question]) -> List[Question]:
        min_docs = APP_CONFIG.tier0.question_min_evidence_docs
        filtered: List[Question] = []
        for q in questions:
            evidence_count = len(q.evidence_doc_ids)
            if q.contradiction_source:
                if evidence_count >= 2:
                    filtered.append(q)
                continue
            if evidence_count >= min_docs:
                filtered.append(q)
        return filtered

    def _apply_answerability_precheck(self, questions: List[Question]) -> List[Question]:
        checked: List[Question] = []
        for q in questions:
            result = self.answerability.check(q.question_text)
            q.answerability_doc_count = result.doc_count
            q.answerability_status = result.status
            q.answerability_sample = result.sample_doc_ids
            if result.status == "ok":
                checked.append(q)
        return checked or questions

    def _sanitize_time_window(self, question: Question, notebook: ResearchNotebook) -> Question:
        if not question.time_window:
            return question

        allowable_years = set()
        pattern_source = question.pattern_source or ""
        contradiction_source = question.contradiction_source or ""
        time_range = ""
        if question.pattern_source and question.pattern_source in notebook.patterns:
            time_range = notebook.patterns[question.pattern_source].time_range or ""

        for text in [pattern_source, contradiction_source, time_range]:
            for year in re.findall(r"(18\\d{2}|19\\d{2}|20\\d{2})", text):
                allowable_years.add(year)

        if not allowable_years:
            question.time_window = None
            return question

        question_years = set(re.findall(r"(18\\d{2}|19\\d{2}|20\\d{2})", question.time_window or ""))
        if question_years and not question_years.issubset(allowable_years):
            question.time_window = None
        return question

    def _deduplicate(self, questions: List[Question]) -> List[Question]:
        seen = set()
        unique: List[Question] = []
        for q in questions:
            key = q.question_text.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(q)
        return unique

    def _ensure_diversity(self, questions: List[Question]) -> List[Question]:
        buckets = defaultdict(list)
        for q in questions:
            buckets[q.question_type].append(q)

        # If already diverse enough, return
        if len(buckets) >= self.config.min_types_represented:
            return questions

        # Otherwise, keep at least one from each type if available
        diversified: List[Question] = []
        for qtype in QuestionType:
            if buckets.get(qtype):
                diversified.append(buckets[qtype][0])

        # Fill remaining slots with highest scoring
        remaining = [q for q in questions if q not in diversified]
        remaining_sorted = sorted(
            remaining,
            key=lambda q: q.validation_score or 0,
            reverse=True,
        )
        diversified.extend(remaining_sorted)

        return diversified

    def _rank_and_select(self, questions: List[Question]) -> List[Question]:
        sorted_questions = sorted(
            questions,
            key=lambda q: q.validation_score or 0,
            reverse=True,
        )
        if len(sorted_questions) >= self.config.target_questions:
            return sorted_questions[: self.config.target_questions]
        return sorted_questions[: max(self.config.min_questions, len(sorted_questions))]


# Convenience helper

def generate_questions(notebook: ResearchNotebook) -> QuestionBatch:
    pipeline = QuestionGenerationPipeline()
    return pipeline.generate(notebook)
