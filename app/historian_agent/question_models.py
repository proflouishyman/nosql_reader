# app/historian_agent/question_models.py
# Created: 2026-02-05
# Purpose: Data models for research question generation and validation

"""
Question Models - Type-safe internal representations with dict serialization.

Pattern: "Lists internally, dicts externally"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from historian_agent.tier0_utils import parse_llm_json


# ============================================================================
# Enums
# ============================================================================

class QuestionType(Enum):
    """Historical question typology."""
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    CHANGE_OVER_TIME = "change"
    DISTRIBUTIONAL = "distributional"
    INSTITUTIONAL = "institutional"
    SCOPE_CONDITIONS = "scope"


class ValidationStatus(Enum):
    """Question validation status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_REFINEMENT = "needs_refinement"
    REJECTED = "rejected"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Question:
    """Research question (internal representation)."""

    question_text: str
    question_type: QuestionType

    why_interesting: str
    time_window: Optional[str] = None
    entities_involved: List[str] = field(default_factory=list)

    evidence_doc_ids: List[str] = field(default_factory=list)
    evidence_block_ids: List[str] = field(default_factory=list)
    pattern_source: Optional[str] = None
    contradiction_source: Optional[str] = None

    generation_method: str = "llm_generated"

    validation_score: Optional[int] = None
    validation_status: Optional[ValidationStatus] = None
    answerability_score: Optional[int] = None
    significance_score: Optional[int] = None
    specificity_score: Optional[int] = None
    evidence_based_score: Optional[int] = None
    critique: Optional[str] = None

    answerability_doc_count: Optional[int] = None
    answerability_status: Optional[str] = None
    answerability_sample: List[str] = field(default_factory=list)

    refinement_count: int = 0
    original_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question_text,
            "type": self.question_type.value,
            "why_interesting": self.why_interesting,
            "time_window": self.time_window,
            "entities_involved": self.entities_involved,
            "evidence_count": len(self.evidence_doc_ids),
            "evidence_sample": self.evidence_doc_ids[:5],
            "pattern_source": self.pattern_source,
            "contradiction_source": self.contradiction_source,
            "generation_method": self.generation_method,
            "validation": {
                "score": self.validation_score,
                "status": self.validation_status.value if self.validation_status else None,
                "answerability": self.answerability_score,
                "significance": self.significance_score,
                "specificity": self.specificity_score,
                "evidence_based": self.evidence_based_score,
                "critique": self.critique,
            } if self.validation_score is not None else None,
            "answerability_precheck": {
                "doc_count": self.answerability_doc_count,
                "status": self.answerability_status,
                "sample_doc_ids": self.answerability_sample,
            } if self.answerability_doc_count is not None else None,
            "refinement_count": self.refinement_count,
            "original_question": self.original_question,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        qtype = data.get("type", "causal")
        if isinstance(qtype, str):
            try:
                qtype = QuestionType(qtype)
            except ValueError:
                qtype = QuestionType.CAUSAL

        return cls(
            question_text=str(data.get("question", "")),
            question_type=qtype,
            why_interesting=str(data.get("why_interesting", "")),
            time_window=data.get("time_window"),
            entities_involved=data.get("entities_involved", []) or [],
            evidence_doc_ids=data.get("evidence_doc_ids", []) or data.get("evidence_sample", []) or [],
            evidence_block_ids=data.get("evidence_block_ids", []) or [],
            pattern_source=data.get("pattern_source"),
            contradiction_source=data.get("contradiction_source"),
            generation_method=data.get("generation_method", "llm_generated"),
        )


@dataclass
class QuestionValidation:
    """Validation results for a research question."""

    total_score: int
    answerability: int
    significance: int
    specificity: int
    evidence_based: int
    critique: str
    suggestions: List[str] = field(default_factory=list)
    status: ValidationStatus = field(init=False)

    def __post_init__(self) -> None:
        if self.total_score >= 80:
            self.status = ValidationStatus.EXCELLENT
        elif self.total_score >= 70:
            self.status = ValidationStatus.GOOD
        elif self.total_score >= 60:
            self.status = ValidationStatus.ACCEPTABLE
        elif self.total_score >= 50:
            self.status = ValidationStatus.NEEDS_REFINEMENT
        else:
            self.status = ValidationStatus.REJECTED

    def apply_to_question(self, question: Question) -> None:
        question.validation_score = self.total_score
        question.validation_status = self.status
        question.answerability_score = self.answerability
        question.significance_score = self.significance
        question.specificity_score = self.specificity
        question.evidence_based_score = self.evidence_based
        question.critique = self.critique


@dataclass
class QuestionBatch:
    """Batch of questions (internal collection)."""

    questions: List[Question] = field(default_factory=list)
    generation_strategy: str = "mixed"
    total_candidates: int = 0
    total_validated: int = 0
    total_accepted: int = 0

    def add(self, question: Question) -> None:
        self.questions.append(question)

    def filter_by_status(self, status: ValidationStatus) -> List[Question]:
        return [q for q in self.questions if q.validation_status == status]

    def filter_by_type(self, qtype: QuestionType) -> List[Question]:
        return [q for q in self.questions if q.question_type == qtype]

    def get_top_n(self, n: int) -> List[Question]:
        sorted_questions = sorted(
            [q for q in self.questions if q.validation_score is not None],
            key=lambda q: q.validation_score,
            reverse=True,
        )
        return sorted_questions[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "questions": [q.to_dict() for q in self.questions],
            "metadata": {
                "generation_strategy": self.generation_strategy,
                "total_candidates": self.total_candidates,
                "total_validated": self.total_validated,
                "total_accepted": self.total_accepted,
                "final_count": len(self.questions),
            },
        }

    def to_list(self) -> List[Dict[str, Any]]:
        return [q.to_dict() for q in self.questions]


# ============================================================================
# Helper Functions
# ============================================================================


def parse_llm_question_response(response_text: str) -> List[Question]:
    data = parse_llm_json(response_text, default=[])

    if isinstance(data, dict):
        if "questions" in data:
            questions_data = data.get("questions", [])
        else:
            questions_data = [data]
    elif isinstance(data, list):
        questions_data = data
    else:
        questions_data = []

    questions: List[Question] = []
    for q_data in questions_data:
        if not isinstance(q_data, dict):
            continue
        try:
            questions.append(Question.from_dict(q_data))
        except Exception:
            continue

    return questions


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except Exception:
            return 0
    if isinstance(value, dict):
        for key in ("score", "value", "total", "number"):
            if key in value:
                return _coerce_int(value[key])
        for v in value.values():
            coerced = _coerce_int(v)
            if coerced:
                return coerced
        return 0
    if isinstance(value, list) and value:
        return _coerce_int(value[0])
    return 0


def parse_validation_response(response_text: str) -> QuestionValidation:
    data = parse_llm_json(response_text, default={})

    return QuestionValidation(
        total_score=_coerce_int(data.get("score", 0)),
        answerability=_coerce_int(data.get("answerability", 0)),
        significance=_coerce_int(data.get("significance", 0)),
        specificity=_coerce_int(data.get("specificity", 0)),
        evidence_based=_coerce_int(data.get("evidence_based", 0)),
        critique=str(data.get("critique", "")),
        suggestions=data.get("suggestions", []) if isinstance(data.get("suggestions"), list)
        else [str(data.get("suggestions", ""))],
    )
