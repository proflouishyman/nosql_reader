# app/historian_agent/question_validator.py
# Created: 2026-02-05
# Purpose: Adversarial validation and refinement for research questions

"""
Question Validator - Layer 3: adversarial validation and refinement.
"""

from __future__ import annotations

from typing import List, Dict, Any

from rag_base import debug_print
from llm_abstraction import LLMClient
from config import APP_CONFIG

from question_models import Question, QuestionValidation, ValidationStatus, parse_validation_response
from research_notebook import ResearchNotebook


VALIDATION_PROMPT = """You are a historian evaluating a research question.

CLOSED-WORLD RULES:
- Use ONLY the corpus knowledge summary provided.
- Do NOT introduce outside context.
- If a time period is not explicit in the corpus, penalize specificity.

QUESTION: {question}

CORPUS KNOWLEDGE SUMMARY:
{notebook_summary}

Evaluate on four criteria (0-25 each):
1. Answerability: Can this be answered with the available documents?
2. Historical Significance: Would historians care? Does it address mechanisms or change?
3. Specificity: Is it well-defined (time, entities, scope)?
4. Evidence-Based: Is it grounded in observed patterns or contradictions?

Return JSON:
{{
  "score": 0-100,
  "answerability": 0-25,
  "significance": 0-25,
  "specificity": 0-25,
  "evidence_based": 0-25,
  "critique": "...",
  "suggestions": ["..."]
}}
"""

REFINE_PROMPT = """You are refining a research question.

CLOSED-WORLD RULES:
- Use ONLY the corpus summary provided.
- Do NOT introduce outside context.
- If time period is not explicit, leave time_window empty.

ORIGINAL QUESTION: {question}

CRITIQUE:
{critique}

CORPUS SUMMARY:
{notebook_summary}

TASK: Rewrite the question to address the critique.
- Keep it answerable with the corpus
- Add specificity (time, entities)
- Make it causal/comparative if possible

Return JSON:
{{
  "question": "...",
  "type": "{qtype}",
  "why_interesting": "...",
  "entities_involved": ["..."],
  "time_window": "..."
}}
"""


class QuestionValidator:
    """Validate and refine research questions."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def validate(self, question: Question, notebook: ResearchNotebook) -> QuestionValidation:
        summary = notebook.get_summary()
        prompt = VALIDATION_PROMPT.format(
            question=question.question_text,
            notebook_summary=summary,
        )

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a strict historical reviewer."},
                {"role": "user", "content": prompt},
            ],
            profile="verifier",
            temperature=0.0,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if not response.success:
            return QuestionValidation(
                total_score=0,
                answerability=0,
                significance=0,
                specificity=0,
                evidence_based=0,
                critique="Validation failed",
                suggestions=[],
            )

        validation = parse_validation_response(response.content)
        validation.apply_to_question(question)
        return validation

    def refine(self, question: Question, notebook: ResearchNotebook) -> Question:
        summary = notebook.get_summary()
        prompt = REFINE_PROMPT.format(
            question=question.question_text,
            critique=question.critique or "",
            notebook_summary=summary,
            qtype=question.question_type.value,
        )

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You refine research questions."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.3,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if not response.success:
            return question

        from question_models import parse_llm_question_response

        candidates = parse_llm_question_response(response.content)
        if not candidates:
            return question

        refined = candidates[0]
        refined.question_type = question.question_type
        refined.pattern_source = question.pattern_source
        refined.contradiction_source = question.contradiction_source
        refined.evidence_doc_ids = list(question.evidence_doc_ids)
        refined.evidence_block_ids = list(question.evidence_block_ids)
        refined.original_question = question.original_question or question.question_text
        refined.refinement_count = question.refinement_count + 1
        return refined

    def validate_and_refine(
        self,
        question: Question,
        notebook: ResearchNotebook,
        max_refinements: int,
        min_score_refine: int = 50,
        min_score_accept: int = 60,
    ) -> Question:
        validation = self.validate(question, notebook)

        attempts = 0
        while (
            attempts < max_refinements
            and question.validation_score is not None
            and question.validation_score >= min_score_refine
            and question.validation_score < min_score_accept
        ):
            attempts += 1
            question = self.refine(question, notebook)
            validation = self.validate(question, notebook)

        return question


def validate_question_batch(
    questions: List[Question],
    notebook: ResearchNotebook,
    max_refinements: int,
    min_score_refine: int,
    min_score_accept: int,
) -> List[Question]:
    validator = QuestionValidator()
    validated: List[Question] = []

    for question in questions:
        validated.append(
            validator.validate_and_refine(
                question,
                notebook,
                max_refinements,
                min_score_refine=min_score_refine,
                min_score_accept=min_score_accept,
            )
        )

    return validated
