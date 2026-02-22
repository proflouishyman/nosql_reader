# app/historian_agent/question_typology.py
# Created: 2026-02-05
# Purpose: Historical question typology with type-specific generation

"""
Question Typology - Generate questions by historical type.

Layer 2: Type-specific generation logic.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from rag_base import debug_print
from llm_abstraction import LLMClient
from config import APP_CONFIG

from historian_agent.question_models import Question, QuestionType, parse_llm_question_response
from historian_agent.research_notebook import ResearchNotebook, Pattern, Contradiction, Entity


# ============================================================================
# Prompts
# ============================================================================

CAUSAL_PROMPT = """You are a historian generating a CAUSAL research question.

CLOSED-WORLD RULES:
- Use ONLY the evidence below.
- Do NOT introduce outside knowledge or context.
- If time period is not explicit, leave time_window empty.

PATTERN:
{pattern_text}

EVIDENCE:
- {evidence_count} documents
- Time range: {time_range}
- Entities: {entities}

TASK: Generate ONE specific causal question (WHY/HOW).
Return JSON:
{{
  "question": "...",
  "type": "causal",
  "why_interesting": "...",
  "entities_involved": ["..."],
  "time_window": "YYYY-YYYY"
}}
"""

COMPARATIVE_PROMPT = """You are a historian generating a COMPARATIVE research question.

CLOSED-WORLD RULES:
- Use ONLY the evidence below.
- Do NOT introduce outside knowledge or context.
- If time period is not explicit, leave time_window empty.

COMPARISON BASIS:
{comparison_basis}

TASK: Generate ONE specific comparative question.
Return JSON:
{{
  "question": "...",
  "type": "comparative",
  "why_interesting": "...",
  "entities_involved": ["..."],
  "time_window": "YYYY-YYYY"
}}
"""

CHANGE_OVER_TIME_PROMPT = """You are a historian generating a CHANGE-OVER-TIME question.

CLOSED-WORLD RULES:
- Use ONLY the evidence below.
- Do NOT introduce outside knowledge or context.
- If time period is not explicit, leave time_window empty.

TEMPORAL EVIDENCE:
{temporal_evidence}

TASK: Generate ONE question about transformation over time.
Return JSON:
{{
  "question": "...",
  "type": "change",
  "why_interesting": "...",
  "entities_involved": ["..."],
  "time_window": "YYYY-YYYY"
}}
"""

DISTRIBUTIONAL_PROMPT = """You are a historian generating a DISTRIBUTIONAL question.

CLOSED-WORLD RULES:
- Use ONLY the evidence below.
- Do NOT introduce outside knowledge or context.
- If time period is not explicit, leave time_window empty.

EVIDENCE OF DIFFERENTIAL IMPACTS:
{distribution_evidence}

TASK: Generate ONE question about who benefited/suffered.
Return JSON:
{{
  "question": "...",
  "type": "distributional",
  "why_interesting": "...",
  "entities_involved": ["..."],
  "time_window": "YYYY-YYYY"
}}
"""

INSTITUTIONAL_PROMPT = """You are a historian generating an INSTITUTIONAL question.

CLOSED-WORLD RULES:
- Use ONLY the evidence below.
- Do NOT introduce outside knowledge or context.
- If time period is not explicit, leave time_window empty.

INSTITUTIONAL EVIDENCE:
{institutional_evidence}

TASK: Generate ONE question about rules/practices/criteria.
Return JSON:
{{
  "question": "...",
  "type": "institutional",
  "why_interesting": "...",
  "entities_involved": ["..."],
  "time_window": "YYYY-YYYY"
}}
"""

SCOPE_PROMPT = """You are a historian generating a SCOPE CONDITIONS question.

CLOSED-WORLD RULES:
- Use ONLY the evidence below.
- Do NOT introduce outside knowledge or context.
- If time period is not explicit, leave time_window empty.

PATTERN / BOUNDARY EVIDENCE:
{scope_evidence}

TASK: Generate ONE question about where/when a pattern applies or fails.
Return JSON:
{{
  "question": "...",
  "type": "scope",
  "why_interesting": "...",
  "entities_involved": ["..."],
  "time_window": "YYYY-YYYY"
}}
"""

QUESTION_REPAIR_PROMPT = """You are a strict JSON formatter.

INPUT (raw model output):
{raw_output}

TASK:
- Convert the input into a JSON array of objects with fields:
  - question
  - type
  - why_interesting
  - entities_involved
  - time_window
- Do NOT add new content. Only reformat what is present.
- If no valid question exists, return an empty JSON array: []

Return ONLY JSON.
"""

# ============================================================================
# Generator
# ============================================================================

class TypedQuestionGenerator:
    """Generate questions by historical type."""

    def __init__(self) -> None:
        self.llm = LLMClient()

    def generate_by_type(
        self,
        qtype: QuestionType,
        context: Dict[str, Any],
        max_questions: int = 5,
    ) -> List[Question]:
        if qtype == QuestionType.CAUSAL:
            return self._generate_causal(context, max_questions)
        if qtype == QuestionType.COMPARATIVE:
            return self._generate_comparative(context, max_questions)
        if qtype == QuestionType.CHANGE_OVER_TIME:
            return self._generate_change_over_time(context, max_questions)
        if qtype == QuestionType.DISTRIBUTIONAL:
            return self._generate_distributional(context, max_questions)
        if qtype == QuestionType.INSTITUTIONAL:
            return self._generate_institutional(context, max_questions)
        if qtype == QuestionType.SCOPE_CONDITIONS:
            return self._generate_scope(context, max_questions)
        return []

    def _generate_causal(self, context: Dict[str, Any], max_questions: int) -> List[Question]:
        patterns: List[Pattern] = context.get("patterns", [])
        patterns = self._filter_patterns(patterns)
        patterns = sorted(patterns, key=lambda p: len(p.evidence_doc_ids), reverse=True)
        entities = self._top_entities(context)

        questions: List[Question] = []
        for pattern in patterns[:max_questions]:
            prompt = CAUSAL_PROMPT.format(
                pattern_text=pattern.pattern_text,
                evidence_count=len(pattern.evidence_doc_ids),
                time_range=pattern.time_range or "unknown",
                entities=", ".join(entities) or "unknown",
            )
            generated = self._generate_from_prompt(prompt, QuestionType.CAUSAL)
            for q in generated:
                q.pattern_source = pattern.pattern_text
                q.evidence_doc_ids = list(pattern.evidence_doc_ids)
                q.evidence_block_ids = list(pattern.evidence_block_ids)
                questions.append(q)

        return questions

    def _generate_comparative(self, context: Dict[str, Any], max_questions: int) -> List[Question]:
        contradictions: List[Contradiction] = context.get("contradictions", [])
        questions: List[Question] = []

        for contra in contradictions[:max_questions]:
            basis = (
                f"Source A: {contra.claim_a}\n"
                f"Source B: {contra.claim_b}\n"
                f"Context: {contra.context}"
            )
            prompt = COMPARATIVE_PROMPT.format(comparison_basis=basis)
            generated = self._generate_from_prompt(prompt, QuestionType.COMPARATIVE)
            for q in generated:
                q.contradiction_source = f"{contra.source_a} vs {contra.source_b}"
                doc_a = contra.source_a.split("::")[0] if isinstance(contra.source_a, str) else contra.source_a
                doc_b = contra.source_b.split("::")[0] if isinstance(contra.source_b, str) else contra.source_b
                q.evidence_doc_ids = [doc_a, doc_b]
                questions.append(q)

        if not questions:
            entities = self._top_entities(context)
            basis = f"Compare patterns across entities: {', '.join(entities) or 'groups'}"
            prompt = COMPARATIVE_PROMPT.format(comparison_basis=basis)
            questions.extend(self._generate_from_prompt(prompt, QuestionType.COMPARATIVE))

        return questions[:max_questions]

    def _generate_change_over_time(self, context: Dict[str, Any], max_questions: int) -> List[Question]:
        temporal_map: Dict[str, List[str]] = context.get("temporal_map", {})
        questions: List[Question] = []

        if temporal_map:
            sample = list(temporal_map.items())[:max_questions]
            for year, events in sample:
                evidence = f"Year {year}: {', '.join(events[:3])}"
                prompt = CHANGE_OVER_TIME_PROMPT.format(temporal_evidence=evidence)
                questions.extend(self._generate_from_prompt(prompt, QuestionType.CHANGE_OVER_TIME))

        if not questions:
            patterns: List[Pattern] = self._filter_patterns(context.get("patterns", []))
            for pattern in patterns[:max_questions]:
                evidence = f"Pattern: {pattern.pattern_text} (time range: {pattern.time_range})"
                prompt = CHANGE_OVER_TIME_PROMPT.format(temporal_evidence=evidence)
                questions.extend(self._generate_from_prompt(prompt, QuestionType.CHANGE_OVER_TIME))

        return questions[:max_questions]

    def _generate_distributional(self, context: Dict[str, Any], max_questions: int) -> List[Question]:
        entities = self._top_entities(context)
        evidence = f"Entities with many documents: {', '.join(entities) or 'groups'}"
        prompt = DISTRIBUTIONAL_PROMPT.format(distribution_evidence=evidence)
        questions = self._generate_from_prompt(prompt, QuestionType.DISTRIBUTIONAL)
        return questions[:max_questions]

    def _generate_institutional(self, context: Dict[str, Any], max_questions: int) -> List[Question]:
        patterns: List[Pattern] = self._filter_patterns(context.get("patterns", []))
        institutional = [p for p in patterns if "policy" in p.pattern_type.lower() or "procedure" in p.pattern_type.lower()]
        candidates = institutional or patterns
        questions: List[Question] = []
        for pattern in candidates[:max_questions]:
            evidence = f"Pattern: {pattern.pattern_text}"
            prompt = INSTITUTIONAL_PROMPT.format(institutional_evidence=evidence)
            generated = self._generate_from_prompt(prompt, QuestionType.INSTITUTIONAL)
            for q in generated:
                q.pattern_source = pattern.pattern_text
                q.evidence_doc_ids = list(pattern.evidence_doc_ids)
                questions.append(q)
        return questions[:max_questions]

    def _generate_scope(self, context: Dict[str, Any], max_questions: int) -> List[Question]:
        patterns: List[Pattern] = self._filter_patterns(context.get("patterns", []))
        questions: List[Question] = []
        for pattern in patterns[:max_questions]:
            evidence = f"Pattern: {pattern.pattern_text} (time: {pattern.time_range})"
            prompt = SCOPE_PROMPT.format(scope_evidence=evidence)
            generated = self._generate_from_prompt(prompt, QuestionType.SCOPE_CONDITIONS)
            for q in generated:
                q.pattern_source = pattern.pattern_text
                q.evidence_doc_ids = list(pattern.evidence_doc_ids)
                questions.append(q)
        return questions[:max_questions]

    def _generate_from_prompt(self, prompt: str, qtype: QuestionType) -> List[Question]:
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian generating research questions using only the provided evidence."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.3,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if not response.success:
            return []

        questions = parse_llm_question_response(response.content)
        if not questions and response.content.strip():
            repair_prompt = QUESTION_REPAIR_PROMPT.format(raw_output=response.content)
            repair = self.llm.generate(
                messages=[
                    {"role": "system", "content": "You only reformat text into strict JSON."},
                    {"role": "user", "content": repair_prompt},
                ],
                profile="verifier",
                temperature=0.0,
                timeout=APP_CONFIG.tier0.llm_timeout,
            )
            if repair.success:
                questions = parse_llm_question_response(repair.content)
        for q in questions:
            q.question_type = qtype
        return questions

    def _top_entities(self, context: Dict[str, Any], limit: int = 5) -> List[str]:
        entities: List[Entity] = context.get("entities", [])
        entities_sorted = sorted(entities, key=lambda e: e.document_count, reverse=True)
        return [e.name for e in entities_sorted[:limit]]

    def _filter_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        min_docs = APP_CONFIG.tier0.question_min_evidence_docs
        return [p for p in patterns if len(p.evidence_doc_ids) >= min_docs]


# Convenience wrapper

def generate_typed_questions(
    notebook: ResearchNotebook,
    max_questions_per_type: int = 5,
) -> List[Question]:
    generator = TypedQuestionGenerator()
    context = {
        "patterns": list(notebook.patterns.values()),
        "contradictions": notebook.contradictions,
        "temporal_map": dict(notebook.temporal_map),
        "entities": list(notebook.entities.values()),
    }

    questions: List[Question] = []
    for qtype in QuestionType:
        questions.extend(generator.generate_by_type(qtype, context, max_questions_per_type))

    return questions
