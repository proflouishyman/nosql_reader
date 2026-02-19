# app/historian_agent/question_pipeline.py
# Created: 2026-02-05
# Purpose: Question generation pipeline with typology and validation

"""
Question Generation Pipeline - Orchestrates typed generation and validation.

Layer 4: Pipeline orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from collections import defaultdict

from rag_base import debug_print
from config import APP_CONFIG

import re

from historian_agent.question_models import Question, QuestionBatch, QuestionType, ValidationStatus
from historian_agent.question_typology import TypedQuestionGenerator
from historian_agent.question_validator import QuestionValidator
from historian_agent.question_answerability import AnswerabilityChecker
from historian_agent.research_notebook import ResearchNotebook
from historian_agent.tier0_utils import parse_llm_json

PLACEHOLDER_RE = re.compile(r"\[[^\\]]+\\]|\\{[^\\}]+\\}")

CAUSAL_MARKERS = ("why", "how", "what explains", "what caused", "what led", "what contributed", "what resulted")
COMPARATIVE_MARKERS = ("between", "compare", "difference", "differ", "versus", "vs.")
DISTRIBUTIONAL_MARKERS = ("who", "which group", "differ", "difference", "distribution", "disproportion")
INSTITUTIONAL_MARKERS = ("criteria", "rule", "policy", "procedure", "eligibility", "requirement", "standard")
SCOPE_MARKERS = ("where", "when", "under what conditions", "scope", "limits", "apply")
INJURY_KEYWORDS = ("injury", "injuries", "accident", "disablement", "disabled", "wound")
JOB_KEYWORDS = ("job", "occupation", "department", "laborer", "brakeman", "conductor", "worker", "track")
PROCESS_KEYWORDS = ("process", "procedure", "workflow", "review", "approval", "adjudication", "paperwork", "timeline", "delay", "administrative", "record", "filing", "certification")  # Added process-family aliases so workflow-focused lenses can influence ranking.
MEANINGFULNESS_GATE_PROMPT = """You are auditing whether a generated question set is meaningful enough for historical synthesis.

CLOSED-WORLD RULES:
- Use ONLY the run context and questions below.
- Do NOT introduce outside context.
- Prefer false negatives to false positives.

RUN CONTEXT:
- Research lens: {lens_terms}
- Documents read: {documents_read}
- Pattern count: {pattern_count}
- Contradiction count: {contradiction_count}
- Entity count: {entity_count}

QUESTIONS:
{question_summaries}

DECISION RULES:
- ACTION=stop if most questions are rejected, off-domain, or too vague.
- ACTION=stop if only one credible question remains and it is a synthetic seed question.
- ACTION=stop if notebook signal is thin (no patterns and no contradictions) and fewer than 2 credible questions remain.
- ACTION=proceed_with_caution only when at least one credible question exists but risk is still notable.

Return ONLY JSON:
{{
  "action": "proceed|proceed_with_caution|stop",
  "reason": "short explanation",
  "issues": ["..."],
  "confidence": "low|medium|high"
}}
"""  # Added LLM-based quality gate prompt so the pipeline can explicitly halt weak question sets before essay synthesis.

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

    def generate(self, notebook: ResearchNotebook, research_lens: List[str] | None = None) -> QuestionBatch:
        debug_print("Starting question generation pipeline")
        lens_terms = self._normalize_lens_terms(research_lens)  # Added lens normalization so ranking can prioritize user-selected historian themes.

        candidates = self._generate_candidates(notebook)
        debug_print(f"Generated {len(candidates)} candidates")

        validated = self._validate_candidates(candidates, notebook)
        debug_print(f"Validated {len(validated)} candidates")
        validated = self._fallback_validation_scores(validated)

        validated = self._improve_quality(validated, notebook)
        debug_print(f"Quality pass kept {len(validated)} candidates")

        filtered = self._apply_answerability_precheck(validated)
        debug_print(f"Answerability precheck kept {len(filtered)} candidates")
        filtered = self._filter_rejected(filtered)
        debug_print(f"Rejected-status filter kept {len(filtered)} candidates")

        filtered = self._filter_by_score(filtered)
        debug_print(f"Filtered to {len(filtered)} (score >= {self.config.min_score_accept})")

        filtered = self._filter_by_evidence(filtered)
        debug_print(f"Evidence filter kept {len(filtered)}")
        if not filtered:
            # Added fallback so strict gates do not collapse the run to zero essay-driving questions.
            filtered = self._fallback_candidates(validated, lens_terms)
            debug_print(f"Fallback selected {len(filtered)} evidence-backed candidates")

        sanitized = [self._sanitize_time_window(q, notebook) for q in filtered]
        sanitized = self._inject_lens_seed_questions(sanitized, notebook, lens_terms)  # Added unified seeding so the active lens can add one targeted structural question without overriding archive-driven variation.

        deduped = self._deduplicate(sanitized)
        debug_print(f"Deduplicated to {len(deduped)}")

        if self.config.enforce_type_diversity:
            diverse = self._ensure_diversity(deduped)
        else:
            diverse = deduped

        final = self._rank_and_select(diverse, lens_terms)
        debug_print(f"Selected {len(final)} final questions")
        quality_gate = self._assess_meaningfulness(final, notebook, lens_terms)
        debug_print(f"Meaningfulness gate action={quality_gate.get('action')} confidence={quality_gate.get('confidence')}")

        return QuestionBatch(
            questions=final,
            generation_strategy="typed_validated",
            total_candidates=len(candidates),
            total_validated=len(validated),
            total_accepted=len(filtered),
            quality_gate=quality_gate,
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

    def _fallback_validation_scores(self, questions: List[Question]) -> List[Question]:
        scores = [q.validation_score for q in questions if q.validation_score is not None]
        if not scores:
            return questions
        if len(set(scores)) > 1:
            return questions
        debug_print("Validation scores collapsed; applying heuristic scoring fallback.")
        for q in questions:
            total, a, s, sp, e = self._heuristic_score(q)
            q.validation_score = total
            q.answerability_score = a
            q.significance_score = s
            q.specificity_score = sp
            q.evidence_based_score = e
        return questions

    def _heuristic_score(self, q: Question) -> tuple[int, int, int, int, int]:
        evidence = len(q.evidence_doc_ids)
        answerability = min(25, 5 + evidence * 2)
        significance = 10
        if q.question_type in {QuestionType.CAUSAL, QuestionType.COMPARATIVE}:
            significance += 6
        if q.pattern_source or q.contradiction_source:
            significance += 5
        significance = min(25, significance)

        specificity = 8
        if q.time_window:
            specificity += 6
        if q.entities_involved:
            specificity += 6
        if len(q.question_text.split()) > 10:
            specificity += 4
        specificity = min(25, specificity)

        evidence_based = 6
        if q.pattern_source or q.contradiction_source:
            evidence_based += 8
        evidence_based += min(11, evidence)
        evidence_based = min(25, evidence_based)

        total = min(100, answerability + significance + specificity + evidence_based)
        return total, answerability, significance, specificity, evidence_based

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

    def _fallback_candidates(self, questions: List[Question], lens_terms: List[str]) -> List[Question]:
        """
        Keep a small evidence-backed fallback set when score/evidence gates remove everything.
        """
        candidates: List[Question] = []
        for q in questions:
            if q.validation_status == ValidationStatus.REJECTED:
                continue  # Added hard rejection guard so failed questions cannot re-enter via fallback.
            if q.validation_score is not None and q.validation_score < self.config.min_score_refine:
                continue  # Added minimum score guard so fallback keeps only minimally viable questions.
            if q.evidence_doc_ids:
                candidates.append(q)
                continue
            if q.answerability_sample:
                q.evidence_doc_ids = list(q.answerability_sample)
                candidates.append(q)

        ranked = sorted(
            candidates,
            key=lambda q: (
                self._question_lens_score(q, lens_terms),  # Added lens-first sorting so fallback questions reflect user-defined interests.
                q.validation_score or 0,
                len(q.evidence_doc_ids),
            ),
            reverse=True,
        )
        fallback_n = max(1, self.config.min_questions)
        return ranked[:fallback_n]

    def _filter_rejected(self, questions: List[Question]) -> List[Question]:
        """Drop explicitly rejected questions before final score/evidence filtering."""
        filtered = [q for q in questions if q.validation_status != ValidationStatus.REJECTED]
        return filtered or questions  # Preserve backward compatibility when status metadata is missing.

    def _normalize_lens_terms(self, research_lens: List[str] | None) -> List[str]:
        """Normalize lens terms to lowercase deduplicated phrases."""
        if not research_lens:
            return []
        terms: List[str] = []
        seen = set()
        for raw in research_lens:
            term = str(raw or "").strip().lower()
            if not term or term in seen:
                continue
            seen.add(term)
            terms.append(term)
        # Added alias expansion so domain-equivalent words (injury/accident, job/occupation) influence ranking consistently.
        if any(any(k in term for k in INJURY_KEYWORDS) for term in terms):
            for alias in INJURY_KEYWORDS:
                if alias not in seen:
                    seen.add(alias)
                    terms.append(alias)
        if any(any(k in term for k in JOB_KEYWORDS) for term in terms):
            for alias in JOB_KEYWORDS:
                if alias not in seen:
                    seen.add(alias)
                    terms.append(alias)
        if any(any(k in term for k in PROCESS_KEYWORDS) for term in terms):
            for alias in PROCESS_KEYWORDS:
                if alias not in seen:
                    seen.add(alias)
                    terms.append(alias)  # Added process alias expansion so varied user phrasing still maps to the same process focus.
        return terms[:20]

    def _question_lens_score(self, question: Question, lens_terms: List[str]) -> int:
        """Score question text alignment to active lens terms."""
        if not lens_terms:
            return 0
        text = (question.question_text or "").lower()
        score = 0
        for term in lens_terms:
            if term in text:
                score += 3
            else:
                # Added token fallback so multi-word lens phrases still influence scoring.
                for token in [t for t in term.split() if len(t) >= 4]:
                    if token in text:
                        score += 1
        return score

    def _lens_requests_injury_and_job(self, lens_terms: List[str]) -> bool:
        """Return True when lens terms ask for both injury and job/occupation analysis."""
        if not lens_terms:
            return False
        has_injury = any(any(k in term for k in INJURY_KEYWORDS) for term in lens_terms)
        has_job = any(any(k in term for k in JOB_KEYWORDS) for term in lens_terms)
        return has_injury and has_job

    def _lens_requests_process(self, lens_terms: List[str]) -> bool:
        """Return True when lens terms ask for administrative process or workflow analysis."""
        if not lens_terms:
            return False
        return any(any(k in term for k in PROCESS_KEYWORDS) for term in lens_terms)

    def _inject_lens_seed_questions(
        self,
        questions: List[Question],
        notebook: ResearchNotebook,
        lens_terms: List[str],
    ) -> List[Question]:
        """Inject at most one targeted seed per supported lens family."""
        seeded = self._inject_injury_job_lens_question(questions, notebook, lens_terms)
        return self._inject_process_lens_question(seeded, notebook, lens_terms)

    def _inject_injury_job_lens_question(
        self,
        questions: List[Question],
        notebook: ResearchNotebook,
        lens_terms: List[str],
    ) -> List[Question]:
        """Inject one comparative injury-by-occupation question when lens requests it and none exists."""
        if not self._lens_requests_injury_and_job(lens_terms):
            return questions

        for q in questions:
            text = (q.question_text or "").lower()
            if any(k in text for k in INJURY_KEYWORDS) and any(k in text for k in JOB_KEYWORDS):
                return questions

        evidence_doc_ids: List[str] = []
        for q in questions:
            evidence_doc_ids.extend(q.evidence_doc_ids)

        for pattern in notebook.patterns.values():
            text = (pattern.pattern_text or "").lower()
            if any(k in text for k in INJURY_KEYWORDS):
                evidence_doc_ids.extend(pattern.evidence_doc_ids)

        for indicator in notebook.group_indicators.values():
            group_type = (indicator.group_type or "").lower()
            label = (indicator.label or "").lower()
            if group_type in {"occupation", "class"} or any(k in label for k in JOB_KEYWORDS):
                evidence_doc_ids.extend(indicator.evidence_doc_ids)

        deduped_ids: List[str] = []
        seen = set()
        for doc_id in evidence_doc_ids:
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            deduped_ids.append(doc_id)

        seeded = Question(
            question_text="How did kinds of injuries differ across occupations and departments, and which job types appeared most often in disablement records?",
            question_type=QuestionType.COMPARATIVE,
            why_interesting="This directly tests whether risk and recorded injury burden were patterned by work role rather than evenly distributed.",
            evidence_doc_ids=deduped_ids[:12],
            generation_method="lens_seeded",
            validation_score=max(self.config.min_score_accept + 15, 65),
        )
        return questions + [seeded]

    def _inject_process_lens_question(
        self,
        questions: List[Question],
        notebook: ResearchNotebook,
        lens_terms: List[str],
    ) -> List[Question]:
        """Inject one process-oriented question when the lens requests workflow/process analysis."""
        if not self._lens_requests_process(lens_terms):
            return questions

        for q in questions:
            text = (q.question_text or "").lower()
            if any(k in text for k in PROCESS_KEYWORDS):
                return questions

        evidence_doc_ids: List[str] = []
        for q in questions:
            evidence_doc_ids.extend(q.evidence_doc_ids)

        for pattern in notebook.patterns.values():
            text = (pattern.pattern_text or "").lower()
            if any(k in text for k in PROCESS_KEYWORDS):
                evidence_doc_ids.extend(pattern.evidence_doc_ids)

        for contra in notebook.contradictions:
            evidence_doc_ids.extend([contra.source_a, contra.source_b])

        deduped_ids: List[str] = []
        seen = set()
        for doc_id in evidence_doc_ids:
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            deduped_ids.append(doc_id)

        seeded = Question(
            question_text="How did claim-processing procedures (medical certification, superintendent review, and return-to-duty approval) vary across divisions and over time?",
            question_type=QuestionType.INSTITUTIONAL,
            why_interesting="This centers institutional workflow while testing whether administrative practice was consistent or uneven across the system.",
            evidence_doc_ids=deduped_ids[:12],
            generation_method="lens_seeded_process",
            validation_score=max(self.config.min_score_accept + 12, 62),
        )
        return questions + [seeded]

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

    def _assess_meaningfulness(
        self,
        questions: List[Question],
        notebook: ResearchNotebook,
        lens_terms: List[str],
    ) -> Dict[str, Any]:
        """Decide whether downstream synthesis should proceed based on question quality."""
        heuristic = self._heuristic_meaningfulness(questions, notebook)
        llm_gate = self._llm_meaningfulness(questions, notebook, lens_terms)
        if heuristic.get("action") == "stop":
            # Heuristic stop takes precedence because it guards known failure modes deterministically.
            if llm_gate:
                heuristic.setdefault("issues", []).extend(llm_gate.get("issues", []))
            return heuristic
        if llm_gate and llm_gate.get("action") == "stop":
            return llm_gate  # Added LLM veto so weak-but-plausible sets can still be halted before synthesis.
        if llm_gate and llm_gate.get("action") == "proceed_with_caution":
            heuristic["action"] = "proceed_with_caution"
            heuristic.setdefault("issues", []).extend(llm_gate.get("issues", []))
            heuristic["reason"] = llm_gate.get("reason") or heuristic.get("reason")
        return heuristic

    def _heuristic_meaningfulness(self, questions: List[Question], notebook: ResearchNotebook) -> Dict[str, Any]:
        """Deterministic safety checks for degenerate question sets."""
        min_docs = APP_CONFIG.tier0.question_min_evidence_docs
        credible = [
            q for q in questions
            if q.validation_status != ValidationStatus.REJECTED
            and (q.validation_score is None or q.validation_score >= self.config.min_score_refine)
            and len(q.evidence_doc_ids) >= min_docs
        ]
        seeded_only = bool(credible) and all((q.generation_method or "").startswith("lens_seeded") for q in credible)
        pattern_count = len(notebook.patterns)
        contradiction_count = len(notebook.contradictions)

        if not credible:
            return {
                "action": "stop",
                "reason": "No credible questions remained after validation and evidence checks.",
                "issues": ["all_questions_rejected_or_weak"],
                "confidence": "high",
            }
        if pattern_count == 0 and contradiction_count == 0 and len(credible) < 2:
            return {
                "action": "stop",
                "reason": "Notebook signal is too thin (no patterns/contradictions) to support synthesis.",
                "issues": ["thin_notebook_signal", "too_few_credible_questions"],
                "confidence": "high",
            }
        if len(credible) == 1 and seeded_only:
            return {
                "action": "stop",
                "reason": "Only one seed question remained; synthesis would overfit to injected guidance.",
                "issues": ["single_seeded_question"],
                "confidence": "high",
            }
        if len(credible) < 2:
            return {
                "action": "proceed_with_caution",
                "reason": "Question set is narrowly supported; synthesis quality may be unstable.",
                "issues": ["low_question_count"],
                "confidence": "medium",
            }
        return {
            "action": "proceed",
            "reason": "Question set passed deterministic quality checks.",
            "issues": [],
            "confidence": "medium",
        }

    def _llm_meaningfulness(
        self,
        questions: List[Question],
        notebook: ResearchNotebook,
        lens_terms: List[str],
    ) -> Dict[str, Any]:
        """Run an LLM audit that can flag semantically incoherent question sets."""
        if not questions:
            return {
                "action": "stop",
                "reason": "No questions were produced.",
                "issues": ["empty_question_set"],
                "confidence": "high",
            }

        summaries = self._question_gate_summaries(questions)
        prompt = MEANINGFULNESS_GATE_PROMPT.format(
            lens_terms=", ".join(lens_terms) if lens_terms else "(none)",
            documents_read=notebook.corpus_map.get("total_documents_read", 0),
            pattern_count=len(notebook.patterns),
            contradiction_count=len(notebook.contradictions),
            entity_count=len(notebook.entities),
            question_summaries=summaries,
        )

        response = self.validator.llm.generate(
            messages=[
                {"role": "system", "content": "You are a strict historical research quality auditor."},
                {"role": "user", "content": prompt},
            ],
            profile="verifier",
            temperature=0.0,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )
        if not response.success:
            return {}

        parsed = parse_llm_json(response.content, default={})
        if not isinstance(parsed, dict):
            return {}

        action = str(parsed.get("action", "")).strip().lower()
        if action not in {"proceed", "proceed_with_caution", "stop"}:
            return {}

        reason = str(parsed.get("reason", "")).strip()
        issues = parsed.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        confidence = str(parsed.get("confidence", "low")).strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        return {
            "action": action,
            "reason": reason or "LLM quality gate decision.",
            "issues": [str(issue) for issue in issues if str(issue).strip()],
            "confidence": confidence,
        }

    def _question_gate_summaries(self, questions: List[Question], limit: int = 8) -> str:
        """Render compact question summaries for the meaningfulness gate prompt."""
        lines: List[str] = []
        for idx, question in enumerate(questions[:limit], start=1):
            status = question.validation_status.value if question.validation_status else "unknown"
            line = (
                f"{idx}. score={question.validation_score} status={status} "
                f"method={question.generation_method} evidence={len(question.evidence_doc_ids)} "
                f"question={question.question_text}"
            )
            lines.append(line)
        return "\n".join(lines)

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

    def _rank_and_select(self, questions: List[Question], lens_terms: List[str]) -> List[Question]:
        sorted_questions = sorted(
            questions,
            key=lambda q: (
                self._question_lens_score(q, lens_terms),  # Added lens-aware rank boost for final question selection.
                q.validation_score or 0,
                len(q.evidence_doc_ids),
            ),
            reverse=True,
        )
        target_n = (
            self.config.target_questions
            if len(sorted_questions) >= self.config.target_questions
            else max(self.config.min_questions, len(sorted_questions))
        )
        if not lens_terms:
            return sorted_questions[:target_n]

        lens_cap = max(1, int(target_n * 0.7))  # Cap lens-dominant slots so archive-driven counterevidence still surfaces in the final set.
        selected: List[Question] = []
        selected_keys = set()
        lens_selected = 0
        for q in sorted_questions:
            key = q.question_text.strip().lower()
            if key in selected_keys:
                continue
            aligned = self._question_lens_score(q, lens_terms) > 0
            if aligned and lens_selected >= lens_cap:
                continue
            selected.append(q)
            selected_keys.add(key)
            if aligned:
                lens_selected += 1
            if len(selected) >= target_n:
                break

        for q in sorted_questions:
            if len(selected) >= target_n:
                break
            key = q.question_text.strip().lower()
            if key in selected_keys:
                continue
            selected.append(q)
            selected_keys.add(key)

        return selected


# Convenience helper

def generate_questions(notebook: ResearchNotebook, research_lens: List[str] | None = None) -> QuestionBatch:
    pipeline = QuestionGenerationPipeline()
    return pipeline.generate(notebook, research_lens=research_lens)
