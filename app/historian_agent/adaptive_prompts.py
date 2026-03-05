# app/historian_agent/adaptive_prompts.py
# Purpose: Centralized prompt variants for adaptive explorer A/B/C testing.

from __future__ import annotations

from typing import Dict

from config import APP_CONFIG


VALID_PROMPT_VARIANTS = {"v1", "v2", "v3"}


def normalize_prompt_variant(raw: str) -> str:
    """Return a stable prompt variant key for adaptive mode."""
    value = str(raw or "").strip().lower()
    return value if value in VALID_PROMPT_VARIANTS else "v1"


def resolve_prompt_variant(override: str | None = None) -> str:
    """Resolve runtime override first, then config default."""
    if override:
        return normalize_prompt_variant(override)
    return normalize_prompt_variant(getattr(APP_CONFIG.tier0, "adaptive_prompt_variant", "v1"))


ADAPTIVE_BATCH_SYSTEM_MESSAGES: Dict[str, str] = {
    "v1": (
        "You are a social historian performing INDUCTIVE analysis of archival documents. "
        "Find patterns across documents and avoid single-person factoids."
    ),
    "v2": (
        "You are a social historian building an evidence graph while reading. "
        "Prioritize cross-document mechanisms, contrasts, and change-over-time patterns."
    ),
    "v3": (
        "You are a historian constructing question threads from micro observations to meso/macro hypotheses. "
        "Do not output isolated trivia; output analytically useful questions with evidence anchors."
    ),
}


ADAPTIVE_BATCH_ANALYSIS_PROMPTS: Dict[str, str] = {
    "v1": """You are a historian systematically reading archival documents.

CLOSED-WORLD RULES:
- Use ONLY the documents provided below.
- Do NOT use outside knowledge or assumptions.
- If a fact is not in the documents, do not include it.
- Prioritize USER RESEARCH LENS topics when evidence exists.

PRIOR KNOWLEDGE (summary from previous batches):
{prior_knowledge}

USER RESEARCH LENS (prioritized topics for this run):
{research_lens}

DOCUMENTS (closed-world JSON objects):
{documents_json}

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
""",
    "v2": """You are a historian systematically reading archival documents.

CLOSED-WORLD RULES:
- Use ONLY the documents below.
- If a claim is not in the documents, omit it.
- Prefer false negatives over false positives.
- Questions must be useful for cross-document explanation, not retrieval.

ANALYTIC TARGET:
- Surface mechanisms, comparisons, and change-over-time.
- Track differences across axes in the research lens.
- Tie every pattern/question to concrete evidence blocks.

PRIOR KNOWLEDGE (summary from previous batches):
{prior_knowledge}

USER RESEARCH LENS (prioritized topics for this run):
{research_lens}

DOCUMENTS (closed-world JSON objects):
{documents_json}

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
    {{
      "question": str,
      "why_interesting": str,
      "evidence_needed": str,
      "related_entities": [],
      "time_window": str,
      "evidence_blocks": ["block_id"],
      "axis_tags": [str],
      "suggested_level": "micro|meso_hint"
    }}
  ],
  "temporal_events": {{"year": [str]}}
}}

QUESTION RULES:
- Avoid single-person lookup questions.
- Use "why/how/compare/change" framing whenever possible.
- Return [] if no meaningful cross-document question exists.
""",
    "v3": """You are a historian reading an archive as a question-building process.

CLOSED-WORLD RULES:
- Use ONLY DOCUMENTS.
- No outside facts, no inferred identities.
- Every evidence reference must be a valid block_id.

GOAL:
- Build a ladder from narrow observations to broader hypotheses.
- Preserve unrelated directions when evidence diverges.
- Prefer "why/how/change/compare" framing over "what happened".

PRIOR KNOWLEDGE:
{prior_knowledge}

RESEARCH LENS:
{research_lens}

DOCUMENTS:
{documents_json}

Return ONLY valid JSON:
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
    {{
      "question": str,
      "why_interesting": str,
      "evidence_needed": str,
      "related_entities": [],
      "time_window": str,
      "evidence_blocks": ["block_id"],
      "axis_tags": [str],
      "suggested_level": "micro|meso_hint|macro_hint",
      "parent_hint": str
    }}
  ],
  "temporal_events": {{"year": [str]}}
}}

QUESTION QUALITY FILTER:
- Reject person-specific trivia.
- Keep only questions that can accumulate evidence across documents.
- Include at most 6 questions, highest analytic value first.
""",
}


WHY_HOW_PROMPTS: Dict[str, str] = {
    "v1": """QUESTION: {question_text}

Historian's rule: good historical questions ask "why", "how", "how did X change
over time", or invite comparison — NOT "what happened" or purely descriptive questions.

Evaluate:
1. If already "why", "how", "change over time", or comparative → return UNCHANGED
2. If "what happened" or purely descriptive → rewrite as "why" or "how",
   preserving exact subject, scope, time period, and people. Change framing only.

Return JSON only: {{"question": str, "question_type": str, "changed": bool}}
Valid question_type: "why", "how", "compare", "change_continuity", "explain", "what"
""",
    "v2": """QUESTION: {question_text}

You are enforcing historian-grade question framing.

Rules:
- Prefer "why/how/compare/change_continuity/explain".
- Keep scope identical to the original wording.
- Do not broaden to abstract generalities.
- If the input is a direct factual lookup that must stay micro, keep "what".

Return JSON only: {{"question": str, "question_type": str, "changed": bool}}
Valid question_type: "why", "how", "compare", "change_continuity", "explain", "what"
""",
    "v3": """QUESTION: {question_text}

Transform weak archival questions into analytic ones when possible.

Hard constraints:
- Preserve actors, place, and period from the input.
- Rewrite descriptive/lookup framing into "why/how/compare/change" if feasible.
- Use "what" only when the question is strictly micro-factual and cannot be reframed
  without changing meaning.

Return JSON only: {{"question": str, "question_type": str, "changed": bool}}
Valid question_type: "why", "how", "compare", "change_continuity", "explain", "what"
""",
}


PROMOTION_PROMPTS: Dict[str, str] = {
    "v1": """You are promoting a historical research question to a higher analytical level.

CURRENT QUESTION ({current_level}): {question_text}
SUB-QUESTIONS:
{children_text}
EVIDENCE SUMMARY:
{evidence_summary}
{tension_note}

Write a single broader question at the {target_level} level that:
- Asks "why", "how", "how did X change over time", or invites structural comparison
- Is NOT a "what happened" question
- Does not invent facts not in the evidence summary
- Is specific enough to be answerable within this corpus

Return JSON only: {{"question": str, "question_type": str}}
Valid question_type: "why", "how", "compare", "change_continuity", "explain"
""",
    "v2": """You are promoting archival questions from local observations to higher-level explanation.

CURRENT QUESTION ({current_level}): {question_text}
SUB-QUESTIONS:
{children_text}
EVIDENCE SUMMARY:
{evidence_summary}
{tension_note}

Write ONE {target_level} question that:
- Preserves the specific phenomenon in the current question
- Adds explanatory leverage (mechanism, comparison, or change over time)
- Avoids generic wording ("How did injury happen?") and avoids "what happened"
- Stays answerable from the corpus evidence represented here

Return JSON only: {{"question": str, "question_type": str}}
Valid question_type: "why", "how", "compare", "change_continuity", "explain"
""",
    "v3": """You are building a multi-level historian question graph.

CURRENT QUESTION ({current_level}): {question_text}
SUB-QUESTIONS:
{children_text}
EVIDENCE SUMMARY:
{evidence_summary}
{tension_note}

Produce ONE {target_level} question that integrates sub-questions while retaining contestation.
- Must be why/how/compare/change_continuity/explain
- Must include at least one differentiating axis or condition implied by evidence
- Must not resolve contradictions; preserve them as analytic tension
- Must not be a "what" question

Return JSON only: {{"question": str, "question_type": str}}
Valid question_type: "why", "how", "compare", "change_continuity", "explain"
""",
}


CHANGE_CONTINUITY_PROMPTS: Dict[str, str] = {
    "v1": """These questions address a common phenomenon across different time periods:
{questions}

Time span in corpus evidence: {min_year}–{max_year}

Write a single "change and continuity" question asking how or why this phenomenon
changed (or persisted) across this period. Must be "how" or "why". No invented facts.

Return JSON only: {{"question": str}}
""",
    "v2": """These questions concern the same phenomenon over time:
{questions}

Evidence years: {min_year}–{max_year}

Write one change/continuity question that:
- asks how/why the pattern changed, persisted, or diverged
- names the relevant period span implicitly or explicitly
- remains answerable from this corpus (no invented factors)

Return JSON only: {{"question": str}}
""",
    "v3": """You are generating a historian change-and-continuity question.

QUESTIONS:
{questions}
TIME SPAN: {min_year}–{max_year}

Return one question that foregrounds mechanism of change OR persistence across periods.
Avoid generic wording. Keep it analyzable with available archival evidence.

Return JSON only: {{"question": str}}
""",
}


SEED_EXTRACTION_PROMPTS: Dict[str, str] = {
    "v1": """You are extracting broad historical research seeds.

INPUT:
{text}

Rules:
- Return 2 to {max_questions} questions.
- Use only why/how/compare/change_continuity/explain framing.
- Do not generate "what happened" questions.
- Do not invent names, dates, or places not in INPUT.

Return JSON array with objects:
{{"question": str, "question_type": str, "tags": [str]}}
Allowed tags: {axes}
""",
    "v2": """You are extracting historian seed hypotheses before corpus reading.

INPUT:
{text}

Rules:
- Return 2 to {max_questions} broad, testable questions.
- Prefer explanatory framing (why/how/change/compare).
- Keep each seed on a distinct analytical dimension when possible.
- Never fabricate names, dates, locations, or institutions.

Return JSON array:
{{"question": str, "question_type": str, "tags": [str]}}
Allowed tags: {axes}
""",
    "v3": """Convert the research statement into high-value archival seed questions.

INPUT:
{text}

Requirements:
- 2 to {max_questions} seeds maximum.
- Each seed should be broad enough for corpus-wide inquiry but specific enough to falsify.
- Only why/how/compare/change_continuity/explain.
- No invented specifics.

Return JSON array only:
{{"question": str, "question_type": str, "tags": [str]}}
Allowed tags: {axes}
""",
}


EXPAND_DIRECTION_PROMPT = """QUESTION A: {q_a}
QUESTION B: {q_b}

One may be broader, or they may be lateral.
Reply with exactly one word: BROADER_A, BROADER_B, or LATERAL.
"""


def get_batch_analysis_prompt(variant: str) -> str:
    key = normalize_prompt_variant(variant)
    return ADAPTIVE_BATCH_ANALYSIS_PROMPTS.get(key, ADAPTIVE_BATCH_ANALYSIS_PROMPTS["v1"])


def get_batch_system_message(variant: str) -> str:
    key = normalize_prompt_variant(variant)
    return ADAPTIVE_BATCH_SYSTEM_MESSAGES.get(key, ADAPTIVE_BATCH_SYSTEM_MESSAGES["v1"])


def get_why_how_prompt(variant: str) -> str:
    key = normalize_prompt_variant(variant)
    return WHY_HOW_PROMPTS.get(key, WHY_HOW_PROMPTS["v1"])


def get_promotion_prompt(variant: str) -> str:
    key = normalize_prompt_variant(variant)
    return PROMOTION_PROMPTS.get(key, PROMOTION_PROMPTS["v1"])


def get_change_continuity_prompt(variant: str) -> str:
    key = normalize_prompt_variant(variant)
    return CHANGE_CONTINUITY_PROMPTS.get(key, CHANGE_CONTINUITY_PROMPTS["v1"])


def get_seed_extraction_prompt(variant: str) -> str:
    key = normalize_prompt_variant(variant)
    return SEED_EXTRACTION_PROMPTS.get(key, SEED_EXTRACTION_PROMPTS["v1"])

