# app/historian_agent/adaptive_prompts.py
# Purpose: Centralized prompt variants for adaptive explorer A/B/C testing.

from __future__ import annotations

from typing import Dict

from config import APP_CONFIG


VALID_PROMPT_VARIANTS = {"v1", "v2", "v3", "v4", "v5"}


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
    "v4": (
        "You are a social historian building an auditable evidence graph from archival documents. "
        "Output only high-value, cross-document questions with explicit block-level evidence anchors."
    ),
    "v5": (
        "You are a historian generating research questions as an archive-reading process: "
        "observation -> puzzle -> system-level explanation. Prioritize causal, institutional, "
        "social-structure, change-over-time, and experience/meaning questions."
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
    "v4": """INSTRUCTIONS (follow in order):
1. Read <PRIOR_KNOWLEDGE>, <RESEARCH_LENS>, and <DOCUMENTS>.
2. Extract only claims supported by explicit document block ids.
3. Produce high-value historian questions for cross-document explanation.
4. Return JSON only. Do not include commentary.

<CLOSED_WORLD_RULES>
- Use only facts present in <DOCUMENTS>.
- Never invent names, dates, places, institutions, or causal claims.
- Every pattern, contradiction, and question must cite at least one valid block_id from the input.
- Prefer false negatives over false positives.
</CLOSED_WORLD_RULES>

<QUESTION_RUBRIC>
- Keep only questions that can aggregate evidence across multiple documents or groups.
- Prefer "why/how/compare/change_continuity/explain" framing.
- Avoid single-person retrieval questions unless they explicitly test a broader mechanism.
- Include at most 6 questions ranked by analytic value.
- If no valid analytic question exists, return "questions": [].
</QUESTION_RUBRIC>

<GOOD_BAD_EXAMPLES>
GOOD: "How did injury compensation differ across occupations, and why?"
BAD: "What happened to John Smith on March 3?"
GOOD: "Why do contradictory injury reports cluster in specific years or offices?"
</GOOD_BAD_EXAMPLES>

<PRIOR_KNOWLEDGE>
{prior_knowledge}
</PRIOR_KNOWLEDGE>

<RESEARCH_LENS>
{research_lens}
</RESEARCH_LENS>

<DOCUMENTS>
{documents_json}
</DOCUMENTS>

Return EXACT JSON schema:
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
""",
    "v5": """You are reading archival evidence the way historians generate questions.

PROCESS (follow in order):
1) Identify observations in documents (repeated categories, anomalies, absences, contradictions).
2) Convert observations into puzzles ("why this pattern? why this record form? why this silence?").
3) Convert puzzles into system-level questions (institutions, incentives, hierarchy, change over time).
4) Keep only high-value questions that can aggregate evidence across multiple documents.

HISTORIAN QUESTION TYPES (target these):
- causal: why did X happen
- institutional: how did systems/organizations work
- social_structure: how hierarchy/power/inequality operated
- change_over_time: how/why something changed or persisted
- experience_meaning: how actors interpreted their world

QUALITY FILTER:
- Reject factoid/trivia prompts ("when", "who", "what year", "how many") unless tied to broader explanation.
- Reject vague pseudo-analytic prompts ("what impact", "what role", "what was life like") unless rewritten with mechanism and scope.
- Prefer questions where historians could reasonably disagree.
- Each question must cite evidence blocks.
- If no valid historian-grade question exists, return "questions": [].

CLOSED-WORLD RULES:
- Use only facts present in DOCUMENTS.
- Never invent names, dates, places, institutions, or causes.

PRIOR_KNOWLEDGE:
{prior_knowledge}

RESEARCH_LENS:
{research_lens}

DOCUMENTS:
{documents_json}

Return ONLY JSON schema:
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
    "v4": """TASK: Enforce historian-grade question framing.

QUESTION:
{question_text}

Rules:
- Keep the same actors, place, period, and scope.
- If descriptive/"what happened", reframe to why/how/compare/change when possible.
- Use "what" only when reframing would change meaning and the question is strictly micro-factual.

Decision examples:
- Input: "What injury did Adams sustain?" -> keep as "what" (micro factual).
- Input: "What happened to brakemen injuries over time?" -> rewrite as change/continuity.
- Input: "Why were reports delayed?" -> unchanged.

Return JSON only:
{{"question": str, "question_type": str, "changed": bool}}
Valid question_type: "why", "how", "compare", "change_continuity", "explain", "what"
""",
    "v5": """TASK: Rewrite to historian-grade framing when possible.

QUESTION:
{question_text}

Rules:
- Keep actors, place, and period fixed.
- Prefer "why/how/compare/change_continuity/explain".
- Block factoid framing ("when/who/what year/how many") unless the question is strictly micro-factual.
- Prefer questions that imply mechanism and allow disagreement.

Return JSON only:
{{"question": str, "question_type": str, "changed": bool}}
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
    "v4": """TASK: Promote a question to a higher analytical level without losing evidence constraints.

CURRENT QUESTION ({current_level}): {question_text}
SUB-QUESTIONS:
{children_text}
EVIDENCE SUMMARY:
{evidence_summary}
{tension_note}

Requirements:
- Output one {target_level} question only.
- Keep the same phenomenon and corpus scope.
- Increase explanatory leverage (mechanism, comparison, or change over time).
- Preserve contradictions as tension; do not resolve uncertainty.
- Must be answerable using the available corpus evidence.
- Must not be a "what" question.

Return JSON only: {{"question": str, "question_type": str}}
Valid question_type: "why", "how", "compare", "change_continuity", "explain"
""",
    "v5": """TASK: Promote archival questions using historian logic.

CURRENT QUESTION ({current_level}): {question_text}
SUB-QUESTIONS:
{children_text}
EVIDENCE SUMMARY:
{evidence_summary}
{tension_note}

Promotion requirements:
- Move from document-level puzzle to system-level explanation.
- Keep causal/institutional/social-structure/change dimensions explicit when supported.
- Preserve contradiction/tension if present.
- Keep question answerable by this corpus.
- Do not output "what" questions.

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
    "v4": """TASK: Write one historian-grade change/continuity question.

QUESTIONS:
{questions}
TIME SPAN: {min_year}–{max_year}

Rules:
- Ask how/why a pattern changed, persisted, or diverged across this span.
- Keep wording specific enough to test in this corpus.
- Do not invent causes not implied by the evidence context.
- Avoid generic phrasing.

Return JSON only: {{"question": str}}
""",
    "v5": """TASK: Write one change/continuity question at historian quality.

QUESTIONS:
{questions}
TIME SPAN: {min_year}–{max_year}

Rules:
- Ask how/why a pattern changed, persisted, or diverged across this span.
- Tie change to institutional, social, or causal mechanisms when evidence allows.
- Avoid generic phrasing and invented factors.

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
    "v4": """TASK: Extract seed questions for a historian before corpus reading.

INPUT:
{text}

Rules:
- Return 2 to {max_questions} seeds.
- Seeds must be broad but falsifiable within corpus reading.
- Allowed types: why, how, compare, change_continuity, explain.
- Do not output "what happened" questions.
- Do not invent names, dates, places, or institutions.
- Prefer distinct analytical dimensions when possible.

Output JSON array only:
{{"question": str, "question_type": str, "tags": [str]}}
Allowed tags: {axes}
""",
    "v5": """TASK: Extract seed questions from the research statement for archival inquiry.

INPUT:
{text}

Rules:
- Return 2 to {max_questions} seeds.
- Seeds must be explanation-oriented and testable in corpus reading.
- Prioritize these types: causal, institutional, social-structure, change-over-time, experience/meaning.
- Use only allowed question types: why, how, compare, change_continuity, explain.
- No invented specifics.

Output JSON array only:
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
