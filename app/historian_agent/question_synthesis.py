# app/historian_agent/question_synthesis.py
# Created: 2026-02-06
# Purpose: Synthesize question hierarchy, buckets, and gaps

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import re

from rag_base import debug_print
from config import APP_CONFIG
from llm_abstraction import LLMClient
from tier0_utils import parse_llm_json
from question_models import Question
from research_notebook import ResearchNotebook, Pattern


@dataclass
class ThemeDefinition:
    key: str
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)


THEMES: List[ThemeDefinition] = [
    ThemeDefinition(
        key="institutional_welfare",
        name="Institutional Welfare Logic",
        description="Rules, eligibility, and governance of Relief/benefit systems",
        keywords=["relief", "feature", "membership", "eligibility", "rules", "procedure", "policy"],
    ),
    ThemeDefinition(
        key="medical_certification",
        name="Medical Certification as Labor Control",
        description="Medical exams, disability certificates, return-to-work decisions",
        keywords=["medical", "exam", "doctor", "certificate", "disability", "return to duty", "x-ray"],
    ),
    ThemeDefinition(
        key="mobility_regional",
        name="Mobility & Regional Labor Markets",
        description="Transfers, relocations, and geographic variance",
        keywords=["transfer", "relocation", "division", "region", "cumberland", "keyser", "lorain", "parkersburg"],
    ),
    ThemeDefinition(
        key="record_keeping",
        name="Record-Keeping & Administrative Reliability",
        description="Contradictions, name variants, certificate numbers, and data quality",
        keywords=["certificate", "record", "number", "reported", "returned", "conflict", "contradiction"],
    ),
    ThemeDefinition(
        key="benefit_economics",
        name="Benefit Economics & Reciprocity",
        description="Contributions, payouts, costs, and worker benefit burden",
        keywords=["contribution", "wage", "payment", "benefit", "dues", "monthly"],
    ),
]


THEME_SYNTHESIS_PROMPT = """You are a historian synthesizing research themes from questions.

CLOSED-WORLD RULES:
- Use ONLY the questions and patterns provided below.
- Do NOT introduce outside context.
- Create theme buckets that best explain the evidence.

CRONON-STYLE GUIDELINES:
- Make themes specific in time/place/actors when possible.
- Emphasize "why then" and "why there" when evidence allows.
- Favor questions that can be filled into: "I am studying ___ because I want to know ___ in order to help my readers understand ___."

QUESTIONS (sample):
{questions}

PATTERNS (sample):
{patterns}

TASK:
Generate {theme_count} theme buckets. For each, return:
- name: short theme name
- description: 1-2 sentences
- keywords: 6-12 lowercase keywords/phrases for assignment
- scope: {{
    "time": "year range or period if evident",
    "place": "locations/divisions if evident",
    "actors": ["groups or institutions if evident"]
  }}

Return ONLY JSON array:
[
  {{
    "name": "...",
    "description": "...",
    "keywords": ["...", "..."],
    "scope": {{"time": "...", "place": "...", "actors": ["...", "..."]}}
  }}
]
"""


GRAND_NARRATIVE_PROMPT = """You are a historian formulating a highest-order research question.

CLOSED-WORLD RULES:
- Use ONLY the questions and patterns provided below.
- Do NOT introduce outside context.

CRONON-STYLE FRAME:
1) Fill this structure:
   "I am studying ____ because I want to know ____ in order to help my readers understand ____."
2) Explicitly address:
   - Why then? (why this time period)
   - Why there? (why this place/organization)
3) Identify assumptions or terms that need definition.

QUESTIONS (sample):
{questions}

PATTERNS (sample):
{patterns}

THEMES (sample):
{themes}

Return ONLY JSON:
{{
  "question": "single concise question",
  "purpose_statement": "I am studying ... because ... in order to ...",
  "scope": {{"time": "...", "place": "...", "actors": ["...", "..."]}},
  "why_then": "...",
  "why_there": "...",
  "assumptions_to_check": ["...", "..."],
  "terms_to_define": ["...", "..."]
}}
"""


GRAND_NARRATIVE_REPAIR_PROMPT = """You are auditing a highest-order research question for evidence alignment.

CLOSED-WORLD RULES:
- Use ONLY the questions and patterns provided below.
- Remove or rewrite any claims that are not supported.
- If a field cannot be supported, set it to "" or "insufficient evidence".

QUESTIONS (sample):
{questions}

PATTERNS (sample):
{patterns}

PROPOSED JSON:
{proposed}

Return ONLY corrected JSON with the same schema as the proposed JSON.
"""

GROUP_DIFFERENCE_PROMPT = """You are a historian looking for differences across groups over time.

CLOSED-WORLD RULES:
- Use ONLY the questions and patterns provided below.
- Do NOT introduce outside context.
- If evidence of group differences is not present, return an empty list.

GROUP AXES TO CONSIDER:
{group_axes}

QUESTIONS (sample):
{questions}

PATTERNS (sample):
{patterns}

Return ONLY JSON array of up to {max_questions} items:
[
  {{
    "question": "comparison question",
    "groups": ["group A", "group B"],
    "time_window": "if evident",
    "evidence_basis": "short phrase referencing patterns/questions"
  }}
]
"""


GAP_AXES = [
    {
        "key": "demographic",
        "title": "Demographic coverage",
        "keywords": ["women", "female", "race", "black", "african", "age", "youth", "child", "ethnicity", "immigrant", "national origin"],
        "questions": [
            "Were women represented in the Relief Department records, and if so, how were their cases handled?",
            "Do the records indicate differences in treatment by race or age?",
        ],
    },
    {
        "key": "group_difference",
        "title": "Group differences (race/gender/class/ethnicity/origin/occupation)",
        "keywords": ["gender", "race", "class", "ethnicity", "national origin", "occupation", "job class", "laborer", "engineer", "brakeman"],
        "questions": [
            "How did Relief Department policies differ by occupation or job class?",
            "Do the records show differences in access or outcomes by gender, race, or national origin?",
        ],
    },
    {
        "key": "economic",
        "title": "Economic magnitude",
        "keywords": ["wage", "salary", "dollar", "contribution", "benefit amount", "payout"],
        "questions": [
            "What were the contribution rates and benefit payouts for Relief Feature members?",
            "How did benefit levels compare to wages over time?",
        ],
    },
    {
        "key": "labor_relations",
        "title": "Labor relations",
        "keywords": ["union", "strike", "grievance", "dispute", "brotherhood"],
        "questions": [
            "What role (if any) did unions play in Relief Department policies or access?",
            "Did Relief benefits intersect with labor disputes?",
        ],
    },
    {
        "key": "family_impact",
        "title": "Family impact",
        "keywords": ["widow", "dependent", "family", "children", "beneficiary"],
        "questions": [
            "How were dependents or surviving family members treated in Relief cases?",
        ],
    },
    {
        "key": "comparative",
        "title": "Comparative context",
        "keywords": ["other railroad", "pennsylvania", "industry", "comparison"],
        "questions": [
            "How did B&O Relief practices compare to other railroads or industries?",
        ],
    },
]


class QuestionSynthesizer:
    """Synthesize questions into buckets, hierarchy, and gaps."""

    def __init__(self) -> None:
        self.llm = LLMClient()
        self._theme_cache: List[ThemeDefinition] | None = None

    def bucket_questions(self, questions: List[Question], themes: List[ThemeDefinition]) -> Dict[str, List[Question]]:
        buckets: Dict[str, List[Question]] = {t.key: [] for t in themes}
        buckets["other"] = []

        for question in questions:
            bucket_key = self._assign_bucket(question, themes)
            buckets.setdefault(bucket_key, []).append(question)

        return buckets

    def build_agenda(self, notebook: ResearchNotebook, questions: List[Question]) -> Dict[str, Any]:
        if not APP_CONFIG.tier0.synthesis_enabled:
            return {}

        themes = self._get_themes(notebook, questions)
        buckets = self.bucket_questions(questions, themes)
        themes_out: List[Dict[str, Any]] = []

        for theme in themes:
            theme_questions = buckets.get(theme.key, [])
            if not theme_questions:
                continue

            sorted_qs = sorted(
                theme_questions,
                key=lambda q: q.validation_score or 0,
                reverse=True,
            )

            broad = sorted_qs[0]
            sub_questions = [q.to_dict() for q in sorted_qs[1:6]]

            theme_patterns = self._patterns_for_theme(notebook, theme)

            themes_out.append({
                "theme": theme.name,
                "theme_key": theme.key,
                "theme_description": theme.description,
                "theme_keywords": theme.keywords,
                "overview_question": broad.to_dict(),
                "sub_questions": sub_questions,
                "inductive_logic": {
                    "patterns_observed": [p.pattern_text for p in theme_patterns[:3]],
                    "inference": f"Observed patterns suggest {theme.description.lower()} as a recurring mechanism in the corpus.",
                },
                "evidence_base": {
                    "pattern_count": len(theme_patterns),
                    "question_count": len(theme_questions),
                    "confidence_distribution": self._confidence_distribution(theme_patterns),
                },
            })

        group_difference_questions = self._group_difference_questions(notebook, questions)
        gaps = self._identify_gaps(notebook, questions)
        if not group_difference_questions:
            gaps.append({
                "gap": "Group differences (race/gender/class/ethnicity/origin/occupation)",
                "missing_evidence": ["race", "gender", "class", "ethnicity", "national origin", "occupation"],
                "suggested_questions": [
                    "How did Relief Department outcomes differ by occupation or job class?",
                    "Do the records show differences in access or outcomes by gender, race, or national origin?",
                ],
            })
        grand_question = self._grand_narrative_question(themes_out, notebook, questions)

        return {
            "grand_narrative": grand_question,
            "theme_definitions": [
                {
                    "key": t.key,
                    "name": t.name,
                    "description": t.description,
                    "keywords": t.keywords,
                }
                for t in themes
            ],
            "themes": themes_out,
            "group_difference_questions": group_difference_questions,
            "gaps": gaps,
        }

    def _assign_bucket(self, question: Question, themes: List[ThemeDefinition]) -> str:
        text = " ".join([
            question.question_text,
            question.why_interesting or "",
            question.pattern_source or "",
            question.contradiction_source or "",
        ]).lower()

        best_key = "other"
        best_score = 0

        for theme in themes:
            score = sum(1 for kw in theme.keywords if kw in text)
            if score > best_score:
                best_score = score
                best_key = theme.key

        return best_key

    def _patterns_for_theme(self, notebook: ResearchNotebook, theme: ThemeDefinition) -> List[Pattern]:
        patterns = list(notebook.patterns.values())
        if not theme.keywords:
            return patterns
        filtered = []
        for pattern in patterns:
            text = pattern.pattern_text.lower()
            if any(kw in text for kw in theme.keywords):
                filtered.append(pattern)
        if not filtered:
            return patterns
        return filtered

    def _get_themes(self, notebook: ResearchNotebook, questions: List[Question]) -> List[ThemeDefinition]:
        if self._theme_cache:
            return self._theme_cache

        if not APP_CONFIG.tier0.synthesis_dynamic or not questions:
            self._theme_cache = THEMES
            return THEMES

        generated = self._generate_dynamic_themes(notebook, questions)
        if generated:
            self._theme_cache = generated
            return generated

        self._theme_cache = THEMES
        return THEMES

    def _generate_dynamic_themes(
        self,
        notebook: ResearchNotebook,
        questions: List[Question],
    ) -> List[ThemeDefinition]:
        question_sample = sorted(
            questions,
            key=lambda q: q.validation_score or 0,
            reverse=True,
        )[: APP_CONFIG.tier0.synthesis_max_question_sample]
        pattern_sample = list(notebook.patterns.values())[: APP_CONFIG.tier0.synthesis_max_pattern_sample]

        questions_text = "\n".join(f"- {q.question_text}" for q in question_sample) or "- (none)"
        patterns_text = "\n".join(f"- {p.pattern_text}" for p in pattern_sample) or "- (none)"

        prompt = THEME_SYNTHESIS_PROMPT.format(
            questions=questions_text,
            patterns=patterns_text,
            theme_count=APP_CONFIG.tier0.synthesis_theme_count,
        )

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You synthesize research themes from evidence only."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.2,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if not response.success:
            debug_print("Theme synthesis failed; falling back to static themes.")
            return []

        data = parse_llm_json(response.content, default=[])
        if not isinstance(data, list):
            return []

        themes: List[ThemeDefinition] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            description = str(item.get("description") or "").strip()
            keywords = item.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(k).lower() for k in keywords if k]
            key = self._slugify(name)
            themes.append(ThemeDefinition(key=key, name=name, description=description, keywords=keywords))

        return themes

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        return slug or "theme"

    def _confidence_distribution(self, patterns: List[Pattern]) -> Dict[str, int]:
        dist = {"high": 0, "medium": 0, "low": 0}
        for pattern in patterns:
            if pattern.confidence in dist:
                dist[pattern.confidence] += 1
        return dist

    def _identify_gaps(self, notebook: ResearchNotebook, questions: List[Question]) -> List[Dict[str, Any]]:
        text_blob = " ".join([
            q.question_text for q in questions
        ] + [
            p.pattern_text for p in notebook.patterns.values()
        ]).lower()

        gaps = []
        for axis in GAP_AXES:
            if not any(kw in text_blob for kw in axis["keywords"]):
                gaps.append({
                    "gap": axis["title"],
                    "missing_evidence": axis["keywords"],
                    "suggested_questions": axis["questions"],
                })
        return gaps

    def _grand_narrative_question(
        self,
        themes_out: List[Dict[str, Any]],
        notebook: ResearchNotebook,
        questions: List[Question],
    ) -> Dict[str, Any]:
        if not themes_out:
            return {}

        theme_names = [t["theme"] for t in themes_out[:3]]
        question_sample = sorted(
            questions,
            key=lambda q: q.validation_score or 0,
            reverse=True,
        )[: APP_CONFIG.tier0.synthesis_max_question_sample]
        pattern_sample = list(notebook.patterns.values())[: APP_CONFIG.tier0.synthesis_max_pattern_sample]

        questions_text = "\n".join(f"- {q.question_text}" for q in question_sample) or "- (none)"
        patterns_text = "\n".join(f"- {p.pattern_text}" for p in pattern_sample) or "- (none)"
        themes_text = "\n".join(f"- {name}" for name in theme_names) or "- (none)"

        prompt = GRAND_NARRATIVE_PROMPT.format(
            questions=questions_text,
            patterns=patterns_text,
            themes=themes_text,
        )

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian refining a research question."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.2,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if not response.success:
            question = (
                "How did the Relief Department shape worker welfare and labor control, "
                f"as reflected across {', '.join(theme_names)} in the archival record?"
            )
            return {
                "question": question,
                "themes": theme_names,
            }

        data = parse_llm_json(response.content, default={})
        if not isinstance(data, dict):
            return {}

        repair_prompt = GRAND_NARRATIVE_REPAIR_PROMPT.format(
            questions=questions_text,
            patterns=patterns_text,
            proposed=json.dumps(data, ensure_ascii=True),
        )
        repair_response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You enforce closed-world evidence alignment."},
                {"role": "user", "content": repair_prompt},
            ],
            profile="verifier",
            temperature=0.0,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        repaired = parse_llm_json(repair_response.content, default={}) if repair_response.success else data
        if not isinstance(repaired, dict):
            repaired = data

        repaired["themes"] = theme_names
        return repaired

    def _group_difference_questions(
        self,
        notebook: ResearchNotebook,
        questions: List[Question],
    ) -> List[Dict[str, Any]]:
        if not questions:
            return []

        question_sample = sorted(
            questions,
            key=lambda q: q.validation_score or 0,
            reverse=True,
        )[: APP_CONFIG.tier0.synthesis_max_question_sample]
        pattern_sample = list(notebook.patterns.values())[: APP_CONFIG.tier0.synthesis_max_pattern_sample]

        questions_text = "\n".join(f"- {q.question_text}" for q in question_sample) or "- (none)"
        patterns_text = "\n".join(f"- {p.pattern_text}" for p in pattern_sample) or "- (none)"

        group_axes = [
            "race",
            "gender",
            "class",
            "ethnicity",
            "national origin",
            "occupation",
            "job category",
        ]
        prompt = GROUP_DIFFERENCE_PROMPT.format(
            group_axes=", ".join(group_axes),
            questions=questions_text,
            patterns=patterns_text,
            max_questions=5,
        )

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You identify group-comparison questions from evidence."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.2,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if not response.success:
            return []

        data = parse_llm_json(response.content, default=[])
        if isinstance(data, dict) and "questions" in data:
            data = data.get("questions", [])
        if not isinstance(data, list):
            return []

        cleaned = []
        for item in data:
            if not isinstance(item, dict) or not item.get("question"):
                continue
            item.setdefault("groups", [])
            item.setdefault("time_window", "")
            item.setdefault("evidence_basis", "")
            cleaned.append(item)
        return cleaned
