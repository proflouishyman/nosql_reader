# app/historian_agent/question_synthesis.py
# Created: 2026-02-06
# Purpose: Synthesize question hierarchy, buckets, and gaps

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import hashlib
import pickle
from pathlib import Path

from rag_base import debug_print
from config import APP_CONFIG
from llm_abstraction import LLMClient, LLMResponse
from tier0_utils import parse_llm_json, CheckpointManager
from question_models import Question
from research_notebook import ResearchNotebook, Pattern
from historian_agent.embeddings import EmbeddingService
from historian_agent.recursive_synthesis import RecursiveSynthesizer


@dataclass
class ThemeDefinition:
    key: str
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)


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
- Put boundaries on scope so themes are researchable and distinct.
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
Themes must be distinct and non-overlapping; avoid near-synonyms.

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


THEME_EXPANSION_PROMPT = """You are a historian expanding theme coverage.

CLOSED-WORLD RULES:
- Use ONLY the questions and patterns provided below.
- Do NOT introduce outside context.

EXISTING THEMES:
{themes}

QUESTIONS (sample):
{questions}

PATTERNS (sample):
{patterns}

TASK:
Propose up to {theme_count} NEW, distinct themes that do NOT overlap with existing themes.
Each theme must be substantively different.

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

THEME_REPAIR_PROMPT = """You are a strict JSON formatter.

INPUT (raw model output):
{raw_output}

TASK:
- Convert the input into a JSON array of theme objects with fields:
  - name
  - description
  - keywords
  - scope
- Do NOT add new content. Only reformat what is present.
- If no valid themes exist, return [].

Return ONLY JSON.
"""

INDUCTIVE_LOGIC_PROMPT = """You are a historian performing inductive reasoning.

CLOSED-WORLD RULES:
- Use ONLY the evidence provided below.
- Do NOT introduce outside context.
- If a conclusion is speculative, phrase it as tentative ("suggests", "may indicate").

THEME:
{theme_name}

EVIDENCE (patterns with counts):
{patterns}

CONTRADICTIONS (count={contradiction_count}):
{contradictions}

TASK:
Return ONLY JSON:
{{
  "patterns_observed": [str],
  "contradictions_observed": int,
  "inference": "short, cautious inference grounded in evidence",
  "historical_argument": "1-2 sentences connecting evidence to significance without adding facts"
}}
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
3) Put boundaries on scope (time/place/actors) and, if evident, name the document set or archive.
4) Identify assumptions or terms that need definition.

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

GRAND_NARRATIVE_FALLBACK = "How did the Relief Department shape worker welfare and labor control, as reflected in the archival record?"


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

NARRATIVE_SYNTHESIS_PROMPT = """You are a historian writing a concise synthesis narrative.

CLOSED-WORLD RULES:
- Use ONLY the evidence provided below.
- Do NOT introduce outside context or facts.
- Historians prefer false negatives to false positives; be cautious.

CRONON-STYLE GUIDELINES:
- Make the narrative specific in time/place/actors when the evidence supports it.
- Emphasize "why then" and "why there" when possible.
- Highlight patterns, contradictions, and gaps without overclaiming.

THEMES (summary with inductive logic):
{themes}

GROUP COMPARISONS:
{group_comparisons}

TEMPORAL QUESTIONS:
{temporal_questions}

GAPS:
{gaps}

CORPUS MAP:
{corpus_map}

TASK:
Write a synthesis narrative (3-5 paragraphs) that:
1) Summarizes the most defensible patterns.
2) Names the main tensions or contradictions.
3) Identifies what is missing or uncertain.
4) Sets up the strongest research agenda implied by the archive.

Return ONLY JSON:
{{
  "narrative": "multi-paragraph narrative",
  "key_claims": ["short claim 1", "short claim 2"],
  "limits": "1-3 sentences on uncertainty or missing evidence"
}}
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
        self._llm_cache = _LLMCache(
            cache_dir=Path(APP_CONFIG.tier0.llm_cache_dir),
            enabled=APP_CONFIG.tier0.llm_cache_enabled,
        )
        self._checkpoint = CheckpointManager(Path(APP_CONFIG.tier0.synthesis_checkpoint_dir))
        self._embedding_cache = _EmbeddingCache(Path(APP_CONFIG.tier0.synthesis_embed_cache))
        self._embedder: Optional[EmbeddingService] = None
        self._embedder_failed = False

    def bucket_questions(self, questions: List[Question], themes: List[ThemeDefinition]) -> Dict[str, List[Question]]:
        if APP_CONFIG.tier0.synthesis_semantic_assignment:
            semantic = self._bucket_questions_semantic(questions, themes)
            if semantic:
                return semantic

        buckets: Dict[str, List[Question]] = {t.key: [] for t in themes}
        buckets["other"] = []

        for question in questions:
            bucket_key = self._assign_bucket(question, themes)
            buckets.setdefault(bucket_key, []).append(question)

        return buckets

    def build_agenda(self, notebook: ResearchNotebook, questions: List[Question]) -> Dict[str, Any]:
        if not APP_CONFIG.tier0.synthesis_enabled:
            return {}

        checksum = self._synthesis_checksum(notebook, questions)
        cached = self._checkpoint.load_latest("synthesis")
        if cached and cached.get("checksum") == checksum:
            payload = cached.get("payload")
            if isinstance(payload, dict):
                if APP_CONFIG.tier0.recursive_enabled and "recursive_synthesis" not in payload:
                    recursive = RecursiveSynthesizer()
                    payload["recursive_synthesis"] = recursive.build(
                        notebook,
                        questions,
                        payload.get("themes", []),
                        grand_narrative=payload.get("grand_narrative"),
                        group_comparisons=payload.get("group_difference_questions"),
                        contradiction_questions=payload.get("contradiction_questions"),
                    )
                    self._checkpoint.save("synthesis", payload, checksum)
                return payload

        themes = self._get_themes(notebook, questions)
        buckets = self.bucket_questions(questions, themes)
        themes, buckets = self._ensure_theme_diversity(themes, buckets, questions)
        themes_out: List[Dict[str, Any]] = []
        contradiction_questions = self._contradiction_questions(notebook)
        temporal_questions = self._temporal_questions(notebook)

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
            sub_candidates = sorted_qs[1:8]
            sub_deduped = self._dedupe_questions_semantic(sub_candidates)
            sub_questions = [self._attach_supporting_documents(q.to_dict()) for q in sub_deduped[:6]]

            clusters = self._cluster_questions_semantic(sorted_qs)
            clustered_questions = self._format_question_clusters(sorted_qs, clusters)

            theme_patterns = self._patterns_for_theme(notebook, theme)
            theme_contradictions = self._contradictions_for_theme(notebook, theme)
            inductive_logic = self._inductive_logic_for_theme(
                theme,
                theme_patterns,
                theme_contradictions,
            )

            themes_out.append({
                "theme": theme.name,
                "theme_key": theme.key,
                "theme_description": theme.description,
                "theme_keywords": theme.keywords,
                "theme_scope": theme.scope,
                "overview_question": self._attach_supporting_documents(broad.to_dict()),
                "sub_questions": sub_questions,
                "question_clusters": clustered_questions,
                "inductive_logic": inductive_logic,
                "evidence_base": {
                    "pattern_count": len(theme_patterns),
                    "question_count": len(theme_questions),
                    "contradiction_count": len(theme_contradictions),
                    "confidence_distribution": self._confidence_distribution(theme_patterns),
                },
            })

        group_difference_questions = self._group_difference_questions(notebook)
        group_difference_questions = [
            self._attach_supporting_documents(q) for q in group_difference_questions
        ]
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
        narrative = self._narrative_summary(
            notebook,
            themes_out,
            group_difference_questions,
            temporal_questions,
            gaps,
        )
        hierarchy = self._build_hierarchy(
            grand_question,
            themes_out,
            contradiction_questions,
            group_difference_questions,
        )

        agenda = {
            "grand_narrative": grand_question,
            "narrative": narrative,
            "theme_definitions": [
                {
                    "key": t.key,
                    "name": t.name,
                    "description": t.description,
                    "keywords": t.keywords,
                    "scope": t.scope,
                }
                for t in themes
            ],
            "themes": themes_out,
            "group_difference_questions": group_difference_questions,
            "contradiction_questions": contradiction_questions,
            "temporal_questions": temporal_questions,
            "hierarchy": hierarchy,
            "gaps": gaps,
        }

        self._checkpoint.save("synthesis", agenda, checksum)
        if APP_CONFIG.tier0.recursive_enabled:
            recursive = RecursiveSynthesizer()
            agenda["recursive_synthesis"] = recursive.build(
                notebook,
                questions,
                themes_out,
                grand_narrative=grand_question,
                group_comparisons=group_difference_questions,
                contradiction_questions=contradiction_questions,
            )
        return agenda

    def _ensure_theme_diversity(
        self,
        themes: List[ThemeDefinition],
        buckets: Dict[str, List[Question]],
        questions: List[Question],
    ) -> Tuple[List[ThemeDefinition], Dict[str, List[Question]]]:
        if not questions:
            return themes, buckets

        non_empty = [t for t in themes if buckets.get(t.key)]
        min_themes = APP_CONFIG.tier0.synthesis_min_themes
        if len(non_empty) >= min_themes:
            return themes, buckets

        clustered_themes, clustered_buckets = self._cluster_questions_into_themes(questions, min_themes)
        if clustered_themes:
            return clustered_themes, clustered_buckets
        return themes, buckets

    def _cluster_questions_into_themes(
        self,
        questions: List[Question],
        target: int,
    ) -> Tuple[List[ThemeDefinition], Dict[str, List[Question]]]:
        if not questions:
            return [], {}
        if len(questions) <= target:
            themes = []
            buckets: Dict[str, List[Question]] = {}
            for idx, q in enumerate(questions, 1):
                key = f"cluster_{idx}"
                name = f"Theme {idx}"
                themes.append(ThemeDefinition(key=key, name=name, description="Derived from question cluster.", keywords=[]))
                buckets[key] = [q]
            return themes, buckets

        vectors = self._get_embeddings([self._question_text(q) for q in questions])
        if vectors is None:
            return [], {}

        # seed with highest-scoring questions
        seeds = sorted(range(len(questions)), key=lambda i: questions[i].validation_score or 0, reverse=True)[:target]
        clusters: Dict[int, List[int]] = {seed: [seed] for seed in seeds}
        similarities = _cosine_similarity_matrix(vectors, vectors)

        for idx in range(len(questions)):
            if idx in seeds:
                continue
            best_seed = None
            best_sim = -1.0
            for seed in seeds:
                sim = float(similarities[idx][seed])
                if sim > best_sim:
                    best_sim = sim
                    best_seed = seed
            if best_seed is None:
                continue
            clusters[best_seed].append(idx)

        themes: List[ThemeDefinition] = []
        buckets: Dict[str, List[Question]] = {}
        for cluster_idx, (seed, members) in enumerate(clusters.items(), 1):
            cluster_questions = [questions[i] for i in members]
            keywords = self._extract_keywords(cluster_questions)
            name = " / ".join([k.title() for k in keywords[:2]]) or f"Theme {cluster_idx}"
            key = self._slugify(name or f"theme_{cluster_idx}")
            themes.append(ThemeDefinition(key=key, name=name, description="Derived from clustered questions.", keywords=keywords))
            buckets[key] = cluster_questions

        return themes, buckets

    def _cluster_questions_semantic(self, questions: List[Question]) -> List[List[int]]:
        if len(questions) <= 1 or not APP_CONFIG.tier0.synthesis_semantic_assignment:
            return [[i] for i in range(len(questions))]

        vectors = self._get_embeddings([self._question_text(q) for q in questions])
        if vectors is None:
            return [[i] for i in range(len(questions))]

        threshold = APP_CONFIG.tier0.synthesis_cluster_threshold
        similarities = _cosine_similarity_matrix(vectors, vectors)
        clusters: List[List[int]] = []

        for idx in range(len(questions)):
            placed = False
            for cluster in clusters:
                rep_idx = cluster[0]
                if float(similarities[idx][rep_idx]) >= threshold:
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])

        return clusters

    def _format_question_clusters(
        self,
        questions: List[Question],
        clusters: List[List[int]],
    ) -> List[Dict[str, Any]]:
        clustered = []
        for cluster in clusters:
            cluster_questions = [questions[i] for i in cluster]
            cluster_questions = sorted(cluster_questions, key=lambda q: q.validation_score or 0, reverse=True)
            rep = cluster_questions[0]
            related = [self._attach_supporting_documents(q.to_dict()) for q in cluster_questions[1:]]
            clustered.append({
                "representative_question": self._attach_supporting_documents(rep.to_dict()),
                "related_questions": related,
            })
        return clustered

    def _extract_keywords(self, questions: List[Question]) -> List[str]:
        stop = {
            "the", "and", "of", "to", "in", "for", "did", "does", "do", "how",
            "what", "why", "when", "where", "were", "was", "with", "on", "by",
            "from", "a", "an", "is", "are", "as", "between", "over", "into",
        }
        counts: Dict[str, int] = {}
        for q in questions:
            tokens = re.findall(r"[a-z0-9]+", q.question_text.lower())
            for token in tokens:
                if token in stop or len(token) < 3:
                    continue
                counts[token] = counts.get(token, 0) + 1
        return [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:6]]

    def _attach_supporting_documents(self, question_dict: Dict[str, Any]) -> Dict[str, Any]:
        docs = set(question_dict.get("evidence_sample") or [])
        precheck = question_dict.get("answerability_precheck") or {}
        for doc_id in precheck.get("sample_doc_ids") or []:
            docs.add(doc_id)
        question_dict["supporting_documents"] = list(docs)[:10]
        return question_dict

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
        patterns = sorted(
            notebook.patterns.values(),
            key=lambda p: len(p.evidence_doc_ids),
            reverse=True,
        )
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

    def _top_patterns(self, notebook: ResearchNotebook, limit: int) -> List[Pattern]:
        patterns = sorted(
            notebook.patterns.values(),
            key=lambda p: len(p.evidence_doc_ids),
            reverse=True,
        )
        return patterns[:limit]

    def _get_themes(self, notebook: ResearchNotebook, questions: List[Question]) -> List[ThemeDefinition]:
        if self._theme_cache:
            return self._theme_cache

        if not APP_CONFIG.tier0.synthesis_dynamic or not questions:
            self._theme_cache = THEMES
            return THEMES

        generated = self._generate_dynamic_themes(notebook, questions)
        if generated:
            target = APP_CONFIG.tier0.synthesis_theme_count
            merged = self._merge_overlapping_themes(generated)
            if len(merged) < target:
                extra = self._expand_themes(notebook, questions, merged, target - len(merged))
                merged = self._merge_overlapping_themes(merged + extra)
            if len(merged) < target:
                fallback = self._fallback_themes_from_patterns(notebook, merged, target - len(merged))
                merged = self._merge_overlapping_themes(merged + fallback)

            if merged:
                merged = merged[:target]
                self._theme_cache = merged
                return merged

        self._theme_cache = THEMES
        return THEMES

    def _fallback_themes_from_patterns(
        self,
        notebook: ResearchNotebook,
        existing: List[ThemeDefinition],
        max_new: int,
    ) -> List[ThemeDefinition]:
        if max_new <= 0:
            return []

        existing_keys = {t.key for t in existing}
        pattern_types = [p.pattern_type or "unknown" for p in notebook.patterns.values()]
        if not pattern_types:
            return []

        counts: Dict[str, int] = {}
        for ptype in pattern_types:
            counts[ptype] = counts.get(ptype, 0) + 1

        sorted_types = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        themes: List[ThemeDefinition] = []
        for ptype, count in sorted_types:
            if len(themes) >= max_new:
                break
            name = ptype.replace("_", " ").strip().title()
            key = self._slugify(name)
            if key in existing_keys:
                continue
            keywords = [w for w in re.split(r"\\s+", name.lower()) if w]
            themes.append(ThemeDefinition(
                key=key,
                name=name,
                description=f"Recurring '{ptype}' patterns observed across {count} instances.",
                keywords=keywords,
                scope={},
            ))
            existing_keys.add(key)

        return themes

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
        pattern_sample = self._top_patterns(
            notebook,
            APP_CONFIG.tier0.synthesis_max_pattern_sample,
        )

        questions_text = "\n".join(f"- {q.question_text}" for q in question_sample) or "- (none)"
        patterns_text = "\n".join(f"- {p.pattern_text}" for p in pattern_sample) or "- (none)"

        prompt = THEME_SYNTHESIS_PROMPT.format(
            questions=questions_text,
            patterns=patterns_text,
            theme_count=APP_CONFIG.tier0.synthesis_theme_count,
        )

        response = self._generate_cached(
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
        if not isinstance(data, list) or not data:
            repair_prompt = THEME_REPAIR_PROMPT.format(raw_output=response.content)
            repair = self._generate_cached(
                messages=[
                    {"role": "system", "content": "You only reformat text into strict JSON."},
                    {"role": "user", "content": repair_prompt},
                ],
                profile="verifier",
                temperature=0.0,
                timeout=APP_CONFIG.tier0.llm_timeout,
            )
            if repair.success:
                data = parse_llm_json(repair.content, default=[])
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
            scope = item.get("scope") or {}
            if not isinstance(scope, dict):
                scope = {}
            key = self._slugify(name)
            themes.append(ThemeDefinition(key=key, name=name, description=description, keywords=keywords, scope=scope))

        return themes

    def _expand_themes(
        self,
        notebook: ResearchNotebook,
        questions: List[Question],
        existing: List[ThemeDefinition],
        max_new: int,
    ) -> List[ThemeDefinition]:
        if max_new <= 0:
            return []

        question_sample = sorted(
            questions,
            key=lambda q: q.validation_score or 0,
            reverse=True,
        )[: APP_CONFIG.tier0.synthesis_max_question_sample]
        pattern_sample = self._top_patterns(
            notebook,
            APP_CONFIG.tier0.synthesis_max_pattern_sample,
        )

        questions_text = "\n".join(f"- {q.question_text}" for q in question_sample) or "- (none)"
        patterns_text = "\n".join(f"- {p.pattern_text}" for p in pattern_sample) or "- (none)"
        themes_text = "\n".join(f"- {t.name}: {t.description}" for t in existing) or "- (none)"

        prompt = THEME_EXPANSION_PROMPT.format(
            themes=themes_text,
            questions=questions_text,
            patterns=patterns_text,
            theme_count=max_new,
        )

        response = self._generate_cached(
            messages=[
                {"role": "system", "content": "You expand research themes without overlap."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.2,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if not response.success:
            return []

        data = parse_llm_json(response.content, default=[])
        if not isinstance(data, list) or not data:
            repair_prompt = THEME_REPAIR_PROMPT.format(raw_output=response.content)
            repair = self._generate_cached(
                messages=[
                    {"role": "system", "content": "You only reformat text into strict JSON."},
                    {"role": "user", "content": repair_prompt},
                ],
                profile="verifier",
                temperature=0.0,
                timeout=APP_CONFIG.tier0.llm_timeout,
            )
            if repair.success:
                data = parse_llm_json(repair.content, default=[])
        if not isinstance(data, list):
            return []

        existing_keys = {t.key for t in existing}
        new_themes: List[ThemeDefinition] = []
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
            scope = item.get("scope") or {}
            if not isinstance(scope, dict):
                scope = {}
            key = self._slugify(name)
            if key in existing_keys:
                continue
            new_themes.append(ThemeDefinition(key=key, name=name, description=description, keywords=keywords, scope=scope))
            existing_keys.add(key)
            if len(new_themes) >= max_new:
                break

        return new_themes

    def _merge_overlapping_themes(self, themes: List[ThemeDefinition]) -> List[ThemeDefinition]:
        if len(themes) <= 1:
            return themes

        merge_threshold = getattr(APP_CONFIG.tier0, "synthesis_theme_merge_threshold", None)
        if merge_threshold is None:
            merge_threshold = APP_CONFIG.tier0.synthesis_dedupe_threshold

        theme_texts = [self._theme_text(t) for t in themes]
        vectors = self._get_embeddings(theme_texts)
        index_map = {id(theme): idx for idx, theme in enumerate(themes)}

        merged: List[ThemeDefinition] = []
        if vectors is not None:
            similarities = _cosine_similarity_matrix(vectors, vectors)
            for idx, theme in enumerate(themes):
                placed = False
                for kept in merged:
                    kept_idx = index_map.get(id(kept), 0)
                    if float(similarities[idx][kept_idx]) >= merge_threshold:
                        self._merge_theme_into(kept, theme)
                        placed = True
                        break
                if not placed:
                    merged.append(theme)
            return merged

        # Fallback: keyword overlap
        for theme in themes:
            placed = False
            for kept in merged:
                overlap = set(theme.keywords or []) & set(kept.keywords or [])
                denom = max(1, len(set(theme.keywords or []) | set(kept.keywords or [])))
                if overlap and (len(overlap) / denom) >= 0.6:
                    self._merge_theme_into(kept, theme)
                    placed = True
                    break
            if not placed:
                merged.append(theme)

        return merged

    def _merge_theme_into(self, base: ThemeDefinition, other: ThemeDefinition) -> None:
        base.keywords = sorted(set(base.keywords + other.keywords))
        if len(other.description) > len(base.description):
            base.description = other.description
            base.name = base.name or other.name
        base.scope = self._merge_scope(base.scope, other.scope)

    def _merge_scope(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(a or {})
        for key, value in (b or {}).items():
            if key not in merged or not merged.get(key):
                merged[key] = value
            elif isinstance(value, list):
                existing = merged.get(key)
                if not isinstance(existing, list):
                    merged[key] = value
                else:
                    merged[key] = sorted(set(existing + value))
        return merged

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
        ] + [
            g.label for g in notebook.group_indicators.values()
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

    def _synthesis_checksum(self, notebook: ResearchNotebook, questions: List[Question]) -> str:
        payload = {
            "questions": [
                {
                    "text": q.question_text,
                    "score": q.validation_score,
                    "pattern": q.pattern_source,
                    "contradiction": q.contradiction_source,
                }
                for q in questions
            ],
            "patterns": [
                {
                    "text": p.pattern_text,
                    "count": len(p.evidence_doc_ids),
                    "confidence": p.confidence,
                }
                for p in notebook.patterns.values()
            ],
            "contradictions": [
                {
                    "a": c.claim_a,
                    "b": c.claim_b,
                    "type": c.contradiction_type,
                }
                for c in notebook.contradictions
            ],
            "group_indicators": [
                {
                    "type": g.group_type,
                    "label": g.label,
                    "count": len(g.evidence_doc_ids),
                    "confidence": g.confidence,
                }
                for g in notebook.group_indicators.values()
            ],
            "corpus_map": {
                "time_coverage": notebook.corpus_map.get("time_coverage"),
                "density_by_decade": notebook.corpus_map.get("density_by_decade"),
                "temporal_gaps": notebook.corpus_map.get("temporal_gaps"),
                "peak_period": notebook.corpus_map.get("peak_period"),
            },
            "config": {
                "synthesis_dynamic": APP_CONFIG.tier0.synthesis_dynamic,
                "synthesis_semantic_assignment": APP_CONFIG.tier0.synthesis_semantic_assignment,
                "synthesis_theme_count": APP_CONFIG.tier0.synthesis_theme_count,
                "synthesis_min_themes": APP_CONFIG.tier0.synthesis_min_themes,
                "synthesis_max_question_sample": APP_CONFIG.tier0.synthesis_max_question_sample,
                "synthesis_max_pattern_sample": APP_CONFIG.tier0.synthesis_max_pattern_sample,
                "synthesis_assign_min_sim": APP_CONFIG.tier0.synthesis_assign_min_sim,
                "synthesis_dedupe_threshold": APP_CONFIG.tier0.synthesis_dedupe_threshold,
                "synthesis_cluster_threshold": APP_CONFIG.tier0.synthesis_cluster_threshold,
                "synthesis_theme_merge_threshold": APP_CONFIG.tier0.synthesis_theme_merge_threshold,
                "synthesis_narrative_enabled": APP_CONFIG.tier0.synthesis_narrative_enabled,
                "synthesis_narrative_max_themes": APP_CONFIG.tier0.synthesis_narrative_max_themes,
                "question_target_count": APP_CONFIG.tier0.question_target_count,
                "question_per_type": APP_CONFIG.tier0.question_per_type,
                "question_min_score": APP_CONFIG.tier0.question_min_score,
            },
            "models": {
                "quality": APP_CONFIG.llm_profiles.get("quality", {}).get("model"),
                "verifier": APP_CONFIG.llm_profiles.get("verifier", {}).get("model"),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _contradictions_for_theme(
        self,
        notebook: ResearchNotebook,
        theme: ThemeDefinition,
    ) -> List[Any]:
        if not theme.keywords:
            return notebook.contradictions
        matches = []
        for contra in notebook.contradictions:
            text = " ".join([
                str(contra.claim_a or ""),
                str(contra.claim_b or ""),
                str(contra.context or ""),
            ]).lower()
            if any(kw in text for kw in theme.keywords):
                matches.append(contra)
        return matches or notebook.contradictions

    def _inductive_logic_for_theme(
        self,
        theme: ThemeDefinition,
        patterns: List[Pattern],
        contradictions: List[Any],
    ) -> Dict[str, Any]:
        pattern_summaries = [
            f"{p.pattern_text} (n={len(p.evidence_doc_ids)})"
            for p in patterns[:5]
        ]
        contradiction_summaries = [
            f"{c.claim_a} vs {c.claim_b}"
            for c in contradictions[:3]
        ]

        prompt = INDUCTIVE_LOGIC_PROMPT.format(
            theme_name=theme.name,
            patterns="\n".join(f"- {p}" for p in pattern_summaries) or "- (none)",
            contradictions="\n".join(f"- {c}" for c in contradiction_summaries) or "- (none)",
            contradiction_count=len(contradictions),
        )

        response = self._generate_cached(
            messages=[
                {"role": "system", "content": "You synthesize cautious inductive logic."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.1,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if response.success:
            data = parse_llm_json(response.content, default={})
            if isinstance(data, dict):
                data.setdefault("patterns_observed", pattern_summaries[:3])
                data.setdefault("contradictions_observed", len(contradictions))
                return data

        return {
            "patterns_observed": pattern_summaries[:3],
            "contradictions_observed": len(contradictions),
            "inference": f"Evidence suggests {theme.description.lower()} as a recurring dynamic, but details remain to be verified.",
            "historical_argument": "Additional evidence is needed to establish mechanisms and scope without overclaiming.",
        }

    def _contradiction_questions(self, notebook: ResearchNotebook) -> List[Dict[str, Any]]:
        questions: List[Dict[str, Any]] = []
        for contra in notebook.contradictions:
            claim_a = contra.claim_a.strip()
            claim_b = contra.claim_b.strip()
            if not claim_a or not claim_b:
                continue
            if contra.contradiction_type == "name_variant_or_ocr":
                question = "How do name variants or OCR errors affect record linkage for this individual?"
            elif contra.contradiction_type == "certificate_number_conflict":
                question = "What explains conflicting certificate numbers in the records for this case?"
            elif contra.contradiction_type == "date_conflict":
                question = "Why do the documents disagree on the date of the same event?"
            else:
                question = "What explains the discrepancy between two records describing the same case?"

            questions.append({
                "question": question,
                "contradiction_type": contra.contradiction_type,
                "claims": [claim_a, claim_b],
                "sources": [contra.source_a, contra.source_b],
                "context": contra.context,
            })

        max_micro = 30
        return questions[:max_micro]

    def _temporal_questions(self, notebook: ResearchNotebook) -> List[Dict[str, Any]]:
        corpus_map = notebook.corpus_map
        time_coverage = corpus_map.get("time_coverage", {})
        density = corpus_map.get("density_by_decade", {})
        if not time_coverage or not density:
            return []

        questions: List[Dict[str, Any]] = []
        start = time_coverage.get("start")
        end = time_coverage.get("end")
        if start and end:
            questions.append({
                "question": f"How did Relief Department practices change between {start} and {end}?",
                "evidence_basis": "time_coverage",
            })

        peak = corpus_map.get("peak_period")
        if peak:
            questions.append({
                "question": f"Why does documentation peak in the {peak}s, and what changed in that decade?",
                "evidence_basis": "density_by_decade",
            })

        gaps = corpus_map.get("temporal_gaps", [])
        if gaps:
            questions.append({
                "question": f"What explains sparse or missing records in the {', '.join(gaps[:3])}s?",
                "evidence_basis": "temporal_gaps",
            })

        return questions

    def _build_hierarchy(
        self,
        grand_question: Dict[str, Any],
        themes_out: List[Dict[str, Any]],
        contradiction_questions: List[Dict[str, Any]],
        group_difference_questions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        level_2 = [
            {
                "theme": theme["theme"],
                "question": theme["overview_question"],
            }
            for theme in themes_out
        ]
        level_3 = {
            theme["theme_key"]: theme.get("question_clusters", theme.get("sub_questions", []))
            for theme in themes_out
        }
        level_4 = contradiction_questions[:25]

        return {
            "level_1_grand": grand_question,
            "level_2_thematic": level_2,
            "level_3_specific": level_3,
            "level_3_group_comparisons": group_difference_questions,
            "level_4_micro": level_4,
        }

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
        pattern_sample = self._top_patterns(
            notebook,
            APP_CONFIG.tier0.synthesis_max_pattern_sample,
        )

        questions_text = "\n".join(f"- {q.question_text}" for q in question_sample) or "- (none)"
        patterns_text = "\n".join(f"- {p.pattern_text}" for p in pattern_sample) or "- (none)"
        themes_text = "\n".join(f"- {name}" for name in theme_names) or "- (none)"

        prompt = GRAND_NARRATIVE_PROMPT.format(
            questions=questions_text,
            patterns=patterns_text,
            themes=themes_text,
        )

        response = self._generate_cached(
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
        repair_response = self._generate_cached(
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

        if not isinstance(repaired, dict) or not repaired.get("question"):
            question = GRAND_NARRATIVE_FALLBACK
            return {
                "question": question,
                "themes": theme_names,
            }

        question_text = str(repaired.get("question") or "").strip()
        scope = repaired.get("scope") or {}
        if self._grand_too_specific(question_text, scope):
            question_text = (
                "How did the Relief Department shape worker welfare and labor control, "
                f"as reflected across {', '.join(theme_names)} in the archival record?"
            )
            repaired["question"] = question_text
            repaired["purpose_statement"] = (
                "I am studying the Relief Department archive because I want to know how its policies "
                "structured worker welfare and control in order to help my readers understand institutional "
                "welfare regimes in early twentieth-century railroads."
            )

        repaired["themes"] = theme_names
        return repaired

    def _grand_too_specific(self, question: str, scope: Dict[str, Any]) -> bool:
        if not question:
            return True
        if re.search(r"\b[a-f0-9]{8,}\b", question.lower()):
            return True
        name_hits = re.findall(r\"\\b[A-Z][a-z]+ [A-Z][a-z]+\\b\", question)
        if name_hits and len(question.split()) < 18:
            return True
        actors = scope.get(\"actors\") if isinstance(scope, dict) else None
        if isinstance(actors, list) and 0 < len(actors) <= 2:
            return True
        return False

    def _narrative_summary(
        self,
        notebook: ResearchNotebook,
        themes_out: List[Dict[str, Any]],
        group_difference_questions: List[Dict[str, Any]],
        temporal_questions: List[Dict[str, Any]],
        gaps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not APP_CONFIG.tier0.synthesis_narrative_enabled:
            return {}
        if not themes_out:
            return {}

        max_themes = APP_CONFIG.tier0.synthesis_narrative_max_themes
        theme_lines = []
        for theme in themes_out[:max_themes]:
            inductive = theme.get("inductive_logic") or {}
            inference = inductive.get("inference") or ""
            argument = inductive.get("historical_argument") or ""
            evidence = theme.get("evidence_base") or {}
            theme_lines.append(
                f"- {theme.get('theme')} | {theme.get('theme_description')} | "
                f"patterns={evidence.get('pattern_count')}, "
                f"contradictions={evidence.get('contradiction_count')} | "
                f"inference={inference} | argument={argument}"
            )

        themes_text = "\n".join(theme_lines) or "- (none)"
        group_text = "\n".join(
            f"- {q.get('question')} (evidence: {q.get('evidence_basis', '')})"
            for q in group_difference_questions[:5]
        ) or "- (none)"
        temporal_text = "\n".join(
            f"- {q.get('question')} (evidence: {q.get('evidence_basis', '')})"
            for q in temporal_questions[:5]
        ) or "- (none)"
        gaps_text = "\n".join(f"- {g.get('gap')}: {', '.join(g.get('missing_evidence', []))}" for g in gaps[:6]) or "- (none)"

        corpus_map = {
            "time_coverage": notebook.corpus_map.get("time_coverage"),
            "density_by_decade": notebook.corpus_map.get("density_by_decade"),
            "temporal_gaps": notebook.corpus_map.get("temporal_gaps"),
            "peak_period": notebook.corpus_map.get("peak_period"),
        }

        prompt = NARRATIVE_SYNTHESIS_PROMPT.format(
            themes=themes_text,
            group_comparisons=group_text,
            temporal_questions=temporal_text,
            gaps=gaps_text,
            corpus_map=json.dumps(corpus_map, ensure_ascii=True),
        )

        response = self._generate_cached(
            messages=[
                {"role": "system", "content": "You write careful historical syntheses."},
                {"role": "user", "content": prompt},
            ],
            profile="quality",
            temperature=0.2,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        if not response.success:
            return {}

        data = parse_llm_json(response.content, default={})
        if not isinstance(data, dict):
            return {}
        return data

    def _group_difference_questions(
        self,
        notebook: ResearchNotebook,
    ) -> List[Dict[str, Any]]:
        indicators = list(notebook.group_indicators.values())
        if not indicators:
            return []

        min_docs = APP_CONFIG.tier0.group_indicator_min_docs
        grouped: Dict[str, List[Any]] = {}
        for indicator in indicators:
            if indicator.confidence not in {"medium", "high"}:
                continue
            if len(indicator.evidence_doc_ids) < min_docs:
                continue
            grouped.setdefault(indicator.group_type, []).append(indicator)

        questions: List[Dict[str, Any]] = []
        for group_type, items in grouped.items():
            items = sorted(items, key=lambda i: len(i.evidence_doc_ids), reverse=True)
            if len(items) < 2:
                continue
            first, second = items[0], items[1]
            label_a = first.label
            label_b = second.label
            question = self._format_group_difference_question(group_type, label_a, label_b)
            questions.append({
                "question": question,
                "groups": [label_a, label_b],
                "group_type": group_type,
                "evidence_basis": f"{label_a} (n={len(first.evidence_doc_ids)}), {label_b} (n={len(second.evidence_doc_ids)})",
                "time_window": "",
            })

        return questions

    def _format_group_difference_question(self, group_type: str, a: str, b: str) -> str:
        if group_type == "occupation":
            return f"How did Relief Department outcomes differ between {a} and {b} occupations?"
        if group_type == "gender":
            return f"Do the records show differences in Relief outcomes between {a} and {b}?"
        if group_type == "race":
            return f"Do the records show differences in Relief outcomes between {a} and {b}?"
        if group_type == "ethnicity":
            return f"Do the records show differences in Relief outcomes between {a} and {b} ethnic groups?"
        if group_type == "national_origin":
            return f"Do the records show differences in Relief outcomes between workers of {a} and {b} origin?"
        if group_type == "class":
            return f"How did Relief outcomes differ between {a} and {b} class categories?"
        return f"How did Relief outcomes differ between {a} and {b} groups?"

    def _generate_cached(
        self,
        messages: List[Dict[str, str]],
        profile: str,
        temperature: float,
        timeout: int,
    ) -> LLMResponse:
        if not APP_CONFIG.tier0.llm_cache_enabled:
            return self.llm.generate(
                messages=messages,
                profile=profile,
                temperature=temperature,
                timeout=timeout,
            )

        profile_cfg = APP_CONFIG.llm_profiles.get(profile, {})
        cache_payload = {
            "profile": profile,
            "provider": profile_cfg.get("provider"),
            "model": profile_cfg.get("model"),
            "temperature": temperature,
            "messages": messages,
        }

        cached = self._llm_cache.get(cache_payload)
        if cached is not None:
            return _CachedResponse(content=cached)

        response = self.llm.generate(
            messages=messages,
            profile=profile,
            temperature=temperature,
            timeout=timeout,
        )

        if response.success:
            self._llm_cache.set(cache_payload, response.content)

        return response

    def _bucket_questions_semantic(
        self,
        questions: List[Question],
        themes: List[ThemeDefinition],
    ) -> Optional[Dict[str, List[Question]]]:
        if not questions or not themes:
            return None

        embeddings = self._get_embeddings_for_assignment(questions, themes)
        if embeddings is None:
            return None

        question_vectors, theme_vectors = embeddings
        if question_vectors is None or theme_vectors is None:
            return None

        similarities = _cosine_similarity_matrix(question_vectors, theme_vectors)
        buckets: Dict[str, List[Question]] = {t.key: [] for t in themes}
        buckets["other"] = []

        min_sim = APP_CONFIG.tier0.synthesis_assign_min_sim
        for q_idx, question in enumerate(questions):
            row = similarities[q_idx]
            best_idx = int(row.argmax())
            best_sim = float(row[best_idx])
            if best_sim < min_sim:
                buckets["other"].append(question)
                continue
            buckets[themes[best_idx].key].append(question)

        return buckets

    def _dedupe_questions_semantic(self, questions: List[Question]) -> List[Question]:
        if len(questions) <= 1 or not APP_CONFIG.tier0.synthesis_semantic_assignment:
            return questions

        vectors = self._get_embeddings([self._question_text(q) for q in questions])
        if vectors is None:
            return questions

        threshold = APP_CONFIG.tier0.synthesis_dedupe_threshold
        similarities = _cosine_similarity_matrix(vectors, vectors)

        keep: List[Question] = []
        for idx, question in enumerate(questions):
            if not keep:
                keep.append(question)
                continue
            is_duplicate = False
            for kept_idx, kept_question in enumerate(keep):
                kept_orig_idx = questions.index(kept_question)
                if float(similarities[idx][kept_orig_idx]) >= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(question)
        return keep

    def _get_embeddings_for_assignment(
        self,
        questions: List[Question],
        themes: List[ThemeDefinition],
    ) -> Optional[Tuple[Any, Any]]:
        question_texts = [self._question_text(q) for q in questions]
        theme_texts = [self._theme_text(t) for t in themes]

        question_vectors = self._get_embeddings(question_texts)
        if question_vectors is None:
            return None

        theme_vectors = self._get_embeddings(theme_texts)
        if theme_vectors is None:
            return None

        return question_vectors, theme_vectors

    def _question_text(self, question: Question) -> str:
        return " ".join([
            question.question_text,
            question.why_interesting or "",
            question.pattern_source or "",
            question.contradiction_source or "",
        ]).strip()

    def _theme_text(self, theme: ThemeDefinition) -> str:
        return " ".join([
            theme.name,
            theme.description,
            "keywords: " + ", ".join(theme.keywords or []),
        ]).strip()

    def _get_embeddings(self, texts: List[str]):
        if not texts:
            return None

        if self._embedder_failed:
            return None

        if self._embedder is None:
            try:
                self._embedder = EmbeddingService(
                    provider=APP_CONFIG.tier0.synthesis_embed_provider,
                    model=APP_CONFIG.tier0.synthesis_embed_model,
                    timeout=APP_CONFIG.tier0.synthesis_embed_timeout,
                )
            except Exception as exc:
                debug_print(f"Embedding service unavailable: {exc}")
                self._embedder_failed = True
                return None

        cached = self._embedding_cache.get_batch(texts, self._embedder.model_name)
        missing = [text for text, vec in cached if vec is None]

        if missing:
            try:
                vectors = self._embedder.embed_documents(missing)
            except Exception as exc:
                debug_print(f"Embedding generation failed: {exc}")
                self._embedder_failed = True
                return None

            for text, vec in zip(missing, vectors):
                self._embedding_cache.set(text, self._embedder.model_name, vec.tolist())

            self._embedding_cache.flush()

            cached = self._embedding_cache.get_batch(texts, self._embedder.model_name)

        vectors = [vec for _text, vec in cached if vec is not None]
        if len(vectors) != len(texts):
            return None

        array = _to_numpy(vectors)
        if array is None:
            return None
        return array


@dataclass
class _CachedResponse:
    content: str
    success: bool = True


class _LLMCache:
    def __init__(self, cache_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.cache_dir = cache_dir
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, payload: Dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, payload: Dict[str, Any]) -> Optional[str]:
        if not self.enabled:
            return None
        key = self._key(payload)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            return None
        return data.get("response")

    def set(self, payload: Dict[str, Any], response: str) -> None:
        if not self.enabled:
            return
        key = self._key(payload)
        path = self.cache_dir / f"{key}.json"
        data = {
            "response": response,
            "profile": payload.get("profile"),
            "provider": payload.get("provider"),
            "model": payload.get("model"),
        }
        try:
            path.write_text(json.dumps(data, ensure_ascii=True))
        except Exception:
            return


class _EmbeddingCache:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache: Dict[str, List[float]] = {}
        if cache_path.exists():
            try:
                self.cache = pickle.loads(cache_path.read_bytes())
            except Exception:
                self.cache = {}

    def _key(self, text: str, model: str) -> str:
        encoded = f"{model}:{text}".encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_batch(self, texts: List[str], model: str) -> List[Tuple[str, Optional[List[float]]]]:
        results: List[Tuple[str, Optional[List[float]]]] = []
        for text in texts:
            key = self._key(text, model)
            results.append((text, self.cache.get(key)))
        return results

    def set(self, text: str, model: str, vector: List[float]) -> None:
        key = self._key(text, model)
        self.cache[key] = vector

    def flush(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(pickle.dumps(self.cache))
        except Exception:
            return


def _to_numpy(vectors: List[List[float]]):
    try:
        import numpy as np
    except Exception:
        return None
    return np.array(vectors, dtype=float)


def _cosine_similarity_matrix(a, b):
    try:
        import numpy as np
    except Exception:
        # Fallback: no numpy, return zeros
        return [[0.0 for _ in range(len(b))] for _ in range(len(a))]

    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T
