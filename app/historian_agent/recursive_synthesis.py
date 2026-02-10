# app/historian_agent/recursive_synthesis.py
# Created: 2026-02-09
# Purpose: Recursive historian synthesis (drill-down to trivial answers, then re-synthesize)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import uuid
import re

from rag_base import DocumentStore, MongoDBConnection, debug_print
from llm_abstraction import LLMClient
from config import APP_CONFIG
from tier0_utils import parse_llm_json
from historian_agent.question_models import Question
from historian_agent.research_notebook import ResearchNotebook, Pattern, Contradiction


SUBQUESTION_PROMPT = """You are a historian refining a research question into smaller, more specific sub-questions.

CLOSED-WORLD RULES:
- Use ONLY the evidence hints below.
- Do NOT add outside knowledge.
- Historians prefer false negatives to false positives.
- Favor interpretive questions (why/how) over descriptive ones.
- If evidence does not name groups, do NOT invent group comparisons.
- Make the question narrower in time/place/actors if possible.
- Every sub-question MUST cite one or more doc_ids from the allowed list.
- If you cannot anchor a sub-question to a doc_id, do NOT include it.

PARENT QUESTION:
{question}

ALLOWED DOC IDS:
{allowed_doc_ids}

EVIDENCE EXCERPTS (from the archive):
{evidence_hints}

TASK:
Generate {count} sub-questions that are more specific and more directly answerable.
Each sub-question should narrow scope (time/place/actors) or target a mechanism.

Return ONLY JSON array of objects:
[
  {{
    "question": "sub-question text",
    "doc_ids": ["...","..."]
  }}
]
"""


LEAF_ANSWER_PROMPT = """You are a historian answering a specific question using ONLY the provided documents.

CLOSED-WORLD RULES:
- Use ONLY the sources below.
- If the answer is not supported, say \"insufficient evidence\".
- Prefer false negatives to false positives.
- Write in the past tense.
- Avoid vague generalizations; be specific.
- Avoid presentism; treat the past on its own terms.
- Topic sentence must be a declarative claim (not a question).
- If evidence is a form/questionnaire without responses, interpret it as a requirement or institutional practice, not an outcome.
- Do not infer outcomes not documented; prefer "the records show that the institution asked for X" over "workers did X."
- Only make strong causal or comparative claims when at least two independent sources corroborate them.
- For group-comparison questions, only answer if the groups are explicitly named in sources.
- Define any key term using the sources if ambiguity would change interpretation.

QUESTION:
{question}

SOURCE INDEX (labels -> doc ids):
{source_index}

SOURCES:
{sources}

Return ONLY JSON:
{{
  "topic_sentence": "...",
  "evidence": [
    {{"source_label": "Source 1", "doc_id": "...", "quote": "...", "reason": "..."}}
  ],
  "analysis": "1-2 sentences explaining how the evidence supports the topic sentence (or why evidence is insufficient).",
  "uncertainty": "...",
  "missing_evidence": "..."
}}
"""


THEME_SYNTHESIS_PROMPT = """You are a historian synthesizing a theme from leaf answers.

CLOSED-WORLD RULES:
- Use ONLY the leaf answers below.
- Do NOT add outside facts.
- Each paragraph must start with a topic sentence.
- Support each topic sentence with evidence citations.
- Avoid repeating the same evidence in multiple paragraphs.
- Prefer interpretive statements over description when evidence permits.
- Each paragraph should follow claim -> evidence -> analysis.
- Explicitly note contradictions and archival silences when they affect the theme.
- Order paragraphs chronologically when evidence provides dates.
- If evidence is form-like, interpret it as institutional practice rather than lived outcome.

THEME:
{theme}

LEAF ANSWERS:
{leaf_answers}

Return ONLY JSON:
{{
  "theme_summary": "...",
  "paragraphs": [
    {{
      "subquestion": "...",
      "topic_sentence": "...",
      "evidence_sentences": ["... [doc_id]", "... [doc_id]"],
      "analysis_sentence": "..."
    }}
  ],
  "contradictions": ["..."],
  "gaps": ["..."]
}}
"""


ESSAY_PROMPT = """You are a historian writing a long-form essay from theme summaries.

CLOSED-WORLD RULES:
- Use ONLY the theme summaries below.
- Do NOT add outside facts.
- Prefer false negatives to false positives.
- Write in the past tense, be specific, avoid presentism.

THEME SUMMARIES:
{theme_summaries}

TASK:
Write a structured essay that:
1) Defines scope, source limits, and evidence base (selection/interpretation).
2) Opens with a thesis and a purpose statement in the Cronon form:
   \"I am studying ____ because I want to know ____ in order to help my readers understand ____.\"\n
3) Provides a brief roadmap (2-3 reasons/themes).
4) Presents theme-by-theme synthesis with topic sentence -> evidence -> analysis.
5) Includes at least one paragraph that addresses an alternative explanation or counterargument (using evidence).
6) Highlights contradictions and archival silences when they affect interpretation.
7) Identifies gaps and next questions.
8) Minimum length: {min_words} words.
9) For each theme, include 2-4 paragraphs built from the provided theme paragraphs.
10) Each paragraph must include at least one evidence citation in [doc_id] format.
11) End with a short conclusion that ties themes together without introducing new facts.
12) The thesis must be an argument (not a summary) and should preview 2-3 reasons.
13) Topic sentences must be arguable claims supported by evidence.
14) Where evidence is thin, acknowledge limits rather than speculate.

Return ONLY JSON:
{{
  "essay": "...",
  "sections": ["...", "..."]
}}
"""


@dataclass
class EvidencePack:
    doc_ids: List[str] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)


@dataclass
class QuestionNode:
    node_id: str
    question_text: str
    level: int
    parent_id: Optional[str]
    evidence: EvidencePack
    children: List["QuestionNode"] = field(default_factory=list)
    answer: Optional[Dict[str, Any]] = None


def _json_safe(value: Any) -> Any:
    try:
        from bson import ObjectId
    except Exception:  # pragma: no cover - bson might be unavailable in some contexts
        ObjectId = None

    if ObjectId is not None and isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _extract_years(text: str) -> List[int]:
    if not text:
        return []
    years = set()
    for match in re.findall(r"\b(?:18|19|20)\d{2}\b", text):
        try:
            years.add(int(match))
        except ValueError:
            continue
    return sorted(years)


class RecursiveSynthesizer:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.leaf_profile = APP_CONFIG.tier0.recursive_leaf_profile
        self.writer_profile = APP_CONFIG.tier0.recursive_writer_profile
        self.doc_store = DocumentStore()
        self._mongo = MongoDBConnection()
        self._leaf_coll = self._mongo.get_collection(APP_CONFIG.tier0.leaf_answers_collection)
        self._ensure_indexes()

    def _normalize_question(self, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _ensure_indexes(self) -> None:
        try:
            self._leaf_coll.create_index([("run_id", 1), ("question_id", 1)], name="run_question")
            self._leaf_coll.create_index([("created_at", -1)], name="created_at")
        except Exception as exc:
            debug_print(f"Leaf answer index error: {exc}")

    def build(
        self,
        notebook: ResearchNotebook,
        questions: List[Question],
        themes_out: List[Dict[str, Any]],
        grand_narrative: Optional[Dict[str, Any]] = None,
        group_comparisons: Optional[List[Dict[str, Any]]] = None,
        contradiction_questions: Optional[List[Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not APP_CONFIG.tier0.recursive_enabled:
            return {}

        run_id = run_id or datetime.utcnow().strftime("recursive_%Y%m%d_%H%M%S")
        question_map = {self._normalize_question(q.question_text): q for q in questions}

        theme_trees: List[Dict[str, Any]] = []
        leaf_answers: List[Dict[str, Any]] = []
        seen_leaf_ids = set()

        for theme in themes_out:
            theme_name = theme.get("theme") or "Theme"
            seed_questions = []

            overview = theme.get("overview_question") or {}
            if overview.get("question"):
                seed_questions.append(overview.get("question"))

            for sub in theme.get("sub_questions") or []:
                if isinstance(sub, dict) and sub.get("question"):
                    seed_questions.append(sub.get("question"))

            # Deduplicate seeds
            seed_questions = list(dict.fromkeys(seed_questions))
            if not seed_questions:
                continue

            tree_nodes = []
            for seed in seed_questions:
                node = self._build_node(
                    seed,
                    notebook,
                    question_map,
                    level=0,
                    parent_id=None,
                    run_id=run_id,
                )
                tree_nodes.append(node)

            theme_leaf_answers = self._collect_leaf_answers(tree_nodes)
            for ans in theme_leaf_answers:
                qid = ans.get("question_id")
                if qid and qid not in seen_leaf_ids:
                    leaf_answers.append(ans)
                    seen_leaf_ids.add(qid)

            theme_trees.append({
                "theme": theme_name,
                "nodes": [self._node_to_dict(n) for n in tree_nodes],
                "leaf_answers": theme_leaf_answers,
            })

        if group_comparisons:
            group_seeds = [
                gc.get("question") for gc in group_comparisons
                if isinstance(gc, dict) and gc.get("question")
            ]
            group_seeds = list(dict.fromkeys(group_seeds))
            if group_seeds:
                group_nodes = [
                    self._build_node(
                        seed,
                        notebook,
                        question_map,
                        level=0,
                        parent_id=None,
                        run_id=run_id,
                    )
                    for seed in group_seeds
                ]
                group_leaf_answers = self._collect_leaf_answers(group_nodes)
                for ans in group_leaf_answers:
                    qid = ans.get("question_id")
                    if qid and qid not in seen_leaf_ids:
                        leaf_answers.append(ans)
                        seen_leaf_ids.add(qid)
                theme_trees.append({
                    "theme": "Group Comparisons",
                    "nodes": [self._node_to_dict(n) for n in group_nodes],
                    "leaf_answers": group_leaf_answers,
                })

        if contradiction_questions:
            contra_seeds = [
                cq.get("question") for cq in contradiction_questions
                if isinstance(cq, dict) and cq.get("question")
            ]
            contra_seeds = list(dict.fromkeys(contra_seeds))
            if contra_seeds:
                contra_nodes = [
                    self._build_node(
                        seed,
                        notebook,
                        question_map,
                        level=0,
                        parent_id=None,
                        run_id=run_id,
                    )
                    for seed in contra_seeds
                ]
                contra_leaf_answers = self._collect_leaf_answers(contra_nodes)
                for ans in contra_leaf_answers:
                    qid = ans.get("question_id")
                    if qid and qid not in seen_leaf_ids:
                        leaf_answers.append(ans)
                        seen_leaf_ids.add(qid)
                theme_trees.append({
                    "theme": "Contradictions & Recordkeeping",
                    "nodes": [self._node_to_dict(n) for n in contra_nodes],
                    "leaf_answers": contra_leaf_answers,
                })

        theme_summaries = []
        for theme in theme_trees:
            summary = self._synthesize_theme(theme["theme"], theme.get("leaf_answers", []))
            theme_summaries.append({
                "theme": theme["theme"],
                "summary": summary,
            })

        essay = self._synthesize_essay(theme_summaries, grand_narrative)

        return _json_safe({
            "run_id": run_id,
            "theme_trees": theme_trees,
            "leaf_answers": leaf_answers,
            "theme_summaries": theme_summaries,
            "grand_narrative": grand_narrative or {},
            "essay": essay,
        })

    def _build_node(
        self,
        question_text: str,
        notebook: ResearchNotebook,
        question_map: Dict[str, Question],
        level: int,
        parent_id: Optional[str],
        run_id: str,
        evidence_override: Optional[EvidencePack] = None,
    ) -> QuestionNode:
        node_id = str(uuid.uuid4())
        evidence = evidence_override or self._build_evidence_pack(question_text, notebook, question_map)

        node = QuestionNode(
            node_id=node_id,
            question_text=question_text,
            level=level,
            parent_id=parent_id,
            evidence=evidence,
        )

        if self._is_trivial(evidence, level):
            node.answer = self._answer_leaf(node, run_id)
            return node

        if level >= APP_CONFIG.tier0.recursive_max_depth:
            node.answer = self._answer_leaf(node, run_id)
            return node

        sub_questions = self._generate_subquestions(node, notebook)
        if not sub_questions:
            node.answer = self._answer_leaf(node, run_id)
            return node

        for sub in sub_questions:
            if not isinstance(sub, dict):
                continue
            sub_text = str(sub.get("question") or "").strip()
            sub_doc_ids = sub.get("doc_ids") or []
            if not sub_text or not sub_doc_ids:
                continue
            sub_pack = EvidencePack(doc_ids=sub_doc_ids, block_ids=[])
            child = self._build_node(
                sub_text,
                notebook,
                question_map,
                level=level + 1,
                parent_id=node_id,
                run_id=run_id,
                evidence_override=sub_pack,
            )
            node.children.append(child)

        return node

    def _collect_leaf_answers(self, nodes: List[QuestionNode]) -> List[Dict[str, Any]]:
        answers: List[Dict[str, Any]] = []
        for node in nodes:
            if node.answer:
                answers.append(node.answer)
            if node.children:
                answers.extend(self._collect_leaf_answers(node.children))
        return answers

    def _is_trivial(self, evidence: EvidencePack, level: int) -> bool:
        min_docs = APP_CONFIG.tier0.recursive_min_docs
        max_docs = APP_CONFIG.tier0.recursive_max_docs
        count = len(evidence.doc_ids)
        if count == 0:
            return True
        if count < min_docs:
            return True
        return count >= min_docs and count <= max_docs

    def _build_evidence_pack(
        self,
        question_text: str,
        notebook: ResearchNotebook,
        question_map: Dict[str, Question],
    ) -> EvidencePack:
        doc_ids: List[str] = []
        block_ids: List[str] = []

        q = question_map.get(self._normalize_question(question_text))
        if q:
            doc_ids.extend(q.evidence_doc_ids or [])
            doc_ids.extend(q.answerability_sample or [])
            block_ids.extend(q.evidence_block_ids or [])

            if q.pattern_source:
                for pattern in notebook.patterns.values():
                    if pattern.pattern_text == q.pattern_source:
                        doc_ids.extend(pattern.evidence_doc_ids or [])
                        block_ids.extend(pattern.evidence_block_ids or [])

            if q.contradiction_source:
                for contra in notebook.contradictions:
                    key = f"{contra.source_a} vs {contra.source_b}"
                    if key == q.contradiction_source:
                        doc_ids.extend([contra.source_a, contra.source_b])

        doc_ids = [str(d).split("::")[0] for d in doc_ids if d]
        block_ids = [str(b) for b in block_ids if b]

        # unique, preserve order
        doc_ids = list(dict.fromkeys(doc_ids))
        block_ids = list(dict.fromkeys(block_ids))

        # cap evidence pack
        max_docs = APP_CONFIG.tier0.recursive_max_docs
        if len(doc_ids) > max_docs:
            doc_ids = doc_ids[:max_docs]

        return EvidencePack(doc_ids=doc_ids, block_ids=block_ids)

    def _generate_subquestions(self, node: QuestionNode, notebook: ResearchNotebook) -> List[str]:
        allowed_doc_ids = list(node.evidence.doc_ids or [])
        if not allowed_doc_ids:
            return []

        sources_text, _ = self._build_evidence_snippets(node.question_text, allowed_doc_ids)
        evidence_blocks = sources_text.split("\n\n") if sources_text else []
        evidence_hints = "\n\n".join(evidence_blocks[:3]) if evidence_blocks else "- (none)"

        prompt = SUBQUESTION_PROMPT.format(
            question=node.question_text,
            evidence_hints=evidence_hints,
            allowed_doc_ids=", ".join(allowed_doc_ids[:8]),
            count=APP_CONFIG.tier0.recursive_subquestion_count,
        )

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You break questions into answerable sub-questions."},
                {"role": "user", "content": prompt},
            ],
            profile=self.leaf_profile,
            temperature=0.2,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        data = parse_llm_json(response.content, default=[]) if response.success else []
        if not isinstance(data, list):
            return []

        min_docs = getattr(APP_CONFIG.tier0, "recursive_subquestion_min_docs", 1)
        cleaned: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, str):
                continue
            if not isinstance(item, dict):
                continue
            qtext = str(item.get("question") or "").strip()
            doc_ids = item.get("doc_ids") or []
            if isinstance(doc_ids, str):
                doc_ids = [d.strip() for d in doc_ids.split(",") if d.strip()]
            if not isinstance(doc_ids, list):
                doc_ids = []
            doc_ids = [str(d).split("::")[0] for d in doc_ids if d]
            doc_ids = [d for d in doc_ids if d in allowed_doc_ids]
            if not qtext or len(doc_ids) < min_docs:
                continue
            cleaned.append({"question": qtext, "doc_ids": list(dict.fromkeys(doc_ids))})

        return cleaned

    def _answer_leaf(self, node: QuestionNode, run_id: str) -> Optional[Dict[str, Any]]:
        if not node.evidence.doc_ids:
            payload = {
                "run_id": run_id,
                "question_id": node.node_id,
                "question_text": node.question_text,
                "level": node.level,
                "evidence_doc_ids": [],
                "topic_sentence": "insufficient evidence",
                "evidence": [],
                "analysis": "",
                "uncertainty": "No documents available for this question.",
                "missing_evidence": "Document evidence was not found.",
                "model": APP_CONFIG.llm_profiles.get(self.leaf_profile, {}).get("model")
                or APP_CONFIG.llm_profiles.get("quality", {}).get("model"),
                "profile": self.leaf_profile,
                "created_at": datetime.utcnow().isoformat(),
            }
            try:
                self._leaf_coll.insert_one(payload)
            except Exception as exc:
                debug_print(f"Leaf answer insert failed: {exc}")
            return payload

        sources_text, sources_index = self._build_evidence_snippets(node.question_text, node.evidence.doc_ids)
        if not sources_text:
            return None

        source_index_text = sources_index.get("text", "- (none)")
        prompt = LEAF_ANSWER_PROMPT.format(
            question=node.question_text,
            source_index=source_index_text,
            sources=sources_text,
        )

        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You answer using only the sources."},
                {"role": "user", "content": prompt},
            ],
            profile=self.leaf_profile,
            temperature=0.1,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )

        data = parse_llm_json(response.content, default={}) if response.success else {}
        if not isinstance(data, dict):
            data = {}

        topic_sentence = data.get("topic_sentence") or data.get("answer") or "insufficient evidence"
        analysis_text = data.get("analysis", "")
        if isinstance(topic_sentence, str):
            lowered = topic_sentence.strip().lower()
            if (
                topic_sentence.strip().endswith("?")
                or lowered == node.question_text.lower()
                or lowered.startswith("the sources addressed")
            ):
                if analysis_text:
                    topic_sentence = analysis_text.split(".")[0].strip() + "."
                else:
                    topic_sentence = f"The sources addressed whether {node.question_text.rstrip('?').lower()}."
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        label_to_id = sources_index.get("label_to_id", {})
        label_to_year = sources_index.get("label_to_year", {})
        doc_id_to_year = sources_index.get("doc_id_to_year", {})
        normalized_evidence = []
        evidence_quotes = []
        evidence_years = set()
        for item in evidence:
            if not isinstance(item, dict):
                continue
            label = item.get("source_label")
            doc_id = item.get("doc_id") or (label_to_id.get(label) if label else None)
            quote = item.get("quote", "")
            evidence_quotes.append(quote)
            for year in _extract_years(quote):
                evidence_years.add(year)
            year_hint = None
            if doc_id and doc_id in doc_id_to_year:
                year_hint = doc_id_to_year.get(doc_id)
            elif label and label in label_to_year:
                year_hint = label_to_year.get(label)
            if year_hint:
                try:
                    evidence_years.add(int(year_hint))
                except (TypeError, ValueError):
                    pass
            normalized_evidence.append({
                "source_label": label,
                "doc_id": doc_id,
                "quote": quote,
                "reason": item.get("reason", ""),
            })

        if evidence_quotes:
            lower_quotes = " ".join(evidence_quotes).lower()
            form_markers = [
                "questions to be asked",
                "have you ever",
                "are you now in good health",
                "i hereby apply",
                "i hereby certify",
                "were you ever employed",
            ]
            form_like = "?" in " ".join(evidence_quotes) or any(m in lower_quotes for m in form_markers)
            risky_markers = [
                "increased",
                "decreased",
                "caused",
                "led to",
                "resulted",
                "more likely",
                "less likely",
                "improved",
                "worsened",
            ]
            if form_like and any(m in topic_sentence.lower() for m in risky_markers):
                topic_sentence = (
                    "The surviving forms indicate that the institution asked about "
                    f"{node.question_text.rstrip('?').lower()}, but they do not record outcomes."
                )
                if not analysis_text:
                    analysis_text = "The evidence consists of application questions rather than reported outcomes."
            if len({ev.get('doc_id') for ev in normalized_evidence if ev.get('doc_id')}) < 2 and any(m in topic_sentence.lower() for m in risky_markers):
                topic_sentence = (
                    "The available record is limited and does not support a strong causal or comparative claim."
                )
                if not analysis_text:
                    analysis_text = "Only a single source supports this point; additional corroboration is needed."

        payload = {
            "run_id": run_id,
            "question_id": node.node_id,
            "question_text": node.question_text,
            "level": node.level,
            "evidence_doc_ids": node.evidence.doc_ids,
            "evidence_block_ids": node.evidence.block_ids,
            "topic_sentence": topic_sentence,
            "evidence": normalized_evidence,
            "analysis": analysis_text,
            "evidence_years": sorted(evidence_years),
            "uncertainty": data.get("uncertainty", ""),
            "missing_evidence": data.get("missing_evidence", ""),
            "model": APP_CONFIG.llm_profiles.get(self.leaf_profile, {}).get("model")
            or APP_CONFIG.llm_profiles.get("quality", {}).get("model"),
            "profile": self.leaf_profile,
            "created_at": datetime.utcnow().isoformat(),
        }

        try:
            self._leaf_coll.insert_one(payload)
        except Exception as exc:
            debug_print(f"Leaf answer insert failed: {exc}")

        return payload

    def _build_evidence_snippets(self, question_text: str, doc_ids: List[str]) -> Tuple[str, Dict[str, Any]]:
        if not doc_ids:
            return "", {}

        tokens = set(re.findall(r"[a-z0-9]{3,}", question_text.lower()))
        entities = set(re.findall(r"\b[A-Z][A-Za-z&.\-]{2,}\b", question_text))
        years = set(re.findall(r"\b(?:18|19|20)\d{2}\b", question_text))
        boilerplate_markers = [
            "questions to be asked",
            "have you ever",
            "are you now in good health",
            "do you",
            "i hereby apply",
            "i hereby certify",
        ]
        parent_meta = self.doc_store.hydrate_parent_metadata(doc_ids)
        snippets = []
        label_to_id: Dict[str, str] = {}
        label_to_year: Dict[str, int] = {}
        doc_id_to_year: Dict[str, int] = {}

        def score_text(text: str) -> int:
            if not tokens or not text:
                return 0
            lower = text.lower()
            score = sum(1 for t in tokens if t in lower)
            if entities:
                score += sum(2 for e in entities if e.lower() in lower)
            if years:
                score += sum(2 for y in years if y in lower)
            if any(marker in lower for marker in boilerplate_markers):
                score -= 2
            if lower.count("?") >= 3:
                score -= 1
            return score

        def extract_window(text: str) -> str:
            lower = text.lower()
            best_idx = None
            for term in sorted(tokens, key=len, reverse=True):
                idx = lower.find(term)
                if idx >= 0:
                    best_idx = idx
                    break
            if best_idx is None:
                return text[:700]
            start = max(0, best_idx - 300)
            end = min(len(text), best_idx + 400)
            return text[start:end]

        source_idx = 1
        for doc_id in doc_ids:
            # Try chunked structure first
            chunks = list(
                self.doc_store.chunks_coll.find({"document_id": doc_id}).sort("chunk_index", 1)
            )
            scored: List[Tuple[int, str]] = []
            if chunks:
                for ch in chunks:
                    text = ch.get("text") or ch.get("ocr_text") or ""
                    if not text:
                        continue
                    snippet = extract_window(text.strip().replace("\n", " "))
                    scored.append((score_text(snippet), snippet[:900]))

                scored.sort(key=lambda x: x[0], reverse=True)
                if scored and scored[0][0] <= 0:
                    scored = scored[:1]
                else:
                    scored = scored[:2]
            else:
                # fallback to embedded text
                try:
                    from bson import ObjectId
                    oid = ObjectId(doc_id)
                    query = {"_id": oid}
                except Exception:
                    query = {"_id": doc_id}
                doc = self.doc_store.documents_coll.find_one(
                    query,
                    {"ocr_text": 1, "content": 1, "text": 1, "summary": 1, "filename": 1},
                )
                text = ""
                if doc:
                    text = doc.get("ocr_text") or doc.get("content") or doc.get("text") or doc.get("summary") or ""
                if text:
                    snippet = extract_window(text.strip().replace("\n", " "))
                    scored = [(score_text(snippet), snippet[:900])]

            for _, text in scored:
                label = f"Source {source_idx}"
                meta = parent_meta.get(doc_id, {})
                meta_bits = []
                if meta.get("filename"):
                    meta_bits.append(meta.get("filename"))
                if meta.get("source_type"):
                    meta_bits.append(str(meta.get("source_type")))
                year_value = meta.get("metadata", {}).get("year")
                if year_value:
                    meta_bits.append(str(year_value))
                    try:
                        year_int = int(year_value)
                        label_to_year[label] = year_int
                        doc_id_to_year[doc_id] = year_int
                    except (TypeError, ValueError):
                        pass
                meta_str = " | ".join(meta_bits)
                snippets.append((label, doc_id, text, meta_str))
                label_to_id[label] = doc_id
                source_idx += 1

        if not snippets:
            return "", {}

        sources_text = "\n\n".join([
            f"[{label} doc_id:{doc_id}{' | ' + meta if meta else ''}]\n{snippet}"
            for label, doc_id, snippet, meta in snippets
        ])
        source_index_text = "\n".join([
            f"{label}: {doc_id}"
            for label, doc_id, _, _ in snippets
        ])

        return sources_text, {
            "text": source_index_text or "- (none)",
            "label_to_id": label_to_id,
            "label_to_year": label_to_year,
            "doc_id_to_year": doc_id_to_year,
        }

    def _synthesize_theme(self, theme: str, leaf_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        subset = [a for a in leaf_answers if a.get("question_text")]
        max_leaves = APP_CONFIG.tier0.recursive_theme_max_leaves
        subset = sorted(
            subset,
            key=lambda a: (a.get("evidence_years") or [9999])[0]
            if isinstance(a.get("evidence_years"), list) and a.get("evidence_years")
            else 9999,
        )
        subset = subset[:max_leaves]
        formatted = []
        for item in subset[:10]:
            evidence = item.get("evidence") or []
            evidence_lines = []
            for ev in evidence[:3]:
                doc_id = ev.get("doc_id") or ev.get("source_label") or ""
                quote = (ev.get("quote") or "")[:140].replace("\n", " ")
                if quote:
                    evidence_lines.append(f"{quote} [{doc_id}]")
            evidence_lines = evidence_lines[:2]
            evidence_block = " | ".join(evidence_lines) if evidence_lines else "(no evidence)"
            formatted.append(
                f"Q: {item.get('question_text')}\n"
                f"Topic: {item.get('topic_sentence')}\n"
                f"Evidence: {evidence_block}\n"
                f"Analysis: {item.get('analysis', '')}"
            )
        hints = "\n\n".join(formatted) or "- (none)"
        if APP_CONFIG.tier0.recursive_theme_use_llm:
            prompt = THEME_SYNTHESIS_PROMPT.format(theme=theme, leaf_answers=hints)
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": "You synthesize theme summaries."},
                    {"role": "user", "content": prompt},
                ],
                profile=self.writer_profile,
                temperature=0.2,
                timeout=APP_CONFIG.tier0.llm_timeout,
            )
            data = parse_llm_json(response.content, default={}) if response.success else {}
            if isinstance(data, dict) and data.get("paragraphs"):
                return data

        # Deterministic theme assembly from leaf answers
        paragraphs = []
        gaps = []
        for item in subset:
            evidence = item.get("evidence") or []
            if not evidence:
                gaps.append(item.get("question_text"))
                continue
            evidence_sentences = []
            for ev in evidence:
                quote = (ev.get("quote") or "").strip()
                doc_id = ev.get("doc_id") or ev.get("source_label") or ""
                if quote:
                    evidence_sentences.append(f"{quote} [{doc_id}]")
            paragraphs.append({
                "subquestion": item.get("question_text"),
                "topic_sentence": item.get("topic_sentence"),
                "evidence_sentences": evidence_sentences or ["(no evidence)"],
                "analysis_sentence": item.get("analysis", ""),
                "approx_years": item.get("evidence_years", []),
            })

        paragraphs = sorted(
            paragraphs,
            key=lambda p: (p.get("approx_years") or [9999])[0]
            if isinstance(p.get("approx_years"), list)
            else 9999,
        )

        return {
            "theme_summary": f"{theme}: {len(paragraphs)} evidence-backed subquestions synthesized.",
            "paragraphs": paragraphs,
            "contradictions": [],
            "gaps": gaps,
        }

    def _synthesize_essay(
        self,
        theme_summaries: List[Dict[str, Any]],
        grand_narrative: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not theme_summaries:
            return {}
        grand_narrative = grand_narrative or {}
        hints = "\n\n".join([
            f"Theme: {t.get('theme')}\n"
            f"Summary: {t.get('summary', {}).get('theme_summary', '')}\n"
            f"Paragraphs: {t.get('summary', {}).get('paragraphs', [])}"
            for t in theme_summaries
        ])
        if grand_narrative:
            hints = (
                f"Grand Narrative: {grand_narrative.get('question', '')}\n"
                f"Purpose: {grand_narrative.get('purpose_statement', '')}\n"
                f"Scope: {grand_narrative.get('scope', {})}\n\n"
                + hints
            )
        prompt = ESSAY_PROMPT.format(
            theme_summaries=hints,
            min_words=APP_CONFIG.tier0.essay_min_words,
        )
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You write long-form essays from evidence."},
                {"role": "user", "content": prompt},
            ],
            profile=self.writer_profile,
            temperature=0.2,
            timeout=APP_CONFIG.tier0.llm_timeout,
        )
        data = parse_llm_json(response.content, default={}) if response.success else {}
        if isinstance(data, dict) and data.get("essay"):
            if self._validate_essay(data.get("essay", "")):
                return data

        # Fallback: assemble essay directly from theme paragraphs
        sections, body_parts = self._assemble_essay(theme_summaries, grand_narrative)
        essay_text = "\n".join(body_parts).strip()
        return {"essay": essay_text, "sections": sections}

    def _validate_essay(self, essay: str) -> bool:
        if not essay:
            return False
        words = len(str(essay).split())
        if words < APP_CONFIG.tier0.essay_min_words:
            return False
        citations = re.findall(r"\[[^\]]+\]", essay)
        if len(citations) < max(3, len(essay.split()) // 300):
            return False
        return True

    def _assemble_essay(
        self,
        theme_summaries: List[Dict[str, Any]],
        grand_narrative: Optional[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        sections: List[str] = []
        body_parts: List[str] = []

        intro = self._build_intro(grand_narrative, theme_summaries)
        body_parts.append(intro)

        for theme in theme_summaries:
            name = theme.get("theme", "Theme")
            summary = theme.get("summary", {})
            paragraphs = summary.get("paragraphs", [])
            sections.append(name)
            body_parts.append(f"\n{name}\n")
            paragraphs = sorted(
                paragraphs,
                key=lambda p: (p.get("approx_years") or [9999])[0]
                if isinstance(p.get("approx_years"), list)
                else 9999,
            )
            for para in paragraphs:
                body_parts.append(self._paragraph_from_summary(para))

        body_parts.append(self._build_counterargument(theme_summaries))
        body_parts.append(self._build_gaps(theme_summaries))
        body_parts.append(self._build_conclusion(theme_summaries))

        if len(" ".join(body_parts).split()) < APP_CONFIG.tier0.essay_min_words:
            body_parts.append(self._build_methods_note())
            body_parts.extend(self._build_evidence_register(theme_summaries))
        if len(" ".join(body_parts).split()) < APP_CONFIG.tier0.essay_min_words:
            body_parts.extend(self._build_detailed_walkthrough(theme_summaries))

        return sections, body_parts

    def _build_intro(
        self,
        grand_narrative: Optional[Dict[str, Any]],
        theme_summaries: List[Dict[str, Any]],
    ) -> str:
        purpose = ""
        thesis = ""
        terms = []
        scope = {}
        if grand_narrative:
            purpose = grand_narrative.get("purpose_statement") or ""
            thesis = grand_narrative.get("question") or ""
            terms = grand_narrative.get("terms_to_define") or []
            scope = grand_narrative.get("scope") or {}
        themes = [t.get("theme") for t in theme_summaries if t.get("theme")]
        roadmap = ", ".join(themes[:3])
        if not purpose and thesis:
            clause = self._to_purpose_clause(thesis)
            purpose = (
                "I am studying the Relief Department archive because I want to know "
                f"{clause} in order to help my readers understand "
                f"{roadmap or 'the institution’s welfare practices'}."
            )
        intro_bits = []
        if purpose:
            intro_bits.append(purpose)
        if thesis:
            intro_bits.append(f"Thesis: {thesis}")
        if scope:
            intro_bits.append(f"Scope: {scope}.")
        if terms:
            intro_bits.append(f"Key terms to define: {', '.join([str(t) for t in terms][:4])}.")
        if roadmap:
            intro_bits.append(f"This essay follows evidence on {roadmap}.")
        intro_bits.append(
            "Scope and evidence base: the archive is fragmentary, so the argument relies on the surviving records and treats silence as meaningful."
        )
        intro = " ".join(intro_bits).strip()
        return self._ensure_length(intro, APP_CONFIG.tier0.essay_intro_min_words)

    def _build_counterargument(self, theme_summaries: List[Dict[str, Any]]) -> str:
        return self._ensure_length(
            "Counterargument and limits: Some interpretations could be read as routine administrative practice rather than deliberate labor control. Where the evidence consists mainly of standardized forms or single-source claims, the analysis remains tentative and the archive does not resolve alternative explanations.",
            APP_CONFIG.tier0.essay_paragraph_min_words,
        )

    def _build_gaps(self, theme_summaries: List[Dict[str, Any]]) -> str:
        gaps = []
        for theme in theme_summaries:
            summary = theme.get("summary", {})
            missing = summary.get("gaps") or []
            for gap in missing:
                if gap:
                    gaps.append(gap)
        if gaps:
            gap_line = "Gaps and next questions: " + "; ".join(list(dict.fromkeys(gaps))[:6])
        else:
            gap_line = "Gaps and next questions: The archive is thin on demographic variation and comparative context, so these lines of inquiry remain open."
        return self._ensure_length(gap_line, APP_CONFIG.tier0.essay_paragraph_min_words)

    def _to_purpose_clause(self, thesis: str) -> str:
        if not thesis:
            return "what the records reveal"
        text = thesis.strip()
        if text.endswith("?"):
            text = text[:-1]
        if text:
            text = text[0].lower() + text[1:]
        prefixes = [
            ("How did ", "how "),
            ("Why did ", "why "),
            ("What was ", "what "),
            ("What were ", "what "),
            ("Did ", "whether "),
            ("Do ", "whether "),
            ("Does ", "whether "),
        ]
        for prefix, replacement in prefixes:
            if thesis.startswith(prefix):
                rest = thesis[len(prefix):].strip()
                if rest.endswith("?"):
                    rest = rest[:-1]
                if rest:
                    rest = rest[0].lower() + rest[1:]
                if prefix.startswith("How") or prefix.startswith("Why"):
                    rest = self._past_tense_first_verb(rest)
                return replacement + rest
        return text or thesis

    def _past_tense_first_verb(self, clause: str) -> str:
        tokens = clause.split()
        skip = {"the", "a", "an", "of", "in", "on", "for", "and", "or", "to"}
        for idx, tok in enumerate(tokens):
            if tok.lower() in skip:
                continue
            if tok and tok[0].islower():
                if not tok.endswith("ed"):
                    if tok.endswith("e"):
                        tok = tok + "d"
                    else:
                        tok = tok + "ed"
                tokens[idx] = tok
                break
        return " ".join(tokens)

    def _build_conclusion(self, theme_summaries: List[Dict[str, Any]]) -> str:
        return self._ensure_length(
            "Conclusion: Taken together, the themes suggest a relief system that recorded health, eligibility, and employment decisions in ways that shaped worker experience while leaving important silences. The evidence does not resolve every contradiction, but it does delineate the institutional logic that can be investigated further through deeper archival work.",
            APP_CONFIG.tier0.essay_conclusion_min_words,
        )

    def _build_methods_note(self) -> str:
        return self._ensure_length(
            "Method and limitations: This essay assembles topic sentences and evidence excerpts from the archive without adding outside context. Where evidence is sparse or limited to standardized forms, the analysis remains cautious and highlights the need for additional records.",
            APP_CONFIG.tier0.essay_paragraph_min_words,
        )

    def _build_evidence_register(self, theme_summaries: List[Dict[str, Any]]) -> List[str]:
        parts = ["\nEvidence register\n"]
        for theme in theme_summaries:
            name = theme.get("theme", "Theme")
            summary = theme.get("summary", {})
            paragraphs = summary.get("paragraphs", [])
            if not paragraphs:
                continue
            parts.append(f"{name}:")
            for para in paragraphs:
                topic = (para.get("topic_sentence") or "").strip()
                evidence = " ".join(para.get("evidence_sentences", []))
                if topic or evidence:
                    parts.append(f"- {topic} {evidence}".strip())
        return parts

    def _build_detailed_walkthrough(self, theme_summaries: List[Dict[str, Any]]) -> List[str]:
        parts = ["\nDetailed evidence walkthrough\n"]
        for theme in theme_summaries:
            name = theme.get("theme", "Theme")
            summary = theme.get("summary", {})
            paragraphs = summary.get("paragraphs", [])
            if not paragraphs:
                continue
            parts.append(f"{name}:")
            for para in paragraphs:
                subq = para.get("subquestion") or ""
                topic = para.get("topic_sentence") or ""
                evidence = " ".join(para.get("evidence_sentences", []))
                analysis = para.get("analysis_sentence", "")
                if subq:
                    parts.append(f"Subquestion: {subq}")
                if topic:
                    parts.append(f"Topic: {topic}")
                if evidence:
                    parts.append(f"Evidence: {evidence}")
                if analysis:
                    parts.append(f"Analysis: {analysis}")
        return parts

    def _paragraph_from_summary(self, para: Dict[str, Any]) -> str:
        topic = (para.get("topic_sentence") or "").strip()
        evidence = para.get("evidence_sentences") or []
        analysis = (para.get("analysis_sentence") or "").strip()
        parts = []
        if topic:
            parts.append(topic.rstrip(".") + ".")
        if evidence:
            parts.append(" ".join(evidence))
        if analysis:
            parts.append(analysis.rstrip(".") + ".")
        text = " ".join(parts).strip()
        if not text:
            text = "Insufficient evidence for this point."
        text = self._ensure_length(text, APP_CONFIG.tier0.essay_paragraph_min_words)
        return self._cap_length(text, APP_CONFIG.tier0.essay_paragraph_max_words)

    def _cap_length(self, text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).rstrip() + "..."

    def _ensure_length(self, text: str, min_words: int) -> str:
        words = text.split()
        if len(words) >= min_words:
            return text
        fillers = [
            " The surviving records are incomplete; where the evidence is thin, the analysis remains cautious and provisional.",
            " The evidence base favors administrative documentation, which requires careful interpretation rather than assumption.",
            " Archival silence here should be treated as a gap to investigate, not as a basis for inference.",
        ]
        idx = 0
        while len(text.split()) < min_words:
            text += fillers[idx % len(fillers)]
            idx += 1
        return text

    def _node_to_dict(self, node: QuestionNode) -> Dict[str, Any]:
        return {
            "node_id": node.node_id,
            "question": node.question_text,
            "level": node.level,
            "parent_id": node.parent_id,
            "evidence_doc_ids": node.evidence.doc_ids,
            "children": [self._node_to_dict(c) for c in node.children],
            "answer": node.answer,
        }
