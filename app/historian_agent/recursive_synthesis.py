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

PARENT QUESTION:
{question}

EVIDENCE HINTS (from the archive):
{evidence_hints}

TASK:
Generate {count} sub-questions that are more specific and more directly answerable.
Each sub-question should narrow scope (time/place/actors) or target a mechanism.

Return ONLY JSON array:
["sub-question 1", "sub-question 2", ...]
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
1) Defines scope and evidence base.
2) Presents theme-by-theme synthesis.
3) Highlights contradictions.
4) Identifies gaps and next questions.
5) Minimum length: {min_words} words.
6) For each theme, include 2-4 paragraphs built from the provided theme paragraphs.
7) Each paragraph must include at least one evidence citation in [doc_id] format.
8) Use a clear thesis and brief roadmap in the opening.
9) End with a short conclusion that ties themes together without introducing new facts.

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


class RecursiveSynthesizer:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.leaf_profile = APP_CONFIG.tier0.recursive_leaf_profile
        self.writer_profile = APP_CONFIG.tier0.recursive_writer_profile
        self.doc_store = DocumentStore()
        self._mongo = MongoDBConnection()
        self._leaf_coll = self._mongo.get_collection(APP_CONFIG.tier0.leaf_answers_collection)
        self._ensure_indexes()

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
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not APP_CONFIG.tier0.recursive_enabled:
            return {}

        run_id = run_id or datetime.utcnow().strftime("recursive_%Y%m%d_%H%M%S")
        question_map = {q.question_text: q for q in questions}

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

        theme_summaries = []
        for theme in theme_trees:
            summary = self._synthesize_theme(theme["theme"], theme.get("leaf_answers", []))
            theme_summaries.append({
                "theme": theme["theme"],
                "summary": summary,
            })

        essay = self._synthesize_essay(theme_summaries)

        return _json_safe({
            "run_id": run_id,
            "theme_trees": theme_trees,
            "leaf_answers": leaf_answers,
            "theme_summaries": theme_summaries,
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
    ) -> QuestionNode:
        node_id = str(uuid.uuid4())
        evidence = self._build_evidence_pack(question_text, notebook, question_map)

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
            child = self._build_node(
                sub,
                notebook,
                question_map,
                level=level + 1,
                parent_id=node_id,
                run_id=run_id,
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
        if count == 0 and level > 0:
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

        q = question_map.get(question_text)
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
        hints = []
        for pattern in list(notebook.patterns.values())[:5]:
            hints.append(f"- {pattern.pattern_text} (n={len(pattern.evidence_doc_ids)})")
        for contra in list(notebook.contradictions)[:3]:
            hints.append(f"- {contra.claim_a} vs {contra.claim_b}")
        evidence_hints = "\n".join(hints) or "- (none)"

        prompt = SUBQUESTION_PROMPT.format(
            question=node.question_text,
            evidence_hints=evidence_hints,
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
        return [str(q).strip() for q in data if str(q).strip()]

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
                "model": APP_CONFIG.llm_profiles.get("quality", {}).get("model"),
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
        normalized_evidence = []
        evidence_quotes = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            label = item.get("source_label")
            doc_id = item.get("doc_id") or (label_to_id.get(label) if label else None)
            quote = item.get("quote", "")
            evidence_quotes.append(quote)
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
            "uncertainty": data.get("uncertainty", ""),
            "missing_evidence": data.get("missing_evidence", ""),
            "model": APP_CONFIG.llm_profiles.get("quality", {}).get("model"),
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
        label_to_name: Dict[str, str] = {}

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
                if meta.get("metadata", {}).get("year"):
                    meta_bits.append(str(meta.get("metadata", {}).get("year")))
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
        }

    def _synthesize_theme(self, theme: str, leaf_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        subset = [a for a in leaf_answers if a.get("question_text")]
        max_leaves = APP_CONFIG.tier0.recursive_theme_max_leaves
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
            })

        return {
            "theme_summary": f"{theme}: {len(paragraphs)} evidence-backed subquestions synthesized.",
            "paragraphs": paragraphs,
            "contradictions": [],
            "gaps": gaps,
        }

    def _synthesize_essay(self, theme_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not theme_summaries:
            return {}
        hints = "\n\n".join([
            f"Theme: {t.get('theme')}\n"
            f"Summary: {t.get('summary', {}).get('theme_summary', '')}\n"
            f"Paragraphs: {t.get('summary', {}).get('paragraphs', [])}"
            for t in theme_summaries
        ])
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
            if len(str(data.get("essay", "")).split()) >= APP_CONFIG.tier0.essay_min_words:
                return data

        # Fallback: assemble essay directly from theme paragraphs
        sections = []
        body_parts = []
        body_parts.append("Scope and evidence base: This essay synthesizes the available theme paragraphs and cited excerpts only.")
        for theme in theme_summaries:
            name = theme.get("theme", "Theme")
            summary = theme.get("summary", {})
            paragraphs = summary.get("paragraphs", [])
            sections.append(name)
            body_parts.append(f"\n{name}\n")
            for para in paragraphs:
                ts = para.get("topic_sentence") or ""
                ev = " ".join(para.get("evidence_sentences", []))
                analysis = para.get("analysis_sentence", "")
                body_parts.append(f"{ts} {ev} {analysis}".strip())
        if len(" ".join(body_parts).split()) < APP_CONFIG.tier0.essay_min_words:
            body_parts.append("\nMethod and limitations: This synthesis relies on a constrained set of excerpts. Where evidence is thin or absent, conclusions remain provisional and indicate gaps for future archival work.")
            body_parts.append("\nEvidence register:")
            for theme in theme_summaries:
                name = theme.get("theme", "Theme")
                summary = theme.get("summary", {})
                paragraphs = summary.get("paragraphs", [])
                if not paragraphs:
                    continue
                body_parts.append(f"{name}:")
                for para in paragraphs:
                    ts = para.get("topic_sentence") or ""
                    ev = " ".join(para.get("evidence_sentences", []))
                    body_parts.append(f"- {ts} {ev}".strip())
            body_parts.append("\nDetailed evidence walkthrough:")
            for theme in theme_summaries:
                name = theme.get("theme", "Theme")
                summary = theme.get("summary", {})
                paragraphs = summary.get("paragraphs", [])
                if not paragraphs:
                    continue
                body_parts.append(f"{name}:")
                for para in paragraphs:
                    subq = para.get("subquestion") or ""
                    ts = para.get("topic_sentence") or ""
                    ev = " ".join(para.get("evidence_sentences", []))
                    analysis = para.get("analysis_sentence", "")
                    body_parts.append(f"Subquestion: {subq}")
                    body_parts.append(f"Topic: {ts}")
                    body_parts.append(f"Evidence: {ev}")
                    if analysis:
                        body_parts.append(f"Analysis: {analysis}")

        return {
            "essay": "\n".join(body_parts).strip(),
            "sections": sections,
        }

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
