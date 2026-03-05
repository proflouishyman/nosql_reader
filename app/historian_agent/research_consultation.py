# app/historian_agent/research_consultation.py
# Purpose: Structured research consultation contract for adaptive exploration.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import APP_CONFIG
from historian_agent.adaptive_prompts import get_seed_extraction_prompt, resolve_prompt_variant
from historian_agent.tier0_utils import parse_llm_json


_CURATED_AXES = {
    "time / periodization": "time",
    "time": "time",
    "place / geography": "place",
    "place": "place",
    "occupation / job category": "occupation",
    "occupation": "occupation",
    "ethnicity / national origin": "ethnicity",
    "ethnicity": "ethnicity",
    "gender": "gender",
    "unionization / labor organizing": "unionization",
    "unionization": "unionization",
    "management / supervision": "management",
    "management": "management",
    "legislation / policy": "legislation",
    "legislation": "legislation",
    "wages / compensation": "wages",
    "wages": "wages",
    "family / dependents": "family",
    "family": "family",
}

_VALID_SORTS = {"archival", "chronological", "record_type", "person", "balanced"}


@dataclass
class ResearchBrief:
    """Research consultation payload persisted with adaptive run metadata."""

    primary_lens: str
    prior_hypotheses: str = ""
    axes: List[str] = field(default_factory=list)
    surprise_expectations: str = ""
    exclusions: str = ""
    sort_order: str = "archival"
    confirmed: bool = False
    created_at: str = ""

    def to_legacy_lens(self) -> str:
        """Compatibility shim for legacy prompt fields that expect a single string."""
        parts = [self.primary_lens.strip()]
        if self.prior_hypotheses.strip():
            parts.append(f"Prior hypotheses: {self.prior_hypotheses.strip()}")
        if self.exclusions.strip():
            parts.append(f"Excluding: {self.exclusions.strip()}")
        return " ".join([part for part in parts if part]).strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_lens": self.primary_lens,
            "prior_hypotheses": self.prior_hypotheses,
            "axes": list(self.axes),
            "surprise_expectations": self.surprise_expectations,
            "exclusions": self.exclusions,
            "sort_order": self.sort_order,
            "confirmed": bool(self.confirmed),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ResearchBrief":
        return normalize_brief(payload)


def _normalize_sort(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in _VALID_SORTS:
        return raw
    fallback = str(getattr(APP_CONFIG.tier0, "adaptive_default_sort", "archival")).strip().lower()
    return fallback if fallback in _VALID_SORTS else "archival"


def _normalize_axes(values: Any) -> List[str]:
    candidates: List[str] = []
    if isinstance(values, str):
        candidates = [part.strip() for part in values.split(",")]
    elif isinstance(values, Iterable):
        for item in values:
            candidates.extend([part.strip() for part in str(item).split(",")])

    cleaned: List[str] = []
    seen = set()
    for item in candidates:
        if not item:
            continue
        key = _CURATED_AXES.get(item.strip().lower(), item.strip().lower())
        if key and key not in seen:
            seen.add(key)
            cleaned.append(key)

    if cleaned:
        return cleaned

    defaults = getattr(APP_CONFIG.tier0, "question_axes", []) or []
    return [str(axis).strip().lower() for axis in defaults if str(axis).strip()]


def normalize_brief(payload: Any) -> ResearchBrief:
    """Normalize API/UI payload into a stable ResearchBrief contract."""
    if isinstance(payload, ResearchBrief):
        brief = payload
    elif isinstance(payload, dict):
        brief = ResearchBrief(
            primary_lens=str(payload.get("primary_lens") or payload.get("research_lens") or "").strip(),
            prior_hypotheses=str(payload.get("prior_hypotheses") or "").strip(),
            axes=_normalize_axes(payload.get("axes") or payload.get("question_axes") or []),
            surprise_expectations=str(payload.get("surprise_expectations") or "").strip(),
            exclusions=str(payload.get("exclusions") or "").strip(),
            sort_order=_normalize_sort(payload.get("sort_order")),
            confirmed=bool(payload.get("confirmed", False)),
            created_at=str(payload.get("created_at") or "").strip(),
        )
    else:
        text = str(payload or "").strip()
        brief = ResearchBrief(
            primary_lens=text,
            axes=_normalize_axes([]),
            sort_order=_normalize_sort(None),
        )

    if not brief.created_at:
        brief.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if not brief.axes:
        brief.axes = _normalize_axes([])
    brief.sort_order = _normalize_sort(brief.sort_order)
    return brief


def _topic_from_text(text: str) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "worker injury patterns in this corpus"
    clauses = re.split(r"[.;:!?]", cleaned)
    topic = clauses[0].strip()
    words = topic.split()
    if len(words) > 18:
        topic = " ".join(words[:18]).rstrip(",")
    return topic or "worker injury patterns in this corpus"


def heuristic_seed_questions(
    text: str,
    axes: List[str],
    max_questions: int = 3,
) -> List[Tuple[str, str, List[str]]]:
    """Deterministic fallback so adaptive mode always has usable seed candidates."""
    topic = _topic_from_text(text)
    axis_set = set(axes)
    out: List[Tuple[str, str, List[str]]] = []

    out.append((
        f"How did {topic} change over time across this corpus?",
        "change_continuity",
        [axis for axis in ["time"] if axis in axis_set] or axes[:1],
    ))
    out.append((
        f"Why did {topic} vary across different worker groups and institutional settings?",
        "why",
        [axis for axis in ["group", "occupation", "ethnicity", "institution", "management"] if axis in axis_set] or axes[:2],
    ))
    out.append((
        f"How did place and organizational context shape {topic}?",
        "how",
        [axis for axis in ["place", "institution", "management"] if axis in axis_set] or axes[:2],
    ))

    deduped: List[Tuple[str, str, List[str]]] = []
    seen = set()
    for question, qtype, tags in out:
        key = question.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((question, qtype, [t for t in tags if t]))
        if len(deduped) >= max_questions:
            break
    return deduped


def _seed_prompt(text: str, axes: List[str], max_questions: int, prompt_variant: Optional[str] = None) -> str:
    variant = resolve_prompt_variant(prompt_variant)
    template = get_seed_extraction_prompt(variant)
    return template.format(text=text, max_questions=max_questions, axes=", ".join(axes))


def extract_seed_questions(
    text: str,
    axes: List[str],
    llm: Optional[Any],
    model: Optional[str],
    timeout_s: int,
    max_questions: int,
    prompt_variant: Optional[str] = None,
) -> List[Tuple[str, str, List[str]]]:
    """
    Use LLM extraction when available and fall back to deterministic questions.
    Fallback is intentionally conservative to avoid returning zero seeds.
    """
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    if llm is not None:
        try:
            response = llm.generate(
                messages=[
                    {"role": "system", "content": "You extract broad historical research questions."},
                    {"role": "user", "content": _seed_prompt(cleaned, axes, max_questions, prompt_variant=prompt_variant)},
                ],
                model=model or APP_CONFIG.tier0.ledger_expand_model,
                timeout=timeout_s,
                temperature=0.2,
            )
            if response.success:
                parsed = parse_llm_json(response.content, default=[])
                if isinstance(parsed, list):
                    seeded: List[Tuple[str, str, List[str]]] = []
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        q = str(item.get("question") or "").strip()
                        qt = str(item.get("question_type") or "why").strip().lower()
                        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
                        cleaned_tags = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
                        if not q or qt == "what":
                            continue
                        seeded.append((q, qt, cleaned_tags))
                    if seeded:
                        return seeded[:max_questions]
        except Exception:
            # Graceful fallback below keeps adaptive mode available even if model parsing fails.
            pass

    return heuristic_seed_questions(cleaned, axes, max_questions=max_questions)


def recommend_sort_order(brief: ResearchBrief) -> str:
    """Heuristic recommendation used by reflection output."""
    text = " ".join([
        brief.primary_lens.lower(),
        brief.prior_hypotheses.lower(),
    ])
    if any(tok in text for tok in ["box", "folder", "finding aid", "series", "record group", "archive"]):
        return "archival"
    if any(tok in text for tok in ["change over time", "chronolog", "period", "before", "after", "decade"]):
        return "chronological"
    if any(tok in text for tok in ["person", "workers", "biograph", "prosopograph", "life history"]):
        return "person"
    if any(tok in text for tok in ["form", "record type", "genre", "document type"]):
        return "record_type"
    return brief.sort_order or _normalize_sort(None)


def build_reflection(
    brief: ResearchBrief,
    llm: Optional[Any] = None,
    prompt_variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate consultation reflection + proposed seeds for user confirmation."""
    recommended_sort = recommend_sort_order(brief)
    proposed_from_hypotheses = extract_seed_questions(
        text=brief.prior_hypotheses,
        axes=brief.axes,
        llm=llm,
        model=APP_CONFIG.tier0.ledger_expand_model,
        timeout_s=APP_CONFIG.tier0.llm_light_timeout,
        max_questions=2,
        prompt_variant=prompt_variant,
    )
    proposed_from_lens = extract_seed_questions(
        text=brief.primary_lens,
        axes=brief.axes,
        llm=llm,
        model=APP_CONFIG.tier0.ledger_expand_model,
        timeout_s=APP_CONFIG.tier0.llm_light_timeout,
        max_questions=3,
        prompt_variant=prompt_variant,
    )

    combined: List[Dict[str, Any]] = []
    seen = set()
    for question, qtype, tags in (proposed_from_hypotheses + proposed_from_lens):
        key = question.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        combined.append({
            "question": question,
            "question_type": qtype,
            "tags": tags,
            "source": "prior_hypotheses" if (question, qtype, tags) in proposed_from_hypotheses else "primary_lens",
        })
        if len(combined) >= 4:
            break

    uncertain = "Could you clarify the strongest boundary (time, place, or group) you want this run to prioritize first?"
    if not brief.primary_lens.strip():
        uncertain = "Your primary lens is empty. Please provide one sentence on what drew you to this collection."

    reflection = (
        f"Core focus: {brief.primary_lens.strip() or 'not specified'}.\n"
        f"Proposed axes: {', '.join(brief.axes) if brief.axes else '(none)'}.\n"
        f"Recommended reading order: {recommended_sort}.\n"
        f"Exclusions noted: {brief.exclusions.strip() or '(none)'}.\n"
        f"Open question: {uncertain}"
    )

    return {
        "reflection": reflection,
        "recommended_sort_order": recommended_sort,
        "proposed_seeds": combined,
        "axes": list(brief.axes),
        "uncertainty": uncertain,
    }
