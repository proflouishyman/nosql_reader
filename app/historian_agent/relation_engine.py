# app/historian_agent/relation_engine.py
# Purpose: Relation decisions, seed guard, and question framing enforcement.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import uuid

from config import APP_CONFIG
from llm_abstraction import LLMClient

from historian_agent.adaptive_prompts import (
    EXPAND_DIRECTION_PROMPT,
    get_why_how_prompt,
    resolve_prompt_variant,
)
from historian_agent.question_graph import QuestionNode, RelationDecisionLog
from historian_agent.tier0_utils import parse_llm_json


@dataclass
class RelationDecision:
    action: str
    target: Optional[QuestionNode] = None
    lateral_link: Optional[QuestionNode] = None
    edge: Optional[str] = None
    expand_call_result: Optional[str] = None
    budget_skipped: bool = False


class _BudgetProtocol:
    def request(self, tier: str) -> bool:  # pragma: no cover - structural protocol
        raise NotImplementedError


def enforce_why_how(
    question_text: str,
    level: str,
    llm: LLMClient,
    budget: _BudgetProtocol,
    prompt_variant: Optional[str] = None,
) -> Tuple[str, str, bool]:
    """
    Returns (final_text, question_type, was_rewritten).

    On budget exhaustion, LLM failure, parse failure, or timeout fallback:
    returns original text with question_type="what".
    """
    if not budget.request("light"):
        return question_text, "what", False

    variant = resolve_prompt_variant(prompt_variant)
    try:
        response = llm.generate(
            messages=[
                {"role": "system", "content": "You enforce historical question framing."},
                {"role": "user", "content": get_why_how_prompt(variant).format(question_text=question_text)},
            ],
            model=APP_CONFIG.tier0.ledger_expand_model,
            timeout=APP_CONFIG.tier0.llm_light_timeout,
            temperature=0.0,
        )
    except Exception:
        return question_text, "what", False

    if not response.success:
        return question_text, "what", False

    result = parse_llm_json(response.content, default={})
    if not isinstance(result, dict):
        return question_text, "what", False

    final_text = str(result.get("question") or question_text).strip() or question_text
    q_type = str(result.get("question_type") or "what").strip().lower()
    if q_type not in {"why", "how", "compare", "change_continuity", "explain", "what"}:
        q_type = "what"

    if q_type == "what" and level in ("meso", "macro"):
        return question_text, "what", False

    changed = bool(result.get("changed", False))
    return final_text, q_type, changed


def _expand_call(q_a: str, q_b: str, llm: LLMClient) -> str:
    try:
        response = llm.generate(
            messages=[
                {"role": "system", "content": "You compare question specificity."},
                {"role": "user", "content": EXPAND_DIRECTION_PROMPT.format(q_a=q_a, q_b=q_b)},
            ],
            model=APP_CONFIG.tier0.ledger_expand_model,
            timeout=APP_CONFIG.tier0.llm_light_timeout,
            temperature=0.0,
        )
    except Exception:
        return "LATERAL"

    if not response.success:
        return "LATERAL"

    content = (response.content or "").strip().upper()
    for token in ("BROADER_A", "BROADER_B", "LATERAL"):
        if token in content:
            return token
    return "LATERAL"


def _decide_relation_standard(
    candidate_text: str,
    similar_nodes: List[Tuple[QuestionNode, float]],
    llm: LLMClient,
    budget: _BudgetProtocol,
) -> RelationDecision:
    if not similar_nodes:
        return RelationDecision(action="new_thread")

    best_node, best_sim = similar_nodes[0]
    cfg = APP_CONFIG.tier0

    if best_sim >= cfg.ledger_dedupe_threshold:
        return RelationDecision(action="merge", target=best_node, edge="overlaps_with")

    if best_sim >= cfg.ledger_merge_threshold:
        if budget.request("light"):
            direction = _expand_call(candidate_text, best_node.question_text, llm)
            skipped = False
        else:
            direction = "LATERAL"
            skipped = True

        if direction == "BROADER_A":
            return RelationDecision(
                action="generalize",
                target=best_node,
                edge="generalizes",
                expand_call_result=direction,
                budget_skipped=skipped,
            )
        if direction == "BROADER_B":
            return RelationDecision(
                action="decompose",
                target=best_node,
                edge="decomposes_into",
                expand_call_result=direction,
                budget_skipped=skipped,
            )
        return RelationDecision(
            action="lateral",
            target=best_node,
            edge="overlaps_with",
            expand_call_result=direction,
            budget_skipped=skipped,
        )

    if best_sim >= cfg.ledger_lateral_threshold:
        return RelationDecision(action="new_thread", lateral_link=best_node, edge="overlaps_with")

    return RelationDecision(action="new_thread")


def decide_relation_with_seed_guard(
    candidate_node: QuestionNode,
    similar_nodes: List[Tuple[QuestionNode, float]],
    llm: LLMClient,
    budget: _BudgetProtocol,
    batch_label: str,
) -> Tuple[RelationDecision, RelationDecisionLog]:
    """
    Compare best-seed and best-non-seed candidates separately to avoid over-attaching
    emergent micro questions to broad seed macros.
    """
    log_id = str(uuid.uuid4())
    cfg = APP_CONFIG.tier0

    decision = RelationDecision(action="new_thread")
    seed_guard_fired = False
    best_match_sim: Optional[float] = None

    seed_matches = [(n, s) for n, s in similar_nodes if n.origin == "seed"]
    non_seed_matches = [(n, s) for n, s in similar_nodes if n.origin != "seed"]

    best_seed = seed_matches[0] if seed_matches else None
    best_non_seed = non_seed_matches[0] if non_seed_matches else None

    if (
        best_seed is not None
        and candidate_node.level == "micro"
        and best_seed[0].level == "macro"
        and best_seed[1] >= cfg.ledger_lateral_threshold
    ):
        seed_node, seed_sim = best_seed
        non_seed_sim = best_non_seed[1] if best_non_seed else 0.0
        margin = float(cfg.seed_attach_margin)
        if seed_sim >= (non_seed_sim + margin):
            seed_guard_fired = True
            decision = RelationDecision(action="decompose", target=seed_node, edge="decomposes_into")
            best_match_sim = seed_sim

    if not seed_guard_fired:
        decision = _decide_relation_standard(
            candidate_text=candidate_node.question_text,
            similar_nodes=similar_nodes,
            llm=llm,
            budget=budget,
        )
        if similar_nodes:
            best_match_sim = similar_nodes[0][1]

    matched = decision.target or decision.lateral_link
    log = RelationDecisionLog(
        log_id=log_id,
        candidate_text=candidate_node.question_text,
        matched_node_id=matched.node_id if matched else None,
        matched_node_text=matched.question_text if matched else None,
        similarity_score=best_match_sim,
        threshold_used=float(cfg.ledger_lateral_threshold),
        action=decision.action,
        edge_type=decision.edge,
        expand_call_result=decision.expand_call_result,
        seed_guard_fired=seed_guard_fired,
        budget_skipped=decision.budget_skipped,
        batch_label=batch_label,
        timestamp=datetime.now().isoformat(),
    )

    return decision, log
