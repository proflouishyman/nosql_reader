# app/historian_agent/defrag_engine.py
# Purpose: Adaptive question defragmentation and promotion logic.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Literal, Set
import uuid

from bson import ObjectId

from config import APP_CONFIG, Tier0Config
from llm_abstraction import LLMClient
from rag_base import DocumentStore

from historian_agent.adaptive_prompts import (
    get_change_continuity_prompt,
    get_promotion_prompt,
    resolve_prompt_variant,
)
from historian_agent.question_graph import (
    DefragSnapshot,
    QuestionEdge,
    QuestionGraph,
    QuestionNode,
    QuestionThread,
    RelationDecisionLog,
)
from historian_agent.tier0_utils import parse_llm_json

@dataclass
class LLMBudget:
    cap: int
    enforce: bool
    used: int = 0
    skipped: int = 0

    def request(self, tier: Literal["light", "medium", "heavy"]) -> bool:
        cost = {"light": 1, "medium": 2, "heavy": 3}[tier]
        if self.enforce and (self.used + cost) > self.cap:
            self.skipped += 1
            return False
        self.used += cost
        return True

    def reset(self) -> None:
        self.used = 0
        self.skipped = 0


def _summarize_evidence(graph: QuestionGraph, node_id: str, max_items: int = 6) -> str:
    evidence = graph.get_evidence(node_id)
    if not evidence:
        return "- (no evidence yet)"
    lines = []
    for link in evidence[:max_items]:
        lines.append(
            f"- {link.evidence_type} | doc={link.doc_id} block={link.block_id} | {link.note[:120]}"
        )
    return "\n".join(lines)


def _fetch_years_from_docs(doc_ids: List[str], doc_store: DocumentStore) -> List[int]:
    if not doc_ids:
        return []

    object_ids = []
    for doc_id in doc_ids:
        try:
            object_ids.append(ObjectId(doc_id))
        except Exception:
            continue

    query = {"_id": {"$in": object_ids}} if object_ids else {"_id": {"$in": []}}
    cursor = doc_store.documents_coll.find(query, {"year": 1})

    years: List[int] = []
    for row in cursor:
        year = row.get("year")
        if isinstance(year, int):
            years.append(year)
        elif isinstance(year, str) and year.isdigit():
            years.append(int(year))

    return sorted(set(years))


def _dedup_pass(graph: QuestionGraph, interval_index: int, batch_label: str) -> int:
    cfg = APP_CONFIG.tier0
    owners = [n for n in graph.nodes.values() if n.evidence_owner]
    merges = 0
    seen: Set[str] = set()

    for node in owners:
        if node.node_id in seen:
            continue
        similar = graph.find_similar(node.question_text, threshold=cfg.ledger_dedupe_threshold)
        for other, sim in similar:
            if other.node_id == node.node_id or other.node_id in seen:
                continue

            # Keep higher-priority node as canonical, but seed is always canonical if present.
            if node.origin == "seed" and other.origin != "seed":
                canonical, absorbed = node, other
            elif other.origin == "seed" and node.origin != "seed":
                canonical, absorbed = other, node
            else:
                canonical, absorbed = (
                    (node, other)
                    if node.priority >= other.priority
                    else (other, node)
                )

            log = RelationDecisionLog(
                log_id=str(uuid.uuid4()),
                candidate_text=absorbed.question_text,
                matched_node_id=canonical.node_id,
                matched_node_text=canonical.question_text,
                similarity_score=sim,
                threshold_used=cfg.ledger_dedupe_threshold,
                action="merge",
                edge_type="overlaps_with",
                expand_call_result=None,
                seed_guard_fired=False,
                budget_skipped=False,
                batch_label=batch_label,
                timestamp=datetime.now().isoformat(),
            )
            graph.add_decision_log(log)
            graph.merge_into(
                canonical_id=canonical.node_id,
                absorbed_id=absorbed.node_id,
                relation="overlaps_with",
                batch_label=batch_label,
                decision_log_id=log.log_id,
            )
            merges += 1
            seen.add(absorbed.node_id)

    return merges


def _rebuild_threads(graph: QuestionGraph, batch_label: str) -> None:
    owners = [node for node in graph.nodes.values() if node.evidence_owner]
    owner_ids = {node.node_id for node in owners}

    adjacency: Dict[str, Set[str]] = {nid: set() for nid in owner_ids}
    for edge in graph.edges:
        if edge.source_id in owner_ids and edge.target_id in owner_ids:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

    visited: Set[str] = set()
    graph.threads = {}

    for node_id in owner_ids:
        if node_id in visited:
            continue

        stack = [node_id]
        component: List[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency.get(current, set()) - visited)

        roots: List[str] = []
        for nid in component:
            parent_exists = any(
                e.target_id == nid and e.source_id in component
                for e in graph.edges
            )
            if not parent_exists:
                roots.append(nid)
        if not roots:
            roots = component[:1]

        axis_coverage = {axis: 0 for axis in graph.axes}
        has_seed = False
        has_emergent = False
        for nid in component:
            node = graph.nodes.get(nid)
            if node is None:
                continue
            has_seed = has_seed or node.origin == "seed"
            has_emergent = has_emergent or node.origin == "emergent"
            for tag in node.tags:
                if tag in axis_coverage:
                    axis_coverage[tag] += 1

        thread_id = str(uuid.uuid4())
        graph.threads[thread_id] = QuestionThread(
            thread_id=thread_id,
            root_node_ids=roots,
            active_score=max((graph.nodes[r].priority for r in roots if r in graph.nodes), default=0.0),
            last_updated_batch=batch_label,
            axis_coverage=axis_coverage,
            has_seed_root=has_seed,
            has_emergent_root=has_emergent,
        )


def _promote_question(
    node: QuestionNode,
    graph: QuestionGraph,
    target_level: Literal["meso", "macro"],
    llm: LLMClient,
    budget: LLMBudget,
    cfg: Tier0Config,
    prompt_variant: Optional[str] = None,
) -> Optional[QuestionNode]:
    if not budget.request("medium"):
        return None

    children = graph.get_children(node.node_id)
    evidence_summary = _summarize_evidence(graph, node.node_id)
    tension_note = ""
    if node.tension_score > 0.3:
        tension_note = (
            f"\nNOTE: This question has contradicting evidence "
            f"(tension_score={node.tension_score:.2f}). Preserve this tension — the contradiction is a finding."
        )

    try:
        variant = resolve_prompt_variant(prompt_variant)
        prompt_template = get_promotion_prompt(variant)
        response = llm.generate(
            messages=[
                {"role": "system", "content": "You promote historical questions to higher levels."},
                {"role": "user", "content": prompt_template.format(
                    current_level=node.level,
                    question_text=node.question_text,
                    children_text="\n".join(f"- {c.question_text}" for c in children[:6]) or "- (none)",
                    evidence_summary=evidence_summary,
                    tension_note=tension_note,
                    target_level=target_level,
                )},
            ],
            model=cfg.ledger_expand_model,
            timeout=cfg.llm_medium_timeout,
            temperature=0.2,
        )
    except Exception:
        return None

    if not response.success:
        return None

    parsed = parse_llm_json(response.content, default={})
    if not isinstance(parsed, dict):
        return None

    question_text = str(parsed.get("question") or "").strip()
    q_type = str(parsed.get("question_type") or "what").strip().lower()
    if not question_text or q_type == "what":
        return None

    promoted = QuestionNode(
        node_id=str(uuid.uuid4()),
        question_text=question_text,
        level=target_level,
        status="open",
        origin=node.origin,
        priority=min(node.priority + 0.1, 1.0),
        tags=list(node.tags),
        question_type=q_type,  # type: ignore[arg-type]
        confidence=node.confidence,
        generation=node.generation + 1,
        first_seen_doc=node.first_seen_doc,
        first_seen_batch=node.first_seen_batch,
        embedding=None,
        is_leaf=False,
        merge_count=node.merge_count,
        original_text=node.question_text,
        tension_score=node.tension_score,
        evidence_owner=True,
    )
    return promoted


def _check_change_continuity(
    thread: QuestionThread,
    graph: QuestionGraph,
    doc_store: DocumentStore,
    llm: LLMClient,
    budget: LLMBudget,
    cfg: Tier0Config,
    prompt_variant: Optional[str] = None,
) -> Optional[QuestionNode]:
    if not budget.request("light"):
        return None

    evidence = []
    for nid in thread.root_node_ids:
        evidence.extend(graph.get_evidence(nid))

    doc_ids = sorted({e.doc_id for e in evidence})
    years = _fetch_years_from_docs(doc_ids, doc_store)
    if not years:
        return None

    if (max(years) - min(years)) < cfg.change_continuity_min_year_span:
        return None

    roots = [graph.nodes[nid] for nid in thread.root_node_ids if nid in graph.nodes]
    if not roots:
        return None

    if any(node.question_type == "change_continuity" for node in roots):
        return None

    try:
        variant = resolve_prompt_variant(prompt_variant)
        prompt_template = get_change_continuity_prompt(variant)
        response = llm.generate(
            messages=[
                {"role": "system", "content": "You formulate change/continuity questions from archival evidence."},
                {"role": "user", "content": prompt_template.format(
                    questions="\n".join(f"- {node.question_text}" for node in roots),
                    min_year=min(years),
                    max_year=max(years),
                )},
            ],
            model=cfg.ledger_expand_model,
            timeout=cfg.llm_light_timeout,
            temperature=0.2,
        )
    except Exception:
        return None

    if not response.success:
        return None

    parsed = parse_llm_json(response.content, default={})
    if not isinstance(parsed, dict):
        return None

    question_text = str(parsed.get("question") or "").strip()
    if not question_text:
        return None

    tag_set: Set[str] = set()
    for node in roots:
        tag_set.update(node.tags)
    tag_set.add("time")

    return QuestionNode(
        node_id=str(uuid.uuid4()),
        question_text=question_text,
        level="meso",
        status="open",
        origin="emergent",
        priority=0.7,
        tags=list(tag_set),
        question_type="change_continuity",
        confidence=0.5,
        generation=1,
        first_seen_doc="",
        first_seen_batch="defrag",
        embedding=None,
        is_leaf=False,
        merge_count=0,
        original_text=None,
        tension_score=0.0,
        evidence_owner=True,
    )


def defrag(
    graph: QuestionGraph,
    interval_index: int,
    docs_read: int,
    llm: LLMClient,
    doc_store: DocumentStore,
    config: Optional[Tier0Config] = None,
    batch_label: Optional[str] = None,
    budget: Optional[LLMBudget] = None,
    prompt_variant: Optional[str] = None,
) -> DefragSnapshot:
    cfg = config or APP_CONFIG.tier0
    label = batch_label or f"defrag-{interval_index}"
    budget = budget or LLMBudget(cap=cfg.llm_budget_per_interval, enforce=cfg.llm_budget_enforce)

    merges_performed = _dedup_pass(graph, interval_index=interval_index, batch_label=label)
    _rebuild_threads(graph, batch_label=label)

    promotions_performed = 0
    emergent_promoted = 0

    owners = [node for node in graph.nodes.values() if node.evidence_owner]
    for node in owners:
        if node.level == "micro":
            docs_for_node = {e.doc_id for e in graph.get_evidence(node.node_id)}
            if len(docs_for_node) < cfg.promote_micro_min_docs:
                continue
            if any(parent.level == "meso" for parent in graph.get_parents(node.node_id)):
                continue
            promoted = _promote_question(node, graph, "meso", llm, budget, cfg, prompt_variant=prompt_variant)
            if promoted is None:
                continue
            graph.add_node(promoted)
            log = RelationDecisionLog(
                log_id=str(uuid.uuid4()),
                candidate_text=promoted.question_text,
                matched_node_id=node.node_id,
                matched_node_text=node.question_text,
                similarity_score=None,
                threshold_used=0.0,
                action="promote",
                edge_type="generalizes",
                expand_call_result=None,
                seed_guard_fired=False,
                budget_skipped=False,
                batch_label=label,
                timestamp=datetime.now().isoformat(),
            )
            graph.add_decision_log(log)
            graph.add_edge(QuestionEdge(
                edge_id=str(uuid.uuid4()),
                source_id=promoted.node_id,
                target_id=node.node_id,
                relation="generalizes",
                confidence=0.8,
                created_at_batch=label,
                decision_log_id=log.log_id,
            ))
            promotions_performed += 1
            if node.origin == "emergent":
                emergent_promoted += 1

        elif node.level == "meso":
            children = [c for c in graph.get_children(node.node_id) if c.evidence_owner]
            children_with_evidence = [c for c in children if graph.get_evidence(c.node_id)]
            if len(children_with_evidence) < cfg.promote_meso_min_children:
                continue
            if any(parent.level == "macro" for parent in graph.get_parents(node.node_id)):
                continue
            promoted = _promote_question(node, graph, "macro", llm, budget, cfg, prompt_variant=prompt_variant)
            if promoted is None:
                continue
            graph.add_node(promoted)
            log = RelationDecisionLog(
                log_id=str(uuid.uuid4()),
                candidate_text=promoted.question_text,
                matched_node_id=node.node_id,
                matched_node_text=node.question_text,
                similarity_score=None,
                threshold_used=0.0,
                action="promote",
                edge_type="generalizes",
                expand_call_result=None,
                seed_guard_fired=False,
                budget_skipped=False,
                batch_label=label,
                timestamp=datetime.now().isoformat(),
            )
            graph.add_decision_log(log)
            graph.add_edge(QuestionEdge(
                edge_id=str(uuid.uuid4()),
                source_id=promoted.node_id,
                target_id=node.node_id,
                relation="generalizes",
                confidence=0.8,
                created_at_batch=label,
                decision_log_id=log.log_id,
            ))
            promotions_performed += 1
            if node.origin == "emergent":
                emergent_promoted += 1

    # Refresh threads after potential promotions.
    _rebuild_threads(graph, batch_label=label)

    for thread in list(graph.threads.values()):
        cc_node = _check_change_continuity(
            thread,
            graph,
            doc_store,
            llm,
            budget,
            cfg,
            prompt_variant=prompt_variant,
        )
        if cc_node is None:
            continue
        graph.add_node(cc_node)
        for root in thread.root_node_ids:
            if root not in graph.nodes:
                continue
            log = RelationDecisionLog(
                log_id=str(uuid.uuid4()),
                candidate_text=cc_node.question_text,
                matched_node_id=root,
                matched_node_text=graph.nodes[root].question_text,
                similarity_score=None,
                threshold_used=0.0,
                action="change_continuity",
                edge_type="generalizes",
                expand_call_result=None,
                seed_guard_fired=False,
                budget_skipped=False,
                batch_label=label,
                timestamp=datetime.now().isoformat(),
            )
            graph.add_decision_log(log)
            graph.add_edge(QuestionEdge(
                edge_id=str(uuid.uuid4()),
                source_id=cc_node.node_id,
                target_id=root,
                relation="generalizes",
                confidence=0.75,
                created_at_batch=label,
                decision_log_id=log.log_id,
            ))
        promotions_performed += 1
        emergent_promoted += 1

    _rebuild_threads(graph, batch_label=label)

    # Recompute thread active scores from root priorities.
    for thread in graph.threads.values():
        weights = [
            graph.attention_weight(graph.nodes[nid], current_interval=interval_index, docs_read=docs_read)
            for nid in thread.root_node_ids
            if nid in graph.nodes
        ]
        thread.active_score = max(weights) if weights else 0.0

    high_tension_threshold = float(cfg.tension_threshold)
    high_tension_nodes = [n.node_id for n in graph.get_high_tension_nodes(high_tension_threshold)]

    seed_owner_nodes = [n for n in graph.nodes.values() if n.origin == "seed" and n.evidence_owner]
    seed_with_evidence = sum(1 for n in seed_owner_nodes if graph.get_evidence(n.node_id))
    seed_without_evidence = max(0, len(seed_owner_nodes) - seed_with_evidence)

    axis_gaps = graph.get_axis_gaps(docs_read=docs_read)
    attention_priorities = [
        node.node_id
        for node in graph.get_active_hypotheses(
            max_count=cfg.ledger_max_active,
            current_interval=interval_index,
            docs_read=docs_read,
        )
    ]

    owner_count = max(1, len([n for n in graph.nodes.values() if n.evidence_owner]))
    over_merge_risk = float(merges_performed) / float(owner_count)

    recent = graph.defrag_snapshots[-2:] if graph.defrag_snapshots else []
    zero_streak = all(s.merges_performed == 0 for s in recent) and merges_performed == 0
    under_merge_risk = 1.0 if zero_streak else 0.0

    llm_used = budget.used
    llm_skipped = budget.skipped

    snapshot = DefragSnapshot(
        interval_index=interval_index,
        docs_read_at_defrag=docs_read,
        merges_performed=merges_performed,
        promotions_performed=promotions_performed,
        seed_questions_with_evidence=seed_with_evidence,
        seed_questions_unconfirmed=seed_without_evidence,
        emergent_questions_promoted=emergent_promoted,
        unresolved_threads=len(graph.threads),
        high_tension_nodes=high_tension_nodes,
        over_merge_risk=over_merge_risk,
        under_merge_risk=under_merge_risk,
        attention_priorities=attention_priorities,
        axis_gaps=axis_gaps,
        llm_calls_this_interval=llm_used,
        llm_calls_skipped_budget=llm_skipped,
    )
    graph.defrag_snapshots.append(snapshot)

    # Reset after snapshot capture so metrics remain accurate.
    budget.reset()
    return snapshot
