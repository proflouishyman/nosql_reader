# app/historian_agent/question_graph.py
# Purpose: Adaptive question-graph notebook structures and graph operations.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Literal
import math
import re
import uuid

from config import APP_CONFIG


QUESTION_GRAPH_VERSION = "1.0"


@dataclass
class QuestionNode:
    node_id: str
    question_text: str
    level: Literal["micro", "meso", "macro"]
    status: Literal["open", "partial", "answered"]
    origin: Literal["seed", "emergent"]
    priority: float
    tags: List[str]
    question_type: Literal["why", "how", "compare", "change_continuity", "explain", "what"]
    confidence: float
    generation: int
    first_seen_doc: str
    first_seen_batch: str
    embedding: Optional[List[float]]
    is_leaf: bool
    merge_count: int
    original_text: Optional[str]
    tension_score: float
    evidence_owner: bool = True
    canonical_node_id: Optional[str] = None
    last_evidence_interval: int = 0


@dataclass
class EvidenceLink:
    link_id: str
    question_id: str
    doc_id: str
    block_id: str
    evidence_type: Literal["confirms", "complicates", "extends", "contradicts", "absence"]
    strength: float
    note: str
    batch_label: str

    @property
    def dedup_key(self) -> Tuple[str, str, str, str]:
        return (self.question_id, self.doc_id, self.block_id, self.evidence_type)


@dataclass
class QuestionEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: Literal["overlaps_with", "generalizes", "decomposes_into", "answered_by", "tensions_with"]
    confidence: float
    created_at_batch: str
    decision_log_id: str


@dataclass
class QuestionThread:
    thread_id: str
    root_node_ids: List[str]
    active_score: float
    last_updated_batch: str
    axis_coverage: Dict[str, int]
    has_seed_root: bool
    has_emergent_root: bool


@dataclass
class DefragSnapshot:
    interval_index: int
    docs_read_at_defrag: int
    merges_performed: int
    promotions_performed: int
    seed_questions_with_evidence: int
    seed_questions_unconfirmed: int
    emergent_questions_promoted: int
    unresolved_threads: int
    high_tension_nodes: List[str]
    over_merge_risk: float
    under_merge_risk: float
    attention_priorities: List[str]
    axis_gaps: List[str]
    llm_calls_this_interval: int
    llm_calls_skipped_budget: int


@dataclass
class RelationDecisionLog:
    log_id: str
    candidate_text: str
    matched_node_id: Optional[str]
    matched_node_text: Optional[str]
    similarity_score: Optional[float]
    threshold_used: float
    action: str
    edge_type: Optional[str]
    expand_call_result: Optional[str]
    seed_guard_fired: bool
    budget_skipped: bool
    batch_label: str
    timestamp: str


class QuestionGraph:
    def __init__(self, axes: List[str], schema_version: str = QUESTION_GRAPH_VERSION):
        self.schema_version = schema_version
        self.nodes: Dict[str, QuestionNode] = {}
        self.edges: List[QuestionEdge] = []
        self.threads: Dict[str, QuestionThread] = {}
        self.evidence: List[EvidenceLink] = []
        self.defrag_snapshots: List[DefragSnapshot] = []
        self.decision_log: List[RelationDecisionLog] = []
        self.axes: List[str] = list(axes)
        self._embedding_cache: Dict[str, List[float]] = {}

    def add_node(self, node: QuestionNode) -> str:
        if not node.node_id:
            node.node_id = str(uuid.uuid4())
        if node.canonical_node_id is None:
            node.canonical_node_id = node.node_id
        node.tags = list(dict.fromkeys([str(tag).strip() for tag in node.tags if str(tag).strip()]))
        self.nodes[node.node_id] = node
        return node.node_id

    def add_edge(self, edge: QuestionEdge) -> None:
        self.edges.append(edge)

    def add_decision_log(self, log: RelationDecisionLog) -> None:
        self.decision_log.append(log)

    def get_node(self, node_id: str) -> Optional[QuestionNode]:
        return self.nodes.get(node_id)

    def canonical_id(self, node_id: str) -> str:
        current = node_id
        seen = set()
        while current in self.nodes and current not in seen:
            seen.add(current)
            node = self.nodes[current]
            parent = node.canonical_node_id or current
            if parent == current:
                return current
            current = parent
        return node_id

    def get_children(self, node_id: str) -> List[QuestionNode]:
        return [
            self.nodes[e.target_id]
            for e in self.edges
            if e.source_id == node_id and e.target_id in self.nodes
        ]

    def get_parents(self, node_id: str) -> List[QuestionNode]:
        return [
            self.nodes[e.source_id]
            for e in self.edges
            if e.target_id == node_id and e.source_id in self.nodes
        ]

    def get_evidence(self, node_id: str) -> List[EvidenceLink]:
        canonical = self.canonical_id(node_id)
        return [e for e in self.evidence if e.question_id == canonical]

    def add_evidence(self, link: EvidenceLink, interval_index: int = 0) -> bool:
        node = self.nodes.get(link.question_id)
        if node is None:
            raise ValueError(f"Unknown node: {link.question_id}")
        if not node.evidence_owner:
            raise ValueError(
                f"Node {link.question_id} is non-canonical (absorbed). "
                "Add evidence to its canonical parent instead."
            )

        canonical_id = self.canonical_id(link.question_id)
        if canonical_id != link.question_id:
            raise ValueError(
                f"Evidence must target canonical node {canonical_id}, got {link.question_id}."
            )

        existing_idx = None
        for idx, existing in enumerate(self.evidence):
            if existing.dedup_key == link.dedup_key:
                existing_idx = idx
                break

        if existing_idx is not None:
            # Keep stronger evidence when dedup keys collide.
            if link.strength > self.evidence[existing_idx].strength:
                self.evidence[existing_idx] = link
                self.update_tension_score(link.question_id)
            return False

        self.evidence.append(link)
        node.last_evidence_interval = max(node.last_evidence_interval, int(interval_index))
        self.update_tension_score(link.question_id)
        return True

    def update_tension_score(self, node_id: str) -> None:
        canonical = self.canonical_id(node_id)
        node = self.nodes.get(canonical)
        if node is None:
            return

        evidence = self.get_evidence(canonical)
        confirms = sum(1 for e in evidence if e.evidence_type in ("confirms", "extends"))
        contradicts = sum(1 for e in evidence if e.evidence_type == "contradicts")
        smoothing = max(1, int(getattr(APP_CONFIG.tier0, "tension_smoothing", 3)))

        node.tension_score = float(contradicts) / float(confirms + contradicts + smoothing)

    def get_high_tension_nodes(self, threshold: float = 0.4) -> List[QuestionNode]:
        nodes = [
            n for n in self.nodes.values()
            if n.evidence_owner and n.tension_score > threshold
        ]
        nodes.sort(key=lambda n: n.tension_score, reverse=True)
        return nodes

    def attention_weight(self, node: QuestionNode, current_interval: int = 0, docs_read: int = 0) -> float:
        evidence = self.get_evidence(node.node_id)
        confirms = sum(1 for e in evidence if e.evidence_type in ("confirms", "extends"))
        absences = sum(1 for e in evidence if e.evidence_type == "absence")
        contradicts = sum(1 for e in evidence if e.evidence_type == "contradicts")
        batches_stale = max(0, current_interval - int(node.last_evidence_interval))

        seed_boost = 0.0
        if node.origin == "seed" and confirms == 0:
            decay_docs = int(getattr(APP_CONFIG.tier0, "seed_decay_after_docs", 20))
            if docs_read <= decay_docs:
                seed_boost = float(getattr(APP_CONFIG.tier0, "seed_priority_boost", 0.6))

        return (
            node.merge_count * 3.0
            + float(confirms)
            + float(absences) * 2.0
            + float(contradicts) * 3.0
            + (1.0 - self.saturation_score(node)) * 5.0
            - float(batches_stale) * 0.5
            + seed_boost
        )

    def saturation_score(self, node: QuestionNode) -> float:
        evidence = self.get_evidence(node.node_id)
        if not evidence:
            return 0.0

        type_set = set(e.evidence_type for e in evidence)
        diversity_penalty = 1.0 / float(max(1, len(type_set)))
        growth_component = min(1.0, float(len(evidence)) / 20.0)
        score = (growth_component * 0.6) + (diversity_penalty * 0.4)
        return max(0.0, min(1.0, score))

    def get_active_hypotheses(
        self,
        max_count: int = 15,
        current_interval: int = 0,
        docs_read: int = 0,
    ) -> List[QuestionNode]:
        scored = [
            (node, self.attention_weight(node, current_interval=current_interval, docs_read=docs_read))
            for node in self.nodes.values()
            if node.evidence_owner
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        if not scored:
            return []

        result = [node for node, _ in scored[:max_count]]
        has_seed = any(node.origin == "seed" for node in result)
        has_micro = any(node.level == "micro" for node in result)

        if not has_seed:
            top_seed = next((node for node, _ in scored if node.origin == "seed"), None)
            if top_seed is not None:
                if len(result) < max_count:
                    result.append(top_seed)
                else:
                    result[-1] = top_seed

        if not has_micro:
            top_micro = next((node for node, _ in scored if node.level == "micro"), None)
            if top_micro is not None and result:
                if len(result) < max_count:
                    result.append(top_micro)
                else:
                    result[-1] = top_micro

        deduped: Dict[str, QuestionNode] = {}
        for node in result:
            deduped[node.node_id] = node
        return list(deduped.values())[:max_count]

    def find_similar(self, text: str, threshold: float = 0.6) -> List[Tuple[QuestionNode, float]]:
        query = (text or "").strip()
        if not query:
            return []

        query_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", query.lower()))
        scores: List[Tuple[QuestionNode, float]] = []
        for node in self.nodes.values():
            if not node.evidence_owner:
                continue
            node_text = node.question_text or ""
            seq = SequenceMatcher(None, query.lower(), node_text.lower()).ratio()
            node_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", node_text.lower()))
            jaccard = 0.0
            union = len(query_tokens | node_tokens)
            if union:
                jaccard = float(len(query_tokens & node_tokens)) / float(union)
            score = max(seq, jaccard)
            if score >= threshold:
                scores.append((node, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def embed(self, text: str) -> List[float]:
        key = (text or "").strip().lower()
        if key in self._embedding_cache:
            return self._embedding_cache[key]

        tokens = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", key)
        vec = [0.0] * 64
        if not tokens:
            self._embedding_cache[key] = vec
            return vec

        for token in tokens:
            idx = hash(token) % 64
            vec[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]

        self._embedding_cache[key] = vec
        return vec

    def merge_into(
        self,
        canonical_id: str,
        absorbed_id: str,
        relation: str,
        batch_label: str,
        decision_log_id: str,
    ) -> None:
        if canonical_id == absorbed_id:
            return
        canonical = self.nodes[canonical_id]
        absorbed = self.nodes[absorbed_id]

        # Seeds always remain canonical when colliding with emergent nodes.
        if absorbed.origin == "seed" and canonical.origin == "emergent":
            canonical_id, absorbed_id = absorbed_id, canonical_id
            canonical = self.nodes[canonical_id]
            absorbed = self.nodes[absorbed_id]

        canonical_map: Dict[Tuple[str, str, str, str], int] = {
            link.dedup_key: idx for idx, link in enumerate(self.evidence)
            if link.question_id == canonical_id
        }

        absorbed_links = [link for link in self.evidence if link.question_id == absorbed_id]
        for link in absorbed_links:
            transferred = EvidenceLink(
                link_id=link.link_id,
                question_id=canonical_id,
                doc_id=link.doc_id,
                block_id=link.block_id,
                evidence_type=link.evidence_type,
                strength=link.strength,
                note=link.note,
                batch_label=link.batch_label,
            )
            existing_idx = canonical_map.get(transferred.dedup_key)
            if existing_idx is None:
                self.evidence.append(transferred)
                canonical_map[transferred.dedup_key] = len(self.evidence) - 1
            else:
                if transferred.strength > self.evidence[existing_idx].strength:
                    self.evidence[existing_idx] = transferred

        absorbed.evidence_owner = False
        absorbed.canonical_node_id = canonical_id

        edge_relation = relation if relation in {
            "overlaps_with", "decomposes_into", "generalizes", "answered_by", "tensions_with"
        } else "overlaps_with"

        self.add_edge(QuestionEdge(
            edge_id=str(uuid.uuid4()),
            source_id=canonical_id,
            target_id=absorbed_id,
            relation=edge_relation,
            confidence=1.0,
            created_at_batch=batch_label,
            decision_log_id=decision_log_id,
        ))

        canonical.merge_count += 1
        self.update_tension_score(canonical_id)

    def get_axis_gaps(self, docs_read: int, gap_threshold: float = 0.20) -> List[str]:
        if docs_read < 20:
            return []

        total_coverage: Dict[str, int] = {axis: 0 for axis in self.axes}
        for thread in self.threads.values():
            for axis, count in thread.axis_coverage.items():
                if axis in total_coverage:
                    total_coverage[axis] += int(count)

        if not total_coverage:
            return []

        max_coverage = max(total_coverage.values()) or 1
        return [
            axis for axis, count in total_coverage.items()
            if (float(count) / float(max_coverage)) < gap_threshold
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "axes": list(self.axes),
            "nodes": {key: asdict(node) for key, node in self.nodes.items()},
            "edges": [asdict(edge) for edge in self.edges],
            "threads": {key: asdict(thread) for key, thread in self.threads.items()},
            "evidence": [asdict(link) for link in self.evidence],
            "defrag_snapshots": [asdict(snapshot) for snapshot in self.defrag_snapshots],
            "decision_log": [asdict(item) for item in self.decision_log],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionGraph":
        version = str(data.get("schema_version", "0.9"))
        payload = data
        if version != QUESTION_GRAPH_VERSION:
            payload = _migrate(dict(data), from_version=version)

        graph = cls(
            axes=payload.get("axes", []),
            schema_version=str(payload.get("schema_version", QUESTION_GRAPH_VERSION)),
        )

        for key, raw in (payload.get("nodes") or {}).items():
            raw = dict(raw or {})
            raw.setdefault("node_id", key)
            raw.setdefault("canonical_node_id", raw.get("node_id"))
            raw.setdefault("evidence_owner", True)
            raw.setdefault("last_evidence_interval", 0)
            graph.nodes[str(key)] = QuestionNode(**raw)

        for raw in (payload.get("edges") or []):
            try:
                graph.edges.append(QuestionEdge(**raw))
            except Exception:
                continue

        for key, raw in (payload.get("threads") or {}).items():
            try:
                graph.threads[str(key)] = QuestionThread(**raw)
            except Exception:
                continue

        for raw in (payload.get("evidence") or []):
            try:
                graph.evidence.append(EvidenceLink(**raw))
            except Exception:
                continue

        for raw in (payload.get("defrag_snapshots") or []):
            try:
                graph.defrag_snapshots.append(DefragSnapshot(**raw))
            except Exception:
                continue

        for raw in (payload.get("decision_log") or []):
            try:
                graph.decision_log.append(RelationDecisionLog(**raw))
            except Exception:
                continue

        return graph


def _migrate(data: Dict[str, Any], from_version: str) -> Dict[str, Any]:
    nodes = data.get("nodes") or {}
    if from_version == "0.9":
        for key, node in nodes.items():
            if not isinstance(node, dict):
                continue
            node.setdefault("node_id", key)
            node.setdefault("origin", "emergent")
            node.setdefault("tension_score", 0.0)
            node.setdefault("original_text", None)
            node.setdefault("question_type", "why")
            node.setdefault("evidence_owner", True)
            node.setdefault("canonical_node_id", node.get("node_id") or key)
            node.setdefault("last_evidence_interval", 0)
        data["schema_version"] = QUESTION_GRAPH_VERSION
        data.setdefault("decision_log", [])
        data.setdefault("defrag_snapshots", [])
        data.setdefault("threads", {})
        data.setdefault("edges", [])
        data.setdefault("evidence", [])
        data.setdefault("axes", list(getattr(APP_CONFIG.tier0, "question_axes", [])))
        return data

    # Unknown previous versions: fill missing fields conservatively.
    for key, node in nodes.items():
        if not isinstance(node, dict):
            continue
        node.setdefault("node_id", key)
        node.setdefault("origin", "emergent")
        node.setdefault("question_type", "why")
        node.setdefault("tension_score", 0.0)
        node.setdefault("evidence_owner", True)
        node.setdefault("canonical_node_id", node.get("node_id") or key)
        node.setdefault("last_evidence_interval", 0)

    data["schema_version"] = QUESTION_GRAPH_VERSION
    data.setdefault("decision_log", [])
    data.setdefault("defrag_snapshots", [])
    data.setdefault("threads", {})
    data.setdefault("edges", [])
    data.setdefault("evidence", [])
    data.setdefault("axes", list(getattr(APP_CONFIG.tier0, "question_axes", [])))
    return data
