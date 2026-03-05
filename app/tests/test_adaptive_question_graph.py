from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


_BASE = Path(__file__).resolve().parents[1] / "historian_agent"
_PKG = types.ModuleType("historian_agent")
_PKG.__path__ = [str(_BASE)]
sys.modules.setdefault("historian_agent", _PKG)


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _BASE / file_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


question_graph_mod = _load("historian_agent.question_graph", "question_graph.py")
relation_engine_mod = _load("historian_agent.relation_engine", "relation_engine.py")

EvidenceLink = question_graph_mod.EvidenceLink
QuestionGraph = question_graph_mod.QuestionGraph
QuestionNode = question_graph_mod.QuestionNode
decide_relation_with_seed_guard = relation_engine_mod.decide_relation_with_seed_guard


class DummyBudget:
    def __init__(self):
        self.used = 0

    def request(self, tier: str) -> bool:
        self.used += 1
        return True


class DummyLLM:
    class _Resp:
        def __init__(self, content: str):
            self.success = True
            self.content = content

    def generate(self, **kwargs):
        return DummyLLM._Resp("LATERAL")


def _node(node_id: str, text: str, origin: str = "emergent", level: str = "micro") -> QuestionNode:
    return QuestionNode(
        node_id=node_id,
        question_text=text,
        level=level,  # type: ignore[arg-type]
        status="open",
        origin=origin,  # type: ignore[arg-type]
        priority=0.5,
        tags=[],
        question_type="why",
        confidence=0.5,
        generation=0,
        first_seen_doc="",
        first_seen_batch="test",
        embedding=None,
        is_leaf=False,
        merge_count=0,
        original_text=None,
        tension_score=0.0,
        evidence_owner=True,
    )


def _evidence(question_id: str, strength: float, ev_type: str = "extends") -> EvidenceLink:
    return EvidenceLink(
        link_id=f"{question_id}-{strength}",
        question_id=question_id,
        doc_id="doc-1",
        block_id="doc-1::b0",
        evidence_type=ev_type,  # type: ignore[arg-type]
        strength=strength,
        note="note",
        batch_label="b1",
    )


def test_merge_into_transfers_evidence_and_keeps_stronger_duplicate() -> None:
    graph = QuestionGraph(axes=["time"])
    canonical = _node("seed", "Seed question", origin="seed", level="macro")
    absorbed = _node("emergent", "Emergent question", origin="emergent", level="micro")
    graph.add_node(canonical)
    graph.add_node(absorbed)

    graph.add_evidence(_evidence("seed", strength=0.4))
    graph.add_evidence(_evidence("emergent", strength=0.9))

    graph.merge_into(
        canonical_id="seed",
        absorbed_id="emergent",
        relation="overlaps_with",
        batch_label="b1",
        decision_log_id="log-1",
    )

    links = graph.get_evidence("seed")
    assert len(links) == 1
    assert links[0].strength == pytest.approx(0.9)
    assert graph.nodes["emergent"].evidence_owner is False
    assert graph.nodes["emergent"].canonical_node_id == "seed"


def test_add_evidence_rejects_non_canonical_absorbed_node() -> None:
    graph = QuestionGraph(axes=["time"])
    canonical = _node("seed", "Seed question", origin="seed", level="macro")
    absorbed = _node("emergent", "Emergent question", origin="emergent", level="micro")
    graph.add_node(canonical)
    graph.add_node(absorbed)
    graph.merge_into(
        canonical_id="seed",
        absorbed_id="emergent",
        relation="overlaps_with",
        batch_label="b1",
        decision_log_id="log-1",
    )

    with pytest.raises(ValueError):
        graph.add_evidence(_evidence("emergent", strength=0.5))


def test_tension_smoothing_prevents_early_spike() -> None:
    graph = QuestionGraph(axes=["time"])
    node = _node("n1", "Why did claims diverge?")
    graph.add_node(node)
    graph.add_evidence(_evidence("n1", strength=0.8, ev_type="contradicts"))

    # With smoothing=3 default: 1 / (0 + 1 + 3) = 0.25
    assert graph.nodes["n1"].tension_score == pytest.approx(0.25, rel=1e-6)


def test_from_dict_migrates_legacy_v09_payload() -> None:
    legacy = {
        "schema_version": "0.9",
        "axes": ["time"],
        "nodes": {
            "n1": {
                "question_text": "Why were claims delayed?",
                "level": "micro",
                "status": "open",
                "priority": 0.5,
                "tags": [],
                "confidence": 0.5,
                "generation": 0,
                "first_seen_doc": "",
                "first_seen_batch": "seed",
                "embedding": None,
                "is_leaf": False,
                "merge_count": 0,
            }
        },
        "edges": [],
        "threads": {},
        "evidence": [],
    }
    graph = QuestionGraph.from_dict(legacy)
    assert graph.schema_version == "1.0"
    assert graph.nodes["n1"].origin == "emergent"
    assert graph.nodes["n1"].question_type == "why"
    assert graph.nodes["n1"].tension_score == pytest.approx(0.0)


def test_seed_guard_requires_material_similarity_margin() -> None:
    budget = DummyBudget()
    llm = DummyLLM()
    candidate = _node("cand", "How did brakemen ankle injuries vary?", origin="emergent", level="micro")

    seed = _node("seed", "How did workers experience occupational injury?", origin="seed", level="macro")
    non_seed = _node("other", "How did brakemen injuries differ by division?", origin="emergent", level="meso")

    # Seed not materially better than non-seed -> guard should not fire.
    decision_a, log_a = decide_relation_with_seed_guard(
        candidate_node=candidate,
        similar_nodes=[(seed, 0.72), (non_seed, 0.70)],
        llm=llm,
        budget=budget,
        batch_label="b1",
    )
    assert log_a.seed_guard_fired is False
    assert decision_a.action in {"new_thread", "lateral", "generalize", "decompose", "merge"}

    budget_b = DummyBudget()
    # Seed materially better by margin >= 0.08 -> guard should fire.
    decision_b, log_b = decide_relation_with_seed_guard(
        candidate_node=candidate,
        similar_nodes=[(seed, 0.82), (non_seed, 0.70)],
        llm=llm,
        budget=budget_b,
        batch_label="b1",
    )
    assert log_b.seed_guard_fired is True
    assert decision_b.action == "decompose"
    # Guard path should not consume budget in logging side effects.
    assert budget_b.used == 0
