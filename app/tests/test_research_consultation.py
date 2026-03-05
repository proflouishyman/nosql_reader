from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


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


consultation_mod = _load("historian_agent.research_consultation", "research_consultation.py")

ResearchBrief = consultation_mod.ResearchBrief
build_reflection = consultation_mod.build_reflection
extract_seed_questions = consultation_mod.extract_seed_questions
normalize_brief = consultation_mod.normalize_brief
recommend_sort_order = consultation_mod.recommend_sort_order


def test_normalize_brief_maps_axes_and_sort_defaults() -> None:
    brief = normalize_brief({
        "primary_lens": "I want to compare compensation outcomes.",
        "axes": ["Time / periodization", "Ethnicity / national origin", "custom_axis"],
        "sort_order": "not-a-sort",
    })
    assert brief.primary_lens.startswith("I want")
    assert "time" in brief.axes
    assert "ethnicity" in brief.axes
    assert "custom_axis" in brief.axes
    assert brief.sort_order in {"archival", "chronological", "record_type", "person", "balanced"}


def test_research_brief_legacy_lens_includes_hypotheses_and_exclusions() -> None:
    brief = ResearchBrief(
        primary_lens="Injury patterns among railroad workers",
        prior_hypotheses="Irish workers faced worse assignments",
        exclusions="Management-level staff",
        axes=["time"],
        sort_order="archival",
    )
    legacy = brief.to_legacy_lens()
    assert "Injury patterns among railroad workers" in legacy
    assert "Prior hypotheses:" in legacy
    assert "Excluding:" in legacy


def test_extract_seed_questions_fallback_generates_non_what_questions() -> None:
    seeds = extract_seed_questions(
        text="I am studying how compensation changed for immigrant workers over time.",
        axes=["time", "ethnicity", "occupation"],
        llm=None,
        model=None,
        timeout_s=5,
        max_questions=3,
    )
    assert len(seeds) >= 1
    assert all(seed[1] != "what" for seed in seeds)


def test_reflection_recommends_chronological_for_change_lens() -> None:
    brief = normalize_brief({
        "primary_lens": "I want to track change over time in injury compensation.",
        "prior_hypotheses": "",
        "axes": ["time", "occupation"],
        "sort_order": "archival",
    })
    assert recommend_sort_order(brief) == "chronological"
    reflection = build_reflection(brief, llm=None)
    assert "Core focus:" in reflection["reflection"]
    assert reflection["recommended_sort_order"] == "chronological"
    assert isinstance(reflection["proposed_seeds"], list)
