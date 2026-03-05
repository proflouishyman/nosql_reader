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


prompts_mod = _load("historian_agent.adaptive_prompts", "adaptive_prompts.py")


def test_normalize_prompt_variant_defaults_to_v1() -> None:
    assert prompts_mod.normalize_prompt_variant("unknown") == "v1"
    assert prompts_mod.normalize_prompt_variant("") == "v1"


def test_batch_prompt_variants_are_distinct() -> None:
    v1 = prompts_mod.get_batch_analysis_prompt("v1")
    v2 = prompts_mod.get_batch_analysis_prompt("v2")
    v3 = prompts_mod.get_batch_analysis_prompt("v3")
    v4 = prompts_mod.get_batch_analysis_prompt("v4")
    assert v1 != v2
    assert v2 != v3
    assert v3 != v4
    assert "question-building process" in v3
    assert "INSTRUCTIONS (follow in order)" in v4


def test_seed_prompt_variants_render_expected_placeholders() -> None:
    template = prompts_mod.get_seed_extraction_prompt("v2")
    rendered = template.format(text="lens", max_questions=3, axes="time, place")
    assert "2 to 3" in rendered
    assert "Allowed tags: time, place" in rendered


def test_v4_is_valid_prompt_variant() -> None:
    assert prompts_mod.normalize_prompt_variant("v4") == "v4"
