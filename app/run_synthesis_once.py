#!/usr/bin/env python3
"""
Run question generation + synthesis for a saved notebook with current env config.

Env:
  NOTEBOOK_PATH: path to notebook JSON (required)
  OUT_PATH: output JSON path (required)
  REGEN_QUESTIONS: 1 to regenerate questions, 0 to load from QUESTIONS_PATH
  QUESTIONS_PATH: optional path to precomputed questions list (list[dict])
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime

from config import APP_CONFIG
from historian_agent.research_notebook import ResearchNotebook
from historian_agent.question_pipeline import QuestionGenerationPipeline
from historian_agent.question_synthesis import QuestionSynthesizer
from historian_agent.question_models import Question


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def main() -> None:
    notebook_path = Path(_require_env("NOTEBOOK_PATH"))
    out_path = Path(_require_env("OUT_PATH"))
    regen_questions = os.environ.get("REGEN_QUESTIONS", "1") == "1"
    questions_path = os.environ.get("QUESTIONS_PATH")

    notebook = ResearchNotebook.load(notebook_path)

    if regen_questions:
        pipeline = QuestionGenerationPipeline()
        batch = pipeline.generate(notebook)
        questions = batch.questions
        question_dicts = [q.to_dict() for q in questions]
        generation_meta = {
            "total_candidates": batch.total_candidates,
            "total_validated": batch.total_validated,
            "total_accepted": batch.total_accepted,
            "generation_strategy": batch.generation_strategy,
        }
    else:
        if not questions_path:
            raise SystemExit("QUESTIONS_PATH required when REGEN_QUESTIONS=0")
        question_dicts = json.loads(Path(questions_path).read_text())
        questions = [Question.from_dict(q) for q in question_dicts]
        generation_meta = {"generation_strategy": "precomputed"}

    synthesizer = QuestionSynthesizer()
    agenda = synthesizer.build_agenda(notebook, questions)

    payload = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "llm_model": APP_CONFIG.llm_profiles.get("quality", {}).get("model"),
            "verifier_model": APP_CONFIG.llm_profiles.get("verifier", {}).get("model"),
            "question_target_count": APP_CONFIG.tier0.question_target_count,
            "question_per_type": APP_CONFIG.tier0.question_per_type,
            "question_min_score": APP_CONFIG.tier0.question_min_score,
            "synthesis_max_question_sample": APP_CONFIG.tier0.synthesis_max_question_sample,
            "synthesis_max_pattern_sample": APP_CONFIG.tier0.synthesis_max_pattern_sample,
            "synthesis_theme_count": APP_CONFIG.tier0.synthesis_theme_count,
            "synthesis_min_themes": APP_CONFIG.tier0.synthesis_min_themes,
            "synthesis_dynamic": APP_CONFIG.tier0.synthesis_dynamic,
            "synthesis_semantic_assignment": APP_CONFIG.tier0.synthesis_semantic_assignment,
            "synthesis_narrative_enabled": APP_CONFIG.tier0.synthesis_narrative_enabled,
        },
        "question_generation": generation_meta,
        "questions": question_dicts,
        "question_synthesis": agenda,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"saved {out_path}")
    print(f"questions {len(question_dicts)}")
    print(f"themes {len(agenda.get('themes', []))}")
    print(f"group_comparisons {len(agenda.get('group_difference_questions', []))}")
    print(f"contradictions {len(agenda.get('contradiction_questions', []))}")


if __name__ == "__main__":
    main()
