#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

NOTEBOOK = "/app/logs/corpus_exploration/exploration_20260209_012130/20260209_012130_notebook.json"
OUTPUT_DIR = Path("/Users/louishyman/coding/nosql/nosql_reader/app/logs/synthesis_matrix_20260209_high")
INCREMENTS = [48, 60, 72, 84, 96]

def run_step(target: int) -> Path:
    out_path = Path(f"/app/logs/synthesis_matrix_20260209_high/synth_llama3.1_8b_q{target}_s{target}_p{target}.json")
    if (OUTPUT_DIR / out_path.name).exists():
        return out_path

    cmd = [
        "docker", "compose",
        "-f", "/Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml",
        "exec", "-T", "app", "sh", "-lc",
        " ".join([
            "PYTHONPATH=/app:/app/historian_agent",
            f"NOTEBOOK_PATH={NOTEBOOK}",
            f"OUT_PATH={out_path}",
            "REGEN_QUESTIONS=1",
            "LLM_MODEL=llama3.1:8b",
            "LLM_FAST_MODEL=llama3.1:8b",
            "VERIFIER_MODEL=llama3.1:8b",
            f"TIER0_QUESTION_TARGET_COUNT={target}",
            "TIER0_QUESTION_PER_TYPE=10",
            "TIER0_QUESTION_MIN_SCORE=60",
            f"TIER0_SYNTHESIS_MAX_QUESTION_SAMPLE={target}",
            f"TIER0_SYNTHESIS_MAX_PATTERN_SAMPLE={target}",
            "TIER0_SYNTHESIS_SEMANTIC_ASSIGNMENT=0",
            "TIER0_LLM_TIMEOUT=600",
            "python /app/run_synthesis_once.py",
        ])
    ]
    subprocess.run(cmd, check=True)
    return out_path

def metrics(path: Path) -> dict:
    data = json.loads((OUTPUT_DIR / path.name).read_text())
    synth = data.get("question_synthesis", {})
    narrative = synth.get("narrative", {}) or {}
    return {
        "questions": len(data.get("questions", [])),
        "themes": len(synth.get("themes", [])),
        "narrative_words": len((narrative.get("narrative") or "").split()),
    }

def improved(prev: dict, cur: dict) -> bool:
    if not prev:
        return True
    if cur["themes"] > prev["themes"]:
        return True
    if cur["questions"] > prev["questions"]:
        return True
    if cur["narrative_words"] > prev["narrative_words"] * 1.05:
        return True
    return False

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prev = {}
    for target in INCREMENTS:
        out_path = OUTPUT_DIR / f"synth_llama3.1_8b_q{target}_s{target}_p{target}.json"
        if not out_path.exists():
            print(f"running q{target}")
            run_step(target)
        else:
            print(f"exists q{target}")
        cur = metrics(out_path)
        print(f"q{target} metrics: {cur}")
        if prev and not improved(prev, cur):
            print(f"stopping at q{target} (no improvement)")
            break
        prev = cur
        time.sleep(2)

if __name__ == "__main__":
    main()
