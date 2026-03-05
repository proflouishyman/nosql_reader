#!/usr/bin/env python3
"""Run adaptive prompt A/B/C benchmarks against /api/rag/explore_corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import re


DEFAULT_BRIEF: Dict[str, Any] = {
    "primary_lens": (
        "I want to understand how railroad injury experience varied across worker groups "
        "and how compensation outcomes changed over time."
    ),
    "prior_hypotheses": (
        "Some occupations and ethnic groups were routed into higher-risk work, and "
        "compensation outcomes were uneven."
    ),
    "axes": [
        "time",
        "place",
        "occupation",
        "ethnicity",
        "unionization",
        "management",
        "legislation",
    ],
    "surprise_expectations": (
        "It would be surprising if injury burden and compensation outcomes were "
        "uniform across occupations and groups."
    ),
    "exclusions": "",
    "sort_order": "archival",
    "confirmed": True,
}


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object response from API.")
    return parsed


def _capture_ollama_models() -> List[str]:
    """Capture locally available Ollama models for benchmark provenance."""
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    # Skip header row when present.
    if lines and lines[0].lower().startswith("name"):
        lines = lines[1:]
    models: List[str] = []
    for line in lines:
        model = line.split()[0].strip()
        if model:
            models.append(model)
    return models


def _variant_payload(
    args: argparse.Namespace,
    variant: str,
    brief: Dict[str, Any],
    ledger_model: str,
) -> Dict[str, Any]:
    """Build a stable benchmark request for one prompt variant."""
    payload: Dict[str, Any] = {
        "mode": "adaptive",
        "strategy": args.strategy,
        "total_budget": args.documents,
        "save_notebook": False,
        "sort_order": args.sort_order,
        "prompt_variant": variant,
        "research_brief": {
            **brief,
            "sort_order": args.sort_order,
            "confirmed": True,
        },
    }
    if ledger_model:
        # Keep model override explicit per run so matrix experiments are reproducible.
        payload["ledger_model"] = ledger_model
    return payload


def _summarize_variant(
    variant: str,
    requested_model: str,
    report: Dict[str, Any],
    elapsed_s: float,
) -> Dict[str, Any]:
    metadata = report.get("exploration_metadata") if isinstance(report.get("exploration_metadata"), dict) else {}
    graph = report.get("question_graph") if isinstance(report.get("question_graph"), dict) else {}
    by_level = graph.get("by_level") if isinstance(graph.get("by_level"), dict) else {}
    by_origin = graph.get("by_origin") if isinstance(graph.get("by_origin"), dict) else {}
    defrag = graph.get("defrag_snapshots") if isinstance(graph.get("defrag_snapshots"), list) else []

    promotions = 0
    merges = 0
    budget_skips = 0
    llm_calls = 0
    for snap in defrag:
        if not isinstance(snap, dict):
            continue
        promotions += int(snap.get("promotions_performed") or 0)
        merges += int(snap.get("merges_performed") or 0)
        budget_skips += int(snap.get("llm_calls_skipped_budget") or 0)
        llm_calls += int(snap.get("llm_calls_this_interval") or 0)

    top_questions = report.get("questions") if isinstance(report.get("questions"), list) else []
    quality = _quality_metrics(top_questions)

    return {
        "prompt_variant": variant,
        "requested_model": requested_model or "(default)",
        "effective_model": str(metadata.get("ledger_model") or requested_model or "(default)"),
        "duration_seconds": round(elapsed_s, 2),
        "documents_read": int(metadata.get("documents_read") or 0),
        "batches_processed": int(metadata.get("batches_processed") or 0),
        "questions_top_count": len(report.get("questions") or []),
        "graph_total_nodes": int(graph.get("total_nodes") or 0),
        "graph_macro_nodes": int(by_level.get("macro") or 0),
        "graph_meso_nodes": int(by_level.get("meso") or 0),
        "graph_micro_nodes": int(by_level.get("micro") or 0),
        "graph_seed_nodes": int(by_origin.get("seed") or 0),
        "graph_emergent_nodes": int(by_origin.get("emergent") or 0),
        "graph_edges": int(graph.get("total_edges") or 0),
        "decision_log_count": int(graph.get("decision_log_count") or 0),
        "seed_questions_confirmed": int(graph.get("seed_questions_confirmed") or 0),
        "seed_questions_unconfirmed": int(graph.get("seed_questions_unconfirmed") or 0),
        "change_continuity_questions": int(graph.get("change_continuity_questions") or 0),
        "high_tension_count": len(graph.get("high_tension_nodes") or []),
        "defrag_merges_total": merges,
        "defrag_promotions_total": promotions,
        "llm_calls_total": llm_calls,
        "llm_calls_skipped_budget_total": budget_skips,
        "quality_analytic_ratio": quality["analytic_ratio"],
        "quality_factoid_count": quality["factoid_count"],
        "quality_vague_count": quality["vague_count"],
        "quality_questions_evaluated": quality["question_count"],
    }


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "status",
        "model",
        "variant",
        "secs",
        "docs",
        "nodes",
        "macro",
        "meso",
        "micro",
        "edges",
        "logs",
        "seed_ok",
        "promotions",
        "analytic",
        "factoid",
        "vague",
    ]
    print(" | ".join(headers))
    print("-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-")
    for row in rows:
        print(
            f"{row.get('status', 'ok')} | "
            f"{row.get('effective_model', row.get('requested_model', '(default)'))} | "
            f"{row.get('prompt_variant', '')} | "
            f"{row.get('duration_seconds', 0)} | "
            f"{row.get('documents_read', 0)} | "
            f"{row.get('graph_total_nodes', 0)} | "
            f"{row.get('graph_macro_nodes', 0)} | "
            f"{row.get('graph_meso_nodes', 0)} | "
            f"{row.get('graph_micro_nodes', 0)} | "
            f"{row.get('graph_edges', 0)} | "
            f"{row.get('decision_log_count', 0)} | "
            f"{row.get('seed_questions_confirmed', 0)} | "
            f"{row.get('defrag_promotions_total', 0)} | "
            f"{row.get('quality_analytic_ratio', 0)} | "
            f"{row.get('quality_factoid_count', 0)} | "
            f"{row.get('quality_vague_count', 0)}"
        )


def _quality_metrics(questions: List[Any]) -> Dict[str, Any]:
    """
    Lightweight historian-quality proxy from top-level synthesized questions.
    Used for A/B comparisons, not as final truth.
    """
    if not isinstance(questions, list):
        return {"question_count": 0, "analytic_ratio": 0.0, "factoid_count": 0, "vague_count": 0}

    factoid_prefix = re.compile(r"^(when|who|what year|how many|where)\b", re.IGNORECASE)
    vague_markers = (
        "what impact",
        "what role",
        "what was life like",
        "how important",
        "what challenges",
    )
    mechanism_markers = (
        "why",
        "how",
        "change",
        "across",
        "vary",
        "institution",
        "system",
        "power",
        "hierarch",
        "incentive",
    )

    total = 0
    factoid = 0
    vague = 0
    analytic = 0

    for item in questions:
        text = ""
        if isinstance(item, dict):
            text = str(item.get("question") or "").strip()
        elif isinstance(item, str):
            text = item.strip()
        if not text:
            continue
        total += 1
        lowered = text.lower()

        if factoid_prefix.match(lowered):
            factoid += 1
        if any(marker in lowered for marker in vague_markers):
            vague += 1
        if any(marker in lowered for marker in mechanism_markers):
            analytic += 1

    ratio = round((analytic / total), 3) if total else 0.0
    return {
        "question_count": total,
        "analytic_ratio": ratio,
        "factoid_count": factoid,
        "vague_count": vague,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:5001")
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--strategy", default="balanced")
    parser.add_argument("--sort-order", default="archival")
    parser.add_argument("--variants", default="v4,v5,v6")
    parser.add_argument("--ledger-model", default="")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model list or 'auto' to pick from ollama list.",
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--out-dir", default="logs/prompt_variant_benchmarks")
    parser.add_argument("--brief-json", default="")
    args = parser.parse_args()

    variants = [item.strip().lower() for item in args.variants.split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one prompt variant is required.")

    brief = DEFAULT_BRIEF
    if args.brief_json:
        parsed = json.loads(args.brief_json)
        if not isinstance(parsed, dict):
            raise ValueError("--brief-json must decode to an object.")
        brief = parsed

    results: List[Dict[str, Any]] = []
    endpoint = f"{args.base_url.rstrip('/')}/api/rag/explore_corpus"
    models = _capture_ollama_models()

    # Build model test set: explicit list wins, then single override, else default server config.
    model_candidates: List[str] = []
    if args.models.strip():
        if args.models.strip().lower() == "auto":
            # Filter out non-text and remote/cloud entries so A/B/C stays comparable.
            filtered_models = []
            for name in models:
                lowered = name.lower()
                if any(token in lowered for token in ("vl", "vision", "embedding", "ocr", "cloud")):
                    continue
                filtered_models.append(name)
            preferred_families = ("qwen", "llama", "gemma", "mistral")
            for family in preferred_families:
                match = next((m for m in filtered_models if m.lower().startswith(family)), None)
                if match and match not in model_candidates:
                    model_candidates.append(match)
            if not model_candidates:
                model_candidates = filtered_models[:3] if filtered_models else models[:3]
        else:
            model_candidates = [item.strip() for item in args.models.split(",") if item.strip()]
    elif args.ledger_model.strip():
        model_candidates = [args.ledger_model.strip()]
    else:
        model_candidates = [""]

    for model_name in model_candidates:
        for variant in variants:
            payload = _variant_payload(args, variant, brief, ledger_model=model_name)
            start = time.perf_counter()
            print(
                f"[benchmark] running model={model_name or '(default)'} variant={variant} docs={args.documents}",
                flush=True,
            )
            try:
                report = _post_json(endpoint, payload, timeout_s=args.timeout)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                elapsed = time.perf_counter() - start
                print(
                    f"[benchmark] error model={model_name or '(default)'} variant={variant}: HTTP {exc.code}",
                    flush=True,
                )
                results.append({
                    "status": "error",
                    "prompt_variant": variant,
                    "requested_model": model_name or "(default)",
                    "effective_model": model_name or "(default)",
                    "duration_seconds": round(elapsed, 2),
                    "error": f"HTTP {exc.code}: {detail[:240]}",
                })
                continue
            except Exception as exc:
                elapsed = time.perf_counter() - start
                print(
                    f"[benchmark] error model={model_name or '(default)'} variant={variant}: {exc}",
                    flush=True,
                )
                results.append({
                    "status": "error",
                    "prompt_variant": variant,
                    "requested_model": model_name or "(default)",
                    "effective_model": model_name or "(default)",
                    "duration_seconds": round(elapsed, 2),
                    "error": str(exc),
                })
                continue
            elapsed = time.perf_counter() - start
            summary = _summarize_variant(variant, model_name, report, elapsed)
            summary["status"] = "ok"
            results.append(summary)

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "endpoint": endpoint,
        "documents": args.documents,
        "strategy": args.strategy,
        "sort_order": args.sort_order,
        "variants": variants,
        "models_tested": [name or "(default)" for name in model_candidates],
        "ollama_models": models,
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"adaptive_prompt_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    _print_table(results)
    print(f"\nSaved benchmark report: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI top-level fallback
        print(f"[benchmark] failed: {exc}", file=sys.stderr)
        raise
