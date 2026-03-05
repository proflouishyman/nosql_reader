# app/historian_agent/incremental_explorer.py
# Purpose: Adaptive attentive corpus exploration with incremental question graph updates.

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import APP_CONFIG
from llm_abstraction import LLMClient

from historian_agent.adaptive_prompts import (
    get_batch_analysis_prompt,
    get_batch_system_message,
    resolve_prompt_variant,
)
from historian_agent.corpus_explorer import CorpusExplorer
from historian_agent.defrag_engine import LLMBudget, defrag
from historian_agent.question_graph import (
    EvidenceLink,
    QuestionEdge,
    QuestionGraph,
    QuestionNode,
)
from historian_agent.question_models import Question, QuestionType
from historian_agent.research_consultation import (
    ResearchBrief,
    extract_seed_questions,
    normalize_brief,
)
from historian_agent.relation_engine import decide_relation_with_seed_guard, enforce_why_how
from historian_agent.tier0_utils import save_with_timestamp


class IncrementalCorpusExplorer(CorpusExplorer):
    """Adaptive Tier 0 explorer that accumulates a question graph while reading."""

    def __init__(self) -> None:
        super().__init__()
        self.graph: Optional[QuestionGraph] = None
        self._docs_read_for_graph = 0
        self._sub_batches_seen = 0
        self._defrag_index = 0
        self._interval_budget = LLMBudget(
            cap=APP_CONFIG.tier0.llm_budget_per_interval,
            enforce=APP_CONFIG.tier0.llm_budget_enforce,
        )
        # Prompt variant is run-scoped so A/B/C tests can override without changing contracts.
        self._prompt_variant: str = resolve_prompt_variant(None)
        self._seed_node_ids: List[str] = []
        self._research_brief: Optional[ResearchBrief] = None

    def explore(
        self,
        strategy: Optional[str] = None,
        total_budget: Optional[int] = None,
        year_range: Optional[Tuple[int, int]] = None,
        save_notebook: Optional[bool] = None,
        research_lens: Optional[Any] = None,
        sort_order: Optional[str] = None,
        research_brief: Optional[Any] = None,
        prompt_variant: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = datetime.now()

        cfg = APP_CONFIG.tier0
        strategy = strategy or cfg.exploration_strategy
        total_budget = total_budget or cfg.exploration_budget
        save_notebook = cfg.notebook_auto_save if save_notebook is None else save_notebook
        self._prompt_variant = resolve_prompt_variant(prompt_variant)
        # Build a stable consultation object no matter whether caller sends raw lens text or full brief.
        brief_payload = research_brief if research_brief is not None else {
            "primary_lens": research_lens or "",
            "axes": list(getattr(cfg, "question_axes", [])),
            "sort_order": sort_order or getattr(cfg, "adaptive_default_sort", "archival"),
            "confirmed": bool(research_brief),
        }
        self._research_brief = normalize_brief(brief_payload)
        if sort_order:
            self._research_brief = normalize_brief({
                **self._research_brief.to_dict(),
                "sort_order": sort_order,
            })
        self._research_lens = self._normalize_research_lens(self._research_brief.to_legacy_lens())

        self.notebook = type(self.notebook)()
        self.graph = self._initialize_graph(self._research_brief)
        self._docs_read_for_graph = 0
        self._sub_batches_seen = 0
        self._defrag_index = 0
        self._interval_budget.reset()

        self.logger.log(
            "start",
            f"adaptive strategy={strategy} budget={total_budget} sort={self._research_brief.sort_order} prompt={self._prompt_variant}",
        )
        if self._research_lens:
            self.logger.log("lens", "; ".join(self._research_lens))

        strata = self._build_strata(
            strategy=strategy,
            total_budget=total_budget,
            year_range=year_range,
            sort_order=self._research_brief.sort_order,
        )
        missing_sort_fields = sum(getattr(stratum, "missing_sort_fields", 0) for stratum in strata)
        if missing_sort_fields:
            self.logger.log(
                "sort_missing",
                f"{missing_sort_fields} docs missing primary field for sort={self._research_brief.sort_order}",
                level="WARN",
            )
        self.logger.log("stratification", f"{len(strata)} batches")

        for idx, stratum in enumerate(strata):
            self.logger.log("batch", f"{idx + 1}/{len(strata)} {stratum.label}")
            try:
                self._process_stratum_adaptive(stratum)
            except Exception as exc:
                self.logger.log("batch_error", f"{stratum.label}: {exc}", level="WARN")

        # Final defrag for trailing interval state.
        self._run_defrag(batch_label="final")

        # Keep notebook serialization compatible.
        self.notebook.graph = self.graph

        corpus_map = self._generate_corpus_map()
        questions = self._export_graph_questions()
        question_synthesis = self._generate_question_synthesis_from_graph()

        report = {
            "corpus_map": corpus_map,
            "questions": questions,
            "question_synthesis": question_synthesis,
            "patterns": self._export_patterns(),
            "entities": self._export_entities(),
            "contradictions": self._export_contradictions(),
            "group_indicators": self._export_group_indicators(),
            "notebook_summary": self.notebook.get_summary(),
            "question_graph": self._export_graph_summary(),
            "exploration_metadata": {
                "mode": "adaptive",
                "strategy": strategy,
                "total_budget": total_budget,
                "documents_read": self.notebook.corpus_map.get("total_documents_read", 0),
                "batches_processed": self.notebook.corpus_map.get("batches_processed", 0),
                "duration_seconds": max(0.0, (datetime.now() - start_time).total_seconds()),
                "timestamp": datetime.now().isoformat(),
                "research_lens": list(self._research_lens),
                "research_brief": self._research_brief.to_dict() if self._research_brief else None,
                "sort_order": self._research_brief.sort_order if self._research_brief else None,
                "prompt_variant": self._prompt_variant,
                "missing_primary_sort_fields": missing_sort_fields,
            },
        }

        if save_notebook:
            run_dir = f"adaptive_exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            notebook_path = save_with_timestamp(
                content=self.notebook.to_dict(),
                base_dir=Path(cfg.notebook_save_dir),
                filename_prefix="notebook",
                subdirectory=run_dir,
            )
            report_path = save_with_timestamp(
                content=report,
                base_dir=Path(cfg.notebook_save_dir),
                filename_prefix="report",
                subdirectory=run_dir,
            )
            report["notebook_path"] = str(notebook_path)
            report["report_path"] = str(report_path)

        self._persist_run(report, strategy, total_budget, year_range)
        return report

    def _initialize_graph(self, brief: ResearchBrief) -> QuestionGraph:
        # Axes are run-scoped and come from consultation, not static env defaults.
        axes = list(brief.axes) if brief.axes else list(APP_CONFIG.tier0.question_axes)
        graph = QuestionGraph(axes=axes)
        self._seed_node_ids = []

        if not brief.primary_lens.strip() and not brief.prior_hypotheses.strip():
            return graph

        if not self._interval_budget.request("heavy"):
            return graph

        seed_limit = max(1, int(APP_CONFIG.tier0.seed_max_questions))
        prior_candidates = extract_seed_questions(
            text=brief.prior_hypotheses,
            axes=axes,
            llm=self.llm,
            model=APP_CONFIG.tier0.ledger_expand_model,
            timeout_s=APP_CONFIG.tier0.llm_heavy_timeout,
            max_questions=min(3, seed_limit),
            prompt_variant=self._prompt_variant,
        )
        lens_candidates = extract_seed_questions(
            text=brief.primary_lens,
            axes=axes,
            llm=self.llm,
            model=APP_CONFIG.tier0.ledger_expand_model,
            timeout_s=APP_CONFIG.tier0.llm_heavy_timeout,
            max_questions=seed_limit,
            prompt_variant=self._prompt_variant,
        )

        # Deduplicate across hypothesis-derived and lens-derived seeds.
        seen = set()
        merged: List[Tuple[str, str, List[str], float]] = []
        for question, q_type, tags in prior_candidates:
            key = question.lower().strip()
            if not key or key in seen or q_type == "what":
                continue
            seen.add(key)
            merged.append((question, q_type, tags, 0.7))
        for question, q_type, tags in lens_candidates:
            key = question.lower().strip()
            if not key or key in seen or q_type == "what":
                continue
            seen.add(key)
            merged.append((question, q_type, tags, 0.5))
        merged = merged[:seed_limit]

        for question, q_type, tags, confidence in merged:
            node = QuestionNode(
                node_id=str(uuid.uuid4()),
                question_text=question,
                level="macro",
                status="open",
                origin="seed",
                priority=float(APP_CONFIG.tier0.seed_priority_boost),
                tags=[str(tag).strip().lower() for tag in tags if str(tag).strip()],
                question_type=q_type,  # type: ignore[arg-type]
                confidence=float(confidence),
                generation=0,
                first_seen_doc="",
                first_seen_batch="seed",
                embedding=None,
                is_leaf=False,
                merge_count=0,
                original_text=None,
                tension_score=0.0,
                evidence_owner=True,
            )
            graph.add_node(node)
            self._seed_node_ids.append(node.node_id)

        return graph

    def _process_stratum_adaptive(self, stratum) -> None:
        sub_batch_size = max(1, APP_CONFIG.tier0.sub_batch_docs)

        if stratum.stream:
            batch_index = 0
            for sub_docs in self.reader.iter_stratum_batches(stratum, sub_batch_size):
                batch_index += 1
                self._process_sub_batch(sub_docs, stratum_label=stratum.label, batch_index=batch_index)
            return

        docs = self.reader.read_stratum(stratum)
        if not docs:
            self.logger.log("batch_skip", f"{stratum.label} empty")
            return

        processed = 0
        batch_index = 0
        while processed < len(docs):
            batch_index += 1
            sub_docs = docs[processed:processed + sub_batch_size]
            processed += len(sub_docs)
            self._process_sub_batch(sub_docs, stratum_label=stratum.label, batch_index=batch_index)

    def _build_strata(
        self,
        strategy: str,
        total_budget: int,
        year_range: Optional[Tuple[int, int]],
        sort_order: Optional[str] = None,
    ):
        """
        Adaptive mode uses researcher-selected reading order as the primary traversal decision.
        Legacy strategy heuristics remain available via the "balanced" sort fallback.
        """
        chosen_sort = (sort_order or getattr(APP_CONFIG.tier0, "adaptive_default_sort", "archival")).strip().lower()
        if strategy in {"full", "exhaustive"}:
            return self.stratifier.full_corpus_stratification(total_budget=total_budget)
        return self.stratifier.build_ordered_strategy(
            total_budget=total_budget,
            sort_order=chosen_sort,
            year_range=year_range,
        )

    def _process_sub_batch(self, docs: List[Dict[str, Any]], stratum_label: str, batch_index: int) -> None:
        batch_label = f"{stratum_label} [{batch_index}]"
        doc_objects, consumed = self.reader.build_document_objects(
            docs,
            APP_CONFIG.tier0.batch_max_chars,
        )
        if consumed <= 0:
            consumed = 1

        if not doc_objects:
            self.logger.log("batch_skip", f"{batch_label} no blocks")
            return

        stats = self.reader.compute_object_stats(doc_objects)
        findings = self._analyze_batch(doc_objects)
        findings["stats"] = stats

        self.notebook.integrate_batch_findings(findings, batch_label)

        self._sub_batches_seen += 1
        self._docs_read_for_graph += len(doc_objects)

        self._integrate_questions(findings, batch_label)
        self._integrate_evidence(findings, batch_label)

        self.logger.log(
            "batch_done",
            f"{batch_label} docs={len(doc_objects)} questions={len(findings.get('questions', []))}",
        )

        if self._sub_batches_seen % max(1, APP_CONFIG.tier0.ledger_merge_interval) == 0:
            self._run_defrag(batch_label=batch_label)

    def _integrate_questions(self, findings: Dict[str, Any], batch_label: str) -> None:
        if self.graph is None:
            return

        for q in findings.get("questions", []) or []:
            if not isinstance(q, dict):
                continue
            raw_text = str(q.get("question") or "").strip()
            if not raw_text:
                continue

            final_text, q_type, changed = enforce_why_how(
                question_text=raw_text,
                level="micro",
                llm=self.llm,
                budget=self._interval_budget,
                prompt_variant=self._prompt_variant,
            )
            node = QuestionNode(
                node_id=str(uuid.uuid4()),
                question_text=final_text,
                level="micro",
                status="open",
                origin="emergent",
                priority=0.5,
                tags=self._infer_tags(final_text),
                question_type=q_type,  # type: ignore[arg-type]
                confidence=0.5,
                generation=0,
                first_seen_doc="",
                first_seen_batch=batch_label,
                embedding=None,
                is_leaf=False,
                merge_count=0,
                original_text=raw_text if changed else None,
                tension_score=0.0,
                evidence_owner=True,
            )
            node.embedding = self.graph.embed(node.question_text)

            similar = self.graph.find_similar(
                node.question_text,
                threshold=APP_CONFIG.tier0.ledger_lateral_threshold,
            )
            decision, log = decide_relation_with_seed_guard(
                candidate_node=node,
                similar_nodes=similar,
                llm=self.llm,
                budget=self._interval_budget,
                batch_label=batch_label,
            )
            self.graph.add_decision_log(log)

            self.graph.add_node(node)
            if decision.action == "merge" and decision.target is not None:
                self.graph.merge_into(
                    canonical_id=decision.target.node_id,
                    absorbed_id=node.node_id,
                    relation=decision.edge or "overlaps_with",
                    batch_label=batch_label,
                    decision_log_id=log.log_id,
                )
            elif decision.action == "generalize" and decision.target is not None:
                self.graph.add_edge(QuestionEdge(
                    edge_id=str(uuid.uuid4()),
                    source_id=node.node_id,
                    target_id=decision.target.node_id,
                    relation="generalizes",
                    confidence=0.75,
                    created_at_batch=batch_label,
                    decision_log_id=log.log_id,
                ))
            elif decision.action == "decompose" and decision.target is not None:
                self.graph.add_edge(QuestionEdge(
                    edge_id=str(uuid.uuid4()),
                    source_id=decision.target.node_id,
                    target_id=node.node_id,
                    relation="decomposes_into",
                    confidence=0.75,
                    created_at_batch=batch_label,
                    decision_log_id=log.log_id,
                ))
            elif decision.action == "lateral" and decision.target is not None:
                self.graph.add_edge(QuestionEdge(
                    edge_id=str(uuid.uuid4()),
                    source_id=node.node_id,
                    target_id=decision.target.node_id,
                    relation="overlaps_with",
                    confidence=0.6,
                    created_at_batch=batch_label,
                    decision_log_id=log.log_id,
                ))
            elif decision.lateral_link is not None:
                self.graph.add_edge(QuestionEdge(
                    edge_id=str(uuid.uuid4()),
                    source_id=node.node_id,
                    target_id=decision.lateral_link.node_id,
                    relation="overlaps_with",
                    confidence=0.6,
                    created_at_batch=batch_label,
                    decision_log_id=log.log_id,
                ))

            # Attach direct evidence from question-level block references when provided.
            # This gives promotions and seed confirmation a concrete signal instead of
            # relying only on later semantic pattern matching.
            raw_blocks = q.get("evidence_blocks")
            if isinstance(raw_blocks, list) and raw_blocks:
                evidence_targets = {self.graph.canonical_id(node.node_id)}
                if (
                    decision.action == "decompose"
                    and decision.target is not None
                    and decision.target.origin == "seed"
                ):
                    evidence_targets.add(self.graph.canonical_id(decision.target.node_id))

                for target_id in evidence_targets:
                    for block in raw_blocks[:6]:
                        block_id = str(block).strip()
                        if not block_id:
                            continue
                        doc_id = block_id.split("::")[0] if "::" in block_id else block_id
                        link = EvidenceLink(
                            link_id=str(uuid.uuid4()),
                            question_id=target_id,
                            doc_id=doc_id,
                            block_id=block_id,
                            evidence_type="extends",
                            strength=0.7,
                            note=f"Question-linked evidence: {final_text[:180]}",
                            batch_label=batch_label,
                        )
                        try:
                            self.graph.add_evidence(link, interval_index=self._defrag_index)
                        except Exception:
                            continue

    def _integrate_evidence(self, findings: Dict[str, Any], batch_label: str) -> None:
        if self.graph is None:
            return

        for pattern in findings.get("patterns", []) or []:
            if not isinstance(pattern, dict):
                continue
            text = str(pattern.get("pattern") or pattern.get("pattern_text") or "").strip()
            if not text:
                continue
            similar = self.graph.find_similar(text, threshold=APP_CONFIG.tier0.ledger_lateral_threshold)
            if not similar:
                continue
            node = similar[0][0]
            canonical_id = self.graph.canonical_id(node.node_id)
            blocks = pattern.get("evidence_blocks") or []
            confidence = str(pattern.get("confidence") or "low").lower()
            strength = {"low": 0.4, "medium": 0.65, "high": 0.85}.get(confidence, 0.4)
            for block in blocks[:6]:
                block_id = str(block)
                doc_id = block_id.split("::")[0] if "::" in block_id else block_id
                link = EvidenceLink(
                    link_id=str(uuid.uuid4()),
                    question_id=canonical_id,
                    doc_id=doc_id,
                    block_id=block_id,
                    evidence_type="extends",
                    strength=float(strength),
                    note=text[:240],
                    batch_label=batch_label,
                )
                try:
                    self.graph.add_evidence(link, interval_index=self._defrag_index)
                except Exception:
                    continue

        for contra in findings.get("contradictions", []) or []:
            if not isinstance(contra, dict):
                continue
            text = f"{contra.get('claim_a', '')} | {contra.get('claim_b', '')}".strip()
            if not text:
                continue
            similar = self.graph.find_similar(text, threshold=APP_CONFIG.tier0.ledger_lateral_threshold)
            if not similar:
                continue
            node = similar[0][0]
            canonical_id = self.graph.canonical_id(node.node_id)
            source_a = str(contra.get("source_a") or "")
            source_b = str(contra.get("source_b") or "")
            conflict_block = source_a if "::" in source_a else source_b
            conflict_doc = conflict_block.split("::")[0] if "::" in conflict_block else (source_a or source_b)
            if not conflict_doc:
                continue
            note = str(contra.get("context") or text)[:240]
            if conflict_block:
                note = f"{note} [block:{conflict_block}]"
            link = EvidenceLink(
                link_id=str(uuid.uuid4()),
                question_id=canonical_id,
                doc_id=str(conflict_doc),
                block_id=str(conflict_block or conflict_doc),
                evidence_type="contradicts",
                strength=0.8,
                note=note,
                batch_label=batch_label,
            )
            try:
                self.graph.add_evidence(link, interval_index=self._defrag_index)
            except Exception:
                continue

    def _run_defrag(self, batch_label: str) -> None:
        if self.graph is None:
            return
        self._defrag_index += 1
        snapshot = defrag(
            graph=self.graph,
            interval_index=self._defrag_index,
            docs_read=self._docs_read_for_graph,
            llm=self.llm,
            doc_store=self.doc_store,
            config=APP_CONFIG.tier0,
            batch_label=batch_label,
            budget=self._interval_budget,
            prompt_variant=self._prompt_variant,
        )
        self.logger.log(
            "defrag",
            f"interval={snapshot.interval_index} merges={snapshot.merges_performed} promotions={snapshot.promotions_performed} llm={snapshot.llm_calls_this_interval}/{APP_CONFIG.tier0.llm_budget_per_interval}",
        )

    def _infer_tags(self, text: str) -> List[str]:
        lowered = text.lower()
        tags: List[str] = []
        for axis in APP_CONFIG.tier0.question_axes:
            token = str(axis).strip().lower()
            if not token:
                continue
            if token in lowered:
                tags.append(token)
        if not tags:
            if any(tok in lowered for tok in ["year", "decade", "time", "period"]):
                tags.append("time")
            if any(tok in lowered for tok in ["place", "location", "division", "city", "state"]):
                tags.append("place")
            if any(tok in lowered for tok in ["occupation", "worker", "job", "department"]):
                tags.append("group")
        return tags

    def _batch_analysis_prompt_template(self) -> str:
        # Adaptive mode can swap prompt behavior by run without changing JSON output contracts.
        return get_batch_analysis_prompt(self._prompt_variant)

    def _batch_analysis_system_message(self) -> str:
        # Keep system behavior aligned with the selected adaptive prompt variant.
        return get_batch_system_message(self._prompt_variant)

    def _export_graph_questions(self) -> List[Dict[str, Any]]:
        if self.graph is None:
            return []

        nodes = [
            node for node in self.graph.nodes.values()
            if node.evidence_owner and node.level in {"meso", "macro"}
        ]
        nodes.sort(
            key=lambda n: (
                self.graph.attention_weight(n, current_interval=self._defrag_index, docs_read=self._docs_read_for_graph),
                n.priority,
            ),
            reverse=True,
        )

        output: List[Dict[str, Any]] = []
        for node in nodes[: max(1, APP_CONFIG.tier0.question_target_count)]:
            evidence = self.graph.get_evidence(node.node_id)
            output.append({
                "question": node.question_text,
                "type": node.question_type,
                "level": node.level,
                "origin": node.origin,
                "priority": round(node.priority, 3),
                "tension_score": round(node.tension_score, 3),
                "evidence_count": len(evidence),
                "evidence_sample": [e.doc_id for e in evidence[:5]],
                "validation": {
                    "score": int(max(0, min(100, round(node.priority * 100)))),
                    "status": "acceptable" if node.level == "meso" else "good",
                },
            })
        return output

    def _node_to_question_model(self, node: QuestionNode) -> Optional[Question]:
        mapping = {
            "why": QuestionType.CAUSAL,
            "how": QuestionType.CAUSAL,
            "compare": QuestionType.COMPARATIVE,
            "change_continuity": QuestionType.CHANGE_OVER_TIME,
            "explain": QuestionType.INSTITUTIONAL,
            "what": QuestionType.SCOPE_CONDITIONS,
        }
        qtype = mapping.get(node.question_type)
        if qtype is None:
            return None

        evidence = self.graph.get_evidence(node.node_id) if self.graph else []
        return Question(
            question_text=node.question_text,
            question_type=qtype,
            why_interesting=f"Adaptive notebook node ({node.level}, {node.origin}).",
            time_window=None,
            entities_involved=[],
            evidence_doc_ids=[e.doc_id for e in evidence[:20]],
            evidence_block_ids=[e.block_id for e in evidence[:20]],
            generation_method="adaptive_graph",
            validation_score=int(max(0, min(100, round(node.priority * 100)))),
        )

    def _generate_question_synthesis_from_graph(self) -> Dict[str, Any]:
        if not APP_CONFIG.tier0.synthesis_enabled:
            return {}
        if self.graph is None:
            return {}

        candidates = [
            node for node in self.graph.nodes.values()
            if node.evidence_owner and node.level in {"meso", "macro"}
        ]
        if not candidates:
            return {}

        models = []
        for node in candidates[: max(1, APP_CONFIG.tier0.question_target_count)]:
            qm = self._node_to_question_model(node)
            if qm is not None:
                models.append(qm)
        if not models:
            return {}

        return self.question_synthesizer.build_agenda(self.notebook, models)

    def _export_graph_summary(self) -> Dict[str, Any]:
        if self.graph is None:
            return {}

        owner_nodes = [node for node in self.graph.nodes.values() if node.evidence_owner]
        by_level: Dict[str, int] = {"micro": 0, "meso": 0, "macro": 0}
        by_origin: Dict[str, int] = {"seed": 0, "emergent": 0}
        by_status: Dict[str, int] = {"open": 0, "partial": 0, "answered": 0}

        for node in owner_nodes:
            by_level[node.level] = by_level.get(node.level, 0) + 1
            by_origin[node.origin] = by_origin.get(node.origin, 0) + 1
            by_status[node.status] = by_status.get(node.status, 0) + 1

        seed_nodes = [node for node in owner_nodes if node.origin == "seed"]
        seed_confirmed = sum(1 for node in seed_nodes if len(self.graph.get_evidence(node.node_id)) > 0)
        seed_unconfirmed = max(0, len(seed_nodes) - seed_confirmed)

        high_tension = []
        for node in self.graph.get_high_tension_nodes(APP_CONFIG.tier0.tension_threshold):
            evidence = self.graph.get_evidence(node.node_id)
            confirms = sum(1 for e in evidence if e.evidence_type in ("confirms", "extends"))
            contradicts = sum(1 for e in evidence if e.evidence_type == "contradicts")
            high_tension.append({
                "question": node.question_text,
                "tension_score": round(node.tension_score, 3),
                "confirms": confirms,
                "contradicts": contradicts,
            })

        change_nodes = [node for node in owner_nodes if node.question_type == "change_continuity"]
        surprise_absences = sum(1 for e in self.graph.evidence if e.evidence_type == "absence")

        return {
            "total_nodes": len(owner_nodes),
            "by_level": by_level,
            "by_origin": by_origin,
            "by_status": by_status,
            "seed_questions_confirmed": seed_confirmed,
            "seed_questions_unconfirmed": seed_unconfirmed,
            "high_tension_nodes": high_tension,
            "change_continuity_questions": len(change_nodes),
            "total_edges": len(self.graph.edges),
            "surprise_absences": surprise_absences,
            "defrag_snapshots": [asdict(snapshot) for snapshot in self.graph.defrag_snapshots],
            "tree": [],
            "decision_log_count": len(self.graph.decision_log),
        }
