#!/usr/bin/env python3
"""
End-to-end CLI smoke suite for Historical Document Reader.

Runs against the Flask app using the Flask test client and live MongoDB data.
Designed for quick full-path validation after backend/frontend changes.

Usage:
  python /app/util/e2e_cli_suite.py
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from bson import ObjectId

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from database_setup import get_client, get_db
from main import cache
from routes import app
from util.unique_terms_contract_smoke import run_unique_terms_contract_check


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


class E2ESuite:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def _ok(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult(name=name, passed=True, detail=detail))
        print(f"[PASS] {name}" + (f" :: {detail}" if detail else ""))

    def _fail(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult(name=name, passed=False, detail=detail))
        print(f"[FAIL] {name}" + (f" :: {detail}" if detail else ""))

    def check(self, name: str, fn: Callable[[], str | None]) -> None:
        try:
            detail = fn() or ""
            self._ok(name, detail)
        except AssertionError as exc:
            self._fail(name, str(exc))
        except Exception as exc:  # pragma: no cover - defensive harness behavior
            tb = traceback.format_exc(limit=2)
            self._fail(name, f"{exc}\n{tb}")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def as_dict(payload: Any, context: str) -> dict[str, Any]:
    require(isinstance(payload, dict), f"{context}: expected JSON object, got {type(payload).__name__}")
    return payload


def main() -> int:
    suite = E2ESuite()

    db_client = get_client()
    db = get_db(db_client)

    context: dict[str, Any] = {
        "sample_doc_id": None,
        "sample_doc_with_refs_id": None,
        "sample_entity_id": None,
        "network_enabled": True,
    }

    def check_environment_contract() -> str:
        mongo_uri = (os.environ.get("APP_MONGO_URI") or os.environ.get("MONGO_URI") or "").strip()
        require(bool(mongo_uri), "APP_MONGO_URI or MONGO_URI must be set")

        db_name = (os.environ.get("DB_NAME") or "railroad_documents").strip()
        require(bool(db_name), "DB_NAME resolved to an empty value")

        chroma_dir = (os.environ.get("CHROMA_PERSIST_DIRECTORY") or "/data/chroma_db").strip()
        require(bool(chroma_dir), "CHROMA_PERSIST_DIRECTORY resolved to an empty value")

        return f"db={db_name} chroma_dir={chroma_dir}"

    suite.check("Environment contract", check_environment_contract)

    def check_python_dependency_contract() -> str:
        modules = ["openai", "spacy", "chromadb", "numpy", "pymongo"]
        for module_name in modules:
            importlib.import_module(module_name)
        return f"imported={','.join(modules)}"

    suite.check("Python dependency imports", check_python_dependency_contract)

    def check_spacy_model_contract() -> str:
        import spacy

        nlp = spacy.load("en_core_web_lg")
        doc = nlp("William Akel worked for the Baltimore and Ohio Railroad in 1928.")
        require(len(doc) > 0, "spaCy tokenization returned no tokens")
        require(len(doc.ents) > 0, "spaCy model produced zero entities on a simple sample")
        return f"entities={len(doc.ents)}"

    suite.check("spaCy model load + NER sanity", check_spacy_model_contract)

    def check_chromadb_contract() -> str:
        import chromadb
        from chromadb.config import Settings

        persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY", "/data/chroma_db")
        if not Path(persist_dir).exists():
            return f"skipped (persist dir not mounted): {persist_dir}"

        chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        collection_names = {collection.name for collection in chroma_client.list_collections()}
        if "historian_documents" not in collection_names:
            return "skipped (historian_documents collection missing)"

        collection = chroma_client.get_collection("historian_documents")
        sample = collection.get(limit=1, include=["embeddings"])
        raw_embeddings = sample.get("embeddings")  # Avoid ndarray truthiness checks that raise ValueError.
        if raw_embeddings is None:
            return "skipped (historian_documents has no embeddings)"
        # Normalize to list-like access for both numpy arrays and Python lists.
        if hasattr(raw_embeddings, "tolist"):
            embeddings = raw_embeddings.tolist()
        else:
            embeddings = list(raw_embeddings)
        if len(embeddings) == 0:
            return "skipped (historian_documents has no embeddings)"
        embedding_dim = len(embeddings[0])
        require(embedding_dim > 0, "embedding dimension must be > 0")
        return f"collection=historian_documents dim={embedding_dim}"

    suite.check("Chroma vector-store contract", check_chromadb_contract)

    def check_historian_agent_imports() -> str:
        modules = [
            "historian_agent.rag_query_handler",
            "historian_agent.adversarial_rag",
            "historian_agent.iterative_adversarial_agent",
        ]
        for module_name in modules:
            importlib.import_module(module_name)
        return f"imported={','.join(modules)}"

    suite.check("Historian-agent module imports", check_historian_agent_imports)

    with app.test_client() as client:
        def check_db_preflight() -> str:
            total_docs = db["documents"].count_documents({})
            total_linked = db["linked_entities"].count_documents({})
            total_edges = db["network_edges"].count_documents({}) if "network_edges" in db.list_collection_names() else 0

            require(total_docs > 0, "documents collection is empty")
            require(total_linked > 0, "linked_entities collection is empty")
            require(total_edges > 0, "network_edges collection missing or empty")

            sample_doc = db["documents"].find_one({}, {"_id": 1})
            require(sample_doc is not None, "could not load sample document")
            context["sample_doc_id"] = str(sample_doc["_id"])

            doc_with_refs = db["documents"].find_one(
                {"entity_refs": {"$exists": True, "$ne": []}},
                {"_id": 1, "entity_refs": 1},
            )
            require(doc_with_refs is not None, "no document with non-empty entity_refs")
            context["sample_doc_with_refs_id"] = str(doc_with_refs["_id"])

            resolved_entity_id = None
            for ref in doc_with_refs.get("entity_refs", []):
                ref_id = str(ref)
                if not ObjectId.is_valid(ref_id):
                    continue
                linked = db["linked_entities"].find_one({"_id": ObjectId(ref_id)}, {"_id": 1})
                if linked:
                    resolved_entity_id = str(linked["_id"])
                    break

            if not resolved_entity_id:
                linked = db["linked_entities"].find_one({}, {"_id": 1})
                require(linked is not None, "no linked_entities records found")
                resolved_entity_id = str(linked["_id"])

            context["sample_entity_id"] = resolved_entity_id
            return f"docs={total_docs}, linked_entities={total_linked}, network_edges={total_edges}"

        suite.check("DB preflight and sample IDs", check_db_preflight)

        def check_document_structure_contract() -> str:
            sample_doc = db["documents"].find_one(
                {
                    "$or": [
                        {"content": {"$exists": True, "$type": "string", "$ne": ""}},
                        {"ocr_text": {"$exists": True, "$type": "string", "$ne": ""}},
                        {"summary": {"$exists": True, "$type": "string", "$ne": ""}},
                    ]
                },
                {"_id": 1, "content": 1, "ocr_text": 1, "summary": 1, "sections": 1},
            )
            require(sample_doc is not None, "documents collection has no sample document")

            populated_text_fields = [
                field_name
                for field_name in ("content", "ocr_text", "summary")
                if isinstance(sample_doc.get(field_name), str) and sample_doc.get(field_name).strip()
            ]
            require(populated_text_fields, "sample document has no populated content/ocr_text/summary field")

            sections_value = sample_doc.get("sections")
            if sections_value is not None:
                require(isinstance(sections_value, list), "document.sections must be a list when present")

            return f"text_fields={','.join(populated_text_fields)}"

        suite.check("Document structure contract", check_document_structure_contract)

        def check_core_pages() -> str:
            for path in ["/", "/search", "/historian-agent", "/network-analysis", "/help"]:
                resp = client.get(path)
                require(resp.status_code == 200, f"{path} returned {resp.status_code}")
            return "core pages returned HTTP 200"

        suite.check("Core page routes", check_core_pages)

        def check_tooltip_standard_contract() -> str:
            tooltip_paths = [
                "/",
                "/search",
                "/search-terms",
                "/historian-agent",
                "/corpus-explorer",
                "/network-analysis",
                "/database-info",
                "/settings",
                "/help",
            ]
            total_targets = 0
            for path in tooltip_paths:
                resp = client.get(path)
                require(resp.status_code == 200, f"{path} returned {resp.status_code}")
                body = resp.data.decode("utf-8", errors="ignore")
                # Contract: pages must load the shared AppHelpTooltips runtime from base.html.
                require("js/help-tooltips.js" in body, f"{path} missing help-tooltips.js include")
                # Contract: pages should expose at least one data-help target for consistent UX.
                page_targets = body.count("data-help=")
                require(page_targets > 0, f"{path} has zero data-help targets")
                total_targets += page_targets
            return f"paths={len(tooltip_paths)} total_data_help_targets={total_targets}"

        suite.check("Tooltip standard contract", check_tooltip_standard_contract)

        def check_static_assets() -> str:
            for path in [
                "/static/js/network.js",
                "/static/js/network_statistics.js",
                "/static/js/d3.v7.min.js",
                "/static/style.css",
            ]:
                resp = client.get(path)
                require(resp.status_code == 200, f"{path} returned {resp.status_code}")
                require(len(resp.data) > 0, f"{path} returned empty body")
            return "network/static assets accessible"

        suite.check("Static assets", check_static_assets)

        def check_search_contract() -> str:
            payload = {
                "searchField1": "",
                "searchTerm1": "",
                "operator1": "contains",
                "condition": "and",
                "page": 1,
                "per_page": 20,
            }
            resp = client.post("/search", json=payload)
            require(resp.status_code == 200, f"/search returned {resp.status_code}")
            data = as_dict(resp.get_json(silent=True), "/search")
            require("search_id" in data, "/search missing search_id")
            require("documents" in data and isinstance(data["documents"], list), "/search missing documents list")
            require(len(data["documents"]) > 0, "/search returned zero documents")

            first = data["documents"][0]
            first_id = first.get("_id")
            require(first_id, "first search result missing _id")

            doc_resp = client.get(f"/document/{first_id}?search_id={data['search_id']}")
            require(doc_resp.status_code == 200, f"document detail from search context returned {doc_resp.status_code}")
            require(b"Previous" in doc_resp.data and b"Next" in doc_resp.data, "document page missing Prev/Next controls")
            return f"search_id={data['search_id']} documents={len(data['documents'])}"

        suite.check("Search API + document navigation context", check_search_contract)

        def check_unique_terms_explorer_contract() -> str:
            # Added dedicated contract smoke so unique_terms regressions fail the master suite.
            return run_unique_terms_contract_check(client, db)

        suite.check("Unique terms explorer contract", check_unique_terms_explorer_contract)

        def check_network_types_and_stats() -> str:
            types_resp = client.get("/api/network/types")
            require(types_resp.status_code == 200, f"/api/network/types returned {types_resp.status_code}")
            types_data = as_dict(types_resp.get_json(silent=True), "/api/network/types")
            types = types_data.get("types", [])
            require(isinstance(types, list), "types payload is not a list")
            context["network_enabled"] = not bool(types_data.get("disabled", False))

            stats_resp = client.get("/api/network/stats")
            require(stats_resp.status_code == 200, f"/api/network/stats returned {stats_resp.status_code}")
            stats_data = as_dict(stats_resp.get_json(silent=True), "/api/network/stats")
            require("exists" in stats_data, "stats missing exists flag")
            return f"types={len(types)} network_enabled={context['network_enabled']}"

        suite.check("Network types/stats endpoints", check_network_types_and_stats)

        def check_network_global_contract() -> str:
            resp = client.get("/api/network/global?min_weight=3&limit=200&person_min_mentions=3")
            require(resp.status_code == 200, f"/api/network/global returned {resp.status_code}")
            data = as_dict(resp.get_json(silent=True), "/api/network/global")
            require("nodes" in data and isinstance(data["nodes"], list), "global missing nodes list")
            require("edges" in data and isinstance(data["edges"], list), "global missing edges list")
            require("stats" in data and isinstance(data["stats"], dict), "global missing stats object")
            return f"nodes={len(data['nodes'])}, edges={len(data['edges'])}"

        suite.check("Network global endpoint contract", check_network_global_contract)

        def check_network_targeted_endpoints() -> str:
            sample_doc_id = context.get("sample_doc_with_refs_id")
            sample_entity_id = context.get("sample_entity_id")
            require(sample_doc_id, "sample_doc_with_refs_id unavailable")
            require(sample_entity_id, "sample_entity_id unavailable")

            doc_resp = client.get(f"/api/network/document/{sample_doc_id}?min_weight=1")
            require(doc_resp.status_code == 200, f"document endpoint returned {doc_resp.status_code}")
            doc_data = as_dict(doc_resp.get_json(silent=True), "document endpoint")
            require("nodes" in doc_data and "edges" in doc_data, "document endpoint missing nodes/edges")

            rel_resp = client.get(f"/api/network/related/{sample_doc_id}?limit=8")
            require(rel_resp.status_code == 200, f"related endpoint returned {rel_resp.status_code}")
            rel_data = as_dict(rel_resp.get_json(silent=True), "related endpoint")
            require("related" in rel_data and isinstance(rel_data["related"], list), "related endpoint missing related list")

            ego_resp = client.get(f"/api/network/entity/{sample_entity_id}?limit=20")
            require(ego_resp.status_code == 200, f"entity endpoint returned {ego_resp.status_code}")
            ego_data = as_dict(ego_resp.get_json(silent=True), "entity endpoint")
            require("entity" in ego_data and "edges" in ego_data and "metrics" in ego_data, "entity endpoint shape mismatch")

            metrics_resp = client.get(f"/api/network/metrics/{sample_entity_id}")
            require(metrics_resp.status_code == 200, f"metrics endpoint returned {metrics_resp.status_code}")
            metrics_data = as_dict(metrics_resp.get_json(silent=True), "metrics endpoint")
            require("degree" in metrics_data and "weighted_degree" in metrics_data, "metrics endpoint missing degree fields")
            return "document, related, entity, metrics endpoints returned expected shapes"

        suite.check("Network targeted endpoints", check_network_targeted_endpoints)

        def check_network_invalid_id_handling() -> str:
            bad_entity_resp = client.get("/api/network/entity/not-a-valid-id")
            require(bad_entity_resp.status_code in (404, 503), f"bad entity id status unexpected: {bad_entity_resp.status_code}")

            bad_doc_resp = client.get("/api/network/document/not-a-valid-id")
            require(bad_doc_resp.status_code in (404, 503), f"bad document id status unexpected: {bad_doc_resp.status_code}")

            for resp in [bad_entity_resp, bad_doc_resp]:
                payload = resp.get_json(silent=True)
                require(isinstance(payload, dict) and "error" in payload, "error payload missing 'error' key")
            return "invalid IDs return controlled error payloads"

        suite.check("Network invalid ID handling", check_network_invalid_id_handling)

        def check_document_network_markup() -> str:
            sample_doc_id = context.get("sample_doc_with_refs_id")
            require(sample_doc_id, "sample_doc_with_refs_id unavailable")
            resp = client.get(f"/document/{sample_doc_id}")
            require(resp.status_code == 200, f"/document/<id> returned {resp.status_code}")
            body = resp.data.decode("utf-8", errors="ignore")
            require("Related Documents via Network" in body, "document page missing related-docs network section")
            require("network-context-graph" in body, "document page missing network context graph container")
            require("data-entity-id" in body, "document page missing entity-id markup for interaction")
            return "document template includes network context and entity hooks"

        suite.check("Document page network integration markup", check_document_network_markup)

        def check_network_viewer_flow() -> str:
            launch = client.get("/network/viewer-launch?min_weight=3&limit=200", follow_redirects=False)
            require(launch.status_code in (302, 303), f"viewer launch returned {launch.status_code}")
            location = launch.headers.get("Location")
            require(location and "/network/viewer-results" in location, "viewer launch missing viewer-results redirect")

            parsed = urlparse(location)
            query = parse_qs(parsed.query)
            search_id = (query.get("search_id") or [""])[0]
            require(search_id, "viewer redirect missing search_id")

            results_resp = client.get(location)
            require(results_resp.status_code == 200, f"viewer results returned {results_resp.status_code}")
            body = results_resp.data.decode("utf-8", errors="ignore")
            require("Network Document Viewer" in body, "viewer results page missing heading")

            ordered_ids = cache.get(f"search_{search_id}") or []
            if ordered_ids:
                first_id = str(ordered_ids[0])
                doc_resp = client.get(f"/document/{first_id}?search_id={search_id}&origin=network")
                require(doc_resp.status_code == 200, f"viewer document open returned {doc_resp.status_code}")
            return f"search_id={search_id} cached_docs={len(ordered_ids)}"

        suite.check("Network viewer launch/results flow", check_network_viewer_flow)

        def check_network_statistics_endpoints() -> str:
            endpoints = [
                ("/api/network/statistics/graph_summary?min_weight=3", "graph_summary"),
                ("/api/network/statistics/available_attributes?min_weight=3", "available_attributes"),
                ("/api/network/statistics/assortativity?attribute=type&n_permutations=20", "assortativity"),
                ("/api/network/statistics/mixing_matrix?attribute=type", "mixing_matrix"),
                ("/api/network/statistics/degree_distribution", "degree_distribution"),
                ("/api/network/statistics/communities", "communities"),
                ("/api/network/statistics/gatekeepers?limit=5", "gatekeepers"),
                ("/api/network/statistics/comparison?n_simulations=10", "comparison"),
                ("/api/network/statistics/summary?n_permutations=10&n_simulations=10", "summary"),
            ]

            for path, label in endpoints:
                resp = client.get(path)
                require(resp.status_code == 200, f"{label} returned {resp.status_code}")
                payload = resp.get_json(silent=True)
                require(isinstance(payload, dict), f"{label} did not return JSON object")
                # Some endpoints can soft-fail with {error: ...} while still 200.
                if "error" in payload:
                    # graph too small for stats is a valid soft-failure case.
                    require("too small" in str(payload["error"]).lower(), f"{label} unexpected error: {payload['error']}")

            return "all statistics endpoints responded with valid JSON"

        suite.check("Network statistics endpoints", check_network_statistics_endpoints)

        def check_network_analysis_markup() -> str:
            resp = client.get("/network-analysis")
            require(resp.status_code == 200, f"/network-analysis returned {resp.status_code}")
            body = resp.data.decode("utf-8", errors="ignore")
            for token in [
                "id=\"network-graph\"",
                "id=\"reset-network-controls\"",
                "id=\"statistics-panel\"",
                "network_statistics.js",
            ]:
                require(token in body, f"/network-analysis missing token: {token}")
            return "network analysis page includes graph, reset button, and statistics panel"

        suite.check("Network analysis page markup", check_network_analysis_markup)

    total = len(suite.results)
    failed = suite.failed
    passed = total - failed

    print("\n=== E2E CLI SUMMARY ===")
    print(f"Total checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
