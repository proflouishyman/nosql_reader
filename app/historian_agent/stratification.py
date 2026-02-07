# app/historian_agent/stratification.py
# Created: 2026-02-05
# Purpose: Stratification strategies and notebook-style document objects

"""
Stratification - Divide corpus into meaningful reading batches.

Adds Notebook-LLM style document objects with semantic chunking.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

from rag_base import DocumentStore, debug_print, MongoDBConnection
from config import APP_CONFIG

from historian_agent.semantic_chunker import SemanticChunker
from historian_agent.tier0_models import DocumentObject


# ============================================================================
# Stratum Definition
# ============================================================================

@dataclass
class Stratum:
    """One stratum (reading batch) of the corpus."""
    stratum_type: str
    label: str
    filters: Dict[str, Any]
    sample_size: int
    priority: int = 1
    stream: bool = False


# ============================================================================
# Stratification Strategies
# ============================================================================

class CorpusStratifier:
    """Creates systematic sampling strategies for corpus exploration."""

    def __init__(self):
        self.doc_store = DocumentStore()
        self.documents_coll = self.doc_store.documents_coll

    def temporal_stratification(
        self,
        year_range: Optional[Tuple[int, int]] = None,
        docs_per_year: int = 50
    ) -> List[Stratum]:
        debug_print("Building temporal stratification")

        pipeline = [
            {"$match": {"year": {"$ne": None}}},
            {"$group": {"_id": "$year", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]

        year_counts = list(self.documents_coll.aggregate(pipeline))
        if not year_counts:
            debug_print("No year data found in corpus")
            return []

        if year_range:
            year_counts = [
                y for y in year_counts
                if year_range[0] <= y["_id"] <= year_range[1]
            ]

        strata: List[Stratum] = []
        for year_data in year_counts:
            year = year_data["_id"]
            count = year_data["count"]
            sample_size = min(docs_per_year, count)
            strata.append(Stratum(
                stratum_type="temporal",
                label=f"Year {year}",
                filters={"year": year},
                sample_size=sample_size,
                priority=1,
            ))

        debug_print(f"Created {len(strata)} temporal strata")
        return strata

    def genre_stratification(self, docs_per_type: int = 100) -> List[Stratum]:
        debug_print("Building genre stratification")

        pipeline = [
            {"$match": {"document_type": {"$ne": None}}},
            {"$group": {"_id": "$document_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]

        type_counts = list(self.documents_coll.aggregate(pipeline))
        if not type_counts:
            debug_print("No document_type data found")
            return []

        strata: List[Stratum] = []
        for type_data in type_counts:
            doc_type = type_data["_id"]
            count = type_data["count"]
            sample_size = min(docs_per_type, count)
            strata.append(Stratum(
                stratum_type="genre",
                label=f"{doc_type}",
                filters={"document_type": doc_type},
                sample_size=sample_size,
                priority=2,
            ))

        debug_print(f"Created {len(strata)} genre strata")
        return strata

    def biographical_stratification(
        self,
        min_docs_per_person: int = 10,
        max_people: int = 50,
        docs_per_person: int = 20
    ) -> List[Stratum]:
        debug_print("Building biographical stratification")

        pipeline = [
            {"$match": {"person_id": {"$ne": None}}},
            {"$group": {
                "_id": {
                    "person_id": "$person_id",
                    "person_name": "$person_name",
                    "person_folder": "$person_folder",
                },
                "count": {"$sum": 1},
            }},
            {"$match": {"count": {"$gte": min_docs_per_person}}},
            {"$sort": {"count": -1}},
            {"$limit": max_people},
        ]

        people = list(self.documents_coll.aggregate(pipeline))
        if not people:
            debug_print("No person data found")
            return []

        strata: List[Stratum] = []
        for person_data in people:
            person_info = person_data["_id"]
            count = person_data["count"]
            sample_size = min(docs_per_person, count)
            label = f"{person_info.get('person_name', 'Unknown')} (ID: {person_info.get('person_id')}, {count} docs)"

            strata.append(Stratum(
                stratum_type="biographical",
                label=label,
                filters={"person_id": person_info.get("person_id")},
                sample_size=sample_size,
                priority=3,
            ))

        debug_print(f"Created {len(strata)} biographical strata")
        return strata

    def collection_stratification(self, docs_per_collection: int = 100) -> List[Stratum]:
        debug_print("Building collection stratification")

        pipeline = [
            {"$match": {"collection": {"$ne": None}}},
            {"$group": {"_id": "$collection", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]

        collections = list(self.documents_coll.aggregate(pipeline))
        if not collections:
            debug_print("No collection data found")
            return []

        strata: List[Stratum] = []
        for coll_data in collections:
            collection = coll_data["_id"]
            count = coll_data["count"]
            sample_size = min(docs_per_collection, count)
            strata.append(Stratum(
                stratum_type="collection",
                label=f"Collection: {collection}",
                filters={"collection": collection},
                sample_size=sample_size,
                priority=2,
            ))

        debug_print(f"Created {len(strata)} collection strata")
        return strata

    def spatial_stratification(self, docs_per_box: int = 50) -> List[Stratum]:
        debug_print("Building spatial stratification")

        pipeline = [
            {"$match": {"archive_structure.physical_box": {"$ne": None}}},
            {"$group": {"_id": "$archive_structure.physical_box", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]

        boxes = list(self.documents_coll.aggregate(pipeline))
        if not boxes:
            debug_print("No physical_box data found")
            return []

        strata: List[Stratum] = []
        for box_data in boxes:
            box = box_data["_id"]
            count = box_data["count"]
            sample_size = min(docs_per_box, count)
            strata.append(Stratum(
                stratum_type="spatial",
                label=f"Box: {box}",
                filters={"archive_structure.physical_box": box},
                sample_size=sample_size,
                priority=2,
            ))

        debug_print(f"Created {len(strata)} spatial strata")
        return strata

    def build_comprehensive_strategy(
        self,
        total_budget: int = 2000,
        strategy: str = "balanced"
    ) -> List[Stratum]:
        debug_print(f"Building comprehensive strategy: {strategy}, budget: {total_budget}")

        all_strata: List[Stratum] = []

        if strategy in {"full", "exhaustive"}:
            all_strata = self.full_corpus_stratification(total_budget=total_budget)
        elif strategy == "temporal":
            all_strata = self.temporal_stratification(docs_per_year=50)
        elif strategy == "biographical":
            all_strata = self.biographical_stratification(
                min_docs_per_person=10,
                max_people=50,
                docs_per_person=30,
            )
        elif strategy == "genre":
            all_strata = self.genre_stratification(docs_per_type=100)
        elif strategy == "balanced":
            temporal = self.temporal_stratification(docs_per_year=25)
            genre = self.genre_stratification(docs_per_type=50)
            biographical = self.biographical_stratification(
                min_docs_per_person=10,
                max_people=20,
                docs_per_person=20,
            )
            collection = self.collection_stratification(docs_per_collection=50)
            all_strata = temporal + genre + biographical + collection
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        all_strata.sort(key=lambda s: s.priority)
        if strategy not in {"full", "exhaustive"}:
            all_strata = self._trim_to_budget(all_strata, total_budget)

        debug_print(
            f"Final strategy: {len(all_strata)} strata, ~{sum(s.sample_size for s in all_strata)} docs"
        )
        return all_strata

    def full_corpus_stratification(self, total_budget: Optional[int] = None) -> List[Stratum]:
        debug_print("Building full-corpus stratification")

        total_docs = self.documents_coll.count_documents({})
        if total_budget is not None:
            sample_size = min(total_budget, total_docs)
        else:
            sample_size = total_docs

        return [
            Stratum(
                stratum_type="full",
                label="Full Corpus",
                filters={},
                sample_size=sample_size,
                priority=1,
                stream=True,
            )
        ]

    def _trim_to_budget(self, strata: List[Stratum], budget: int) -> List[Stratum]:
        total = 0
        trimmed: List[Stratum] = []
        for stratum in strata:
            if total + stratum.sample_size <= budget:
                trimmed.append(stratum)
                total += stratum.sample_size
            else:
                remaining = budget - total
                if remaining > 0:
                    stratum.sample_size = remaining
                    trimmed.append(stratum)
                break
        return trimmed


# ============================================================================
# Stratum Reader
# ============================================================================

class StratumReader:
    """Reads documents from a stratum and builds document objects."""

    def __init__(self):
        self.doc_store = DocumentStore()
        self.documents_coll = self.doc_store.documents_coll
        self.chunker = SemanticChunker(
            enabled=APP_CONFIG.tier0.semantic_chunking,
            max_block_chars=APP_CONFIG.tier0.block_max_chars,
            max_blocks_per_doc=APP_CONFIG.tier0.max_blocks_per_doc,
            fallback_chunk_chars=APP_CONFIG.tier0.block_max_chars,
            fallback_chunk_overlap=max(50, APP_CONFIG.tier0.block_max_chars // 10),
        )

    def read_stratum(self, stratum: Stratum) -> List[Dict[str, Any]]:
        debug_print(f"Reading stratum: {stratum.label} ({stratum.sample_size} docs)")

        query = stratum.filters.copy()
        pipeline = [
            {"$match": query},
            {"$sample": {"size": stratum.sample_size}},
        ]

        docs = list(self.documents_coll.aggregate(pipeline))
        debug_print(f"Retrieved {len(docs)} documents from {stratum.label}")
        return docs

    def iter_stratum_batches(self, stratum: Stratum, batch_size: int):
        """Stream documents for a stratum in batches (for full-corpus runs)."""
        debug_print(f"Streaming stratum: {stratum.label} (limit {stratum.sample_size} docs)")

        query = stratum.filters.copy()
        mongo = MongoDBConnection()
        with mongo.client.start_session() as session:
            cursor = self.documents_coll.find(query, no_cursor_timeout=True, session=session)

            if stratum.sample_size:
                cursor = cursor.limit(stratum.sample_size)

            batch: List[Dict[str, Any]] = []
            try:
                for doc in cursor:
                    batch.append(doc)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch
            finally:
                cursor.close()

    def build_document_objects(
        self,
        docs: List[Dict[str, Any]],
        max_chars: int
    ) -> List[DocumentObject]:
        """Build notebook-style document objects with semantic blocks."""
        objects: List[DocumentObject] = []
        total_chars = 0

        for doc in docs:
            doc_obj = self._document_to_object(doc)
            if not doc_obj.blocks:
                continue

            estimated = len(json.dumps(doc_obj.to_prompt_dict(), ensure_ascii=True))
            if total_chars + estimated > max_chars and objects:
                break

            objects.append(doc_obj)
            total_chars += estimated

        return objects

    def format_document_objects_for_llm(self, objects: List[DocumentObject]) -> str:
        """Format document objects as JSON for LLM context."""
        return json.dumps([obj.to_prompt_dict() for obj in objects], ensure_ascii=True, indent=2)

    def compute_batch_stats(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute stats for a batch (code-side, not LLM)."""
        stats = {
            "docs_in_batch": len(docs),
            "by_year": defaultdict(int),
            "by_collection": defaultdict(int),
            "by_document_type": defaultdict(int),
            "by_person": defaultdict(int),
            "earliest_year": None,
            "latest_year": None,
        }

        for doc in docs:
            year = doc.get("year")
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            if isinstance(year, int):
                stats["by_year"][str(year)] += 1
                if stats["earliest_year"] is None or year < stats["earliest_year"]:
                    stats["earliest_year"] = year
                if stats["latest_year"] is None or year > stats["latest_year"]:
                    stats["latest_year"] = year

            collection = doc.get("collection")
            if collection:
                stats["by_collection"][str(collection)] += 1

            doc_type = doc.get("document_type") or doc.get("type")
            if doc_type:
                stats["by_document_type"][str(doc_type)] += 1

            person = doc.get("person_name") or doc.get("person_id")
            if person:
                stats["by_person"][str(person)] += 1

        # Convert defaultdicts for JSON compatibility
        stats["by_year"] = dict(stats["by_year"])
        stats["by_collection"] = dict(stats["by_collection"])
        stats["by_document_type"] = dict(stats["by_document_type"])
        stats["by_person"] = dict(stats["by_person"])

        return stats

    def compute_object_stats(self, objects: List[DocumentObject]) -> Dict[str, Any]:
        """Compute stats for document objects actually sent to the LLM."""
        stats = {
            "docs_in_batch": len(objects),
            "by_year": defaultdict(int),
            "by_collection": defaultdict(int),
            "by_document_type": defaultdict(int),
            "by_person": defaultdict(int),
            "earliest_year": None,
            "latest_year": None,
        }

        for obj in objects:
            if isinstance(obj.year, int):
                stats["by_year"][str(obj.year)] += 1
                if stats["earliest_year"] is None or obj.year < stats["earliest_year"]:
                    stats["earliest_year"] = obj.year
                if stats["latest_year"] is None or obj.year > stats["latest_year"]:
                    stats["latest_year"] = obj.year

            if obj.collection:
                stats["by_collection"][str(obj.collection)] += 1

            if obj.document_type:
                stats["by_document_type"][str(obj.document_type)] += 1

            if obj.person_name:
                stats["by_person"][str(obj.person_name)] += 1

        stats["by_year"] = dict(stats["by_year"])
        stats["by_collection"] = dict(stats["by_collection"])
        stats["by_document_type"] = dict(stats["by_document_type"])
        stats["by_person"] = dict(stats["by_person"])

        return stats

    def _document_to_object(self, doc: Dict[str, Any]) -> DocumentObject:
        doc_id = str(doc.get("_id", "unknown"))
        blocks = self.chunker.chunk_document(doc_id, doc)
        return DocumentObject(
            doc_id=doc_id,
            filename=str(doc.get("filename", "Unknown")),
            year=doc.get("year"),
            document_type=str(doc.get("document_type") or doc.get("type") or "unknown"),
            person_name=doc.get("person_name"),
            collection=doc.get("collection"),
            metadata={
                "source_type": doc.get("source_type"),
                "person_id": doc.get("person_id"),
            },
            blocks=blocks,
        )
