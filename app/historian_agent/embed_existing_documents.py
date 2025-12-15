# 2025-12-15 13:46 America/New_York
# Purpose: One-time migration to chunk existing MongoDB documents, generate embeddings (via Ollama on host), store chunks, and index them in ChromaDB, with verbose logging and memory diagnostics.

"""
Migration Script: Embed Existing Documents (Ollama-first)

This script processes all existing documents in MongoDB:
1. Chunks documents intelligently
2. Generates vector embeddings (via Ollama embeddings API on the host)
3. Stores chunks in MongoDB collection
4. Indexes embeddings in vector store (ChromaDB)

Usage examples:
    python embed_existing_documents.py --batch 20 --provider ollama --model qwen3-embedding:0.6b --reset
    python embed_existing_documents.py --batch-size 20 --provider ollama --model qwen3-embedding:0.6b

Notes:
- Your provided Ollama URL is /api/generate, embeddings must use /api/embeddings.
  This script derives the embeddings endpoint automatically from OLLAMA_URL.
- Designed to avoid Linux Docker CPU PyTorch OOM by pushing embeddings to host Ollama.
"""

import argparse
import faulthandler
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests
from pymongo import MongoClient
from tqdm import tqdm

# Add app directory to Python path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

# Import RAG components
from chunking import DocumentChunker, Chunk
from embeddings import EmbeddingService  # kept for local/openai, we add ollama path below
from vector_store import get_vector_store


# -------------------------
# Configuration constants
# -------------------------
DEFAULT_MONGO_URI = (
    os.environ.get("APP_MONGO_URI")
    or os.environ.get("MONGO_URI")
    or "mongodb://admin:secret@mongodb:27017/admin"
)
DEFAULT_DB_NAME = "railroad_documents"

DEFAULT_BATCH_SIZE = 100
DEFAULT_PROVIDER = "ollama"

# Default to smaller Qwen3 embedding model
DEFAULT_MODEL = "qwen3-embedding:0.6b"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_VECTOR_STORE = "chroma"

DEFAULT_EMBED_SUBBATCH = 32
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 1.7

DEFAULT_CONTENT_FIELDS = ("title", "content", "ocr_text", "summary", "description")

# User-provided Ollama URL, note, this is the generate endpoint, embeddings uses /api/embeddings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")


# -------------------------
# Logging helpers
# -------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    log_file: str,
    log_level: str,
    json_logs: bool,
    noisy_libs: bool,
) -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    formatter: logging.Formatter
    if json_logs:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    if not noisy_libs:
        for lib in ["pymongo", "sentence_transformers", "transformers", "urllib3", "httpx", "chromadb", "requests"]:
            logging.getLogger(lib).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def get_rss_kb_linux() -> Optional[int]:
    """
    Best-effort RSS, in KiB, from /proc/self/status (Linux containers).
    """
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return _safe_int(parts[1], default=0)
    except Exception:
        return None
    return None


def fmt_bytes(kib: Optional[int]) -> str:
    if kib is None:
        return "n/a"
    b = kib * 1024
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if b < 1024 or unit == "TiB":
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} B"


@dataclass
class Timer:
    name: str
    logger: logging.Logger
    extra: Optional[Dict[str, Any]] = None
    start: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        self.logger.debug(f"START {self.name}", extra={"extra": self.extra or {}})
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        payload = dict(self.extra or {})
        payload["elapsed_s"] = round(elapsed, 6)
        if exc is None:
            self.logger.debug(f"END {self.name}", extra={"extra": payload})
        else:
            payload["exc_type"] = getattr(exc_type, "__name__", str(exc_type))
            self.logger.error(f"FAIL {self.name}: {exc}", exc_info=True, extra={"extra": payload})


def derive_ollama_embeddings_url(ollama_url: str) -> str:
    """
    User gave /api/generate. Embeddings must use /api/embeddings.
    We derive robustly without being fragile about trailing slashes.
    """
    u = (ollama_url or "").strip()
    if not u:
        u = "http://host.docker.internal:11434/api/generate"
    u = u.rstrip("/")
    if u.endswith("/api/generate"):
        return u[: -len("/api/generate")] + "/api/embeddings"
    if u.endswith("/api/embeddings"):
        return u
    # If they gave base like http://host:11434, append embeddings route
    if "/api/" not in u:
        return u + "/api/embeddings"
    # Some other api path, best effort replace last segment
    if u.endswith("/generate"):
        return u[: -len("/generate")] + "embeddings"
    return u + "/embeddings"


# -------------------------
# Ollama embedding client
# -------------------------
class OllamaEmbeddingClient:
    def __init__(
        self,
        embeddings_url: str,
        model: str,
        timeout_s: int,
        max_retries: int,
        backoff_base: float,
        logger: logging.Logger,
    ):
        self.embeddings_url = embeddings_url
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.logger = logger

        self._dimension: Optional[int] = None

        self.logger.info(f"Ollama embeddings configured: url={self.embeddings_url}, model={self.model}")

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Calls Ollama embeddings endpoint once per text.
        This is the most compatible behavior across Ollama versions.
        """
        vectors: List[List[float]] = []

        for i, text in enumerate(texts):
            payload = {"model": self.model, "prompt": text}
            attempt = 0

            while True:
                attempt += 1
                t0 = time.perf_counter()
                try:
                    r = requests.post(self.embeddings_url, json=payload, timeout=self.timeout_s)
                    dt = time.perf_counter() - t0

                    if r.status_code != 200:
                        body = r.text[:1000]
                        self.logger.error(
                            f"Ollama embeddings HTTP {r.status_code} item={i}/{len(texts)} time={dt:.3f}s body={body}"
                        )
                        raise RuntimeError(f"Ollama embeddings HTTP {r.status_code}")

                    data = r.json()
                    emb = data.get("embedding")
                    if emb is None or not isinstance(emb, list) or len(emb) == 0:
                        self.logger.error(f"Ollama returned empty embedding item={i}, keys={list(data.keys())}")
                        raise RuntimeError("Ollama returned empty embedding")

                    if self._dimension is None:
                        self._dimension = len(emb)
                        self.logger.info(f"Ollama embedding dimension inferred: {self._dimension}")

                    vectors.append(emb)
                    self.logger.debug(
                        f"Ollama embedded item={i+1}/{len(texts)} chars={len(text)} dim={len(emb)} time={dt:.3f}s"
                    )
                    break

                except Exception as e:
                    if attempt >= self.max_retries:
                        self.logger.error(
                            f"Ollama embedding failed after attempts={attempt} item={i}/{len(texts)}: {e}",
                            exc_info=True,
                        )
                        raise
                    sleep_s = self.backoff_base ** (attempt - 1)
                    self.logger.warning(
                        f"Ollama embedding attempt={attempt} failed item={i}/{len(texts)} err={e}, retry_in={sleep_s:.2f}s"
                    )
                    time.sleep(sleep_s)

        return vectors


# -------------------------
# Migration implementation
# -------------------------
class DocumentEmbeddingMigration:
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        batch_size: int,
        provider: str,
        model: str,
        chunk_size: int,
        chunk_overlap: int,
        vector_store_type: str,
        content_fields: Sequence[str],
        embed_subbatch: int,
        logger: logging.Logger,
        openai_api_key: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_timeout_s: int = DEFAULT_TIMEOUT_S,
        ollama_max_retries: int = DEFAULT_MAX_RETRIES,
        ollama_backoff_base: float = DEFAULT_BACKOFF_BASE,
    ):
        self.logger = logger
        self.batch_size = int(batch_size)
        self.embed_subbatch = max(1, int(embed_subbatch))
        self.content_fields = tuple(content_fields)

        self.provider = provider
        self.model = model

        self.logger.info("=" * 80)
        self.logger.info("Initializing Document Embedding Migration")
        self.logger.info("=" * 80)
        self.logger.info(
            f"Config: db={db_name}, provider={provider}, model={model}, doc_batch={self.batch_size}, "
            f"embed_subbatch={self.embed_subbatch}, chunk_size={chunk_size}, overlap={chunk_overlap}, "
            f"vector_store={vector_store_type}, fields={list(self.content_fields)}, rss={fmt_bytes(get_rss_kb_linux())}"
        )

        # MongoDB
        with Timer("mongo_connect", self.logger, {"mongo_uri_prefix": mongo_uri[:60], "db": db_name}):
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.documents_collection = self.db["documents"]
            self.chunks_collection = self.db["document_chunks"]

        self._create_indexes()

        # Chunker
        with Timer("init_chunker", self.logger, {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}):
            self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Embeddings provider
        self.embedding_service: Optional[EmbeddingService] = None
        self.ollama_client: Optional[OllamaEmbeddingClient] = None

        if provider == "ollama":
            emb_url = derive_ollama_embeddings_url(ollama_url or OLLAMA_URL)
            self.ollama_client = OllamaEmbeddingClient(
                embeddings_url=emb_url,
                model=model,
                timeout_s=ollama_timeout_s,
                max_retries=ollama_max_retries,
                backoff_base=ollama_backoff_base,
                logger=self.logger,
            )
        else:
            # Keep legacy behavior for local/openai
            with Timer("init_embedding_service", self.logger, {"provider": provider, "model": model}):
                self.embedding_service = EmbeddingService(
                    provider=provider,
                    model=model,
                    api_key=openai_api_key,
                    batch_size=self.embed_subbatch,
                )

        # Vector store
        with Timer("init_vector_store", self.logger, {"type": vector_store_type}):
            self.vector_store = get_vector_store(
                store_type=vector_store_type,
                collection=self.chunks_collection if vector_store_type == "mongo" else None,
            )

        self.logger.info(f"Initialization complete, rss={fmt_bytes(get_rss_kb_linux())}")

    def _create_indexes(self) -> None:
        try:
            with Timer("create_indexes", self.logger):
                self.chunks_collection.create_index("document_id")
                self.chunks_collection.create_index("chunk_id", unique=True)
                self.chunks_collection.create_index([("text", "text")])
            self.logger.info("Indexes ensured on document_chunks collection")
        except Exception as e:
            self.logger.warning(f"Index creation issue (may already exist): {e}", exc_info=True)

    def reset_vector_store(self) -> None:
        if hasattr(self.vector_store, "reset"):
            self.vector_store.reset()
        else:
            raise RuntimeError("Vector store does not support reset()")

    def run(self, skip_existing: bool = True, limit: Optional[int] = None) -> Dict[str, Any]:
        self.logger.info("=" * 80)
        self.logger.info("Starting document embedding migration")
        self.logger.info("=" * 80)

        start_time = datetime.now()
        stats: Dict[str, Any] = {
            "total_documents": 0,
            "skipped": 0,
            "processed": 0,
            "total_chunks": 0,
            "failed": 0,
            "errors": [],
        }

        query: Dict[str, Any] = {}
        if skip_existing:
            with Timer("load_existing_doc_ids", self.logger):
                existing_parent_ids = set(self.chunks_collection.distinct("document_id"))
            if existing_parent_ids:
                query["_id"] = {"$nin": list(existing_parent_ids)}
                self.logger.info(f"Skipping {len(existing_parent_ids)} documents that already have chunks")

        total_docs = self.documents_collection.count_documents(query)
        stats["total_documents"] = total_docs

        if limit:
            total_docs = min(total_docs, limit)
            self.logger.info(f"Limiting processing to {limit} documents")

        self.logger.info(f"Found {total_docs} documents to process")
        if total_docs == 0:
            return stats

        projection = {"_id": 1}
        for f in self.content_fields:
            projection[f] = 1

        cursor = (
            self.documents_collection.find(query, projection=projection)
            .sort("_id", 1)
            .limit(limit or 0)
        )

        processed_seen = 0
        with tqdm(total=total_docs, desc="Processing documents", unit="docs") as pbar:
            batch: List[Dict[str, Any]] = []
            for document in cursor:
                batch.append(document)
                if len(batch) >= self.batch_size:
                    batch_stats = self._process_batch(batch, batch_index=(processed_seen // self.batch_size))
                    self._update_stats(stats, batch_stats)
                    pbar.update(len(batch))
                    processed_seen += len(batch)
                    batch = []

                    self.logger.info(
                        f"Heartbeat docs_seen={processed_seen} rss={fmt_bytes(get_rss_kb_linux())} "
                        f"processed={stats['processed']} failed={stats['failed']} chunks={stats['total_chunks']}"
                    )

            if batch:
                batch_stats = self._process_batch(batch, batch_index=(processed_seen // self.batch_size))
                self._update_stats(stats, batch_stats)
                pbar.update(len(batch))
                processed_seen += len(batch)

        duration = datetime.now() - start_time
        stats["duration_seconds"] = duration.total_seconds()
        stats["avg_time_per_doc"] = duration.total_seconds() / stats["processed"] if stats["processed"] else 0.0

        self.logger.info("=" * 80)
        self.logger.info("Migration Complete")
        self.logger.info("=" * 80)
        self.logger.info(f"Total documents matched: {stats['total_documents']}")
        self.logger.info(f"Processed: {stats['processed']}")
        self.logger.info(f"Skipped: {stats['skipped']}")
        self.logger.info(f"Failed: {stats['failed']}")
        self.logger.info(f"Total chunks created: {stats['total_chunks']}")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Average: {stats['avg_time_per_doc']:.4f}s per processed document")
        self.logger.info(f"Final rss: {fmt_bytes(get_rss_kb_linux())}")

        if stats["errors"]:
            self.logger.warning(f"Encountered {len(stats['errors'])} errors, see log for details")

        return stats

    def _process_batch(self, documents: List[Dict[str, Any]], batch_index: int) -> Dict[str, Any]:
        batch_stats: Dict[str, Any] = {
            "processed": 0,
            "skipped": 0,
            "total_chunks": 0,
            "failed": 0,
            "errors": [],
        }

        self.logger.info(f"Batch start index={batch_index} doc_count={len(documents)} rss={fmt_bytes(get_rss_kb_linux())}")

        all_chunks: List[Chunk] = []

        # Step 1, chunking
        with Timer("batch_chunking", self.logger, {"batch_index": batch_index, "docs": len(documents)}):
            for doc_i, document in enumerate(documents):
                doc_id = str(document.get("_id"))
                try:
                    field_sizes = {}
                    for f in self.content_fields:
                        v = document.get(f)
                        field_sizes[f] = len(v) if isinstance(v, str) else (0 if v is None else len(str(v)))

                    self.logger.debug(
                        f"Chunking doc={doc_i+1}/{len(documents)} id={doc_id} field_sizes={field_sizes}",
                        extra={"extra": {"doc_id": doc_id, "field_sizes": field_sizes, "batch_index": batch_index}},
                    )

                    chunks = self.chunker.chunk_document(document, content_fields=self.content_fields)

                    if not chunks:
                        self.logger.warning(f"Document {doc_id} produced no chunks")
                        batch_stats["skipped"] += 1
                        continue

                    all_chunks.extend(chunks)
                    batch_stats["processed"] += 1
                    batch_stats["total_chunks"] += len(chunks)

                except Exception as e:
                    self.logger.error(f"Error chunking document {doc_id}: {e}", exc_info=True)
                    batch_stats["failed"] += 1
                    batch_stats["errors"].append({"doc_id": doc_id, "error": str(e), "stage": "chunking"})

        if not all_chunks:
            self.logger.info(f"Batch end index={batch_index}, no chunks, skipped={batch_stats['skipped']}, failed={batch_stats['failed']}")
            return batch_stats

        chunk_lengths = [len(getattr(c, "content", "") or "") for c in all_chunks]
        self.logger.info(
            f"Chunk stats index={batch_index} docs_ok={batch_stats['processed']} chunks={len(all_chunks)} "
            f"chars_total={sum(chunk_lengths)} chars_min={min(chunk_lengths)} "
            f"chars_p50={sorted(chunk_lengths)[len(chunk_lengths)//2]} chars_max={max(chunk_lengths)} "
            f"rss={fmt_bytes(get_rss_kb_linux())}"
        )

        # Step 2, embeddings, sub-batched
        embeddings: List[Any] = []
        with Timer("batch_embedding", self.logger, {"batch_index": batch_index, "chunks": len(all_chunks), "provider": self.provider}):
            try:
                texts = [c.content for c in all_chunks]
                n = len(texts)

                for start in range(0, n, self.embed_subbatch):
                    end = min(start + self.embed_subbatch, n)
                    sub = texts[start:end]
                    sub_chars = sum(len(s) for s in sub)

                    self.logger.info(
                        f"Embedding subbatch index={batch_index} range={start}:{end} count={len(sub)} chars={sub_chars} rss={fmt_bytes(get_rss_kb_linux())}"
                    )

                    with Timer("embed_subbatch", self.logger, {"batch_index": batch_index, "start": start, "end": end, "count": len(sub), "chars": sub_chars}):
                        if self.provider == "ollama":
                            if not self.ollama_client:
                                raise RuntimeError("Ollama client not initialized")
                            sub_emb = self.ollama_client.embed_texts(sub)
                        else:
                            if not self.embedding_service:
                                raise RuntimeError("EmbeddingService not initialized")
                            sub_emb = self.embedding_service.embed_documents(sub, show_progress=False)

                    if sub_emb is None or len(sub_emb) != len(sub):
                        raise RuntimeError(f"Embedding count mismatch: got {0 if sub_emb is None else len(sub_emb)} for {len(sub)}")

                    embeddings.extend(sub_emb)

                if len(embeddings) != len(all_chunks):
                    raise RuntimeError(f"Total embedding mismatch: got {len(embeddings)} for {len(all_chunks)} chunks")

                # Attach embeddings
                for chunk, emb in zip(all_chunks, embeddings):
                    chunk.embedding = emb

                if self.provider == "ollama" and self.ollama_client and self.ollama_client.dimension is not None:
                    self.logger.info(f"Ollama dimension={self.ollama_client.dimension} (default for model={self.model})")

                self.logger.info(f"Embeddings complete index={batch_index} embeddings={len(embeddings)} rss={fmt_bytes(get_rss_kb_linux())}")

            except Exception as e:
                self.logger.error(f"Error generating embeddings index={batch_index}: {e}", exc_info=True)
                batch_stats["errors"].append({"error": str(e), "stage": "embedding", "batch_index": batch_index})
                batch_stats["failed"] += batch_stats["processed"]
                batch_stats["processed"] = 0
                return batch_stats

        # Step 3, Mongo insert
        with Timer("batch_mongo_insert", self.logger, {"batch_index": batch_index, "chunks": len(all_chunks)}):
            try:
                chunk_dicts = [c.to_dict() for c in all_chunks]
                if chunk_dicts:
                    self.chunks_collection.insert_many(chunk_dicts, ordered=False)
                self.logger.info(f"Mongo insert complete index={batch_index} inserted={len(chunk_dicts)}")
            except Exception as e:
                self.logger.error(f"Error inserting chunks into MongoDB index={batch_index}: {e}", exc_info=True)
                batch_stats["errors"].append({"error": str(e), "stage": "mongodb_insert", "batch_index": batch_index})

        # Step 4, vector store add
        with Timer("batch_vector_store_add", self.logger, {"batch_index": batch_index, "chunks": len(all_chunks)}):
            try:
                self.vector_store.add_chunks(all_chunks)
                self.logger.info(f"Vector store add complete index={batch_index} added={len(all_chunks)}")
            except Exception as e:
                self.logger.error(f"Error adding chunks to vector store index={batch_index}: {e}", exc_info=True)
                batch_stats["errors"].append({"error": str(e), "stage": "vector_store", "batch_index": batch_index})

        self.logger.info(f"Batch end index={batch_index} rss={fmt_bytes(get_rss_kb_linux())}")
        return batch_stats

    def _update_stats(self, stats: Dict[str, Any], batch_stats: Dict[str, Any]) -> None:
        stats["processed"] += batch_stats["processed"]
        stats["skipped"] += batch_stats["skipped"]
        stats["total_chunks"] += batch_stats["total_chunks"]
        stats["failed"] += batch_stats["failed"]
        stats["errors"].extend(batch_stats["errors"])


# -------------------------
# Signal diagnostics
# -------------------------
def install_crash_handlers(logger: logging.Logger) -> None:
    faulthandler.enable(all_threads=True)

    def _handler(sig, frame):
        logger.critical(f"Received signal {sig}, dumping traceback", extra={"extra": {"signal": sig}})
        try:
            faulthandler.dump_traceback(all_threads=True)
        except Exception:
            logger.critical("Failed dumping traceback", exc_info=True)

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _handler)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate existing documents to chunked+embedded format")

    parser.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI, help="MongoDB connection URI")
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME, help="Database name")

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Documents per batch")
    parser.add_argument("--batch", dest="batch_size", type=int, help="Alias for --batch-size")

    parser.add_argument("--provider", choices=["ollama", "local", "openai"], default=DEFAULT_PROVIDER, help="Embedding provider")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")

    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlap between chunks")
    parser.add_argument("--vector-store", choices=["chroma", "mongo"], default=DEFAULT_VECTOR_STORE, help="Vector store type")

    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip documents that already have chunks")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Process all documents (reprocess existing)")
    parser.add_argument("--limit", type=int, help="Limit number of documents (for testing)")
    parser.add_argument("--reset", action="store_true", help="Reset vector store before migration (deletes all existing data)")

    parser.add_argument("--embed-subbatch", type=int, default=DEFAULT_EMBED_SUBBATCH, help="Embedding subbatch size")
    parser.add_argument("--content-fields", nargs="+", default=list(DEFAULT_CONTENT_FIELDS), help="Fields to concatenate for chunking")

    # Ollama specifics
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help="Ollama URL, you can pass /api/generate, we derive /api/embeddings")

    # Logging
    parser.add_argument("--log-file", default="embed_migration.ollama.log", help="Log file path")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"), help="Log level, e.g. DEBUG, INFO")
    parser.add_argument("--json-logs", action="store_true", help="Emit JSON log lines")
    parser.add_argument("--noisy-libs", action="store_true", help="Allow third-party libs to log more")

    args = parser.parse_args()

    logger = setup_logging(
        log_file=args.log_file,
        log_level=args.log_level,
        json_logs=args.json_logs,
        noisy_libs=args.noisy_libs,
    )
    install_crash_handlers(logger)

    logger.info("Starting migration with parameters:")
    logger.info(f"  MongoDB URI: {str(args.mongo_uri)[:60]}...")
    logger.info(f"  Database: {args.db_name}")
    logger.info(f"  Provider: {args.provider}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Doc batch size: {args.batch_size}")
    logger.info(f"  Embed subbatch: {args.embed_subbatch}")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  Chunk overlap: {args.chunk_overlap}")
    logger.info(f"  Vector store: {args.vector_store}")
    logger.info(f"  Skip existing: {args.skip_existing}")
    logger.info(f"  Content fields: {args.content_fields}")
    logger.info(f"  Ollama URL: {args.ollama_url}")
    logger.info(f"  Derived embeddings URL: {derive_ollama_embeddings_url(args.ollama_url)}")
    logger.info(f"  RSS at start: {fmt_bytes(get_rss_kb_linux())}")


    


    try:
        migration = DocumentEmbeddingMigration(
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            batch_size=args.batch_size,
            provider=args.provider,
            model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            vector_store_type=args.vector_store,
            content_fields=args.content_fields,
            embed_subbatch=args.embed_subbatch,
            logger=logger,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            ollama_url=args.ollama_url,
        )
    except Exception as e:
        logger.error(f"Failed to initialize migration: {e}", exc_info=True)
        sys.exit(1)

    if args.reset:
        logger.warning("RESETTING VECTOR STORE, all existing embeddings will be deleted")
        try:
            migration.reset_vector_store()
            logger.info("Vector store reset complete")
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}", exc_info=True)
            sys.exit(1)


    # Preflight: verify Ollama model is available BEFORE processing docs
    try:
        emb = migration.ollama_client.embed_texts(["preflight test"])
        logger.info(f"Ollama preflight OK, dim={len(emb[0])}")
    except Exception as e:
        logger.error(
            "Ollama preflight FAILED. Most likely the model is not pulled on the Ollama host.\n"
            "Fix: run `ollama pull qwen3-embedding:0.6b` on the host, then rerun.\n"
            f"Error: {e}",
            exc_info=True,
        )
        sys.exit(2)


    try:
        stats = migration.run(skip_existing=args.skip_existing, limit=args.limit)
        exit_code = 0 if stats.get("failed", 0) == 0 else 1
        logger.info(f"Migration exiting with code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user (Ctrl+C)")
        logger.info("Progress has been saved, you can resume with --skip-existing")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed with unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
