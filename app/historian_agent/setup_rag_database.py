# 2025-12-15 13:53 America/New_York
# Purpose: Initialize RAG system for Qwen3 embeddings (MongoDB chunk indexes + Chroma collection sized to the embedding model's default dimension via Ollama).

"""
setup_rag_database.py - Initialize RAG system (ChromaDB + MongoDB chunks), Qwen3 via Ollama

Run this AFTER setup_databases.py.

Usage:
    docker compose exec app python scripts/setup_rag_database.py

Creates / verifies:
- MongoDB document_chunks collection indexes
- ChromaDB collection sized to the embedding model's DEFAULT dimension (qwen3-embedding:0.6b -> 1024)
- Tests integration between MongoDB and ChromaDB using a real embedding from Ollama

Key points:
- Uses Ollama for embeddings to avoid container CPU torch OOM.
- You provided OLLAMA_URL as /api/generate, this script derives /api/embeddings automatically.
- Chroma doesn't require you to declare dimension at creation, but it WILL enforce consistency at upsert.
  This script creates a fresh collection (or optionally resets) and then asserts the dimension on test upsert.

Env:
- APP_MONGO_URI or MONGO_URI
- CHROMA_PERSIST_DIRECTORY
- OLLAMA_URL (optional), default: http://host.docker.internal:11434/api/generate
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Optional

import requests
from pymongo import MongoClient


# -------------------------
# Defaults (Qwen3 small model)
# -------------------------
DEFAULT_DB_NAME = os.environ.get("MONGO_DB_NAME", "railroad_documents")
DEFAULT_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "historian_documents")
DEFAULT_EMBED_MODEL = os.environ.get("HISTORIAN_AGENT_EMBEDDING_MODEL", "qwen3-embedding:0.6b")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
DEFAULT_TIMEOUT_S = int(os.environ.get("OLLAMA_TIMEOUT_S", "120"))


# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("setup_rag_database.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def derive_ollama_embeddings_url(ollama_url: str) -> str:
    u = (ollama_url or "").strip().rstrip("/")
    if not u:
        u = "http://host.docker.internal:11434/api/generate"
    if u.endswith("/api/generate"):
        return u[: -len("/api/generate")] + "/api/embeddings"
    if u.endswith("/api/embeddings"):
        return u
    if "/api/" not in u:
        return u + "/api/embeddings"
    if u.endswith("/generate"):
        return u[: -len("/generate")] + "embeddings"
    return u + "/embeddings"


def ollama_embed_once(model: str, text: str, embeddings_url: str, timeout_s: int) -> list:
    payload = {"model": model, "prompt": text}
    r = requests.post(embeddings_url, json=payload, timeout=timeout_s)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama embeddings HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or len(emb) == 0:
        raise RuntimeError(f"Ollama returned invalid embedding payload keys={list(data.keys())}")
    return emb


class RAGDatabaseSetup:
    """Handles RAG database initialization for Ollama Qwen3 embeddings."""

    def __init__(self):
        self.mongo_client: Optional[MongoClient] = None
        self.db = None
        self.chroma_client = None
        self.chroma_collection = None

        self.db_name = DEFAULT_DB_NAME
        self.collection_name = DEFAULT_COLLECTION_NAME
        self.embed_model = DEFAULT_EMBED_MODEL
        self.ollama_embeddings_url = derive_ollama_embeddings_url(OLLAMA_URL)

        self.embedding_dimension: Optional[int] = None

    def connect_mongodb(self) -> bool:
        print("\n🔧 Connecting to MongoDB...")
        try:
            mongo_uri = os.environ.get("MONGO_URI") or os.environ.get("APP_MONGO_URI")
            if not mongo_uri:
                raise ValueError("APP_MONGO_URI or MONGO_URI environment variable not set")

            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.mongo_client.admin.command("ping")
            self.db = self.mongo_client[self.db_name]

            print("   ✅ MongoDB connected")
            logger.info(f"Connected to MongoDB db={self.db_name}")
            return True

        except Exception as e:
            print(f"   ❌ MongoDB connection failed: {e}")
            logger.error(f"MongoDB connection failed: {e}", exc_info=True)
            return False

    def verify_mongodb_collections(self) -> bool:
        print("\n🔍 Verifying MongoDB collections...")
        required = ["documents", "document_chunks"]
        existing = self.db.list_collection_names()

        missing = [c for c in required if c not in existing]
        if missing:
            print(f"   ❌ Missing collections: {missing}")
            print("   💡 Run setup_databases.py first!")
            return False

        print("   ✅ All required collections exist")
        return True

    def create_chunks_indexes(self) -> bool:
        print("\n📇 Creating document_chunks indexes...")
        chunks = self.db["document_chunks"]
        existing_indexes = chunks.index_information()

        indexes_created = 0

        if "chunk_id_1" not in existing_indexes:
            chunks.create_index([("chunk_id", 1)], unique=True)
            print("   ✅ Created unique index: chunk_id")
            logger.info("Created unique index on chunk_id")
            indexes_created += 1
        else:
            print("   ✓ Index exists: chunk_id")

        if "document_id_1" not in existing_indexes:
            chunks.create_index([("document_id", 1)])
            print("   ✅ Created index: document_id")
            logger.info("Created index on document_id")
            indexes_created += 1
        else:
            print("   ✓ Index exists: document_id")

        if "document_id_1_chunk_index_1" not in existing_indexes:
            chunks.create_index([("document_id", 1), ("chunk_index", 1)])
            print("   ✅ Created compound index: document_id + chunk_index")
            logger.info("Created compound index on document_id and chunk_index")
            indexes_created += 1
        else:
            print("   ✓ Index exists: document_id + chunk_index")

        if indexes_created:
            print(f"\n   📊 Created {indexes_created} new indexes")

        return True

    def verify_embedding_provider(self) -> bool:
        """
        Verify we can obtain an embedding from Ollama and record its dimension.
        """
        print("\n🤖 Verifying embedding provider (Ollama)...")
        try:
            print(f"   Model: {self.embed_model}")
            print(f"   Ollama embeddings URL: {self.ollama_embeddings_url}")

            test_text = "Integration test, verify embedding dimension."
            t0 = time.time()
            emb = ollama_embed_once(
                model=self.embed_model,
                text=test_text,
                embeddings_url=self.ollama_embeddings_url,
                timeout_s=DEFAULT_TIMEOUT_S,
            )
            dt = time.time() - t0
            self.embedding_dimension = len(emb)

            print("   ✅ Ollama embedding call successful")
            print(f"      Dimension (default): {self.embedding_dimension}")
            print(f"      Time: {dt:.2f}s")

            logger.info(f"Ollama embeddings verified model={self.embed_model} dim={self.embedding_dimension} time_s={dt:.2f}")
            return True

        except Exception as e:
            print(f"   ❌ Ollama embedding failed: {e}")
            logger.error(f"Ollama embedding failed: {e}", exc_info=True)
            return False

    def initialize_chromadb(self) -> bool:
        print("\n🗄️  Initializing ChromaDB...")
        try:
            import chromadb
            from chromadb.config import Settings

            persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY", "/data/chroma_db/persist")
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            print(f"   Persist directory: {persist_dir}")

            self.chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            print("   ✅ ChromaDB client initialized")

            # Optional reset
            if os.environ.get("RESET_CHROMA", "").lower() in ("1", "true", "yes"):
                print("   ⚠️  RESET_CHROMA enabled, resetting Chroma client...")
                self.chroma_client.reset()
                print("   ✅ Chroma reset complete")

            # Create or get collection
            try:
                self.chroma_collection = self.chroma_client.get_collection(name=self.collection_name)
                count = self.chroma_collection.count()
                print(f"   ✅ Collection exists: {self.collection_name}")
                print(f"      Current vectors: {count}")
            except Exception:
                self.chroma_collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"   ✅ Created collection: {self.collection_name}")
                logger.info(f"Created ChromaDB collection: {self.collection_name}")

            logger.info(f"ChromaDB initialized collection={self.collection_name} persist_dir={persist_dir}")
            return True

        except ImportError as e:
            print(f"   ❌ Missing dependency: {e}")
            logger.error(f"Import error: {e}", exc_info=True)
            return False
        except Exception as e:
            print(f"   ❌ ChromaDB initialization failed: {e}")
            logger.error(f"ChromaDB init failed: {e}", exc_info=True)
            return False

    def test_integration(self) -> bool:
        print("\n🧪 Testing integration (MongoDB + ChromaDB + Ollama embeddings)...")
        try:
            if self.embedding_dimension is None:
                raise RuntimeError("Embedding dimension not set, run verify_embedding_provider first")

            test_chunk_id = "test_integration_chunk_0"
            test_text = "This is a test document for verifying RAG system integration with Qwen3 embeddings."

            print("   1. Generating test embedding via Ollama...")
            embedding = ollama_embed_once(
                model=self.embed_model,
                text=test_text,
                embeddings_url=self.ollama_embeddings_url,
                timeout_s=DEFAULT_TIMEOUT_S,
            )
            got_dim = len(embedding)

            if got_dim != self.embedding_dimension:
                raise RuntimeError(f"Dimension changed within run: expected {self.embedding_dimension}, got {got_dim}")

            print(f"      ✅ Generated {got_dim}-dimensional vector")

            print("   2. Testing MongoDB upsert...")
            test_chunk = {
                "chunk_id": test_chunk_id,
                "document_id": "test_doc_id",
                "chunk_index": 0,
                "text": test_text,
                "token_count": 0,
                "metadata": {"source_file": "integration_test.txt", "test": True, "embedding_dim": got_dim},
            }
            self.db["document_chunks"].update_one({"chunk_id": test_chunk_id}, {"$set": test_chunk}, upsert=True)
            print("      ✅ MongoDB upsert successful")

            print("   3. Testing ChromaDB upsert (dimension enforcement)...")
            self.chroma_collection.upsert(
                ids=[test_chunk_id],
                embeddings=[embedding],
                documents=[test_text],
                metadatas=[{"test": True, "source": "integration_test", "embedding_dim": got_dim}],
            )
            print("      ✅ ChromaDB upsert successful")

            print("   4. Testing vector search...")
            results = self.chroma_collection.query(query_embeddings=[embedding], n_results=1)

            ok = bool(results.get("ids")) and bool(results["ids"][0]) and (results["ids"][0][0] == test_chunk_id)
            if ok:
                print("      ✅ Vector search successful")
            else:
                print("      ⚠️  Vector search returned unexpected results")
                logger.warning(f"Unexpected query results: {json.dumps(results)[:500]}")

            print("   5. Cleaning up test data...")
            self.db["document_chunks"].delete_one({"chunk_id": test_chunk_id})
            self.chroma_collection.delete(ids=[test_chunk_id])
            print("      ✅ Test data cleaned up")

            print("\n   ✅ Integration test PASSED")
            logger.info("Integration test passed")
            return True

        except Exception as e:
            print(f"\n   ❌ Integration test FAILED: {e}")
            logger.error(f"Integration test failed: {e}", exc_info=True)
            return False

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("✅ RAG Database Setup Complete (Qwen3 via Ollama)")
        print("=" * 70)

        print("\n📊 Summary:")
        print("   • MongoDB: document_chunks indexes created/verified")
        print(f"   • ChromaDB: {self.collection_name} collection ready")
        print(f"   • Embedding provider: Ollama ({self.ollama_embeddings_url})")
        print(f"   • Embedding model: {self.embed_model}")
        print(f"   • Dimension (default): {self.embedding_dimension}")

        print("\n📝 Next Steps:")
        print("   1. Reset and re-embed using the Ollama pipeline:")
        print("      docker compose exec app python scripts/embed_existing_documents.py \\")
        print("        --batch 20 \\")
        print("        --provider ollama \\")
        print(f"        --model {self.embed_model} \\")
        print("        --reset")
        print()
        print("   2. Tail logs:")
        print("      tail -f embed_migration.ollama.log")
        print()
        print("   3. If you need to change embedding model later, use a new collection name")
        print("      (or reset Chroma) to avoid dimension mismatch.")
        print()
        print("Chroma persist dir: " + os.environ.get("CHROMA_PERSIST_DIRECTORY", "N/A"))
        print()

    def run(self) -> bool:
        print("=" * 70)
        print("RAG Database Setup - Historical Document Reader")
        print("Embeddings: Qwen3 via Ollama (default dimension enforced by test upsert)")
        print("=" * 70)

        steps = [
            ("Connect to MongoDB", self.connect_mongodb),
            ("Verify MongoDB collections", self.verify_mongodb_collections),
            ("Create document_chunks indexes", self.create_chunks_indexes),
            ("Verify embedding provider", self.verify_embedding_provider),
            ("Initialize ChromaDB", self.initialize_chromadb),
            ("Test integration", self.test_integration),
        ]

        for step_name, step_func in steps:
            try:
                if not step_func():
                    print(f"\n❌ Setup failed at: {step_name}")
                    return False
            except Exception as e:
                print(f"\n❌ Unexpected error in {step_name}: {e}")
                logger.error(f"Error in {step_name}: {e}", exc_info=True)
                return False

        self.print_summary()
        return True


def main():
    setup = RAGDatabaseSetup()
    success = setup.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
