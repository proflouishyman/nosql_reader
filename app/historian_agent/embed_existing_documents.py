"""
Migration Script: Embed Existing Documents

This script processes all existing documents in MongoDB:
1. Chunks documents intelligently
2. Generates vector embeddings
3. Stores chunks in MongoDB collection
4. Indexes embeddings in vector store (ChromaDB)

Usage:
    python embed_existing_documents.py --batch-size 100 --provider local

This is a ONE-TIME migration that should be run after initial deployment.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional
from pathlib import Path

# Add app directory to Python path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from pymongo import MongoClient
from tqdm import tqdm

# Import RAG components
from chunking import DocumentChunker, Chunk
from embeddings import EmbeddingService
from vector_store import get_vector_store

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embed_migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_MONGO_URI = os.environ.get('APP_MONGO_URI') or os.environ.get('MONGO_URI') or "mongodb://admin:secret@mongodb:27017/admin"
DEFAULT_DB_NAME = 'railroad_documents'
DEFAULT_BATCH_SIZE = 100
DEFAULT_PROVIDER = 'local'
DEFAULT_MODEL = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_VECTOR_STORE = 'chroma'


class DocumentEmbeddingMigration:
    """Handles migration of existing documents to chunked+embedded format."""
    
    def __init__(
        self,
        mongo_uri: str,
        db_name: str = DEFAULT_DB_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_provider: str = DEFAULT_PROVIDER,
        embedding_model: str = DEFAULT_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        openai_api_key: Optional[str] = None,
        vector_store_type: str = DEFAULT_VECTOR_STORE,
    ):
        """
        Initialize migration.
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
            batch_size: Number of documents to process at once
            embedding_provider: 'local' or 'openai'
            embedding_model: Model name for embeddings
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            openai_api_key: OpenAI API key (if using OpenAI)
            vector_store_type: 'chroma' or 'mongo'
        """
        logger.info("="*60)
        logger.info("Initializing Document Embedding Migration")
        logger.info("="*60)
        
        # MongoDB setup
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.documents_collection = self.db["documents"]
        self.chunks_collection = self.db["document_chunks"]
        
        # Create indexes on chunks collection
        self._create_indexes()
        
        # Initialize RAG components
        logger.info(f"Initializing document chunker (size={chunk_size}, overlap={chunk_overlap})")
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        logger.info(f"Initializing embedding service (provider={embedding_provider}, model={embedding_model})")
        self.embedding_service = EmbeddingService(
            provider=embedding_provider,
            model=embedding_model,
            api_key=openai_api_key,
            batch_size=batch_size,
        )
        
        logger.info(f"Initializing vector store (type={vector_store_type})")
        self.vector_store = get_vector_store(
            store_type=vector_store_type,
            collection=self.chunks_collection if vector_store_type == "mongo" else None,
        )
        
        self.batch_size = batch_size
        
        logger.info(
            f"Migration initialized successfully: "
            f"db={db_name}, batch_size={batch_size}, "
            f"embeddings={embedding_provider}/{embedding_model}, "
            f"vector_store={vector_store_type}"
        )
    
    def _create_indexes(self):
        """Create necessary indexes on chunks collection."""
        try:
            # Index on document_id for fast lookup
            self.chunks_collection.create_index("document_id")
            # Index on chunk_id for fast lookup
            self.chunks_collection.create_index("chunk_id", unique=True)
            # Text index on text field for keyword search
            self.chunks_collection.create_index([("text", "text")])
            logger.info("Created indexes on document_chunks collection")
        except Exception as e:
            logger.warning(f"Error creating indexes (may already exist): {e}")
    
    def run(
        self,
        skip_existing: bool = True,
        limit: Optional[int] = None,
    ) -> dict:
        """
        Run the migration.
        
        Args:
            skip_existing: Skip documents that already have chunks
            limit: Maximum number of documents to process (for testing)
            
        Returns:
            Dictionary with migration statistics
        """
        logger.info("=" * 60)
        logger.info("Starting document embedding migration")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        stats = {
            "total_documents": 0,
            "skipped": 0,
            "processed": 0,
            "total_chunks": 0,
            "failed": 0,
            "errors": [],
        }
        
        # Get documents to process
        query = {}
        if skip_existing:
            # Find documents that don't have chunks yet
            existing_parent_ids = set(
                self.chunks_collection.distinct("document_id")
            )
            if existing_parent_ids:
                query["_id"] = {"$nin": list(existing_parent_ids)}
                logger.info(f"Skipping {len(existing_parent_ids)} documents that already have chunks")
        
        total_docs = self.documents_collection.count_documents(query)
        stats["total_documents"] = total_docs
        
        if limit:
            total_docs = min(total_docs, limit)
            logger.info(f"Limiting processing to {limit} documents")
        
        logger.info(f"Found {total_docs} documents to process")
        
        if total_docs == 0:
            logger.info("No documents to process")
            return stats
        
        # Process in batches
        cursor = self.documents_collection.find(query).limit(limit or 0)
        
        with tqdm(total=total_docs, desc="Processing documents", unit="docs") as pbar:
            batch = []
            
            for document in cursor:
                batch.append(document)
                
                if len(batch) >= self.batch_size:
                    batch_stats = self._process_batch(batch)
                    self._update_stats(stats, batch_stats)
                    pbar.update(len(batch))
                    batch = []
            
            # Process remaining documents
            if batch:
                batch_stats = self._process_batch(batch)
                self._update_stats(stats, batch_stats)
                pbar.update(len(batch))
        
        # Calculate duration
        duration = datetime.now() - start_time
        stats["duration_seconds"] = duration.total_seconds()
        stats["avg_time_per_doc"] = (
            duration.total_seconds() / stats["processed"]
            if stats["processed"] > 0
            else 0
        )
        
        # Log summary
        logger.info("=" * 60)
        logger.info("Migration Complete!")
        logger.info("=" * 60)
        logger.info(f"Total documents: {stats['total_documents']}")
        logger.info(f"Processed: {stats['processed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total chunks created: {stats['total_chunks']}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Average: {stats['avg_time_per_doc']:.2f}s per document")
        
        if stats["errors"]:
            logger.warning(f"Encountered {len(stats['errors'])} errors")
            logger.info("Check embed_migration.log for details")
        
        return stats
    
    def _process_batch(self, documents: List[dict]) -> dict:
        """Process a batch of documents."""
        batch_stats = {
            "processed": 0,
            "skipped": 0,
            "total_chunks": 0,
            "failed": 0,
            "errors": [],
        }
        
        all_chunks = []
        
        # Step 1: Chunk all documents in batch
        logger.debug(f"Chunking {len(documents)} documents...")
        for document in documents:
            doc_id = str(document["_id"])
            
            try:
                # Chunk document - use content_fields parameter name
                chunks = self.chunker.chunk_document(
                    document,
                    content_fields=("title", "content", "ocr_text", "summary")
                )
                
                if not chunks:
                    logger.warning(f"Document {doc_id} produced no chunks")
                    batch_stats["skipped"] += 1
                    continue
                
                all_chunks.extend(chunks)
                batch_stats["processed"] += 1
                batch_stats["total_chunks"] += len(chunks)
                
            except Exception as e:
                logger.error(f"Error chunking document {doc_id}: {e}", exc_info=True)
                batch_stats["failed"] += 1
                batch_stats["errors"].append({
                    "doc_id": doc_id,
                    "error": str(e),
                    "stage": "chunking"
                })
        
        if not all_chunks:
            return batch_stats
        
        # Step 2: Generate embeddings for all chunks
        try:
            logger.debug(f"Generating embeddings for {len(all_chunks)} chunks...")
            
            # Extract text from chunks (use .content property)
            chunk_texts = [chunk.content for chunk in all_chunks]
            
            # Generate embeddings
            embeddings = self.embedding_service.embed_documents(
                chunk_texts,
                show_progress=False,
            )
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            batch_stats["errors"].append({
                "error": str(e),
                "stage": "embedding"
            })
            # Mark all chunks in this batch as failed
            batch_stats["failed"] += batch_stats["processed"]
            batch_stats["processed"] = 0
            return batch_stats
        
        # Step 3: Store chunks in MongoDB
        try:
            chunk_dicts = [chunk.to_dict() for chunk in all_chunks]
            if chunk_dicts:
                self.chunks_collection.insert_many(chunk_dicts, ordered=False)
                logger.debug(f"Inserted {len(chunk_dicts)} chunks into MongoDB")
        except Exception as e:
            logger.error(f"Error inserting chunks into MongoDB: {e}", exc_info=True)
            batch_stats["errors"].append({
                "error": str(e),
                "stage": "mongodb_insert"
            })
        
        # Step 4: Add to vector store
        try:
            self.vector_store.add_chunks(all_chunks)
            logger.debug(f"Added {len(all_chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}", exc_info=True)
            batch_stats["errors"].append({
                "error": str(e),
                "stage": "vector_store"
            })
        
        return batch_stats
    
    def _update_stats(self, stats: dict, batch_stats: dict):
        """Update overall stats with batch stats."""
        stats["processed"] += batch_stats["processed"]
        stats["skipped"] += batch_stats["skipped"]
        stats["total_chunks"] += batch_stats["total_chunks"]
        stats["failed"] += batch_stats["failed"]
        stats["errors"].extend(batch_stats["errors"])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate existing documents to chunked+embedded format"
    )
    parser.add_argument(
        "--mongo-uri",
        default=DEFAULT_MONGO_URI,
        help="MongoDB connection URI"
    )
    parser.add_argument(
        "--db-name",
        default=DEFAULT_DB_NAME,
        help="Database name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of documents to process at once"
    )
    parser.add_argument(
        "--provider",
        choices=["local", "openai"],
        default=os.environ.get("HISTORIAN_AGENT_EMBEDDING_PROVIDER", DEFAULT_PROVIDER),
        help="Embedding provider"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HISTORIAN_AGENT_EMBEDDING_MODEL", DEFAULT_MODEL),
        help="Embedding model name"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Characters per chunk"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--vector-store",
        choices=["chroma", "mongo"],
        default=DEFAULT_VECTOR_STORE,
        help="Vector store type"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip documents that already have chunks"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Process all documents (reprocess existing)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of documents (for testing)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before migration (deletes all existing data)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting migration with parameters:")
    logger.info(f"  MongoDB URI: {args.mongo_uri[:50]}...")
    logger.info(f"  Database: {args.db_name}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Provider: {args.provider}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  Chunk overlap: {args.chunk_overlap}")
    logger.info(f"  Vector store: {args.vector_store}")
    logger.info(f"  Skip existing: {args.skip_existing}")
    if args.limit:
        logger.info(f"  Limit: {args.limit} documents")
    
    # Initialize migration
    try:
        migration = DocumentEmbeddingMigration(
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            batch_size=args.batch_size,
            embedding_provider=args.provider,
            embedding_model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            vector_store_type=args.vector_store,
        )
    except Exception as e:
        logger.error(f"Failed to initialize migration: {e}", exc_info=True)
        sys.exit(1)
    
    # Reset if requested
    if args.reset:
        logger.warning("⚠️  RESETTING VECTOR STORE - All existing embeddings will be deleted!")
        try:
            if hasattr(migration.vector_store, 'reset'):
                migration.vector_store.reset()
            logger.info("✅ Vector store reset complete")
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")
            sys.exit(1)
    
    # Run migration
    try:
        stats = migration.run(
            skip_existing=args.skip_existing,
            limit=args.limit,
        )
        
        # Exit with success if no failures
        exit_code = 0 if stats["failed"] == 0 else 1
        logger.info(f"Migration exiting with code: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user (Ctrl+C)")
        logger.info("Progress has been saved. You can resume with --skip-existing")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed with unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()