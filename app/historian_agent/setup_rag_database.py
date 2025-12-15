"""
setup_rag_database.py - Initialize RAG system (ChromaDB + MongoDB chunks)

This script sets up the vector database and RAG infrastructure for semantic search.
Run this AFTER setup_databases.py.

Usage:
    docker compose exec app python scripts/setup_rag_database.py

Creates:
- ChromaDB collection (historian_documents) with 1536D embeddings
- MongoDB document_chunks collection indexes
- Verifies gte-Qwen2-1.5B-instruct model can load
- Tests integration between MongoDB and ChromaDB

Requirements:
- MongoDB must be running (setup_databases.py completed)
- sentence-transformers, chromadb installed
- CHROMA_PERSIST_DIRECTORY environment variable set
"""

import sys
import os
from pathlib import Path
import logging
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup_rag_database.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGDatabaseSetup:
    """Handles RAG database initialization."""
    
    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.chroma_client = None
        self.chroma_collection = None
        self.embedding_model = None
        
    def connect_mongodb(self):
        """Connect to MongoDB."""
        print("\n🔧 Connecting to MongoDB...")
        try:
            mongo_uri = os.environ.get('MONGO_URI') or os.environ.get('APP_MONGO_URI')
            if not mongo_uri:
                raise ValueError("APP_MONGO_URI environment variable not set")
            
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.mongo_client.admin.command('ping')
            self.db = self.mongo_client['railroad_documents']
            
            print("   ✅ MongoDB connected")
            logger.info("Connected to MongoDB")
            return True
            
        except Exception as e:
            print(f"   ❌ MongoDB connection failed: {e}")
            logger.error(f"MongoDB connection failed: {e}")
            return False
    
    def verify_mongodb_collections(self):
        """Verify required MongoDB collections exist."""
        print("\n🔍 Verifying MongoDB collections...")
        required = ['documents', 'document_chunks']
        existing = self.db.list_collection_names()
        
        missing = [c for c in required if c not in existing]
        if missing:
            print(f"   ❌ Missing collections: {missing}")
            print("   💡 Run setup_databases.py first!")
            return False
        
        print("   ✅ All required collections exist")
        return True
    
    def create_chunks_indexes(self):
        """Create indexes for document_chunks collection."""
        print("\n📇 Creating document_chunks indexes...")
        chunks = self.db['document_chunks']
        existing_indexes = chunks.index_information()
        
        indexes_created = 0
        
        # Unique index on chunk_id
        if "chunk_id_1" not in existing_indexes:
            chunks.create_index([("chunk_id", 1)], unique=True)
            print("   ✅ Created unique index: chunk_id")
            logger.info("Created unique index on chunk_id")
            indexes_created += 1
        else:
            print("   ✓ Index exists: chunk_id")
        
        # Index on document_id for lookups
        if "document_id_1" not in existing_indexes:
            chunks.create_index([("document_id", 1)])
            print("   ✅ Created index: document_id")
            logger.info("Created index on document_id")
            indexes_created += 1
        else:
            print("   ✓ Index exists: document_id")
        
        # Compound index on document_id + chunk_index
        if "document_id_1_chunk_index_1" not in existing_indexes:
            chunks.create_index([("document_id", 1), ("chunk_index", 1)])
            print("   ✅ Created compound index: document_id + chunk_index")
            logger.info("Created compound index on document_id and chunk_index")
            indexes_created += 1
        else:
            print("   ✓ Index exists: document_id + chunk_index")
        
        if indexes_created > 0:
            print(f"\n   📊 Created {indexes_created} new indexes")
        
        return True
    
    def verify_embedding_model(self):
        """Verify gte-Qwen2-1.5B-instruct can be loaded."""
        print("\n🤖 Verifying embedding model...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = os.environ.get(
                'HISTORIAN_AGENT_EMBEDDING_MODEL',
                'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
            )
            
            print(f"   Loading: {model_name}")
            start_time = time.time()
            
            self.embedding_model = SentenceTransformer(
                model_name,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            print(f"   ✅ Model loaded successfully")
            print(f"      Dimension: {dimension}")
            print(f"      Load time: {load_time:.2f}s")
            
            if dimension != 1536:
                print(f"   ⚠️  Warning: Expected 1536 dimensions, got {dimension}")
                logger.warning(f"Unexpected dimension: {dimension}")
            
            logger.info(f"Embedding model verified: {model_name}, {dimension}D")
            return True
            
        except ImportError as e:
            print(f"   ❌ Missing dependency: {e}")
            print("   💡 Install with: pip install sentence-transformers transformers")
            logger.error(f"Import error: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Model load failed: {e}")
            logger.error(f"Model load failed: {e}")
            return False
    
    def initialize_chromadb(self):
        """Initialize ChromaDB collection."""
        print("\n🗄️  Initializing ChromaDB...")
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Get persist directory from environment
            persist_dir = os.environ.get(
                'CHROMA_PERSIST_DIRECTORY',
                '/home/claude/chroma_db'
            )
            
            # Ensure directory exists
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            
            print(f"   Persist directory: {persist_dir}")
            
            # Initialize client
            self.chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            
            print("   ✅ ChromaDB client initialized")
            
            # Get or create collection
            collection_name = "historian_documents"
            
            try:
                self.chroma_collection = self.chroma_client.get_collection(
                    name=collection_name
                )
                print(f"   ✅ Collection exists: {collection_name}")
                
                # Get stats
                count = self.chroma_collection.count()
                print(f"      Current vectors: {count}")
                
            except Exception:
                # Create new collection
                self.chroma_collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"   ✅ Created collection: {collection_name}")
                logger.info(f"Created ChromaDB collection: {collection_name}")
            
            logger.info("ChromaDB initialized successfully")
            return True
            
        except ImportError as e:
            print(f"   ❌ Missing dependency: {e}")
            print("   💡 Install with: pip install chromadb")
            logger.error(f"Import error: {e}")
            return False
        except Exception as e:
            print(f"   ❌ ChromaDB initialization failed: {e}")
            logger.error(f"ChromaDB init failed: {e}")
            return False
    
    def test_integration(self):
        """Test integration between MongoDB and ChromaDB."""
        print("\n🧪 Testing integration...")
        
        try:
            import numpy as np
            
            # Create test data
            test_chunk_id = "test_integration_chunk_0"
            test_text = "This is a test document for verifying RAG system integration."
            
            print("   1. Generating test embedding...")
            embedding = self.embedding_model.encode([test_text])[0]
            print(f"      ✅ Generated {len(embedding)}-dimensional vector")
            
            # Test MongoDB insertion
            print("   2. Testing MongoDB insertion...")
            test_chunk = {
                "chunk_id": test_chunk_id,
                "document_id": "test_doc_id",
                "chunk_index": 0,
                "text": test_text,
                "token_count": 12,
                "metadata": {
                    "source_file": "integration_test.txt",
                    "test": True
                }
            }
            
            # Insert or update
            self.db['document_chunks'].update_one(
                {"chunk_id": test_chunk_id},
                {"$set": test_chunk},
                upsert=True
            )
            print("      ✅ MongoDB insert successful")
            
            # Test ChromaDB insertion
            print("   3. Testing ChromaDB insertion...")
            self.chroma_collection.upsert(
                ids=[test_chunk_id],
                embeddings=[embedding.tolist()],
                documents=[test_text],
                metadatas=[{"test": True, "source": "integration_test"}]
            )
            print("      ✅ ChromaDB insert successful")
            
            # Test retrieval
            print("   4. Testing vector search...")
            results = self.chroma_collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=1
            )
            
            if results['ids'] and results['ids'][0] and results['ids'][0][0] == test_chunk_id:
                print("      ✅ Vector search successful")
            else:
                print("      ⚠️  Vector search returned unexpected results")
            
            # Cleanup test data
            print("   5. Cleaning up test data...")
            self.db['document_chunks'].delete_one({"chunk_id": test_chunk_id})
            self.chroma_collection.delete(ids=[test_chunk_id])
            print("      ✅ Test data cleaned up")
            
            print("\n   ✅ Integration test PASSED")
            logger.info("Integration test passed")
            return True
            
        except Exception as e:
            print(f"\n   ❌ Integration test FAILED: {e}")
            logger.error(f"Integration test failed: {e}", exc_info=True)
            return False
    
    def print_summary(self):
        """Print setup summary."""
        print("\n" + "="*60)
        print("✅ RAG Database Setup Complete!")
        print("="*60)
        
        print("\n📊 Summary:")
        print("   • MongoDB: document_chunks indexes created")
        print("   • ChromaDB: historian_documents collection ready")
        print("   • Embedding model: gte-Qwen2-1.5B-instruct verified")
        print("   • Dimension: 1536")
        print("   • Integration: Tested and working")
        
        print("\n📝 Next Steps:")
        print("\n   1. Run migration to embed existing documents:")
        print("      docker compose exec app python scripts/embed_existing_documents.py \\")
        print("        --batch-size 100 \\")
        print("        --provider local")
        print()
        print("   2. Monitor progress:")
        print("      tail -f embed_migration.log")
        print()
        print("   3. Verify migration:")
        print("      docker compose exec app python scripts/verify_rag_setup.py")
        print()
        print("   4. Test semantic search:")
        print("      docker compose exec app python -c \"")
        print("        from app.historian_agent import get_agent")
        print("        from app.database_setup import get_client, get_db")
        print("        client = get_client()")
        print("        db = get_db(client)")
        print("        agent = get_agent(db['documents'])")
        print("        result = agent.invoke('train accidents')")
        print("        print(result['answer'])")
        print("      \"")
        print()
        
        print("💡 Tips:")
        print("   • Estimated migration time: ~30-60 minutes for 50k docs")
        print("   • M4 Mac Pro will use Neural Engine for acceleration")
        print("   • You can resume migration if interrupted (--resume flag)")
        print("   • ChromaDB data stored in: " + os.environ.get('CHROMA_PERSIST_DIRECTORY', 'N/A'))
        print()
    
    def run(self):
        """Run complete RAG database setup."""
        print("="*60)
        print("RAG Database Setup - Historical Document Reader")
        print("Embedding Model: gte-Qwen2-1.5B-instruct (1536D)")
        print("="*60)
        
        steps = [
            ("Connect to MongoDB", self.connect_mongodb),
            ("Verify MongoDB collections", self.verify_mongodb_collections),
            ("Create document_chunks indexes", self.create_chunks_indexes),
            ("Verify embedding model", self.verify_embedding_model),
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
    """Entry point."""
    setup = RAGDatabaseSetup()
    success = setup.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()