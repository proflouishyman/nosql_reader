#!/usr/bin/env python3
"""
RAG System Test Script

Tests all RAG components independently and together to verify the system
is working correctly before frontend integration.

Usage:
    python scripts/test_rag_system.py [--full]

Options:
    --full    Run full test including actual document processing (slower)
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Any

# Add app directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import numpy as np
from pymongo import MongoClient

# Import RAG components
from chunking import DocumentChunker, Chunk
from embeddings import EmbeddingService
from vector_store import VectorStoreManager, get_vector_store
from retrievers import VectorRetriever, KeywordRetriever, HybridRetriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSystemTest:
    """Comprehensive test suite for RAG system."""
    
    def __init__(self):
        self.mongo_uri = os.environ.get('APP_MONGO_URI') or os.environ.get('MONGO_URI') or "mongodb://admin:secret@mongodb:27017/admin"
        self.db_name = 'railroad_documents'
        self.test_results = {}
        
    def print_header(self, text: str):
        """Print formatted test section header."""
        print("\n" + "="*70)
        print(f"  {text}")
        print("="*70)
    
    def print_test(self, test_name: str, passed: bool, details: str = ""):
        """Print test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {test_name}")
        if details:
            print(f"       {details}")
        self.test_results[test_name] = passed
    
    # ==================== Test 1: Chunking ====================
    
    def test_chunking(self) -> bool:
        """Test document chunking functionality."""
        self.print_header("TEST 1: Document Chunking")
        
        try:
            # Initialize chunker
            chunker = DocumentChunker(
                chunk_size=500,
                chunk_overlap=100,
            )
            self.print_test("Chunker initialization", True, "Created with size=500, overlap=100")
            
            # Test document
            test_doc = {
                "_id": "test_doc_001",
                "title": "Baltimore & Ohio Railroad Safety Report 1925",
                "content": "This is a test document. " * 100,  # ~500 chars
                "ocr_text": "Additional OCR content here. " * 50,
                "date": "1925-06-15"
            }
            
            # Chunk document
            chunks = chunker.chunk_document(test_doc)
            
            # Verify chunks created
            passed = len(chunks) > 0
            self.print_test(
                "Chunk generation",
                passed,
                f"Generated {len(chunks)} chunks from test document"
            )
            
            if not passed:
                return False
            
            # Verify chunk structure
            chunk = chunks[0]
            has_required_fields = all([
                hasattr(chunk, 'chunk_id'),
                hasattr(chunk, 'document_id'),
                hasattr(chunk, 'text'),
                hasattr(chunk, 'content'),  # Property alias
                hasattr(chunk, 'token_count'),
                hasattr(chunk, 'metadata'),
            ])
            self.print_test(
                "Chunk structure",
                has_required_fields,
                f"Chunk ID: {chunk.chunk_id[:30]}..."
            )
            
            # Verify metadata preserved
            has_metadata = 'title' in chunk.metadata
            self.print_test(
                "Metadata preservation",
                has_metadata,
                f"Metadata: {list(chunk.metadata.keys())}"
            )
            
            # Verify to_dict() works
            chunk_dict = chunk.to_dict()
            dict_valid = 'chunk_id' in chunk_dict and 'text' in chunk_dict
            self.print_test(
                "MongoDB serialization",
                dict_valid,
                "to_dict() produces valid MongoDB document"
            )
            
            return all([passed, has_required_fields, has_metadata, dict_valid])
            
        except Exception as e:
            self.print_test("Chunking system", False, f"Error: {str(e)}")
            logger.exception("Chunking test failed")
            return False
    
    # ==================== Test 2: Embeddings ====================
    
    def test_embeddings(self) -> bool:
        """Test embedding generation."""
        self.print_header("TEST 2: Embedding Generation")
        
        try:
            # Initialize embedding service
            embedding_service = EmbeddingService(
                provider="local",
                model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
            )
            dimension = embedding_service.get_embedding_dimension()
            self.print_test(
                "Embedding service initialization",
                True,
                f"Model loaded, dimension={dimension}"
            )
            
            # Test single embedding
            test_text = "Baltimore and Ohio Railroad historical document"
            embedding = embedding_service.embed_query(test_text)
            
            single_valid = (
                isinstance(embedding, np.ndarray) and
                len(embedding) == dimension and
                not np.all(embedding == 0)
            )
            self.print_test(
                "Single embedding generation",
                single_valid,
                f"Generated {len(embedding)}D vector"
            )
            
            # Test batch embeddings
            test_texts = [
                "Train derailment investigation report",
                "Employee safety guidelines and procedures",
                "Locomotive maintenance schedule"
            ]
            embeddings = embedding_service.embed_documents(test_texts, show_progress=False)
            
            batch_valid = (
                isinstance(embeddings, np.ndarray) and
                embeddings.shape == (len(test_texts), dimension) and
                not np.all(embeddings == 0)
            )
            self.print_test(
                "Batch embedding generation",
                batch_valid,
                f"Generated {embeddings.shape} matrix"
            )
            
            # Test embedding similarity
            embedding1 = embedding_service.embed_query("train accident report")
            embedding2 = embedding_service.embed_query("railway collision investigation")
            embedding3 = embedding_service.embed_query("chocolate cake recipe")
            
            sim_12 = embedding_service.compute_similarity(embedding1, embedding2)
            sim_13 = embedding_service.compute_similarity(embedding1, embedding3)
            
            similarity_valid = sim_12 > sim_13  # Related texts should be more similar
            self.print_test(
                "Semantic similarity",
                similarity_valid,
                f"Related: {sim_12:.3f}, Unrelated: {sim_13:.3f}"
            )
            
            return all([single_valid, batch_valid, similarity_valid])
            
        except Exception as e:
            self.print_test("Embedding system", False, f"Error: {str(e)}")
            logger.exception("Embedding test failed")
            return False
    
    # ==================== Test 3: Vector Store ====================
    
    def test_vector_store(self) -> bool:
        """Test vector store operations."""
        self.print_header("TEST 3: Vector Store (ChromaDB)")
        
        try:
            # Initialize vector store
            vector_store = get_vector_store(store_type="chroma")
            stats = vector_store.get_stats()
            self.print_test(
                "Vector store initialization",
                True,
                f"Collection: {stats.get('collection_name')}, Count: {stats.get('total_chunks', 0)}"
            )
            
            # Create test chunks with embeddings
            embedding_service = EmbeddingService(provider="local")
            
            test_chunks = []
            test_texts = [
                "Railroad safety regulations for locomotive operation",
                "Employee training procedures and certification requirements",
                "Track maintenance schedule and inspection protocols"
            ]
            
            for i, text in enumerate(test_texts):
                chunk = Chunk(
                    chunk_id=f"test_vector_chunk_{i}",
                    document_id="test_vector_doc",
                    chunk_index=i,
                    text=text,
                    token_count=len(text.split()),
                    metadata={"test": True, "source": "vector_test"},
                    embedding=embedding_service.embed_query(text)
                )
                test_chunks.append(chunk)
            
            # Test add_chunks
            vector_store.add_chunks(test_chunks)
            self.print_test(
                "Add chunks to vector store",
                True,
                f"Added {len(test_chunks)} test chunks"
            )
            
            # Test search
            query_text = "locomotive safety rules"
            query_embedding = embedding_service.embed_query(query_text)
            results = vector_store.search(query_embedding, k=2)
            
            search_valid = (
                len(results) > 0 and
                'content' in results[0] and
                'score' in results[0]
            )
            self.print_test(
                "Vector similarity search",
                search_valid,
                f"Found {len(results)} results, top score: {results[0].get('score', 0):.3f}"
            )
            
            if search_valid:
                print(f"       Top result: {results[0]['content'][:60]}...")
            
            # Test retrieval
            retrieved_chunk = vector_store.get_chunk(test_chunks[0].chunk_id)
            retrieval_valid = retrieved_chunk is not None
            self.print_test(
                "Chunk retrieval by ID",
                retrieval_valid,
                f"Retrieved chunk: {test_chunks[0].chunk_id}"
            )
            
            # Cleanup test chunks
            vector_store.delete_chunks([c.chunk_id for c in test_chunks])
            self.print_test(
                "Cleanup test data",
                True,
                "Deleted test chunks"
            )
            
            return all([search_valid, retrieval_valid])
            
        except Exception as e:
            self.print_test("Vector store system", False, f"Error: {str(e)}")
            logger.exception("Vector store test failed")
            return False
    
    # ==================== Test 4: MongoDB Integration ====================
    
    def test_mongodb(self) -> bool:
        """Test MongoDB connection and collections."""
        self.print_header("TEST 4: MongoDB Integration")
        
        try:
            # Connect to MongoDB
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            self.print_test("MongoDB connection", True, f"Connected to {self.db_name}")
            
            db = client[self.db_name]
            
            # Check required collections
            collections = db.list_collection_names()
            required = ['documents', 'document_chunks']
            has_collections = all(c in collections for c in required)
            self.print_test(
                "Required collections exist",
                has_collections,
                f"Found: {', '.join(required)}"
            )
            
            # Check documents collection
            doc_count = db['documents'].count_documents({})
            self.print_test(
                "Documents collection",
                doc_count > 0,
                f"{doc_count:,} documents available"
            )
            
            # Check document_chunks collection
            chunk_count = db['document_chunks'].count_documents({})
            chunks_exist = chunk_count > 0
            self.print_test(
                "Document chunks collection",
                chunks_exist,
                f"{chunk_count:,} chunks {'exist' if chunks_exist else 'will be created during migration'}"
            )
            
            # Check indexes
            indexes = db['document_chunks'].index_information()
            has_indexes = 'chunk_id_1' in indexes or 'document_id_1' in indexes
            self.print_test(
                "Chunk indexes",
                has_indexes,
                f"{len(indexes)} indexes present"
            )
            
            client.close()
            return all([has_collections, doc_count > 0])
            
        except Exception as e:
            self.print_test("MongoDB integration", False, f"Error: {str(e)}")
            logger.exception("MongoDB test failed")
            return False
    
    # ==================== Test 5: Retrievers ====================
    
    def test_retrievers(self) -> bool:
        """Test retriever functionality."""
        self.print_header("TEST 5: Retrievers (Vector & Hybrid)")
        
        try:
            # Setup
            embedding_service = EmbeddingService(provider="local")
            vector_store = get_vector_store(store_type="chroma")
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            db = client[self.db_name]
            chunks_collection = db['document_chunks']
            
            # Create test data
            test_texts = [
                "Train derailment caused by track defect in 1923",
                "Employee safety regulations updated in 1924",
                "Locomotive maintenance procedures manual"
            ]
            
            test_chunks = []
            for i, text in enumerate(test_texts):
                chunk = Chunk(
                    chunk_id=f"test_retriever_chunk_{i}",
                    document_id="test_retriever_doc",
                    chunk_index=i,
                    text=text,
                    token_count=len(text.split()),
                    metadata={"test": True, "source": "retriever_test"},
                    embedding=embedding_service.embed_query(text)
                )
                test_chunks.append(chunk)
            
            # Add to vector store and MongoDB
            vector_store.add_chunks(test_chunks)
            chunks_collection.insert_many([c.to_dict() for c in test_chunks])
            
            # Test VectorRetriever
            vector_retriever = VectorRetriever(
                vector_store=vector_store,
                embedding_service=embedding_service,
                mongo_collection=chunks_collection,
                top_k=2
            )
            
            vector_results = vector_retriever.get_relevant_documents("train accident investigation")
            vector_valid = len(vector_results) > 0
            self.print_test(
                "VectorRetriever",
                vector_valid,
                f"Retrieved {len(vector_results)} documents"
            )
            
            if vector_valid and vector_results:
                print(f"       Top result: {vector_results[0].page_content[:60]}...")
                print(f"       Score: {vector_results[0].metadata.get('score', 0):.3f}")
            
            # Cleanup
            vector_store.delete_chunks([c.chunk_id for c in test_chunks])
            chunks_collection.delete_many({"chunk_id": {"$in": [c.chunk_id for c in test_chunks]}})
            
            client.close()
            return vector_valid
            
        except Exception as e:
            self.print_test("Retriever system", False, f"Error: {str(e)}")
            logger.exception("Retriever test failed")
            return False
    
    # ==================== Test 6: End-to-End ====================
    
    def test_end_to_end(self, test_real_docs: bool = False) -> bool:
        """Test complete RAG pipeline end-to-end."""
        self.print_header("TEST 6: End-to-End Pipeline")
        
        try:
            if not test_real_docs:
                print("⏭️  Skipping real document test (use --full flag to enable)")
                return True
            
            # Connect to MongoDB
            client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            db = client[self.db_name]
            
            # Get a sample document
            sample_doc = db['documents'].find_one()
            if not sample_doc:
                self.print_test("End-to-end test", False, "No documents in database")
                return False
            
            doc_id = str(sample_doc['_id'])
            self.print_test(
                "Sample document loaded",
                True,
                f"ID: {doc_id}"
            )
            
            # Step 1: Chunk document
            chunker = DocumentChunker()
            chunks = chunker.chunk_document(sample_doc)
            self.print_test(
                "Document chunked",
                len(chunks) > 0,
                f"Generated {len(chunks)} chunks"
            )
            
            # Step 2: Generate embeddings
            embedding_service = EmbeddingService(provider="local")
            texts = [chunk.content for chunk in chunks]
            embeddings = embedding_service.embed_documents(texts[:5], show_progress=False)  # Just first 5
            
            for i, chunk in enumerate(chunks[:5]):
                chunk.embedding = embeddings[i]
            
            self.print_test(
                "Embeddings generated",
                True,
                f"Created {len(embeddings)} embeddings"
            )
            
            # Step 3: Store in vector store
            vector_store = get_vector_store(store_type="chroma")
            vector_store.add_chunks(chunks[:5])
            self.print_test(
                "Stored in vector DB",
                True,
                f"Added {len(chunks[:5])} chunks to ChromaDB"
            )
            
            # Step 4: Search
            query = "What does this document discuss?"
            query_embedding = embedding_service.embed_query(query)
            results = vector_store.search(query_embedding, k=3)
            
            search_works = len(results) > 0
            self.print_test(
                "Semantic search works",
                search_works,
                f"Found {len(results)} relevant chunks"
            )
            
            if search_works:
                print(f"       Query: '{query}'")
                print(f"       Top result: {results[0]['content'][:80]}...")
                print(f"       Score: {results[0].get('score', 0):.3f}")
            
            # Cleanup
            vector_store.delete_chunks([c.chunk_id for c in chunks[:5]])
            self.print_test("Cleanup", True, "Removed test data")
            
            client.close()
            return search_works
            
        except Exception as e:
            self.print_test("End-to-end pipeline", False, f"Error: {str(e)}")
            logger.exception("End-to-end test failed")
            return False
    
    # ==================== Run All Tests ====================
    
    def run_all_tests(self, full: bool = False):
        """Run complete test suite."""
        print("\n" + "="*70)
        print("  RAG SYSTEM TEST SUITE")
        print("="*70)
        print(f"Testing RAG components before frontend integration")
        print(f"Database: {self.db_name}")
        print(f"Full test mode: {'Yes' if full else 'No (use --full for complete tests)'}")
        
        # Run tests
        tests = [
            ("Chunking", lambda: self.test_chunking()),
            ("Embeddings", lambda: self.test_embeddings()),
            ("Vector Store", lambda: self.test_vector_store()),
            ("MongoDB", lambda: self.test_mongodb()),
            ("Retrievers", lambda: self.test_retrievers()),
            ("End-to-End", lambda: self.test_end_to_end(full)),
        ]
        
        for test_name, test_func in tests:
            try:
                passed = test_func()
                if not passed and test_name != "End-to-End":
                    print(f"\n⚠️  {test_name} test failed, but continuing...")
            except Exception as e:
                print(f"\n❌ {test_name} test crashed: {e}")
                logger.exception(f"{test_name} test exception")
        
        # Print summary
        self.print_header("TEST SUMMARY")
        
        total = len(self.test_results)
        passed = sum(1 for v in self.test_results.values() if v)
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! System ready for frontend integration.")
            return True
        else:
            print(f"\n⚠️  {failed} test(s) failed. Review errors above.")
            print("\nFailed tests:")
            for name, result in self.test_results.items():
                if not result:
                    print(f"  ❌ {name}")
            return False


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Test RAG system components")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test including real document processing (slower)"
    )
    
    args = parser.parse_args()
    
    tester = RAGSystemTest()
    success = tester.run_all_tests(full=args.full)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()