#!/usr/bin/env python3
"""
Dimension Verification Script

Verifies that all components are using consistent embedding dimensions.
This prevents "dimension mismatch" errors during search.

Usage:
    python verify_dimensions.py
"""

import sys
import os
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from pymongo import MongoClient
import chromadb
from chromadb.config import Settings


def check_chromadb_dimension():
    """Check what dimension ChromaDB collection expects."""
    print("\n🔍 Checking ChromaDB...")
    
    persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY", "/data/chroma_db/persist")
    
    try:
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection("historian_documents")
        
        # Get a sample to determine dimension
        results = collection.get(limit=1, include=["embeddings"])
        
        embeddings = results.get("embeddings")

        if embeddings is None or len(embeddings) == 0:
            print(f"   ⚠️  No embeddings found in collection")
            return None

        # embeddings[0] may be a list or numpy array — handle both
        first = embeddings[0]
        try:
            dim = len(first)
        except Exception:
            # fallback: coerce to array and read shape/size
            import numpy as _np
            arr = _np.asarray(first)
            dim = int(arr.shape[-1]) if arr.ndim > 0 else int(arr.size)

        print(f"   ✅ ChromaDB dimension: {dim}")
        return dim
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def check_mongodb_chunks():
    """Check if MongoDB has chunks and sample one."""
    print("\n🔍 Checking MongoDB...")
    
    mongo_uri = (
        os.environ.get("APP_MONGO_URI") 
        or os.environ.get("MONGO_URI") 
        or "mongodb://admin:secret@mongodb:27017/admin"
    )
    
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client["railroad_documents"]
        
        chunk_count = db["document_chunks"].count_documents({})
        print(f"   ✅ Total chunks: {chunk_count:,}")
        
        if chunk_count > 0:
            sample = db["document_chunks"].find_one()
            doc_id = sample.get("document_id", "unknown")
            text_len = len(sample.get("text", ""))
            print(f"   ✅ Sample chunk: {doc_id}")
            print(f"      Text length: {text_len} chars")
            return True
        else:
            print(f"   ⚠️  No chunks found - migration not run yet?")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def check_embedding_service():
    """Check what dimension the embedding service will produce."""
    print("\n🔍 Checking Embedding Service...")
    
    provider = os.environ.get("HISTORIAN_AGENT_EMBEDDING_PROVIDER", "ollama")
    model = os.environ.get("HISTORIAN_AGENT_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    
    print(f"   Provider: {provider}")
    print(f"   Model: {model}")
    
    try:
        from embeddings import EmbeddingService
        
        service = EmbeddingService(provider=provider, model=model)
        dimension = service.get_embedding_dimension()
        
        print(f"   ✅ Service dimension: {dimension}")
        
        # Test actual embedding generation
        test_embedding = service.embed_query("test")
        actual_dim = len(test_embedding)
        
        if actual_dim == dimension:
            print(f"   ✅ Verified: Generated {actual_dim}D embedding")
        else:
            print(f"   ⚠️  Mismatch: Expected {dimension}, got {actual_dim}")
        
        return actual_dim
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*60)
    print("  DIMENSION VERIFICATION")
    print("="*60)
    print("Checking that all components use consistent dimensions...")
    
    # Check all components
    chroma_dim = check_chromadb_dimension()
    mongo_ok = check_mongodb_chunks()
    service_dim = check_embedding_service()
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    all_ok = True
    
    if chroma_dim and service_dim:
        if chroma_dim == service_dim:
            print(f"✅ PASS: All dimensions match ({chroma_dim}D)")
        else:
            print(f"❌ FAIL: Dimension mismatch!")
            print(f"   ChromaDB: {chroma_dim}D")
            print(f"   Service:  {service_dim}D")
            print("\n💡 Fix: Reset ChromaDB and re-migrate with consistent model")
            all_ok = False
    else:
        print("⚠️  WARNING: Could not verify all dimensions")
        all_ok = False
    
    if not mongo_ok:
        print("⚠️  WARNING: No chunks in MongoDB - migration incomplete?")
        all_ok = False
    
    if all_ok:
        print("\n🎉 System is ready for search/retrieval!")
        print("\n📝 Configuration:")
        print(f"   Dimension: {chroma_dim}D")
        print(f"   Provider: {os.environ.get('HISTORIAN_AGENT_EMBEDDING_PROVIDER', 'ollama')}")
        print(f"   Model: {os.environ.get('HISTORIAN_AGENT_EMBEDDING_MODEL', 'qwen3-embedding:0.6b')}")
        return 0
    else:
        print("\n⚠️  Issues detected - review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())