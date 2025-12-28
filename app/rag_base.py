# app/rag_base.py
# Shared base components for RAG handlers
# STANDALONE VERSION - No factory dependencies

"""
Shared infrastructure for all RAG handlers.

Provides:
- DocumentStore: MongoDB operations and document expansion
- Utility functions: token counting, debug logging
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

from config import APP_CONFIG


# ============================================================================
# Utilities
# ============================================================================

def count_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Simple approximation: ~4 characters per token for English.
    This is a rough estimate - actual tokenization varies by model.
    
    Args:
        text: Text to count tokens for
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return len(text) // 4


def debug_print(*args, **kwargs):
    """
    Print debug messages if debug mode is enabled.
    
    Args:
        *args: Messages to print
        **kwargs: Additional arguments (passed to print)
    """
    if APP_CONFIG.debug_mode:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [DEBUG]", *args, **kwargs, file=sys.stderr)
        sys.stderr.flush()


# ============================================================================
# MongoDB Connection (Singleton Pattern)
# ============================================================================

class MongoDBConnection:
    """
    Singleton MongoDB connection manager.
    
    Provides single shared connection with proper connection pooling.
    """
    _instance = None
    _client = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._connect()
    
    def _connect(self):
        """Establish MongoDB connection."""
        debug_print(f"Connecting to MongoDB: {APP_CONFIG.database.uri}")
        
        self._client = MongoClient(
            APP_CONFIG.database.uri,
            serverSelectionTimeoutMS=APP_CONFIG.database.server_selection_timeout_ms,
            connectTimeoutMS=APP_CONFIG.database.connection_timeout_ms,
            maxPoolSize=10,
            minPoolSize=1,
        )
        
        # Test connection
        try:
            self._client.admin.command('ping')
            debug_print("MongoDB connection successful")
        except Exception as e:
            debug_print(f"MongoDB connection failed: {e}")
            raise
        
        self._db = self._client[APP_CONFIG.database.db_name]
    
    @property
    def client(self):
        """Get MongoDB client."""
        if self._client is None:
            self._connect()
        return self._client
    
    @property
    def db(self):
        """Get database."""
        if self._db is None:
            self._connect()
        return self._db
    
    def get_collection(self, collection_name: str):
        """Get a collection by name."""
        return self.db[collection_name]


# ============================================================================
# DocumentStore
# ============================================================================

class DocumentStore:
    """
    Handles MongoDB document operations and parent document expansion.
    
    This is the critical shared component that implements your
    small-to-big retrieval strategy.
    """
    
    def __init__(self):
        """Initialize document store with MongoDB connection."""
        debug_print("Initializing DocumentStore")
        
        # Get singleton MongoDB connection
        mongo = MongoDBConnection()
        
        # Get collections
        self.documents_coll = mongo.get_collection(APP_CONFIG.database.documents_collection)
        self.chunks_coll = mongo.get_collection(APP_CONFIG.database.chunks_collection)
        
        debug_print(
            f"DocumentStore ready: "
            f"docs={APP_CONFIG.database.documents_collection}, "
            f"chunks={APP_CONFIG.database.chunks_collection}"
        )
    
    def hydrate_parent_metadata(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch parent document metadata for a list of document IDs.
        
        Args:
            doc_ids: List of document ObjectIds (as strings)
        
        Returns:
            Dict mapping document_id -> metadata dict
        
        Example:
            >>> meta = store.hydrate_parent_metadata(['507f1f77bcf86cd799439011'])
            >>> meta['507f1f77bcf86cd799439011']['filename']
            'Relief_Record_1234.pdf'
        """
        from bson import ObjectId
        
        if not doc_ids:
            return {}
        
        # Convert string IDs to ObjectIds
        try:
            object_ids = [ObjectId(doc_id) for doc_id in doc_ids]
        except Exception as e:
            debug_print(f"Error converting doc_ids to ObjectIds: {e}")
            return {}
        
        # Fetch documents
        cursor = self.documents_coll.find(
            {"_id": {"$in": object_ids}},
            {
                "filename": 1,
                "source_type": 1,
                "upload_date": 1,
                "metadata": 1,
            }
        )
        
        # Build mapping
        result = {}
        for doc in cursor:
            doc_id_str = str(doc["_id"])
            result[doc_id_str] = {
                "filename": doc.get("filename", "Unknown"),
                "source_type": doc.get("source_type", "Unknown"),
                "upload_date": doc.get("upload_date"),
                "metadata": doc.get("metadata", {}),
            }
        
        return result
    
    def get_full_document_text(
        self,
        doc_ids: List[str]
    ) -> Tuple[str, Dict[str, str], float]:
        """
        Fetch complete text for parent documents (small-to-big expansion).
        
        This is your special sauce - retrieving full documents from chunks.
        
        Args:
            doc_ids: List of parent document IDs
        
        Returns:
            Tuple of (full_text, mapping, fetch_time) where:
                - full_text: Combined text of all documents
                - mapping: Dict of {label: document_id}
                - fetch_time: Time taken to fetch documents
        
        Example:
            >>> text, mapping, time = store.get_full_document_text(['doc1', 'doc2'])
            >>> mapping
            {'Source 1': 'doc1', 'Source 2': 'doc2'}
        """
        from bson import ObjectId
        
        start = time.time()
        
        if not doc_ids:
            return "", {}, 0.0
        
        debug_print(f"Fetching full text for {len(doc_ids)} documents")
        
        # Convert to ObjectIds
        try:
            object_ids = [ObjectId(doc_id) for doc_id in doc_ids]
        except Exception as e:
            debug_print(f"Error converting doc_ids: {e}")
            return "", {}, time.time() - start
        
        # Fetch parent metadata for source labels
        parent_meta = self.hydrate_parent_metadata(doc_ids)
        
        # Build full text by fetching ALL chunks for each document
        full_text_parts = []
        mapping = {}
        
        for idx, doc_id in enumerate(doc_ids, 1):
            obj_id = ObjectId(doc_id)
            
            # Fetch all chunks for this document, sorted by chunk_index
            chunks = list(
                self.chunks_coll.find(
                    {"document_id": obj_id}
                ).sort("chunk_index", 1)
            )
            
            if not chunks:
                debug_print(f"No chunks found for document {doc_id}")
                continue
            
            # Combine chunk text
            doc_text_parts = []
            for chunk in chunks:
                # Try both 'text' and 'ocr_text' fields
                text = chunk.get("text") or chunk.get("ocr_text", "")
                if text:
                    doc_text_parts.append(text)
            
            if doc_text_parts:
                doc_text = "\n".join(doc_text_parts)
                
                # Get filename from parent metadata
                filename = parent_meta.get(doc_id, {}).get("filename", f"Document {idx}")
                
                # Add to full text with source label
                source_label = f"Source {idx}"
                full_text_parts.append(f"[{source_label}: {filename}]\n{doc_text}")
                
                # Add to mapping
                mapping[source_label] = doc_id
        
        full_text = "\n\n".join(full_text_parts)
        fetch_time = time.time() - start
        
        debug_print(
            f"Fetched {len(full_text)} characters from {len(mapping)} documents "
            f"in {fetch_time:.2f}s"
        )
        
        return full_text, mapping, fetch_time


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "DocumentStore",
    "count_tokens",
    "debug_print",
]