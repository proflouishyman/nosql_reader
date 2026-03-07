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
from typing import Any, Dict, List, Tuple, Optional
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
        message = " ".join(str(arg) for arg in args)
        print(f"[{timestamp}] [DEBUG]", message, file=sys.stderr)
        sys.stderr.flush()
        # Mirror debug output into SSE stream without impacting core flow.
        try:
            from log_stream import push
            push(f"[DEBUG] {message}", level="debug", source="[RAG]")
        except Exception:
            pass


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
    ) -> Tuple[str, List[Dict[str, str]], float]:
        """
        Fetch complete text for parent documents (small-to-big expansion).
        
        Supports TWO data structures:
        1. Chunked: Separate chunks collection with document_id references
        2. Embedded: Text stored directly in document (ocr_text, content, etc.)
        
        Args:
            doc_ids: List of parent document IDs
        
        Returns:
            Tuple of (full_text, sources_list, fetch_time) where:
                - full_text: Combined text of all documents
                - sources_list: List of source dicts [{"label": "Source 1", "id": "...", ...}]
                - fetch_time: Time taken to fetch documents
        """
        from bson import ObjectId
        
        start = time.time()
        
        if not doc_ids:
            return "", [], 0.0
        
        debug_print(f"Fetching full text for {len(doc_ids)} documents")
        
        # Convert to ObjectIds
        try:
            object_ids = [ObjectId(doc_id) for doc_id in doc_ids]
        except Exception as e:
            debug_print(f"Error converting doc_ids: {e}")
            return "", [], time.time() - start
        
        # Fetch parent metadata for source labels
        parent_meta = self.hydrate_parent_metadata(doc_ids)
        
        # Helper function to strip file extensions
        def strip_extensions(filename: str) -> str:
            """Strip common file extensions from filename for display."""
            extensions = ['.json', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pdf']
            display = filename
            while True:
                stripped = False
                for ext in extensions:
                    if display.lower().endswith(ext):
                        display = display[:-len(ext)]
                        stripped = True
                        break
                if not stripped:
                    break
            return display
        
        # Try to build full text from chunks first
        full_text_parts = []
        sources_list = []  # Internal list format
        docs_found_in_chunks = set()
        
        for idx, doc_id in enumerate(doc_ids, 1):
            
            # Fetch all chunks for this document, sorted by chunk_index
            chunks = list(
                self.chunks_coll.find(
                    {"document_id": doc_id}  # Use string directly (not ObjectId)
                ).sort("chunk_index", 1)
            )
            
            if chunks:
                # Found chunks - use chunked structure
                docs_found_in_chunks.add(doc_id)
                
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
                    display_name = strip_extensions(filename)
                    
                    # Add to full text with source label
                    source_label = f"Source {idx}"
                    full_text_parts.append(f"[{source_label}: {display_name}]\n{doc_text}")
                    
                    # Add to sources list (internal format)
                    sources_list.append({
                        "label": source_label,
                        "id": doc_id,
                        "filename": filename,
                        "display_name": display_name
                    })
        
        # If no chunks found for ANY documents, fall back to embedded text structure
        if not docs_found_in_chunks:
            debug_print("No chunks found - using embedded text structure")
            
            # Fetch documents directly and extract embedded text
            docs = list(
                self.documents_coll.find(
                    {"_id": {"$in": object_ids}},
                    {
                        "filename": 1,
                        "ocr_text": 1,
                        "content": 1,
                        "text": 1,
                        "summary": 1,
                    }
                )
            )
            
            for idx, doc in enumerate(docs, 1):
                # Try multiple text fields in order of preference
                text = (
                    doc.get("ocr_text") or 
                    doc.get("content") or 
                    doc.get("text") or 
                    doc.get("summary") or 
                    ""
                )
                
                if text and text.strip():
                    doc_id = str(doc["_id"])
                    filename = doc.get("filename", f"Document {idx}")
                    display_name = strip_extensions(filename)
                    
                    source_label = f"Source {idx}"
                    full_text_parts.append(f"[{source_label}: {display_name}]\n{text}")
                    
                    # Add to sources list (internal format)
                    sources_list.append({
                        "label": source_label,
                        "id": doc_id,
                        "filename": filename,
                        "display_name": display_name
                    })
                else:
                    debug_print(f"No text found in document {doc.get('_id')}")
        
        # Handle mixed case: some docs have chunks, others don't
        elif len(docs_found_in_chunks) < len(doc_ids):
            debug_print(
                f"Mixed structure: {len(docs_found_in_chunks)} with chunks, "
                f"{len(doc_ids) - len(docs_found_in_chunks)} without"
            )
            
            # Fetch remaining documents without chunks
            missing_ids = [
                ObjectId(doc_id) 
                for doc_id in doc_ids 
                if doc_id not in docs_found_in_chunks
            ]
            
            if missing_ids:
                docs = list(
                    self.documents_coll.find(
                        {"_id": {"$in": missing_ids}},
                        {
                            "filename": 1,
                            "ocr_text": 1,
                            "content": 1,
                            "text": 1,
                            "summary": 1,
                        }
                    )
                )
                
                current_idx = len(sources_list) + 1
                for doc in docs:
                    text = (
                        doc.get("ocr_text") or 
                        doc.get("content") or 
                        doc.get("text") or 
                        doc.get("summary") or 
                        ""
                    )
                    
                    if text and text.strip():
                        doc_id = str(doc["_id"])
                        filename = doc.get("filename", f"Document {current_idx}")
                        display_name = strip_extensions(filename)
                        
                        source_label = f"Source {current_idx}"
                        full_text_parts.append(f"[{source_label}: {display_name}]\n{text}")
                        
                        # Add to sources list (internal format)
                        sources_list.append({
                            "label": source_label,
                            "id": doc_id,
                            "filename": filename,
                            "display_name": display_name
                        })
                        current_idx += 1
        
        full_text = "\n\n".join(full_text_parts)
        fetch_time = time.time() - start
        
        debug_print(
            f"Fetched {len(full_text)} characters from {len(sources_list)} documents "
            f"in {fetch_time:.2f}s"
        )
        
        return full_text, sources_list, fetch_time

    def fetch_block_snippets(
        self,
        block_ids: List[str],
        max_chars: int = 320,
    ) -> Dict[str, str]:
        """
        Resolve explorer block IDs (for example `doc_id::b0`) to short text snippets.

        This powers question-graph evidence drill-down in the UI. The lookup strategy is:
        1. Try chunk text by document + inferred block index.
        2. Fallback to first chunk for the document.
        3. Fallback to embedded OCR/content text from the parent document.
        """
        from bson import ObjectId

        snippets: Dict[str, str] = {}
        if not block_ids:
            return snippets

        # Parse block identifiers once to avoid repeated regex work.
        parsed_items: List[Tuple[str, Optional[int], str]] = []
        doc_ids: List[str] = []
        for raw in block_ids:
            block_id = str(raw or "").strip()
            if not block_id:
                continue
            doc_id = block_id.split("::", 1)[0]
            block_index: Optional[int] = None
            if "::b" in block_id:
                try:
                    block_index = int(block_id.rsplit("::b", 1)[-1])
                except ValueError:
                    block_index = None
            parsed_items.append((doc_id, block_index, block_id))
            doc_ids.append(doc_id)

        if not parsed_items:
            return snippets

        unique_doc_ids = sorted(set(doc_ids))

        # Fetch chunk rows for all relevant docs in one pass.
        chunk_rows = list(
            self.chunks_coll.find(
                {"document_id": {"$in": unique_doc_ids}},
                {"document_id": 1, "chunk_index": 1, "text": 1, "ocr_text": 1},
            ).sort([("document_id", 1), ("chunk_index", 1)])
        )
        chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for row in chunk_rows:
            doc_id = str(row.get("document_id") or "")
            if not doc_id:
                continue
            chunks_by_doc.setdefault(doc_id, []).append(row)

        # Parent document fallback for docs missing chunk text.
        object_ids = []
        object_to_doc: Dict[ObjectId, str] = {}
        for doc_id in unique_doc_ids:
            try:
                oid = ObjectId(doc_id)
                object_ids.append(oid)
                object_to_doc[oid] = doc_id
            except Exception:
                continue
        parent_text_by_doc: Dict[str, str] = {}
        if object_ids:
            parent_rows = self.documents_coll.find(
                {"_id": {"$in": object_ids}},
                {"ocr_text": 1, "content": 1, "text": 1, "summary": 1},
            )
            for row in parent_rows:
                oid = row.get("_id")
                doc_id = object_to_doc.get(oid)
                if not doc_id:
                    continue
                text = (
                    row.get("ocr_text")
                    or row.get("content")
                    or row.get("text")
                    or row.get("summary")
                    or ""
                )
                parent_text_by_doc[doc_id] = str(text)

        for doc_id, block_index, block_id in parsed_items:
            if block_id in snippets:
                continue

            chunk_candidates = chunks_by_doc.get(doc_id) or []
            snippet_text = ""
            if chunk_candidates:
                chosen = None
                if block_index is not None:
                    for row in chunk_candidates:
                        if int(row.get("chunk_index") or 0) == block_index:
                            chosen = row
                            break
                    if chosen is None and 0 <= block_index < len(chunk_candidates):
                        chosen = chunk_candidates[block_index]
                if chosen is None:
                    chosen = chunk_candidates[0]
                snippet_text = str(chosen.get("text") or chosen.get("ocr_text") or "")

            if not snippet_text:
                snippet_text = parent_text_by_doc.get(doc_id, "")

            cleaned = " ".join(snippet_text.split())
            if not cleaned:
                snippets[block_id] = ""
                continue
            if len(cleaned) > max_chars:
                cleaned = f"{cleaned[:max_chars].rstrip()}..."
            snippets[block_id] = cleaned

        return snippets


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "DocumentStore",
    "count_tokens",
    "debug_print",
]
