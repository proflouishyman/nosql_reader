"""
Document chunking service for the Historian Agent RAG system.

This module handles intelligent document splitting with overlap and metadata preservation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import logging

import numpy as np

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CONTEXT_FIELDS = ["title", "content", "ocr_text", "summary", "description"]
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""]


@dataclass
class Chunk:
    """
    Represents a single chunk of a document with metadata.
    
    This is the core data structure used throughout the RAG system.
    Compatible with LangChain and ChromaDB requirements.
    """
    
    chunk_id: str
    document_id: str  # Parent document ID
    chunk_index: int
    text: str  # Main content field
    token_count: int
    metadata: Dict[str, Any]
    overlap_previous: Optional[str] = None
    overlap_next: Optional[str] = None
    embedding: Optional[np.ndarray] = None  # Added for migration script
    
    @property
    def content(self) -> str:
        """Alias for text field for consistency with retrievers."""
        return self.text
    
    @property
    def source_document_id(self) -> str:
        """Alias for document_id for backward compatibility."""
        return self.document_id
    
    @property
    def chunk_text(self) -> str:
        """Alias for text field for backward compatibility."""
        return self.text
    
    @property
    def chunk_tokens(self) -> int:
        """Alias for token_count for backward compatibility."""
        return self.token_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for MongoDB storage."""
        result = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "overlap_previous": self.overlap_previous,
            "overlap_next": self.overlap_next,
        }
        
        # Don't store embedding in MongoDB (stored in ChromaDB)
        # Numpy arrays aren't JSON serializable anyway
        
        return result


class DocumentChunker:
    """
    Intelligent document chunking with overlap and metadata preservation.
    
    This class splits documents into smaller chunks optimized for embedding
    and retrieval while preserving context through overlap.
    """
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
        context_fields: Optional[List[str]] = None,
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters/tokens
            chunk_overlap: Number of characters/tokens to overlap between chunks
            separators: Custom separators for splitting (defaults to sensible values)
            context_fields: Document fields to include in chunks (content_fields alias supported)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_fields = context_fields or DEFAULT_CONTEXT_FIELDS
        
        if separators is None:
            self.separators = DEFAULT_SEPARATORS
        else:
            self.separators = separators
        
        # Initialize text splitter
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain is required for document chunking. "
                "Install with: pip install langchain-text-splitters"
            )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=self._count_tokens,
            is_separator_regex=False,
        )
        
        # Initialize tokenizer for accurate token counting
        self._init_tokenizer()
        
        logger.info(
            f"DocumentChunker initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, fields={self.context_fields}"
        )
    
    def _init_tokenizer(self) -> None:
        """Initialize the tokenizer for token counting."""
        if tiktoken is not None:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info("Initialized tiktoken tokenizer for accurate token counting")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}. Using character count approximation.")
                self.tokenizer = None
        else:
            logger.warning("tiktoken not available. Using character count / 4 as token approximation.")
            self.tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken if available, otherwise estimate.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}. Using approximation.")
        
        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(text) // 4
    
    def extract_text_fields(self, document: Dict[str, Any]) -> str:
        """
        Extract and concatenate relevant text fields from a document.
        
        Args:
            document: MongoDB document dictionary
            
        Returns:
            Combined text from all relevant fields
        """
        text_parts = []
        
        for field in self.context_fields:
            value = document.get(field)
            if value:
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, list):
                    # Join list items with spaces
                    text_parts.append(" ".join(str(item) for item in value))
                elif isinstance(value, dict):
                    # Join dictionary values
                    text_parts.append(" ".join(str(v) for v in value.values()))
        
        # Join all parts with double newline for paragraph separation
        combined_text = "\n\n".join(text_parts)
        return combined_text.strip()
    
    def extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from document for chunk enrichment.
        
        Args:
            document: MongoDB document dictionary
            
        Returns:
            Dictionary of metadata fields
        """
        metadata = {}
        
        # Standard metadata fields
        metadata_fields = [
            "title", "name", "document_title",
            "date", "year", "document_date",
            "source", "archive", "collection",
            "document_type", "type",
            "author", "creator",
            "location", "place",
            "filename", "file_path"
        ]
        
        for field in metadata_fields:
            if field in document:
                value = document[field]
                # Convert to simple types for metadata
                if isinstance(value, (str, int, float, bool)):
                    metadata[field] = value
                elif isinstance(value, list) and value:
                    metadata[field] = value[0] if len(value) == 1 else value[:10]  # Limit arrays
        
        # Extract entities if available
        if "extracted_entities" in document:
            entities = document["extracted_entities"]
            if isinstance(entities, list) and entities:
                # Store entity texts for quick reference
                entity_texts = [e.get("text") for e in entities if isinstance(e, dict)]
                if entity_texts:
                    metadata["entities"] = entity_texts[:10]  # Limit to first 10
        
        return metadata
    
    def chunk_document(
        self,
        document: Dict[str, Any],
        content_fields: Optional[tuple] = None
    ) -> List[Chunk]:
        """
        Split a document into overlapping chunks with metadata.
        
        Args:
            document: MongoDB document dictionary
            content_fields: Optional override for context_fields (for backward compatibility)
            
        Returns:
            List of Chunk objects
        """
        # Support content_fields parameter name for migration script compatibility
        if content_fields is not None:
            old_fields = self.context_fields
            self.context_fields = list(content_fields)
        
        try:
            # Extract text content
            text = self.extract_text_fields(document)
            
            if not text:
                logger.warning(f"No text content found in document {document.get('_id')}")
                return []
            
            # Extract metadata
            metadata = self.extract_metadata(document)
            
            # Get document ID
            doc_id = str(document.get("_id", "unknown"))
            
            # Split text into chunks
            try:
                chunk_texts = self.text_splitter.split_text(text)
            except Exception as e:
                logger.error(f"Error splitting document {doc_id}: {e}")
                return []
            
            if not chunk_texts:
                logger.warning(f"No chunks generated for document {doc_id}")
                return []
            
            # Create enriched chunk objects
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                # Generate chunk ID
                chunk_id = f"{doc_id}_chunk_{i:03d}"
                
                # Count tokens
                chunk_tokens = self._count_tokens(chunk_text)
                
                # Get overlap text if available
                overlap_prev = None
                overlap_next = None
                
                if i > 0 and self.chunk_overlap > 0:
                    # Get overlap from previous chunk
                    prev_text = chunk_texts[i - 1]
                    overlap_prev = prev_text[-self.chunk_overlap:]
                
                if i < len(chunk_texts) - 1 and self.chunk_overlap > 0:
                    # Get overlap into next chunk
                    next_text = chunk_texts[i + 1]
                    overlap_next = next_text[:self.chunk_overlap]
                
                # Create chunk object
                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    chunk_index=i,
                    text=chunk_text,
                    token_count=chunk_tokens,
                    metadata=metadata.copy(),
                    overlap_previous=overlap_prev,
                    overlap_next=overlap_next,
                )
                
                chunks.append(chunk)
            
            logger.debug(f"Created {len(chunks)} chunks for document {doc_id}")
            return chunks
            
        finally:
            # Restore original context_fields if we temporarily changed them
            if content_fields is not None:
                self.context_fields = old_fields
    
    def chunk_documents_batch(
        self, 
        documents: Iterable[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        Chunk multiple documents in batch.
        
        Args:
            documents: Iterable of MongoDB document dictionaries
            
        Returns:
            List of all Chunk objects from all documents
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                doc_id = doc.get("_id", "unknown")
                logger.error(f"Error chunking document {doc_id}: {e}", exc_info=True)
                continue
        
        logger.info(f"Created {len(all_chunks)} total chunks from batch")
        return all_chunks


def get_optimal_chunk_size(model_name: str) -> int:
    """
    Get optimal chunk size based on embedding model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Recommended chunk size in tokens
    """
    # Model-specific recommendations
    model_limits = {
        # OpenAI models
        "text-embedding-3-large": 8191,
        "text-embedding-3-small": 8191,
        "text-embedding-ada-002": 8191,
        # HuggingFace sentence-transformers
        "all-MiniLM-L6-v2": 256,
        "all-mpnet-base-v2": 384,
        # Qwen2 models
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct": 8192,
        "gte-Qwen2-1.5B-instruct": 8192,
    }
    
    # Get limit for model
    max_tokens = model_limits.get(model_name, 512)
    
    # Use 80% of max to leave room for special tokens
    optimal = int(max_tokens * 0.8)
    
    # Reasonable bounds
    return min(max(optimal, 256), 2000)