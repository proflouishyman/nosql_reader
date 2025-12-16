"""
Vector store manager for the Historian Agent RAG system.

This module handles ChromaDB operations for storing and retrieving document embeddings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import os
from pathlib import Path

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None  # type: ignore

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_COLLECTION_NAME = "historian_documents"
DEFAULT_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIRECTORY", "/data/chroma_db/persist")
DEFAULT_DISTANCE_METRIC = "cosine"


class VectorStoreManager:
    """
    Manager for ChromaDB vector store operations.
    
    Handles storing document chunks with embeddings and performing similarity searches.
    """
    
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ):
        """
        Initialize the vector store manager.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
            embedding_function: Optional custom embedding function
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install with: pip install chromadb"
            )
        
        self.collection_name = collection_name
        
        # Set up persistence directory
        if persist_directory is None:
            persist_directory = DEFAULT_PERSIST_DIR
        
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Initialize or get collection
        self.embedding_function = embedding_function
        self.collection = self._get_or_create_collection()
        
        logger.info(
            f"Vector store initialized: {collection_name} at {persist_directory}"
        )
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": DEFAULT_DISTANCE_METRIC},
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    def add_chunks(self, chunks: List[Any]) -> None:
        """
        Add document chunks with embeddings to the vector store.
        
        This method accepts Chunk objects from chunking.py and extracts
        the necessary fields for ChromaDB storage.
        
        Args:
            chunks: List of Chunk objects (from chunking.py)
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        # Extract data from Chunk objects
        chunk_ids = []
        chunk_texts = []
        embeddings = []
        metadatas = []
        
        for chunk in chunks:
            # Validate that chunk has embedding
            if chunk.embedding is None:
                logger.warning(f"Chunk {chunk.chunk_id} has no embedding, skipping")
                continue
            
            chunk_ids.append(chunk.chunk_id)
            chunk_texts.append(chunk.text)  # Use .text field
            embeddings.append(chunk.embedding)
            
            # Prepare metadata
            metadata = chunk.metadata.copy() if chunk.metadata else {}
            metadata["document_id"] = chunk.document_id
            metadata["chunk_index"] = chunk.chunk_index
            metadata["token_count"] = chunk.token_count
            metadatas.append(metadata)
        
        if not chunk_ids:
            logger.warning("No valid chunks with embeddings to add")
            return
        
        # Convert embeddings to numpy array then to list
        embeddings_array = np.array(embeddings)
        embeddings_list = embeddings_array.tolist()
        
        # Clean metadata to ensure ChromaDB compatibility
        cleaned_metadatas = []
        for metadata in metadatas:
            cleaned = {}
            for key, value in metadata.items():
                # ChromaDB only supports str, int, float, bool
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    cleaned[key] = ", ".join(str(v) for v in value)
                elif value is None:
                    cleaned[key] = ""
                else:
                    cleaned[key] = str(value)
            cleaned_metadatas.append(cleaned)
        
        try:
            # Add to collection
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings_list,
                documents=chunk_texts,
                metadatas=cleaned_metadatas,
            )
            logger.info(f"Added {len(chunk_ids)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector store.
        
        TRUTHINESS-SAFE: Never tests arrays for truthiness.
        
        Args:
            query_embedding: Query embedding vector (numpy array or list)
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of dictionaries with keys: chunk_id, content, metadata, score, distance
        """
        try:
            # Convert numpy to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding_list = query_embedding.tolist()
            else:
                query_embedding_list = query_embedding
            
            # Build where clause from filters
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
            
            # TRUTHINESS-SAFE: Check length without testing truthiness
            ids = results.get("ids")
            docs = results.get("documents")
            metas = results.get("metadatas")
            dists = results.get("distances")
            
            # Safe check - no truthiness on arrays
            if ids is None or len(ids) == 0 or len(ids[0]) == 0:
                return []
            
            # Build output list
            output = []
            for i in range(len(ids[0])):
                output.append({
                    "chunk_id": ids[0][i],
                    "content": docs[0][i] if docs is not None and len(docs) > 0 else None,
                    "metadata": metas[0][i] if metas is not None and len(metas) > 0 else {},
                    "score": 1.0 - dists[0][i] if dists is not None and len(dists) > 0 else 0.0,
                    "distance": dists[0][i] if dists is not None and len(dists) > 0 else 0.0,
                })
            
            logger.debug(f"Retrieved {len(output)} results from vector search")
            return output
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search and return LangChain Documents.
        
        This is an alternative interface that returns LangChain Document objects.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of LangChain Document objects with scores
        """
        search_results = self.search(query_embedding, k, filters)
        
        documents = []
        for result in search_results:
            metadata = result["metadata"].copy()
            metadata["chunk_id"] = result["chunk_id"]
            metadata["score"] = result["score"]
            metadata["distance"] = result["distance"]
            
            documents.append(
                Document(
                    page_content=result["content"],
                    metadata=metadata,
                )
            )
        
        return documents
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from filters dictionary.
        
        Args:
            filters: Dictionary of filter conditions
            
        Returns:
            ChromaDB where clause
        """
        where_conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values - use $in operator
                where_conditions.append({key: {"$in": value}})
            elif isinstance(value, dict):
                # Range query (e.g., date range)
                if "$gte" in value or "$lte" in value or "$gt" in value or "$lt" in value:
                    where_conditions.append({key: value})
            else:
                # Exact match
                where_conditions.append({key: {"$eq": value}})
        
        # Combine conditions with AND
        if len(where_conditions) == 1:
            return where_conditions[0]
        elif len(where_conditions) > 1:
            return {"$and": where_conditions}
        else:
            return {}
    
    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """
        Delete chunks from the vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        try:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks from vector store")
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            raise
    
    def delete_by_document_id(self, document_id: str) -> None:
        """
        Delete all chunks belonging to a specific document.
        
        Args:
            document_id: Source document ID
        """
        try:
            # Query for chunks with this document ID
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}}
            )
            
            # TRUTHINESS-SAFE: Check length directly
            if len(results.get("ids", [])) > 0:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} chunks for document {document_id}"
                )
        except Exception as e:
            logger.error(f"Error deleting chunks by document ID: {e}")
            raise
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by ID.
        
        TRUTHINESS-SAFE: Never tests arrays for truthiness.
        
        Args:
            chunk_id: Chunk ID to retrieve
            
        Returns:
            Dictionary with chunk data or None if not found
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            # TRUTHINESS-SAFE: Check length directly - works for both list and numpy array
            ids = results.get("ids")
            if ids is None or len(ids) == 0:
                return None
            
            docs = results.get("documents")
            metas = results.get("metadatas")
            embs = results.get("embeddings")
            
            # Safe extraction with length checks
            content = docs[0] if docs is not None and len(docs) > 0 else None
            metadata = metas[0] if metas is not None and len(metas) > 0 else {}
            embedding = embs[0] if embs is not None and len(embs) > 0 else None
            
            return {
                "chunk_id": ids[0],
                "content": content,
                "metadata": metadata,
                "embedding": embedding,
            }
            
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats: total_chunks, collection_name, etc.
        """
        try:
            count = self.collection.count()
            return {
                "type": "chromadb",
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "type": "chromadb",
                "collection_name": self.collection_name,
                "error": str(e),
            }
    
    def reset(self) -> None:
        """
        Reset the vector store by deleting and recreating the collection.
        
        WARNING: This will delete all data in the collection!
        """
        logger.warning(f"Resetting vector store collection: {self.collection_name}")
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Error deleting collection (may not exist): {e}")
        
        # Recreate collection
        self.collection = self._get_or_create_collection()
        logger.info("Vector store reset complete")


def get_vector_store(
    store_type: str = "chroma",
    collection: Optional[Any] = None,
    **kwargs
) -> VectorStoreManager:
    """
    Factory function to get a vector store instance.
    
    Args:
        store_type: Type of vector store ("chroma" or "mongo")
        collection: Optional MongoDB collection (for mongo store type)
        **kwargs: Additional arguments passed to VectorStoreManager
        
    Returns:
        VectorStoreManager instance
    """
    if store_type == "chroma":
        return VectorStoreManager(**kwargs)
    elif store_type == "mongo":
        raise NotImplementedError("MongoDB vector search not yet implemented")
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")