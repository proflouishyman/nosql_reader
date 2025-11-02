"""
Vector store manager for the Historian Agent RAG system.

This module handles ChromaDB operations for storing and retrieving document embeddings.
"""

# Added so the agent can persist and query embeddings per RAG plan.  # change rationale comment

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import logging
import os
from pathlib import Path

import numpy as np

if TYPE_CHECKING:  # Added for type checking of optional chunk objects.
    from .chunking import DocumentChunk

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None  # type: ignore

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manager for ChromaDB vector store operations.
    
    Handles storing document chunks with embeddings and performing similarity searches.
    """
    
    def __init__(
        self,
        collection_name: str = "historian_documents",
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
            persist_directory = os.environ.get(
                "RAG_VECTOR_STORE_PATH",
                "/app/vector_store"
            )
        
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
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    def add_chunks(
        self,
        chunk_ids: List[Any],  # Accepts IDs or DocumentChunk objects when used with script helper.
        chunk_texts: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add document chunks with embeddings to the vector store.

        Args:
            chunk_ids: Either a list of chunk IDs or DocumentChunk objects
            chunk_texts: List of chunk text content
            embeddings: Numpy array of embeddings (n_chunks, embedding_dim)
            metadatas: List of metadata dictionaries for each chunk
        """
        if not chunk_ids:
            logger.warning("No chunks provided to add to vector store")
            return

        if chunk_texts is None or embeddings is None or metadatas is None:
            chunk_objects: List["DocumentChunk"] = chunk_ids  # type: ignore[assignment]
            chunk_ids = [chunk.chunk_id for chunk in chunk_objects]
            chunk_texts = [chunk.chunk_text for chunk in chunk_objects]
            metadatas = [chunk.metadata for chunk in chunk_objects]

            embedding_vectors = []
            for chunk in chunk_objects:
                vector = getattr(chunk, "embedding", None)
                if vector is None:
                    raise ValueError("Chunk object missing embedding for vector store upload")
                embedding_vectors.append(np.array(vector, dtype=np.float32))

            embeddings = (
                np.vstack(embedding_vectors)
                if embedding_vectors
                else np.zeros((0, 0), dtype=np.float32)
            )

        # Validate inputs
        n_chunks = len(chunk_ids)
        if len(chunk_texts) != n_chunks:
            raise ValueError("Number of texts must match number of IDs")
        if embeddings.shape[0] != n_chunks:
            raise ValueError("Number of embeddings must match number of IDs")
        if len(metadatas) != n_chunks:
            raise ValueError("Number of metadatas must match number of IDs")
        
        # Convert embeddings to list format
        embeddings_list = embeddings.tolist()
        
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
            logger.info(f"Added {n_chunks} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of LangChain Document objects with scores
        """
        try:
            # Convert embedding to list
            query_embedding_list = query_embedding.tolist()
            
            # Build where clause from filters
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k,
                where=where_clause,
            )
            
            # Convert to LangChain Documents
            documents = []
            if results["documents"] and results["documents"][0]:
                for i, doc_text in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    
                    # Add score (convert distance to similarity)
                    metadata["score"] = 1 - distance  # Cosine distance to similarity
                    metadata["chunk_id"] = results["ids"][0][i]
                    
                    documents.append(
                        Document(
                            page_content=doc_text,
                            metadata=metadata,
                        )
                    )
            
            logger.info(f"Retrieved {len(documents)} documents from vector search")
            return documents

        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise

    # Added adapter returning dict payloads to satisfy VectorRetriever expectations.  # change rationale comment
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return raw search payloads expected by retrievers."""

        documents = self.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
        )

        results: List[Dict[str, Any]] = []
        for doc in documents:
            metadata = dict(doc.metadata)
            score = float(metadata.get("score", 0.0))
            results.append(
                {
                    "chunk_id": metadata.get("chunk_id"),
                    "content": doc.page_content,
                    "metadata": metadata,
                    "score": score,
                    "distance": 1.0 - score,
                }
            )

        return results
    
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
                where={"source_document_id": {"$eq": document_id}}
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} chunks for document {document_id}"
                )
        except Exception as e:
            logger.error(f"Error deleting chunks by document ID: {e}")
            raise
    
    def get_chunk(self, chunk_id: str) -> Optional[Document]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Chunk ID to retrieve
            
        Returns:
            LangChain Document or None if not found
        """
        try:
            results = self.collection.get(ids=[chunk_id])
            
            if results["documents"]:
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                return Document(
                    page_content=results["documents"][0],
                    metadata=metadata,
                )
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def count_chunks(self) -> int:
        """Get total number of chunks in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error counting chunks: {e}")
            return 0
    
    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """
        Peek at some chunks in the collection.
        
        Args:
            limit: Number of chunks to retrieve
            
        Returns:
            Dictionary with chunk data
        """
        try:
            return self.collection.peek(limit=limit)
        except Exception as e:
            logger.error(f"Error peeking at collection: {e}")
            return {}
    
    def reset_collection(self) -> None:
        """Delete all data in the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            logger.warning(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            count = self.count_chunks()
            peek_data = self.peek(limit=1)
            
            info = {
                "name": self.collection_name,
                "count": count,
                "persist_directory": self.persist_directory,
            }
            
            # Add embedding dimension if available
            if peek_data.get("embeddings"):
                info["embedding_dimension"] = len(peek_data["embeddings"][0])
            
            return info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"name": self.collection_name, "error": str(e)}
    
    def search_by_text(
        self,
        query_text: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Search using query text (ChromaDB will generate embedding).
        
        Args:
            query_text: Query text
            k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of LangChain Documents
        """
        if self.embedding_function is None:
            raise ValueError(
                "Collection must have embedding function to search by text"
            )
        
        try:
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                where=where_clause,
            )
            
            documents = []
            if results["documents"] and results["documents"][0]:
                for i, doc_text in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 0
                    
                    metadata["score"] = 1 - distance
                    metadata["chunk_id"] = results["ids"][0][i]
                    
                    documents.append(
                        Document(
                            page_content=doc_text,
                            metadata=metadata,
                        )
                    )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            raise


def create_vector_store_from_documents(
    documents: List[Dict[str, Any]],
    embedding_service: Any,
    chunker: Any,
    collection_name: str = "historian_documents",
    persist_directory: Optional[str] = None,
    batch_size: int = 100,
) -> VectorStoreManager:
    """
    Create and populate a vector store from documents.
    
    Args:
        documents: List of MongoDB documents
        embedding_service: EmbeddingService instance
        chunker: DocumentChunker instance
        collection_name: Name for the collection
        persist_directory: Where to persist the database
        batch_size: Batch size for processing
        
    Returns:
        Initialized VectorStoreManager
    """
    # Initialize vector store
    vector_store = VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    
    # Process documents in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        logger.info(
            f"Processing batch {i // batch_size + 1} "
            f"({i + 1}-{min(i + batch_size, len(documents))} of {len(documents)})"
        )
        
        # Chunk documents
        chunks = chunker.chunk_documents_batch(batch)
        
        if not chunks:
            continue
        
        # Extract data
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        chunk_texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings
        embeddings = embedding_service.generate_embeddings_batch(
            chunk_texts,
            show_progress=True,
        )
        
        # Add to vector store
        vector_store.add_chunks(chunk_ids, chunk_texts, embeddings, metadatas)
    
    logger.info(f"Vector store created with {vector_store.count_chunks()} chunks")
    return vector_store


# Added factory helper to align migration script with runtime configuration.  # change rationale comment
def get_vector_store(
    store_type: str = "chroma",
    collection: Optional[Any] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = "historian_document_chunks",
) -> VectorStoreManager:
    """Factory helper to initialise the configured vector store."""

    store_type = (store_type or "chroma").lower()
    if store_type != "chroma":
        raise ValueError(f"Unsupported vector store type: {store_type}")

    return VectorStoreManager(
        collection_name=collection_name,
        persist_directory=persist_directory or os.environ.get("CHROMA_PERSIST_DIRECTORY"),
    )
