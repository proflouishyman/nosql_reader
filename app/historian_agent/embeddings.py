"""
Embedding service for the Historian Agent RAG system.

This module handles embedding generation for documents and queries using
multiple providers (local sentence-transformers or OpenAI).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import logging
import os

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


# Configuration constants
DEFAULT_PROVIDER = "local"
DEFAULT_LOCAL_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 32
DEFAULT_DIMENSION_LOCAL = 1536  # For gte-Qwen2-1.5B-instruct
DEFAULT_DIMENSION_OPENAI = 1536


class EmbeddingService:
    """
    Service for generating document and query embeddings.
    
    Supports multiple embedding providers:
    - local: HuggingFace Sentence Transformers (gte-Qwen2-1.5B-instruct by default)
    - openai: OpenAI embeddings API
    """
    
    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: str = None,
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the embedding service.
        
        Args:
            provider: Embedding provider ("local" or "openai")
            model: Name of the embedding model (defaults based on provider)
            api_key: API key for provider (OpenAI only)
            dimension: Target embedding dimension (optional, inferred from model)
            batch_size: Batch size for processing multiple texts
        """
        self.provider = provider.lower()
        self.batch_size = batch_size
        
        # Set model defaults based on provider
        if model is None:
            self.model_name = DEFAULT_LOCAL_MODEL if self.provider == "local" else DEFAULT_OPENAI_MODEL
        else:
            self.model_name = model
        
        self.dimension = dimension
        self.api_key = api_key
        
        # Initialize provider-specific client
        if self.provider == "openai":
            self._init_openai_client()
        elif self.provider == "local":
            self._init_local_model()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}. Use 'local' or 'openai'")
        
        logger.info(
            f"Initialized {self.provider} embedding service: "
            f"model={self.model_name}, dimension={self.dimension}, batch_size={batch_size}"
        )
    
    def _init_openai_client(self) -> None:
        """Initialize OpenAI client."""
        if OpenAI is None:
            raise ImportError(
                "OpenAI is required for OpenAI embeddings. "
                "Install with: pip install openai"
            )
        
        # Get API key from parameter or environment
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required but not provided")
        
        self.client = OpenAI(api_key=api_key)
        
        # Set default dimension if not specified
        if self.dimension is None:
            dimension_map = {
                "text-embedding-3-large": 3072,
                "text-embedding-3-small": 1536,
                "text-embedding-ada-002": 1536,
            }
            self.dimension = dimension_map.get(self.model_name, DEFAULT_DIMENSION_OPENAI)
        
        logger.info(f"OpenAI client initialized: model={self.model_name}, dimension={self.dimension}")
    
    def _init_local_model(self) -> None:
        """Initialize local Sentence Transformer model."""
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        try:
            # Load model with trust_remote_code for Qwen2 models
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True
            )
            
            # Get dimension from model
            if self.dimension is None:
                self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Local model loaded: {self.model_name}, "
                f"dimension={self.dimension}"
            )
        except Exception as e:
            logger.error(f"Failed to load local model {self.model_name}: {e}")
            raise
    
    # Primary interface methods (used by migration script)
    
    def embed_documents(
        self, 
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple documents (batch).
        
        This is the primary method used by the migration script.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return np.array([])
        
        try:
            if self.provider == "openai":
                return self._generate_openai_embeddings_batch(texts, show_progress)
            elif self.provider == "local":
                return self._generate_local_embeddings_batch(texts, show_progress)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        This is used by retrievers for search queries.
        
        Args:
            text: Query text to embed
            
        Returns:
            Numpy array of embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.dimension, dtype=np.float32)
        
        try:
            if self.provider == "openai":
                return self._generate_openai_embedding(text)
            elif self.provider == "local":
                return self._generate_local_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    # Legacy interface methods (for backward compatibility)
    
    def generate_embedding(self, text: str, prefix: Optional[str] = None) -> np.ndarray:
        """Legacy method name - calls embed_query."""
        if prefix:
            text = f"{prefix}{text}"
        return self.embed_query(text)
    
    def generate_embeddings_batch(
        self, 
        texts: List[str],
        prefix: Optional[str] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Legacy method name - calls embed_documents."""
        if prefix:
            texts = [f"{prefix}{text}" for text in texts]
        return self.embed_documents(texts, show_progress)
    
    # Internal implementation methods
    
    def _generate_openai_embedding(self, text: str) -> np.ndarray:
        """Generate single embedding using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimension if self.dimension else None,
            )
            
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    def _generate_local_embedding(self, text: str) -> np.ndarray:
        """Generate single embedding using local model."""
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise
    
    def _generate_openai_embeddings_batch(
        self, 
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for batch using OpenAI API."""
        embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    dimensions=self.dimension if self.dimension else None,
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                if show_progress:
                    logger.info(
                        f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} texts"
                    )
                    
            except Exception as e:
                logger.error(f"Error processing batch {i}-{i+self.batch_size}: {e}")
                # Add zero vectors for failed batch
                embeddings.extend([np.zeros(self.dimension) for _ in batch])
        
        return np.array(embeddings, dtype=np.float32)
    
    def _generate_local_embeddings_batch(
        self, 
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for batch using local model."""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Local batch embedding failed: {e}")
            raise
    
    # Utility methods
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        return self.dimension
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Clamp to [0, 1]
        return float(max(0.0, min(1.0, similarity)))
    
    def compute_similarities_batch(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarities between a query and multiple documents.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            document_embeddings: Document embeddings matrix (2D array)
            
        Returns:
            Array of similarity scores
        """
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize documents
        doc_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1  # Avoid division by zero
        docs_normalized = document_embeddings / doc_norms
        
        # Compute similarities
        similarities = np.dot(docs_normalized, query_norm)
        
        # Clamp to [0, 1]
        return np.clip(similarities, 0.0, 1.0)


def get_recommended_embedding_model(
    use_case: str = "general",
    prefer_local: bool = True,
) -> Dict[str, Any]:
    """
    Get recommended embedding model configuration for a use case.
    
    Args:
        use_case: Type of use case ("general", "semantic", "speed")
        prefer_local: Prefer local models over API-based ones (default: True for this project)
        
    Returns:
        Dictionary with provider and model configuration
    """
    if prefer_local:
        # Local models (free, private, no API costs)
        recommendations = {
            "general": {
                "provider": "local",
                "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                "dimension": 1536,
            },
            "semantic": {
                "provider": "local",
                "model": "sentence-transformers/all-mpnet-base-v2",
                "dimension": 768,
            },
            "speed": {
                "provider": "local",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
        }
    else:
        # OpenAI models (paid, but potentially faster)
        recommendations = {
            "general": {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "dimension": 3072,
            },
            "semantic": {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "dimension": 1536,  # Reduced dimension for speed
            },
            "speed": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "dimension": 1536,
            },
        }
    
    return recommendations.get(use_case, recommendations["general"])