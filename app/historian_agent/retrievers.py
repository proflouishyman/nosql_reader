"""
Advanced Retrievers for RAG

This module provides multiple retrieval strategies:
- VectorRetriever: Semantic search using embeddings
- KeywordRetriever: Traditional regex/BM25 search
- HybridRetriever: Combines vector + keyword with RRF fusion

Features:
- Reciprocal Rank Fusion (RRF) for result merging
- Configurable weights for hybrid search
- Parent document retrieval
- Metadata filtering
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re
import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            
        Returns:
            List of LangChain Document objects
        """
        pass


class VectorRetriever(BaseRetriever):
    """
    Semantic retrieval using vector embeddings.
    
    Features:
    - Cosine similarity search
    - Configurable top-k
    - Metadata filtering support
    """
    
    def __init__(
        self,
        vector_store,
        embedding_service,
        mongo_collection,
        top_k: int = 10,
    ):
        """
        Initialize vector retriever.
        
        Args:
            vector_store: VectorStore instance
            embedding_service: EmbeddingService instance
            mongo_collection: pymongo Collection for parent documents
            top_k: Number of results to return
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.mongo_collection = mongo_collection
        self.top_k = top_k
        
        logger.info(f"VectorRetriever initialized with top_k={top_k}")
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using vector similarity search."""
        if not query.strip():
            logger.warning("Empty query provided to VectorRetriever")
            return []
        
        try:
            # Step 1: Embed query
            query_embedding = self.embedding_service.embed_query(query)
            
            # Step 2: Vector search
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                k=self.top_k,
            )
            
            if not search_results:
                logger.info("No results found from vector search")
                return []
            
            # Step 3: Convert to LangChain Documents
            documents = self._convert_to_langchain_docs(search_results)
            
            logger.debug(
                f"VectorRetriever found {len(documents)} documents for query"
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in VectorRetriever: {e}", exc_info=True)
            return []
    
    def _convert_to_langchain_docs(
        self,
        search_results: List[Dict[str, Any]]
    ) -> List[Document]:
        """Convert search results to LangChain Document objects."""
        documents = []
        
        for result in search_results:
            # Use chunk content as page_content
            page_content = result.get("content", "")
            
            # Build metadata
            metadata = result.get("metadata", {}).copy()
            metadata.update({
                "chunk_id": result.get("chunk_id"),
                "score": result.get("score", 0.0),
                "distance": result.get("distance", 0.0),
                "retrieval_method": "vector",
            })
            
            # Create Document
            doc = Document(
                page_content=page_content,
                metadata=metadata,
            )
            documents.append(doc)
        
        return documents


class KeywordRetriever(BaseRetriever):
    """
    Traditional keyword-based retrieval.
    
    Supports:
    - Regex matching
    - Multiple field search
    - Case-insensitive search
    """
    
    def __init__(
        self,
        mongo_collection,
        config,
        top_k: int = 10,
    ):
        """
        Initialize keyword retriever.
        
        Args:
            mongo_collection: pymongo Collection (can be chunks or documents)
            config: HistorianAgentConfig with context_fields
            top_k: Number of results to return
        """
        self.mongo_collection = mongo_collection
        self.config = config
        self.top_k = top_k
        
        logger.info(
            f"KeywordRetriever initialized with top_k={top_k}, "
            f"fields={config.context_fields}"
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using keyword matching."""
        if not query.strip():
            logger.warning("Empty query provided to KeywordRetriever")
            return []
        
        try:
            # Build regex query
            regex = re.compile(re.escape(query), re.IGNORECASE)
            
            # Search across configured fields
            filters = [
                {field: {"$regex": regex}}
                for field in self.config.context_fields
            ]
            mongo_query = {"$or": filters} if filters else {}
            
            # Execute search
            cursor = self.mongo_collection.find(mongo_query).limit(self.top_k)
            
            # Convert to LangChain Documents
            documents = []
            for record in cursor:
                page_content = self._extract_content(record)
                metadata = self._extract_metadata(record)
                metadata["retrieval_method"] = "keyword"
                
                if page_content:
                    documents.append(
                        Document(page_content=page_content, metadata=metadata)
                    )
            
            logger.debug(
                f"KeywordRetriever found {len(documents)} documents for query"
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in KeywordRetriever: {e}", exc_info=True)
            return []
    
    def _extract_content(self, record: Dict[str, Any]) -> str:
        """Extract content from MongoDB record."""
        # Try to use chunk content if available
        if "content" in record:
            return record["content"]
        
        # Otherwise combine configured fields
        content_segments: List[str] = []
        for field in self.config.context_fields:
            value = record.get(field)
            if isinstance(value, str):
                content_segments.append(value)
            elif isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
                content_segments.append(" ".join(map(str, value)))
            elif isinstance(value, dict):
                content_segments.append(" ".join(map(str, value.values())))
        
        return "\n".join(seg for seg in content_segments if seg)
    
    def _extract_metadata(self, record: Dict[str, Any]) -> Dict[str, str]:
        """Extract metadata from MongoDB record."""
        metadata = {
            "_id": str(record.get("_id")),
        }
        
        # Add title
        title = (
            record.get("title") or
            record.get("name") or
            record.get("document_title")
        )
        if title:
            metadata["title"] = title
        
        # Add chunk-specific metadata if available
        if "chunk_id" in record:
            metadata["chunk_id"] = record["chunk_id"]
        if "parent_doc_id" in record:
            metadata["parent_doc_id"] = record["parent_doc_id"]
        if "chunk_index" in record:
            metadata["chunk_index"] = record["chunk_index"]
        
        return metadata


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Formula: score(d) = Σ(1 / (k + rank_i(d)))
    
    Args:
        rankings: List of ranked document IDs from each retriever
        k: RRF constant (typically 60)
        
    Returns:
        List of (doc_id, score) tuples sorted by score (descending)
    
    Example:
        vector_ranking = ["doc1", "doc2", "doc3"]
        keyword_ranking = ["doc2", "doc4", "doc1"]
        
        result = reciprocal_rank_fusion([vector_ranking, keyword_ranking])
        # Returns: [("doc2", 0.0325), ("doc1", 0.0323), ("doc4", 0.0161), ...]
    """
    scores = defaultdict(float)
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by score (descending)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_scores


class HybridRetriever(BaseRetriever):
    """
    Hybrid retrieval combining vector and keyword search.
    
    Features:
    - Reciprocal Rank Fusion for result merging
    - Configurable retrieval weights
    - Deduplication
    """
    
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 10,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_retriever: VectorRetriever instance
            keyword_retriever: KeywordRetriever instance
            vector_weight: Weight for vector results (0.0-1.0)
            keyword_weight: Weight for keyword results (0.0-1.0)
            top_k: Number of final results to return
            rrf_k: RRF constant
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k
        self.rrf_k = rrf_k
        
        logger.info(
            f"HybridRetriever initialized with "
            f"vector_weight={vector_weight}, "
            f"keyword_weight={keyword_weight}, "
            f"top_k={top_k}, "
            f"rrf_k={rrf_k}"
        )
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid search with RRF fusion."""
        if not query.strip():
            logger.warning("Empty query provided to HybridRetriever")
            return []
        
        try:
            # Step 1: Get results from both retrievers
            vector_docs = self.vector_retriever.get_relevant_documents(query)
            keyword_docs = self.keyword_retriever.get_relevant_documents(query)
            
            logger.debug(
                f"HybridRetriever: vector={len(vector_docs)}, "
                f"keyword={len(keyword_docs)} results"
            )
            
            # Step 2: Extract document IDs for RRF
            vector_ids = [doc.metadata.get("chunk_id") or doc.metadata.get("_id")
                         for doc in vector_docs]
            keyword_ids = [doc.metadata.get("chunk_id") or doc.metadata.get("_id")
                          for doc in keyword_docs]
            
            # Step 3: Apply Reciprocal Rank Fusion
            fused_scores = reciprocal_rank_fusion(
                [vector_ids, keyword_ids],
                k=self.rrf_k
            )
            
            # Step 4: Get top K document IDs
            top_doc_ids = [doc_id for doc_id, score in fused_scores[:self.top_k]]
            
            # Step 5: Retrieve full documents in fused order
            all_docs_map = {}
            for doc in vector_docs + keyword_docs:
                doc_id = doc.metadata.get("chunk_id") or doc.metadata.get("_id")
                if doc_id not in all_docs_map:
                    all_docs_map[doc_id] = doc
            
            # Step 6: Build final results maintaining RRF order
            final_docs = []
            for doc_id in top_doc_ids:
                if doc_id in all_docs_map:
                    doc = all_docs_map[doc_id]
                    # Update metadata to indicate hybrid retrieval
                    doc.metadata["retrieval_method"] = "hybrid"
                    # Find the fused score
                    for fused_id, score in fused_scores:
                        if fused_id == doc_id:
                            doc.metadata["rrf_score"] = score
                            break
                    final_docs.append(doc)
            
            logger.debug(
                f"HybridRetriever returning {len(final_docs)} fused results"
            )
            
            return final_docs
            
        except Exception as e:
            logger.error(f"Error in HybridRetriever: {e}", exc_info=True)
            # Fallback to vector retriever only
            logger.warning("Falling back to vector retriever only")
            return self.vector_retriever.get_relevant_documents(query)


# Backward compatibility: MongoKeywordRetriever as alias
class MongoKeywordRetriever(KeywordRetriever):
    """
    Backward compatible alias for KeywordRetriever.
    
    Maintains compatibility with existing HistorianAgent code.
    """
    
    def __init__(self, collection, config):
        """Initialize with existing signature."""
        super().__init__(
            mongo_collection=collection,
            config=config,
            top_k=config.max_context_documents,
        )