#!/usr/bin/env python3
"""
Re-ranking Module for RAG

Implements cross-encoder reranking to improve retrieval quality.
Takes initial retrieval results and rescores them based on query-document relevance.

Architecture:
  Initial Results → Cross-Encoder Scoring → Temporal Boost → Entity Overlap → Final Ranking

Features:
- Cross-encoder models for semantic relevance
- Temporal relevance boosting (recent documents scored higher)
- Entity overlap scoring
- Configurable weights for different signals

Usage:
  from reranking import DocumentReranker
  
  reranker = DocumentReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
  reranked = reranker.rerank(query, documents, top_k=5)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TEMPORAL_WEIGHT = 0.1
DEFAULT_ENTITY_WEIGHT = 0.05
DEFAULT_CROSS_ENCODER_WEIGHT = 0.85


@dataclass
class RerankedDocument:
    """Document with reranking scores."""
    document: Any  # LangChain Document object
    original_score: float
    cross_encoder_score: float
    temporal_score: float
    entity_overlap_score: float
    final_score: float
    rank: int


class DocumentReranker:
    """
    Rerank retrieved documents using cross-encoder and other signals.
    
    Cross-encoders jointly encode query and document, providing better
    relevance scores than bi-encoders (embeddings) alone.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        cross_encoder_weight: float = DEFAULT_CROSS_ENCODER_WEIGHT,
        temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
        entity_weight: float = DEFAULT_ENTITY_WEIGHT,
        use_gpu: bool = False,
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name
            cross_encoder_weight: Weight for cross-encoder score
            temporal_weight: Weight for temporal relevance boost
            entity_weight: Weight for entity overlap
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.cross_encoder_weight = cross_encoder_weight
        self.temporal_weight = temporal_weight
        self.entity_weight = entity_weight
        
        # Validate weights sum to 1.0
        total_weight = cross_encoder_weight + temporal_weight + entity_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(
                f"Weights sum to {total_weight:.3f}, normalizing to 1.0"
            )
            norm = 1.0 / total_weight
            self.cross_encoder_weight *= norm
            self.temporal_weight *= norm
            self.entity_weight *= norm
        
        logger.info(f"Initializing DocumentReranker with {model_name}")
        logger.info(f"  Cross-encoder weight: {self.cross_encoder_weight:.3f}")
        logger.info(f"  Temporal weight: {self.temporal_weight:.3f}")
        logger.info(f"  Entity weight: {self.entity_weight:.3f}")
        
        # Load cross-encoder model
        try:
            device = "cuda" if use_gpu else "cpu"
            self.model = CrossEncoder(model_name, device=device)
            logger.info(f"✓ Loaded cross-encoder on {device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Any],
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> List[RerankedDocument]:
        """
        Rerank documents using multiple signals.
        
        Args:
            query: User query string
            documents: List of LangChain Document objects
            top_k: Number of top documents to return (None = all)
            return_scores: Whether to include detailed scores
            
        Returns:
            List of RerankedDocument objects sorted by final score
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        logger.info(f"Reranking {len(documents)} documents for query: {query[:100]}...")
        
        # Step 1: Cross-encoder scoring
        cross_encoder_scores = self._compute_cross_encoder_scores(query, documents)
        
        # Step 2: Temporal relevance boost
        temporal_scores = self._compute_temporal_scores(documents)
        
        # Step 3: Entity overlap scoring
        entity_scores = self._compute_entity_overlap_scores(query, documents)
        
        # Step 4: Combine scores
        reranked_docs = []
        for i, doc in enumerate(documents):
            # Get original score from metadata
            original_score = doc.metadata.get("score", 0.0)
            
            # Compute weighted final score
            final_score = (
                self.cross_encoder_weight * cross_encoder_scores[i] +
                self.temporal_weight * temporal_scores[i] +
                self.entity_weight * entity_scores[i]
            )
            
            reranked_doc = RerankedDocument(
                document=doc,
                original_score=original_score,
                cross_encoder_score=cross_encoder_scores[i],
                temporal_score=temporal_scores[i],
                entity_overlap_score=entity_scores[i],
                final_score=final_score,
                rank=0,  # Will be set after sorting
            )
            reranked_docs.append(reranked_doc)
        
        # Sort by final score
        reranked_docs.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign ranks
        for rank, doc in enumerate(reranked_docs, start=1):
            doc.rank = rank
        
        # Truncate to top_k
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]
        
        logger.info(f"✓ Reranking complete, returning top {len(reranked_docs)} documents")
        
        # Log score distribution
        if reranked_docs:
            scores = [d.final_score for d in reranked_docs]
            logger.info(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")
            logger.info(f"  Mean score: {np.mean(scores):.3f}")
        
        return reranked_docs
    
    def _compute_cross_encoder_scores(
        self,
        query: str,
        documents: List[Any],
    ) -> List[float]:
        """
        Score query-document pairs using cross-encoder.
        
        Cross-encoders jointly encode query and document for better
        relevance scores than separate embeddings.
        """
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            content = doc.page_content
            pairs.append([query, content])
        
        # Batch predict
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            # Convert to Python floats and normalize to [0, 1]
            scores = [float(s) for s in scores]
            
            # Normalize scores to [0, 1] using sigmoid
            scores = [1.0 / (1.0 + np.exp(-s)) for s in scores]
            
            logger.debug(f"Cross-encoder scores: {scores[:3]}...")
            return scores
            
        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            # Fallback to uniform scores
            return [0.5] * len(documents)
    
    def _compute_temporal_scores(
        self,
        documents: List[Any],
    ) -> List[float]:
        """
        Boost scores for more recent documents.
        
        Uses exponential decay: score = exp(-λ * years_ago)
        """
        scores = []
        current_year = datetime.now().year
        
        for doc in documents:
            # Try to extract date from metadata
            date_str = doc.metadata.get("date", "")
            
            if date_str:
                # Try to parse year
                year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
                if year_match:
                    year = int(year_match.group())
                    years_ago = current_year - year
                    
                    # Exponential decay with λ = 0.05 (half-life ~14 years)
                    score = np.exp(-0.05 * max(0, years_ago))
                    scores.append(float(score))
                    continue
            
            # No date or parse failed - neutral score
            scores.append(0.5)
        
        logger.debug(f"Temporal scores: {scores[:3]}...")
        return scores
    
    def _compute_entity_overlap_scores(
        self,
        query: str,
        documents: List[Any],
    ) -> List[float]:
        """
        Score based on entity overlap between query and document.
        
        Simple heuristic: count shared capitalized words, names, dates.
        """
        # Extract entities from query (simple heuristic)
        query_entities = self._extract_entities(query)
        
        if not query_entities:
            # No entities to match
            return [0.5] * len(documents)
        
        scores = []
        for doc in documents:
            content = doc.page_content
            doc_entities = self._extract_entities(content)
            
            # Compute overlap
            if doc_entities:
                overlap = len(query_entities & doc_entities)
                score = overlap / len(query_entities)
                scores.append(min(1.0, score))
            else:
                scores.append(0.0)
        
        logger.debug(f"Entity overlap scores: {scores[:3]}...")
        return scores
    
    def _extract_entities(self, text: str) -> set:
        """
        Extract entities using simple heuristics.
        
        Looks for:
        - Capitalized words (names, places)
        - Dates (YYYY, MM/DD/YYYY)
        - Railroad-specific terms
        """
        entities = set()
        
        # Capitalized words (2+ letters)
        cap_words = re.findall(r'\b[A-Z][a-z]{1,}\b', text)
        entities.update(cap_words)
        
        # Years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        entities.update(years)
        
        # Common railroad terms
        railroad_terms = [
            "locomotive", "engine", "train", "car", "track",
            "accident", "derailment", "collision", "injury",
            "B&O", "Baltimore", "Ohio", "Railroad", "Railway"
        ]
        
        text_lower = text.lower()
        for term in railroad_terms:
            if term.lower() in text_lower:
                entities.add(term)
        
        return entities
    
    def get_reranked_documents_only(
        self,
        reranked_results: List[RerankedDocument],
    ) -> List[Any]:
        """
        Extract just the Document objects from reranked results.
        
        Useful for passing to downstream components that expect Document objects.
        """
        return [r.document for r in reranked_results]


# ============================================================================
# Utility Functions
# ============================================================================

def print_reranking_results(
    reranked_docs: List[RerankedDocument],
    top_n: int = 5,
):
    """
    Pretty-print reranking results for debugging.
    
    Args:
        reranked_docs: List of RerankedDocument objects
        top_n: Number of top results to display
    """
    print("\n" + "="*70)
    print("RERANKING RESULTS")
    print("="*70)
    
    for doc in reranked_docs[:top_n]:
        print(f"\nRank {doc.rank}: Final Score = {doc.final_score:.3f}")
        print(f"  Original: {doc.original_score:.3f}")
        print(f"  Cross-Encoder: {doc.cross_encoder_score:.3f}")
        print(f"  Temporal: {doc.temporal_score:.3f}")
        print(f"  Entity Overlap: {doc.entity_overlap_score:.3f}")
        print(f"  Document ID: {doc.document.metadata.get('document_id', 'N/A')[:16]}...")
        print(f"  Content: {doc.document.page_content[:100]}...")
    
    print("="*70 + "\n")


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Test reranking with sample data."""
    import sys
    
    # This would normally come from rag_query_handler
    print("Note: This is a module meant to be imported by adversarial_rag.py")
    print("To test reranking, use: python adversarial_rag.py --test-rerank")
    return 0


if __name__ == "__main__":
    sys.exit(main())