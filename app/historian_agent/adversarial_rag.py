#!/usr/bin/env python3
"""
Adversarial RAG - Enhanced Query Processing with Two-Pass Validation

This script orchestrates a complete RAG pipeline with:
1. Initial retrieval (rag_query_handler)
2. Cross-encoder reranking (reranking)
3. Answer generation (via rag_query_handler)
4. Adversarial critique and validation

Architecture:
  Query → RAG Retrieval → Reranking → Answer Generation → Critique → Final Response

The adversarial layer acts as a "second opinion" to:
- Verify citation accuracy
- Flag unsupported claims
- Identify contradictions in sources
- Score confidence (0-1)
- Suggest revisions if needed

Usage:
  # Basic query with adversarial validation
  python adversarial_rag.py "What caused train accidents in the 1920s?"
  
  # With custom parameters
  python adversarial_rag.py "safety violations" --top-k 10 --rerank-top-k 5
  
  # Skip critique for faster results
  python adversarial_rag.py "quick question" --skip-critique
  
  # Test reranking only
  python adversarial_rag.py --test-rerank "sample query"
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_query_handler import RAGQueryHandler, RAGQueryResult
from reranking import DocumentReranker, print_reranking_results

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_RERANK_TOP_K = 10  # Rerank top 10 from initial retrieval
DEFAULT_FINAL_TOP_K = 5    # Use top 5 after reranking for answer

# Critique configuration
DEFAULT_CRITIQUE_MODEL = "gpt-oss:20b"  # Same model for consistency
DEFAULT_CRITIQUE_TEMPERATURE = 0.3  # Lower temp for more analytical critique

# Reranking weights
DEFAULT_CROSS_ENCODER_WEIGHT = 0.85
DEFAULT_TEMPORAL_WEIGHT = 0.10
DEFAULT_ENTITY_WEIGHT = 0.05

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CritiqueResult:
    """Result from adversarial critique."""
    confidence_score: float  # 0-1
    issues_found: List[str]
    unsupported_claims: List[str]
    citation_errors: List[str]
    contradictions: List[str]
    recommendation: str  # "accept", "revise", "reject"
    revised_answer: Optional[str] = None


@dataclass
class AdversarialRAGResult:
    """Complete result from adversarial RAG pipeline."""
    query: str
    initial_answer: str
    final_answer: str
    sources: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]
    reranking_stats: Dict[str, Any]
    critique: CritiqueResult
    conversation_id: str
    total_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert critique to dict
        result['critique'] = asdict(self.critique)
        return result


class AdversarialRAGHandler:
    """
    Orchestrates the complete adversarial RAG pipeline.
    """
    
    def __init__(
        self,
        # RAG handler params
        mongo_uri: str = None,
        db_name: str = None,
        embedding_provider: str = None,
        embedding_model: str = None,
        llm_provider: str = None,
        llm_model: str = None,
        ollama_base_url: str = None,
        temperature: float = None,
        
        # Retrieval params
        initial_top_k: int = None,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        max_context_tokens: int = None,
        
        # Reranking params
        cross_encoder_model: str = None,
        cross_encoder_weight: float = DEFAULT_CROSS_ENCODER_WEIGHT,
        temporal_weight: float = DEFAULT_TEMPORAL_WEIGHT,
        entity_weight: float = DEFAULT_ENTITY_WEIGHT,
        
        # Critique params
        critique_model: str = DEFAULT_CRITIQUE_MODEL,
        critique_temperature: float = DEFAULT_CRITIQUE_TEMPERATURE,
        skip_critique: bool = False,
    ):
        """
        Initialize adversarial RAG handler.
        
        Args:
            initial_top_k: Number of documents to retrieve initially (default from RAG handler)
            rerank_top_k: Number of documents to rerank (should be >= final_top_k)
            final_top_k: Number of documents to use for answer generation
            skip_critique: If True, skip adversarial validation step
        """
        logger.info("Initializing Adversarial RAG Handler")
        
        self.initial_top_k = initial_top_k
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k
        self.skip_critique = skip_critique
        self.critique_model = critique_model
        self.critique_temperature = critique_temperature
        
        # Validate top_k hierarchy
        if self.rerank_top_k < self.final_top_k:
            logger.warning(
                f"rerank_top_k ({self.rerank_top_k}) < final_top_k ({self.final_top_k}), "
                f"adjusting rerank_top_k to {self.final_top_k}"
            )
            self.rerank_top_k = self.final_top_k
        
        # Initialize RAG handler
        logger.info("Initializing RAG query handler...")
        self.rag_handler = RAGQueryHandler(
            mongo_uri=mongo_uri,
            db_name=db_name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_base_url=ollama_base_url,
            temperature=temperature,
            top_k=initial_top_k or self.rerank_top_k,  # Retrieve more for reranking
            max_context_tokens=max_context_tokens,
        )
        
        # Initialize reranker
        logger.info("Initializing document reranker...")
        self.reranker = DocumentReranker(
            model_name=cross_encoder_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder_weight=cross_encoder_weight,
            temporal_weight=temporal_weight,
            entity_weight=entity_weight,
            use_gpu=False,  # Set to True if GPU available
        )
        
        logger.info("✓ Adversarial RAG Handler initialized")
    
    def process_query(
        self,
        question: str,
        conversation_id: str = None,
        chat_history: List[Dict[str, str]] = None,
    ) -> AdversarialRAGResult:
        """
        Process a query through the complete adversarial pipeline.
        
        Steps:
        1. Initial retrieval (more documents than needed)
        2. Rerank using cross-encoder
        3. Generate answer using top-k reranked documents
        4. Critique answer for accuracy and citations
        5. Revise if needed
        
        Args:
            question: User's question
            conversation_id: Conversation identifier
            chat_history: Previous conversation turns
            
        Returns:
            AdversarialRAGResult with answer and validation
        """
        start_time = time.time()
        
        logger.info(f"\n{'='*70}")
        logger.info("ADVERSARIAL RAG PIPELINE")
        logger.info(f"{'='*70}")
        logger.info(f"Query: {question}")
        
        # ====================================================================
        # STEP 1: Initial Retrieval
        # ====================================================================
        logger.info("\n[1/4] Initial retrieval...")
        
        # Use rag_handler directly but don't call process_query yet
        # We need to intercept after retrieval but before answer generation
        documents = self.rag_handler.hybrid_retriever.get_relevant_documents(question)
        
        initial_retrieval_time = time.time() - start_time
        logger.info(f"  Retrieved {len(documents)} documents in {initial_retrieval_time:.2f}s")
        
        # ====================================================================
        # STEP 2: Reranking
        # ====================================================================
        logger.info(f"\n[2/4] Reranking top {self.rerank_top_k} documents...")
        
        rerank_start = time.time()
        
        # Rerank documents
        reranked_results = self.reranker.rerank(
            query=question,
            documents=documents[:self.rerank_top_k],
            top_k=self.final_top_k,
        )
        
        rerank_time = time.time() - rerank_start
        logger.info(f"  Reranked to top {len(reranked_results)} in {rerank_time:.2f}s")
        
        # Extract reranked documents
        reranked_documents = [r.document for r in reranked_results]
        
        # Log reranking impact
        if reranked_results:
            logger.info(f"  Score improvement:")
            for i, r in enumerate(reranked_results[:3], 1):
                logger.info(f"    #{i}: {r.original_score:.3f} → {r.final_score:.3f}")
        
        # ====================================================================
        # STEP 3: Answer Generation
        # ====================================================================
        logger.info(f"\n[3/4] Generating answer from top {len(reranked_documents)} documents...")
        
        gen_start = time.time()
        
        # Assemble context from reranked documents
        context, sources = self.rag_handler._assemble_context(reranked_documents)
        
        # Generate answer
        initial_answer = self.rag_handler._generate_answer(
            question, context, chat_history or []
        )
        
        gen_time = time.time() - gen_start
        logger.info(f"  Generated answer in {gen_time:.2f}s")
        logger.info(f"  Answer length: {len(initial_answer)} chars")
        
        # ====================================================================
        # STEP 4: Adversarial Critique
        # ====================================================================
        if self.skip_critique:
            logger.info("\n[4/4] Skipping critique (disabled)")
            critique = CritiqueResult(
                confidence_score=1.0,
                issues_found=[],
                unsupported_claims=[],
                citation_errors=[],
                contradictions=[],
                recommendation="accept",
            )
            final_answer = initial_answer
        else:
            logger.info("\n[4/4] Performing adversarial critique...")
            
            critique_start = time.time()
            
            critique = self._critique_answer(
                question=question,
                answer=initial_answer,
                sources=sources,
                context=context,
            )
            
            critique_time = time.time() - critique_start
            logger.info(f"  Critique complete in {critique_time:.2f}s")
            logger.info(f"  Confidence: {critique.confidence_score:.2f}")
            logger.info(f"  Recommendation: {critique.recommendation}")
            
            # Decide on final answer
            if critique.recommendation == "accept" or critique.confidence_score >= 0.7:
                final_answer = initial_answer
            elif critique.revised_answer:
                final_answer = critique.revised_answer
                logger.info("  Using revised answer from critique")
            else:
                final_answer = initial_answer
                logger.info("  Keeping original answer despite low confidence")
        
        # ====================================================================
        # Prepare Result
        # ====================================================================
        total_time = time.time() - start_time
        
        result = AdversarialRAGResult(
            query=question,
            initial_answer=initial_answer,
            final_answer=final_answer,
            sources=sources,
            retrieval_stats={
                "num_documents_retrieved": len(documents),
                "num_documents_reranked": len(reranked_results),
                "num_documents_used": len(reranked_documents),
                "retrieval_time_seconds": initial_retrieval_time,
                "reranking_time_seconds": rerank_time,
            },
            reranking_stats={
                "model": self.reranker.model_name,
                "cross_encoder_weight": self.reranker.cross_encoder_weight,
                "temporal_weight": self.reranker.temporal_weight,
                "entity_weight": self.reranker.entity_weight,
            },
            critique=critique,
            conversation_id=conversation_id or self._generate_conversation_id(),
            total_time_seconds=total_time,
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✓ Pipeline complete in {total_time:.2f}s")
        logger.info(f"{'='*70}\n")
        
        return result
    
    def _critique_answer(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        context: str,
    ) -> CritiqueResult:
        """
        Perform adversarial critique of the generated answer.
        
        Uses a second LLM to:
        - Verify citation accuracy
        - Flag unsupported claims
        - Identify contradictions
        - Score confidence
        """
        # Build critique prompt
        critique_prompt = self._build_critique_prompt(
            question, answer, sources, context
        )
        
        # Call LLM for critique
        try:
            critique_response = self._call_ollama_for_critique(critique_prompt)
            
            # Parse critique response
            critique = self._parse_critique_response(critique_response)
            
            return critique
            
        except Exception as e:
            logger.error(f"Critique failed: {e}", exc_info=True)
            # Return default critique
            return CritiqueResult(
                confidence_score=0.8,
                issues_found=[f"Critique failed: {str(e)}"],
                unsupported_claims=[],
                citation_errors=[],
                contradictions=[],
                recommendation="accept",
            )
    
    def _build_critique_prompt(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        context: str,
    ) -> str:
        """Build the critique prompt."""
        sources_list = "\n".join([
            f"- Document {i+1}: {s.get('title', 'N/A')} (ID: {s.get('document_id', 'N/A')[:16]}...)"
            for i, s in enumerate(sources)
        ])
        
        prompt = f"""You are a critical evaluator reviewing a historical research answer for accuracy and proper citation.

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

SOURCE DOCUMENTS USED:
{sources_list}

FULL CONTEXT PROVIDED TO ANSWERER:
{context[:2000]}...

YOUR TASK:
Critically evaluate the answer for:
1. **Citation Accuracy**: Are claims properly supported by the provided sources?
2. **Unsupported Claims**: Are there statements not backed by the context?
3. **Contradictions**: Do any sources contradict each other or the answer?
4. **Completeness**: Does the answer fully address the question?

Provide your critique in this EXACT format:

CONFIDENCE_SCORE: <0.0 to 1.0>
RECOMMENDATION: <accept|revise|reject>

ISSUES_FOUND:
- <issue 1>
- <issue 2>
...

UNSUPPORTED_CLAIMS:
- <claim 1>
- <claim 2>
...

CITATION_ERRORS:
- <error 1>
- <error 2>
...

CONTRADICTIONS:
- <contradiction 1>
- <contradiction 2>
...

SUGGESTED_REVISION:
<revised answer if needed, or "NONE" if answer is acceptable>

Be thorough but fair. Minor issues don't require rejection."""
        
        return prompt
    
    def _call_ollama_for_critique(self, prompt: str) -> str:
        """Call Ollama for critique generation."""
        import requests
        
        url = f"{self.rag_handler.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.critique_model,
            "prompt": prompt,
            "temperature": self.critique_temperature,
            "stream": False,
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    def _parse_critique_response(self, response: str) -> CritiqueResult:
        """Parse structured critique response."""
        # Extract confidence score
        confidence_match = re.search(r'CONFIDENCE_SCORE:\s*([\d.]+)', response)
        confidence_score = float(confidence_match.group(1)) if confidence_match else 0.8
        
        # Extract recommendation
        rec_match = re.search(r'RECOMMENDATION:\s*(\w+)', response)
        recommendation = rec_match.group(1).lower() if rec_match else "accept"
        
        # Extract lists
        def extract_list(section_name: str) -> List[str]:
            pattern = f'{section_name}:(.*?)(?=\n\n[A-Z_]+:|$)'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                items = re.findall(r'-\s*(.+)', content)
                return [item.strip() for item in items if item.strip()]
            return []
        
        issues_found = extract_list('ISSUES_FOUND')
        unsupported_claims = extract_list('UNSUPPORTED_CLAIMS')
        citation_errors = extract_list('CITATION_ERRORS')
        contradictions = extract_list('CONTRADICTIONS')
        
        # Extract suggested revision
        revision_match = re.search(r'SUGGESTED_REVISION:\s*(.+)', response, re.DOTALL)
        if revision_match:
            revision = revision_match.group(1).strip()
            revised_answer = None if revision.upper() == "NONE" else revision
        else:
            revised_answer = None
        
        return CritiqueResult(
            confidence_score=confidence_score,
            issues_found=issues_found,
            unsupported_claims=unsupported_claims,
            citation_errors=citation_errors,
            contradictions=contradictions,
            recommendation=recommendation,
            revised_answer=revised_answer,
        )
    
    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID."""
        import uuid
        return str(uuid.uuid4())
    
    def close(self):
        """Close connections."""
        if hasattr(self, 'rag_handler'):
            self.rag_handler.close()


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Command-line interface for adversarial RAG."""
    import argparse
    import re
    
    parser = argparse.ArgumentParser(
        description="Adversarial RAG with reranking and critique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python adversarial_rag.py "What caused train accidents?"
  
  # With reranking
  python adversarial_rag.py "safety violations" --rerank-top-k 10 --final-top-k 5
  
  # Skip critique for speed
  python adversarial_rag.py "quick question" --skip-critique
  
  # Test reranking
  python adversarial_rag.py --test-rerank "sample query"
        """
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask"
    )
    parser.add_argument(
        "--initial-top-k",
        type=int,
        help="Number of documents to retrieve initially (default: 20)"
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=DEFAULT_RERANK_TOP_K,
        help=f"Number of documents to rerank (default: {DEFAULT_RERANK_TOP_K})"
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=DEFAULT_FINAL_TOP_K,
        help=f"Number of documents to use for answer (default: {DEFAULT_FINAL_TOP_K})"
    )
    parser.add_argument(
        "--skip-critique",
        action="store_true",
        help="Skip adversarial critique (faster)"
    )
    parser.add_argument(
        "--test-rerank",
        action="store_true",
        help="Test reranking only (no answer generation)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    
    args = parser.parse_args()
    
    if not args.question:
        parser.error("Question is required")
    
    try:
        # Initialize handler
        handler = AdversarialRAGHandler(
            initial_top_k=args.initial_top_k,
            rerank_top_k=args.rerank_top_k,
            final_top_k=args.final_top_k,
            skip_critique=args.skip_critique,
        )
        
        if args.test_rerank:
            # Test reranking only
            logger.info("Testing reranking only...")
            documents = handler.rag_handler.hybrid_retriever.get_relevant_documents(args.question)
            reranked = handler.reranker.rerank(args.question, documents[:10], top_k=5)
            print_reranking_results(reranked)
            return 0
        
        # Full pipeline
        result = handler.process_query(args.question)
        
        # Output result
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("\n" + "="*70)
            print("ADVERSARIAL RAG RESULT")
            print("="*70)
            print(f"\nQuestion: {result.query}")
            print(f"\nFinal Answer:\n{result.final_answer}")
            
            if result.initial_answer != result.final_answer:
                print(f"\n[Note: Answer was revised by critique]")
            
            print(f"\nCritique:")
            print(f"  Confidence: {result.critique.confidence_score:.2f}")
            print(f"  Recommendation: {result.critique.recommendation}")
            if result.critique.issues_found:
                print(f"  Issues: {len(result.critique.issues_found)}")
                for issue in result.critique.issues_found[:3]:
                    print(f"    - {issue}")
            
            print(f"\nSources: {len(result.sources)} documents")
            for i, source in enumerate(result.sources[:3], 1):
                print(f"  {i}. Doc {source['document_id'][:8]}... "
                      f"(score: {source['score']:.3f})")
            
            print(f"\nPerformance:")
            print(f"  Total time: {result.total_time_seconds:.2f}s")
            print(f"  Retrieval: {result.retrieval_stats['retrieval_time_seconds']:.2f}s")
            print(f"  Reranking: {result.retrieval_stats['reranking_time_seconds']:.2f}s")
            print("="*70 + "\n")
        
        handler.close()
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())