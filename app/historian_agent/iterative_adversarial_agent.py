#!/usr/bin/env python3
"""
Tiered Historian Agent with Adversarial Verification

LOGIC OVERVIEW:
===============

This agent implements a two-tier progressive investigation strategy that
adapts based on verification quality rather than arbitrary confidence heuristics.

TIER 1 - Quick Draft:
---------------------
1. Execute standard RAG query with hybrid retrieval + reranking
2. Generate initial answer from top chunks
3. Run adversarial verification on the answer
4. If verification score >= 90/100 → Return answer (high confidence)
5. If verification score < 90/100 → Escalate to Tier 2

TIER 2 - Deep Investigation:
-----------------------------
1. Generate 3 alternative search queries using LLM
2. Execute retrieval for each alternative query
3. Collect and deduplicate document IDs (up to PARENT_RETRIEVAL_CAP)
4. Fetch full document text for all unique documents
5. Synthesize comprehensive answer from expanded context
6. Run adversarial verification on expanded answer
7. Return final answer with verification report

VERIFICATION SYSTEM:
--------------------
- Uses AdversarialRAGHandler.verify() method
- Automatically calculates adaptive timeout based on token count
- Formula: (tokens / 40) * 1.2, bounded [30s, 300s]
- Returns structured verdict with:
  * citation_score: 0-100 (percentage of claims supported)
  * is_accurate: boolean
  * reasoning: detailed explanation
  * fallback_used: whether timeout occurred

KEY DESIGN DECISIONS:
---------------------
- Escalation threshold: 90/100 (CONFIDENCE_THRESHOLD * 100)
- Document cap: 8 documents max (PARENT_RETRIEVAL_CAP)
- Verifier model: qwen2.5:32b (from .env: VERIFIER_MODEL)
- Generator model: configurable (from .env: LLM_MODEL)
- Multi-query count: 3 alternative perspectives

BACKWARD COMPATIBILITY:
-----------------------
- verify() method takes sources as List[Dict] with metadata
- Must reconstruct source list from handler's sources dict
- Metrics dict from RAG handler is passed to verifier for context
- Full document text is fetched via handler.get_full_document_text()

ERROR HANDLING:
---------------
- Multi-query generation: Falls back to generic queries if LLM fails
- Document retrieval: Continues with available documents if some fail
- Verification: Uses fallback score (75/100) if verifier times out
- All failures are logged via debug_event() when DEBUG=1
"""

import sys
import os
import time
import re
import json
import requests
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

from .rag_query_handler import RAGQueryHandler, count_tokens
from .adversarial_rag import AdversarialRAGHandler

load_dotenv()

# ====================================================================
# Configuration - Read from .env
# ====================================================================
DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"
PARENT_RETRIEVAL_CAP = int(os.environ.get("PARENT_RETRIEVAL_CAP", 8))
CONFIDENCE_THRESHOLD = 0.9  # 90/100 verification score threshold

# LLM Configuration from .env
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", "qwen2.5:32b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")


def debug_event(category: str, msg: str, icon: str = "⚙️", level: str = "INFO"):
    """
    Print debug events to stderr if DEBUG mode is enabled.
    
    Args:
        category: Event category (e.g., "Tier 1", "Verification")
        msg: Message to display
        icon: Emoji icon for visual clarity
        level: Log level (INFO, WARN, ERROR)
    """
    if DEBUG:
        timestamp = time.strftime("%H:%M:%S")
        sys.stderr.write(f"{icon} [{timestamp}] [{category.upper()}] {msg}\n")


class TieredHistorianAgent:
    """
    Multi-tier investigation agent with adversarial verification.
    
    Escalates from quick answers to comprehensive multi-query searches
    based on verification confidence scores from adversarial RAG.
    
    Attributes:
        handler: RAGQueryHandler for standard retrieval and generation
        adversarial_handler: AdversarialRAGHandler for verification
        verifier_model: Name of the verifier model (from .env)
        llm_model: Name of the generator model (from .env)
    """
    
    def __init__(self):
        """
        Initialize RAG handler and adversarial verifier.
        
        All configuration is read from .env file via environment variables:
        - LLM_MODEL: Model for answer generation
        - VERIFIER_MODEL: Model for adversarial verification
        - OLLAMA_BASE_URL: Ollama API endpoint
        - MONGO_URI: MongoDB connection string
        """
        # Initialize handlers
        self.handler = RAGQueryHandler()
        
        # Initialize adversarial handler
        # It reads all config from .env via environment variables
        self.adversarial_handler = AdversarialRAGHandler()
        
        # Store model names for logging
        self.verifier_model = VERIFIER_MODEL
        self.llm_model = LLM_MODEL
        
        debug_event(
            "Init",
            f"Initialized with Generator: {self.llm_model}, Verifier: {self.verifier_model}",
            icon="🤖"
        )
    
    def generate_multi_queries(self, original_question: str) -> List[str]:
        """
        Generate 3 alternative search queries for expanded investigation.
        
        Uses the LLM to generate different search angles that might uncover
        information missed in the initial retrieval.
        
        Args:
            original_question: The user's original query
            
        Returns:
            List of 3 alternative search queries (or fallback queries if generation fails)
            
        Example:
            >>> agent.generate_multi_queries("Were brakemen paid better than firemen?")
            [
                "brakemen firemen wage comparison records",
                "railroad occupation salary differences",
                "B&O employment compensation by position"
            ]
        """
        prompt = f"""TASK: The user asked: "{original_question}"

Generate 3 NEW, DIFFERENT search queries to find missing details or alternative perspectives.

Requirements:
- Each query should explore a different angle or time period
- Focus on what might have been missed in the initial search
- Be specific and searchable
- Avoid simply rephrasing the original question

Output ONLY a JSON list of strings: ["query1", "query2", "query3"]
"""
        
        try:
            debug_event(
                "Multi-Query Gen",
                "Generating alternative search queries...",
                icon="🔄"
            )
            
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.7, "num_predict": 200}
                },
                timeout=30
            )
            
            resp.raise_for_status()
            response_text = resp.json().get("response", "[]")
            
            # Clean up response (remove markdown if present)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1]
            if response_text.endswith("```"):
                response_text = response_text.rsplit("```", 1)[0]
            
            queries = json.loads(response_text.strip())
            
            if not isinstance(queries, list) or len(queries) == 0:
                raise ValueError("Invalid query format returned")
            
            debug_event(
                "Multi-Query Gen",
                f"Generated {len(queries)} queries",
                icon="✅"
            )
            
            return queries[:3]  # Ensure we only return 3 queries
            
        except Exception as e:
            debug_event(
                "Multi-Query Gen",
                f"Failed to generate queries: {str(e)}. Using fallback queries.",
                icon="⚠️",
                level="WARN"
            )
            
            # Fallback queries if generation fails
            return [
                f"{original_question} detailed information",
                f"{original_question} historical context",
                f"{original_question} related documents"
            ]
    
    def reconstruct_sources_list(self, sources_dict: Dict[str, str]) -> List[Dict]:
        """
        Convert sources dict to list format expected by verify().
        
        The RAG handler returns sources as {label: doc_id}, but the
        adversarial verifier expects a list of dicts with metadata.
        
        Args:
            sources_dict: Dict mapping source labels to document IDs
                Example: {"Source 1": "507f1f77bcf86cd799439011", ...}
            
        Returns:
            List of source dicts with metadata
                Example: [{"metadata": {"document_id": "507f1f77..."}}, ...]
        """
        return [
            {
                "metadata": {
                    "document_id": doc_id
                }
            }
            for doc_id in sources_dict.values()
        ]
    
    def verify_answer(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        metrics: Dict
    ) -> Dict:
        """
        Verify answer using adversarial RAG with adaptive timeout.
        
        This method wraps the AdversarialRAGHandler.verify() call and
        handles the verification workflow. The verify() method internally:
        1. Expands context by fetching full documents
        2. Calculates adaptive timeout based on token count
        3. Calls verifier model (qwen2.5:32b) to check claims
        4. Returns structured verdict
        
        Args:
            question: Original question
            answer: Generated answer to verify
            sources: List of source dicts with document_id in metadata
            metrics: Metrics dict from RAG generation (for context)
            
        Returns:
            Verification result dict with:
                - citation_score: 0-100 (percentage of claims supported)
                - is_accurate: boolean
                - reasoning: detailed explanation from verifier
                - fallback_used: True if verification timed out
                
        Example:
            >>> verdict = agent.verify_answer(
            ...     question="Were brakemen paid more?",
            ...     answer="No, firemen earned more...",
            ...     sources=[{"metadata": {"document_id": "abc123"}}],
            ...     metrics={"sources": {...}}
            ... )
            >>> verdict
            {
                "citation_score": 85,
                "is_accurate": True,
                "reasoning": "Claims 1-3 supported, claim 4 partially supported...",
                "fallback_used": False
            }
        """
        debug_event(
            "Verification",
            f"Running adversarial verification with {self.verifier_model}",
            icon="🛡️"
        )
        
        try:
            # Call verify() - it handles adaptive timeout internally
            verdict = self.adversarial_handler.verify(
                answer=answer,
                sources=sources,
                question=question,
                metrics=metrics
            )
            
            score = verdict.get('citation_score', 0)
            fallback = verdict.get('fallback_used', False)
            
            status = "✅" if score >= 90 else "⚠️" if score >= 70 else "❌"
            fallback_msg = " (fallback)" if fallback else ""
            
            debug_event(
                "Verification",
                f"{status} Score: {score}/100{fallback_msg}",
                icon="📊"
            )
            
            return verdict
            
        except Exception as e:
            debug_event(
                "Verification",
                f"Verification failed: {str(e)}. Using fallback score.",
                icon="❌",
                level="ERROR"
            )
            
            # Return fallback verdict if verification completely fails
            return {
                'is_accurate': False,
                'citation_score': 75,
                'reasoning': f"⚠️ Verification system error: {str(e)}. Answer generated from retrieved sources but not verified.",
                'fallback_used': True
            }
    
    def attach_verification_report(
        self,
        answer: str,
        verification_score: int,
        reasoning: str
    ) -> str:
        """
        Attach verification report to answer if score < 90.
        
        Args:
            answer: The generated answer
            verification_score: Score from 0-100
            reasoning: Verifier's detailed reasoning
            
        Returns:
            Answer with verification report appended (if score < 90)
        """
        if verification_score >= 90:
            debug_event(
                "Report",
                "Score >= 90, no verification report needed",
                icon="✅"
            )
            return answer
        
        debug_event(
            "Report",
            f"Attaching verification report (score: {verification_score}/100)",
            icon="📋"
        )
        
        separator = "\n\n" + "─" * 40 + "\n"
        report = f"{separator}🛡️ **VERIFICATION REPORT (Score: {verification_score}/100)**\n\n"
        report += f"**Judge's Reasoning:**\n{reasoning}\n"
        
        return answer + report
    

def run(self, question: str) -> Dict[str, Any]:
    """
    Modern API: execute tiered investigation and return a structured dict.

    Keeps backwards compatibility with the older investigate() tuple API by
    simply adapting its outputs.
    """
    answer, sources, metrics_list, duration = self.investigate(question)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "metrics_list": metrics_list,
        "duration": duration,
    }

def investigate(self, question: str) -> Tuple[str, Dict, List[Dict], float]:
    """
    Execute tiered investigation with adaptive escalation.
    
    WORKFLOW:
    ---------
    
    TIER 1 (Quick Pass):
    1. Execute hybrid retrieval (vector + keyword, RRF fusion)
    2. Rerank top 40 → 10 chunks with cross-encoder
    3. Generate answer from top chunks
    4. Verify answer with adversarial RAG
    5. IF score >= 90 → Return answer (done)
    6. IF score < 90 → Continue to Tier 2
    
    TIER 2 (Deep Investigation):
    1. Generate 3 alternative search queries
    2. Execute retrieval for each query (3 docs per query)
    3. Collect all document IDs (Tier 1 + multi-query)
    4. Deduplicate and cap at PARENT_RETRIEVAL_CAP
    5. Fetch full document text for all unique docs
    6. Generate comprehensive answer from expanded context
    7. Verify expanded answer
    8. Return final answer with verification report
    
    Args:
        question: User's research question
        
    Returns:
        Tuple of (answer, sources, all_metrics, total_time) where:
            - answer: Final answer with optional verification report
            - sources: Dict mapping source labels to document IDs
            - all_metrics: List of metrics dicts from each stage
            - total_time: Total investigation time in seconds
            
    Example:
        >>> agent = TieredHistorianAgent()
        >>> answer, sources, metrics, time = agent.investigate(
        ...     "Were brakemen paid better than firemen?"
        ... )
        >>> print(f"Answer (verified): {answer}")
        >>> print(f"Used {len(sources)} sources in {time:.1f}s")
    """
    all_metrics = []
    overall_start = time.time()
    
    debug_event(
        "Start",
        f"Beginning investigation: '{question}'",
        icon="🔍"
    )
    
    # ================================================================
    # TIER 1: Initial Quick Pass
    # ================================================================
    debug_event(
        "Tier 1",
        "Executing hybrid retrieval + reranking...",
        icon="📝"
    )
    
    t1_ans, t1_met = self.handler.process_query(
        f"Create a detailed table for: {question}",
        label="T1_DRAFT"
    )
    
    t1_duration = time.time() - overall_start
    
    debug_event(
        "Tier 1",
        f"Generated answer ({len(t1_ans)} chars) from {len(t1_met['sources'])} sources in {t1_duration:.2f}s",
        icon="✅"
    )
    
    all_metrics.append({
        "stage": "Tier 1: Initial Draft",
        "duration": t1_duration,
        **t1_met
    })
    
    # ================================================================
    # TIER 1: Adversarial Verification
    # ================================================================
    debug_event(
        "Tier 1",
        "Running adversarial verification...",
        icon="🧐"
    )
    
    verification_start = time.time()
    
    # Reconstruct sources list from metrics dict
    sources_list = self.reconstruct_sources_list(t1_met["sources"])
    
    # Verify the answer (uses verifier model with adaptive timeout)
    verdict = self.verify_answer(question, t1_ans, sources_list, t1_met)
    verification_score = verdict.get('citation_score', 0)
    verification_duration = time.time() - verification_start
    
    debug_event(
        "Decision",
        f"Tier 1 Verification: {verification_score}/100 (took {verification_duration:.2f}s)",
        icon="⚖️"
    )
    
    # Add verification to metrics
    all_metrics.append({
        "stage": "Tier 1: Verification",
        "duration": verification_duration,
        "score": verification_score,
        "reasoning_length": len(verdict.get('reasoning', '')),
        "fallback_used": verdict.get('fallback_used', False),
        "is_accurate": verdict.get('is_accurate', False)
    })
    
    # ================================================================
    # DECISION POINT: Escalate to Tier 2?
    # ================================================================
    # Convert 0-100 score to 0.0-1.0 for comparison with threshold
    confidence = verification_score / 100.0
    
    if confidence >= CONFIDENCE_THRESHOLD:
        debug_event(
            "Complete",
            f"High confidence ({verification_score}/100 >= {int(CONFIDENCE_THRESHOLD*100)}). Returning Tier 1 answer.",
            icon="✅"
        )
        
        # Attach verification report if score < 90
        final_answer = self.attach_verification_report(
            t1_ans,
            verification_score,
            verdict['reasoning']
        )
        
        total_time = time.time() - overall_start
        
        debug_event(
            "Complete",
            f"Total time: {total_time:.2f}s",
            icon="🏁"
        )
        
        return final_answer, t1_met["sources"], all_metrics, total_time
    
    # ================================================================
    # TIER 2: Multi-Query Expansion
    # ================================================================
    debug_event(
        "Tier 2",
        f"Low verification score ({verification_score}/100 < {int(CONFIDENCE_THRESHOLD*100)}). Escalating to Multi-Query expansion...",
        icon="🚀"
    )
    
    tier2_start = time.time()
    
    # Step 1: Generate alternative search queries
    new_queries = self.generate_multi_queries(question)
    
    debug_event(
        "Tier 2",
        f"Generated {len(new_queries)} alternative queries",
        icon="📋"
    )
    
    # Step 2: Collect document IDs from multi-query results
    expanded_ids = list(t1_met["sources"].values())
    
    debug_event(
        "Tier 2",
        f"Starting with {len(expanded_ids)} documents from Tier 1",
        icon="📚"
    )
    
    for idx, q in enumerate(new_queries, 1):
        debug_event(
            "Multi-Query",
            f"Query {idx}/{len(new_queries)}: {q}",
            icon="🔍"
        )
        
        try:
            # Retrieve and rerank for each query
            chunks = self.handler.hybrid_retriever.get_relevant_documents(q)
            reranked = self.handler.reranker.rerank(q, chunks, top_k=3)
            
            # Extract document IDs
            new_ids = [
                c.metadata.get("document_id")
                for c in reranked
                if c.metadata.get("document_id")
            ]
            
            expanded_ids.extend(new_ids)
            
            debug_event(
                "Multi-Query",
                f"Found {len(new_ids)} additional documents",
                icon="📄"
            )
            
        except Exception as e:
            debug_event(
                "Multi-Query",
                f"Query failed: {str(e)}",
                icon="⚠️",
                level="WARN"
            )
            continue
    
    # Step 3: De-duplicate and cap document count
    unique_ids = list(dict.fromkeys(expanded_ids))[:PARENT_RETRIEVAL_CAP]
    
    debug_event(
        "Expansion",
        f"Collected {len(expanded_ids)} total docs → {len(unique_ids)} unique (capped at {PARENT_RETRIEVAL_CAP})",
        icon="📊"
    )
    
    # Step 4: Fetch full document text
    debug_event(
        "Expansion",
        f"Fetching full text for {len(unique_ids)} documents...",
        icon="📖"
    )
    
    full_text, t2_map, io_time = self.handler.get_full_document_text(unique_ids)
    
    debug_event(
        "Expansion",
        f"Retrieved {len(full_text)} chars in {io_time:.2f}s",
        icon="✅"
    )
    
    # Step 5: Generate comprehensive answer from expanded context
    debug_event(
        "Tier 2",
        "Synthesizing comprehensive answer from expanded context...",
        icon="🔨"
    )
    
    synthesis_start = time.time()
    
    final_ans, t2_met = self.handler.process_query(
        f"Using the full document text below, answer comprehensively: {question}. "
        f"Merge duplicate events and reconcile any conflicting information. "
        f"Provide specific citations from the documents.",
        context=full_text,
        label="T2_ASSEMBLY"
    )
    
    synthesis_duration = time.time() - synthesis_start
    
    debug_event(
        "Tier 2",
        f"Generated expanded answer ({len(final_ans)} chars) in {synthesis_duration:.2f}s",
        icon="✅"
    )
    
    t2_met["retrieval_time"] = io_time
    t2_met["synthesis_time"] = synthesis_duration
    
    tier2_duration = time.time() - tier2_start
    
    all_metrics.append({
        "stage": "Tier 2: Multi-Query Assembly",
        "duration": tier2_duration,
        **t2_met
    })
    
    # ================================================================
    # TIER 2: Final Verification
    # ================================================================
    debug_event(
        "Tier 2",
        "Running final adversarial verification...",
        icon="🛡️"
    )
    
    t2_verification_start = time.time()
    
    # Reconstruct sources for verification (Tier 1 + Tier 2)
    combined_sources = {**t1_met["sources"], **t2_map}
    t2_sources = self.reconstruct_sources_list(combined_sources)
    
    # Verify Tier 2 answer
    t2_verdict = self.verify_answer(question, final_ans, t2_sources, t2_met)
    t2_score = t2_verdict.get('citation_score', 0)
    t2_verification_duration = time.time() - t2_verification_start
    
    debug_event(
        "Final Score",
        f"Tier 2 Verification: {t2_score}/100 (took {t2_verification_duration:.2f}s)",
        icon="🎯"
    )
    
    all_metrics.append({
        "stage": "Tier 2: Final Verification",
        "duration": t2_verification_duration,
        "score": t2_score,
        "reasoning_length": len(t2_verdict.get('reasoning', '')),
        "fallback_used": t2_verdict.get('fallback_used', False),
        "is_accurate": t2_verdict.get('is_accurate', False)
    })
    
    # Attach verification report
    final_answer = self.attach_verification_report(
        final_ans,
        t2_score,
        t2_verdict['reasoning']
    )
    
    # ================================================================
    # Return Results
    # ================================================================
    total_time = time.time() - overall_start
    
    debug_event(
        "Complete",
        f"Investigation complete. Total time: {total_time:.2f}s, Sources: {len(combined_sources)}",
        icon="🏁"
    )
    
    return final_answer, combined_sources, all_metrics, total_time


# ====================================================================
# CLI Interface (for testing)
# ====================================================================


def build_agent_from_env() -> "TieredHistorianAgent":
    """
    Factory used by routes.py for lazy, single-instance initialization.
    """
    return TieredHistorianAgent()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tiered_investigator.py 'your question here'")
        print("\nExample:")
        print("  DEBUG_MODE=1 python tiered_investigator.py 'Were brakemen paid better than firemen?'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    print(f"\n{'='*70}")
    print("TIERED HISTORIAN AGENT")
    print(f"{'='*70}")
    print(f"Question: {question}")
    print(f"{'='*70}\n")

    # IMPORTANT: always build via the factory
    agent = build_agent_from_env()

    try:
        answer, sources, metrics, total_time = agent.investigate(question)

        print(answer)
        print(f"\n{'='*70}")
        print("Investigation Complete")
        print(f"{'='*70}")
        print(f"Sources Used: {len(sources)}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"{'='*70}\n")

        if DEBUG:
            print("\nDETAILED METRICS:")
            print("=" * 70)
            for m in metrics:
                stage = m.get("tier") or m.get("stage") or "unknown"
                print(f"\n{stage}:")
                for k, v in m.items():
                    if k in ("tier", "stage"):
                        continue
                    if isinstance(v, float):
                        print(f"  {k}: {v:.2f}")
                    else:
                        print(f"  {k}: {v}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        if DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)
