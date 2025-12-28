# app/historian_agent/iterative_adversarial_agent.py
# UPDATED: 2025-12-28 - Integrated with llm_abstraction layer

"""
Tiered Historian Agent with Adversarial Verification

CHANGES FROM ORIGINAL:
- Removed LLM_MODEL, VERIFIER_MODEL, OLLAMA_BASE_URL env reads
- Uses LLMClient from llm_abstraction for multi-query generation
- Preserves EXACT tier escalation logic
- Uses RAGQueryHandler and AdversarialRAGHandler (which use llm_abstraction)
- Backward compatible with existing routes
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

from .rag_query_handler import RAGQueryHandler, count_tokens, debug_print
from .adversarial_rag import AdversarialRAGHandler
from llm_abstraction import LLMClient
from config import APP_CONFIG

load_dotenv()

# Configuration
PARENT_RETRIEVAL_CAP = int(os.environ.get("PARENT_RETRIEVAL_CAP", 8))
CONFIDENCE_THRESHOLD = 0.9  # 90/100 verification score threshold


def debug_event(category: str, msg: str, icon: str = "⚙️", level: str = "INFO"):
    """Print debug events if debug mode enabled."""
    if APP_CONFIG.debug_mode:
        timestamp = time.strftime("%H:%M:%S")
        sys.stderr.write(f"{icon} [{timestamp}] [{category.upper()}] {msg}\n")


class TieredHistorianAgent:
    """
    Multi-tier investigation agent with adversarial verification.
    
    Pipeline (YOUR EXISTING LOGIC - unchanged):
        Tier 1: Quick answer + verification
        If score >= 90 → Return
        If score < 90 → Tier 2: Multi-query expansion + comprehensive answer
    """
    
    def __init__(self):
        """Initialize tiered agent."""
        # Initialize handlers (use llm_abstraction internally)
        self.handler = RAGQueryHandler()
        self.adversarial_handler = AdversarialRAGHandler()
        
        # LLM client for multi-query generation
        self.llm = LLMClient()
        
        debug_event(
            "Init",
            "Initialized tiered agent with adversarial verification",
            icon="🤖"
        )
    
    def generate_multi_queries(self, original_question: str) -> List[str]:
        """
        Generate 3 alternative search queries.
        
        YOUR EXISTING LOGIC - preserved!
        Only change: Uses llm_abstraction instead of raw requests.
        
        Args:
            original_question: The user's original query
        
        Returns:
            List of 3 alternative search queries
        """
        debug_event("Multi-Query", "Generating alternative queries...", icon="🔄")
        
        # YOUR EXISTING PROMPT - unchanged!
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
            # UPDATED: Use llm_abstraction
            response = self.llm.generate_simple(
                prompt=prompt,
                profile="fast",  # Use fast model for query generation
                temperature=0.7,
                max_tokens=200,
                response_format="json"
            )
            
            if not response.success:
                raise Exception(str(response.error))
            
            # Parse JSON response
            content = response.content.strip()
            
            # Clean up markdown if present
            if content.startswith("```json"):
                content = content.split("```json")[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            
            queries = json.loads(content.strip())
            
            if not isinstance(queries, list) or len(queries) == 0:
                raise ValueError("Invalid query format")
            
            debug_event(
                "Multi-Query",
                f"Generated {len(queries)} queries",
                icon="✅"
            )
            
            return queries[:3]  # Take first 3
        
        except Exception as e:
            debug_event(
                "Multi-Query",
                f"Generation failed: {str(e)}, using fallback",
                icon="⚠️",
                level="WARN"
            )
            
            # YOUR EXISTING FALLBACK - unchanged!
            return [
                f"{original_question} detailed information",
                f"{original_question} historical context",
                f"{original_question} related documents"
            ]
    
    def tier1_quick_answer(
        self,
        question: str
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Tier 1: Quick answer using standard RAG.
        
        YOUR EXISTING LOGIC - unchanged!
        """
        debug_event("Tier 1", "Starting quick answer...", icon="🚀")
        
        t_start = time.time()
        answer, metrics = self.handler.process_query(question, label="TIER1")
        latency = time.time() - t_start
        
        debug_event("Tier 1", f"Generated in {latency:.1f}s", icon="✅")
        
        return answer, metrics, latency
    
    def tier2_deep_investigation(
        self,
        question: str
    ) -> Tuple[str, Dict[str, str], List[Dict], float]:
        """
        Tier 2: Deep investigation with multi-query expansion.
        
        YOUR EXISTING PIPELINE - unchanged!
        
        Returns:
            (answer, sources, tier_metrics_list, latency)
        """
        debug_event("Tier 2", "Starting deep investigation...", icon="🔬")
        
        t_start = time.time()
        tier_metrics = []
        
        # 1. Generate alternative queries (uses llm_abstraction)
        alt_queries = self.generate_multi_queries(question)
        
        debug_event("Tier 2", f"Alternative queries: {alt_queries}", icon="📝")
        
        # 2. Execute retrieval for each query (YOUR EXISTING LOGIC)
        all_doc_ids = set()
        
        for idx, query in enumerate(alt_queries, 1):
            debug_event("Tier 2", f"Query {idx}/3: {query[:60]}...", icon="🔍")
            
            try:
                chunks = self.handler.hybrid_retriever.get_relevant_documents(query)
                
                for chunk in chunks:
                    doc_id = chunk.metadata.get("document_id")
                    if doc_id:
                        all_doc_ids.add(doc_id)
                
                tier_metrics.append({
                    "tier": f"2.{idx}",
                    "query": query,
                    "chunks_found": len(chunks),
                    "stage": "retrieval"
                })
            
            except Exception as e:
                debug_event(
                    "Tier 2",
                    f"Query {idx} failed: {str(e)}",
                    icon="⚠️",
                    level="ERROR"
                )
        
        # 3. Cap to parent_retrieval_cap
        unique_doc_ids = list(all_doc_ids)[:PARENT_RETRIEVAL_CAP]
        
        debug_event(
            "Tier 2",
            f"Collected {len(unique_doc_ids)} documents (cap: {PARENT_RETRIEVAL_CAP})",
            icon="📚"
        )
        
        # 4. Fetch full document text
        if unique_doc_ids:
            full_text, sources, fetch_time = self.handler.get_full_document_text(unique_doc_ids)
            
            tier_metrics.append({
                "tier": "2.expansion",
                "doc_count": len(unique_doc_ids),
                "fetch_time": fetch_time,
                "stage": "expansion"
            })
        else:
            debug_event("Tier 2", "No documents found", icon="⚠️", level="WARN")
            full_text = "No relevant documents found."
            sources = {}
        
        # 5. Synthesize answer from expanded context
        debug_event("Tier 2", "Synthesizing answer...", icon="🧠")
        
        answer, synth_metrics = self.handler.process_query(
            f"Using the full document text below, answer comprehensively: {question}. "
            f"Merge duplicate events and reconcile any conflicting information. "
            f"Provide specific citations from the documents.",
            context=full_text,
            label="TIER2"
        )
        
        tier_metrics.append({
            "tier": "2.synthesis",
            "llm_time": synth_metrics.get("llm_time", 0),
            "tokens": synth_metrics.get("tokens", 0),
            "stage": "synthesis"
        })
        
        latency = time.time() - t_start
        
        debug_event("Tier 2", f"Investigation complete in {latency:.1f}s", icon="✅")
        
        return answer, sources, tier_metrics, latency
    
    def investigate(self, question: str) -> Tuple[str, Dict, List[Dict], float]:
        """
        Main investigation with confidence-based escalation.
        
        YOUR EXISTING FLOW - unchanged!
        
        Returns:
            (final_answer, sources, all_metrics, total_time)
        """
        total_start = time.time()
        all_metrics = []
        
        debug_event("Investigation", f"Starting: {question[:60]}...", icon="🎯")
        
        # ========================================
        # TIER 1: Quick Answer + Verification
        # ========================================
        
        tier1_answer, tier1_metrics, tier1_time = self.tier1_quick_answer(question)
        
        all_metrics.append({
            "tier": "1",
            "time": tier1_time,
            "tokens": tier1_metrics.get("tokens", 0),
            "doc_count": tier1_metrics.get("doc_count", 0),
            "stage": "tier1_generation"
        })
        
        # Verify Tier 1 answer
        debug_event("Verification", "Verifying Tier 1...", icon="⚖️")
        
        verify_start = time.time()
        tier1_verdict = self.adversarial_handler.verify_citations(
            question,
            tier1_answer,
            tier1_metrics.get("context", "")
        )
        verify_time = time.time() - verify_start
        
        score = tier1_verdict.get("citation_score", 0)
        
        all_metrics.append({
            "tier": "1.verification",
            "time": verify_time,
            "score": score,
            "is_accurate": tier1_verdict.get("is_accurate", False),
            "stage": "tier1_verification"
        })
        
        debug_event("Verification", f"Tier 1 Score: {score}/100", icon="📊")
        
        # ========================================
        # Decision: Return or Escalate?
        # ========================================
        
        threshold = int(CONFIDENCE_THRESHOLD * 100)
        
        if score >= threshold:
            # High confidence - return Tier 1
            debug_event(
                "Decision",
                f"Score {score} >= {threshold}, returning Tier 1",
                icon="✅"
            )
            
            total_time = time.time() - total_start
            sources = tier1_metrics.get("sources", {})
            
            return tier1_answer, sources, all_metrics, total_time
        
        # ========================================
        # TIER 2: Deep Investigation
        # ========================================
        
        debug_event(
            "Decision",
            f"Score {score} < {threshold}, escalating to Tier 2",
            icon="⬆️"
        )
        
        tier2_answer, tier2_sources, tier2_metrics, tier2_time = (
            self.tier2_deep_investigation(question)
        )
        
        all_metrics.extend(tier2_metrics)
        
        # Verify Tier 2 answer
        debug_event("Verification", "Verifying Tier 2...", icon="⚖️")
        
        tier2_context, _, _ = self.handler.get_full_document_text(
            list(tier2_sources.values())
        )
        
        verify_start = time.time()
        tier2_verdict = self.adversarial_handler.verify_citations(
            question,
            tier2_answer,
            tier2_context
        )
        verify_time = time.time() - verify_start
        
        final_score = tier2_verdict.get("citation_score", 0)
        
        all_metrics.append({
            "tier": "2.verification",
            "time": verify_time,
            "score": final_score,
            "is_accurate": tier2_verdict.get("is_accurate", False),
            "stage": "tier2_verification"
        })
        
        debug_event("Verification", f"Tier 2 Score: {final_score}/100", icon="📊")
        
        # Attach report if needed
        final_answer = tier2_answer
        
        if final_score < 90:
            report = f"\n\n---\n**Verification Report:**\n"
            report += f"- Citation Score: {final_score}/100\n"
            report += f"- Reasoning: {tier2_verdict.get('reasoning', 'N/A')}\n"
            
            if tier2_verdict.get('fallback_used'):
                report += "- Note: Automated verification unavailable\n"
            
            final_answer += report
        
        total_time = time.time() - total_start
        
        debug_event(
            "Complete",
            f"Total: {total_time:.1f}s, Score: {final_score}/100",
            icon="🏁"
        )
        
        return final_answer, tier2_sources, all_metrics, total_time


# Factory function for routes.py
def build_agent_from_env() -> TieredHistorianAgent:
    """Factory for backward compatibility."""
    return TieredHistorianAgent()


# CLI Test Interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python iterative_adversarial_agent.py 'your question'")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    agent = TieredHistorianAgent()
    
    try:
        answer, sources, metrics, total_time = agent.investigate(question)
        
        print(answer)
        print(f"\n{'='*70}")
        print("INVESTIGATION SUMMARY")
        print(f"{'='*70}")
        print(f"Sources: {len(sources)}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"{'='*70}\n")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
