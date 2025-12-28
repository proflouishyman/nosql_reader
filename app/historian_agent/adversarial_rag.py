# app/historian_agent/adversarial_rag.py
# UPDATED: 2025-12-28 - Integrated with llm_abstraction layer

"""
Adversarial RAG Handler - Verification layer on top of basic RAG.

CHANGES FROM ORIGINAL:
- Removed VERIFIER_MODEL, OLLAMA_BASE_URL env reads
- Uses LLMClient from llm_abstraction for verification calls
- Preserves EXACT verification logic (prompts, retry, fallback)
- Uses RAGQueryHandler (which now uses llm_abstraction)
- Backward compatible with existing routes
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .rag_query_handler import RAGQueryHandler, count_tokens, debug_print
from llm_abstraction import LLMClient, LLMResponse
from config import APP_CONFIG


# Configuration
LOG_DIR = Path("/app/logs/adversarial")
_log_file = None


# ============================================================================
# Logging
# ============================================================================

def _init_log_file():
    """Initialize debug log file."""
    global _log_file
    
    if not APP_CONFIG.debug_mode or _log_file is not None:
        return
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"adversarial_{timestamp}.log"
    
    try:
        _log_file = open(log_path, 'w', encoding='utf-8')
        _log_file.write(f"=== Adversarial RAG Debug Log ===\n")
        _log_file.write(f"Started: {datetime.now().isoformat()}\n")
        _log_file.write(f"Verifier Profile: verifier\n")
        _log_file.write("="*60 + "\n\n")
        _log_file.flush()
    except Exception as e:
        sys.stderr.write(f"[ADVERSARIAL] Failed to create log: {e}\n")
        sys.stderr.flush()
        _log_file = None


def debug_step(step_name: str, detail: str = "", icon: str = "", level: str = "INFO"):
    """Print debug information and log to file."""
    if not APP_CONFIG.debug_mode:
        return
    
    if _log_file is None:
        _init_log_file()
    
    timestamp = time.strftime("%H:%M:%S")
    stderr_msg = f"{icon} [{timestamp}] [ADVERSARIAL] {step_name.upper()}\n"
    if detail:
        stderr_msg += f"   {detail}\n"
    sys.stderr.write(stderr_msg)
    sys.stderr.flush()
    
    if _log_file is not None:
        try:
            log_msg = f"[{timestamp}] [{level}] {step_name.upper()}\n"
            if detail:
                log_msg += f"  {detail}\n"
            _log_file.write(log_msg)
            _log_file.flush()
        except Exception:
            pass


# ============================================================================
# Adversarial RAG Handler
# ============================================================================

class AdversarialRAGHandler:
    """
    RAG with adversarial verification.
    
    Pipeline (YOUR EXISTING LOGIC - unchanged):
        1. Generate answer (via RAGQueryHandler)
        2. Extract exact context used
        3. Verify claims against sources (separate LLM call)
        4. Attach verification report if score < 90
    """
    
    def __init__(self):
        """Initialize adversarial handler."""
        debug_step("Init", "Initializing Adversarial Handler...", icon="🔰")
        
        # Initialize RAG handler (uses llm_abstraction internally)
        self.rag_handler = RAGQueryHandler()
        
        # LLM client for verification
        self.llm = LLMClient()
        
        debug_step("Init", "Verifier: profile='verifier'", icon="✅")
    
    def verify_citations(
        self,
        question: str,
        answer: str,
        sources_text: str
    ) -> Dict[str, Any]:
        """
        Verify answer claims against source text.
        
        YOUR EXISTING VERIFICATION LOGIC - preserved!
        Only change: Uses llm_abstraction instead of raw requests.
        
        Args:
            question: Original question
            answer: Generated answer to verify
            sources_text: Complete source text
        
        Returns:
            Verification verdict dict
        """
        debug_step("Verification", "Starting citation verification", icon="⚖️")
        
        # YOUR EXISTING PROMPT - unchanged!
        system_prompt = """You are a fact-checking expert reviewing claims made in historical document research.

Your task: Verify that every factual claim in the ANSWER is directly supported by the SOURCE TEXT.

Rules:
1. Check ONLY what is explicitly stated in the sources
2. Do NOT use background knowledge or inference
3. Do NOT accept paraphrased claims without direct evidence
4. If a date/name/number appears in the answer, it MUST appear in sources

Output Format (JSON):
{
    "is_accurate": true/false,
    "citation_score": 0-100,
    "reasoning": "Detailed explanation"
}"""
        
        user_prompt = f"""QUESTION:
{question}

ANSWER TO VERIFY:
{answer}

SOURCE TEXT:
{sources_text}

TASK: Verify each factual claim. Report citation_score as percentage with direct support."""
        
        # Calculate adaptive timeout (YOUR EXISTING LOGIC)
        token_count = count_tokens(system_prompt + user_prompt)
        timeout = max(30, min(300, int((token_count / 40) * 1.2)))
        
        debug_step(
            "Verification",
            f"Tokens: {token_count}, Timeout: {timeout}s",
            icon="🔋"
        )
        
        # UPDATED: Use llm_abstraction with retry
        try:
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                profile="verifier",  # Uses verifier config
                temperature=0.0,
                max_tokens=500,
                retry=True,
                max_retries=3,
                timeout=timeout,
                response_format="json"  # Provider-specific hint
            )
            
            if not response.success:
                raise Exception(str(response.error))
            
            # Parse JSON response
            verdict = json.loads(response.content)
            
            # Validate and clamp score
            verdict["citation_score"] = max(0, min(100, int(verdict.get("citation_score", 0))))
            verdict["fallback_used"] = False
            
            debug_step(
                "Verification",
                f"Score: {verdict['citation_score']}/100",
                icon="📊"
            )
            
            return verdict
        
        except Exception as e:
            debug_step(
                "Verification",
                f"Verification failed: {str(e)}",
                icon="⚠️",
                level="ERROR"
            )
        
        # YOUR EXISTING FALLBACK - unchanged!
        debug_step("Fallback", "Using fallback verdict", icon="⚠️", level="WARN")
        
        return {
            "is_accurate": True,
            "citation_score": 75,
            "reasoning": (
                "Verification system unavailable. "
                "Answer generated from retrieved sources but not independently verified. "
                "Manual review recommended."
            ),
            "fallback_used": True
        }
    
    def process_query(self, question: str) -> tuple:
        """
        Process query through adversarial pipeline.
        
        YOUR EXISTING PIPELINE - unchanged!
        
        Returns:
            (answer, latency, sources)
        """
        start = time.time()
        
        # 1. Generate Answer (uses llm_abstraction internally)
        debug_step("Generation", "Starting RAG query...", icon="🚀")
        ans, metrics = self.rag_handler.process_query(question, label="ADVERSARIAL_GEN")
        
        # 2. Extract Context
        verify_text = metrics.get('context', "")
        
        if not verify_text:
            debug_step(
                "Context",
                "Context not in metrics, fetching full documents",
                icon="⚠️",
                level="WARN"
            )
            source_ids = list(metrics.get('sources', {}).values())
            
            if source_ids:
                verify_text, _, _ = self.rag_handler.get_full_document_text(source_ids)
            else:
                verify_text = "No sources found."
        
        # 3. Verify Answer
        debug_step("Verification", "Running adversarial verification", icon="⚖️")
        verdict = self.verify_citations(question, ans, verify_text)
        
        score = verdict.get('citation_score', 0)
        
        # 4. Attach Report if Needed (YOUR EXISTING LOGIC)
        if score < 90:
            report = f"\n\n---\n**Verification Report:**\n"
            report += f"- Citation Score: {score}/100\n"
            report += f"- Reasoning: {verdict.get('reasoning', 'N/A')}\n"
            
            if verdict.get('fallback_used'):
                report += "- Note: Automated verification unavailable\n"
            
            ans += report
        
        latency = time.time() - start
        sources = metrics.get('sources', {})
        
        debug_step("Complete", f"Total: {latency:.1f}s, Score: {score}/100", icon="🏁")
        
        return ans, latency, sources
    
    def close(self):
        """Clean up resources."""
        global _log_file
        
        if _log_file is not None:
            try:
                _log_file.write(f"\n{'='*60}\n")
                _log_file.write(f"Session ended: {datetime.now().isoformat()}\n")
                _log_file.close()
            except Exception:
                pass
            finally:
                _log_file = None
        
        self.rag_handler.close()


# CLI Test Interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python adversarial_rag.py 'question'")
    
    handler = AdversarialRAGHandler()
    try:
        query = sys.argv[1]
        ans, lat, src = handler.process_query(query)
        
        print("\n" + "═" * 60)
        print(f"  ADVERSARIAL RESPONSE ({lat:.2f}s)")
        print("─" * 60)
        print(ans)
        print("═" * 60)
        print(f"\nSources: {len(src)} documents")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        handler.close()
