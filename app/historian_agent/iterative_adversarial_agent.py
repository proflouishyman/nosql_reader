#!/usr/bin/env python3
"""
Tiered Historian Agent with Adversarial Verification - FIXED

CRITICAL FIX: run() method must be INSIDE the class, not at module level!
"""

import sys
import os
import time
import re
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv

# Import from same package
try:
    from .rag_query_handler import RAGQueryHandler, count_tokens
    from .adversarial_rag import AdversarialRAGHandler
except ImportError:
    # Fallback for when module is run directly
    from rag_query_handler import RAGQueryHandler, count_tokens
    from adversarial_rag import AdversarialRAGHandler

load_dotenv()

# Configuration
DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"
PARENT_RETRIEVAL_CAP = int(os.environ.get("PARENT_RETRIEVAL_CAP", 8))
CONFIDENCE_THRESHOLD = 0.9

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", "qwen2.5:32b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

# Logging Setup
LOG_DIR = Path("/app/logs/tiered_agent")
_log_file = None

def _init_log_file():
    """Initialize log file with timestamp if in DEBUG mode."""
    global _log_file
    
    if not DEBUG or _log_file is not None:
        return
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"tiered_{timestamp}.log"
    
    try:
        _log_file = open(log_path, 'w', encoding='utf-8')
        _log_file.write(f"=== Tiered Historian Agent Debug Log ===\n")
        _log_file.write(f"Started: {datetime.now().isoformat()}\n")
        _log_file.write(f"LLM Model: {LLM_MODEL}\n")
        _log_file.write(f"Verifier Model: {VERIFIER_MODEL}\n")
        _log_file.write("="*60 + "\n\n")
        _log_file.flush()
        sys.stderr.write(f"🔍 [TIERED] Logging to: {log_path}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"⚠️ [TIERED] Failed to create log file: {e}\n")
        sys.stderr.flush()
        _log_file = None


def debug_event(category: str, msg: str, icon: str = "⚙️", level: str = "INFO"):
    """Print debug events to stderr and write to log file."""
    if not DEBUG:
        return
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    output = f"{icon} [{timestamp}] [{category.upper()}] {msg}"
    
    sys.stderr.write(output + "\n")
    sys.stderr.flush()
    
    if _log_file:
        try:
            _log_file.write(f"[{timestamp}] [{level}] [{category}] {msg}\n")
            _log_file.flush()
        except:
            pass


def log_prompt(stage: str, prompt: str):
    """Log a full LLM prompt to file."""
    if not DEBUG or not _log_file:
        return
    
    try:
        _log_file.write(f"\n{'-'*70}\n")
        _log_file.write(f"{stage} PROMPT ({len(prompt)} chars, ~{len(prompt)//4} tokens)\n")
        _log_file.write(f"{'-'*70}\n")
        _log_file.write(prompt)
        _log_file.write(f"\n{'-'*70}\n\n")
        _log_file.flush()
    except:
        pass


def log_response(stage: str, response: str):
    """Log a full LLM response to file."""
    if not DEBUG or not _log_file:
        return
    
    try:
        _log_file.write(f"\n{'-'*70}\n")
        _log_file.write(f"{stage} RESPONSE ({len(response)} chars)\n")
        _log_file.write(f"{'-'*70}\n")
        _log_file.write(response[:5000])
        if len(response) > 5000:
            _log_file.write(f"\n... [truncated {len(response) - 5000} chars] ...\n")
        _log_file.write(f"\n{'-'*70}\n\n")
        _log_file.flush()
    except:
        pass


class TieredHistorianAgent:
    """Two-tier investigation agent with robust logging."""
    
    def __init__(self):
        """Initialize handlers and logging."""
        debug_event("Init", "Initializing TieredHistorianAgent...", icon="🚀")
        _init_log_file()
        
        self.handler = RAGQueryHandler()
        self.adversarial_handler = AdversarialRAGHandler()
        self.llm_model = LLM_MODEL
        self.verifier_model = VERIFIER_MODEL
        
        debug_event("Init", f"LLM: {self.llm_model}", icon="🤖")
        debug_event("Init", "Initialization complete", icon="✅")
    
    
    def generate_multi_queries(self, original_question: str, answer: str, verification_reasoning: str) -> List[str]:
        """
        Generate diagnostic search queries based on verification failures.
        
        Extracts the specific unsupported claims from verification reasoning and
        creates targeted search queries to find documents containing that evidence.
        """
        prompt = f"""You are analyzing why certain claims in an answer could not be verified against source documents.

ORIGINAL QUESTION:
{original_question}

ANSWER THAT WAS GENERATED:
{answer[:1500]}

VERIFIER'S REASONING (what couldn't be confirmed):
{verification_reasoning}

TASK: Extract the SPECIFIC unsupported claims and create 3 targeted search queries to find evidence for them.

STRATEGY:
1. Identify which specific claims the verifier said were "not supported" or "not found"
2. Extract the key terms/concepts from those unsupported claims
3. Create search queries using those exact terms

EXAMPLES:

Example 1:
Verifier says: "Claims about 'Burst knee cap' not found in sources"
Extract: burst knee cap
Query: "burst knee cap"

Example 2:
Verifier says: "The wage amount '$2.50/hour' could not be verified"
Extract: $2.50/hour, wage amount
Query: "wage 2.50 dollar hour payment"

Example 3:
Verifier says: "Claims about 'elected in 1923' and 'served until 1931' not supported"
Extract: elected 1923, served 1931
Queries: ["elected 1923", "served 1931 term"]

Example 4:
Verifier says: "Information about 'transferred to Chicago division' not in source text"
Extract: transferred Chicago division
Query: "transferred Chicago division"

YOUR TASK:
- Look at what the verifier said was NOT supported
- Pull out the key nouns, numbers, dates, names, or specific terms from those claims
- Create 3 search queries using those extracted terms
- DO NOT add generic words like "detailed" or "information"
- DO use the actual words from the unsupported claims

Return ONLY a JSON array:
["query_with_unsupported_terms_1", "query_with_unsupported_terms_2", "query_with_unsupported_terms_3"]"""
        
        log_prompt("Multi-Query Generation", prompt)
        
        try:
            debug_event("Multi-Query Gen", "Extracting unsupported claims and generating targeted queries...", icon="🔄")
            
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
            log_response("Multi-Query Generation", response_text)
            
            # Aggressive cleaning for JSON extraction
            response_text = response_text.strip()
            
            # Remove markdown code blocks
            if "```" in response_text:
                response_text = re.sub(r"```(?:json)?", "", response_text)
            
            # Find JSON array pattern - look for [...] even if surrounded by text
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            else:
                # Fallback: try to find just the array content
                if "[" in response_text and "]" in response_text:
                    start = response_text.find("[")
                    end = response_text.rfind("]") + 1
                    response_text = response_text[start:end]
            
            queries = json.loads(response_text.strip())
            
            if not isinstance(queries, list) or len(queries) == 0:
                raise ValueError("Invalid query format returned")
            
            debug_event("Multi-Query Gen", f"Generated {len(queries)} queries", icon="✅")
            return queries[:3]
            
        except Exception as e:
            debug_event("Multi-Query Gen", f"Failed: {str(e)}. Using fallback queries.", icon="⚠️", level="WARN")
            return [
                f"{original_question} detailed information",
                f"{original_question} historical context",
                f"{original_question} related documents"
            ]
    
    
    def reconstruct_sources_list(self, sources_dict: Dict[str, str]) -> List[Dict]:
        """Convert sources dict to list format expected by verify()."""
        return [
            {"metadata": {"document_id": doc_id}}
            for doc_id in sources_dict.values()
        ]
    
    
    def sources_dict_to_list(self, sources_dict: Dict[str, str]) -> List[Dict[str, str]]:
        """Convert internal sources dict to frontend-friendly list format."""
        return [
            {"id": doc_id, "label": label}
            for label, doc_id in sources_dict.items()
        ]
    
    
    def verify_answer(self, question: str, answer: str, sources: List[Dict], metrics: Dict) -> Dict:
        """Verify answer using adversarial RAG with adaptive timeout."""
        debug_event("Verification", f"Running adversarial verification with {self.verifier_model}", icon="🛡️")
        
        # Log verification inputs for debugging
        if _log_file:
            try:
                _log_file.write(f"\n{'='*60}\n")
                _log_file.write(f"VERIFICATION INPUT\n")
                _log_file.write(f"{'='*60}\n")
                _log_file.write(f"Question: {question}\n")
                _log_file.write(f"Answer Length: {len(answer)} chars\n")
                _log_file.write(f"Sources Count: {len(sources)}\n")
                _log_file.write(f"Context in metrics: {'context' in metrics}\n")
                if 'context' in metrics:
                    context_preview = metrics['context']
                    _log_file.write(f"Context Length: {len(context_preview)} chars\n")
                    _log_file.write(f"\nContext Preview (first 2000 chars):\n")
                    _log_file.write(f"{'-'*60}\n")
                    _log_file.write(f"{context_preview[:2000]}\n")
                    _log_file.write(f"{'-'*60}\n")
                    
                    # Check if context looks empty/useless
                    if len(context_preview.strip()) < 500:
                        _log_file.write(f"⚠️ WARNING: Context seems very short!\n")
                    if context_preview.count('---') > len(sources) * 2:
                        _log_file.write(f"⚠️ WARNING: Lots of document headers but little content!\n")
                else:
                    _log_file.write(f"⚠️ WARNING: No context in metrics!\n")
                _log_file.write(f"{'='*60}\n\n")
                _log_file.flush()
            except:
                pass
        
        try:
            # CRITICAL: Pass the context explicitly to verify()
            # The verify() method accepts context as a keyword argument
            context_to_verify = metrics.get('context', '')
            
            if not context_to_verify:
                debug_event("Verification", "⚠️ No context in metrics, verify() will fetch documents", icon="⚠️", level="WARN")
            else:
                debug_event("Verification", f"Passing {len(context_to_verify)} chars of context to verifier", icon="📄")
            
            # Log a sample of the context for debugging
            if _log_file and context_to_verify:
                try:
                    _log_file.write(f"Context Sample (first 500 chars):\n")
                    _log_file.write(f"{context_to_verify[:500]}\n")
                    _log_file.write(f"...\n")
                    _log_file.write(f"Context Sample (last 500 chars):\n")
                    _log_file.write(f"{context_to_verify[-500:]}\n\n")
                    _log_file.flush()
                except:
                    pass
            
            verdict = self.adversarial_handler.verify(
                answer=answer,
                sources=sources,
                question=question,
                metrics=metrics,
                context=context_to_verify  # ← Pass context explicitly!
            )
            
            score = verdict.get('citation_score', 0)
            fallback = verdict.get('fallback_used', False)
            
            status = "✅" if score >= 90 else "⚠️" if score >= 70 else "❌"
            fallback_msg = " (fallback)" if fallback else ""
            
            debug_event("Verification", f"{status} Score: {score}/100{fallback_msg}", icon="📊")
            return verdict
            
        except Exception as e:
            debug_event("Verification", f"Verification failed: {str(e)}. Using fallback score.", icon="❌", level="ERROR")
            return {
                'is_accurate': False,
                'citation_score': 75,
                'reasoning': f"⚠️ Verification system error: {str(e)}. Answer generated from retrieved sources but not verified.",
                'fallback_used': True
            }
    
    
    def attach_verification_report(self, answer: str, verification_score: int, reasoning: str) -> str:
        """Attach verification report to answer if score < 90."""
        if verification_score >= 90:
            debug_event("Report", "Score >= 90, no verification report needed", icon="✅")
            return answer
        
        debug_event("Report", f"Attaching verification report (score: {verification_score}/100)", icon="📋")
        
        separator = "\n\n" + "─" * 40 + "\n"
        report = f"{separator}🛡️ **VERIFICATION REPORT (Score: {verification_score}/100)**\n\n"
        report += f"**Judge's Reasoning:**\n{reasoning}\n"
        
        return answer + report
    
    
    # CRITICAL: This MUST be a class method (indented under class)
    def run(self, question: str) -> Dict[str, Any]:
        """
        Modern API: execute tiered investigation and return a structured dict.
        
        This is what routes.py calls!
        """
        answer, sources, metrics_list, duration = self.investigate(question)
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "metrics_list": metrics_list,
            "duration": duration,
        }
    
    
    def investigate(self, question: str) -> Tuple[str, List[Dict[str, str]], List[Dict], float]:
        """Execute tiered investigation with adaptive escalation."""
        all_metrics = []
        overall_start = time.time()
        
        debug_event("Start", f"Beginning investigation: '{question}'", icon="🔍")
        
        # ================================================================
        # TIER 1: Initial Quick Pass
        # ================================================================
        debug_event("Tier 1", "Executing hybrid retrieval + reranking...", icon="📝")
        
        # 2025-12-23 19:xx ET
        # Purpose: Extract only source-grounded claims with explicit citations, no synthesis requirements.
        tier1_prompt = f"""You will be given a QUESTION and a set of SOURCE DOCUMENTS.

Your job: produce ONLY claims that are EXPLICITLY supported by the sources.
If a claim cannot be directly supported by a specific quote/span in the sources, DO NOT include it.

Output format: a JSON array of objects, each object:
{{
  "claim": "...",
  "sources": ["DocNameOrID#location", "DocNameOrID#location"]
}}

Rules:
- Every claim MUST have at least one source pointer.
- Use only information stated in the sources, no general knowledge, no inference, no summarizing across documents.
- Claims may be partial or narrow, that is fine.
- If the sources do not support any claims that answer the QUESTION, output [] (empty array).
- Do not output any text outside the JSON.

QUESTION: {question}
"""

        
        log_prompt("Tier 1 Generation", tier1_prompt)
        
        t1_ans, t1_met = self.handler.process_query(tier1_prompt, label="T1_DRAFT")
        log_response("Tier 1 Generation", t1_ans)
        
        t1_duration = time.time() - overall_start
        
        debug_event("Tier 1", f"Generated answer ({len(t1_ans)} chars) from {len(t1_met['sources'])} sources in {t1_duration:.2f}s", icon="✅")
        
        all_metrics.append({
            "stage": "Tier 1: Initial Draft",
            "duration": t1_duration,
            **t1_met
        })
        
        # ================================================================
        # TIER 1: Adversarial Verification
        # ================================================================
        debug_event("Tier 1", "Running adversarial verification...", icon="🧐")
        
        verification_start = time.time()
        sources_list = self.reconstruct_sources_list(t1_met["sources"])
        verdict = self.verify_answer(question, t1_ans, sources_list, t1_met)
        
        verification_score = verdict.get('citation_score', 0)
        verification_duration = time.time() - verification_start
        
        debug_event("Decision", f"Tier 1 Verification: {verification_score}/100 (took {verification_duration:.2f}s)", icon="⚖️")
        
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
        confidence = verification_score / 100.0
        
        if confidence >= CONFIDENCE_THRESHOLD:
            debug_event("Complete", f"High confidence ({verification_score}/100 >= {int(CONFIDENCE_THRESHOLD*100)}). Returning Tier 1 answer.", icon="🏁")
            
            total_time = time.time() - overall_start
            sources_list_output = self.sources_dict_to_list(t1_met["sources"])
            
            final_answer = self.attach_verification_report(t1_ans, verification_score, verdict['reasoning'])
            return final_answer, sources_list_output, all_metrics, total_time
        
        # ================================================================
        # TIER 2: Deep Investigation
        # ================================================================
        debug_event("Tier 2", f"Low verification score ({verification_score}/100 < 90). Escalating to Multi-Query expansion...", icon="🚀")
        
        tier2_start = time.time()
        
        # Generate diagnostic queries based on what verification couldn't confirm
        alt_queries = self.generate_multi_queries(
            original_question=question,
            answer=t1_ans,
            verification_reasoning=verdict.get('reasoning', '')
        )
        
        debug_event("Tier 2", f"Generated {len(alt_queries)} alternative queries", icon="📋")
        
        all_doc_ids = set(t1_met["sources"].values())
        debug_event("Tier 2", f"Starting with {len(all_doc_ids)} documents from Tier 1", icon="📚")
        
        for i, query in enumerate(alt_queries, 1):
            debug_event("Multi-Query", f"Query {i}/{len(alt_queries)}: {query}", icon="🔍")
            docs = self.handler.hybrid_retriever.get_relevant_documents(query)
            new_ids = {d.metadata.get("document_id") for d in docs[:3] if d.metadata.get("document_id")}
            debug_event("Multi-Query", f"Found {len(new_ids)} additional documents", icon="📄")
            all_doc_ids.update(new_ids)
        
        unique_ids = list(all_doc_ids)[:PARENT_RETRIEVAL_CAP]
        debug_event("Expansion", f"Collected {len(all_doc_ids)} total docs → {len(unique_ids)} unique (capped at {PARENT_RETRIEVAL_CAP})", icon="📊")
        
        debug_event("Expansion", f"Fetching full text for {len(unique_ids)} documents...", icon="📖")
        
        io_start = time.time()
        full_text, t2_map, _ = self.handler.get_full_document_text(unique_ids)
        io_time = time.time() - io_start
        
        debug_event("Expansion", f"Retrieved {len(full_text)} chars in {io_time:.2f}s", icon="✅")
        
        # t2_map now contains {filename: doc_id} for all documents in unique_ids
        if _log_file:
            try:
                _log_file.write(f"Tier 2 document mapping:\n")
                for fname, doc_id in t2_map.items():
                    _log_file.write(f"  {fname} -> {doc_id}\n")
                _log_file.flush()
            except:
                pass
        
        debug_event("Tier 2", "Synthesizing comprehensive answer from expanded context...", icon="🔨")
        
        synthesis_start = time.time()
        
        tier2_prompt = f"""Answer this question comprehensively: {question}

CRITICAL DEDUPLICATION RULES:
- If the same injury/event appears in multiple documents, list it ONCE
- Combine duplicate entries with multiple source citations
- Format: Injury Type | Description | Source(s)
- Example: "Sprained ankle | Ligamentous injury... | RDApp-X.json, RDApp-Y.json"

DO NOT create separate rows for identical entries.
DO NOT add "(repeated)" labels.
Merge duplicate events and reconcile any conflicting information.

Provide specific citations from the documents in a professional table format."""
        
        log_prompt("Tier 2 Synthesis", tier2_prompt)
        
        # Use process_query with the expanded context
        # IMPORTANT: When context is provided, process_query still sets metrics["context"]
        # So we must use the t2_map we got from get_full_document_text() for sources
        final_ans, t2_met = self.handler.process_query(tier2_prompt, context=full_text, label="T2_ASSEMBLY")
        
        log_response("Tier 2 Synthesis", final_ans)
        
        # ERROR HANDLING: Check if LLM returned empty response
        if not final_ans or len(final_ans.strip()) < 100:
            debug_event("Tier 2", f"⚠️ LLM returned insufficient response ({len(final_ans)} chars). Falling back to Tier 1.", icon="⚠️", level="ERROR")
            
            if _log_file:
                try:
                    _log_file.write(f"\n⚠️ TIER 2 SYNTHESIS FAILED\n")
                    _log_file.write(f"Response length: {len(final_ans)} chars\n")
                    _log_file.write(f"Context length: {len(full_text)} chars\n")
                    _log_file.write(f"Falling back to Tier 1 answer\n\n")
                    _log_file.flush()
                except:
                    pass
            
            # Fall back to Tier 1 answer instead of returning empty
            final_ans = t1_ans
            t2_met = t1_met.copy()
            t2_met["fallback_to_tier1"] = True
            combined_sources = t1_met["sources"]  # Use only Tier 1 sources
            
            debug_event("Fallback", "Using Tier 1 answer due to Tier 2 failure", icon="🔙")
        else:
            # Normal flow - Tier 2 succeeded
            synthesis_duration = time.time() - synthesis_start
            debug_event("Tier 2", f"Generated expanded answer ({len(final_ans)} chars) in {synthesis_duration:.2f}s", icon="✅")
            
            t2_met["retrieval_time"] = io_time
            t2_met["synthesis_time"] = synthesis_duration
            
            # CRITICAL FIX: Use t2_map from get_full_document_text, NOT from process_query
            # When process_query receives a context parameter, it doesn't populate metrics["sources"]
            # So t2_map already has the correct {filename: doc_id} mapping for all documents
            
            tier2_duration = time.time() - tier2_start
            
            all_metrics.append({
                "stage": "Tier 2: Multi-Query Assembly",
                "duration": tier2_duration,
                **t2_met
            })
            
            # Combine sources from both tiers
            combined_sources = {**t1_met["sources"], **t2_map}
        
        # Continue with verification regardless of which path we took
        synthesis_duration = time.time() - synthesis_start
        
        # ================================================================
        # TIER 2: Final Verification
        # ================================================================
        debug_event("Tier 2", "Running final adversarial verification...", icon="🛡️")
        
        t2_verification_start = time.time()
        
        # Combine sources from both tiers
        # t1_met["sources"] has Tier 1 sources (filename: doc_id)
        # t2_map has Tier 2 sources (filename: doc_id) from get_full_document_text
        combined_sources = {**t1_met["sources"], **t2_map}
        
        debug_event("Sources", f"T1: {len(t1_met['sources'])}, T2: {len(t2_map)}, Combined: {len(combined_sources)}", icon="📚")
        
        if _log_file:
            try:
                _log_file.write(f"\nSource Combination:\n")
                _log_file.write(f"  Tier 1 sources: {len(t1_met['sources'])}\n")
                _log_file.write(f"  Tier 2 sources: {len(t2_map)}\n")
                _log_file.write(f"  Combined sources: {len(combined_sources)}\n")
                _log_file.write(f"  Documents used in synthesis: {len(unique_ids)}\n")
                if len(combined_sources) != len(unique_ids):
                    _log_file.write(f"  ⚠️ WARNING: Source count mismatch!\n")
                _log_file.flush()
            except:
                pass
        
        t2_sources = self.reconstruct_sources_list(combined_sources)
        
        t2_verdict = self.verify_answer(question, final_ans, t2_sources, t2_met)
        t2_score = t2_verdict.get('citation_score', 0)
        t2_verification_duration = time.time() - t2_verification_start
        
        debug_event("Final Score", f"Tier 2 Verification: {t2_score}/100 (took {t2_verification_duration:.2f}s)", icon="🎯")
        
        all_metrics.append({
            "stage": "Tier 2: Final Verification",
            "duration": t2_verification_duration,
            "score": t2_score,
            "reasoning_length": len(t2_verdict.get('reasoning', '')),
            "fallback_used": t2_verdict.get('fallback_used', False),
            "is_accurate": t2_verdict.get('is_accurate', False)
        })
        
        final_answer = self.attach_verification_report(final_ans, t2_score, t2_verdict['reasoning'])
        
        # ================================================================
        # Return Results
        # ================================================================
        total_time = time.time() - overall_start
        sources_list_output = self.sources_dict_to_list(combined_sources)
        
        debug_event("Complete", f"Investigation complete. Total time: {total_time:.2f}s, Sources: {len(sources_list_output)}", icon="🏁")
        
        return final_answer, sources_list_output, all_metrics, total_time
    
    
    def close(self):
        """Clean up resources and close log file."""
        debug_event("Shutdown", "Closing handlers...", icon="🔚")
        
        global _log_file
        if _log_file is not None:
            try:
                _log_file.write(f"\n{'='*60}\n")
                _log_file.write(f"Session ended: {datetime.now().isoformat()}\n")
                _log_file.write(f"{'='*60}\n")
                _log_file.close()
                debug_event("Shutdown", "Log file closed", icon="✅")
            except Exception as e:
                sys.stderr.write(f"⚠️ [TIERED] Error closing log: {e}\n")
            finally:
                _log_file = None
        
        self.handler.close()
        self.adversarial_handler.close()


# ====================================================================
# Factory Method (for routes.py)
# ====================================================================
def build_agent_from_env() -> TieredHistorianAgent:
    """Factory used by routes.py for lazy initialization."""
    return TieredHistorianAgent()


# ====================================================================
# CLI Test Interface
# ====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python iterative_adversarial_agent.py 'your question here'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    agent = build_agent_from_env()

    try:
        answer, sources, metrics, total_time = agent.investigate(question)
        print(answer)
        print(f"\nSources Used: {len(sources)}")
        print(f"Total Time: {total_time:.2f}s")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        agent.close()