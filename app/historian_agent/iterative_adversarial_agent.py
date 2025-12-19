#!/usr/bin/env python3
import sys, os, time, re, json, requests
from .rag_query_handler import RAGQueryHandler, count_tokens
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"
PARENT_RETRIEVAL_CAP = int(os.environ.get("PARENT_RETRIEVAL_CAP", 8))
CONFIDENCE_THRESHOLD = 0.9

def debug_event(category: str, msg: str, icon: str = "⚙️"):
    if DEBUG: sys.stderr.write(f"\n{icon} [{category.upper()}] {msg}\n")

class TieredHistorianAgent:
    def __init__(self):
        self.handler = RAGQueryHandler()

    def generate_multi_queries(self, original_question):
        """Generate 3 new search angles."""
        prompt = f"""
        TASK: The user asked: "{original_question}"
        Generate 3 NEW, DIFFERENT search queries to find missing details.
        Output JSON list of strings only: ["query1", "query2", "query3"]
        """
        try:
            resp = requests.post(f"{os.environ.get('OLLAMA_BASE_URL')}/api/generate", json={
                "model": os.environ.get("LLM_MODEL"),
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.7}
            })
            return json.loads(resp.json().get("response", "[]"))
        except:
            return [f"{original_question} details", f"{original_question} history"]

    def investigate(self, question):
        all_metrics = []
        overall_start = time.time()
        
        # --- TIER 1: Broad Pass ---
        debug_event("Tier 1", "Initial draft...", icon="📝")
        t1_ans, t1_met = self.handler.process_query(f"Create a detailed table for: {question}", label="T1_DRAFT")
        all_metrics.append({"stage": "Tier 1", **t1_met})
        
        # --- CRITIQUE ---
        debug_event("Critique", "Checking confidence...", icon="🧐")
        crit_ans, _ = self.handler.process_query(
            f"Review this: {t1_ans}. Rate CONFIDENCE (0.0-1.0).", context=t1_ans, label="T1_CRITIQUE"
        )
        # Robust Regex with Fallback
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", crit_ans, re.IGNORECASE)
        conf = float(conf_match.group(1)) if conf_match else 0.5
        debug_event("Decision", f"Confidence: {conf}", icon="⚖️")
        
        if conf >= CONFIDENCE_THRESHOLD:
            return t1_ans, t1_met["sources"], all_metrics, time.time() - overall_start

        # --- TIER 2: Escalation (Multi-Query) ---
        debug_event("Tier 2", "Low confidence. Escalating to Multi-Query...", icon="🚀")
        
        # 1. Generate & Run New Queries
        new_queries = self.generate_multi_queries(question)
        expanded_ids = list(t1_met["sources"].values())
        
        for q in new_queries:
            debug_event("Multi-Query", f"Searching: {q}", icon="Gx")
            # Simulate retrieval step using handler components
            chunks = self.handler.hybrid_retriever.get_relevant_documents(q)
            reranked = self.handler.reranker.rerank(q, chunks, top_k=3)
            expanded_ids.extend([c.metadata.get("document_id") for c in reranked])

        # 2. De-duplicate & Cap
        unique_ids = list(dict.fromkeys(expanded_ids))[:PARENT_RETRIEVAL_CAP]
        
        # 3. Deep Dive Synthesis
        debug_event("Expansion", f"Synthesizing {len(unique_ids)} unique docs...", icon="📖")
        full_text, t2_map, io_time = self.handler.get_full_document_text(unique_ids)
        
        final_ans, t2_met = self.handler.process_query(
            f"Using full text, answer comprehensively: {question}. Merge duplicate events.", 
            context=full_text, label="T2_ASSEMBLY"
        )
        
        t2_met["retrieval_time"] = io_time
        all_metrics.append({"stage": "Tier 2: Multi-Query", **t2_met})
        
        return final_ans, {**t1_met["sources"], **t2_map}, all_metrics, time.time() - overall_start