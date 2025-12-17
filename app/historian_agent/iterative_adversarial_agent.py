#!/usr/bin/env python3
import sys
import os
import time
import re
from rag_query_handler import RAGQueryHandler, TITLE_FIELDS
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.9

def debug_step(iteration: int, msg: str):
    if DEBUG:
        sys.stderr.write(f"\n🔄 [ROUND {iteration}] {msg}\n")
        sys.stderr.flush()

class IterativeHistorianAgent:
    def __init__(self):
        debug_step(0, "Initializing Core Handler & Reranker...")
        self.rag_handler = RAGQueryHandler()
        from reranking import DocumentReranker
        self.reranker = DocumentReranker()

    def process_investigation(self, question: str):
        overall_start = time.time()
        current_search_query = question
        accumulated_context = []
        seen_source_ids = set()
        sources_map = {}
        iteration_logs = []
        final_answer = ""
        iteration = 1
        
        while iteration <= MAX_ITERATIONS:
            debug_step(iteration, f"Searching for: '{current_search_query}'")
            
            # 1. RETRIEVE & RERANK
            raw_docs = self.rag_handler.hybrid_retriever.get_relevant_documents(current_search_query)
            top_docs = self.reranker.rerank(current_search_query, raw_docs, top_k=10)
            
            # 2. HYDRATE & DE-DUPLICATE
            parent_ids = list(set([d.metadata.get("document_id") for d in top_docs]))
            # FIXED: Calling public method
            meta_map = self.rag_handler.hydrate_parent_metadata(parent_ids)
            
            new_finds = []
            for d in top_docs:
                p_id = str(d.metadata.get("document_id"))
                if p_id not in seen_source_ids:
                    parent = meta_map.get(p_id, {})
                    # FIXED: Calling public method
                    fname = self.rag_handler.get_best_field(parent, TITLE_FIELDS, f"Doc-{p_id[:8]}")
                    
                    accumulated_context.append(f"--- SOURCE: {fname} ---\nTEXT: {d.page_content}\n\n")
                    seen_source_ids.add(p_id)
                    sources_map[fname] = p_id
                    new_finds.append(fname)
            
            iteration_logs.append({"round": iteration, "query": current_search_query, "new_files": new_finds})

            # 3. GENERATE DRAFT
            full_context = "".join(accumulated_context)
            # We use the raw LLM logic through the handler to bypass the handler's internal retrieval
            prompt = f"Using this historical context:\n{full_context}\n\nQuestion: {question}\nAnswer with a detailed markdown table:"
            
            draft_answer, _, _ = self.rag_handler.process_query(prompt)
            final_answer = draft_answer

            # 4. CRITIQUE
            critique_prompt = f"Historical Answer: {draft_answer}\n\nIs this complete? If missing details for names/dates, provide NEW_QUERY: <search terms> and CONFIDENCE: <0.0-1.0>."
            critique_raw, _, _ = self.rag_handler.process_query(critique_prompt)
            
            conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", critique_raw)
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            query_match = re.search(r"NEW_QUERY:\s*(.*)", critique_raw)
            new_query = query_match.group(1).strip() if query_match else "NONE"

            if confidence >= CONFIDENCE_THRESHOLD or "NONE" in new_query.upper():
                debug_step(iteration, f"Investigation concludes (Confidence: {confidence})")
                break
                
            current_search_query = new_query
            iteration += 1

        return final_answer, sources_map, time.time() - overall_start, iteration, iteration_logs

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    
    agent = IterativeHistorianAgent()
    try:
        ans, sources, lat, iters, logs = agent.process_investigation(sys.argv[1])
        
        print(f"\n{'='*80}\nFINAL INVESTIGATION REPORT\n{'='*80}")
        print(ans)
        
        print(f"\n{'='*80}\nREVISION SUMMARY (RESEARCH EVOLUTION)\n{'='*80}")
        for log in logs:
            print(f"Round {log['round']}: Searched for '{log['query']}'")
            if log['new_files']:
                print(f"   └─ Uncovered {len(log['new_files'])} new files: {', '.join(log['new_files'][:2])}...")
            else:
                print(f"   └─ Found no new unique documents.")

        print(f"\n{'='*80}\nSOURCES UTILIZED (TAB-SEPARATED):\n{'='*80}")
        for fname in sorted(sources.keys()):
            print(f"{fname}\t{sources[fname]}")
            
    finally:
        agent.rag_handler.close()