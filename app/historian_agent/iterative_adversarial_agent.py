#!/usr/bin/env python3
import sys, os, time, re
from .rag_query_handler import RAGQueryHandler, count_tokens
from dotenv import load_dotenv

load_dotenv()

# --- Configuration & Monitoring ---
DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"
PARENT_RETRIEVAL_CAP = int(os.environ.get("PARENT_RETRIEVAL_CAP", 15))
CONFIDENCE_THRESHOLD = 0.9

def debug_event(category: str, msg: str, icon: str = "⚙️", level: str = "INFO"):
    if not DEBUG:
        return
    ts = time.strftime("%H:%M:%S")
    prefix = "⚠️" if level == "WARN" else icon
    sys.stderr.write(f"\n{prefix} [{ts}] [{category.upper()}] {msg}\n")
    sys.stderr.flush()

class TieredHistorianAgent:
    def __init__(self):
        debug_event("Init", "Initializing Tiered Agent components...", icon="🏛️")
        self.handler = RAGQueryHandler()
        from .reranking import DocumentReranker
        self.reranker = DocumentReranker()
        debug_event("Init", "Reranker and Handler loaded.")

    def get_context_with_ids(self, query):
        debug_event("Retrieval", f"Fetching context for: {query[:50]}...", icon="🔍")
        raw = self.handler.hybrid_retriever.get_relevant_documents(query)
        
        debug_event("Rerank", f"Applying reranker to {len(raw)} candidates...", icon="⚖️")
        top = self.reranker.rerank(query, raw, top_k=10)
        
        doc_ids = [d.metadata.get('document_id') for d in top]
        meta = self.handler.hydrate_parent_metadata(doc_ids)
        
        context_parts = []
        mapping = {}
        for d in top:
            p_id = str(d.metadata.get('document_id'))
            fname = self.handler.get_best_field(meta.get(p_id, {}), ["filename", "title"], f"Doc-{p_id[:8]}")
            context_parts.append(f"--- SOURCE: {fname} (ID: {p_id}) ---\n{d.page_content}")
            mapping[fname] = p_id
            
        return "\n\n".join(context_parts), mapping

    def investigate(self, question):
        all_metrics = []
        overall_start = time.time()
        
        # --- TIER 1: BROAD PASS ---
        debug_event("Tier 1", "Phase Start: Generating initial draft table...", icon="📝")
        t1_ans, t1_met = self.handler.process_query(f"Create a detailed table for: {question}", label="T1_DRAFT")
        all_metrics.append({"stage": "Tier 1: Broad Pass", **t1_met})
        
        debug_event("Critique", "Phase Start: Self-evaluating confidence and identifying gaps...", icon="🧐")
        crit_ans, crit_met = self.handler.process_query(
            f"Review this: {t1_ans}. Rate CONFIDENCE (0-1) and provide expansion queries.", 
            context=t1_ans, label="T1_CRITIQUE"
        )
        all_metrics.append({"stage": "Critique Phase", **crit_met})
        
        # Extract Confidence
        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", crit_ans)
        conf = float(conf_match.group(1)) if conf_match else 0.5
        
        debug_event("Decision", f"Current Confidence Score: {conf} (Threshold: {CONFIDENCE_THRESHOLD})", icon="⚖️")
        
        if conf >= CONFIDENCE_THRESHOLD:
            debug_event("Finish", f"Confidence satisfies threshold. Returning early.", icon="🏁")
            return t1_ans, t1_met["sources"], all_metrics, time.time() - overall_start

        # --- TIER 2: ESCALATION ---
        debug_event("Tier 2", f"Escalation triggered. Expanding context...", icon="🚀", level="WARN")
        
        queries = re.findall(r"['\"](.*?)['\"]", crit_ans)
        t2_ids = list(t1_met["sources"].values())
        
        debug_event("Expansion", f"Fetching FULL TEXT for {min(len(t2_ids), PARENT_RETRIEVAL_CAP)} documents...", icon="📖")
        full_text, t2_map, io_time = self.handler.get_full_document_text(t2_ids[:PARENT_RETRIEVAL_CAP])
        
        # Token Alert before the final heavy call
        t2_tokens = count_tokens(full_text)
        debug_event("Token Check", f"Tier 2 context size: {t2_tokens} tokens", icon="📊", 
                    level="WARN" if t2_tokens > 50000 else "INFO")

        debug_event("Final Synthesis", "Re-constructing final table using expanded document context...", icon="🧠")
        final_ans, t2_met = self.handler.process_query(
            f"Using full text, construct final table: {question}", 
            context=full_text, label="T2_ASSEMBLY"
        )
        t2_met["retrieval_time"] = io_time
        all_metrics.append({"stage": "Tier 2: Deep Dive", **t2_met})
        
        return final_ans, {**t1_met["sources"], **t2_map}, all_metrics, time.time() - overall_start

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        sys.exit("Usage: python iterative_adversarial_agent.py 'question'")
    
    DEBUG = True # Force for CLI
    agent = TieredHistorianAgent()
    
    try:
        query = sys.argv[1]
        ans, sources, metrics, total_dur = agent.investigate(query)
        
        # FINAL REPORT
        print("\n" + "█"*80)
        print(f" FINAL AGENT OUTPUT ({total_dur:.1f}s)")
        print("█"*80)
        print(f"\n{ans}\n")
        
        print("="*80)
        print(f"{'STAGE':<25} | {'TIME':<8} | {'TOKENS':<10} | {'DOCS':<5}")
        print("-" * 80)
        for m in metrics:
            print(f"{m['stage']:<25} | {m['total_time']:>7.1f}s | {m['tokens']:>10} | {m['doc_count']:>5}")
        
        print("\n" + "═"*80)
        print(f"📊 PERFORMANCE SUMMARY")
        print(f"Peak Tokens:   {max(m['tokens'] for m in metrics):,}")
        print(f"Total Sources: {len(sources)}")
        print("═"*80 + "\n")

    except Exception as e:
        debug_event("Crash", f"Pipeline failed: {str(e)}", icon="❌", level="ERROR")
        if DEBUG:
            import traceback
            traceback.print_exc()
    finally:
        agent.handler.close()