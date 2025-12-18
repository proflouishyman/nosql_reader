#!/usr/bin/env python3
import sys, os, time
from rag_query_handler import RAGQueryHandler
from dotenv import load_dotenv

load_dotenv()

# --- Debug Configuration ---
DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"
TOKEN_WARNING_THRESHOLD = 50000  # Alert if context exceeds this

def debug_step(step_name: str, detail: str = "", icon: str = "⚡", level: str = "INFO"):
    """Formatted pipeline tracer."""
    if not DEBUG:
        return
    timestamp = time.strftime("%H:%M:%S")
    color_icon = "⚠️" if level == "WARN" else "❌" if level == "ERROR" else icon
    
    sys.stderr.write(f"{color_icon} [{timestamp}] [PIPELINE] {step_name.upper()}\n")
    if detail:
        sys.stderr.write(f"   └─ {detail}\n")
    sys.stderr.flush()

class AdversarialRAGHandler:
    def __init__(self):
        debug_step("Initialization", "Spinning up Centralized RAG Handler...", icon="🏗️")
        start_time = time.time()
        
        try:
            self.rag_handler = RAGQueryHandler()
            debug_step("Init Complete", f"Sub-handler ready in {time.time() - start_time:.2f}s", icon="✅")
        except Exception as e:
            debug_step("Init Failed", str(e), level="ERROR")
            raise

    def process_query(self, question: str):
        pipeline_start = time.time()
        
        debug_step("Query Received", f"Input: {question[:100]}...", icon="📥")
        
        # We wrap the call to the underlying handler to capture its metrics for our pipeline log
        debug_step("Dispatching", "Routing to RAGQueryHandler.process_query...", icon="🛰️")
        
        ans, metrics = self.rag_handler.process_query(
            question, 
            context="", 
            label="ONE_SHOT_ADVERSARIAL"
        )
        
        # Check for the token bloat issue we discussed
        total_tokens = metrics.get('tokens', 0)
        if total_tokens > TOKEN_WARNING_THRESHOLD:
            debug_step("Token Alert", 
                       f"Payload is {total_tokens} tokens! This will likely cause latency or context overflow.", 
                       level="WARN")

        total_lat = time.time() - pipeline_start
        
        debug_step("Pipeline Finished", 
                   f"Latency: {total_lat:.2f}s | Retrieval: {metrics.get('retrieval_time', 0):.2f}s | LLM: {metrics.get('llm_time', 0):.2f}s", 
                   icon="🏁")
        
        return ans, total_lat, metrics.get('sources', {})

    def close(self):
        if hasattr(self, 'rag_handler'):
            debug_step("Shutdown", "Closing MongoDB connections via sub-handler...", icon="🔌")
            self.rag_handler.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python adversarial_rag.py 'question'")
    
    # Enable debug by default for this execution if you want to see the verbose output
    DEBUG = True
    
    handler = AdversarialRAGHandler()
    try:
        query = sys.argv[1]
        ans, lat, src = handler.process_query(query)
        
        # Clean Output for the User
        print("\n" + "═"*60)
        print(f"🤖 ADVERSARIAL RESPONSE ({lat:.2f}s)")
        print("─"*60)
        print(ans)
        print("═"*60)
        
        if src:
            print("\n📂 DATA SOURCES:")
            for fname, d_id in sorted(src.items()):
                print(f" • {fname.ljust(35)} [{d_id}]")
        print("\n")

    except Exception as e:
        debug_step("Pipeline Crash", f"Fatal error during execution: {str(e)}", level="ERROR")
        if DEBUG:
            import traceback
            traceback.print_exc()
    finally:
        handler.close()