#!/usr/bin/env python3
import sys
import os
import time
from rag_query_handler import RAGQueryHandler
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"

def debug_step(step: str, detail: str = ""):
    if DEBUG:
        sys.stderr.write(f"🚀 [PIPELINE] STEP {step}: {detail}\n")
        sys.stderr.flush()

class AdversarialRAGHandler:
    def __init__(self):
        debug_step("1/3", "Initializing Core Handler...")
        self.rag_handler = RAGQueryHandler()
        # We import here to keep the main scope clean
        from reranking import DocumentReranker
        self.reranker = DocumentReranker()

    def process_query(self, question: str):
        pipeline_start = time.time()
        
        debug_step("2/3", "Starting Reranking Pass...")
        t_start = time.time()
        
        # Initial retrieval
        raw_docs = self.rag_handler.hybrid_retriever.get_relevant_documents(question)
        # Apply reranker
        top_docs = self.reranker.rerank(question, raw_docs, top_k=10)
        
        debug_step("2/3", f"Reranking complete in {time.time() - t_start:.2f}s")

        debug_step("3/3", "Executing Final Answer Generation...")
        ans, sources, base_lat = self.rag_handler.process_query(question)
        
        total_lat = time.time() - pipeline_start
        return ans, sources, total_lat

    def close(self):
        """Cleanly close database connections."""
        if hasattr(self, 'rag_handler') and self.rag_handler:
            self.rag_handler.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python adversarial_rag.py 'question'")
    
    handler = AdversarialRAGHandler()
    try:
        ans, src, lat = handler.process_query(sys.argv[1])
        
        # Print the Answer
        print(f"\nADVERSARIAL ANSWER ({lat:.2f}s):\n{ans}\n")
        
        # Print the Sources
        print("SOURCES:")
        unique_sources = {}
        for s in src:
            unique_sources[s['filename']] = s['id']
        
        # FIXED: Iterate through keys only to avoid the Tuple KeyError
        for fname in sorted(unique_sources.keys()):
            doc_id = unique_sources[fname]
            print(f"{fname}\t{doc_id}")
            
    except Exception as e:
        sys.stderr.write(f"❌ ERROR: {str(e)}\n")
    finally:
        handler.close()