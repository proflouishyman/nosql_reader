#!/usr/bin/env python3
"""
RAG Query Handler - Streamlined Version
Usage: python rag_query_handler.py "What caused train accidents in the 1920s?"
"""

import sys
import os
import logging
import json
import requests
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pymongo import MongoClient

# Import project-specific modules
# (Ensure these files are in the same directory or Python path)
try:
    from chunking import DocumentChunker
    from embeddings import EmbeddingService
    from vector_store import get_vector_store
    from retrievers import VectorRetriever, KeywordRetriever, HybridRetriever
except ImportError as e:
    print(f"Error: Missing required module: {e}")
    sys.exit(1)

# ============================================================================
# GLOBAL CONFIGURATION - MASTER CONTROLS
# ============================================================================
TOP_K = 100
MAX_CONTEXT_TOKENS = 80000
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# MongoDB Configuration
MONGO_URI = "mongodb://admin:secret@mongodb:27017/admin"
DB_NAME = "railroad_documents"

# LLM Configuration
LLM_PROVIDER = "ollama"
LLM_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://host.docker.internal:11434"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 5000

# Embedding Configuration
EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# System Prompt
SYSTEM_PROMPT = """You are a knowledgeable historian assistant specializing in Baltimore & Ohio Railroad history. 
Base your answers on the provided context documents. Cite specific documents when making claims. 
If information is not in the context, say so clearly."""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class SimpleConfig:
    """Simple configuration object for retrievers."""
    def __init__(self, context_fields=None):
        self.context_fields = context_fields or ["title", "content", "ocr_text", "summary", "description"]

@dataclass
class RAGQueryResult:
    answer: str
    sources: List[Dict[str, Any]]
    context_used: str
    retrieval_stats: Dict[str, Any]
    conversation_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ============================================================================
# CORE HANDLER
# ============================================================================

class RAGQueryHandler:
    def __init__(self):
        # Pulling from Global Constants
        self.mongo_uri = os.environ.get('MONGO_URI') or MONGO_URI
        self.db_name = DB_NAME
        self.top_k = TOP_K
        self.llm_model = LLM_MODEL
        self.ollama_base_url = OLLAMA_BASE_URL
        self.temperature = LLM_TEMPERATURE
        
        logger.info(f"--- Initializing RAG Handler (Top-K: {self.top_k}) ---")
        self._init_components()
    
    def _init_components(self):
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.chunks_collection = self.db['document_chunks']
            
            self.embedding_service = EmbeddingService(
                provider=EMBEDDING_PROVIDER,
                model=EMBEDDING_MODEL
            )
            
            self.vector_store = get_vector_store(store_type="chroma")
            
            self.vector_retriever = VectorRetriever(
                vector_store=self.vector_store,
                embedding_service=self.embedding_service,
                mongo_collection=self.chunks_collection,
                top_k=self.top_k
            )
            
            retriever_config = SimpleConfig()
            self.keyword_retriever = KeywordRetriever(
                mongo_collection=self.chunks_collection,
                config=retriever_config,
                top_k=self.top_k
            )
            
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.vector_retriever,
                keyword_retriever=self.keyword_retriever,
                vector_weight=VECTOR_WEIGHT,
                keyword_weight=KEYWORD_WEIGHT,
                top_k=self.top_k
            )
            logger.info("✓ Components Initialized Successfully")
        except Exception as e:
            logger.error(f"Initialization Failed: {e}")
            raise

    def process_query(self, question: str) -> RAGQueryResult:
        start_time = time.time()
        
        # 1. Retrieve
        documents = self.hybrid_retriever.get_relevant_documents(question)
        retrieval_time = time.time() - start_time
        
        # 2. Assemble Context
        context, sources = self._assemble_context(documents)
        
        # 3. Generate Answer
        answer = self._generate_answer(question, context)
        
        return RAGQueryResult(
            answer=answer,
            sources=sources,
            context_used=context[:500] + "...",
            retrieval_stats={
                "num_documents": len(documents),
                "retrieval_time": f"{retrieval_time:.2f}s",
                "top_k": self.top_k
            },
            conversation_id=str(uuid.uuid4())
        )

    def _assemble_context(self, documents: List[Any]) -> tuple[str, List[Dict[str, Any]]]:
        context_parts = []
        sources = []
        total_tokens = 0
        
        for i, doc in enumerate(documents):
            content = doc.page_content
            tokens = len(content) // 4 # Rough token estimation
            
            if total_tokens + tokens > MAX_CONTEXT_TOKENS:
                break
            
            context_parts.append(f"[Document {i+1}]\n{content}\n")
            total_tokens += tokens
            
            metadata = doc.metadata
            sources.append({
                "document_id": str(metadata.get("document_id", "")),
                "title": metadata.get("title", "Unknown Source"),
                "score": metadata.get("score", 0.0)
            })
            
        return "\n".join(context_parts), sources

    def _generate_answer(self, question: str, context: str) -> str:
        if not context:
            return "No relevant historical documents were found."

        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature, "num_predict": LLM_MAX_TOKENS}
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "Error: Empty response from LLM.")
        except Exception as e:
            return f"LLM Error: {str(e)}"

    def close(self):
        self.client.close()

# ============================================================================
# CLEAN CLI EXECUTION
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_query_handler.py \"[Your Question]\"")
        sys.exit(1)

    question = sys.argv[1]
    handler = RAGQueryHandler()
    
    try:
        result = handler.process_query(question)
        
        print("\n" + "="*80)
        print(f"QUERY: {question}")
        print("="*80)
        print(f"\n{result.answer}\n")
        print("="*80)
        print(f"RETRIEVED: {result.retrieval_stats['num_documents']} chunks in {result.retrieval_stats['retrieval_time']}")
        for i, src in enumerate(result.sources, 1):
            print(f"[{i}] {src['title']} (ID: {src['document_id'][:8]})")
        print("="*80 + "\n")
        
    finally:
        handler.close()

if __name__ == "__main__":
    main()