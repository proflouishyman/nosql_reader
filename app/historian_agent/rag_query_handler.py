#!/usr/bin/env python3
"""
RAG Query Handler - Command-line and Flask-compatible

This script provides RAG-enhanced query processing that can be:
1. Run standalone from command line for testing
2. Imported and called from Flask routes
3. Used to replace/enhance the existing historian agent query endpoint

Architecture:
  User Query → Query Processing → Hybrid Retrieval → Context Assembly → LLM Generation

Usage:
  # Command line
  python rag_query_handler.py "What caused train accidents in the 1920s?"
  
  # From Flask
  from rag_query_handler import process_rag_query
  result = process_rag_query(question, conversation_id, chat_history)
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pymongo import MongoClient
from chunking import DocumentChunker
from embeddings import EmbeddingService
from vector_store import get_vector_store
from retrievers import VectorRetriever, KeywordRetriever, HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Simple config object for retrievers
class SimpleConfig:
    """Simple configuration object for retrievers."""
    def __init__(self, context_fields=None):
        self.context_fields = context_fields or ["title", "content", "ocr_text", "summary", "description"]


@dataclass
class RAGQueryResult:
    """Result from RAG query processing."""
    answer: str
    sources: List[Dict[str, Any]]
    context_used: str
    retrieval_stats: Dict[str, Any]
    conversation_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RAGQueryHandler:
    """
    Handles RAG-enhanced query processing.
    
    This class orchestrates:
    - Query preprocessing
    - Hybrid retrieval (vector + keyword)
    - Context assembly with token management
    - LLM generation with citations
    """
    
    def __init__(
        self,
        mongo_uri: str = None,
        db_name: str = "railroad_documents",
        embedding_provider: str = None,
        embedding_model: str = None,
        top_k: int = 5,
        max_context_tokens: int = 4000,
    ):
        """
        Initialize RAG query handler.
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: Database name
            embedding_provider: 'ollama' or 'openai'
            embedding_model: Model name for embeddings
            top_k: Number of documents to retrieve
            max_context_tokens: Maximum tokens for context
        """
        # Load from environment if not provided
        self.mongo_uri = mongo_uri or self._get_mongo_uri()
        self.db_name = db_name
        self.embedding_provider = embedding_provider or os.environ.get(
            'HISTORIAN_AGENT_EMBEDDING_PROVIDER', 'ollama'
        )
        self.embedding_model = embedding_model or os.environ.get(
            'HISTORIAN_AGENT_EMBEDDING_MODEL', 'qwen3-embedding:0.6b'
        )
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        
        logger.info(f"Initializing RAG Query Handler")
        logger.info(f"  MongoDB: {self.mongo_uri[:60]}...")
        logger.info(f"  Database: {self.db_name}")
        logger.info(f"  Embeddings: {self.embedding_provider}/{self.embedding_model}")
        logger.info(f"  Top-K: {self.top_k}")
        
        # Initialize components
        self._init_components()
    
    def _get_mongo_uri(self) -> str:
        """Get MongoDB URI from environment."""
        return (
            os.environ.get('APP_MONGO_URI') or 
            os.environ.get('MONGO_URI') or 
            "mongodb://admin:secret@mongodb:27017/admin"
        )
    
    def _init_components(self):
        """Initialize RAG components."""
        try:
            # MongoDB connection
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.documents_collection = self.db['documents']
            self.chunks_collection = self.db['document_chunks']
            
            logger.info(f"✓ Connected to MongoDB")
            
            # Embedding service
            self.embedding_service = EmbeddingService(
                provider=self.embedding_provider,
                model=self.embedding_model
            )
            logger.info(f"✓ Initialized embedding service")
            
            # Vector store
            self.vector_store = get_vector_store(store_type="chroma")
            logger.info(f"✓ Connected to vector store")
            
            # Retrievers
            self.vector_retriever = VectorRetriever(
                vector_store=self.vector_store,
                embedding_service=self.embedding_service,
                mongo_collection=self.chunks_collection,
                top_k=self.top_k
            )
            
            # Create simple config for keyword retriever
            retriever_config = SimpleConfig(
                context_fields=["title", "content", "ocr_text", "summary", "description"]
            )
            
            self.keyword_retriever = KeywordRetriever(
                mongo_collection=self.chunks_collection,
                config=retriever_config,
                top_k=self.top_k
            )
            
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.vector_retriever,
                keyword_retriever=self.keyword_retriever,
                vector_weight=0.7,  # Weight for vector search
                keyword_weight=0.3,  # Weight for keyword search
                top_k=self.top_k
            )
            logger.info(f"✓ Initialized retrievers")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise
    
    def process_query(
        self,
        question: str,
        conversation_id: str = None,
        chat_history: List[Dict[str, str]] = None,
        use_hybrid: bool = True,
    ) -> RAGQueryResult:
        """
        Process a query using the RAG pipeline.
        
        Args:
            question: User's question
            conversation_id: Conversation identifier
            chat_history: Previous conversation turns
            use_hybrid: Whether to use hybrid retrieval (vs vector only)
            
        Returns:
            RAGQueryResult with answer and sources
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        conversation_id = conversation_id or self._generate_conversation_id()
        chat_history = chat_history or []
        
        logger.info(f"Processing query: {question[:100]}...")
        logger.info(f"Conversation ID: {conversation_id}")
        
        # Step 1: Retrieve relevant documents
        retrieval_start = self._timestamp()
        
        if use_hybrid:
            documents = self.hybrid_retriever.get_relevant_documents(question)
            retrieval_method = "hybrid"
        else:
            documents = self.vector_retriever.get_relevant_documents(question)
            retrieval_method = "vector"
        
        retrieval_time = self._timestamp() - retrieval_start
        
        logger.info(f"Retrieved {len(documents)} documents using {retrieval_method} search")
        logger.info(f"Retrieval took {retrieval_time:.2f}s")
        
        # Step 2: Assemble context
        context, sources = self._assemble_context(documents)
        
        # Step 3: Generate answer (placeholder - will integrate with existing historian agent)
        answer = self._generate_answer(question, context, chat_history)
        
        # Step 4: Prepare result
        result = RAGQueryResult(
            answer=answer,
            sources=sources,
            context_used=context[:500] + "..." if len(context) > 500 else context,
            retrieval_stats={
                "num_documents": len(documents),
                "retrieval_method": retrieval_method,
                "retrieval_time_seconds": retrieval_time,
                "top_k": self.top_k,
            },
            conversation_id=conversation_id
        )
        
        logger.info("✓ Query processed successfully")
        return result
    
    def _assemble_context(
        self, 
        documents: List[Any]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Assemble context from retrieved documents.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            Tuple of (context_text, sources_list)
        """
        context_parts = []
        sources = []
        total_tokens = 0
        
        for i, doc in enumerate(documents):
            # Estimate tokens (rough approximation)
            content = doc.page_content
            tokens = len(content) // 4
            
            if total_tokens + tokens > self.max_context_tokens:
                logger.info(f"Reached token limit, using {i} of {len(documents)} documents")
                break
            
            # Add to context
            context_parts.append(f"[Document {i+1}]\n{content}\n")
            total_tokens += tokens
            
            # Extract source metadata
            metadata = doc.metadata
            source = {
                "document_id": str(metadata.get("document_id", "")),
                "chunk_id": metadata.get("chunk_id", ""),
                "score": metadata.get("score", 0.0),
                "title": metadata.get("title", ""),
                "date": metadata.get("date", ""),
            }
            sources.append(source)
        
        context = "\n".join(context_parts)
        logger.info(f"Assembled context: {total_tokens} tokens from {len(sources)} documents")
        
        return context, sources
    
    def _generate_answer(
        self,
        question: str,
        context: str,
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate answer using LLM.
        
        This is a placeholder that will be replaced with integration
        to the existing historian agent's LLM chain.
        
        Args:
            question: User's question
            context: Assembled context from retrieved documents
            chat_history: Previous conversation
            
        Returns:
            Generated answer
        """
        # TODO: Integrate with existing historian_agent LLM
        # For now, return a placeholder that includes context info
        
        if not context:
            return "I couldn't find relevant information to answer your question."
        
        # Placeholder response
        answer = f"Based on the historical documents, {question.lower()}\n\n"
        answer += "[Note: This is a placeholder. Full LLM integration pending.]"
        
        return answer
    
    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def close(self):
        """Close connections."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Command-line interface for testing RAG queries."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test RAG query processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  python rag_query_handler.py "What caused train accidents?"
  
  # With custom parameters
  python rag_query_handler.py "safety violations" --top-k 10 --hybrid
  
  # Output as JSON
  python rag_query_handler.py "railroad history" --json
        """
    )
    
    parser.add_argument(
        "question",
        help="Question to ask"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=True,
        help="Use hybrid retrieval (default: True)"
    )
    parser.add_argument(
        "--vector-only",
        action="store_true",
        help="Use vector retrieval only"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )
    parser.add_argument(
        "--mongo-uri",
        help="MongoDB URI (default: from environment)"
    )
    parser.add_argument(
        "--db-name",
        default="railroad_documents",
        help="Database name (default: railroad_documents)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize handler
        handler = RAGQueryHandler(
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            top_k=args.top_k,
        )
        
        # Process query
        use_hybrid = not args.vector_only
        result = handler.process_query(
            question=args.question,
            use_hybrid=use_hybrid
        )
        
        # Output result
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print("\n" + "="*70)
            print("QUERY RESULT")
            print("="*70)
            print(f"\nQuestion: {args.question}")
            print(f"\nAnswer:\n{result.answer}")
            print(f"\nSources: {len(result.sources)} documents")
            for i, source in enumerate(result.sources[:3], 1):
                print(f"  {i}. Doc {source['document_id'][:8]}... "
                      f"(score: {source['score']:.3f})")
            if len(result.sources) > 3:
                print(f"  ... and {len(result.sources) - 3} more")
            print(f"\nRetrieval Stats:")
            for key, value in result.retrieval_stats.items():
                print(f"  {key}: {value}")
            print("="*70 + "\n")
        
        handler.close()
        return 0
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())