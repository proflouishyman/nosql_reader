# RAG System Implementation Plan
## Historical Document Reader - Complete RAG Enhancement

**Created:** October 31, 2025  
**Status:** Implementation Ready  
**Priority:** High

---

## Executive Summary

This document outlines the complete implementation of a production-ready Retrieval-Augmented Generation (RAG) system for the Historical Document Reader. The current system has basic LangChain integration but lacks semantic search capabilities. This plan addresses all gaps and provides a phased approach to building a robust RAG pipeline.

---

## Current State Analysis

### What Exists
- ✅ Basic LangChain integration (`historian_agent/__init__.py`)
- ✅ MongoDB document storage
- ✅ Keyword-based retrieval (regex matching)
- ✅ Chat interface with citation support
- ✅ Multi-provider LLM support (OpenAI, Ollama)

### Critical Gaps
- ❌ No vector embeddings
- ❌ No semantic search
- ❌ No document chunking
- ❌ No hybrid retrieval
- ❌ No re-ranking
- ❌ Poor context management

---

## Implementation Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Query Processing Pipeline                       │
│  • Query expansion                                           │
│  • Entity extraction                                         │
│  • Temporal filtering                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Hybrid Retrieval System                         │
│                                                              │
│  ┌──────────────────┐        ┌──────────────────┐          │
│  │  Vector Search   │        │  Keyword Search  │          │
│  │  (ChromaDB)      │        │  (MongoDB)       │          │
│  │  • Embeddings    │        │  • Full-text     │          │
│  │  • Similarity    │        │  • Regex         │          │
│  └────────┬─────────┘        └────────┬─────────┘          │
│           │                           │                     │
│           └────────┬──────────────────┘                     │
│                    │                                        │
│                    ▼                                        │
│           ┌──────────────────┐                             │
│           │  Result Fusion   │                             │
│           │  • RRF Algorithm │                             │
│           └────────┬─────────┘                             │
└────────────────────┼──────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Re-ranking Layer                                │
│  • Cross-encoder scoring                                     │
│  • Temporal relevance boost                                  │
│  • Entity overlap scoring                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Context Assembly                                │
│  • Token budget management                                   │
│  • Source citation preparation                               │
│  • Metadata enrichment                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Generation                                  │
│  • Prompt engineering                                        │
│  • Streaming response                                        │
│  • Citation injection                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Response Post-processing                        │
│  • Citation formatting                                       │
│  • Source metadata                                           │
│  • Confidence scoring                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Algorithm Details

### 1. Document Chunking Algorithm

```
FUNCTION chunk_document(document):
    text = extract_text_fields(document)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    enriched_chunks = []
    FOR i, chunk_text IN enumerate(chunks):
        chunk = {
            "chunk_id": f"{document._id}_chunk_{i:03d}",
            "source_document_id": document._id,
            "chunk_index": i,
            "chunk_text": chunk_text,
            "metadata": extract_metadata(document)
        }
        enriched_chunks.append(chunk)
    
    RETURN enriched_chunks
```

### 2. Hybrid Retrieval Algorithm

```
FUNCTION hybrid_retrieve(query, top_k):
    query_embedding = generate_embedding(query)
    
    vector_results = vector_store.similarity_search(
        query_embedding, 
        k=top_k*2
    )
    
    keyword_results = mongo_search(
        query=query,
        k=top_k*2
    )
    
    fused_results = reciprocal_rank_fusion(
        vector_results,
        keyword_results
    )
    
    RETURN fused_results[:top_k]
```

---

## Implementation Phases

### Phase 1: Foundation (Priority P0)
1. Document chunking system
2. Embedding generation service
3. Vector store integration (ChromaDB)
4. Basic vector search

### Phase 2: Hybrid Retrieval (Priority P1)
5. Enhanced keyword search
6. Reciprocal rank fusion
7. Query preprocessing

### Phase 3: Optimization (Priority P1)
8. Re-ranking system
9. Context management
10. Token budget control

### Phase 4: Advanced (Priority P2)
11. Conversation memory
12. Query routing
13. Monitoring & evaluation

---

## File Structure

```
app/
├── historian_agent/
│   ├── __init__.py                 # Main agent (existing)
│   ├── chunking.py                 # NEW: Document chunking
│   ├── embeddings.py               # NEW: Embedding service
│   ├── vector_store.py             # NEW: ChromaDB interface
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── vector_retriever.py     # NEW: Vector search
│   │   ├── keyword_retriever.py    # Enhanced existing
│   │   └── hybrid_retriever.py     # NEW: Fusion logic
│   ├── reranking.py                # NEW: Re-ranking system
│   ├── context_manager.py          # NEW: Context assembly
│   └── utils.py                    # Shared utilities
```

---

## Success Metrics

- 50% improvement in answer relevance
- 30% reduction in "no answer found"
- < 3 second response time
- User satisfaction > 4/5

---

**Status:** Ready for code implementation
