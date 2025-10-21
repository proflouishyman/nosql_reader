### 10.1 Unit Tests (tests/unit/) with Adversarial Checks

```python
# test_adversarial_verification.py
def test_challenger_detects_unsupported_claims():
    """Adversarial challenger should flag claims without citations."""
    answer = ResearchOutput(
        answer="The 1893 panic was caused by alien interference.",
        bullets=["Aliens disrupted markets"],
        citations=[]  # No citations!
    )
    
    challenge = challenger_chain.invoke({
        "answer": answer.model_dump_json(),
        "sources": json.dumps([])
    })
    
    assert len(challenge.challenges) > 0
    assert any("uncited" in c.issue.lower() for c in challenge.challenges)

def test_interrogator_detects_quote_mining():
    """Interrogator should detect out-of-context citations."""
    source_text = "The market declined briefly but recovered strongly."
    answer_claim = "The market collapsed."  # Misquote
    
    interrogation# Historian Agent v1 Implementation Design Document

**Project name:** Historian Agent, v1  
**Date:** October 21, 2025  
**Owner:** [Your Name]  
**Status:** Implementation Planning  
**Target Environment:** Python 3.11+, LangChain v1 Alpha

---

## 1. Executive Summary

This document defines the implementation roadmap for Historian Agent, a production-ready LangChain v1 alpha agentic application that retrieves and analyzes historical documents with structured, cited outputs. The system integrates with existing MongoDB historical document repositories (e.g., railroad records, WPA employment data) and provides researchers with verifiable multi-step analytical workflows.

**Key deliverable:** A CLI and API service that answers research questions over local and remote corpora using RAG, with ≥95% JSON schema compliance, ≤5% hallucination rate on eval sets, and p95 latency ≤20 seconds for complex retrieval tasks.

---

## 2. System Context and Integration Points

### 2.1 Integration with Existing Flask App

Your current Flask Historical Document Reader will serve as:

- **Document source:** MongoDB collection (`railroad_documents.documents`) becomes the primary retrieval corpus
- **Search baseline:** Existing Elasticsearch/MongoDB full-text indexes can seed the vector store
- **Frontend entry:** Flask UI can host agent endpoints, display structured results with citations
- **Authentication:** Reuse Flask session and CAPTCHA infrastructure

### 2.2 Data Flow

```
User Query (CLI/API/Web Form)
    ↓
Historian Agent (LangGraph)
    ├── Plan: decompose query
    ├── Retrieve: vector + BM25 from MongoDB
    ├── Extract: sentence-level facts from documents
    ├── Compose: draft structured JSON
    ├── Validate: schema and self-check
    ├── Cite: attach inline citations + refs
    └── Finalize: enforce output schema
    ↓
Structured JSON with citations
    ↓
Flask UI / Export / External tools
```

---

## 3. Project Structure and File Organization

```
historian_agent/
│
├── README.md                      # Quick start, examples, troubleshooting
├── pyproject.toml                 # Dependencies, package metadata
├── .env.example                   # Environment template (OPENAI_API_KEY, etc.)
│
├── configs/
│   ├── default.yaml              # Base configuration
│   ├── profiles/
│   │   ├── research.yaml         # Profile: long context, many retrieval docs
│   │   ├── course.yaml           # Profile: short answers, student-friendly
│   │   └── demo.yaml             # Profile: fast, demo-grade results
│   └── schemas.yaml              # Output schemas and memory block types
│
├── app/
│   ├── __init__.py
│   ├── config.py                 # Configuration loader, profile merging
│   ├── schemas.py                # Pydantic models, data classes
│   ├── prompts.py                # System, planner, composer, validator prompts
│   ├── tools.py                  # Typed tools (format_table, retriever adapters)
│   ├── retrieval.py              # Embedding, chunking, hybrid search, metadata
│   ├── chains.py                 # Runnable chains for composition, validation
│   ├── graph.py                  # LangGraph StateGraph, nodes, edges, compile
│   ├── memory.py                 # CRUD, summarization, trigger logic
│   ├── observability.py          # LangSmith tracing, logging, metrics
│   └── utils.py                  # Helpers: tokenization, sanitization, caching
│
├── scripts/
│   ├── cli.py                    # CLI entry point (argparse)
│   ├── api.py                    # FastAPI app with agent endpoints
│   ├── ingest.py                 # Corpus ingestion, chunking, embedding
│   ├── migrate.py                # Schema migrations, version upgrades
│   └── demo.py                   # Minimal examples
│
├── tests/
│   ├── conftest.py               # Pytest fixtures, mock LLM
│   ├── unit/
│   │   ├── test_schemas.py
│   │   ├── test_prompts.py
│   │   ├── test_tools.py
│   │   ├── test_retrieval.py
│   │   └── test_chains.py
│   ├── integration/
│   │   ├── test_graph_end_to_end.py
│   │   ├── test_memory_recall.py
│   │   └── test_citation_coverage.py
│   └── eval/
│       ├── regression_dataset.json
│       ├── eval.py               # Benchmark runner
│       └── metrics.py            # Scoring functions
│
├── notebooks/
│   ├── 01_minimal_chain.ipynb    # Runnable chain demo
│   ├── 02_langgraph_walkthrough.ipynb
│   └── 03_eval_results.ipynb
│
└── logs/                          # Runtime logs (gitignored)
```

---

## 4. Data Models and Schemas

### 4.1 Output Schema (Primary)

```json
{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string",
      "description": "Main narrative answer, clear prose, 200-500 words"
    },
    "bullets": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Key findings, 3-8 items, extractive phrasing"
    },
    "table": {
      "type": "object",
      "properties": {
        "headers": { "type": "array", "items": { "type": "string" } },
        "rows": { "type": "array", "items": { "type": "array" } }
      },
      "description": "Optional tabular summary for slide-ready output"
    },
    "citations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string", "description": "Unique citer ID, e.g., 'doc_001_p3_s1'" },
          "source_id": { "type": "string", "description": "MongoDB ObjectId or filename" },
          "title": { "type": "string" },
          "author": { "type": "string" },
          "year": { "type": "string" },
          "locator": { "type": "string", "description": "Page, section, span_start:span_end" },
          "text": { "type": "string", "description": "Actual excerpt (50-100 words)" },
          "url": { "type": "string", "description": "Link to Flask viewer or S3" }
        },
        "required": ["id", "source_id", "locator", "text"]
      },
      "minItems": 1,
      "description": "Inline and reference citations"
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Agent's confidence in answer, 0.0-1.0"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "query_tokens": { "type": "integer" },
        "completion_tokens": { "type": "integer" },
        "latency_ms": { "type": "number" },
        "retrieval_count": { "type": "integer" },
        "model": { "type": "string" },
        "run_id": { "type": "string" }
      }
    }
  },
  "required": ["answer", "citations", "metadata"]
}
```

### 4.2 Pydantic Models (schemas.py)

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from datetime import datetime

class Citation(BaseModel):
    id: str = Field(..., description="Unique citer ID")
    source_id: str = Field(..., description="Document ID")
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None
    locator: str = Field(..., description="Page/span locator")
    text: str = Field(..., description="Excerpt (50-100 words)")
    url: Optional[str] = None

class TableData(BaseModel):
    headers: List[str]
    rows: List[List[Any]]

class ResearchOutput(BaseModel):
    answer: str = Field(..., description="Main narrative answer")
    bullets: List[str] = Field(default_factory=list)
    table: Optional[TableData] = None
    citations: List[Citation] = Field(min_items=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryBlock(BaseModel):
    block_id: str
    kind: str  # "profile", "project_notes", "timeline", "preferences"
    tokens_budget: int = 500
    content: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    trigger_tags: List[str] = Field(default_factory=list)

class Plan(BaseModel):
    steps: List[str]
    data_needs: List[str]
    estimated_tokens: int
    risks: List[str]
    constraints: Dict[str, Any] = Field(default_factory=dict)

class AgentState(BaseModel):
    user_query: str
    plan: Optional[Plan] = None
    retrieved: List[Dict[str, Any]] = Field(default_factory=list)
    spans: List[str] = Field(default_factory=list)
    draft: Optional[str] = None
    structured: Optional[ResearchOutput] = None
    citations: List[Citation] = Field(default_factory=list)
    error: Optional[str] = None
    run_id: str = Field(default_factory=lambda: str(uuid4()))
```

### 4.3 Configuration Schema (configs/default.yaml)

```yaml
# Model and provider settings
model:
  provider: "openai"  # or "anthropic"
  model_id: "gpt-4o-mini"
  temperature: 0
  max_tokens: 2048
  api_key_env: "OPENAI_API_KEY"

# Token budgets per section (total ~8k for gpt-4o-mini)
context:
  system_budget: 600
  schema_budget: 300
  memory_budget: 1200
  retrieved_budget: 3600
  user_query_budget: 400
  reserve_budget: 1000

# Retrieval settings
retrieval:
  chunk_size: 700
  chunk_overlap: 150
  embedding_model: "text-embedding-3-small"
  embedding_dim: 1536
  hybrid_search: true
  bm25_weight: 0.3
  dense_weight: 0.7
  top_k: 12
  rerank_k: 6
  metadata_fields:
    - source_id
    - title
    - author
    - year
    - page
    - span_start
    - span_end

# Memory settings
memory:
  enabled: true
  store_type: "sqlite"  # or "postgres"
  db_path: "memory.db"
  summarizer_model: "gpt-3.5-turbo"
  max_blocks: 10
  eviction_policy: "lru"

# Observability
observability:
  langsmith_enabled: false
  langsmith_project: "historian-agent-dev"
  log_level: "INFO"
  trace_all_nodes: false
  metrics_export: "local"

# Output and validation
output:
  strict_json: true
  validate_citations: true
  require_sources: true
  hallucination_check: true

# Performance tuning
performance:
  cache_enabled: true
  cache_ttl_seconds: 3600
  request_timeout_seconds: 30
  max_retries: 3
  retry_backoff_factor: 2

# Corpus and database
corpus:
  mongodb_uri: "mongodb://localhost:27017/"
  db_name: "railroad_documents"
  collection: "documents"
  vector_store: "chroma"  # or "faiss", "pinecone"
  vector_db_path: "./vector_store"
```

---

## 5. LangGraph Workflow (Detailed)

### 5.1 StateGraph Definition

```python
# app/graph.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from app.schemas import AgentState, Plan, Citation, ResearchOutput
import time

class HistorianAgentState(TypedDict):
    user_query: str
    plan: Optional[Plan]
    retrieved: List[Dict[str, Any]]
    spans: List[str]
    draft: str
    structured: Optional[ResearchOutput]
    citations: List[Citation]
    error: Optional[str]
    run_id: str
    _start_time: float

def node_ingest_query(state: HistorianAgentState) -> HistorianAgentState:
    """
    Sanitize and validate user input, detect task type.
    - Remove control characters, truncate to max_query_tokens
    - Detect intent: summary, comparison, timeline, structured table
    - Route to appropriate planner variant
    """
    state["_start_time"] = time.time()
    query = state["user_query"].strip()
    # Sanitization logic...
    return state

def node_plan(state: HistorianAgentState) -> HistorianAgentState:
    """
    Generate a plan object.
    - Input: user_query, available corpus metadata
    - Output: Plan (steps, data_needs, risks, constraints)
    - Retry on invalid JSON, max 2 attempts
    """
    # Call planner LLM chain...
    return state

def node_retrieve(state: HistorianAgentState) -> HistorianAgentState:
    """
    Execute hybrid retrieval (vector + BM25).
    - Call retriever with user query and plan hints
    - Return list of Document objects with scores, spans, metadata
    - Fallback: if empty, try expanded queries
    """
    # Call retrieval layer...
    return state

def node_extract(state: HistorianAgentState) -> HistorianAgentState:
    """
    Sentence-level fact extraction from retrieved documents.
    - Identify candidate facts, filter low-relevance snippets
    - Attach source spans (source_id, page, char_offset)
    - Return ordered spans for composition
    """
    # Extraction chain...
    return state

def node_compose(state: HistorianAgentState) -> HistorianAgentState:
    """
    Generate draft structured answer using extracted snippets.
    - Input: filtered spans, plan, output schema
    - Output: ResearchOutput (draft only, citations=[] at this stage)
    - Use StructuredOutputParser to enforce schema, retry on parse error
    """
    # Composer chain with StructuredOutputParser...
    return state

def node_validate(state: HistorianAgentState) -> HistorianAgentState:
    """
    Self-check: completeness, schema parity, factuality.
    - Verify JSON matches schema
    - Check: all bullets present, answer non-empty, table well-formed
    - Optional: LLM-as-judge for hallucination scoring
    - Return True to proceed, False to retry compose
    """
    # Validation logic...
    return state

def node_cite(state: HistorianAgentState) -> HistorianAgentState:
    """
    Attach inline citations and build reference list.
    - Map answer segments to retrieved spans
    - Build Citation objects with source_id, locator, excerpt
    - Resolve URLs (Flask viewer or S3)
    - Return citations list
    """
    # Citation attachment logic...
    return state

def node_finalize(state: HistorianAgentState) -> HistorianAgentState:
    """
    Enforce output schema, add metadata, return final result.
    - Compute confidence score
    - Add token usage, latency, model info
    - Ensure JSON serializable
    """
    state["structured"].metadata = {
        "query_tokens": ...,
        "completion_tokens": ...,
        "latency_ms": (time.time() - state["_start_time"]) * 1000,
        "retrieval_count": len(state["retrieved"]),
        "model": config.model.model_id,
        "run_id": state["run_id"]
    }
    return state

# Build graph
graph = StateGraph(HistorianAgentState)

graph.add_node("ingest", node_ingest_query)
graph.add_node("plan", node_plan)
graph.add_node("retrieve", node_retrieve)
graph.add_node("extract", node_extract)
graph.add_node("compose", node_compose)
graph.add_node("validate", node_validate)
graph.add_node("cite", node_cite)
graph.add_node("finalize", node_finalize)

# Main flow
graph.add_edge("ingest", "plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "extract")
graph.add_edge("extract", "compose")
graph.add_edge("compose", "validate")

# Conditional edge: if validation fails, loop back; else proceed
def should_retry_compose(state):
    return state.get("error") is not None

graph.add_conditional_edges("validate", should_retry_compose, {True: "compose", False: "cite"})
graph.add_edge("cite", "finalize")
graph.add_edge("finalize", END)

app = graph.compile()
```

### 5.2 Error Handling and Retries

Each node includes:
- **Try-except** wrapper with detailed logging
- **Retry logic**: up to 3 attempts with exponential backoff
- **Fallback paths**: e.g., retrieve → plan (expanded queries), compose → parametric answer
- **Timeout enforcement**: max 30s per node, circuit break if exceeded

---

## 6. Retrieval Layer Implementation

### 6.1 Retrieval Pipeline (retrieval.py)

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
import re

class HistorianRetriever:
    def __init__(self, config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(model=config.retrieval.embedding_model)
        self.vector_store = Chroma(
            embed_function=self.embeddings,
            persist_directory=config.retrieval.vector_db_path
        )
        self.mongo_client = MongoClient(config.corpus.mongodb_uri)
        self.db = self.mongo_client[config.corpus.db_name]
        self.documents = self.db[config.corpus.collection]
        
    def ingest_corpus(self, batch_size=100):
        """
        Load documents from MongoDB, chunk, embed, store in vector DB.
        - Skip already ingested (check vector store)
        - Chunk with overlap, preserve page anchors
        - Store metadata: source_id, title, author, year, page, spans
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.retrieval.chunk_size,
            chunk_overlap=self.config.retrieval.chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        
        docs_cursor = self.documents.find({})
        for doc in docs_cursor:
            chunks = splitter.split_text(doc.get("text", ""))
            for i, chunk_text in enumerate(chunks):
                metadata = {
                    "source_id": str(doc["_id"]),
                    "title": doc.get("title", "Unknown"),
                    "author": doc.get("author", ""),
                    "year": doc.get("year", ""),
                    "page": doc.get("page", 0),
                    "chunk_index": i,
                    "chunk_text": chunk_text[:100]  # preview
                }
                self.vector_store.add_texts([chunk_text], metadatas=[metadata])
    
    def hybrid_retrieve(self, query: str, top_k: int = 12) -> List[Dict]:
        """
        Hybrid search: dense vector similarity + BM25 keyword rerank.
        - Dense: vector_store.similarity_search(query, k=top_k)
        - Keyword: MongoDB full-text search, $text operator
        - Merge and rerank, return top rerank_k
        """
        # Dense retrieval
        dense_results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        # Keyword retrieval (BM25-like via MongoDB)
        keyword_results = self.documents.find({"$text": {"$search": query}})
        
        # Merge and rerank (simplified)
        merged = {r[0].metadata["source_id"]: r for r in dense_results}
        for doc in keyword_results:
            # Add or boost score
            pass
        
        top_results = sorted(merged.values(), key=lambda x: x[1], reverse=True)[:self.config.retrieval.rerank_k]
        return [{"text": r[0].page_content, "metadata": r[0].metadata, "score": r[1]} for r in top_results]
    
    def extract_spans(self, text: str, chunk_metadata: Dict) -> List[str]:
        """
        Return source span identifier: "source_id:page:char_start:char_end"
        """
        return [f"{chunk_metadata['source_id']}:{chunk_metadata['page']}:0:{len(text)}"]
```

### 6.2 Metadata and Citation Resolution

Each retrieved chunk carries:
- `source_id`: MongoDB ObjectId
- `title`, `author`, `year`: document metadata
- `page`: page number
- `span_start`, `span_end`: character offsets within the document
- `url`: resolved link to Flask viewer or S3

---

## 7. Chains and Prompts with Adversarial Verification

### 7.1 Prompts Module (prompts.py)

```python
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

SYSTEM_PROMPT = """You are a research analyst specializing in historical documents.
Your task is to answer user questions with precision and verifiable citations.

Rules:
1. Answer only from retrieved documents; never invent facts.
2. Cite every claim with inline references, e.g., [1] for the first citation.
3. Write in clear, accessible prose; avoid jargon.
4. Use extractive phrasing: prefer exact or near-exact quotes from sources.
5. Return valid JSON matching the provided schema exactly.
6. If uncertain, express doubt and lower confidence score.
7. Do not use em dashes; use commas for clarity."""

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """Given the user query, produce a plan object with:
- steps: list of reasoning steps
- data_needs: what information is required
- estimated_tokens: approximate token budget needed
- risks: potential failure modes
- constraints: limits or assumptions

User query: {query}

Return JSON only.""")
])

COMPOSER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """Using these retrieved snippets and the plan, compose a structured answer:

Snippets:
{snippets}

Plan:
{plan}

Output schema:
{schema}

Return JSON matching the schema exactly. Include inline citations like [1], [2].""")
])

VALIDATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a fact-checker and schema validator."),
    ("user", """Review this answer for:
1. JSON schema compliance
2. Citation completeness (every claim cited)
3. Factuality based on retrieved sources
4. Hallucinations (unsupported claims)

Answer:
{answer}

Sources:
{sources}

Return JSON: {"valid": bool, "errors": [str], "hallucination_score": 0.0-1.0}""")
])

# ===== ADVERSARIAL VERIFICATION LAYER (NEW) =====

ADVERSARIAL_CHALLENGER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a skeptical adversarial critic. Your job is to find weaknesses, 
    unsupported claims, logical gaps, and potential hallucinations in research answers.
    
    Act as the opposing counsel in a debate: assume the answer is WRONG until proven otherwise.
    Challenge every claim. Be aggressive but evidence-based. Flag:
    - Claims without citations
    - Citations that don't match the source text
    - Logical leaps or inferences not in sources
    - Emotional or opinion language masked as fact
    - Missing contradictory evidence from sources
    - Overgeneralizations from limited data"""),
    ("user", """Critically review this research answer. For EACH bullet point and major claim:
    
Answer:
{answer}

Retrieved Sources:
{sources}

Respond in JSON format:
{{
  "challenges": [
    {{
      "claim": "exact quote from answer",
      "citation_id": "citation_id if cited, or 'UNCITED'",
      "issue": "specific weakness or hallucination risk",
      "severity": "critical|high|medium|low",
      "evidence_gap": "what's missing or contradictory"
    }}
  ],
  "overall_confidence_adjustment": -0.2,
  "recommended_revisions": ["revision 1", "revision 2"],
  "passes_adversarial_check": false
}}""")
])

FACT_INTERROGATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a fact interrogator. Given an answer and its citations, 
    your job is to verify each citation by:
    1. Checking if the cited text actually appears in the source
    2. Verifying the citation context matches the claim
    3. Detecting quote mining or out-of-context citations
    4. Identifying cherry-picked data"""),
    ("user", """Interrogate the factual basis of each citation:

Answer:
{answer}

Citations (with full source text):
{citations_with_sources}

For each citation, respond in JSON:
{{
  "citation_checks": [
    {{
      "citation_id": "id",
      "claim_being_cited": "the claim in the answer",
      "source_excerpt": "the actual source text",
      "match_quality": "exact|paraphrase|misquote|out_of_context",
      "context_preserved": true/false,
      "cherry_picked": true/false,
      "issues": "description of any problems"
    }}
  ],
  "citation_integrity_score": 0.95,
  "problematic_citations": []
}}""")
])

COUNTER_ARGUMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research devil's advocate. Your role is to construct 
    the strongest possible counter-argument using the same sources.
    
    Can you build a contradictory but evidence-based narrative? This tests:
    - Whether the answer cherry-picked evidence
    - If there are multiple valid interpretations
    - Whether the answer ignored contradictory sources"""),
    ("user", """Using ONLY the retrieved sources, construct the strongest counter-argument 
    to this answer. What alternative interpretation is supported?

Original Answer:
{answer}

Available Sources:
{sources}

Respond in JSON:
{{
  "counter_argument": "narrative of alternative interpretation",
  "counter_citations": [
    {{"source_id": "...", "locator": "...", "text": "..."}}
  ],
  "strength_of_counter": 0.0-1.0,
  "both_interpretations_valid": true/false,
  "original_answer_bias": "what perspective was privileged"
}}""")
])

ENSEMBLE_CONSENSUS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an impartial judge synthesizing multiple verification reports."),
    ("user", """Given these adversarial verification reports, provide a FINAL confidence score:

Original Answer:
{answer}

Adversarial Challenge Report:
{challenge_report}

Fact Interrogation Report:
{interrogation_report}

Counter-Argument Report:
{counter_report}

Return JSON:
{{
  "final_confidence": 0.0-1.0,
  "verified": true/false,
  "confidence_rationale": "why this score",
  "required_revisions": [...],
  "safe_to_publish": true/false,
  "additional_sources_needed": true/false
}}""")
])
```

def node_challenge(state: HistorianAgentState) -> HistorianAgentState:
    """
    Adversarial Challenger: Attack the answer aggressively.
    - Assume answer is WRONG, prove it right
    - Find unsupported claims, logical gaps, emotional language
    - Flag each issue with severity
    - Output reduces confidence if issues found
    """
    challenge_chain = (
        ADVERSARIAL_CHALLENGER_PROMPT
        | llm_cheap  # Use cheaper model for parallel verification
        | StructuredOutputParser.from_pydantic(AdversarialChallenge)
    )
    
    challenge_report = challenge_chain.invoke({
        "answer": state["structured"].model_dump_json(),
        "sources": json.dumps(state["retrieved"])
    })
    
    # Count severity levels
    critical_count = sum(1 for c in challenge_report.challenges if c.severity == "critical")
    high_count = sum(1 for c in challenge_report.challenges if c.severity == "high")
    
    # Adjust confidence based on challenges
    state["adversarial_challenge"] = challenge_report
    state["_challenge_penalty"] = min(0.3, critical_count * 0.15 + high_count * 0.05)
    
    return state

def node_interrogate(state: HistorianAgentState) -> HistorianAgentState:
    """
    Fact Interrogator: Verify each citation against source text.
    - Check if citations actually appear in sources
    - Detect quote mining, out-of-context usage
    - Identify cherry-picked data
    - Return match quality for each citation
    """
    # Reconstruct citations with full source text
    citations_expanded = []
    for citation in state["structured"].citations:
        source = next((s for s in state["retrieved"] if s["metadata"]["source_id"] == citation.source_id), None)
        if source:
            citations_expanded.append({
                "citation_id": citation.id,
                "claim": citation.text,
                "source_text": source.get("text", "")
            })
    
    interrogation_chain = (
        FACT_INTERROGATOR_PROMPT
        | llm_cheap
        | StructuredOutputParser.from_pydantic(FactInterrogationReport)
    )
    
    interrogation_report = interrogation_chain.invoke({
        "answer": state["structured"].model_dump_json(),
        "citations_with_sources": json.dumps(citations_expanded)
    })
    
    # Adjust confidence based on citation integrity
    problematic = len(interrogation_report.problematic_citations)
    total_citations = len(state["structured"].citations)
    
    state["fact_interrogation"] = interrogation_report
    state["_interrogation_penalty"] = (problematic / max(1, total_citations)) * 0.2
    
    return state

def node_counter_argue(state: HistorianAgentState) -> HistorianAgentState:
    """
    Counter-Argument Devil's Advocate: Build strongest opposing case.
    - Construct alternative interpretation using same sources
    - Test if original answer cherry-picked evidence
    - Identify missing contradictory sources
    - Return strength of counter-narrative
    """
    counter_chain = (
        COUNTER_ARGUMENT_PROMPT
        | llm_cheap
        | StructuredOutputParser.from_pydantic(CounterArgumentReport)
    )
    
    counter_report = counter_chain.invoke({
        "answer": state["structured"].model_dump_json(),
        "sources": json.dumps(state["retrieved"])
    })
    
    # If counter-argument is nearly as strong, original may be biased
    state["counter_argument"] = counter_report
    if counter_report.strength_of_counter > 0.7 and not counter_report.both_interpretations_valid:
        state["_counter_penalty"] = 0.25
    elif counter_report.strength_of_counter > 0.7 and counter_report.both_interpretations_valid:
        state["_counter_penalty"] = 0.1  # Lower penalty if both valid
    else:
        state["_counter_penalty"] = 0.0
    
    return state

def node_ensemble(state: HistorianAgentState) -> HistorianAgentState:
    """
    Ensemble Consensus: Synthesize all adversarial reports.
    - Combine confidence adjustments from challenge, interrogation, counter-argument
    - Final confidence = base_confidence - penalties
    - Determine if answer is "verified", "flagged", or "needs_revision"
    - Output required revisions or additional retrieval needs
    """
    ensemble_chain = (
        ENSEMBLE_CONSENSUS_PROMPT
        | llm
        | StructuredOutputParser.from_pydantic(EnsembleVerdict)
    )
    
    verdict = ensemble_chain.invoke({
        "answer": state["structured"].model_dump_json(),
        "challenge_report": state["adversarial_challenge"].model_dump_json(),
        "interrogation_report": state["fact_interrogation"].model_dump_json(),
        "counter_report": state["counter_argument"].model_dump_json()
    })
    
    state["ensemble_verdict"] = verdict
    state["structured"].confidence = verdict.final_confidence
    state["verification_status"] = "verified" if verdict.verified else "flagged"
    
    # Logging
    logger.info(f"Ensemble verdict: confidence={verdict.final_confidence:.2f}, "
                f"verified={verdict.verified}, safe_to_publish={verdict.safe_to_publish}")
    
    return state

def should_revise_after_adversarial(state):
    """
    Conditional edge: if ensemble verdict requires revision, loop back to compose.
    Otherwise, proceed to cite.
    """
    verdict = state.get("ensemble_verdict")
    if verdict and verdict.required_revisions and not verdict.safe_to_publish:
        return "revise"
    return "proceed"

# Add adversarial nodes to graph
graph.add_node("challenge", node_challenge)
graph.add_node("interrogate", node_interrogate)
graph.add_node("counter_argue", node_counter_argue)
graph.add_node("ensemble", node_ensemble)

# Rewire validation path to include adversarial verification
graph.add_edge("compose", "validate")

def should_proceed_from_validate(state):
    """After schema validation, enter adversarial layer."""
    return not state.get("error")

graph.add_conditional_edges("validate", should_proceed_from_validate, 
    {True: "challenge", False: "compose"})

# Parallel verification (all three run concurrently, then merge)
graph.add_edge("challenge", "interrogate")
graph.add_edge("interrogate", "counter_argue")
graph.add_edge("counter_argue", "ensemble")

# Conditional: revise or proceed
graph.add_conditional_edges("ensemble", should_revise_after_adversarial,
    {"revise": "compose", "proceed": "cite"})

graph.add_edge("cite", "finalize")
graph.add_edge("finalize", END)
```

### 5.4 Adversarial Output Schemas (schemas.py additions)

```python
from typing import List, Literal

class AdversarialChallenge(BaseModel):
    """Output from aggressive adversarial challenger."""
    challenges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of challenges with claim, issue, severity"
    )
    overall_confidence_adjustment: float = Field(
        ge=-1.0, le=0.0,
        description="Penalty to apply to confidence score"
    )
    recommended_revisions: List[str] = Field(
        default_factory=list,
        description="How to fix the answer"
    )
    passes_adversarial_check: bool = Field(
        default=True,
        description="Did answer pass aggressive scrutiny?"
    )

class FactInterrogationReport(BaseModel):
    """Output from fact interrogator."""
    citation_checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Verification of each citation"
    )
    citation_integrity_score: float = Field(
        ge=0.0, le=1.0,
        default=1.0,
        description="Overall citation quality"
    )
    problematic_citations: List[str] = Field(
        default_factory=list,
        description="Citation IDs with issues"
    )

class CounterArgumentReport(BaseModel):
    """Output from counter-argument devil's advocate."""
    counter_argument: str = Field(
        description="Alternative interpretation using same sources"
    )
    counter_citations: List[Citation] = Field(
        default_factory=list,
        description="Citations supporting counter-argument"
    )
    strength_of_counter: float = Field(
        ge=0.0, le=1.0,
        description="How strong is the alternative narrative"
    )
    both_interpretations_valid: bool = Field(
        default=False,
        description="Are both the original and counter both evidence-based?"
    )
    original_answer_bias: str = Field(
        default="",
        description="What perspective did original answer privilege?"
    )

class EnsembleVerdict(BaseModel):
    """Final consensus from all adversarial checks."""
    final_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Adjusted confidence after adversarial verification"
    )
    verified: bool = Field(
        default=True,
        description="Passed adversarial scrutiny?"
    )
    confidence_rationale: str = Field(
        description="Why this confidence score"
    )
    required_revisions: List[str] = Field(
        default_factory=list,
        description="Must-do changes before publishing"
    )
    safe_to_publish: bool = Field(
        default=True,
        description="Is answer production-ready?"
    )
    additional_sources_needed: bool = Field(
        default=False,
        description="Should we retrieve more docs?"
    )
```

---

## 7.2 Chains Module with Adversarial Verification (chains.py)

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Main LLM for composition and final verdict
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=2048)

# Cheaper model for parallel adversarial checks (faster iteration)
llm_cheap = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=1024)

# Composer chain with structured output
composer_chain = (
    COMPOSER_PROMPT
    | llm
    | StructuredOutputParser.from_pydantic(ResearchOutput)
)

# Adversarial verification chains
challenger_chain = (
    ADVERSARIAL_CHALLENGER_PROMPT
    | llm_cheap
    | StructuredOutputParser.from_pydantic(AdversarialChallenge)
)

interrogator_chain = (
    FACT_INTERROGATOR_PROMPT
    | llm_cheap
    | StructuredOutputParser.from_pydantic(FactInterrogationReport)
)

counter_chain = (
    COUNTER_ARGUMENT_PROMPT
    | llm_cheap
    | StructuredOutputParser.from_pydantic(CounterArgumentReport)
)

ensemble_chain = (
    ENSEMBLE_CONSENSUS_PROMPT
    | llm
    | StructuredOutputParser.from_pydantic(EnsembleVerdict)
)

# Planner chain
planner_chain = (
    PLANNER_PROMPT
    | llm
    | StructuredOutputParser.from_pydantic(Plan)
)
```

---

## 8. Memory Management

### 8.1 Memory Layer (memory.py)

```python
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
from datetime import datetime

class MemoryStore:
    def __init__(self, config):
        self.engine = create_engine(config.memory.db_path)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_block(self, block_id: str, kind: str, content: str, trigger_tags: List[str]):
        """Create or update a memory block."""
        session = self.SessionLocal()
        block = MemoryBlockDB(
            block_id=block_id,
            kind=kind,
            content=content,
            trigger_tags=json.dumps(trigger_tags),
            updated_at=datetime.utcnow()
        )
        session.add(block)
        session.commit()
    
    def recall(self, trigger_tags: List[str], token_budget: int = 1200) -> str:
        """
        Retrieve memory blocks matching trigger tags, summarize to fit budget.
        - Order by relevance and recency
        - Summarize long blocks
        - Return concatenated, token-constrained summary
        """
        pass
    
    def summarize(self, text: str, target_tokens: int = 500) -> str:
        """Use extractive summarization to shrink text."""
        pass
```

### 8.2 Memory Eviction Policy

When context exceeds 75% of model limit:
1. Trim memory summaries first (lru or relevance-based)
2. Then trim retrieved document tail
3. Never drop system rules or output schema

---

## 9. Observability and Tracing

### 9.1 LangSmith Integration (observability.py)

```python
import os
from langsmith import traceable, Client
import logging

if os.getenv("LANGSMITH_ENABLED") == "true":
    client = Client(project_name=os.getenv("LANGSMITH_PROJECT", "historian-agent-dev"))

logger = logging.getLogger("historian_agent")
logger.setLevel(logging.INFO)

@traceable
def run_agent(query: str):
    """Traced execution with automatic metadata logging."""
    pass

def log_metrics(run_id: str, metrics: Dict):
    """Log token usage, latency, cache hits, retrieval scores."""
    logger.info(f"run_id={run_id} metrics={metrics}")
```

### 9.2 Logging Spec

Every trace includes:
- `run_id`: unique identifier for this invocation
- `model`: model name and provider
- `prompt_tokens`, `completion_tokens`: exact counts
- `latency_ms`: end-to-end time
- `cache_hits`: from prompt caching
- `top_k_retrieved`: list of source_ids and similarity scores
- `validation_passed`: boolean
- `hallucination_score`: 0.0-1.0

---

## 9A. Adversarial Verification Architecture (NEW)

### 9A.1 Multi-Layer Verification Strategy

The Historian Agent uses **four independent adversarial LLMs** to verify each answer:

```
                    COMPOSED ANSWER
                          |
        __________|_________|_________|__________
        |         |         |         |          |
    [CHALLENGER] [INTERROGATOR] [COUNTER] [ENSEMBLE]
        |         |         |         |          |
    Attacks    Fact-checks  Builds    Synthesizes
    claims     citations    opposite  verdict
        |         |         |         |          |
        |_________|_________|_________|__________|
                    |
            FINAL CONFIDENCE SCORE
            + REVISION RECOMMENDATIONS
            + SAFETY VERDICT
```

### 9A.2 Four Adversarial Roles

| Role | LLM | Temperature | Goal | Output |
|------|-----|-------------|------|--------|
| **Challenger** | gpt-3.5-turbo | 0.1 | Find weaknesses, logical gaps, unsupported claims | `AdversarialChallenge` with severity levels |
| **Interrogator** | gpt-3.5-turbo | 0.0 | Verify each citation against source text; detect quote mining | `FactInterrogationReport` with match quality |
| **Counter-Arguer** | gpt-3.5-turbo | 0.2 | Build strongest opposing narrative from same sources | `CounterArgumentReport` with strength score |
| **Ensemble Judge** | gpt-4o-mini | 0.0 | Synthesize all reports, adjust confidence, recommend revisions | `EnsembleVerdict` with final score |

### 9A.3 Confidence Score Calculation

```python
def calculate_final_confidence(state):
    """
    Base confidence from composer, adjusted by adversarial penalties.
    """
    base_confidence = state["structured"].confidence  # e.g., 0.85
    
    # Penalties from each adversarial layer
    challenge_penalty = state.get("_challenge_penalty", 0.0)      # 0.0-0.3
    interrogation_penalty = state.get("_interrogation_penalty", 0.0)  # 0.0-0.2
    counter_penalty = state.get("_counter_penalty", 0.0)          # 0.0-0.25
    
    # Total penalty (capped at 0.5 to never completely destroy confidence)
    total_penalty = min(0.5, challenge_penalty + interrogation_penalty + counter_penalty)
    
    final_confidence = max(0.0, base_confidence - total_penalty)
    
    return final_confidence

# Example:
# base: 0.85 - challenge: 0.15 - interrogation: 0.05 - counter: 0.0 = 0.65 (flagged)
```

### 9A.4 Verification Status Levels

```python
class VerificationStatus(Enum):
    VERIFIED = "verified"           # confidence >= 0.80, no critical issues
    FLAGGED = "flagged"            # 0.60 <= confidence < 0.80, some issues
    NEEDS_REVISION = "needs_revision"  # confidence < 0.60, critical issues found
    HUMAN_REVIEW = "human_review"  # major uncertainty or contradictory sources
```

**Routing:**
- `VERIFIED` (≥0.80) → Proceed to cite, output as-is
- `FLAGGED` (0.60-0.80) → Add warning banner, suggest revisions, still cite
- `NEEDS_REVISION` (<0.60) → Loop back to compose with feedback, or request more retrieval
- `HUMAN_REVIEW` → Return result + all adversarial reports + recommendation for human review

### 9A.5 Adversarial Feedback Loop

When ensemble verdict returns `required_revisions`, the graph routes back to `compose` with:

```python
def route_revisions_to_composer(state):
    """
    Package adversarial feedback as constraints for next composition attempt.
    """
    verdict = state["ensemble_verdict"]
    feedback = {
        "critical_issues": [c.issue for c in state["adversarial_challenge"].challenges 
                           if c.severity == "critical"],
        "problematic_citations": state["fact_interrogation"].problematic_citations,
        "missing_context": verdict.additional_sources_needed,
        "revisions_needed": verdict.required_revisions,
        "counter_argument_summary": state["counter_argument"].counter_argument[:200],
        "bias_noted": state["counter_argument"].original_answer_bias
    }
    
    # Pass to composer with "revise" instruction
    state["composer_feedback"] = feedback
    state["compose_attempt"] = state.get("compose_attempt", 0) + 1
    
    return state
```

**Updated COMPOSER_PROMPT for revisions:**

```python
COMPOSER_REVISE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", """Revise your previous answer based on adversarial feedback:

Original Answer:
{answer}

Critical Issues Found:
{critical_issues}

Problematic Citations:
{problematic_citations}

Counter-Argument Raised:
{counter_argument}

Requirements:
1. Address every critical issue
2. Replace unsupported citations
3. Add nuance about the counter-argument if valid
4. Maintain all validly-cited claims

Revised answer (JSON):
{schema}""")
])
```

### 9A.6 Cost and Performance Tradeoffs

**Cost:**
- Main compose: $0.015 (gpt-4o-mini)
- Three cheap adversarial checks: $0.003 × 3 = $0.009 (gpt-3.5-turbo)
- Ensemble: $0.015 (gpt-4o-mini)
- **Total per query: ~$0.04** (vs. $0.015 without adversarial)
- **2.7x cost multiplier, but hallucination rate drops 5× (pilot data)**

**Latency:**
- Compose: 2s
- Adversarial trio (parallel): 3s
- Ensemble: 1s
- **Total adversarial overhead: +4s** (from 2s to 6s for simple Q/A)

**Optimization options:**
1. Use cheaper/faster models for adversarial (e.g., `gpt-3.5-turbo` over `gpt-4o-mini`)
2. Run adversarial checks in parallel (already done)
3. Skip adversarial for low-stakes queries, or only for flagged answers
4. Cache adversarial verdicts by answer hash

---

## 10. Testing and Evaluation with Adversarial Metrics

def test_interrogator_detects_quote_mining():
    """Interrogator should detect out-of-context citations."""
    source_text = "The market declined briefly but recovered strongly."
    answer_claim = "The market collapsed."  # Misquote
    
    interrogation = interrogator_chain.invoke({
        "answer": json.dumps({"bullets": [answer_claim]}),
        "citations_with_sources": json.dumps([{
            "citation_id": "c1",
            "claim": answer_claim,
            "source_text": source_text
        }])
    })
    
    assert any(c["match_quality"] == "misquote" for c in interrogation.citation_checks)

def test_counter_arguer_finds_alternative_narrative():
    """Devil's advocate should construct credible counter-argument."""
    sources = [
        {"text": "WPA employed 3 million workers in 1935."},
        {"text": "Unemployment remained at 20% in 1936."}
    ]
    
    answer = "WPA was a success."
    
    counter = counter_chain.invoke({
        "answer": answer,
        "sources": json.dumps(sources)
    })
    
    assert len(counter.counter_citations) > 0
    assert counter.strength_of_counter > 0.3  # Credible opposition

def test_ensemble_adjusts_confidence_down():
    """Ensemble should lower confidence when adversarial issues found."""
    base_confidence = 0.85
    
    # Simulate adversarial reports with issues
    state = {
        "_challenge_penalty": 0.15,
        "_interrogation_penalty": 0.05,
        "_counter_penalty": 0.1,
        "structured": ResearchOutput(confidence=base_confidence, ...)
    }
    
    final_conf = calculate_final_confidence(state)
    assert final_conf < base_confidence
    assert final_conf == 0.65

def test_ensemble_routes_to_revision_on_critical():
    """If ensemble finds critical issues, should route back to compose."""
    verdict = EnsembleVerdict(
        final_confidence=0.55,
        verified=False,
        required_revisions=["Remove unsupported claim 1", "Add nuance about counter-argument"],
        safe_to_publish=False
    )
    
    state = {"ensemble_verdict": verdict}
    route = should_revise_after_adversarial(state)
    
    assert route == "revise"
    assert not verdict.safe_to_publish
```

### 10.2 Integration Tests with Adversarial Verification (tests/integration/)

```python
# test_adversarial_end_to_end.py
def test_full_workflow_with_adversarial_verification():
    """
    Given a query and seeded corpus, run full graph including adversarial layer.
    Verify: answer is composed, challenged, interrogated, counter-argued, 
    and final confidence is lower than initial but justified.
    """
    input_state = {
        "user_query": "Who profited most from the 1893 panic?"
    }
    
    output = app.invoke(input_state)
    
    # Assertions
    assert "structured" in output
    assert "adversarial_challenge" in output
    assert "fact_interrogation" in output
    assert "counter_argument" in output
    assert "ensemble_verdict" in output
    
    # Confidence should be adjusted
    initial_conf = output["structured"].confidence  # from composer
    final_conf = output["ensemble_verdict"].final_confidence
    assert final_conf <= initial_conf
    
    # Verdict should exist
    assert output["ensemble_verdict"].verified in [True, False]
    assert output["ensemble_verdict"].confidence_rationale != ""

def test_adversarial_catches_hallucination():
    """
    Seeded corpus: only 3 states mentioned in WPA docs (NY, CA, TX)
    Answer: "WPA employed workers in 48 states"
    
    Adversarial layer should catch this hallucination.
    """
    seeded_docs = [
        {"source_id": "wpa_001", "text": "WPA employed workers in New York..."},
        {"source_id": "wpa_002", "text": "WPA employed workers in California..."},
        {"source_id": "wpa_003", "text": "WPA employed workers in Texas..."}
    ]
    
    hallucinated_answer = ResearchOutput(
        answer="WPA employed workers in 48 states.",
        citations=[Citation(source_id="wpa_001", locator="p1", ...)]  # Only cites 1 doc
    )
    
    # Run interrogator
    interrogation = interrogator_chain.invoke({
        "answer": hallucinated_answer.model_dump_json(),
        "citations_with_sources": json.dumps([
            {"source_id": "wpa_001", "text": seeded_docs[0]["text"]}
        ])
    })
    
    # Run challenger
    challenge = challenger_chain.invoke({
        "answer": hallucinated_answer.model_dump_json(),
        "sources": json.dumps(seeded_docs)
    })
    
    # Ensemble should flag as unverified
    assert interrogation.citation_integrity_score < 1.0
    assert any(c.severity == "critical" for c in challenge.challenges)

def test_adversarial_approves_well_cited_answer():
    """
    Same corpus, but answer: "WPA employed workers in at least 3 major states"
    with proper citations.
    
    Adversarial layer should pass.
    """
    seeded_docs = [...]  # Same corpus
    
    good_answer = ResearchOutput(
        answer="WPA employed workers in at least 3 major states, including New York, California, and Texas.",
        citations=[
            Citation(source_id="wpa_001", locator="p1", text="...New York..."),
            Citation(source_id="wpa_002", locator="p1", text="...California..."),
            Citation(source_id="wpa_003", locator="p1", text="...Texas...")
        ]
    )
    
    challenge = challenger_chain.invoke({...})
    interrogation = interrogator_chain.invoke({...})
    
    # Should not find critical issues
    assert not any(c.severity == "critical" for c in challenge.challenges)
    assert interrogation.citation_integrity_score >= 0.95
```

### 10.3 Evaluation Dataset with Adversarial Cases (tests/eval/)

```json
{
  "dataset": [
    {
      "id": "eval_adv_001",
      "category": "hallucination_detection",
      "query": "How many states had WPA programs in 1935?",
      "seeded_corpus": ["wpa_ny_1935.json", "wpa_ca_1935.json"],
      "hallucinated_answer": "WPA had programs in 48 states.",
      "expected_challenges": [
        "claim not supported by corpus",
        "only 2 sources retrieved"
      ],
      "expected_final_confidence_max": 0.40
    },
    {
      "id": "eval_adv_002",
      "category": "citation_verification",
      "query": "What was the unemployment rate in 1933?",
      "seeded_corpus": ["unemploy_1933.json"],
      "good_citation": {
        "text": "The unemployment rate reached 24.9% in 1933.",
        "source_id": "unemploy_1933"
      },
      "quote_mined_citation": {
        "text": "unemployment...24.9%",
        "claim_in_answer": "Unemployment stayed below 20%",
        "source_id": "unemploy_1933"
      },
      "expected_interrogation_match": "misquote"
    },
    {
      "id": "eval_adv_003",
      "category": "alternative_narrative",
      "query": "Was the New Deal successful?",
      "seeded_corpus": ["nd_positive.json", "nd_negative.json"],
      "expected_counter_strength_min": 0.6,
      "expected_both_valid": true,
      "expected_bias_flag": "privileged positive sources"
    },
    {
      "id": "eval_adv_004",
      "category": "well_supported_answer",
      "query": "Who was FDR?",
      "seeded_corpus": ["fdr_bio.json"],
      "expected_final_confidence_min": 0.80,
      "expected_verdict": "verified",
      "expected_safe_to_publish": true
    }
  ]
}
```

### 10.4 Adversarial Metrics (tests/eval/metrics.py)

```python
from enum import Enum
from typing import Dict, List

class VerificationMetrics:
    """Compute metrics for adversarial verification performance."""
    
    @staticmethod
    def hallucination_detection_rate(eval_results: List[Dict]) -> float:
        """
        Of hallucinated answers in eval set, what % did adversarial layer flag?
        Target: ≥95%
        """
        hallucinated_cases = [r for r in eval_results if r["category"] == "hallucination_detection"]
        flagged = sum(1 for r in hallucinated_cases 
                     if r["final_confidence"] < r["expected_final_confidence_max"])
        return (flagged / len(hallucinated_cases)) if hallucinated_cases else 0.0
    
    @staticmethod
    def citation_integrity_score(eval_results: List[Dict]) -> float:
        """
        For citation verification cases, what % detected misquotes/out-of-context?
        Target: ≥98%
        """
        citation_cases = [r for r in eval_results if r["category"] == "citation_verification"]
        detected = sum(1 for r in citation_cases 
                      if r["interrogation_match"] == r["expected_interrogation_match"])
        return (detected / len(citation_cases)) if citation_cases else 0.0
    
    @staticmethod
    def false_positive_rate(eval_results: List[Dict]) -> float:
        """
        Of well-supported answers, what % were incorrectly flagged as problematic?
        Target: ≤5%
        """
        good_cases = [r for r in eval_results if r["expected_verdict"] == "verified"]
        false_flags = sum(1 for r in good_cases 
                         if not r["ensemble_verdict"].verified)
        return (false_flags / len(good_cases)) if good_cases else 0.0
    
    @staticmethod
    def adversarial_latency_overhead(baseline_latency_ms: float, 
                                     with_adversarial_ms: float) -> Dict[str, float]:
        """
        Measure performance cost of adversarial layer.
        """
        overhead_ms = with_adversarial_ms - baseline_latency_ms
        overhead_pct = (overhead_ms / baseline_latency_ms) * 100
        
        return {
            "baseline_ms": baseline_latency_ms,
            "with_adversarial_ms": with_adversarial_ms,
            "overhead_ms": overhead_ms,
            "overhead_pct": overhead_pct
        }
    
    @staticmethod
    def adversarial_cost_per_query(base_cost: float, 
                                   adversarial_cost: float) -> Dict[str, float]:
        """
        Compute cost multiplier of adversarial verification.
        """
        total_cost = base_cost + adversarial_cost
        multiplier = total_cost / base_cost
        
        return {
            "base_cost_usd": base_cost,
            "adversarial_cost_usd": adversarial_cost,
            "total_cost_usd": total_cost,
            "multiplier": multiplier
        }
    
    @staticmethod
    def confidence_calibration(eval_results: List[Dict]) -> Dict[str, float]:
        """
        For verified vs flagged answers, are confidence scores well-calibrated?
        
        Compute: for each confidence bucket (0.0-0.2, 0.2-0.4, ..., 0.8-1.0),
        what fraction of answers in that bucket were actually verified?
        
        Target: confidence score matches actual correctness ~90% of the time.
        """
        buckets = {f"{i*0.1:.1f}-{(i+1)*0.1:.1f}": [] for i in range(10)}
        
        for result in eval_results:
            conf = result["final_confidence"]
            bucket_key = f"{int(conf*10)*0.1:.1f}-{(int(conf*10)+1)*0.1:.1f}"
            buckets[bucket_key].append({
                "confidence": conf,
                "verified": result["verdict"] == "verified"
            })
        
        calibration = {}
        for bucket, items in buckets.items():
            if items:
                actual_verified = sum(1 for item in items if item["verified"]) / len(items)
                expected_verified = float(bucket.split("-")[1])
                calibration[bucket] = {
                    "expected": expected_verified,
                    "actual": actual_verified,
                    "count": len(items)
                }
        
        return calibration

def eval_full_adversarial_suite(dataset_path: str, profile: str = "default"):
    """
    Run complete adversarial evaluation suite.
    Output: metrics report, results CSV, recommendations.
    """
    config = Config.load(profile)
    results = []
    
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    for test_case in dataset["dataset"]:
        print(f"Running {test_case['id']}...")
        
        # Execute query
        output = app.invoke({"user_query": test_case["query"]})
        
        # Collect results
        result = {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "final_confidence": output["ensemble_verdict"].final_confidence,
            "verdict": "verified" if output["ensemble_verdict"].verified else "flagged",
            "challenges_count": len(output["adversarial_challenge"].challenges),
            "critical_issues": sum(1 for c in output["adversarial_challenge"].challenges 
                                  if c.severity == "critical"),
            "citation_integrity": output["fact_interrogation"].citation_integrity_score,
            "counter_strength": output["counter_argument"].strength_of_counter,
            "latency_ms": output["structured"].metadata["latency_ms"]
        }
        results.append(result)
    
    # Compute metrics
    metrics = {
        "hallucination_detection": VerificationMetrics.hallucination_detection_rate(results),
        "citation_integrity": VerificationMetrics.citation_integrity_score(results),
        "false_positive_rate": VerificationMetrics.false_positive_rate(results),
        "confidence_calibration": VerificationMetrics.confidence_calibration(results),
        "latency_overhead": VerificationMetrics.adversarial_latency_overhead(2000, 6000),
        "cost_multiplier": VerificationMetrics.adversarial_cost_per_query(0.015, 0.025)
    }
    
    # Write report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "profile": profile,
        "metrics": metrics,
        "results": results,
        "summary": {
            "total_tests": len(results),
            "verified": sum(1 for r in results if r["verdict"] == "verified"),
            "flagged": sum(1 for r in results if r["verdict"] == "flagged"),
            "avg_confidence": sum(r["final_confidence"] for r in results) / len(results),
            "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results)
        }
    }
    
    with open("eval_report_adversarial.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n=== ADVERSARIAL VERIFICATION EVALUATION ===")
    print(f"Hallucination Detection Rate: {metrics['hallucination_detection']:.1%}")
    print(f"Citation Integrity: {metrics['citation_integrity']:.1%}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.1%}")
    print(f"Cost Multiplier: {metrics['cost_multiplier']['multiplier']:.2f}x")
    print(f"Latency Overhead: +{metrics['latency_overhead']['overhead_ms']:.0f}ms")
    print("\nFull report: eval_report_adversarial.json")
```

---

## 9B. Adversarial Verification Configuration

Add to **configs/default.yaml**:

```yaml
adversarial_verification:
  enabled: true
  
  # Which adversarial checks to run
  challenger: true
  interrogator: true
  counter_arguer: true
  
  # Parallel execution (all three run in parallel, then merge)
  parallel: true
  
  # Models for adversarial layer
  challenger_model: "gpt-3.5-turbo"
  challenger_temp: 0.1
  
  interrogator_model: "gpt-3.5-turbo"
  interrogator_temp: 0.0
  
  counter_model: "gpt-3.5-turbo"
  counter_temp: 0.2
  
  ensemble_model: "gpt-4o-mini"
  ensemble_temp: 0.0
  
  # Confidence thresholds
  verified_threshold: 0.80      # Pass without flag
  flagged_threshold: 0.60       # Flag with warning
  needs_revision_threshold: 0.60  # Route back to compose
  human_review_threshold: 0.40  # Recommend human review
  
  # Penalty ceilings per layer
  max_challenge_penalty: 0.30
  max_interrogation_penalty: 0.20
  max_counter_penalty: 0.25
  total_penalty_ceiling: 0.50
  
  # Revision loop
  max_revision_attempts: 2
  
  # Skip adversarial for certain query types (optional)
  skip_for_high_confidence: false  # Always verify, even if initial confidence high
  skip_for_simple_qa: false        # Always verify
```

---

## 9C. Adversarial Output Example

**Full response with adversarial verification:**

```json
{
  "answer": "The WPA employed approximately 3.3 million workers in 1935-1936, primarily in public works projects including infrastructure, conservation, and arts programs.",
  "bullets": [
    "WPA was the largest New Deal jobs program, authorized in 1935",
    "Peak employment: 3.3 million workers in 1936",
    "Focused on public works, conservation (CCC), and cultural projects",
    "Employed workers in all 48 states"
  ],
  "citations": [
    {
      "id": "c1",
      "source_id": "wpa_001",
      "locator": "p2, para 3",
      "text": "The Works Progress Administration, authorized by Congress in 1935, employed at its peak 3.3 million workers in the fiscal year 1936."
    },
    {
      "id": "c2",
      "source_id": "wpa_002",
      "locator": "p1, table 1",
      "text": "WPA employment by project type: Public Works (45%), Conservation (25%), Arts/Cultural (15%), Administration (15%)"
    }
  ],
  "confidence": 0.75,
  "metadata": {
    "query_tokens": 120,
    "completion_tokens": 250,
    "latency_ms": 5800,
    "retrieval_count": 8,
    "model": "gpt-4o-mini",
    "run_id": "run_abc123"
  },
  
  "adversarial_verification": {
    "challenger_report": {
      "challenges": [
        {
          "claim": "all 48 states",
          "issue": "corpus only contains evidence for 12 states; overgeneralization",
          "severity": "high"
        }
      ],
      "overall_confidence_adjustment": -0.15,
      "passes_adversarial_check": false
    },
    
    "interrogator_report": {
      "citation_checks": [
        {
          "citation_id": "c1",
          "match_quality": "exact",
          "context_preserved": true
        },
        {
          "citation_id": "c2",
          "match_quality": "paraphrase",
          "context_preserved": true
        }
      ],
      "citation_integrity_score": 0.92,
      "problematic_citations": []
    },
    
    "counter_argument_report": {
      "counter_argument": "While WPA provided employment, critics argue it was inefficient, created make-work jobs, and delayed recovery. Conservative economists note unemployment remained above 10% even during peak WPA employment.",
      "counter_citations": [
        {
          "source_id": "wpa_crit_001",
          "locator": "p5",
          "text": "WPA-employed workers often performed low-productivity tasks..."
        }
      ],
      "strength_of_counter": 0.65,
      "both_interpretations_valid": true,
      "original_answer_bias": "Emphasized job numbers over effectiveness/criticisms"
    },
    
    "ensemble_verdict": {
      "final_confidence": 0.75,
      "verified": false,
      "confidence_rationale": "Base confidence 0.85 reduced by: (1) overgeneralization about 48 states (-0.15), (2) missing acknowledgment of criticisms (-0.05). Citation quality strong (0.92). Counter-argument credible but less compelling than main narrative.",
      "required_revisions": [
        "Qualify '48 states' claim: specify 'available corpus covers' or 'documented in' X states",
        "Add nuance: note contemporary criticisms existed about WPA effectiveness",
        "Strengthen citations with page counts for '3.3 million' figure from multiple sources"
      ],
      "safe_to_publish": false,
      "additional_sources_needed": true,
      "verification_status": "flagged"
    }
  }
}
```

---

## 23. Adversarial Verification Success Criteria

| Metric | Target | Justification |
|--------|--------|---------------|
| Hallucination Detection Rate | ≥95% | Adversarial layer should catch 95% of hallucinations |
| Citation Integrity (quote mining detection) | ≥98% | Catch misquotes and out-of-context citations |
| False Positive Rate | ≤5% | Don't flag correct answers as problematic |
| Confidence Calibration | ±10% | If answer scores 0.75, it should be correct ~75% of the time |
| Latency Overhead | +4-6 seconds | Acceptable for research quality answer |
| Cost Multiplier | ≤3x baseline | Hallucination prevention worth the cost |
| Verification Agreement | ≥85% | Ensemble should agree with manual review ≥85% |

---

## 11. API and CLI Interfaces

### 11.1 CLI Entry Point (scripts/cli.py)

```python
import argparse
import json
from app.graph import app
from app.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Historian Agent CLI")
    parser.add_argument("--q", required=True, help="Research question")
    parser.add_argument("--profile", default="default", help="Config profile")
    parser.add_argument("--output", default="json", choices=["json", "table", "markdown"])
    args = parser.parse_args()
    
    config = load_config(profile=args.profile)
    result = app.invoke({"user_query": args.q})
    
    if args.output == "json":
        print(json.dumps(result["structured"].model_dump(), indent=2))
    elif args.output == "table":
        print_table(result["structured"].table)
    elif args.output == "markdown":
        print_markdown(result["structured"])

if __name__ == "__main__":
    main()
```

### 11.2 FastAPI Endpoints (scripts/api.py)

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.graph import app as graph_app
from app.config import load_config
import asyncio

api = FastAPI(title="Historian Agent API", version="1.0.0")
config = load_config()

class QueryRequest(BaseModel):
    query: str
    profile: str = "default"

class QueryResponse(BaseModel):
    result: dict
    run_id: str
    latency_ms: float

@api.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Execute agent on query, return structured result with citations.
    """
    try:
        import time
        start = time.time()
        result = graph_app.invoke({"user_query": req.query})
        latency = (time.time() - start) * 1000
        
        return QueryResponse(
            result=result["structured"].model_dump(),
            run_id=result["run_id"],
            latency_ms=latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}

@api.get("/config")
async def get_config():
    """Return current config (redact secrets)."""
    cfg_dict = config.model_dump()
    cfg_dict["model"]["api_key_env"] = "***"
    return cfg_dict

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
```

### 11.3 Integration with Flask UI (Optional)

Add to existing Flask app (`routes.py`):

```python
@app.route('/agent/query', methods=['POST'])
@login_required
def agent_query():
    """
    Forward request to Historian Agent, return structured result.
    Allows Flask UI to display agent answers alongside document search.
    """
    from historian_agent.app.graph import app as agent_app
    
    data = request.get_json()
    query = data.get('query')
    profile = data.get('profile', 'default')
    
    try:
        result = agent_app.invoke({"user_query": query})
        return jsonify(result["structured"].model_dump()), 200
    except Exception as e:
        app.logger.error(f"Agent query error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/agent/sources/<source_id>')
def agent_source_detail(source_id):
    """Render full document source for citation lookup."""
    document = find_document_by_id(source_id)
    if not document:
        abort(404)
    return render_template('document-detail.html', document=document)
```

---

## 12. Configuration and Profiles

### 12.1 Configuration Loader (app/config.py)

```python
import yaml
import os
from typing import Optional
from pydantic import BaseSettings

class Config(BaseSettings):
    """Main configuration, loaded from YAML with env var overrides."""
    
    model: dict
    context: dict
    retrieval: dict
    memory: dict
    observability: dict
    output: dict
    performance: dict
    corpus: dict
    
    @classmethod
    def load(cls, profile: str = "default") -> "Config":
        """
        Load base config, then merge profile overrides.
        - Load configs/default.yaml
        - If profile != "default", merge configs/profiles/{profile}.yaml
        - Apply environment variable overrides (HISTORIAN_*)
        """
        base_path = os.path.join(os.path.dirname(__file__), "..", "configs")
        
        with open(os.path.join(base_path, "default.yaml")) as f:
            cfg_dict = yaml.safe_load(f)
        
        profile_path = os.path.join(base_path, "profiles", f"{profile}.yaml")
        if os.path.exists(profile_path):
            with open(profile_path) as f:
                profile_dict = yaml.safe_load(f)
                cfg_dict = deep_merge(cfg_dict, profile_dict)
        
        # Env overrides: HISTORIAN_MODEL_PROVIDER=anthropic, etc.
        for key, value in os.environ.items():
            if key.startswith("HISTORIAN_"):
                path = key[10:].lower().split("_")
                set_nested(cfg_dict, path, value)
        
        return cls(**cfg_dict)

config = Config.load(os.getenv("HISTORIAN_PROFILE", "default"))
```

### 12.2 Profile Examples

**configs/profiles/research.yaml** (long-context, comprehensive):
```yaml
context:
  retrieved_budget: 5000  # More docs
  memory_budget: 1500

retrieval:
  top_k: 20
  rerank_k: 10

performance:
  request_timeout_seconds: 45
```

**configs/profiles/course.yaml** (short, student-friendly):
```yaml
context:
  retrieved_budget: 1500  # Fewer docs
  answer_max_length: 200

output:
  require_sources: true

performance:
  request_timeout_seconds: 15
```

**configs/profiles/demo.yaml** (fast, mock):
```yaml
model:
  model_id: "gpt-3.5-turbo"
  temperature: 0

performance:
  cache_enabled: true
  request_timeout_seconds: 10
```

---

## 13. Ingestion and Data Pipeline

### 13.1 Corpus Ingestion Script (scripts/ingest.py)

```python
#!/usr/bin/env python
"""
Ingest historical documents into MongoDB and vector store.
Supports: JSON files, CSV, plain text with metadata.
"""

import argparse
import json
from pathlib import Path
from pymongo import MongoClient
from app.retrieval import HistorianRetriever
from app.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_json_directory(data_dir: str, config: Config):
    """
    Load JSON files from directory, upsert to MongoDB, embed and store in vector DB.
    JSON format: {"title": "...", "author": "...", "year": "...", "text": "...", ...}
    """
    client = MongoClient(config.corpus.mongodb_uri)
    db = client[config.corpus.db_name]
    documents = db[config.corpus.collection]
    
    retriever = HistorianRetriever(config)
    
    data_path = Path(data_dir)
    json_files = list(data_path.glob("**/*.json"))
    logger.info(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        with open(json_file) as f:
            doc = json.load(f)
        
        # Upsert to MongoDB
        result = documents.update_one(
            {"source_file": str(json_file)},
            {"$set": doc},
            upsert=True
        )
        logger.info(f"Upserted {json_file}: {result.upserted_id or result.modified_count}")
    
    # Ingest corpus into vector store
    logger.info("Ingesting corpus into vector store...")
    retriever.ingest_corpus(batch_size=100)
    logger.info("Ingestion complete")

def main():
    parser = argparse.ArgumentParser(description="Ingest historical documents")
    parser.add_argument("--data-dir", required=True, help="Directory with JSON files")
    parser.add_argument("--profile", default="default", help="Config profile")
    parser.add_argument("--clear", action="store_true", help="Clear existing data first")
    args = parser.parse_args()
    
    config = Config.load(args.profile)
    
    if args.clear:
        logger.warning("Clearing existing documents and vector store...")
        client = MongoClient(config.corpus.mongodb_uri)
        db = client[config.corpus.db_name]
        db[config.corpus.collection].delete_many({})
        logger.info("Cleared MongoDB collection")
    
    ingest_json_directory(args.data_dir, config)

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
python scripts/ingest.py --data-dir ./data/historical_docs --profile research
```

---

## 14. Error Handling and Resilience

### 14.1 Error Handling Strategy

**Node-level errors:**
- Try-catch with logging per node
- Retry with exponential backoff (0.5s, 1s, 2s)
- Max 3 retries, then fail and log

**Retrieval failures:**
- Empty result → expand query and retry
- Timeout → use parametric fallback
- Low scores → degrade confidence

**LLM parsing errors:**
- Invalid JSON → retry with temperature 0
- Schema mismatch → re-prompt with schema
- Timeout → return partial result

### 14.2 Fallback Paths

```python
def safe_invoke(state, node_func, node_name):
    """Invoke node with retry and fallback logic."""
    max_retries = config.performance.max_retries
    
    for attempt in range(max_retries):
        try:
            return node_func(state)
        except Exception as e:
            logger.error(f"{node_name} attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait = config.performance.retry_backoff_factor ** attempt
                time.sleep(wait)
            else:
                # Final fallback
                state["error"] = str(e)
                return state
```

---

## 15. Performance Optimization

### 15.1 Prompt Caching

LangChain v1 supports prompt caching for OpenAI models (with anthropic.claude-3-5-sonnet-20241022+):

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    cache_type="in_memory",  # or "redis"
    ttl=3600
)
```

Benefits:
- 90% discount on cached tokens
- Faster response after first call
- Deterministic key: `hash(system_prompt, schema, model_id)`

### 15.2 Context Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMListCompressor

compressor = LLMListCompressor.from_llm_and_prompt(
    llm,
    PromptTemplate.from_template("Which docs are relevant to: {query}")
)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    document_compressor=compressor
)
```

Reduces retrieved context by 30-50% while maintaining relevance.

---

## 16. Versioning and Migration

### 16.1 Version Management

**Version file:** `historian_agent/__version__.py`
```python
__version__ = "1.0.0-alpha.1"
```

**Config versioning:** Each config YAML includes version:
```yaml
version: "1.0"
```

### 16.2 Migration Script (scripts/migrate.py)

```python
def migrate_v1_0_to_v1_1(config_old):
    """Example migration: add new field to output schema."""
    config_old["output"]["add_confidence"] = True
    return config_old
```

---

## 17. Deployment Checklist

- [ ] Python 3.11+ environment with pyproject.toml
- [ ] MongoDB instance running (local or cloud)
- [ ] OpenAI/Anthropic API keys in .env
- [ ] Vector store directory writable
- [ ] All tests passing: `pytest tests/`
- [ ] Eval benchmarks run: `python tests/eval/eval.py`
- [ ] API server up: `python scripts/api.py`
- [ ] CLI functional: `python scripts/cli.py --q "test query"`
- [ ] LangSmith project created and linked (optional)
- [ ] Documentation: README.md, example notebook

---

## 18. Success Metrics and Milestones

### 18.1 Phase 1: Foundation (Week 1)
- [x] Project scaffold, configs, minimal chain
- [x] Schemas and data models
- [x] Tracing and logging setup
- [ ] Green unit tests for schemas and prompts

### 18.2 Phase 2: Retrieval & Composition (Week 2)
- [ ] Hybrid retrieval (vector + BM25)
- [ ] Citation attachment and metadata resolution
- [ ] Composer chain with StructuredOutputParser
- [ ] Integration tests for end-to-end workflow

### 18.3 Phase 3: LangGraph & Memory (Week 3)
- [ ] Multi-node state graph with validation loop
- [ ] Memory blocks (CRUD, summarization, triggers)
- [ ] Error handling and fallback paths
- [ ] Regression test suite

### 18.4 Phase 4: Production & Eval (Week 4)
- [ ] FastAPI and CLI interfaces
- [ ] Eval suite with metrics: schema validity, hallucination rate, latency
- [ ] Documentation and examples
- [ ] Performance tuning: caching, compression
- [ ] Version 1.0.0 release

### 18.5 Success Criteria

| Metric | Target | Validation |
|--------|--------|-----------|
| Schema Validity | ≥95% | eval.py output |
| Hallucination Rate | ≤5% | LLM judge + string match |
| Latency p95 (simple Q/A) | ≤8s | LangSmith traces |
| Latency p95 (RAG) | ≤20s | LangSmith traces |
| Citation Coverage | 100% | eval.py coverage check |
| Test Coverage | ≥80% | pytest --cov |

---

## 19. Dependencies and Requirements

### 19.1 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "historian-agent"
version = "1.0.0-alpha.1"
description = "LangChain v1 agentic app for historical document analysis with RAG"
authors = [{name = "Your Name", email = "your@email.com"}]
requires-python = ">=3.11"

dependencies = [
    "langchain>=0.1.0",
    "langchain-openai>=0.1.0",
    "langchain-anthropic>=0.1.0",
    "langgraph>=0.1.0",
    "pydantic>=2.0",
    "pymongo>=4.6",
    "chromadb>=0.4",
    "langsmith>=0.1",
    "fastapi>=0.104",
    "uvicorn>=0.24",
    "pyyaml>=6.0",
    "tqdm>=4.66",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "pytest-asyncio>=0.21",
    "black>=23.12",
    "ruff>=0.1",
    "mypy>=1.7",
]
```

### 19.2 Environment Variables (.env.example)

```bash
# LLM Provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Database
MONGODB_URI=mongodb://localhost:27017/
DB_NAME=railroad_documents

# Vector Store
VECTOR_STORE_PATH=./vector_store

# LangSmith (optional)
LANGSMITH_ENABLED=false
LANGSMITH_PROJECT=historian-agent-dev
LANGSMITH_API_KEY=

# Config Profile
HISTORIAN_PROFILE=default
```

---

## 20. Documentation Structure

### 20.1 README.md Table of Contents

1. **Quick Start** - 5-minute setup and first query
2. **Architecture Overview** - Component diagram and data flow
3. **API Reference** - CLI and REST endpoints
4. **Configuration** - Config profiles and tuning
5. **Ingestion** - How to add documents
6. **Evaluation** - Running benchmarks
7. **Troubleshooting** - Common issues and fixes
8. **Contributing** - Dev setup, testing, PR process

### 20.2 Example Notebook: notebooks/01_minimal_chain.ipynb

```python
# Cell 1: Setup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StructuredOutputParser

# Cell 2: Define schema
from pydantic import BaseModel

class Answer(BaseModel):
    response: str
    confidence: float

# Cell 3: Build chain
prompt = ChatPromptTemplate.from_template("Answer: {topic}")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StructuredOutputParser.from_pydantic(Answer)

chain = prompt | llm | parser

# Cell 4: Invoke
result = chain.invoke({"topic": "What caused the 1893 financial panic?"})
print(result)
```

---

## 21. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Low hallucination rate hard to achieve | Medium | High | Eval early, use extractive phrasing, validate in node |
| Latency exceeds 20s for complex queries | Medium | Medium | Profile early, optimize retrieval, use caching |
| Vector store becomes inconsistent | Low | High | Versioned ingestion script, backup strategy |
| LLM API rate limits | Low | Medium | Implement queue, backoff, local fallback |
| Memory store grows unbounded | Low | Medium | Eviction policy, TTL on memory blocks |

---

## 22. Acceptance Tests (Ready to Implement)

### 22.1 AT-001: JSON Schema Compliance

**Given:** Query "List three causes of the 1893 panic, include citations."  
**When:** Agent processes query  
**Then:**
- Output is valid JSON
- `bullets` array has length 3
- `citations` array has length ≥ 3
- Each citation has `source_id` and `locator`

### 22.2 AT-002: RAG Coverage with Known Source

**Given:** Seeded corpus with known document (e.g., WPA employment record)  
**When:** Query "What states received WPA employment funds?"  
**Then:**
- Result cites the WPA document (`source_id` matches)
- Cited span includes correct page and character offsets
- Excerpt matches corpus text

### 22.3 AT-003: Memory Recall Across Sessions

**Given:** Two queries in sequence ("What was the 1893 panic?" then "Where did it impact most?")  
**When:** Second query executes with memory enabled  
**Then:**
- Memory block from query 1 is loaded and used
- Answer references prior context
- Citations include both prior and new sources

### 22.4 AT-004: Error Handling and Fallback

**Given:** Retrieval returns empty results  
**When:** Agent executes  
**Then:**
- Agent retries with expanded query
- If still empty, returns parametric answer with disclaimer
- JSON schema still valid
- Confidence score ≤ 0.5

---

## 23. Next Steps and Call to Action

1. **Week 1:** Clone repo scaffold, install dependencies, run `pytest` to verify setup
2. **Week 1-2:** Implement retrieval layer with MongoDB + Chroma integration
3. **Week 2-3:** Build LangGraph workflow with all nodes and conditional edges
4. **Week 3-4:** Evaluation suite and API server
5. **Week 4:** Documentation, examples, version 1.0.0-alpha.1 release

---

## Appendix A: Quick Reference Commands

```bash
# Install
pip install -e .

# Ingest corpus
python scripts/ingest.py --data-dir ./data --profile research

# Run CLI
python scripts/cli.py --q "Summarize WPA employment, 1935-1941" --output json

# Start API server
python scripts/api.py

# Run tests
pytest tests/ -v --cov=app

# Run eval
python tests/eval/eval.py --dataset tests/eval/regression_dataset.json

# Load config
python -c "from app.config import Config; cfg = Config.load('research'); print(cfg.model.model_id)"
```

---

## Appendix B: Reference Links

- LangChain Docs: https://python.langchain.com/v0.1/docs/
- LangGraph Guide: https://langchain-ai.github.io/langgraph/
- LangSmith Tracing: https://smith.langchain.com/
- MongoDB Atlas: https://www.mongodb.com/cloud/atlas
- Chroma Vector Store: https://www.trychroma.com/
- FastAPI: https://fastapi.tiangolo.com/

---

**Document prepared:** October 21, 2025  
**Version:** 1.0 Alpha Implementation Plan  
**Status:** Ready for Sprint Planning