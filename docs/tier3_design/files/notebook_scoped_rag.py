# app/historian_agent/notebook_scoped_rag.py
# Created: 2025-12-29
# Purpose: Notebook-scoped RAG (Tier 0 → Tier 1/2 integration)

"""
Notebook-Scoped RAG - Investigation bounded by Tier 0 notebook.

KEY INSIGHT from Notebook-LLM analysis:
"Each notebook is a sealed cognitive workspace.
No cross-notebook leakage.
No background training data invoked implicitly."

Your system:
- Tier 0: Reads 2000 documents → Creates ResearchNotebook
- Tier 1/2: Answers questions

Notebook-style integration:
- Tier 0: Creates bounded workspace
- Tier 1/2: Investigates ONLY within that workspace
- No retrieval outside notebook scope
- No background knowledge allowed

This is THE key difference:
Typical RAG: "Search entire corpus for answer"
Notebook RAG: "Search THIS notebook's documents for answer"
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from research_notebook import ResearchNotebook
from rag_base import debug_print
from llm_abstraction import LLMClient


class NotebookScopedRAG:
    """
    RAG system bounded by a Tier 0 research notebook.
    
    Enforces Notebook-LLM principle:
    "If it's not in the notebook, it doesn't exist."
    """
    
    def __init__(self, notebook: ResearchNotebook, rag_handler):
        """
        Initialize notebook-scoped RAG.
        
        Args:
            notebook: ResearchNotebook from Tier 0 exploration
            rag_handler: Your existing RAGQueryHandler
        """
        self.notebook = notebook
        self.rag_handler = rag_handler
        self.llm = LLMClient()
        
        # Extract document scope from notebook
        self.allowed_doc_ids = self._extract_document_scope()
        
        debug_print(f"Notebook scope: {len(self.allowed_doc_ids)} documents")
    
    def _extract_document_scope(self) -> set:
        """
        Extract set of document IDs that are in notebook scope.
        
        These are the ONLY documents RAG is allowed to retrieve.
        """
        doc_ids = set()
        
        # From patterns
        for pattern in self.notebook.patterns.values():
            doc_ids.update(pattern.evidence_doc_ids)
        
        # From entities
        for entity in self.notebook.entities.values():
            doc_ids.update(entity.document_ids)
        
        # From contradictions
        for contradiction in self.notebook.contradictions:
            doc_ids.add(contradiction.source_a)
            doc_ids.add(contradiction.source_b)
        
        return doc_ids
    
    def query(
        self,
        question: str,
        use_strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Answer question using ONLY notebook-scoped documents.
        
        Args:
            question: User question
            use_strict_mode: If True, enforce closed-world assumption
            
        Returns:
            Answer with sources (all from notebook scope)
        """
        debug_print(f"Notebook query: {question}")
        
        # 1. Retrieve documents (filtered to notebook scope)
        documents = self._retrieve_scoped(question)
        
        if not documents:
            return {
                'answer': "No relevant documents found in this notebook.",
                'sources': [],
                'notebook_scope': True,
                'documents_searched': len(self.allowed_doc_ids)
            }
        
        debug_print(f"Retrieved {len(documents)} scoped documents")
        
        # 2. Generate answer with strict notebook prompt
        if use_strict_mode:
            answer = self._generate_with_notebook_constraints(
                question, documents
            )
        else:
            # Fallback to regular generation
            answer = self._generate_regular(question, documents)
        
        # 3. Verify sources are in scope
        answer['notebook_scope'] = True
        answer['documents_searched'] = len(self.allowed_doc_ids)
        answer['notebook_summary'] = self.notebook.get_summary()
        
        return answer
    
    def _retrieve_scoped(self, question: str, top_k: int = 20) -> List[Dict]:
        """
        Retrieve documents, but ONLY from notebook scope.
        
        This is the key enforcement:
        - Typical RAG: Search all 9,600 documents
        - Notebook RAG: Search only the 2,000 in this notebook
        """
        # Retrieve candidates
        all_candidates = self.rag_handler.retrieve_documents(question, top_k=top_k * 2)
        
        # Filter to notebook scope
        scoped = [
            doc for doc in all_candidates
            if doc.get('id', doc.get('_id')) in self.allowed_doc_ids
        ]
        
        debug_print(f"Filtered {len(all_candidates)} → {len(scoped)} (notebook scope)")
        
        return scoped[:top_k]
    
    def _generate_with_notebook_constraints(
        self,
        question: str,
        documents: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate answer with strict Notebook-LLM constraints.
        
        Enforces:
        1. Closed-world assumption
        2. Quote over paraphrase
        3. Citations required
        4. No background knowledge
        """
        # Build context from documents
        context = self._format_notebook_context(documents)
        
        # Notebook-style prompt (very strict)
        prompt = f"""You are a historian analyzing documents from a research notebook.

NOTEBOOK SCOPE:
This notebook contains {len(self.allowed_doc_ids)} documents about B&O Railroad history.
You have access to {len(documents)} relevant documents for this question.

STRICT RULES:
1. CLOSED-WORLD: If information is not in the documents below, it does NOT exist for this answer
2. NO BACKGROUND KNOWLEDGE: Do not use general knowledge about railroads, history, or anything else
3. QUOTE FIRST: Prefer direct quotes over paraphrasing
4. CITE EVERYTHING: Every fact must reference a specific document
5. FLAG GAPS: If critical information is missing, explicitly state "Not found in notebook documents"

QUESTION: {question}

DOCUMENTS:
{context}

ANSWER (following all rules above):"""
        
        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            profile="quality",
            temperature=0.1  # Low temperature for factual accuracy
        )
        
        if not response.success:
            return {
                'answer': f"Generation error: {response.error}",
                'sources': documents,
                'error': True
            }
        
        return {
            'answer': response.content,
            'sources': documents,
            'prompt_used': 'notebook_strict',
            'error': False
        }
    
    def _generate_regular(self, question: str, documents: List[Dict]) -> Dict[str, Any]:
        """Fallback to regular generation (your existing logic)."""
        # Use your existing RAG generation
        return self.rag_handler.generate_answer(question, documents)
    
    def _format_notebook_context(self, documents: List[Dict]) -> str:
        """
        Format documents as notebook context.
        
        Uses Notebook-style structure:
        [Document ID] (from Section X)
        Content...
        """
        formatted = []
        
        for i, doc in enumerate(documents[:15], 1):  # Limit to 15 for context window
            doc_id = doc.get('id', doc.get('_id', f'Unknown_{i}'))
            content = doc.get('content', doc.get('page_content', ''))[:1500]
            
            # Get metadata from notebook
            doc_info = self._get_notebook_metadata(doc_id)
            
            formatted.append(
                f"[Document {i}: {doc_id}]\n"
                f"Metadata: {doc_info}\n"
                f"Content:\n{content}\n"
                f"{'-'*60}\n"
            )
        
        return '\n'.join(formatted)
    
    def _get_notebook_metadata(self, doc_id: str) -> str:
        """Get metadata about document from notebook."""
        metadata = []
        
        # Check if in patterns
        for pattern in self.notebook.patterns.values():
            if doc_id in pattern.evidence_doc_ids:
                metadata.append(f"Pattern: {pattern.pattern_text[:50]}")
                break
        
        # Check entities
        for entity in self.notebook.entities.values():
            if doc_id in entity.document_ids:
                metadata.append(f"Entity: {entity.name}")
                break
        
        return "; ".join(metadata) if metadata else "General document"
    
    def get_notebook_context_summary(self) -> str:
        """
        Get summary of notebook for context.
        
        Useful for showing user what scope they're querying.
        """
        summary = self.notebook.get_summary()
        
        return f"""Notebook Summary:
- Documents read: {summary['documents_read']}
- Time coverage: {summary['time_coverage']['start']}-{summary['time_coverage']['end']}
- Entities tracked: {summary['total_entities']}
- Patterns identified: {summary['total_patterns']}
- High-confidence patterns: {len(summary['high_confidence_patterns'])}

Top entities:
{chr(10).join(f"  - {name} ({count} docs)" for name, count in summary['top_entities'][:5])}

This is your bounded workspace for investigation.
"""


# ============================================================================
# Integration with Tiered Agent
# ============================================================================

def create_notebook_scoped_investigation(
    notebook_path: Path,
    rag_handler
) -> NotebookScopedRAG:
    """
    Create notebook-scoped investigation from saved Tier 0 notebook.
    
    Usage:
        # After Tier 0 exploration
        notebook = ResearchNotebook.load('path/to/notebook.json')
        
        # Create scoped RAG
        scoped_rag = create_notebook_scoped_investigation(notebook, rag_handler)
        
        # Now all queries are bounded to notebook
        result = scoped_rag.query("What caused the 1923 injury spike?")
        # Only searches the 2000 docs in notebook, not all 9600
    """
    notebook = ResearchNotebook.load(notebook_path)
    return NotebookScopedRAG(notebook, rag_handler)


# ============================================================================
# CLI Example
# ============================================================================

def cli_example():
    """
    Example: Notebook-scoped investigation CLI.
    
    python notebook_scoped_rag.py --notebook path/to/notebook.json
    
    > What caused injury spike in 1923?
    [Searches only notebook documents]
    > Answer: Based on documents in this notebook...
    """
    import sys
    from rag_query_handler import RAGQueryHandler
    
    if len(sys.argv) < 2:
        print("Usage: python notebook_scoped_rag.py <notebook.json>")
        sys.exit(1)
    
    notebook_path = Path(sys.argv[1])
    
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        sys.exit(1)
    
    # Load notebook and create scoped RAG
    rag_handler = RAGQueryHandler()
    scoped_rag = create_notebook_scoped_investigation(notebook_path, rag_handler)
    
    # Show notebook summary
    print(scoped_rag.get_notebook_context_summary())
    print("\n" + "="*60)
    print("Notebook-Scoped Investigation")
    print("="*60 + "\n")
    
    # Interactive query loop
    while True:
        question = input("\n> ")
        
        if question.lower() in ('quit', 'exit', 'q'):
            break
        
        if not question.strip():
            continue
        
        # Query with notebook scope
        result = scoped_rag.query(question)
        
        print(f"\nAnswer ({len(result['sources'])} sources):")
        print(result['answer'])
        print(f"\n[Searched {result['documents_searched']} notebook documents]")


if __name__ == "__main__":
    cli_example()
