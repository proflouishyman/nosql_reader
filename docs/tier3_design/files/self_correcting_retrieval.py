# app/historian_agent/self_correcting_retrieval.py
# Created: 2025-12-29
# Purpose: Self-correcting retrieval (Notebook-LLM style)

"""
Self-Correcting Retrieval - Iterative evidence gathering.

Notebook-LLM Principle:
"Uses a loop:
1. Attempt answer
2. Detect missing support
3. Retrieve additional sections
4. Re-evaluate"

Your current Tier 2:
- Generate 3 queries
- Retrieve
- Answer

Notebook-style:
- Attempt answer
- Check: Do I have evidence for each claim?
- If no → Retrieve more
- Re-attempt
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from llm_abstraction import LLMClient
from rag_base import debug_print


@dataclass
class EvidenceGap:
    """Represents a claim without sufficient evidence."""
    claim: str
    required_evidence: str
    current_sources: List[str]
    missing_info: str


class SelfCorrectingRetriever:
    """
    Retriever that iteratively gathers evidence until answer is supported.
    
    Contrasted with your current system:
    - Current: One retrieval pass, hope it works
    - Notebook-style: Loop until evidence sufficient or max iterations
    """
    
    def __init__(self, rag_handler, max_iterations: int = 3):
        """
        Initialize self-correcting retriever.
        
        Args:
            rag_handler: Your existing RAGQueryHandler
            max_iterations: Max retrieval loops
        """
        self.rag_handler = rag_handler
        self.max_iterations = max_iterations
        self.llm = LLMClient()
    
    def retrieve_with_self_correction(
        self,
        question: str,
        initial_documents: List[Dict]
    ) -> Tuple[str, List[Dict], Dict[str, Any]]:
        """
        Retrieve iteratively until answer is supported.
        
        Returns:
            (answer, sources, metadata)
        """
        debug_print("Starting self-correcting retrieval")
        
        current_docs = initial_documents
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            debug_print(f"Iteration {iteration}/{self.max_iterations}")
            
            # 1. Attempt answer with current documents
            answer_attempt = self._attempt_answer(question, current_docs)
            
            # 2. Detect evidence gaps
            gaps = self._detect_evidence_gaps(
                question=question,
                answer=answer_attempt,
                sources=current_docs
            )
            
            if not gaps:
                # Answer is fully supported!
                debug_print(f"Answer fully supported after {iteration} iterations")
                return answer_attempt, current_docs, {'iterations': iteration, 'gaps': []}
            
            debug_print(f"Found {len(gaps)} evidence gaps")
            
            # 3. Retrieve additional evidence for gaps
            additional_docs = self._retrieve_for_gaps(gaps)
            
            if not additional_docs:
                # Can't find more evidence
                debug_print("No additional evidence found, stopping")
                break
            
            # 4. Merge new documents
            current_docs = self._merge_documents(current_docs, additional_docs)
            debug_print(f"Added {len(additional_docs)} documents, total: {len(current_docs)}")
        
        # Final answer with whatever evidence we have
        final_answer = self._attempt_answer(question, current_docs)
        
        return final_answer, current_docs, {
            'iterations': iteration,
            'gaps': [g.claim for g in gaps] if gaps else [],
            'incomplete': len(gaps) > 0
        }
    
    def _attempt_answer(self, question: str, documents: List[Dict]) -> str:
        """Attempt to answer question with current documents."""
        # Format sources
        sources_text = self._format_sources(documents)
        
        # Use strict closed-world prompt
        prompt = f"""You are a historian. Answer ONLY from the sources provided.

SOURCES:
{sources_text}

QUESTION: {question}

RULES:
- If a claim is not in the sources, do NOT include it
- Mark uncertain claims with [UNCERTAIN: ...]
- If you cannot answer, say "Insufficient evidence"

ANSWER:"""
        
        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            profile="quality",
            temperature=0.1
        )
        
        return response.content if response.success else "Error generating answer"
    
    def _detect_evidence_gaps(
        self,
        question: str,
        answer: str,
        sources: List[Dict]
    ) -> List[EvidenceGap]:
        """
        Detect claims in answer that lack source support.
        
        This is the KEY Notebook-LLM feature:
        "I don't have enough evidence yet, go back to the sources."
        """
        # Ask LLM to identify unsupported claims
        sources_text = self._format_sources(sources)
        
        prompt = f"""You are checking if an answer is fully supported by sources.

QUESTION: {question}

ANSWER:
{answer}

SOURCES:
{sources_text}

TASK: Identify claims in the ANSWER that are NOT supported by SOURCES.

For each unsupported claim, explain what additional information would be needed.

Return JSON array:
[
  {{
    "claim": "specific claim from answer",
    "missing_info": "what information is missing from sources"
  }}
]

If all claims are supported, return empty array: []
"""
        
        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            profile="verifier",  # Use verifier for critical evaluation
            temperature=0.0
        )
        
        if not response.success:
            return []
        
        # Parse gaps
        try:
            import json
            from tier0_utils import parse_llm_json
            
            gaps_data = parse_llm_json(response.content, default=[])
            
            gaps = [
                EvidenceGap(
                    claim=g['claim'],
                    required_evidence=g.get('missing_info', 'unknown'),
                    current_sources=[d.get('id', '') for d in sources],
                    missing_info=g.get('missing_info', '')
                )
                for g in gaps_data
            ]
            
            return gaps
            
        except Exception as e:
            debug_print(f"Failed to parse evidence gaps: {e}")
            return []
    
    def _retrieve_for_gaps(self, gaps: List[EvidenceGap]) -> List[Dict]:
        """
        Retrieve additional documents to fill evidence gaps.
        
        Generate targeted queries for missing information.
        """
        additional_docs = []
        
        for gap in gaps[:3]:  # Limit to 3 gaps per iteration
            # Generate search query for missing info
            query = f"{gap.claim} {gap.missing_info}"
            
            # Use your existing retrieval
            try:
                docs = self.rag_handler.retrieve_documents(query, top_k=5)
                additional_docs.extend(docs)
            except Exception as e:
                debug_print(f"Retrieval failed for gap: {e}")
                continue
        
        return additional_docs
    
    def _merge_documents(
        self,
        current: List[Dict],
        additional: List[Dict]
    ) -> List[Dict]:
        """Merge document lists, removing duplicates."""
        seen_ids = {d.get('id', d.get('_id')) for d in current}
        merged = list(current)
        
        for doc in additional:
            doc_id = doc.get('id', doc.get('_id'))
            if doc_id not in seen_ids:
                merged.append(doc)
                seen_ids.add(doc_id)
        
        return merged
    
    def _format_sources(self, documents: List[Dict]) -> str:
        """Format documents for prompt."""
        formatted = []
        for i, doc in enumerate(documents[:20], 1):  # Limit to 20 docs
            content = doc.get('content', doc.get('page_content', ''))[:1000]
            doc_id = doc.get('id', doc.get('_id', f'doc_{i}'))
            formatted.append(f"[Source {i}: {doc_id}]\n{content}\n")
        
        return '\n'.join(formatted)


# ============================================================================
# Integration Example
# ============================================================================

def integrate_with_tiered_agent(agent):
    """
    Add self-correcting retrieval to your tiered agent.
    
    Modifies Tier 2 to use iterative retrieval instead of multi-query.
    """
    corrector = SelfCorrectingRetriever(agent.rag_handler, max_iterations=3)
    
    # In your tier2_deep_investigation():
    # Replace:
    #   answer, sources, metrics = self.rag_handler.query(question)
    # With:
    #   initial_docs = self.rag_handler.retrieve_documents(question)
    #   answer, sources, metrics = corrector.retrieve_with_self_correction(
    #       question, initial_docs
    #   )
    
    return corrector
