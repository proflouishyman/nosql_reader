# app/historian_agent/corpus_explorer.py
# Created: 2025-12-29
# Purpose: Tier 0 - Systematic corpus exploration with cumulative synthesis

"""
Corpus Explorer - Tier 0 of the historian agent system.

This is the "deep map" tier that reads the entire corpus systematically
to discover questions, patterns, and contradictions before any specific
investigation begins.

Architecture:
1. Stratify corpus into readable batches (temporal, genre, biographical)
2. Read each batch with LLM
3. Extract entities, patterns, questions, contradictions
4. Accumulate findings in ResearchNotebook
5. Provide notebook context to each subsequent batch
6. Generate final corpus map and research questions

This replicates how historians actually work: read systematically first,
form questions second, chase specific evidence third.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from rag_base import DocumentStore, debug_print, count_tokens
from llm_abstraction import LLMClient, LLMResponse
from config import APP_CONFIG
from research_notebook import ResearchNotebook
from stratification import CorpusStratifier, StratumReader, Stratum


# ============================================================================
# Prompts
# ============================================================================

BATCH_ANALYSIS_PROMPT = """You are a historian systematically reading through archival documents.

PRIOR KNOWLEDGE (from previous batches):
{prior_knowledge}

NEW BATCH ({batch_label}, {n_docs} documents):
{batch_context}

TASK - Extract findings from this batch:

1. **Entities**: People, organizations, places mentioned
   - Format: {{"name": str, "type": "person|organization|place", "first_seen": doc_id, "context": str}}

2. **Patterns**: Recurring themes, trends, practices
   - What patterns are NEW vs confirming existing patterns?
   - What's SURPRISING or unexpected?
   - Format: {{"pattern": str, "type": str, "evidence": [doc_ids], "confidence": "low|medium|high", "time_range": str}}

3. **Contradictions**: Disagreements between sources
   - Format: {{"claim_a": str, "claim_b": str, "source_a": doc_id, "source_b": doc_id, "context": str}}

4. **Research Questions**: Questions that emerge from this batch
   - What questions does this raise?
   - What would you want to investigate further?
   - Format: {{"question": str, "why_interesting": str, "evidence_needed": str, "time_window": str}}

5. **Temporal Events**: What happens in this time period?
   - Format: {{"year": str, "events": [str]}}

Return ONLY valid JSON:
{{
  "entities": [...],
  "patterns": [...],
  "contradictions": [...],
  "questions": [...],
  "temporal_events": {{"year": [...]}},
  "stats": {{
    "docs_in_batch": int,
    "by_year": {{}},
    "by_collection": {{}},
    "by_document_type": {{}},
    "earliest_year": int,
    "latest_year": int
  }}
}}

Focus on DISCOVERY - what's new, what's surprising, what's contradictory."""

CORPUS_MAP_PROMPT = """You are a historian who has just completed a systematic reading of an archival corpus.

READING SUMMARY:
{notebook_summary}

TASK - Write archive orientation notes:

Based on your systematic reading, provide 5-7 observations about:
1. **Scope**: What does this archive document? What time period, what activities?
2. **Voices**: Whose perspectives are captured? Whose are missing?
3. **Biases**: What selection biases or archival gaps exist?
4. **Surprises**: What was unexpected or contradictory?
5. **Research Potential**: What research questions would this archive support?

Write 2-3 sentences per observation. Be specific and cite patterns/entities you found."""

QUESTION_GENERATION_PROMPT = """You are a historian who has systematically read {docs_read} documents.

FINDINGS:
{notebook_summary}

TASK - Generate research questions:

Based on the patterns, contradictions, and entities you've found, generate 8-12 research questions that:
1. Are answerable with this archive
2. Build on identified patterns or contradictions
3. Connect entities and events
4. Address important historical questions

For each question provide:
- question: The research question
- why_interesting: Why this matters historically
- approach: How you'd investigate it (what documents, what methods)
- entities_involved: Key people/orgs/places

Return JSON array of questions."""


# ============================================================================
# Corpus Explorer
# ============================================================================

class CorpusExplorer:
    """
    Tier 0: Systematic corpus exploration with cumulative synthesis.
    
    This is the "reading pass" that discovers questions before answering them.
    """
    
    def __init__(self):
        """Initialize corpus explorer."""
        debug_print("Initializing CorpusExplorer (Tier 0)")
        
        self.doc_store = DocumentStore()
        self.llm = LLMClient()
        self.stratifier = CorpusStratifier()
        self.reader = StratumReader()
        
        # Notebook for accumulating findings
        self.notebook = ResearchNotebook()
        
        # Logging
        self.log_dir = Path("/app/logs/corpus_exploration")
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def explore(
        self,
        strategy: str = 'balanced',
        total_budget: int = 2000,
        year_range: Optional[Tuple[int, int]] = None,
        save_notebook: bool = True
    ) -> Dict[str, Any]:
        """
        Execute systematic corpus exploration.
        
        Args:
            strategy: 'temporal', 'biographical', 'genre', or 'balanced'
            total_budget: Total number of documents to read
            year_range: Optional (start_year, end_year) filter
            save_notebook: Whether to save notebook to disk
            
        Returns:
            Exploration report with corpus_map, questions, patterns, etc.
        """
        start_time = time.time()
        debug_print(f"Starting corpus exploration: {strategy}, budget: {total_budget}")
        
        # Build stratification strategy
        strata = self._build_strata(strategy, total_budget, year_range)
        
        debug_print(f"Stratification: {len(strata)} batches, ~{sum(s.sample_size for s in strata)} docs")
        
        # Read each stratum and accumulate findings
        for i, stratum in enumerate(strata):
            debug_print(f"Processing batch {i+1}/{len(strata)}: {stratum.label}")
            
            try:
                self._process_stratum(stratum)
            except Exception as e:
                debug_print(f"Error processing {stratum.label}: {e}")
                continue
        
        # Generate final outputs
        corpus_map = self._generate_corpus_map()
        questions = self._generate_research_questions()
        
        # Build report
        report = {
            'corpus_map': corpus_map,
            'questions': questions,
            'patterns': self._export_patterns(),
            'entities': self._export_entities(),
            'contradictions': self._export_contradictions(),
            'notebook_summary': self.notebook.get_summary(),
            'exploration_metadata': {
                'strategy': strategy,
                'total_budget': total_budget,
                'documents_read': self.notebook.corpus_map['total_documents_read'],
                'batches_processed': self.notebook.corpus_map['batches_processed'],
                'duration_seconds': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Save notebook
        if save_notebook:
            notebook_path = self.log_dir / f"notebook_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.notebook.save(str(notebook_path))
            debug_print(f"Notebook saved to {notebook_path}")
            report['notebook_path'] = str(notebook_path)
        
        debug_print(f"Exploration complete: {report['exploration_metadata']['documents_read']} docs, {len(questions)} questions")
        
        return report
    
    def _build_strata(
        self,
        strategy: str,
        total_budget: int,
        year_range: Optional[Tuple[int, int]]
    ) -> List[Stratum]:
        """Build stratification strategy."""
        if strategy == 'temporal' and year_range:
            # Temporal with year range
            strata = self.stratifier.temporal_stratification(
                year_range=year_range,
                docs_per_year=min(50, total_budget // 20)
            )
        else:
            # Use comprehensive strategy
            strata = self.stratifier.build_comprehensive_strategy(
                total_budget=total_budget,
                strategy=strategy
            )
        
        return strata
    
    def _process_stratum(self, stratum: Stratum) -> None:
        """
        Process one stratum: read docs, analyze with LLM, update notebook.
        
        Args:
            stratum: Stratum to process
        """
        # Read documents from stratum
        docs = self.reader.read_stratum(stratum)
        
        if not docs:
            debug_print(f"No documents in {stratum.label}, skipping")
            return
        
        # Format for LLM
        batch_context = self.reader.format_docs_for_llm(docs, max_chars=60000)
        
        # Get prior knowledge from notebook
        prior_knowledge = self.notebook.format_for_llm_context()
        
        # Build prompt
        prompt = BATCH_ANALYSIS_PROMPT.format(
            prior_knowledge=prior_knowledge,
            batch_label=stratum.label,
            n_docs=len(docs),
            batch_context=batch_context
        )
        
        # Call LLM
        debug_print(f"Calling LLM for batch analysis: {count_tokens(prompt)} tokens")
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian performing systematic archival research."},
                {"role": "user", "content": prompt}
            ],
            profile="quality",  # Use quality model for analysis
            temperature=0.3,
            timeout=180.0
        )
        
        if not response.success:
            debug_print(f"LLM call failed: {response.error}")
            return
        
        # Parse findings
        try:
            findings = self._parse_llm_response(response.content)
        except Exception as e:
            debug_print(f"Failed to parse LLM response: {e}")
            return
        
        # Integrate findings into notebook
        self.notebook.integrate_batch_findings(findings, stratum.label)
        
        debug_print(f"Batch complete: {len(findings.get('entities', []))} entities, {len(findings.get('patterns', []))} patterns")
    
    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM JSON response with error handling.
        
        Args:
            content: LLM response text
            
        Returns:
            Parsed findings dict
        """
        # Strip markdown code blocks if present
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        content = content.strip()
        
        # Parse JSON
        findings = json.loads(content)
        
        # Validate required fields
        required = ['entities', 'patterns', 'contradictions', 'questions', 'temporal_events', 'stats']
        for field in required:
            if field not in findings:
                findings[field] = [] if field != 'stats' else {}
        
        return findings
    
    def _generate_corpus_map(self) -> Dict[str, Any]:
        """Generate archive orientation notes."""
        debug_print("Generating corpus map")
        
        notebook_summary = json.dumps(self.notebook.get_summary(), indent=2)
        
        prompt = CORPUS_MAP_PROMPT.format(notebook_summary=notebook_summary)
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian writing archive orientation notes."},
                {"role": "user", "content": prompt}
            ],
            profile="quality",
            temperature=0.4
        )
        
        if not response.success:
            debug_print(f"Failed to generate corpus map: {response.error}")
            archive_notes = "Failed to generate archive notes."
        else:
            archive_notes = response.content
        
        return {
            'statistics': self.notebook.corpus_map,
            'archive_notes': archive_notes
        }
    
    def _generate_research_questions(self) -> List[Dict[str, Any]]:
        """Generate research questions from accumulated findings."""
        debug_print("Generating research questions")
        
        notebook_summary = json.dumps(self.notebook.get_summary(), indent=2)
        docs_read = self.notebook.corpus_map['total_documents_read']
        
        prompt = QUESTION_GENERATION_PROMPT.format(
            docs_read=docs_read,
            notebook_summary=notebook_summary
        )
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian identifying research questions."},
                {"role": "user", "content": prompt}
            ],
            profile="quality",
            temperature=0.5,
            max_tokens=4000
        )
        
        if not response.success:
            debug_print(f"Failed to generate questions: {response.error}")
            return []
        
        try:
            # Parse JSON response
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            
            questions = json.loads(content.strip())
            
            if isinstance(questions, dict) and 'questions' in questions:
                questions = questions['questions']
            
            return questions
            
        except Exception as e:
            debug_print(f"Failed to parse questions: {e}")
            return []
    
    def _export_patterns(self) -> List[Dict[str, Any]]:
        """Export patterns from notebook."""
        return [
            {
                'pattern': p.pattern_text,
                'type': p.pattern_type,
                'confidence': p.confidence,
                'evidence_count': len(p.evidence_doc_ids),
                'time_range': p.time_range,
                'first_noticed': p.first_noticed
            }
            for p in self.notebook.patterns.values()
        ]
    
    def _export_entities(self) -> List[Dict[str, Any]]:
        """Export top entities from notebook."""
        # Sort by document count
        sorted_entities = sorted(
            self.notebook.entities.values(),
            key=lambda e: e.document_count,
            reverse=True
        )
        
        return [
            {
                'name': e.name,
                'type': e.entity_type,
                'document_count': e.document_count,
                'contexts': e.contexts[:3]  # Sample contexts
            }
            for e in sorted_entities[:100]  # Top 100
        ]
    
    def _export_contradictions(self) -> List[Dict[str, Any]]:
        """Export contradictions from notebook."""
        return [
            {
                'claim_a': c.claim_a,
                'claim_b': c.claim_b,
                'source_a': c.source_a,
                'source_b': c.source_b,
                'context': c.context,
                'batch': c.noticed_in_batch
            }
            for c in self.notebook.contradictions
        ]


# ============================================================================
# Convenience Functions
# ============================================================================

def explore_corpus(
    strategy: str = 'balanced',
    total_budget: int = 2000,
    year_range: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Convenience function for corpus exploration.
    
    Args:
        strategy: 'temporal', 'biographical', 'genre', or 'balanced'
        total_budget: Total number of documents to read
        year_range: Optional (start_year, end_year) filter
        
    Returns:
        Exploration report
    """
    explorer = CorpusExplorer()
    return explorer.explore(
        strategy=strategy,
        total_budget=total_budget,
        year_range=year_range,
        save_notebook=True
    )
