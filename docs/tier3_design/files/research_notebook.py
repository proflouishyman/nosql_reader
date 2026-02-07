# app/historian_agent/research_notebook.py
# Created: 2025-12-29
# Purpose: Persistent research notebook for Tier 0 corpus exploration

"""
Research Notebook - Cumulative state for systematic corpus reading.

This is the historian's notebook that accumulates:
- Entities (people, orgs, places) found across batches
- Patterns (recurring themes, trends, mechanisms)
- Contradictions (disagreements between sources)
- Questions (emerging research inquiries)
- Temporal map (what happens when)
- Corpus statistics (coverage, gaps, biases)

Architecture:
- Each batch reading updates the notebook
- Deduplication happens automatically (entities, patterns)
- Confidence scores track pattern strength
- Temporal coverage tracked for gap analysis
"""

import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Entity:
    """Tracked entity (person, organization, place)."""
    name: str
    entity_type: str  # person, organization, place
    first_seen: str  # document_id
    last_seen: str  # document_id
    document_count: int = 1
    contexts: List[str] = field(default_factory=list)  # Sample contexts
    
    def merge(self, other: 'Entity') -> None:
        """Merge another entity occurrence into this one."""
        self.document_count += other.document_count
        self.contexts.extend(other.contexts[:3])  # Keep sample contexts
        self.contexts = self.contexts[:10]  # Cap at 10


@dataclass
class Pattern:
    """Identified pattern across documents."""
    pattern_text: str
    pattern_type: str  # injury_trend, wage_pattern, institutional_practice, etc.
    evidence_doc_ids: List[str] = field(default_factory=list)
    confidence: str = "low"  # low, medium, high
    time_range: Optional[str] = None
    first_noticed: str = ""  # batch label
    
    def add_evidence(self, doc_ids: List[str]) -> None:
        """Add supporting evidence."""
        self.evidence_doc_ids.extend(doc_ids)
        self.evidence_doc_ids = list(set(self.evidence_doc_ids))  # Dedupe
        
        # Update confidence based on evidence count
        if len(self.evidence_doc_ids) >= 20:
            self.confidence = "high"
        elif len(self.evidence_doc_ids) >= 10:
            self.confidence = "medium"


@dataclass
class Contradiction:
    """Noticed contradiction between sources."""
    claim_a: str
    claim_b: str
    source_a: str
    source_b: str
    context: str
    noticed_in_batch: str


@dataclass
class ResearchQuestion:
    """Emerging research question."""
    question: str
    why_interesting: str
    evidence_needed: str
    related_entities: List[str] = field(default_factory=list)
    time_window: Optional[str] = None
    noticed_in_batch: str = ""


# ============================================================================
# Research Notebook
# ============================================================================

class ResearchNotebook:
    """
    Persistent state that accumulates across batches.
    
    This is the core data structure that makes multi-pass reading work.
    Each batch reading updates this notebook, and the notebook provides
    context for the next batch.
    """
    
    def __init__(self):
        """Initialize empty research notebook."""
        # Entity registry - keyed by normalized name
        self.entities: Dict[str, Entity] = {}
        
        # Pattern registry - keyed by pattern text
        self.patterns: Dict[str, Pattern] = {}
        
        # Lists of findings
        self.contradictions: List[Contradiction] = []
        self.questions: List[ResearchQuestion] = []
        
        # Temporal coverage map
        self.temporal_map: Dict[str, List[str]] = defaultdict(list)  # year -> events
        
        # Corpus statistics
        self.corpus_map = {
            'total_documents_read': 0,
            'by_year': defaultdict(int),
            'by_collection': defaultdict(int),
            'by_document_type': defaultdict(int),
            'by_person': defaultdict(int),
            'batches_processed': 0,
            'time_coverage': {'start': None, 'end': None},
            'gaps_identified': []
        }
        
        # Processing metadata
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.batches_log: List[Dict] = []
    
    def integrate_batch_findings(self, findings: Dict[str, Any], batch_label: str) -> None:
        """
        Merge findings from a batch into cumulative knowledge.
        
        Args:
            findings: Dict with keys: entities, patterns, contradictions, questions, temporal_events, stats
            batch_label: Human-readable label for this batch (e.g., "Year 1923")
        """
        # Update entities (with deduplication)
        for entity_dict in findings.get('entities', []):
            self._add_entity(entity_dict)
        
        # Update patterns (with evidence accumulation)
        for pattern_dict in findings.get('patterns', []):
            self._add_pattern(pattern_dict, batch_label)
        
        # Add contradictions
        for contra_dict in findings.get('contradictions', []):
            contra_dict['noticed_in_batch'] = batch_label
            self.contradictions.append(Contradiction(**contra_dict))
        
        # Add questions
        for question_dict in findings.get('questions', []):
            question_dict['noticed_in_batch'] = batch_label
            self.questions.append(ResearchQuestion(**question_dict))
        
        # Update temporal map
        for year, events in findings.get('temporal_events', {}).items():
            self.temporal_map[year].extend(events)
        
        # Update corpus statistics
        self._update_corpus_map(findings.get('stats', {}))
        
        # Log batch
        self.batches_log.append({
            'batch_label': batch_label,
            'processed_at': datetime.now().isoformat(),
            'entities_added': len(findings.get('entities', [])),
            'patterns_added': len(findings.get('patterns', [])),
            'questions_added': len(findings.get('questions', []))
        })
        
        self.corpus_map['batches_processed'] += 1
        self.last_updated = datetime.now().isoformat()
    
    def _add_entity(self, entity_dict: Dict) -> None:
        """Add entity with deduplication."""
        name = entity_dict['name']
        normalized_name = name.lower().strip()
        
        if normalized_name in self.entities:
            # Merge with existing
            existing = self.entities[normalized_name]
            new_entity = Entity(**entity_dict)
            existing.merge(new_entity)
        else:
            # Add new
            self.entities[normalized_name] = Entity(**entity_dict)
    
    def _add_pattern(self, pattern_dict: Dict, batch_label: str) -> None:
        """Add pattern with evidence accumulation."""
        text = pattern_dict['pattern']
        
        if text in self.patterns:
            # Add evidence to existing pattern
            existing = self.patterns[text]
            existing.add_evidence(pattern_dict.get('evidence', []))
        else:
            # Add new pattern
            self.patterns[text] = Pattern(
                pattern_text=text,
                pattern_type=pattern_dict.get('type', 'unknown'),
                evidence_doc_ids=pattern_dict.get('evidence', []),
                confidence=pattern_dict.get('confidence', 'low'),
                time_range=pattern_dict.get('time_range'),
                first_noticed=batch_label
            )
    
    def _update_corpus_map(self, stats: Dict) -> None:
        """Update corpus statistics."""
        self.corpus_map['total_documents_read'] += stats.get('docs_in_batch', 0)
        
        for year, count in stats.get('by_year', {}).items():
            self.corpus_map['by_year'][year] += count
        
        for collection, count in stats.get('by_collection', {}).items():
            self.corpus_map['by_collection'][collection] += count
        
        for doc_type, count in stats.get('by_document_type', {}).items():
            self.corpus_map['by_document_type'][doc_type] += count
        
        for person, count in stats.get('by_person', {}).items():
            self.corpus_map['by_person'][person] += count
        
        # Update time coverage
        if stats.get('earliest_year'):
            if self.corpus_map['time_coverage']['start'] is None:
                self.corpus_map['time_coverage']['start'] = stats['earliest_year']
            else:
                self.corpus_map['time_coverage']['start'] = min(
                    self.corpus_map['time_coverage']['start'],
                    stats['earliest_year']
                )
        
        if stats.get('latest_year'):
            if self.corpus_map['time_coverage']['end'] is None:
                self.corpus_map['time_coverage']['end'] = stats['latest_year']
            else:
                self.corpus_map['time_coverage']['end'] = max(
                    self.corpus_map['time_coverage']['end'],
                    stats['latest_year']
                )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of notebook state for LLM context."""
        return {
            'total_entities': len(self.entities),
            'top_entities': sorted(
                [(e.name, e.document_count) for e in self.entities.values()],
                key=lambda x: x[1],
                reverse=True
            )[:20],
            'total_patterns': len(self.patterns),
            'high_confidence_patterns': [
                p.pattern_text for p in self.patterns.values() if p.confidence == 'high'
            ],
            'total_questions': len(self.questions),
            'total_contradictions': len(self.contradictions),
            'documents_read': self.corpus_map['total_documents_read'],
            'time_coverage': self.corpus_map['time_coverage'],
            'batches_processed': self.corpus_map['batches_processed']
        }
    
    def format_for_llm_context(self) -> str:
        """Format notebook summary as text for LLM context."""
        summary = self.get_summary()
        
        text = f"""PRIOR KNOWLEDGE (from {summary['documents_read']} documents read):

Entities Found ({summary['total_entities']} total):
"""
        for name, count in summary['top_entities'][:10]:
            text += f"  - {name} ({count} documents)\n"
        
        text += f"\nHigh-Confidence Patterns ({len(summary['high_confidence_patterns'])}):\n"
        for pattern in summary['high_confidence_patterns'][:5]:
            text += f"  - {pattern}\n"
        
        text += f"\nResearch Questions ({summary['total_questions']}):\n"
        for q in self.questions[-5:]:  # Last 5 questions
            text += f"  - {q.question}\n"
        
        text += f"\nTime Coverage: {summary['time_coverage']['start']} - {summary['time_coverage']['end']}\n"
        text += f"Batches Processed: {summary['batches_processed']}\n"
        
        return text
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'entities': {k: asdict(v) for k, v in self.entities.items()},
            'patterns': {k: asdict(v) for k, v in self.patterns.items()},
            'contradictions': [asdict(c) for c in self.contradictions],
            'questions': [asdict(q) for q in self.questions],
            'temporal_map': dict(self.temporal_map),
            'corpus_map': dict(self.corpus_map),
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'batches_log': self.batches_log
        }
    
    def save(self, filepath: str) -> None:
        """Save notebook to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ResearchNotebook':
        """Load notebook from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        notebook = cls()
        
        # Restore entities
        notebook.entities = {
            k: Entity(**v) for k, v in data['entities'].items()
        }
        
        # Restore patterns
        notebook.patterns = {
            k: Pattern(**v) for k, v in data['patterns'].items()
        }
        
        # Restore other fields
        notebook.contradictions = [Contradiction(**c) for c in data['contradictions']]
        notebook.questions = [ResearchQuestion(**q) for q in data['questions']]
        notebook.temporal_map = defaultdict(list, data['temporal_map'])
        notebook.corpus_map = data['corpus_map']
        notebook.created_at = data['created_at']
        notebook.last_updated = data['last_updated']
        notebook.batches_log = data['batches_log']
        
        return notebook
