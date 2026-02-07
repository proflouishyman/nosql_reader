# app/historian_agent/question_models.py
# Created: 2025-12-29
# Purpose: Data models for research question generation and validation

"""
Question Models - Type-safe internal representations with dict serialization.

Architecture:
- Internal: Dataclasses for type safety and validation
- External: Dict conversion for JSON/API responses

Pattern: "Lists internally, dicts externally"
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class QuestionType(Enum):
    """Historical question typology."""
    CAUSAL = "causal"                      # Why did X happen?
    COMPARATIVE = "comparative"            # How did X differ from Y?
    CHANGE_OVER_TIME = "change"           # How did X evolve?
    DISTRIBUTIONAL = "distributional"      # Who benefited/suffered?
    INSTITUTIONAL = "institutional"        # What rules governed X?
    SCOPE_CONDITIONS = "scope"            # Where/when did X apply?


class ValidationStatus(Enum):
    """Question validation status."""
    EXCELLENT = "excellent"      # Score >= 80
    GOOD = "good"               # Score >= 70
    ACCEPTABLE = "acceptable"    # Score >= 60
    NEEDS_REFINEMENT = "needs_refinement"  # Score >= 50
    REJECTED = "rejected"        # Score < 50


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Question:
    """
    Research question (internal representation).
    
    Internal use: Type-safe, validated
    External API: Convert to dict via to_dict()
    """
    question_text: str
    question_type: QuestionType
    
    # Historical context
    why_interesting: str
    time_window: Optional[str] = None
    entities_involved: List[str] = field(default_factory=list)
    
    # Evidence grounding
    evidence_doc_ids: List[str] = field(default_factory=list)
    pattern_source: Optional[str] = None
    contradiction_source: Optional[str] = None
    
    # Generation metadata
    generation_method: str = "llm_generated"
    
    # Validation results (populated after validation)
    validation_score: Optional[int] = None
    validation_status: Optional[ValidationStatus] = None
    answerability_score: Optional[int] = None
    significance_score: Optional[int] = None
    specificity_score: Optional[int] = None
    evidence_based_score: Optional[int] = None
    critique: Optional[str] = None
    
    # Refinement
    refinement_count: int = 0
    original_question: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for external API/JSON.
        
        Returns:
            Dict suitable for JSON serialization
        """
        return {
            'question': self.question_text,
            'type': self.question_type.value,
            'why_interesting': self.why_interesting,
            'time_window': self.time_window,
            'entities_involved': self.entities_involved,
            'evidence_count': len(self.evidence_doc_ids),
            'evidence_sample': self.evidence_doc_ids[:5],  # Sample for API
            'pattern_source': self.pattern_source,
            'generation_method': self.generation_method,
            'validation': {
                'score': self.validation_score,
                'status': self.validation_status.value if self.validation_status else None,
                'answerability': self.answerability_score,
                'significance': self.significance_score,
                'specificity': self.specificity_score,
                'evidence_based': self.evidence_based_score,
                'critique': self.critique
            } if self.validation_score is not None else None,
            'refinement_count': self.refinement_count,
            'original_question': self.original_question
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        """
        Create Question from dictionary.
        
        Args:
            data: Dictionary from LLM response or API
            
        Returns:
            Question instance
        """
        # Parse question type
        qtype = data.get('type', 'causal')
        if isinstance(qtype, str):
            qtype = QuestionType(qtype)
        
        return cls(
            question_text=data['question'],
            question_type=qtype,
            why_interesting=data.get('why_interesting', ''),
            time_window=data.get('time_window'),
            entities_involved=data.get('entities_involved', []),
            evidence_doc_ids=data.get('evidence_doc_ids', []),
            pattern_source=data.get('pattern_source'),
            generation_method=data.get('generation_method', 'llm_generated')
        )


@dataclass
class QuestionValidation:
    """
    Validation results for a research question.
    
    Internal: Type-safe validation data
    External: Included in Question.to_dict()
    """
    total_score: int  # 0-100
    
    # Component scores (0-25 each)
    answerability: int
    significance: int
    specificity: int
    evidence_based: int
    
    # Qualitative feedback
    critique: str
    suggestions: List[str] = field(default_factory=list)
    
    # Status determination
    status: ValidationStatus = field(init=False)
    
    def __post_init__(self):
        """Determine status from score."""
        if self.total_score >= 80:
            self.status = ValidationStatus.EXCELLENT
        elif self.total_score >= 70:
            self.status = ValidationStatus.GOOD
        elif self.total_score >= 60:
            self.status = ValidationStatus.ACCEPTABLE
        elif self.total_score >= 50:
            self.status = ValidationStatus.NEEDS_REFINEMENT
        else:
            self.status = ValidationStatus.REJECTED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API."""
        return {
            'score': self.total_score,
            'status': self.status.value,
            'components': {
                'answerability': self.answerability,
                'significance': self.significance,
                'specificity': self.specificity,
                'evidence_based': self.evidence_based
            },
            'critique': self.critique,
            'suggestions': self.suggestions
        }
    
    def apply_to_question(self, question: Question) -> None:
        """
        Apply validation results to a Question instance.
        
        Mutates question in place.
        """
        question.validation_score = self.total_score
        question.validation_status = self.status
        question.answerability_score = self.answerability
        question.significance_score = self.significance
        question.specificity_score = self.specificity
        question.evidence_based_score = self.evidence_based
        question.critique = self.critique


@dataclass
class QuestionBatch:
    """
    Batch of questions (internal collection).
    
    Internal: List of Question objects
    External: List of dicts via to_dict()
    """
    questions: List[Question] = field(default_factory=list)
    
    # Batch metadata
    generation_strategy: str = "mixed"
    total_candidates: int = 0
    total_validated: int = 0
    total_accepted: int = 0
    
    def add(self, question: Question) -> None:
        """Add question to batch."""
        self.questions.append(question)
    
    def filter_by_status(self, status: ValidationStatus) -> List[Question]:
        """Filter questions by validation status."""
        return [q for q in self.questions if q.validation_status == status]
    
    def filter_by_type(self, qtype: QuestionType) -> List[Question]:
        """Filter questions by type."""
        return [q for q in self.questions if q.question_type == qtype]
    
    def get_top_n(self, n: int) -> List[Question]:
        """
        Get top N questions by validation score.
        
        Args:
            n: Number of questions to return
            
        Returns:
            List of top N questions (sorted)
        """
        sorted_questions = sorted(
            [q for q in self.questions if q.validation_score is not None],
            key=lambda q: q.validation_score,
            reverse=True
        )
        return sorted_questions[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for external API."""
        return {
            'questions': [q.to_dict() for q in self.questions],
            'metadata': {
                'generation_strategy': self.generation_strategy,
                'total_candidates': self.total_candidates,
                'total_validated': self.total_validated,
                'total_accepted': self.total_accepted,
                'final_count': len(self.questions)
            }
        }
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        Convert to list of dicts (for backward compatibility).
        
        Returns:
            List of question dicts
        """
        return [q.to_dict() for q in self.questions]


# ============================================================================
# Helper Functions
# ============================================================================

def parse_llm_question_response(response_text: str) -> List[Question]:
    """
    Parse LLM response into Question objects.
    
    Handles JSON arrays or single JSON objects.
    
    Args:
        response_text: LLM response (JSON string)
        
    Returns:
        List of Question objects
    """
    import json
    
    # Strip markdown code blocks
    text = response_text.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()
    
    # Parse JSON
    data = json.loads(text)
    
    # Handle array or single object
    if isinstance(data, list):
        questions_data = data
    elif isinstance(data, dict):
        # Check for questions array in response
        if 'questions' in data:
            questions_data = data['questions']
        else:
            questions_data = [data]
    else:
        raise ValueError(f"Unexpected response format: {type(data)}")
    
    # Convert to Question objects
    questions = []
    for q_data in questions_data:
        try:
            question = Question.from_dict(q_data)
            questions.append(question)
        except Exception as e:
            # Log but continue
            import sys
            print(f"Warning: Failed to parse question: {e}", file=sys.stderr)
            continue
    
    return questions


def parse_validation_response(response_text: str) -> QuestionValidation:
    """
    Parse LLM validation response into QuestionValidation.
    
    Args:
        response_text: LLM validation response (JSON string)
        
    Returns:
        QuestionValidation object
    """
    import json
    
    # Strip markdown
    text = response_text.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()
    
    # Parse JSON
    data = json.loads(text)
    
    return QuestionValidation(
        total_score=data['score'],
        answerability=data.get('answerability', 0),
        significance=data.get('significance', 0),
        specificity=data.get('specificity', 0),
        evidence_based=data.get('evidence_based', 0),
        critique=data.get('critique', ''),
        suggestions=data.get('suggestions', []) if isinstance(data.get('suggestions'), list) 
                   else [data.get('suggestions', '')]
    )
