# app/historian_agent/question_pipeline.py
# Created: 2025-12-29
# Purpose: Question generation pipeline with typology and validation

"""
Question Generation Pipeline - Orchestrates typed generation and validation.

Architecture:
- Layer 4: Orchestration
- Uses Layer 2 (typology) and Layer 3 (validation)
- Returns QuestionBatch internally, List[Dict] externally

Pipeline:
1. Generate candidates by type (30+)
2. Validate each question
3. Refine low-scoring questions
4. Filter by minimum score
5. Deduplicate
6. Rank by validation score
7. Return top N
"""

from typing import List, Dict, Any
from collections import defaultdict

from rag_base import debug_print
from question_models import Question, QuestionBatch, QuestionType, ValidationStatus
from question_typology import TypedQuestionGenerator, generate_typed_questions
from question_validator import QuestionValidator, validate_question_batch
from research_notebook import ResearchNotebook


# ============================================================================
# Pipeline Configuration
# ============================================================================

class PipelineConfig:
    """Configuration for question generation pipeline."""
    
    # Generation
    QUESTIONS_PER_TYPE = 5  # How many to generate per type
    MIN_CANDIDATES = 15     # Minimum total candidates
    MAX_CANDIDATES = 40     # Maximum total candidates
    
    # Validation
    MIN_SCORE_ACCEPT = 60   # Minimum score to accept
    MIN_SCORE_REFINE = 50   # Minimum score to attempt refinement
    MAX_REFINEMENTS = 2     # Max refinement attempts
    
    # Output
    TARGET_QUESTIONS = 12   # Target number of final questions
    MIN_QUESTIONS = 8       # Minimum to return
    
    # Diversity
    ENFORCE_TYPE_DIVERSITY = True  # Ensure multiple types represented
    MIN_TYPES_REPRESENTED = 3      # Minimum number of types in output


# ============================================================================
# Question Pipeline
# ============================================================================

class QuestionGenerationPipeline:
    """
    Orchestrates typed generation and adversarial validation.
    
    Layer 4: Pipeline orchestration
    Internal: Works with Question objects
    External: Returns dicts via to_dict()
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self.generator = TypedQuestionGenerator()
        self.validator = QuestionValidator()
    
    def generate(self, notebook: ResearchNotebook) -> QuestionBatch:
        """
        Run full pipeline: generate, validate, filter, rank.
        
        Args:
            notebook: Research notebook with corpus knowledge
            
        Returns:
            QuestionBatch with validated questions
        """
        debug_print("Starting question generation pipeline")
        
        # Stage 1: Generate candidates by type
        candidates = self._generate_candidates(notebook)
        debug_print(f"Stage 1: Generated {len(candidates)} candidates")
        
        if len(candidates) < self.config.MIN_CANDIDATES:
            debug_print(f"Warning: Only {len(candidates)} candidates (target: {self.config.MIN_CANDIDATES})")
        
        # Stage 2: Validate all candidates
        validated = self._validate_candidates(candidates, notebook)
        debug_print(f"Stage 2: Validated {len(validated)} questions")
        
        # Stage 3: Filter by minimum score
        filtered = self._filter_by_score(validated)
        debug_print(f"Stage 3: Filtered to {len(filtered)} questions (score >= {self.config.MIN_SCORE_ACCEPT})")
        
        # Stage 4: Deduplicate
        deduplicated = self._deduplicate(filtered)
        debug_print(f"Stage 4: Deduplicated to {len(deduplicated)} questions")
        
        # Stage 5: Ensure type diversity
        if self.config.ENFORCE_TYPE_DIVERSITY:
            diverse = self._ensure_diversity(deduplicated)
            debug_print(f"Stage 5: Ensured diversity, {len(diverse)} questions")
        else:
            diverse = deduplicated
        
        # Stage 6: Rank and select top N
        final = self._rank_and_select(diverse)
        debug_print(f"Stage 6: Selected top {len(final)} questions")
        
        # Build batch
        batch = QuestionBatch(
            questions=final,
            generation_strategy="typed_validated",
            total_candidates=len(candidates),
            total_validated=len(validated),
            total_accepted=len(filtered)
        )
        
        return batch
    
    def _generate_candidates(self, notebook: ResearchNotebook) -> List[Question]:
        """
        Generate candidate questions by type.
        
        Uses typed generation from Layer 2.
        """
        candidates = []
        
        # Prepare context
        context = {
            'patterns': list(notebook.patterns.values()),
            'contradictions': notebook.contradictions,
            'temporal_map': dict(notebook.temporal_map),
            'entities': list(notebook.entities.values())
        }
        
        # Generate by type
        types_to_generate = [
            QuestionType.CAUSAL,
            QuestionType.COMPARATIVE,
            QuestionType.CHANGE_OVER_TIME
        ]
        
        for qtype in types_to_generate:
            typed_questions = self.generator.generate_by_type(qtype, context)
            
            # Limit per type
            typed_questions = typed_questions[:self.config.QUESTIONS_PER_TYPE]
            
            candidates.extend(typed_questions)
            debug_print(f"  Generated {len(typed_questions)} {qtype.value} questions")
        
        # Cap total candidates
        if len(candidates) > self.config.MAX_CANDIDATES:
            candidates = candidates[:self.config.MAX_CANDIDATES]
        
        return candidates
    
    def _validate_candidates(
        self,
        candidates: List[Question],
        notebook: ResearchNotebook
    ) -> List[Question]:
        """
        Validate all candidates with refinement.
        
        Uses adversarial validation from Layer 3.
        """
        validated = []
        
        for i, question in enumerate(candidates, 1):
            debug_print(f"  Validating {i}/{len(candidates)}: {question.question_text[:60]}...")
            
            # Validate and refine
            validated_q = self.validator.validate_and_refine(
                question,
                notebook,
                max_refinements=self.config.MAX_REFINEMENTS
            )
            
            validated.append(validated_q)
        
        return validated
    
    def _filter_by_score(self, questions: List[Question]) -> List[Question]:
        """Filter questions by minimum validation score."""
        return [
            q for q in questions
            if q.validation_score is not None
            and q.validation_score >= self.config.MIN_SCORE_ACCEPT
        ]
    
    def _deduplicate(self, questions: List[Question]) -> List[Question]:
        """
        Remove duplicate or very similar questions.
        
        Simple approach: exact text match (could use embeddings for semantic deduplication)
        """
        seen = set()
        unique = []
        
        for q in questions:
            # Normalize for comparison
            normalized = q.question_text.lower().strip()
            
            if normalized not in seen:
                seen.add(normalized)
                unique.append(q)
        
        return unique
    
    def _ensure_diversity(self, questions: List[Question]) -> List[Question]:
        """
        Ensure multiple question types are represented.
        
        If one type dominates, balance it.
        """
        # Count by type
        by_type = defaultdict(list)
        for q in questions:
            by_type[q.question_type].append(q)
        
        # If diversity is already good, return as is
        if len(by_type) >= self.config.MIN_TYPES_REPRESENTED:
            return questions
        
        # Otherwise, try to balance
        debug_print(f"  Warning: Only {len(by_type)} types represented (target: {self.config.MIN_TYPES_REPRESENTED})")
        
        # At minimum, include top question from each type
        balanced = []
        for qtype, qs in by_type.items():
            # Sort by score
            sorted_qs = sorted(qs, key=lambda q: q.validation_score or 0, reverse=True)
            balanced.extend(sorted_qs[:2])  # Top 2 from each type
        
        return balanced
    
    def _rank_and_select(self, questions: List[Question]) -> List[Question]:
        """
        Rank by validation score and select top N.
        
        Args:
            questions: Filtered questions
            
        Returns:
            Top N questions
        """
        # Sort by validation score (descending)
        ranked = sorted(
            questions,
            key=lambda q: (q.validation_score or 0, len(q.evidence_doc_ids)),
            reverse=True
        )
        
        # Select target number
        n = self.config.TARGET_QUESTIONS
        
        # But ensure minimum
        if len(ranked) < self.config.MIN_QUESTIONS:
            debug_print(f"Warning: Only {len(ranked)} questions available (target: {n})")
            return ranked
        
        return ranked[:n]
    
    def generate_to_dict(self, notebook: ResearchNotebook) -> Dict[str, Any]:
        """
        Run pipeline and return dict for external API.
        
        Args:
            notebook: Research notebook
            
        Returns:
            Dict with questions and metadata
        """
        batch = self.generate(notebook)
        return batch.to_dict()
    
    def generate_to_list(self, notebook: ResearchNotebook) -> List[Dict[str, Any]]:
        """
        Run pipeline and return list of dicts (for backward compatibility).
        
        Args:
            notebook: Research notebook
            
        Returns:
            List of question dicts
        """
        batch = self.generate(notebook)
        return batch.to_list()


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_validated_questions(
    notebook: ResearchNotebook,
    target_count: int = 12,
    min_score: int = 60
) -> List[Dict[str, Any]]:
    """
    Generate and validate research questions (simple interface).
    
    Args:
        notebook: Research notebook with corpus knowledge
        target_count: Target number of questions
        min_score: Minimum validation score
        
    Returns:
        List of question dicts
    """
    # Configure pipeline
    config = PipelineConfig()
    config.TARGET_QUESTIONS = target_count
    config.MIN_SCORE_ACCEPT = min_score
    
    # Run pipeline
    pipeline = QuestionGenerationPipeline(config)
    return pipeline.generate_to_list(notebook)


def generate_questions_report(notebook: ResearchNotebook) -> Dict[str, Any]:
    """
    Generate full question report with metadata.
    
    Args:
        notebook: Research notebook
        
    Returns:
        Dict with questions, statistics, and quality metrics
    """
    pipeline = QuestionGenerationPipeline()
    batch = pipeline.generate(notebook)
    
    # Calculate statistics
    avg_score = sum(q.validation_score or 0 for q in batch.questions) / len(batch.questions) if batch.questions else 0
    
    type_distribution = defaultdict(int)
    for q in batch.questions:
        type_distribution[q.question_type.value] += 1
    
    status_distribution = defaultdict(int)
    for q in batch.questions:
        if q.validation_status:
            status_distribution[q.validation_status.value] += 1
    
    return {
        'questions': batch.to_list(),
        'metadata': batch.to_dict()['metadata'],
        'quality_metrics': {
            'average_validation_score': round(avg_score, 1),
            'type_distribution': dict(type_distribution),
            'status_distribution': dict(status_distribution),
            'questions_refined': sum(1 for q in batch.questions if q.refinement_count > 0),
            'average_evidence_count': round(
                sum(len(q.evidence_doc_ids) for q in batch.questions) / len(batch.questions)
                if batch.questions else 0,
                1
            )
        }
    }
