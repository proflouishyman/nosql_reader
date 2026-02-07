# app/historian_agent/question_validator.py
# Created: 2025-12-29
# Purpose: Adversarial validation for research questions

"""
Question Validator - Adversarial checking against evidence.

Architecture:
- Layer 3: Validation logic
- Uses adversarial pattern (generator vs verifier)
- Returns QuestionValidation internally
- Applies validation to Question objects

Validation Criteria:
1. Answerability (0-25): Can this be answered with available docs?
2. Significance (0-25): Is this historically important?
3. Specificity (0-25): Is the question well-defined?
4. Evidence-based (0-25): Grounded in documented patterns?
"""

from typing import Dict, Any, Optional
import json

from rag_base import debug_print
from llm_abstraction import LLMClient
from config import APP_CONFIG
from question_models import Question, QuestionValidation, parse_validation_response
from research_notebook import ResearchNotebook


# ============================================================================
# Validation Prompts
# ============================================================================

QUESTION_VALIDATION_PROMPT = """You are a historian evaluating a research question.

QUESTION TO EVALUATE:
{question_text}

TYPE: {question_type}

CORPUS KNOWLEDGE:
{corpus_summary}

TASK: Evaluate this question on 4 criteria (0-25 points each, total 100):

**1. ANSWERABILITY (0-25): Can this be answered with available documents?**
- Are there sufficient documents on this topic? (need 10+ relevant docs)
- Is the time period covered in the corpus?
- Are key entities/events documented?
- Is scope appropriate (not too broad, not too narrow)?

Score guide:
- 25: Excellent coverage (50+ relevant docs, all entities documented)
- 20: Good coverage (20-50 docs, most entities documented)
- 15: Adequate (10-20 docs, some entities documented)
- 10: Limited (5-10 docs, few entities)
- 0-5: Insufficient (<5 docs or key entities missing)

**2. HISTORICAL SIGNIFICANCE (0-25): Does this matter historically?**
- Addresses causation, mechanisms, or change (not just description)
- Challenges or confirms important narratives
- Reveals power relations, inequality, or institutional logic
- Would scholars in the field care about this?

Score guide:
- 25: Major historical question (causation, change, inequality)
- 20: Important question (mechanisms, comparisons)
- 15: Worthwhile (specific patterns, scope conditions)
- 10: Minor (narrow detail)
- 0-5: Trivial or purely descriptive

**3. SPECIFICITY (0-25): Is the question well-defined?**
- Clear temporal scope (specific years or range)
- Specific entities named (not "workers" but "railroad firemen")
- Specific mechanisms or variables (not vague "factors")
- Avoids vague language ("what happened", "were there", "existed")

Score guide:
- 25: Highly specific (year range, named groups, clear variables)
- 20: Mostly specific (decade, occupational groups, some variables)
- 15: Somewhat specific (era, broad groups)
- 10: Vague (no time period, generic groups)
- 0-5: Extremely vague

**4. EVIDENCE-BASED (0-25): Grounded in documented patterns?**
- Based on patterns found in corpus (not speculation)
- References documented contradictions or anomalies
- Builds on entity co-occurrences or temporal patterns
- Not asking about things absent from corpus

Score guide:
- 25: Directly from strong pattern (high confidence, 30+ docs)
- 20: From medium pattern (20+ docs)
- 15: From weak pattern or entity frequency
- 10: Tangentially related to evidence
- 0-5: Not grounded in corpus at all

**CORPUS SUMMARY FOR REFERENCE:**
Documents read: {docs_read}
Time coverage: {time_coverage}
Top entities: {top_entities}
High-confidence patterns: {patterns}

Return ONLY valid JSON:
{{
  "score": 0-100 (sum of 4 components),
  "answerability": 0-25,
  "significance": 0-25,
  "specificity": 0-25,
  "evidence_based": 0-25,
  "critique": "detailed explanation of scoring",
  "suggestions": "how to improve if score < 80 (single string)"
}}

Be strict. Most questions score 60-75. Only exceptional questions score 80+.
"""

QUESTION_REFINEMENT_PROMPT = """You are a historian refining a research question.

ORIGINAL QUESTION:
{question_text}

VALIDATION CRITIQUE:
Score: {score}/100
{critique}

SUGGESTIONS:
{suggestions}

TASK: Rewrite this question to address the critique.

REQUIREMENTS:
- Keep the core intent
- Address ALL critique points
- Make more specific (add years, entities, mechanisms)
- Ensure answerability with available evidence
- Maintain historical significance

Return JSON:
{{
  "refined_question": "the improved question",
  "improvements_made": ["change1", "change2", "change3"],
  "why_better": "explanation of improvements"
}}
"""


# ============================================================================
# Question Validator
# ============================================================================

class QuestionValidator:
    """
    Adversarial validator for research questions.
    
    Layer 3: Validation logic
    Uses verifier model to critically evaluate questions
    """
    
    def __init__(self):
        """Initialize validator with LLM client."""
        self.llm = LLMClient()
    
    def validate(
        self,
        question: Question,
        notebook: ResearchNotebook
    ) -> QuestionValidation:
        """
        Validate question against corpus evidence.
        
        Args:
            question: Question to validate
            notebook: Research notebook with corpus knowledge
            
        Returns:
            QuestionValidation with scores and critique
        """
        # Build corpus summary
        summary = notebook.get_summary()
        corpus_summary = self._format_corpus_summary(summary)
        
        # Build prompt
        prompt = QUESTION_VALIDATION_PROMPT.format(
            question_text=question.question_text,
            question_type=question.question_type.value,
            corpus_summary=corpus_summary,
            docs_read=summary['documents_read'],
            time_coverage=f"{summary['time_coverage']['start']} - {summary['time_coverage']['end']}",
            top_entities=", ".join([f"{name} ({count})" for name, count in summary['top_entities'][:5]]),
            patterns="; ".join(summary['high_confidence_patterns'][:3])
        )
        
        # Call verifier model
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a strict historian evaluating research questions. Be critical."},
                {"role": "user", "content": prompt}
            ],
            profile="verifier",  # Use verifier model for critical evaluation
            temperature=0.0,  # Deterministic
            max_tokens=1000
        )
        
        if not response.success:
            debug_print(f"Validation failed: {response.error}")
            # Return default low score
            return QuestionValidation(
                total_score=50,
                answerability=10,
                significance=10,
                specificity=15,
                evidence_based=15,
                critique="Validation failed - LLM error"
            )
        
        # Parse validation response
        try:
            validation = parse_validation_response(response.content)
            return validation
        except Exception as e:
            debug_print(f"Failed to parse validation: {e}")
            return QuestionValidation(
                total_score=50,
                answerability=10,
                significance=10,
                specificity=15,
                evidence_based=15,
                critique=f"Parse error: {str(e)}"
            )
    
    def refine(
        self,
        question: Question,
        validation: QuestionValidation
    ) -> Optional[Question]:
        """
        Refine question based on validation critique.
        
        Args:
            question: Original question
            validation: Validation results with critique
            
        Returns:
            Refined Question or None if refinement fails
        """
        if validation.total_score >= 80:
            # No need to refine
            return question
        
        # Build refinement prompt
        prompt = QUESTION_REFINEMENT_PROMPT.format(
            question_text=question.question_text,
            score=validation.total_score,
            critique=validation.critique,
            suggestions="\n".join(validation.suggestions) if validation.suggestions else "Make more specific"
        )
        
        # Generate refinement
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian refining research questions."},
                {"role": "user", "content": prompt}
            ],
            profile="quality",
            temperature=0.3
        )
        
        if not response.success:
            debug_print(f"Refinement failed: {response.error}")
            return None
        
        # Parse refinement
        try:
            # Strip markdown
            text = response.content.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            
            data = json.loads(text.strip())
            
            # Create refined question
            refined = Question(
                question_text=data['refined_question'],
                question_type=question.question_type,
                why_interesting=question.why_interesting,
                time_window=question.time_window,
                entities_involved=question.entities_involved,
                evidence_doc_ids=question.evidence_doc_ids,
                pattern_source=question.pattern_source,
                generation_method=question.generation_method,
                refinement_count=question.refinement_count + 1,
                original_question=question.question_text if question.refinement_count == 0 else question.original_question
            )
            
            return refined
            
        except Exception as e:
            debug_print(f"Failed to parse refinement: {e}")
            return None
    
    def validate_and_refine(
        self,
        question: Question,
        notebook: ResearchNotebook,
        max_refinements: int = 2
    ) -> Question:
        """
        Validate question and refine if needed.
        
        Args:
            question: Question to validate
            notebook: Research notebook
            max_refinements: Maximum refinement attempts
            
        Returns:
            Validated (and possibly refined) question
        """
        current = question
        
        for attempt in range(max_refinements + 1):
            # Validate
            validation = self.validate(current, notebook)
            validation.apply_to_question(current)
            
            debug_print(
                f"Validation attempt {attempt + 1}: "
                f"Score {validation.total_score}/100 ({validation.status.value})"
            )
            
            # If excellent or good, done
            if validation.total_score >= 70:
                return current
            
            # If last attempt or score too low, return as is
            if attempt >= max_refinements or validation.total_score < 50:
                return current
            
            # Try to refine
            refined = self.refine(current, validation)
            if refined is None:
                return current  # Refinement failed, return current
            
            current = refined
        
        return current
    
    def _format_corpus_summary(self, summary: Dict[str, Any]) -> str:
        """Format corpus summary for prompt."""
        return f"""Total documents: {summary['documents_read']}
Time range: {summary['time_coverage']['start']} - {summary['time_coverage']['end']}
Entities found: {summary['total_entities']}
Patterns identified: {summary['total_patterns']}
Questions so far: {summary['total_questions']}
Contradictions: {summary['total_contradictions']}
"""


# ============================================================================
# Batch Validation
# ============================================================================

def validate_question_batch(
    questions: list,  # List[Question]
    notebook: ResearchNotebook,
    max_refinements: int = 2
) -> list:  # List[Question]
    """
    Validate a batch of questions.
    
    Args:
        questions: List of Question objects
        notebook: Research notebook
        max_refinements: Max refinement attempts per question
        
    Returns:
        List of validated Question objects
    """
    validator = QuestionValidator()
    validated = []
    
    for i, question in enumerate(questions, 1):
        debug_print(f"Validating question {i}/{len(questions)}")
        
        validated_q = validator.validate_and_refine(
            question,
            notebook,
            max_refinements=max_refinements
        )
        
        validated.append(validated_q)
    
    return validated
