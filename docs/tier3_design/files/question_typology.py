# app/historian_agent/question_typology.py
# Created: 2025-12-29
# Purpose: Historical question typology with type-specific generation

"""
Question Typology - Generate questions by historical type.

Architecture:
- Layer 2: Type-specific generation logic
- Uses QuestionType enum for classification
- Returns List[Question] internally
- Each type has specialized prompts and logic

Question Types (from historiography):
1. Causal - Why/how did X happen? (mechanisms)
2. Comparative - How did X differ from Y? (across time/space/groups)
3. Change-over-time - How did X evolve? (transformation)
4. Distributional - Who benefited/suffered? (differential impacts)
5. Institutional - What rules governed X? (organizational logic)
6. Scope - Where/when did X apply? (boundaries)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from rag_base import debug_print
from llm_abstraction import LLMClient
from config import APP_CONFIG
from question_models import Question, QuestionType, parse_llm_question_response
from research_notebook import ResearchNotebook, Pattern, Contradiction, Entity


# ============================================================================
# Type-Specific Prompts
# ============================================================================

CAUSAL_PROMPT = """You are a historian generating CAUSAL research questions.

PATTERN IDENTIFIED:
{pattern_text}

EVIDENCE:
- {evidence_count} documents support this pattern
- Time range: {time_range}
- Entities involved: {entities}

TASK: Generate ONE specific causal question that asks WHY or HOW this pattern occurred.

REQUIREMENTS:
✓ Ask about MECHANISMS, not just description
✓ Be SPECIFIC: include time period, entities, concrete events
✓ Avoid vague "what happened" questions
✓ Focus on causation, not correlation

GOOD EXAMPLES:
- "Why did injury rates among railroad firemen spike during the 1923-1925 labor disputes?"
- "How did changes in locomotive technology between 1900-1920 affect firemen burn injury rates?"

BAD EXAMPLES:
- "What happened in 1923?" (descriptive, not causal)
- "Were there injuries?" (yes/no, not mechanistic)

Return JSON:
{{
  "question": "specific causal question",
  "type": "causal",
  "why_interesting": "why this mechanism matters historically",
  "entities_involved": ["entity1", "entity2"],
  "time_window": "YYYY-YYYY or YYYY",
  "mechanism_hypotheses": ["hypothesis1", "hypothesis2"]
}}
"""

COMPARATIVE_PROMPT = """You are a historian generating COMPARATIVE research questions.

{comparison_basis}

TASK: Generate ONE specific comparative question.

REQUIREMENTS:
✓ Compare across TIME, SPACE, or GROUPS
✓ Identify DIFFERENCES or SIMILARITIES
✓ Be SPECIFIC about what's being compared
✓ Specify the dimension of comparison

GOOD EXAMPLES:
- "How did disability claim approval rates differ between Division A and Division C during 1920-1930?"
- "How did wage records for African American vs white workers differ in the 1920s?"
- "How did injury patterns change before vs after the 1918 safety regulations?"

BAD EXAMPLES:
- "What were wages?" (not comparative)
- "Were there differences?" (too vague)

Return JSON:
{{
  "question": "specific comparative question",
  "type": "comparative",
  "why_interesting": "why this comparison matters",
  "entities_involved": ["group1", "group2"],
  "time_window": "YYYY-YYYY",
  "comparison_dimensions": ["dimension1", "dimension2"]
}}
"""

CHANGE_OVER_TIME_PROMPT = """You are a historian generating CHANGE-OVER-TIME questions.

TEMPORAL PATTERN:
{temporal_evidence}

TASK: Generate ONE question about TRANSFORMATION or EVOLUTION.

REQUIREMENTS:
✓ Focus on CHANGE, not static description
✓ Specify START and END points
✓ Ask about TRAJECTORY (gradual, sudden, cyclical?)
✓ Consider TURNING POINTS

GOOD EXAMPLES:
- "How did medical terminology for workplace injuries evolve between 1900-1940?"
- "When did disability claim approval rates shift from 85% to 45% in Division C?"
- "How did the composition of railroad workforce change after WWI?"

BAD EXAMPLES:
- "What was the workforce in 1920?" (static)
- "Did things change?" (too vague)

Return JSON:
{{
  "question": "specific change-over-time question",
  "type": "change",
  "why_interesting": "what this change reveals historically",
  "time_window": "YYYY-YYYY (must be range)",
  "entities_involved": ["entity1"],
  "change_indicators": ["indicator1", "indicator2"]
}}
"""

DISTRIBUTIONAL_PROMPT = """You are a historian generating DISTRIBUTIONAL questions.

EVIDENCE OF DIFFERENTIAL IMPACTS:
{distribution_evidence}

TASK: Generate ONE question about WHO benefited or suffered.

REQUIREMENTS:
✓ Identify GROUPS (by class, race, gender, occupation, etc.)
✓ Ask about DIFFERENTIAL impacts
✓ Consider power and inequality
✓ Be specific about the distribution

GOOD EXAMPLES:
- "Were African American workers disproportionately assigned to hazardous roles in the 1920s?"
- "How did disability benefits vary by wage level and occupation?"
- "Which divisions bore the brunt of injury rates during 1923-1925?"

BAD EXAMPLES:
- "What jobs existed?" (not distributional)
- "Were there workers?" (not about differential impact)

Return JSON:
{{
  "question": "specific distributional question",
  "type": "distributional",
  "why_interesting": "what this reveals about inequality/power",
  "entities_involved": ["group1", "group2"],
  "time_window": "YYYY-YYYY",
  "distribution_dimensions": ["race", "class", "occupation", etc.]
}}
"""

INSTITUTIONAL_PROMPT = """You are a historian generating INSTITUTIONAL questions.

ORGANIZATIONAL CONTEXT:
{institutional_evidence}

TASK: Generate ONE question about RULES, PRACTICES, or PROCEDURES.

REQUIREMENTS:
✓ Ask about FORMAL or INFORMAL rules
✓ Focus on organizational LOGIC
✓ Consider criteria, standards, procedures
✓ Ask HOW institutions functioned

GOOD EXAMPLES:
- "What criteria determined whether disability claims were approved in the 1920s?"
- "How did the Relief Department decide compensation amounts for injured workers?"
- "What procedures governed the reporting of workplace injuries?"

BAD EXAMPLES:
- "How many claims were filed?" (quantitative, not institutional)
- "What was the policy?" (too vague)

Return JSON:
{{
  "question": "specific institutional question",
  "type": "institutional",
  "why_interesting": "what this reveals about organizational logic",
  "entities_involved": ["institution1"],
  "time_window": "YYYY-YYYY",
  "institutional_aspects": ["rules", "criteria", "procedures"]
}}
"""

SCOPE_CONDITIONS_PROMPT = """You are a historian generating SCOPE CONDITION questions.

PATTERN WITH VARIATION:
{scope_evidence}

TASK: Generate ONE question about WHERE or WHEN a pattern APPLIED or DID NOT APPLY.

REQUIREMENTS:
✓ Identify BOUNDARIES of a pattern
✓ Ask about VARIATION across contexts
✓ Consider geographic, temporal, or social scope
✓ Challenge universality

GOOD EXAMPLES:
- "Did wage underreporting occur consistently across all divisions or only in Division C?"
- "Was the spike in injury rates during 1923-1925 limited to certain departments?"
- "Did disability claim approval rates vary by supervisor or medical examiner?"

BAD EXAMPLES:
- "Did X happen?" (yes/no, not about scope)
- "Where did workers work?" (descriptive)

Return JSON:
{{
  "question": "specific scope conditions question",
  "type": "scope",
  "why_interesting": "what variation reveals about the pattern",
  "entities_involved": ["entity1"],
  "time_window": "YYYY-YYYY",
  "scope_dimensions": ["geographic", "temporal", "organizational"]
}}
"""


# ============================================================================
# Type-Specific Generators
# ============================================================================

class TypedQuestionGenerator:
    """
    Generates questions by historical type.
    
    Layer 2: Type-specific generation logic
    Returns: List[Question] for internal use
    """
    
    def __init__(self):
        """Initialize generator with LLM client."""
        self.llm = LLMClient()
    
    def from_pattern_causal(self, pattern: Pattern) -> Optional[Question]:
        """
        Generate causal question from a pattern.
        
        Args:
            pattern: Pattern to convert to question
            
        Returns:
            Question or None if generation fails
        """
        # Build prompt
        entities = ", ".join(pattern.evidence_doc_ids[:5]) if pattern.evidence_doc_ids else "Unknown"
        
        prompt = CAUSAL_PROMPT.format(
            pattern_text=pattern.pattern_text,
            evidence_count=len(pattern.evidence_doc_ids),
            time_range=pattern.time_range or "Unknown",
            entities=entities
        )
        
        # Generate
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian generating causal research questions."},
                {"role": "user", "content": prompt}
            ],
            profile="quality",
            temperature=0.4
        )
        
        if not response.success:
            debug_print(f"Failed to generate causal question: {response.error}")
            return None
        
        # Parse
        try:
            questions = parse_llm_question_response(response.content)
            if questions:
                q = questions[0]
                q.pattern_source = pattern.pattern_text
                q.evidence_doc_ids = pattern.evidence_doc_ids
                q.generation_method = "pattern_causal"
                return q
        except Exception as e:
            debug_print(f"Failed to parse causal question: {e}")
            return None
    
    def from_contradiction_comparative(self, contradiction: Contradiction) -> Optional[Question]:
        """
        Generate comparative question from contradiction.
        
        Args:
            contradiction: Source disagreement
            
        Returns:
            Question or None if generation fails
        """
        comparison_basis = f"""CONTRADICTION FOUND:
Source A ({contradiction.source_a}): {contradiction.claim_a}
Source B ({contradiction.source_b}): {contradiction.claim_b}

Context: {contradiction.context}

This suggests variation that needs comparison.
"""
        
        prompt = COMPARATIVE_PROMPT.format(comparison_basis=comparison_basis)
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian generating comparative research questions."},
                {"role": "user", "content": prompt}
            ],
            profile="quality",
            temperature=0.4
        )
        
        if not response.success:
            return None
        
        try:
            questions = parse_llm_question_response(response.content)
            if questions:
                q = questions[0]
                q.contradiction_source = f"{contradiction.claim_a} vs {contradiction.claim_b}"
                q.evidence_doc_ids = [contradiction.source_a, contradiction.source_b]
                q.generation_method = "contradiction_comparative"
                return q
        except Exception as e:
            debug_print(f"Failed to parse comparative question: {e}")
            return None
    
    def from_temporal_map_change(self, temporal_events: Dict[str, List[str]]) -> List[Question]:
        """
        Generate change-over-time questions from temporal map.
        
        Args:
            temporal_events: Year -> events mapping
            
        Returns:
            List of change questions
        """
        if not temporal_events or len(temporal_events) < 2:
            return []
        
        # Find years with significant events
        years = sorted(temporal_events.keys())
        start_year = years[0]
        end_year = years[-1]
        
        # Build evidence
        temporal_evidence = f"""TIME RANGE: {start_year} - {end_year}

Events by year:
"""
        for year in years[:10]:  # First 10 years
            events = temporal_events[year]
            temporal_evidence += f"{year}: {', '.join(events[:3])}\n"
        
        prompt = CHANGE_OVER_TIME_PROMPT.format(temporal_evidence=temporal_evidence)
        
        response = self.llm.generate(
            messages=[
                {"role": "system", "content": "You are a historian generating change-over-time questions."},
                {"role": "user", "content": prompt}
            ],
            profile="quality",
            temperature=0.5,
            max_tokens=1000
        )
        
        if not response.success:
            return []
        
        try:
            questions = parse_llm_question_response(response.content)
            for q in questions:
                q.generation_method = "temporal_change"
            return questions
        except Exception as e:
            debug_print(f"Failed to parse change questions: {e}")
            return []
    
    def generate_by_type(
        self,
        qtype: QuestionType,
        context: Dict[str, Any]
    ) -> List[Question]:
        """
        Generate questions of specific type.
        
        Args:
            qtype: Question type to generate
            context: Context for generation (patterns, entities, etc.)
            
        Returns:
            List of questions
        """
        if qtype == QuestionType.CAUSAL:
            patterns = context.get('patterns', [])
            return [q for p in patterns[:5] if (q := self.from_pattern_causal(p))]
        
        elif qtype == QuestionType.COMPARATIVE:
            contradictions = context.get('contradictions', [])
            return [q for c in contradictions[:5] if (q := self.from_contradiction_comparative(c))]
        
        elif qtype == QuestionType.CHANGE_OVER_TIME:
            temporal_map = context.get('temporal_map', {})
            return self.from_temporal_map_change(temporal_map)
        
        # Add other types as needed
        else:
            debug_print(f"Question type {qtype} not yet implemented")
            return []


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_typed_questions(notebook: ResearchNotebook) -> List[Question]:
    """
    Generate questions of all types from notebook.
    
    Args:
        notebook: Research notebook with patterns, contradictions, etc.
        
    Returns:
        List of typed questions
    """
    generator = TypedQuestionGenerator()
    questions = []
    
    # Prepare context
    context = {
        'patterns': list(notebook.patterns.values()),
        'contradictions': notebook.contradictions,
        'temporal_map': dict(notebook.temporal_map)
    }
    
    # Generate by type
    for qtype in [QuestionType.CAUSAL, QuestionType.COMPARATIVE, QuestionType.CHANGE_OVER_TIME]:
        typed_questions = generator.generate_by_type(qtype, context)
        questions.extend(typed_questions)
        debug_print(f"Generated {len(typed_questions)} {qtype.value} questions")
    
    return questions
