# app/historian_agent/semantic_chunker.py
# Created: 2025-12-29
# Purpose: Semantic chunking for historical documents (Notebook-LLM style)

"""
Semantic Chunker - Document-aware chunking (not mechanical).

Notebook-LLM Principle:
"Documents are pre-processed into logical units (sections, tables, footnotes),
not arbitrary token windows."

For B&O Railroad documents, logical units are:
- Injury reports (complete case)
- Employment records (complete person)
- Correspondence (complete letter)
- Tables (complete table, not split rows)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class SemanticChunk:
    """
    Document chunk representing a logical unit.
    
    Unlike mechanical chunks (fixed tokens), this represents:
    - Complete injury case
    - Complete employment record
    - Complete table
    - Complete letter/memo
    """
    chunk_id: str
    document_id: str
    chunk_type: str  # "injury_report", "employment_record", "table", "letter"
    content: str
    
    # Structural metadata (Notebook-style)
    section_hierarchy: List[str]  # ["Document A", "Section 2", "Subsection 2.3"]
    parent_document_id: str
    sibling_chunk_ids: List[str]  # Chunks in same logical group
    
    # Citation metadata
    page_numbers: List[int]
    line_numbers: Optional[range] = None
    
    def to_document_object(self) -> Dict[str, Any]:
        """
        Convert to Notebook-style document object.
        
        Returns structured object that model can reason over,
        not just text blob.
        """
        return {
            'id': self.chunk_id,
            'type': self.chunk_type,
            'content': self.content,
            'location': {
                'document': self.parent_document_id,
                'hierarchy': ' > '.join(self.section_hierarchy),
                'pages': self.page_numbers
            },
            'relationships': {
                'siblings': self.sibling_chunk_ids,
                'parent': self.parent_document_id
            }
        }


class SemanticChunker:
    """
    Chunks documents into logical units (Notebook-LLM style).
    
    Contrasted with mechanical chunking:
    - Mechanical: Split every 1000 tokens regardless of content
    - Semantic: Split at logical boundaries (complete cases, tables, etc.)
    """
    
    def __init__(self):
        """Initialize semantic chunker."""
        pass
    
    def chunk_injury_report(self, document: Dict[str, Any]) -> List[SemanticChunk]:
        """
        Chunk injury report into logical units.
        
        Logical units for injury reports:
        1. Header (employee info, date, location)
        2. Injury description (complete narrative)
        3. Medical treatment (complete record)
        4. Witness statements (each statement complete)
        5. Disposition (outcome, compensation)
        
        Unlike mechanical chunking which might split:
        "John Smith suffered burns to left hand..."
        [CHUNK BOUNDARY]
        "...requiring 6 weeks recovery"
        
        Semantic keeps injury description intact.
        """
        chunks = []
        content = document.get('content', '')
        doc_id = document.get('_id', 'unknown')
        
        # Detect sections (simplified - you'd use better parsing)
        sections = self._detect_injury_sections(content)
        
        for i, section in enumerate(sections):
            chunk = SemanticChunk(
                chunk_id=f"{doc_id}_section_{i}",
                document_id=doc_id,
                chunk_type=section['type'],
                content=section['content'],
                section_hierarchy=[
                    document.get('title', 'Untitled'),
                    section['type']
                ],
                parent_document_id=doc_id,
                sibling_chunk_ids=[f"{doc_id}_section_{j}" for j in range(len(sections)) if j != i],
                page_numbers=section.get('pages', [])
            )
            chunks.append(chunk)
        
        return chunks
    
    def _detect_injury_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Detect logical sections in injury report.
        
        Uses domain knowledge of B&O forms:
        - "Employee Name:" → header section
        - "Injury Description:" → injury section
        - "Medical Treatment:" → treatment section
        - "Witness:" → witness section
        """
        sections = []
        
        # Simple heuristic (you'd make this more robust)
        patterns = {
            'header': r'(?:Employee Name|Date of Injury|Location):',
            'injury': r'(?:Injury Description|Nature of Injury|How Injured):',
            'treatment': r'(?:Medical Treatment|Doctor|Hospital|Treatment):',
            'witness': r'(?:Witness|Statement|Testimony):',
            'disposition': r'(?:Disposition|Outcome|Compensation|Claim):'
        }
        
        # Split content by section markers
        current_pos = 0
        current_type = 'header'
        
        for section_type, pattern in patterns.items():
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                # Extract content from current_pos to first match
                if current_pos > 0:
                    section_content = content[current_pos:matches[0].start()].strip()
                    if section_content:
                        sections.append({
                            'type': current_type,
                            'content': section_content,
                            'pages': []  # You'd extract from OCR metadata
                        })
                current_pos = matches[0].start()
                current_type = section_type
        
        # Add final section
        if current_pos < len(content):
            sections.append({
                'type': current_type,
                'content': content[current_pos:].strip(),
                'pages': []
            })
        
        return sections if sections else [{'type': 'full_document', 'content': content, 'pages': []}]
    
    def chunk_table(self, table_content: str, document_id: str) -> SemanticChunk:
        """
        Chunk complete table (never split rows).
        
        Mechanical chunking problem:
        | Name    | Injury      | Date      |
        | Smith   | Burn        | 1923-05-12|
        [CHUNK BOUNDARY - LOSES CONTEXT]
        | Jones   | Fracture    | 1923-05-15|
        
        Semantic chunking: Keep entire table together.
        """
        return SemanticChunk(
            chunk_id=f"{document_id}_table",
            document_id=document_id,
            chunk_type="table",
            content=table_content,
            section_hierarchy=[document_id, "Table"],
            parent_document_id=document_id,
            sibling_chunk_ids=[],
            page_numbers=[]
        )


# ============================================================================
# Integration with Existing RAG
# ============================================================================

def convert_mechanical_to_semantic(
    mechanical_chunks: List[Dict],
    document: Dict[str, Any]
) -> List[SemanticChunk]:
    """
    Convert existing mechanical chunks to semantic chunks.
    
    Backwards compatible with your current system.
    """
    chunker = SemanticChunker()
    
    # Detect document type
    doc_type = document.get('document_type', 'unknown')
    
    if 'injury' in doc_type.lower():
        return chunker.chunk_injury_report(document)
    elif 'table' in doc_type.lower():
        return [chunker.chunk_table(document.get('content', ''), document.get('_id'))]
    else:
        # Fallback: keep mechanical chunks but wrap in semantic structure
        return [
            SemanticChunk(
                chunk_id=chunk.get('id', f"{document.get('_id')}_chunk_{i}"),
                document_id=document.get('_id', 'unknown'),
                chunk_type='text',
                content=chunk.get('content', ''),
                section_hierarchy=[document.get('title', 'Untitled')],
                parent_document_id=document.get('_id'),
                sibling_chunk_ids=[],
                page_numbers=[]
            )
            for i, chunk in enumerate(mechanical_chunks)
        ]
