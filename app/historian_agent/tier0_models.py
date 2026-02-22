# app/historian_agent/tier0_models.py
# Created: 2026-02-05
# Purpose: Tier 0 data models (internal lists, external dicts)

"""
Tier 0 Models

Internal representation uses dataclasses and lists for type safety.
External APIs convert to dictionaries for JSON serialization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentBlock:
    """Logical block within a document (semantic chunk)."""
    block_id: str
    block_type: str
    text: str
    page_numbers: List[int] = field(default_factory=list)

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Minimal dict for LLM prompt context."""
        return {
            "block_id": self.block_id,
            "type": self.block_type,
            "text": self.text,
            "pages": self.page_numbers,
        }


@dataclass
class DocumentObject:
    """Notebook-style document object with structured blocks."""
    doc_id: str
    filename: str
    year: Optional[int]
    document_type: str
    person_name: Optional[str]
    collection: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    blocks: List[DocumentBlock] = field(default_factory=list)

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Convert to dict for LLM context (closed-world document object)."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "year": self.year,
            "document_type": self.document_type,
            "person_name": self.person_name,
            "collection": self.collection,
            "metadata": self.metadata,
            "blocks": [block.to_prompt_dict() for block in self.blocks],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for external serialization."""
        return self.to_prompt_dict()
