# app/historian_agent/semantic_chunker.py
# Created: 2026-02-05
# Purpose: Semantic chunking for Tier 0 (Notebook-LLM style)

"""
Semantic Chunker

Splits documents into logical units instead of fixed token windows.
Falls back to conservative paragraph splitting when semantic cues are missing.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import re

from rag_base import debug_print

try:
    from historian_agent.chunking import DocumentChunker
except Exception:  # pragma: no cover - optional dependency
    DocumentChunker = None  # type: ignore

from historian_agent.tier0_models import DocumentBlock


_SECTION_PATTERNS: List[Tuple[str, str]] = [
    ("header", r"(?:Employee Name|Date of Injury|Location|Department|Division):"),
    ("injury", r"(?:Injury Description|Nature of Injury|How Injured):"),
    ("treatment", r"(?:Medical Treatment|Doctor|Hospital|Treatment):"),
    ("witness", r"(?:Witness|Statement|Testimony):"),
    ("disposition", r"(?:Disposition|Outcome|Compensation|Claim):"),
]


class SemanticChunker:
    """Create logical blocks for a document."""

    def __init__(
        self,
        enabled: bool = True,
        max_block_chars: int = 2000,
        max_blocks_per_doc: int = 12,
        fallback_chunk_chars: int = 1200,
        fallback_chunk_overlap: int = 150,
    ) -> None:
        self.enabled = enabled
        self.max_block_chars = max_block_chars
        self.max_blocks_per_doc = max_blocks_per_doc
        self.fallback_chunk_chars = fallback_chunk_chars
        self.fallback_chunk_overlap = fallback_chunk_overlap
        self._fallback_chunker = self._init_fallback_chunker()

    def _init_fallback_chunker(self) -> Optional[DocumentChunker]:
        if DocumentChunker is None:
            return None
        try:
            return DocumentChunker(
                chunk_size=self.fallback_chunk_chars,
                chunk_overlap=self.fallback_chunk_overlap,
                context_fields=["content", "ocr_text", "text", "structured_data"],
            )
        except Exception as e:
            debug_print(f"SemanticChunker: fallback chunker unavailable: {e}")
            return None

    def chunk_document(self, doc_id: str, document: Dict[str, Any]) -> List[DocumentBlock]:
        """Chunk a document into logical blocks."""
        text = self._extract_text(document)
        if not text:
            return []

        if not self.enabled:
            return [self._make_block(doc_id, 0, "full_document", text)]

        doc_type = (document.get("document_type") or document.get("type") or "").lower()

        if self._looks_like_table(text):
            return [self._make_block(doc_id, 0, "table", text)]

        if "injury" in doc_type or self._has_injury_markers(text):
            sections = self._split_by_markers(text)
            blocks = [
                self._make_block(doc_id, i, section_type, section_text)
                for i, (section_type, section_text) in enumerate(sections)
            ]
            return self._cap_blocks(blocks)

        if len(text) <= self.max_block_chars:
            return [self._make_block(doc_id, 0, "full_document", text)]

        # Fallback chunking for long documents
        blocks = self._fallback_split(doc_id, text)
        return self._cap_blocks(blocks)

    def _fallback_split(self, doc_id: str, text: str) -> List[DocumentBlock]:
        if self._fallback_chunker is not None:
            try:
                fake_doc = {"content": text}
                chunks = self._fallback_chunker.chunk_document(fake_doc)
                return [
                    self._make_block(doc_id, i, "segment", chunk.text)
                    for i, chunk in enumerate(chunks)
                ]
            except Exception as e:
                debug_print(f"SemanticChunker: fallback chunker failed: {e}")

        # Simple paragraph-based split
        return [
            self._make_block(doc_id, i, "segment", chunk)
            for i, chunk in enumerate(self._simple_split(text))
        ]

    def _simple_split(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        blocks: List[str] = []
        current: List[str] = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) + 2 > self.max_block_chars and current:
                blocks.append("\n\n".join(current))
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += len(para) + 2

        if current:
            blocks.append("\n\n".join(current))

        if not blocks:
            blocks = [text]

        return blocks

    def _cap_blocks(self, blocks: List[DocumentBlock]) -> List[DocumentBlock]:
        if len(blocks) <= self.max_blocks_per_doc:
            return blocks

        if self.max_blocks_per_doc <= 1:
            doc_id = blocks[0].block_id.split("::")[0]
            merged_text = "\n\n".join(block.text for block in blocks)
            return [self._make_block(doc_id, 0, "merged", merged_text)]

        # Merge overflow into final block
        kept = blocks[: self.max_blocks_per_doc - 1]
        overflow = blocks[self.max_blocks_per_doc - 1 :]
        merged_text = "\n\n".join(block.text for block in overflow)
        kept.append(self._make_block(kept[0].block_id.split("::")[0], len(kept), "merged", merged_text))
        return kept

    def _make_block(self, doc_id: str, index: int, block_type: str, text: str) -> DocumentBlock:
        block_text = text.strip()
        if len(block_text) > self.max_block_chars:
            block_text = block_text[: self.max_block_chars]
        block_id = f"{doc_id}::b{index}"
        return DocumentBlock(block_id=block_id, block_type=block_type, text=block_text, page_numbers=[])

    def _split_by_markers(self, text: str) -> List[Tuple[str, str]]:
        markers: List[Tuple[int, str]] = []
        for section_type, pattern in _SECTION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                markers.append((match.start(), section_type))
        markers.sort(key=lambda item: item[0])

        if not markers:
            return [("full_document", text)]

        sections: List[Tuple[str, str]] = []
        for idx, (start, section_type) in enumerate(markers):
            end = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((section_type, section_text))

        return sections if sections else [("full_document", text)]

    def _has_injury_markers(self, text: str) -> bool:
        for _, pattern in _SECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _looks_like_table(self, text: str) -> bool:
        if text.count("|") >= 6:
            return True
        if text.count("\t") >= 6:
            return True
        if re.search(r"\n\s*[-=]{3,}\s*\n", text):
            return True
        return False

    def _extract_text(self, document: Dict[str, Any]) -> str:
        for field in ("ocr_text", "content", "text"):
            value = document.get(field)
            if isinstance(value, str) and value.strip():
                return value

        structured = document.get("structured_data")
        if isinstance(structured, dict):
            parts = [str(v) for v in structured.values() if v]
            if parts:
                return " ".join(parts)

        return ""
