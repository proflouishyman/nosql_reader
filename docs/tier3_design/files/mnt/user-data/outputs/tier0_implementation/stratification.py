# app/historian_agent/stratification.py
# Created: 2025-12-29
# Purpose: Stratification strategies for systematic corpus reading

"""
Stratification - Divide corpus into meaningful reading batches.

Historians don't read randomly - they read systematically across:
- Time (chronological slices)
- Genre (document types)
- People (biographical focus)
- Collections (archival provenance)

This module creates sampling strategies that ensure comprehensive coverage
rather than just "top-k most similar".
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from rag_base import DocumentStore, debug_print


# ============================================================================
# Stratum Definition
# ============================================================================

@dataclass
class Stratum:
    """
    One stratum (reading batch) of the corpus.
    
    A stratum is a meaningful subset for focused reading:
    - All documents from year 1923
    - All injury reports
    - All documents for person X
    - All documents in Box 129
    """
    stratum_type: str  # temporal, genre, biographical, collection, spatial
    label: str  # Human-readable label ("Year 1923", "Injury Reports")
    filters: Dict[str, Any]  # MongoDB query filters
    sample_size: int  # How many docs to read from this stratum
    priority: int = 1  # Higher priority = read first


# ============================================================================
# Stratification Strategies
# ============================================================================

class CorpusStratifier:
    """
    Creates systematic sampling strategies for corpus exploration.
    
    Uses actual MongoDB data to build realistic strata.
    """
    
    def __init__(self):
        """Initialize stratifier with DocumentStore."""
        self.doc_store = DocumentStore()
        self.documents_coll = self.doc_store.documents_coll
    
    def temporal_stratification(
        self,
        year_range: Optional[Tuple[int, int]] = None,
        docs_per_year: int = 50
    ) -> List[Stratum]:
        """
        Stratify by time slices (chronological reading).
        
        Args:
            year_range: Optional (start_year, end_year) tuple
            docs_per_year: How many documents to sample per year
            
        Returns:
            List of temporal strata
        """
        debug_print("Building temporal stratification")
        
        # Get year distribution from MongoDB
        pipeline = [
            {'$match': {'year': {'$ne': None}}},
            {'$group': {
                '_id': '$year',
                'count': {'$sum': 1}
            }},
            {'$sort': {'_id': 1}}
        ]
        
        year_counts = list(self.documents_coll.aggregate(pipeline))
        
        if not year_counts:
            debug_print("No year data found in corpus")
            return []
        
        # Apply year range filter if provided
        if year_range:
            year_counts = [
                y for y in year_counts 
                if year_range[0] <= y['_id'] <= year_range[1]
            ]
        
        strata = []
        for year_data in year_counts:
            year = year_data['_id']
            count = year_data['count']
            
            sample_size = min(docs_per_year, count)
            
            strata.append(Stratum(
                stratum_type='temporal',
                label=f'Year {year}',
                filters={'year': year},
                sample_size=sample_size,
                priority=1
            ))
        
        debug_print(f"Created {len(strata)} temporal strata")
        return strata
    
    def genre_stratification(
        self,
        docs_per_type: int = 100
    ) -> List[Stratum]:
        """
        Stratify by document type (genre reading).
        
        Args:
            docs_per_type: How many documents to sample per type
            
        Returns:
            List of genre strata
        """
        debug_print("Building genre stratification")
        
        # Get document type distribution
        pipeline = [
            {'$match': {'document_type': {'$ne': None}}},
            {'$group': {
                '_id': '$document_type',
                'count': {'$sum': 1}
            }},
            {'$sort': {'count': -1}}
        ]
        
        type_counts = list(self.documents_coll.aggregate(pipeline))
        
        if not type_counts:
            debug_print("No document_type data found")
            return []
        
        strata = []
        for type_data in type_counts:
            doc_type = type_data['_id']
            count = type_data['count']
            
            sample_size = min(docs_per_type, count)
            
            strata.append(Stratum(
                stratum_type='genre',
                label=f'{doc_type}',
                filters={'document_type': doc_type},
                sample_size=sample_size,
                priority=2
            ))
        
        debug_print(f"Created {len(strata)} genre strata")
        return strata
    
    def biographical_stratification(
        self,
        min_docs_per_person: int = 10,
        max_people: int = 50,
        docs_per_person: int = 20
    ) -> List[Stratum]:
        """
        Stratify by person (biographical reading).
        
        Focus on people with substantial documentation.
        
        Args:
            min_docs_per_person: Only include people with at least this many docs
            max_people: Maximum number of people to include
            docs_per_person: How many docs to sample per person
            
        Returns:
            List of biographical strata
        """
        debug_print("Building biographical stratification")
        
        # Get people with most documentation
        pipeline = [
            {'$match': {'person_id': {'$ne': None}}},
            {'$group': {
                '_id': {
                    'person_id': '$person_id',
                    'person_name': '$person_name',
                    'person_folder': '$person_folder'
                },
                'count': {'$sum': 1}
            }},
            {'$match': {'count': {'$gte': min_docs_per_person}}},
            {'$sort': {'count': -1}},
            {'$limit': max_people}
        ]
        
        people = list(self.documents_coll.aggregate(pipeline))
        
        if not people:
            debug_print("No person data found")
            return []
        
        strata = []
        for person_data in people:
            person_info = person_data['_id']
            count = person_data['count']
            
            sample_size = min(docs_per_person, count)
            
            # Create human-readable label
            label = f"{person_info['person_name']} (ID: {person_info['person_id']}, {count} docs)"
            
            strata.append(Stratum(
                stratum_type='biographical',
                label=label,
                filters={'person_id': person_info['person_id']},
                sample_size=sample_size,
                priority=3
            ))
        
        debug_print(f"Created {len(strata)} biographical strata")
        return strata
    
    def collection_stratification(
        self,
        docs_per_collection: int = 100
    ) -> List[Stratum]:
        """
        Stratify by collection (archival provenance).
        
        Args:
            docs_per_collection: How many docs to sample per collection
            
        Returns:
            List of collection strata
        """
        debug_print("Building collection stratification")
        
        # Get collection distribution
        pipeline = [
            {'$match': {'collection': {'$ne': None}}},
            {'$group': {
                '_id': '$collection',
                'count': {'$sum': 1}
            }},
            {'$sort': {'count': -1}}
        ]
        
        collections = list(self.documents_coll.aggregate(pipeline))
        
        if not collections:
            debug_print("No collection data found")
            return []
        
        strata = []
        for coll_data in collections:
            collection = coll_data['_id']
            count = coll_data['count']
            
            sample_size = min(docs_per_collection, count)
            
            strata.append(Stratum(
                stratum_type='collection',
                label=f'Collection: {collection}',
                filters={'collection': collection},
                sample_size=sample_size,
                priority=2
            ))
        
        debug_print(f"Created {len(strata)} collection strata")
        return strata
    
    def spatial_stratification(
        self,
        docs_per_box: int = 50
    ) -> List[Stratum]:
        """
        Stratify by physical location (box reading).
        
        Args:
            docs_per_box: How many docs to sample per box
            
        Returns:
            List of spatial strata
        """
        debug_print("Building spatial stratification")
        
        # Get box distribution
        pipeline = [
            {'$match': {'archive_structure.physical_box': {'$ne': None}}},
            {'$group': {
                '_id': '$archive_structure.physical_box',
                'count': {'$sum': 1}
            }},
            {'$sort': {'count': -1}}
        ]
        
        boxes = list(self.documents_coll.aggregate(pipeline))
        
        if not boxes:
            debug_print("No physical_box data found")
            return []
        
        strata = []
        for box_data in boxes:
            box = box_data['_id']
            count = box_data['count']
            
            sample_size = min(docs_per_box, count)
            
            strata.append(Stratum(
                stratum_type='spatial',
                label=f'Box: {box}',
                filters={'archive_structure.physical_box': box},
                sample_size=sample_size,
                priority=2
            ))
        
        debug_print(f"Created {len(strata)} spatial strata")
        return strata
    
    def build_comprehensive_strategy(
        self,
        total_budget: int = 2000,
        strategy: str = 'balanced'
    ) -> List[Stratum]:
        """
        Build comprehensive reading strategy combining multiple approaches.
        
        Args:
            total_budget: Total number of documents to read
            strategy: 'temporal' (chronological focus), 
                     'biographical' (person focus),
                     'genre' (document type focus),
                     'balanced' (mix of all)
            
        Returns:
            Ordered list of strata to read
        """
        debug_print(f"Building comprehensive strategy: {strategy}, budget: {total_budget}")
        
        all_strata = []
        
        if strategy == 'temporal':
            # Focus on chronological coverage
            all_strata = self.temporal_stratification(docs_per_year=50)
            
        elif strategy == 'biographical':
            # Focus on people
            all_strata = self.biographical_stratification(
                min_docs_per_person=10,
                max_people=50,
                docs_per_person=30
            )
            
        elif strategy == 'genre':
            # Focus on document types
            all_strata = self.genre_stratification(docs_per_type=100)
            
        elif strategy == 'balanced':
            # Mix of approaches
            temporal = self.temporal_stratification(docs_per_year=25)
            genre = self.genre_stratification(docs_per_type=50)
            biographical = self.biographical_stratification(
                min_docs_per_person=10,
                max_people=20,
                docs_per_person=20
            )
            collection = self.collection_stratification(docs_per_collection=50)
            
            all_strata = temporal + genre + biographical + collection
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Sort by priority
        all_strata.sort(key=lambda s: s.priority)
        
        # Trim to budget
        all_strata = self._trim_to_budget(all_strata, total_budget)
        
        debug_print(f"Final strategy: {len(all_strata)} strata, ~{sum(s.sample_size for s in all_strata)} docs")
        
        return all_strata
    
    def _trim_to_budget(self, strata: List[Stratum], budget: int) -> List[Stratum]:
        """
        Trim strata list to fit within document budget.
        
        Args:
            strata: List of strata
            budget: Maximum total documents
            
        Returns:
            Trimmed list of strata
        """
        total = 0
        trimmed = []
        
        for stratum in strata:
            if total + stratum.sample_size <= budget:
                trimmed.append(stratum)
                total += stratum.sample_size
            else:
                # Partial stratum to fill budget
                remaining = budget - total
                if remaining > 0:
                    stratum.sample_size = remaining
                    trimmed.append(stratum)
                break
        
        return trimmed


# ============================================================================
# Stratum Reader
# ============================================================================

class StratumReader:
    """Reads documents from a stratum."""
    
    def __init__(self):
        """Initialize reader with DocumentStore."""
        self.doc_store = DocumentStore()
        self.documents_coll = self.doc_store.documents_coll
    
    def read_stratum(self, stratum: Stratum) -> List[Dict[str, Any]]:
        """
        Read documents from a stratum with diversity sampling.
        
        Args:
            stratum: Stratum to read from
            
        Returns:
            List of document dictionaries
        """
        debug_print(f"Reading stratum: {stratum.label} ({stratum.sample_size} docs)")
        
        # Build query from filters
        query = stratum.filters.copy()
        
        # Sample documents
        # Use $sample for random selection within stratum
        pipeline = [
            {'$match': query},
            {'$sample': {'size': stratum.sample_size}}
        ]
        
        docs = list(self.documents_coll.aggregate(pipeline))
        
        debug_print(f"Retrieved {len(docs)} documents from {stratum.label}")
        
        return docs
    
    def format_docs_for_llm(self, docs: List[Dict[str, Any]], max_chars: int = 50000) -> str:
        """
        Format documents as text for LLM reading.
        
        Args:
            docs: List of document dictionaries
            max_chars: Maximum characters to include
            
        Returns:
            Formatted text for LLM
        """
        formatted = []
        total_chars = 0
        
        for i, doc in enumerate(docs):
            # Extract text content
            text_content = self._extract_text(doc)
            
            if not text_content:
                continue
            
            # Build document summary
            doc_summary = f"""--- Document {i+1}: {doc.get('filename', 'Unknown')} ---
Year: {doc.get('year', 'Unknown')}
Type: {doc.get('document_type', 'Unknown')}
Person: {doc.get('person_name', 'Unknown')}
Collection: {doc.get('collection', 'Unknown')}

Content (excerpt):
{text_content[:2000]}
---
"""
            
            if total_chars + len(doc_summary) > max_chars:
                break
            
            formatted.append(doc_summary)
            total_chars += len(doc_summary)
        
        return "\n\n".join(formatted)
    
    def _extract_text(self, doc: Dict[str, Any]) -> str:
        """Extract text content from document."""
        # Try multiple fields
        text_fields = ['ocr_text', 'content', 'text', 'structured_data']
        
        for field in text_fields:
            if field in doc and doc[field]:
                if isinstance(doc[field], str):
                    return doc[field]
                elif isinstance(doc[field], dict):
                    # For structured_data, concatenate values
                    return ' '.join(str(v) for v in doc[field].values() if v)
        
        return ""
