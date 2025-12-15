"""
ner_entity_processor.py - Entity Extraction & Linking for Railroad Employee Records

Extracts entities from structured employee records, normalizes them using LLM,
and creates a linked entity graph for cross-document queries.

Usage:
    docker compose exec app python app/ner_entity_processor.py
    docker compose exec app python setup/ner_processor_llm.py --limit 100  # Test on 100 docs
    docker compose exec app python app/ner_entity_processor.py --force  # Reprocess all
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re

from pymongo import MongoClient
from bson import ObjectId
import requests
from fuzzywuzzy import fuzz
from tqdm import tqdm

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('NERProcessor')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler('ner_processing.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =======================
# Database Connection
# =======================

def get_client():
    """Initialize MongoDB client."""
    mongo_uri = os.environ.get('APP_MONGO_URI') or os.environ.get('MONGO_URI')
    if not mongo_uri:
        raise ValueError("Neither APP_MONGO_URI nor MONGO_URI environment variable is set")
    
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    logger.info("Connected to MongoDB")
    return client

def get_db(client):
    """Return the database instance."""
    return client['railroad_documents']

# =======================
# LLM Helper Functions
# =======================

def call_ollama(prompt: str, model: str = "llama3.1:8b") -> Optional[str]:
    """Call local Ollama instance for entity normalization."""
    ollama_url = os.environ.get('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
    
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for consistency
                    "num_predict": 200
                }
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result.get('response', '').strip()
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return None

def normalize_name_with_llm(name: str) -> Dict[str, str]:
    """Use LLM to parse and normalize names."""
    prompt = f"""Parse this railroad employee name into standard format. Return ONLY a JSON object with no additional text:

Name: "{name}"

Return format:
{{"last": "LastName", "first": "FirstName", "middle": "MiddleInitial", "full": "FirstName MiddleInitial LastName"}}

If middle name is missing, set it to empty string. OReturn ONLY valid JSON with no markdown, no explanation, no extra text."""

    response = call_ollama(prompt)
    if not response:
        # Fallback to simple parsing
        return parse_name_fallback(name)
    
    try:
        # Clean response - remove markdown code blocks if present
        response = response.strip()
        if response.startswith('```'):
            response = response.split('\n', 1)[1]
        if response.endswith('```'):
            response = response.rsplit('\n', 1)[0]
        response = response.strip()
        
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM response for name: {name}")
        return parse_name_fallback(name)

def parse_name_fallback(name: str) -> Dict[str, str]:
    """Fallback name parser without LLM."""
    name = name.strip()
    
    # Handle "Last, First MI" format
    if ',' in name:
        parts = name.split(',', 1)
        last = parts[0].strip()
        rest = parts[1].strip().split()
        first = rest[0] if rest else ""
        middle = rest[1] if len(rest) > 1 else ""
    else:
        # Handle "First MI Last" format
        parts = name.split()
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            middle = parts[1] if len(parts) > 2 else ""
        else:
            first = parts[0] if parts else ""
            last = ""
            middle = ""
    
    full = f"{first} {middle} {last}".strip()
    return {
        "last": last,
        "first": first,
        "middle": middle,
        "full": full
    }

# =======================
# Entity Extraction
# =======================

def extract_employee_ids(text: str, sections: List[Dict]) -> List[str]:
    """Extract employee ID numbers from remarks and OCR text."""
    ids = []
    
    # Pattern: alphanumeric codes like "4967D", "PY418", "R218"
    pattern = r'\b[A-Z]*\d{3,5}[A-Z]?\b'
    
    # From OCR text
    if text and isinstance(text, str):
        matches = re.findall(pattern, text)
        ids.extend(matches)
    
    # From sections
    for section in sections:
        fields = section.get('fields', [])
        for field in fields:
            if isinstance(field, dict):
                for key, value in field.items():
                    if value and isinstance(value, str):
                        matches = re.findall(pattern, value)
                        ids.extend(matches)
    
    return list(set(ids))  # Remove duplicates

def extract_locations(sections: List[Dict]) -> List[str]:
    """Extract location/division names from structured fields."""
    locations = []
    
    for section in sections:
        fields = section.get('fields', [])
        for field in fields:
            if isinstance(field, dict):
                # Check for division, location, department fields
                for key in ['division', 'location', 'department', 'Division', 'Location']:
                    if key in field:
                        value = field[key]
                        if value and isinstance(value, str) and len(value) > 1:
                            locations.append(value.strip())
                
                # Check field_name and value pairs
                if field.get('field_name') in ['Division', 'Location', 'Department']:
                    value = field.get('value')
                    if value and isinstance(value, str) and len(value) > 1:
                        locations.append(value.strip())
    
    return list(set(locations))

def extract_dates(sections: List[Dict]) -> List[str]:
    """Extract dates from structured fields."""
    dates = []
    
    for section in sections:
        fields = section.get('fields', [])
        for field in fields:
            if isinstance(field, dict):
                # Look for date fields
                for key in ['date', 'Date', 'born', 'Born']:
                    if key in field:
                        value = field[key]
                        if value and isinstance(value, str):
                            dates.append(value.strip())
                
                if field.get('field_name') in ['Date', 'Born']:
                    value = field.get('value')
                    if value and isinstance(value, str):
                        dates.append(value.strip())
    
    return list(set(dates))

def extract_occupations(sections: List[Dict]) -> List[str]:
    """Extract occupation/position information."""
    occupations = []
    
    for section in sections:
        fields = section.get('fields', [])
        for field in fields:
            if isinstance(field, dict):
                for key in ['position', 'Position', 'occupation', 'Occupation']:
                    if key in field:
                        value = field[key]
                        if value and isinstance(value, str) and len(value) > 1:
                            # Skip generic values
                            if value.lower() not in ['same', '']:
                                occupations.append(value.strip())
                
                if field.get('field_name') in ['Position', 'Occupation']:
                    value = field.get('value')
                    if value and isinstance(value, str) and len(value) > 1:
                        if value.lower() not in ['same', '']:
                            occupations.append(value.strip())
    
    return list(set(occupations))

def extract_entities_from_document(doc: Dict) -> Dict:
    """Extract all entities from a document."""
    sections = doc.get('sections', [])
    ocr_text = doc.get('ocr_text', '')
    
    entities = {
        'PERSON': [],
        'GPE': [],  # Geo-political entities (locations)
        'DATE': [],
        'OCCUPATION': [],
        'EMPLOYEE_ID': [],
        'ORG': ['Baltimore and Ohio Railroad']  # Implicit
    }
    
    # Extract person name from structured fields
    for section in sections:
        if section.get('section_name') == 'Personal Information':
            fields = section.get('fields', [])
            for field in fields:
                if isinstance(field, dict):
                    if field.get('field_name') == 'Name' or 'name' in field:
                        name = field.get('value') or field.get('name', '')
                        if name:
                            entities['PERSON'].append(name.strip())
    
    # Extract other entities
    entities['GPE'] = extract_locations(sections)
    entities['DATE'] = extract_dates(sections)
    entities['OCCUPATION'] = extract_occupations(sections)
    entities['EMPLOYEE_ID'] = extract_employee_ids(ocr_text, sections)
    
    return entities

# =======================
# Entity Resolution & Linking
# =======================

def find_matching_entity(db, entity_type: str, value: str, threshold: int = 85) -> Optional[Dict]:
    """Find existing entity that matches the given value."""
    linked_entities = db['linked_entities']
    
    # Exact match first
    existing = linked_entities.find_one({
        'type': entity_type,
        '$or': [
            {'canonical_name': value},
            {'variants': value}
        ]
    })
    
    if existing:
        return existing
    
    # For PERSON entities, do fuzzy matching
    if entity_type == 'PERSON':
        all_persons = linked_entities.find({'type': 'PERSON'})
        
        for person in all_persons:
            # Check canonical name
            ratio = fuzz.ratio(value.lower(), person['canonical_name'].lower())
            if ratio >= threshold:
                logger.debug(f"Fuzzy match: '{value}' → '{person['canonical_name']}' ({ratio}%)")
                return person
            
            # Check variants
            for variant in person.get('variants', []):
                ratio = fuzz.ratio(value.lower(), variant.lower())
                if ratio >= threshold:
                    logger.debug(f"Fuzzy match variant: '{value}' → '{variant}' ({ratio}%)")
                    return person
    
    return None

def create_or_update_entity(db, doc_id: str, entity_type: str, value: str, 
                            context: Optional[Dict] = None) -> str:
    """Create new entity or update existing one."""
    linked_entities = db['linked_entities']
    
    # Normalize person names with LLM
    if entity_type == 'PERSON':
        parsed_name = normalize_name_with_llm(value)
        canonical_name = parsed_name['full']
    else:
        canonical_name = value
    
    # Check if entity exists
    existing = find_matching_entity(db, entity_type, value)
    
    if existing:
        # Update existing entity
        entity_id = str(existing['_id'])
        
        updates = {
            '$addToSet': {
                'document_ids': doc_id,
                'variants': value
            },
            '$inc': {'mention_count': 1}
        }
        
        if context:
            updates['$push'] = {'contexts': context}
        
        linked_entities.update_one({'_id': existing['_id']}, updates)
        logger.debug(f"Updated entity: {canonical_name}")
        
        return entity_id
    else:
        # Create new entity
        entity_doc = {
            'canonical_name': canonical_name,
            'type': entity_type,
            'variants': [value] if value != canonical_name else [],
            'document_ids': [doc_id],
            'mention_count': 1,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        if entity_type == 'PERSON':
            entity_doc['parsed_name'] = parsed_name
        
        if context:
            entity_doc['contexts'] = [context]
        
        result = linked_entities.insert_one(entity_doc)
        logger.debug(f"Created new entity: {canonical_name}")
        
        return str(result.inserted_id)

def link_entities_by_employee_id(db, doc_id: str, employee_ids: List[str], person_name: str):
    """Link person entities using employee ID as definitive match.
    
    This creates a single canonical entity for each unique person (identified by employee_id),
    and multiple documents reference that same entity. Documents are NOT merged.
    """
    if not employee_ids or not person_name:
        return
    
    linked_entities = db['linked_entities']
    
    for emp_id in employee_ids:
        # Find existing person entity with this employee ID
        existing_person = linked_entities.find_one({
            'type': 'PERSON',
            'employee_ids': emp_id
        })
        
        if existing_person:
            # Entity exists - just add this document to its reference list
            # and add name variant if it's new
            updates = {
                '$addToSet': {
                    'document_ids': doc_id,
                    'employee_ids': emp_id
                }
            }
            
            # Add name as variant if it's different from canonical
            if person_name != existing_person['canonical_name']:
                updates['$addToSet']['variants'] = person_name
            
            linked_entities.update_one(
                {'_id': existing_person['_id']},
                updates
            )
            logger.debug(f"Linked doc {doc_id} to existing entity via employee_id {emp_id}")
        else:
            # First time seeing this employee ID - add it to the person's entity
            linked_entities.update_one(
                {
                    'type': 'PERSON',
                    '$or': [
                        {'canonical_name': person_name},
                        {'variants': person_name}
                    ]
                },
                {
                    '$addToSet': {
                        'employee_ids': emp_id,
                        'document_ids': doc_id
                    }
                }
            )
            logger.debug(f"Added employee_id {emp_id} to person entity for {person_name}")

# =======================
# Main Processing
# =======================

def process_document(db, doc: Dict, force: bool = False) -> bool:
    """Process a single document for entity extraction."""
    doc_id = str(doc['_id'])
    
    # Skip if already processed (unless force)
    if not force and doc.get('entities_extracted'):
        logger.debug(f"Skipping already processed document: {doc_id}")
        return False
    
    try:
        # Extract entities
        entities = extract_entities_from_document(doc)
        
        # Create entity references
        entity_refs = []
        
        # Process each entity type
        for entity_type, values in entities.items():
            for value in values:
                if not value or value.strip() == '':
                    continue
                
                # Create context for this mention
                context = {
                    'document_id': doc_id,
                    'mention': value,
                    'summary': (doc.get('summary') or '')[:200]
                }
                
                # Create or link entity
                entity_id = create_or_update_entity(db, doc_id, entity_type, value, context)
                entity_refs.append(entity_id)
        
        # Special handling: link by employee ID
        person_names = entities.get('PERSON', [])
        employee_ids = entities.get('EMPLOYEE_ID', [])
        if person_names and employee_ids:
            link_entities_by_employee_id(db, doc_id, employee_ids, person_names[0])
        
        # Update document with extracted entities
        db['documents'].update_one(
            {'_id': doc['_id']},
            {
                '$set': {
                    'entities': entities,
                    'entity_refs': entity_refs,
                    'entities_extracted': True,
                    'entities_extracted_at': datetime.utcnow()
                }
            }
        )
        
        logger.debug(f"Processed document {doc_id}: {len(entity_refs)} entities")
        return True
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
        return False

def process_all_documents(limit: Optional[int] = None, force: bool = False):
    """Process all documents in the database."""
    client = get_client()
    db = get_db(client)
    
    # Build query
    query = {}
    if not force:
        query['entities_extracted'] = {'$ne': True}
    
    total = db['documents'].count_documents(query)
    logger.info(f"Found {total} documents to process")
    
    if total == 0:
        print("✅ All documents already processed!")
        print("Use --force to reprocess all documents")
        return
    
    # Apply limit if specified
    cursor = db['documents'].find(query)
    if limit:
        cursor = cursor.limit(limit)
        total = min(total, limit)
    
    # Process documents
    processed = 0
    failed = 0
    
    print(f"\n🔄 Processing {total} documents...")
    with tqdm(total=total, desc="Extracting entities") as pbar:
        for doc in cursor:
            success = process_document(db, doc, force)
            if success:
                processed += 1
            else:
                failed += 1
            pbar.update(1)
    
    # Print summary
    print("\n" + "="*60)
    print("Entity Extraction Complete!")
    print("="*60)
    print(f"✅ Processed: {processed}")
    print(f"⏭️  Skipped: {total - processed - failed}")
    print(f"❌ Failed: {failed}")
    
    # Entity statistics
    entity_count = db['linked_entities'].count_documents({})
    person_count = db['linked_entities'].count_documents({'type': 'PERSON'})
    location_count = db['linked_entities'].count_documents({'type': 'GPE'})
    
    print(f"\n📊 Entity Statistics:")
    print(f"   Total entities: {entity_count}")
    print(f"   Persons: {person_count}")
    print(f"   Locations: {location_count}")
    
    # Top entities
    print(f"\n👥 Top 10 Most Mentioned Persons:")
    top_persons = db['linked_entities'].find(
        {'type': 'PERSON'}
    ).sort('mention_count', -1).limit(10)
    
    for i, person in enumerate(top_persons, 1):
        name = person['canonical_name']
        count = person.get('mention_count', 0)
        variants = person.get('variants', [])
        print(f"   {i}. {name} ({count} docs)")
        if variants:
            print(f"      Variants: {', '.join(variants[:3])}")

# =======================
# Main Execution
# =======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and link entities from railroad employee records"
    )
    parser.add_argument(
        '--limit',
        type=int,
        help="Limit number of documents to process (for testing)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Reprocess all documents, even if already processed"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("NER Entity Extraction & Linking")
    print("="*60)
    
    process_all_documents(limit=args.limit, force=args.force)