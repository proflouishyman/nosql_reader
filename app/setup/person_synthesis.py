#!/usr/bin/env python3
"""
Person Synthesis System
Generate AI-powered biographical syntheses and embed them in each person's documents

Features:
- Hierarchical synthesis for large document collections (batches of 50)
- Debug mode to save prompts and responses
- Improved progress indicators
- Fixed database save with verification
"""
import os
import json
import requests
import time
from datetime import datetime
from pymongo import MongoClient
from pathlib import Path
import sys

# Configuration
# Use APP_MONGO_URI first (Docker), fallback to MONGO_URI (local)
MONGO_URI = os.environ.get('APP_MONGO_URI') or os.environ.get('MONGO_URI') or "mongodb://admin:secret@mongodb:27017/admin"
DB_NAME = 'railroad_documents'
OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL = "llama3.1:8b"

# Control flags
DRY_RUN = False  # Set to True to test without modifying database
MAX_PERSONS = 10  # Set to number to limit processing (e.g., 10 for testing), None for all
VERBOSE = True
DEBUG = False  # Set to True to save prompts and responses for inspection

def print_startup_diagnostics():
    """Print detailed startup diagnostics"""
    print("="*70)
    print("STARTUP DIAGNOSTICS")
    print("="*70)
    print(f"MongoDB URI: {MONGO_URI.replace('secret', '***')}")
    print(f"Target Database: {DB_NAME}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Model: {MODEL}")
    print(f"\nEnvironment Variables:")
    print(f"  APP_MONGO_URI: {'✓ SET' if os.environ.get('APP_MONGO_URI') else '✗ NOT SET'}")
    print(f"  MONGO_URI: {'✓ SET' if os.environ.get('MONGO_URI') else '✗ NOT SET'}")
    print(f"\nControl Flags:")
    print(f"  DRY_RUN: {DRY_RUN}")
    print(f"  MAX_PERSONS: {MAX_PERSONS}")
    print(f"  VERBOSE: {VERBOSE}")
    print(f"  DEBUG: {DEBUG}")
    print("="*70 + "\n")

class PersonSynthesizer:
    """Generate and store person syntheses"""
    
    def __init__(self, mongo_uri=MONGO_URI, db_name=DB_NAME):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.documents = self.db['documents']
        self.person_syntheses = self.db['person_syntheses']
        
        # Create debug directory if needed
        if DEBUG:
            self.debug_dir = Path("/home/claude/synthesis_debug")
            self.debug_dir.mkdir(exist_ok=True, parents=True)
            print(f"🐛 DEBUG mode enabled - saving to {self.debug_dir}")
    
    def _save_debug_file(self, person_folder, content, file_type, batch_num=None):
        """Save debug file with timestamp"""
        if not DEBUG:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean folder name for filesystem
        safe_folder = person_folder.replace('/', '_').replace(' ', '_')
        
        # Create subdirectory for this person
        person_dir = self.debug_dir / safe_folder
        person_dir.mkdir(exist_ok=True)
        
        # Build filename
        if batch_num is not None:
            filename = f"{timestamp}_batch{batch_num:02d}_{file_type}.txt"
        else:
            filename = f"{timestamp}_{file_type}.txt"
        
        filepath = person_dir / filename
        
        # Save content
        if isinstance(content, dict):
            content = json.dumps(content, indent=2, default=str)
        
        filepath.write_text(content, encoding='utf-8')
        
        if VERBOSE:
            print(f"    🐛 Saved debug file: {filepath.name}")
        
    def get_all_person_folders(self):
        """Get list of all unique person folders, sorted by document count (ascending)"""
        # Get all folders with their document counts
        pipeline = [
            {'$match': {'person_folder': {'$exists': True, '$ne': None}}},
            {'$group': {
                '_id': '$person_folder',
                'doc_count': {'$sum': 1}
            }},
            {'$sort': {'doc_count': 1}}  # Sort ascending (smallest first)
        ]
        
        results = list(self.documents.aggregate(pipeline))
        
        # Extract folder names
        folders = [r['_id'] for r in results]
        
        if VERBOSE:
            print("\nFolder sizes (first 10):")
            for r in results[:10]:
                print(f"  {r['_id']}: {r['doc_count']} documents")
        
        return folders
    
    def get_documents_for_person(self, person_folder):
        """Retrieve all documents for a person folder"""
        docs = list(self.documents.find({
            'person_folder': person_folder
        }).sort('relative_path', 1))
        return docs
    
    def create_synthesis_prompt(self, person_folder, documents, show_progress=True, batch_info=None):
        """Build comprehensive prompt for AI synthesis"""
        
        if not documents:
            return None
        
        # Extract metadata
        person_id = documents[0].get('person_id', 'Unknown')
        person_name = documents[0].get('person_name', 'Unknown')
        
        batch_label = f" - {batch_info}" if batch_info else ""
        
        prompt = f"""You are a historian analyzing Baltimore & Ohio Railroad employment records from the early 20th century.

EMPLOYEE FOLDER: {person_folder}
EMPLOYEE ID: {person_id}
FOLDER NAME: {person_name}
NUMBER OF DOCUMENTS: {len(documents)}{batch_label}

Below are ALL documents from this employee's personnel file. Read carefully and generate a comprehensive biographical synthesis.

"""
        
        # Add progress counter instead of per-doc print
        if show_progress:
            print(f"  Processing {len(documents)} documents...", end='', flush=True)
        
        # Add each document
        for i, doc in enumerate(documents, 1):
            # Simple progress dots every 50 docs
            if show_progress and i % 50 == 0:
                print(f" {i}", end='', flush=True)
            
            filename = doc.get('relative_path', '').split('/')[-1] or f"doc_{i}"
            ocr_text = doc.get('ocr_text', '')
            summary = doc.get('summary', '')
            
            # Truncate very long OCR text (more aggressive for large batches)
            if len(ocr_text) > 1000:
                ocr_text = ocr_text[:1000] + "\n... [truncated for length]"
            
            prompt += f"""
{'='*70}
DOCUMENT {i} of {len(documents)}: {filename}
{'='*70}

OCR TEXT:
{ocr_text}

SUMMARY:
{summary}

"""
        
        if show_progress:
            print()  # New line after progress
        
        prompt += """
{'='*70}
YOUR TASK: GENERATE COMPREHENSIVE BIOGRAPHICAL SYNTHESIS
{'='*70}

Create a structured JSON synthesis with the following schema. Be thorough and evidence-based.

{
  "person_identity": {
    "canonical_name": "Full official name based on most common/official usage",
    "name_variations": [
      {
        "name": "Variation as it appears",
        "confidence": "high/medium/low",
        "sources": ["list document filenames where this variation appears"]
      }
    ],
    "employee_id": "Employee ID number",
    "birth_year": 1895,
    "confidence_notes": "Brief assessment of identity certainty"
  },
  
  "biographical_narrative": "Write 2-4 paragraphs summarizing this person's railroad career. Include: when they started, positions held, major events (injuries, promotions, disputes), family situation, and any notable circumstances. Write in past tense as a professional historian would. Make it readable and informative.",
  
  "family": {
    "spouse": {
      "name": "Spouse name or null",
      "evidence": "How we know this",
      "sources": ["document filenames"]
    },
    "parents": [
      {
        "name": "Parent name",
        "relationship": "father/mother",
        "evidence": "How mentioned",
        "sources": ["document filenames"]
      }
    ],
    "children": []
  },
  
  "addresses": [
    {
      "address": "Full address as stated",
      "date_range": "When associated with this address or 'unknown'",
      "context": "Why this address appears (residence, beneficiary address, etc.)",
      "sources": ["document filenames"]
    }
  ],
  
  "employment_timeline": [
    {
      "date": "YYYY-MM-DD or best available",
      "event_type": "hire/application/promotion/transfer/furlough/resignation/injury",
      "position": "Job title",
      "department": "Department name or null",
      "division": "Division name or null",
      "details": "Brief description of what happened",
      "sources": ["document filenames"]
    }
  ],
  
  "injury_history": [
    {
      "injury_date": "YYYY-MM-DD or best available",
      "injury_type": "Description of injury",
      "body_part": "Affected body part(s)",
      "narrative": "2-3 sentence description of injury, treatment, and outcome",
      "medical_examinations": [
        {
          "date": "YYYY-MM-DD",
          "examiner": "Dr. Name",
          "finding": "disabled/not disabled/able to return/etc",
          "notes": "Key details from examination",
          "sources": ["document filenames"]
        }
      ],
      "payments": [
        {
          "period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
          "days": 10,
          "amount": 50.00,
          "check_number": "Check number if available",
          "sources": ["document filenames"]
        }
      ],
      "medical_dispute": {
        "disputed": true or false,
        "narrative": "If disputed, explain the conflict between doctors or administrators",
        "resolution": "How was it resolved or 'unclear'"
      }
    }
  ],
  
  "medical_examiners": [
    {
      "name": "Dr. Name",
      "district": "District number if available",
      "examinations": [
        {
          "date": "YYYY-MM-DD",
          "finding": "Brief finding"
        }
      ]
    }
  ],
  
  "administrative_events": [
    {
      "date": "YYYY-MM-DD or best available",
      "event": "What happened",
      "significance": "Why this matters",
      "sources": ["document filenames"]
    }
  ],
  
  "document_analysis": {
    "total_documents": 10,
    "date_range": {
      "earliest": "YYYY-MM-DD or best guess",
      "latest": "YYYY-MM-DD or best guess"
    },
    "completeness_assessment": "1-2 sentences: How complete is this record? What's well-documented? What's missing?",
    "data_quality_notes": [
      "List observations about OCR quality, contradictions, clarity, etc."
    ]
  },
  
  "historical_significance": {
    "key_insights": [
      "What does this case reveal about railroad labor, medical practices, administration, etc.?"
    ],
    "comparative_questions": [
      "What questions could be explored by comparing this case to others?"
    ],
    "research_value": "High/Medium/Low"
  },
  
  "confidence_scores": {
    "identity_resolution": 0.95,
    "employment_timeline": 0.85,
    "injury_details": 0.90,
    "overall": 0.88
  }
}

CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON - no preamble, no markdown fences, no explanation
2. Use exact schema above - include all fields even if some are empty/null
3. Always include "sources" arrays with actual document filenames for evidence
4. Be thorough but concise
5. Note contradictions or uncertainties explicitly
6. Assign realistic confidence scores (0.0-1.0) based on evidence quality
7. Write biographical_narrative in clear, professional prose
8. If information is missing, use null or empty arrays - don't invent data
9. For dates, use best available format (prefer YYYY-MM-DD, but accept what documents provide)
10. Document filenames should match the filename shown in DOCUMENT headers above

Begin your response with { and end with }. Return ONLY the JSON object.
"""
        
        return prompt
    
    def call_ollama(self, prompt, model=MODEL):
        """Call Ollama API for synthesis generation"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 6000,
                "top_p": 0.9
            }
        }
        
        try:
            if VERBOSE:
                print(f"    Calling Ollama (model: {model})...")
            
            response = requests.post(OLLAMA_URL, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            return result.get('response', '')
            
        except requests.exceptions.Timeout:
            print(f"    ⚠️  Ollama timeout after 300 seconds")
            return None
        except Exception as e:
            print(f"    ❌ Ollama error: {e}")
            return None
    
    def parse_synthesis(self, response):
        """Parse and validate synthesis JSON"""
        if not response:
            return None
        
        # Clean response
        response = response.strip()
        
        # Remove markdown code fences if present
        if response.startswith('```'):
            lines = response.split('\n')
            # Find first { and last }
            start_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('{'))
            end_idx = next(i for i in range(len(lines)-1, -1, -1) if lines[i].strip().endswith('}'))
            response = '\n'.join(lines[start_idx:end_idx+1])
        
        # Try to parse
        try:
            synthesis = json.loads(response)
            return synthesis
        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parse error: {e}")
            return None
    
    def save_synthesis_to_db(self, person_folder, person_id, documents, synthesis):
        """Save synthesis to person_syntheses collection"""
        
        synthesis_doc = {
            'person_folder': person_folder,
            'person_id': person_id,
            'person_name': documents[0].get('person_name'),
            'num_documents': len(documents),
            'synthesis': synthesis,
            'generated_date': datetime.now(),
            'model': MODEL,
            'version': '1.0'
        }
        
        if DRY_RUN:
            print(f"    [DRY RUN] Would save synthesis to person_syntheses collection")
            return 'dry-run-id'
        
        # Debug: Verify we're using the right database
        if VERBOSE:
            print(f"    Database: {self.db.name}")
            print(f"    Collection: {self.person_syntheses.name}")
        
        try:
            # Delete any existing first (like your test does)
            delete_result = self.person_syntheses.delete_many({'person_folder': person_folder})
            if VERBOSE and delete_result.deleted_count > 0:
                print(f"    Deleted {delete_result.deleted_count} existing syntheses")
            
            # Insert new
            insert_result = self.person_syntheses.insert_one(synthesis_doc)
            
            # Verify it saved
            found = self.person_syntheses.find_one({'_id': insert_result.inserted_id})
            if found:
                if VERBOSE:
                    print(f"    ✓ Verified synthesis saved (ID: {insert_result.inserted_id})")
                    
                # Double-check by querying by person_folder
                found_by_folder = self.person_syntheses.find_one({'person_folder': person_folder})
                if not found_by_folder:
                    print(f"    ⚠️  WARNING: Saved but can't find by person_folder!")
                    
                return insert_result.inserted_id
            else:
                print(f"    ❌ Error: Could not verify saved synthesis")
                return None
                
        except Exception as e:
            print(f"    ❌ Error saving synthesis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def add_synthesis_to_documents(self, person_folder, synthesis):
        """Add synthesis to each document belonging to this person"""
        
        if DRY_RUN:
            print(f"    [DRY RUN] Would add synthesis field to all documents for {person_folder}")
            return 0
        
        # Update all documents with this person_folder
        result = self.documents.update_many(
            {'person_folder': person_folder},
            {
                '$set': {
                    'person_synthesis': synthesis,
                    'synthesis_generated_date': datetime.now(),
                    'synthesis_version': '1.0'
                }
            }
        )
        
        if VERBOSE:
            print(f"    ✓ Added synthesis to {result.modified_count} documents")
        
        return result.modified_count
    
    def _single_synthesis(self, person_folder, documents):
        """Generate synthesis for a single batch of documents"""
        
        prompt = self.create_synthesis_prompt(person_folder, documents, show_progress=True)
        if not prompt:
            return None
        
        # Save prompt if debugging
        self._save_debug_file(person_folder, prompt, "prompt")
        
        print(f"  ✓ Prompt created ({len(prompt):,} characters)")
        print("  🤖 Generating synthesis with Ollama...")
        
        response = self.call_ollama(prompt)
        if not response:
            print(f"  ❌ No response from Ollama")
            return None
        
        # Save raw response if debugging
        self._save_debug_file(person_folder, response, "raw_response")
        
        print(f"  ✓ Received response ({len(response):,} characters)")
        
        synthesis = self.parse_synthesis(response)
        if not synthesis:
            print(f"  ❌ Failed to parse JSON")
            # Error case - always save for debugging (even if DEBUG=False)
            error_dir = Path("/home/claude/synthesis_errors")
            error_dir.mkdir(exist_ok=True, parents=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_folder = person_folder.replace('/', '_').replace(' ', '_')
            error_file = error_dir / f"{safe_folder}_{timestamp}_error.txt"
            error_file.write_text(response)
            print(f"     Raw response saved to: {error_file}")
            return None
        
        # Save parsed synthesis if debugging
        self._save_debug_file(person_folder, synthesis, "synthesis")
        
        print(f"  ✓ Successfully parsed JSON")
        return synthesis
    
    def _hierarchical_synthesis(self, person_folder, documents, batch_size):
        """Generate synthesis in batches, then synthesize the syntheses"""
        
        batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
        print(f"  📦 Split into {len(batches)} batches of ~{batch_size} documents")
        
        batch_syntheses = []
        
        # Synthesize each batch
        for i, batch in enumerate(batches, 1):
            print(f"\n  Batch {i}/{len(batches)} ({len(batch)} documents)...")
            
            prompt = self.create_synthesis_prompt(
                person_folder, 
                batch, 
                show_progress=False,  # Don't spam output
                batch_info=f"Batch {i}/{len(batches)}"
            )
            
            if not prompt:
                print(f"    ❌ Failed to create prompt")
                continue
            
            # Save batch prompt
            self._save_debug_file(person_folder, prompt, "prompt", batch_num=i)
            
            print(f"    Calling Ollama...")
            response = self.call_ollama(prompt)
            
            if not response:
                print(f"    ❌ No response")
                continue
            
            # Save batch response
            self._save_debug_file(person_folder, response, "raw_response", batch_num=i)
            
            synthesis = self.parse_synthesis(response)
            if synthesis:
                batch_syntheses.append(synthesis)
                # Save batch synthesis
                self._save_debug_file(person_folder, synthesis, "synthesis", batch_num=i)
                print(f"    ✓ Batch {i} complete")
            else:
                print(f"    ❌ Parse failed")
                # Save error
                error_dir = Path("/home/claude/synthesis_errors")
                error_dir.mkdir(exist_ok=True, parents=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_folder = person_folder.replace('/', '_').replace(' ', '_')
                error_file = error_dir / f"{safe_folder}_batch{i:02d}_{timestamp}_error.txt"
                error_file.write_text(response)
                print(f"    Raw response saved to: {error_file}")
        
        if not batch_syntheses:
            print(f"  ❌ No successful batch syntheses")
            return None
        
        print(f"\n  ✓ Completed {len(batch_syntheses)} batch syntheses")
        
        # Meta-synthesis: combine all batch syntheses
        print(f"  🔄 Creating meta-synthesis...")
        meta_synthesis = self._meta_synthesis(person_folder, batch_syntheses, len(documents))
        
        return meta_synthesis
    
    def _meta_synthesis(self, person_folder, batch_syntheses, total_docs):
        """Combine multiple batch syntheses into final synthesis"""
        
        prompt = f"""You are combining multiple batch syntheses for employee folder: {person_folder}

Total documents analyzed: {total_docs}
Number of batch syntheses: {len(batch_syntheses)}

Below are {len(batch_syntheses)} separate syntheses, each covering a portion of this person's records.
Your task is to merge them into ONE comprehensive synthesis following the same JSON schema.

"""
        
        for i, batch in enumerate(batch_syntheses, 1):
            prompt += f"\n{'='*70}\nBATCH SYNTHESIS {i}/{len(batch_syntheses)}\n{'='*70}\n"
            prompt += json.dumps(batch, indent=2, default=str)
            prompt += "\n"
        
        prompt += """

YOUR TASK: Merge these batch syntheses into ONE comprehensive synthesis.

Rules:
1. Combine all timeline events chronologically
2. Merge all injury records
3. Consolidate addresses, family members, medical examiners
4. Write a unified biographical narrative that covers the full career
5. Update confidence scores based on complete picture
6. Return ONLY valid JSON following the exact schema from the original prompts

Begin your response with { and end with }. Return ONLY the JSON object.
"""
        
        # Save meta-synthesis prompt
        self._save_debug_file(person_folder, prompt, "meta_prompt")
        
        print(f"    Meta-synthesis prompt: {len(prompt):,} characters")
        print("    Calling Ollama for final synthesis...")
        
        response = self.call_ollama(prompt)
        if not response:
            print(f"    ❌ No response")
            return None
        
        # Save meta-synthesis response
        self._save_debug_file(person_folder, response, "meta_raw_response")
        
        synthesis = self.parse_synthesis(response)
        if not synthesis:
            print(f"    ❌ Parse failed")
            # Save error
            error_dir = Path("/home/claude/synthesis_errors")
            error_dir.mkdir(exist_ok=True, parents=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_folder = person_folder.replace('/', '_').replace(' ', '_')
            error_file = error_dir / f"{safe_folder}_meta_{timestamp}_error.txt"
            error_file.write_text(response)
            print(f"    Raw response saved to: {error_file}")
            return None
        
        # Save final meta-synthesis
        self._save_debug_file(person_folder, synthesis, "meta_synthesis")
        
        print(f"    ✓ Meta-synthesis complete")
        return synthesis
    
    def synthesize_person(self, person_folder):
        """Generate and store synthesis for one person"""
        
        print(f"\n{'='*70}")
        print(f"Processing: {person_folder}")
        print(f"{'='*70}")
        
        # Check if already synthesized
        existing = self.person_syntheses.find_one({'person_folder': person_folder})
        if existing and not DRY_RUN:
            print(f"  ⏭️  Already synthesized (generated {existing.get('generated_date')})")
            print(f"     To regenerate, delete existing synthesis first")
            return {'status': 'skipped', 'reason': 'already_exists'}
        
        # Load documents
        print("  📄 Loading documents...")
        documents = self.get_documents_for_person(person_folder)
        
        if not documents:
            print(f"  ❌ No documents found")
            return {'status': 'error', 'reason': 'no_documents'}
        
        print(f"  ✓ Loaded {len(documents)} documents")
        
        # HIERARCHICAL SYNTHESIS
        BATCH_SIZE = 25  # Smaller batches = faster processing, less timeout risk
        
        if len(documents) <= BATCH_SIZE:
            # Small enough - single synthesis
            print(f"  📝 Creating synthesis prompt...")
            synthesis = self._single_synthesis(person_folder, documents)
        else:
            # Large - batch synthesis then meta-synthesis
            print(f"  📝 Large folder - using hierarchical synthesis")
            synthesis = self._hierarchical_synthesis(person_folder, documents, BATCH_SIZE)
        
        if not synthesis:
            return {'status': 'error', 'reason': 'synthesis_failed'}
        
        # Save to database
        print("  💾 Saving to database...")
        
        # Save to person_syntheses collection
        synthesis_id = self.save_synthesis_to_db(person_folder, documents[0].get('person_id'), documents, synthesis)
        
        if not synthesis_id:
            return {'status': 'error', 'reason': 'database_save_failed'}
        
        # Add synthesis to each document
        doc_count = self.add_synthesis_to_documents(person_folder, synthesis)
        
        print(f"  ✅ SUCCESS - Synthesis complete")
        
        return {
            'status': 'success',
            'person_folder': person_folder,
            'num_documents': len(documents),
            'documents_updated': doc_count,
            'synthesis': synthesis
        }
    
    def process_all_persons(self, max_persons=None):
        """Process all person folders"""
        
        print("="*70)
        print("PERSON SYNTHESIS SYSTEM")
        print("="*70)
        print(f"Mode: {'DRY RUN (no database changes)' if DRY_RUN else 'LIVE (will modify database)'}")
        print(f"Debug: {'ENABLED (saving prompts/responses)' if DEBUG else 'DISABLED'}")
        print(f"Limit: {max_persons if max_persons else 'No limit (process all)'}")
        print("="*70)
        
        # Get all person folders
        folders = self.get_all_person_folders()
        total_folders = len(folders)
        
        print(f"\nFound {total_folders} person folders")
        
        if max_persons:
            folders = folders[:max_persons]
            print(f"Processing first {len(folders)} folders (limited by MAX_PERSONS)")
        
        # Statistics
        stats = {
            'total': len(folders),
            'success': 0,
            'skipped': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Process each folder
        for i, folder in enumerate(folders, 1):
            print(f"\n[{i}/{len(folders)}]")
            
            result = self.synthesize_person(folder)
            
            if result['status'] == 'success':
                stats['success'] += 1
            elif result['status'] == 'skipped':
                stats['skipped'] += 1
            else:
                stats['errors'] += 1
            
            # Rate limiting - brief pause between persons
            if i < len(folders):
                time.sleep(1)
        
        # Final statistics
        elapsed = time.time() - stats['start_time']
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Total folders: {stats['total']}")
        print(f"  ✓ Success:   {stats['success']}")
        print(f"  ⏭️  Skipped:   {stats['skipped']}")
        print(f"  ❌ Errors:    {stats['errors']}")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Avg per person: {elapsed/stats['total']:.1f} seconds")
        print("="*70)
        
        if stats['success'] > 0:
            print(f"\n✓ Syntheses saved to database collection: {self.person_syntheses.name}")
            print(f"✓ Each person's documents now have 'person_synthesis' field")
            
            # Show example query
            print(f"\nExample MongoDB queries:")
            print(f"  # View a synthesis:")
            print(f"  db.person_syntheses.findOne({{'person_folder': '{folders[0]}'}})")
            print(f"  ")
            print(f"  # View a document with embedded synthesis:")
            print(f"  db.documents.findOne({{'person_folder': '{folders[0]}'}}, {{'person_synthesis': 1}})")
            print(f"  ")
            print(f"  # Find all people with medical disputes:")
            print(f"  db.person_syntheses.find({{'synthesis.injury_history.medical_dispute.disputed': true}})")
        
        return stats


def main():
    """Main entry point"""
    
    # Print startup diagnostics FIRST
    print_startup_diagnostics()
    
    # Check for command line flags
    global DRY_RUN, MAX_PERSONS, DEBUG
    
    if '--dry-run' in sys.argv:
        DRY_RUN = True
        print("🔍 DRY RUN MODE - No database changes will be made\n")
    
    if '--debug' in sys.argv:
        DEBUG = True
        print("🐛 DEBUG MODE - Saving prompts and responses\n")
    
    # Check for limit
    for arg in sys.argv:
        if arg.startswith('--limit='):
            MAX_PERSONS = int(arg.split('=')[1])
    
    # Initialize synthesizer
    try:
        print("Connecting to MongoDB...")
        synthesizer = PersonSynthesizer()
        print("✓ Connected to MongoDB")
        
        # Verify database and collections
        print(f"\nDatabase Verification:")
        print(f"  Connected to database: {synthesizer.db.name}")
        print(f"  Available collections: {', '.join(synthesizer.db.list_collection_names())}")
        
        # Check if we can count documents
        doc_count = synthesizer.documents.count_documents({})
        synth_count = synthesizer.person_syntheses.count_documents({})
        print(f"\nCollection Stats:")
        print(f"  Documents in 'documents': {doc_count:,}")
        print(f"  Documents in 'person_syntheses': {synth_count:,}")
        
        # Test write/read to verify permissions
        print(f"\nPermissions Check:")
        test_id = synthesizer.person_syntheses.insert_one({'_test': True, 'timestamp': datetime.now()})
        found = synthesizer.person_syntheses.find_one({'_id': test_id.inserted_id})
        synthesizer.person_syntheses.delete_one({'_id': test_id.inserted_id})
        if found:
            print(f"  ✓ Write/Read/Delete permissions verified")
        else:
            print(f"  ⚠️  Warning: Write succeeded but read failed!")
        
        print()  # Extra line before continuing
        
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure MongoDB is running:")
        print("  sudo systemctl start mongod")
        return 1
    
    # Check Ollama availability
    try:
        response = requests.get("http://host.docker.internal:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"✓ Ollama available, models: {', '.join(model_names)}")
            
            if MODEL not in model_names:
                print(f"⚠️  Warning: {MODEL} not found in available models")
                print(f"   You may need to: ollama pull {MODEL}")
        else:
            print(f"⚠️  Ollama responded but with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        return 1
    
    # Process all persons
    stats = synthesizer.process_all_persons(max_persons=MAX_PERSONS)
    
    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())