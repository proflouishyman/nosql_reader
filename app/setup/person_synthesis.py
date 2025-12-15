#!/usr/bin/env python3
"""
Fixed Person Synthesis System - with better error handling and save verification
"""
import os
import json
import requests
import time
from datetime import datetime
from pymongo import MongoClient
from pathlib import Path
import sys
import traceback

# Configuration
MONGO_URI = os.environ.get('APP_MONGO_URI') or os.environ.get('MONGO_URI')
DB_NAME = 'railroad_documents'
OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL = "llama3.1:8b"

# Control flags
DRY_RUN = False
MAX_PERSONS = 3  # Process just 3 for testing
VERBOSE = True
SAVE_RAW_RESPONSES = True  # Save Ollama responses for debugging

class PersonSynthesizer:
    """Generate and store person syntheses"""
    
    def __init__(self, mongo_uri=MONGO_URI, db_name=DB_NAME):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.documents = self.db['documents']
        self.person_syntheses = self.db['person_syntheses']
        
    def synthesize_person(self, person_folder):
        """Generate and store synthesis for one person"""
        
        print(f"\n{'='*70}")
        print(f"Processing: {person_folder}")
        print(f"{'='*70}")
        
        try:
            # Check if already synthesized
            existing = self.person_syntheses.find_one({'person_folder': person_folder})
            if existing and not DRY_RUN:
                print(f"  ⏭️  Already synthesized (generated {existing.get('generated_date')})")
                return {'status': 'skipped', 'reason': 'already_exists'}
            
            # Load documents
            print("  📄 Loading documents...")
            documents = list(self.documents.find({'person_folder': person_folder}).sort('relative_path', 1))
            
            if not documents:
                print(f"  ❌ No documents found for {person_folder}")
                return {'status': 'error', 'reason': 'no_documents'}
            
            print(f"  ✓ Loaded {len(documents)} documents")
            person_id = documents[0].get('person_id', 'Unknown')
            person_name = documents[0].get('person_name', 'Unknown')
            
            # Create prompt (using your existing create_synthesis_prompt method)
            print("  📝 Creating synthesis prompt...")
            prompt = self.create_synthesis_prompt(person_folder, documents)
            
            if not prompt:
                print(f"  ❌ Failed to create prompt")
                return {'status': 'error', 'reason': 'prompt_creation_failed'}
            
            print(f"  ✓ Prompt created ({len(prompt):,} characters)")
            
            # Save prompt for debugging
            if SAVE_RAW_RESPONSES:
                debug_dir = Path("/app/synthesis_debug")
                debug_dir.mkdir(exist_ok=True)
                prompt_file = debug_dir / f"{person_folder}_prompt.txt"
                prompt_file.write_text(prompt)
                print(f"  💾 Saved prompt to: {prompt_file}")
            
            # Call Ollama
            print("  🤖 Generating synthesis with Ollama...")
            response = self.call_ollama(prompt)
            
            if not response:
                print(f"  ❌ No response from Ollama")
                return {'status': 'error', 'reason': 'ollama_no_response'}
            
            print(f"  ✓ Received response ({len(response):,} characters)")
            
            # Save raw response for debugging
            if SAVE_RAW_RESPONSES:
                response_file = debug_dir / f"{person_folder}_response.txt"
                response_file.write_text(response)
                print(f"  💾 Saved raw response to: {response_file}")
            
            # Parse JSON
            print("  🔍 Parsing JSON...")
            synthesis = self.parse_synthesis(response)
            
            if not synthesis:
                print(f"  ❌ Failed to parse JSON")
                return {'status': 'error', 'reason': 'json_parse_failed'}
            
            print(f"  ✓ Successfully parsed JSON")
            
            # Save to database with explicit verification
            print("  💾 Saving to database...")
            
            # Prepare the document
            synthesis_doc = {
                'person_folder': person_folder,
                'person_id': person_id,
                'person_name': person_name,
                'num_documents': len(documents),
                'synthesis': synthesis,
                'generated_date': datetime.now(),
                'model': MODEL,
                'version': '1.0'
            }
            
            if DRY_RUN:
                print("  [DRY RUN] Would save synthesis")
                return {'status': 'success', 'synthesis': synthesis}
            
            # Delete any existing synthesis for this person
            delete_result = self.person_syntheses.delete_many({'person_folder': person_folder})
            if delete_result.deleted_count > 0:
                print(f"  🗑️  Deleted {delete_result.deleted_count} existing synthesis records")
            
            # Insert the new synthesis
            insert_result = self.person_syntheses.insert_one(synthesis_doc)
            print(f"  ✓ Inserted synthesis with ID: {insert_result.inserted_id}")
            
            # Verify it was saved
            verification = self.person_syntheses.find_one({'_id': insert_result.inserted_id})
            if not verification:
                print(f"  ❌ ERROR: Synthesis was not saved properly!")
                return {'status': 'error', 'reason': 'save_verification_failed'}
            
            print(f"  ✓ Verified: Synthesis saved successfully")
            
            # Add synthesis to documents
            update_result = self.documents.update_many(
                {'person_folder': person_folder},
                {
                    '$set': {
                        'person_synthesis': synthesis,
                        'synthesis_generated_date': datetime.now(),
                        'synthesis_version': '1.0'
                    }
                }
            )
            
            print(f"  ✓ Updated {update_result.modified_count} documents with synthesis")
            
            # Final verification
            final_check = self.person_syntheses.count_documents({'person_folder': person_folder})
            if final_check != 1:
                print(f"  ⚠️  WARNING: Expected 1 synthesis, found {final_check}")
            
            print(f"  ✅ SUCCESS - Synthesis complete and verified")
            
            return {
                'status': 'success',
                'person_folder': person_folder,
                'num_documents': len(documents),
                'synthesis_id': str(insert_result.inserted_id),
                'documents_updated': update_result.modified_count
            }
            
        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            traceback.print_exc()
            return {'status': 'error', 'reason': 'exception', 'error': str(e)}
    
    def create_synthesis_prompt(self, person_folder, documents):
        """Your existing prompt creation method"""
        # Copy your existing create_synthesis_prompt method here
        # I'm abbreviating for space
        if not documents:
            return None
        
        person_id = documents[0].get('person_id', 'Unknown')
        person_name = documents[0].get('person_name', 'Unknown')
        
        prompt = f"""You are a historian analyzing Baltimore & Ohio Railroad employment records.

EMPLOYEE FOLDER: {person_folder}
EMPLOYEE ID: {person_id}
FOLDER NAME: {person_name}
NUMBER OF DOCUMENTS: {len(documents)}

[Rest of your prompt template...]

Return ONLY valid JSON starting with {{ and ending with }}.
"""
        return prompt
    
    def call_ollama(self, prompt, model=MODEL):
        """Your existing Ollama call method"""
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
            response = requests.post(OLLAMA_URL, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            print(f"    ❌ Ollama error: {e}")
            return None
    
    def parse_synthesis(self, response):
        """Your existing parse method with better error handling"""
        if not response:
            return None
        
        response = response.strip()
        
        # Remove markdown code fences
        if response.startswith('```'):
            lines = response.split('\n')
            start_idx = None
            end_idx = None
            
            for i, line in enumerate(lines):
                if line.strip().startswith('{') and start_idx is None:
                    start_idx = i
                if line.strip().endswith('}'):
                    end_idx = i
            
            if start_idx is not None and end_idx is not None:
                response = '\n'.join(lines[start_idx:end_idx+1])
        
        try:
            synthesis = json.loads(response)
            return synthesis
        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parse error: {e}")
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                try:
                    synthesis = json.loads(response[start:end+1])
                    print(f"    ✓ Recovered JSON from response")
                    return synthesis
                except:
                    pass
            return None
    
    def verify_all_syntheses(self):
        """Verify what syntheses are actually in the database"""
        print("\n" + "="*70)
        print("SYNTHESIS VERIFICATION")
        print("="*70)
        
        total = self.person_syntheses.count_documents({})
        print(f"Total syntheses in database: {total}")
        
        syntheses = list(self.person_syntheses.find({}, {'person_folder': 1, 'generated_date': 1}).limit(10))
        
        if syntheses:
            print("\nExisting syntheses:")
            for syn in syntheses:
                print(f"  - {syn['person_folder']}: {syn['generated_date']}")
        
        return total

def main():
    """Main entry point"""
    print("="*70)
    print("PERSON SYNTHESIS SYSTEM - FIXED VERSION")
    print("="*70)
    
    synthesizer = PersonSynthesizer()
    
    # First verify current state
    current_count = synthesizer.verify_all_syntheses()
    
    # Get some person folders to process
    all_folders = synthesizer.documents.distinct('person_folder')
    folders_with_docs = []
    
    for folder in all_folders:
        if folder:  # Skip None/empty
            doc_count = synthesizer.documents.count_documents({'person_folder': folder})
            if doc_count > 0:
                folders_with_docs.append((folder, doc_count))
    
    # Sort by document count and take top 3
    folders_with_docs.sort(key=lambda x: x[1], reverse=False) #to process those with fewer documents first set False, True for more documents first
    
    print(f"\nProcessing {MAX_PERSONS} person folders with most documents:")
    for folder, count in folders_with_docs[:MAX_PERSONS]:
        print(f"  - {folder}: {count} documents")
    
    # Process them
    stats = {
        'success': 0,
        'errors': 0,
        'skipped': 0
    }
    
    for folder, _ in folders_with_docs[:MAX_PERSONS]:
        result = synthesizer.synthesize_person(folder)
        
        if result['status'] == 'success':
            stats['success'] += 1
        elif result['status'] == 'skipped':
            stats['skipped'] += 1
        else:
            stats['errors'] += 1
    
    # Final verification
    final_count = synthesizer.verify_all_syntheses()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Syntheses before: {current_count}")
    print(f"Syntheses after: {final_count}")
    print(f"New syntheses created: {final_count - current_count}")
    print(f"Success: {stats['success']}, Errors: {stats['errors']}, Skipped: {stats['skipped']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())