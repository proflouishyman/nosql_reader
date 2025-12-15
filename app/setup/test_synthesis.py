#!/usr/bin/env python3
"""
Fixed version of synthesis saving - ensures data is actually written
"""
import os
from pymongo import MongoClient
from datetime import datetime

MONGO_URI = os.environ.get('APP_MONGO_URI') or os.environ.get('MONGO_URI')
DB_NAME = 'railroad_documents'

def verify_and_fix_syntheses():
    """Check and fix synthesis saving"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    person_syntheses = db['person_syntheses']
    documents = db['documents']
    
    print("CHECKING SYNTHESIS SAVE ISSUES")
    print("="*80)
    
    # First, let's check what person folders have documents but no synthesis
    print("\n1. PERSON FOLDERS WITH DOCUMENTS BUT NO SYNTHESIS:")
    
    # Get all person folders from documents
    all_folders = documents.distinct('person_folder')
    
    # Get all person folders that have syntheses
    synthesized_folders = person_syntheses.distinct('person_folder')
    
    # Find the difference
    missing_syntheses = [f for f in all_folders if f and f not in synthesized_folders]
    
    print(f"   Total person folders: {len(all_folders)}")
    print(f"   Folders with syntheses: {len(synthesized_folders)}")
    print(f"   Missing syntheses: {len(missing_syntheses)}")
    
    # Show some examples
    if missing_syntheses:
        print(f"\n   Examples of missing syntheses:")
        for folder in missing_syntheses[:10]:
            doc_count = documents.count_documents({'person_folder': folder})
            print(f"   - {folder}: {doc_count} documents")
    
    # Check if 205285-Abrell has documents
    print("\n2. CHECKING SPECIFIC CASE: 205285-Abrell")
    abrell_docs = documents.count_documents({'person_folder': '205285-Abrell'})
    print(f"   Documents found: {abrell_docs}")
    
    if abrell_docs == 0:
        # Maybe the person_folder value is different?
        print("   Searching for similar folders...")
        similar = documents.distinct('person_folder', {'person_folder': {'$regex': 'Abrell', '$options': 'i'}})
        for s in similar:
            count = documents.count_documents({'person_folder': s})
            print(f"   - Found: '{s}' with {count} documents")
    
    # Let's also check the documents collection structure
    print("\n3. SAMPLE DOCUMENT STRUCTURE:")
    sample = documents.find_one({'person_folder': {'$exists': True, '$ne': None}})
    if sample:
        print(f"   Person folder: {sample.get('person_folder')}")
        print(f"   Person ID: {sample.get('person_id')}")
        print(f"   Person name: {sample.get('person_name')}")
        print(f"   Has synthesis: {'person_synthesis' in sample}")
    
    # Test saving a synthesis directly
    print("\n4. TESTING DIRECT SYNTHESIS SAVE:")
    
    test_synthesis = {
        'person_folder': 'TEST-FOLDER',
        'person_id': 'TEST-ID',
        'person_name': 'Test Person',
        'num_documents': 1,
        'synthesis': {
            'test': True,
            'message': 'This is a test synthesis'
        },
        'generated_date': datetime.now(),
        'model': 'test',
        'version': 'test'
    }
    
    # Try insert
    try:
        result = person_syntheses.insert_one(test_synthesis)
        print(f"   ✓ Insert successful: {result.inserted_id}")
        
        # Verify it's there
        found = person_syntheses.find_one({'_id': result.inserted_id})
        if found:
            print("   ✓ Verified: Document exists in database")
        
        # Clean up
        person_syntheses.delete_one({'_id': result.inserted_id})
        print("   ✓ Cleanup complete")
        
    except Exception as e:
        print(f"   ❌ Insert failed: {e}")
    
    # Try upsert
    print("\n5. TESTING UPSERT OPERATION:")
    
    try:
        # First insert
        result1 = person_syntheses.update_one(
            {'person_folder': 'TEST-UPSERT'},
            {'$set': {
                'person_folder': 'TEST-UPSERT',
                'synthesis': {'version': 1},
                'generated_date': datetime.now()
            }},
            upsert=True
        )
        print(f"   First upsert - Inserted: {result1.upserted_id}, Modified: {result1.modified_count}")
        
        # Second update
        result2 = person_syntheses.update_one(
            {'person_folder': 'TEST-UPSERT'},
            {'$set': {
                'synthesis': {'version': 2},
                'generated_date': datetime.now()
            }},
            upsert=True
        )
        print(f"   Second upsert - Inserted: {result2.upserted_id}, Modified: {result2.modified_count}")
        
        # Verify
        found = person_syntheses.find_one({'person_folder': 'TEST-UPSERT'})
        if found:
            print(f"   ✓ Document exists with version: {found.get('synthesis', {}).get('version')}")
        
        # Clean up
        person_syntheses.delete_one({'person_folder': 'TEST-UPSERT'})
        
    except Exception as e:
        print(f"   ❌ Upsert test failed: {e}")
    
    print("\n" + "="*80)

def force_save_synthesis(person_folder, synthesis_data):
    """Force save a synthesis with explicit error checking"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    person_syntheses = db['person_syntheses']
    
    try:
        # Delete any existing
        delete_result = person_syntheses.delete_many({'person_folder': person_folder})
        print(f"Deleted {delete_result.deleted_count} existing syntheses for {person_folder}")
        
        # Insert new
        insert_result = person_syntheses.insert_one(synthesis_data)
        print(f"Inserted new synthesis with ID: {insert_result.inserted_id}")
        
        # Verify
        found = person_syntheses.find_one({'_id': insert_result.inserted_id})
        if found:
            print("✓ Verified: Synthesis saved successfully")
            return True
        else:
            print("❌ Error: Could not verify saved synthesis")
            return False
            
    except Exception as e:
        print(f"❌ Error saving synthesis: {e}")
        return False

if __name__ == "__main__":
    verify_and_fix_syntheses()