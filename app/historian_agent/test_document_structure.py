#!/usr/bin/env python3
"""
Diagnostic script to understand document structure in MongoDB.
Run this inside the Docker container to see what fields your documents actually have.
"""

from pymongo import MongoClient
import os
import sys

# Configuration - load from environment like the main app
MONGO_URI = os.environ.get('APP_MONGO_URI') or os.environ.get('MONGO_URI') or "mongodb://admin:secret@mongodb:27017/admin"
DB_NAME = 'railroad_documents'

def main():
    print("=" * 70)
    print("DOCUMENT STRUCTURE DIAGNOSTIC")
    print("=" * 70)
    print(f"\nMongoDB URI: {MONGO_URI[:60]}...")
    print(f"Database: {DB_NAME}")
    
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    docs_collection = db['documents']
    
    # Total count
    total = docs_collection.count_documents({})
    print(f"\nTotal documents: {total:,}")
    
    # Check for various text fields
    print("\n" + "=" * 70)
    print("TEXT FIELD AVAILABILITY")
    print("=" * 70)
    
    fields_to_check = [
        ("content", {"content": {"$exists": True, "$ne": ""}}),
        ("ocr_text", {"ocr_text": {"$exists": True, "$ne": ""}}),
        ("summary", {"summary": {"$exists": True, "$ne": ""}}),
        ("description", {"description": {"$exists": True, "$ne": ""}}),
        ("title", {"title": {"$exists": True, "$ne": ""}}),
        ("image_metadata", {"image_metadata": {"$exists": True}}),
        ("image_metadata.structured_content.ocr_text", 
         {"image_metadata.structured_content.ocr_text": {"$exists": True, "$ne": ""}}),
    ]
    
    for field_name, query in fields_to_check:
        count = docs_collection.count_documents(query)
        pct = (count / total * 100) if total > 0 else 0
        print(f"{field_name:50s}: {count:6,} ({pct:5.1f}%)")
    
    # Sample documents
    print("\n" + "=" * 70)
    print("SAMPLE DOCUMENTS")
    print("=" * 70)
    
    # Get one document with text in any field
    doc_with_text = docs_collection.find_one({"$or": [
        {"content": {"$exists": True, "$ne": ""}},
        {"ocr_text": {"$exists": True, "$ne": ""}},
        {"summary": {"$exists": True, "$ne": ""}},
        {"image_metadata.structured_content.ocr_text": {"$exists": True, "$ne": ""}},
    ]})
    
    if doc_with_text:
        print(f"\n Document ID: {doc_with_text['_id']}")
        print(f"All fields: {list(doc_with_text.keys())}")
        
        # Check text fields
        text_fields = ['content', 'ocr_text', 'summary', 'description', 'title']
        print("\nTop-level text fields:")
        for field in text_fields:
            val = doc_with_text.get(field)
            if val:
                if isinstance(val, str):
                    print(f"  {field}: {len(val):,} chars - '{val[:80]}...'")
                else:
                    print(f"  {field}: {type(val).__name__}")
            else:
                print(f"  {field}: (not present or empty)")
        
        # Check image_metadata
        if 'image_metadata' in doc_with_text:
            img_meta = doc_with_text['image_metadata']
            print(f"\nimage_metadata present: {type(img_meta).__name__}")
            if isinstance(img_meta, dict):
                print(f"  image_metadata keys: {list(img_meta.keys())}")
                if 'structured_content' in img_meta:
                    struct = img_meta['structured_content']
                    print(f"  structured_content type: {type(struct).__name__}")
                    if isinstance(struct, dict):
                        print(f"  structured_content keys: {list(struct.keys())}")
                        if 'ocr_text' in struct:
                            ocr = struct['ocr_text']
                            if isinstance(ocr, str):
                                print(f"  structured_content.ocr_text: {len(ocr):,} chars")
    else:
        print("\n⚠️  NO DOCUMENTS WITH TEXT FOUND")
    
    # Sample a few more random documents
    print("\n" + "=" * 70)
    print("RANDOM SAMPLE (5 documents)")
    print("=" * 70)
    for i, doc in enumerate(docs_collection.find().limit(5), 1):
        print(f"\n{i}. Document {doc['_id']}")
        print(f"   Fields: {list(doc.keys())}")
        has_text = False
        for field in ['content', 'ocr_text', 'summary', 'description']:
            if doc.get(field):
                has_text = True
                val = doc[field]
                if isinstance(val, str):
                    print(f"   {field}: {len(val)} chars")
        if not has_text:
            print("   ⚠️  No text in standard fields")
            if 'image_metadata' in doc:
                print("   Has image_metadata")
    
    client.close()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)