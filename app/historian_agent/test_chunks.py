#!/usr/bin/env python3
"""Check document_chunks schema to see available fields"""

from pymongo import MongoClient
import json

client = MongoClient("mongodb://admin:secret@mongodb:27017/admin")
db = client["railroad_documents"]
chunks_collection = db["document_chunks"]

print("="*70)
print("DOCUMENT_CHUNKS SCHEMA INSPECTION")
print("="*70)

# Get a sample chunk
sample = chunks_collection.find_one()

if not sample:
    print("No chunks found!")
    exit(1)

print("\nAvailable fields in chunks:")
print("-"*70)
for key in sorted(sample.keys()):
    value = sample[key]
    
    # Format preview based on type
    if isinstance(value, str):
        if len(value) > 100:
            preview = f'"{value[:100]}..."'
        else:
            preview = f'"{value}"'
    elif isinstance(value, (int, float)):
        preview = str(value)
    elif isinstance(value, dict):
        preview = f"{{...}} ({len(value)} keys)"
    elif isinstance(value, list):
        preview = f"[...] ({len(value)} items)"
    else:
        preview = str(type(value))
    
    print(f"  {key:20} : {preview}")

print("\n" + "="*70)
print("KEY FIELDS FOR DISPLAY:")
print("="*70)

# Check specific fields we care about
important_fields = ["filename", "relative_path", "parent_doc_id", "document_id", 
                   "chunk_id", "title", "ocr_text", "summary"]

for field in important_fields:
    if field in sample:
        value = sample[field]
        if isinstance(value, str):
            preview = value[:80] + "..." if len(value) > 80 else value
        else:
            preview = str(value)
        print(f"  ✓ {field}: {preview}")
    else:
        print(f"  ✗ {field}: NOT FOUND")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)

# Check if we need to look up parent document
if "parent_doc_id" in sample or "document_id" in sample:
    parent_id = sample.get("parent_doc_id") or sample.get("document_id")
    print(f"\nChunks reference parent via: {parent_id}")
    print("Checking parent documents collection...")
    
    docs_collection = db["documents"]
    parent_doc = docs_collection.find_one({"_id": parent_id})
    
    if parent_doc:
        print("\nParent document fields:")
        for field in ["filename", "relative_path", "title", "name"]:
            if field in parent_doc:
                value = parent_doc[field]
                preview = value[:80] + "..." if isinstance(value, str) and len(value) > 80 else value
                print(f"  ✓ {field}: {preview}")
    else:
        print("  ✗ Parent document not found")
        print(f"\nYou may need to populate parent_doc_id or copy filename to chunks")

client.close()