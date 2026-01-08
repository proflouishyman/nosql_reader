#!/usr/bin/env python3
"""
Enhanced field analysis script that dives deeper into nested MongoDB structures.
"""

from pymongo import MongoClient
import os
import json
from collections import defaultdict

MONGO_URI = os.environ.get('APP_MONGO_URI') or os.environ.get('MONGO_URI') or "mongodb://admin:secret@mongodb:27017/admin"
DB_NAME = 'railroad_documents'
COLLECTION_NAME = 'documents'
N = 10  # Max unique values to qualify as dropdown

def get_nested_field_names(collection):
    """Get all nested field names from documents."""
    sample_docs = list(collection.find().limit(100))
    
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                items.append((new_key, "array_of_objects"))
            else:
                items.append((new_key, type(v).__name__))
        return dict(items)
    
    field_names = set()
    for doc in sample_docs:
        flat_fields = flatten_dict(doc)
        field_names.update(flat_fields.keys())
    
    return sorted(field_names)

def analyze_field_uniqueness(collection, field_name):
    """Analyze uniqueness for a specific field."""
    pipeline = [
        {"$match": {field_name: {"$exists": True, "$ne": ""}}},
        {"$group": {"_id": f"${field_name}"}},
        {"$count": "unique_values"}
    ]
    
    try:
        result = list(collection.aggregate(pipeline))
        count = result[0]['unique_values'] if result else 0
        return count
    except Exception as e:
        print(f"Error analyzing {field_name}: {e}")
        return 0

def get_sample_values(collection, field_name, limit=5):
    """Get sample values for a field."""
    pipeline = [
        {"$match": {field_name: {"$exists": True, "$ne": ""}}},
        {"$group": {"_id": f"${field_name}"}},
        {"$sort": {"_id": 1}},
        {"$limit": limit}
    ]
    
    try:
        result = list(collection.aggregate(pipeline))
        return [doc['_id'] for doc in result]
    except Exception as e:
        print(f"Error getting values for {field_name}: {e}")
        return []

def analyze_nested_fields():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    print("=" * 70)
    print("DEEP FIELD ANALYSIS")
    print("=" * 70)

    # Get all field names including nested ones
    field_names = get_nested_field_names(collection)
    
    # Filter to focus on likely dropdown candidates
    target_fields = [f for f in field_names if not f.startswith('_id') and 
                     'image_metadata' not in f and 'structured_content' not in f]
    
    print(f"Analyzing {len(target_fields)} fields...")
    
    # Analyze each field
    dropdown_fields = []
    for field in target_fields[:20]:  # Limit to first 20 to avoid overload
        count = analyze_field_uniqueness(collection, field)
        
        if count <= N:
            values = get_sample_values(collection, field)
            print(f"{field:40} | {count:3} unique values")
            if count <= 5:
                print(f"  Values: {values}")
            dropdown_fields.append((field, count, values))
        else:
            print(f"{field:40} | {count:3} unique values (too many for dropdown)")

    print("\n" + "=" * 70)
    print("DEEP STRUCTURE ANALYSIS")
    print("=" * 70)

    # Analyze specific nested structures
    nested_fields = [
        "archive_structure.series",
        "archive_structure.format", 
        "archive_structure.physical_box",
        "metadata.migrated_from"
    ]
    
    for field in nested_fields:
        count = analyze_field_uniqueness(collection, field)
        print(f"{field:35} | {count:4} unique values")
        
        if count <= N:
            values = get_sample_values(collection, field)
            print(f"  Values: {values}")

    # Look for array fields with unique elements
    print("\n" + "=" * 70)
    print("ARRAY FIELD ANALYSIS")
    print("=" * 70)
    
    array_fields = ["sections", "archive_structure.path_components"]
    for field in array_fields:
        try:
            pipeline = [
                {"$match": {field: {"$exists": True}}},
                {"$project": {"array_length": {"$size": f"${field}"}}},
                {"$group": {"_id": "$array_length", "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}}
            ]
            result = list(collection.aggregate(pipeline))
            print(f"{field:30} | Array sizes: {result}")
        except Exception as e:
            print(f"Array analysis failed for {field}: {e}")

    client.close()
    
    # Summary of dropdown candidates
    print("\n" + "=" * 70)
    print("RECOMMENDED DROPDOWN FIELDS")
    print("=" * 70)
    for field, count, values in dropdown_fields:
        print(f"✓ {field} ({count} values)")
        if count <= 5:
            print(f"  Dropdown options: {values}")

if __name__ == "__main__":
    analyze_nested_fields()
