#!/usr/bin/env python3
"""
Fix imports in RAG files for historian_agent subdirectory structure

Usage:
    python fix_rag_imports.py /path/to/app/historian_agent/

This script updates the three RAG files to use relative imports
so they work correctly when placed in the historian_agent subdirectory.
"""

import sys
import os
from pathlib import Path


def fix_rag_query_handler(filepath):
    """Fix imports in rag_query_handler.py"""
    print(f"Fixing imports in {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace absolute imports with relative imports
    replacements = [
        ('from embeddings import', 'from .embeddings import'),
        ('from vector_store import', 'from .vector_store import'),
        ('from retrievers import', 'from .retrievers import'),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ Replaced: {old} → {new}")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✓ {filepath} updated")


def fix_adversarial_rag(filepath):
    """Fix imports in adversarial_rag.py"""
    print(f"Fixing imports in {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace absolute imports with relative imports
    replacements = [
        ('from rag_query_handler import', 'from .rag_query_handler import'),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ Replaced: {old} → {new}")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✓ {filepath} updated")


def fix_iterative_adversarial_agent(filepath):
    """Fix imports in iterative_adversarial_agent.py"""
    print(f"Fixing imports in {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace absolute imports with relative imports
    replacements = [
        ('from rag_query_handler import', 'from .rag_query_handler import'),
        ('from reranking import', 'from .reranking import'),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ Replaced: {old} → {new}")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"  ✓ {filepath} updated")


def ensure_init_file(directory):
    """Ensure __init__.py exists in the directory"""
    init_file = Path(directory) / '__init__.py'
    
    if not init_file.exists():
        print(f"Creating {init_file}...")
        init_file.touch()
        print(f"  ✓ Created __init__.py")
    else:
        print(f"  ✓ __init__.py already exists")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_rag_imports.py /path/to/app/historian_agent/")
        sys.exit(1)
    
    historian_agent_dir = Path(sys.argv[1])
    
    if not historian_agent_dir.exists():
        print(f"Error: Directory {historian_agent_dir} does not exist")
        sys.exit(1)
    
    print(f"\n=== Fixing RAG imports for historian_agent directory ===")
    print(f"Directory: {historian_agent_dir}\n")
    
    # Ensure __init__.py exists
    ensure_init_file(historian_agent_dir)
    print()
    
    # Fix each file
    files_to_fix = [
        ('rag_query_handler.py', fix_rag_query_handler),
        ('adversarial_rag.py', fix_adversarial_rag),
        ('iterative_adversarial_agent.py', fix_iterative_adversarial_agent),
    ]
    
    for filename, fix_func in files_to_fix:
        filepath = historian_agent_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  Warning: {filepath} not found - skipping")
            continue
        
        try:
            fix_func(filepath)
        except Exception as e:
            print(f"❌ Error fixing {filepath}: {e}")
        
        print()
    
    print("=== Import fixing complete ===\n")
    print("Next steps:")
    print("1. Copy the 5 route functions from historian_rag_routes.py to app/routes.py")
    print("2. Add imports to app/routes.py:")
    print("   from historian_agent.rag_query_handler import RAGQueryHandler")
    print("   from historian_agent.adversarial_rag import AdversarialRAGHandler")
    print("   from historian_agent.iterative_adversarial_agent import TieredHistorianAgent")
    print("3. Restart Flask: docker compose restart flask_app")
    print("4. Test: curl -X POST http://localhost:5006/historian-agent/query-basic \\")
    print("         -H 'Content-Type: application/json' -d '{\"question\": \"test\"}'")


if __name__ == '__main__':
    main()