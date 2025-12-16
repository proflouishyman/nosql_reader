#!/usr/bin/env python3
"""
ChromaDB Health Check

Diagnose potential ChromaDB issues causing segfaults.
"""

import os
import sys
from pathlib import Path

def check_disk_space():
    """Check available disk space."""
    print("\n📊 Disk Space Check")
    print("=" * 60)
    try:
        import shutil
        chroma_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY", "/data/chroma_db/persist")
        total, used, free = shutil.disk_usage(chroma_dir)
        
        print(f"ChromaDB directory: {chroma_dir}")
        print(f"Total: {total // (2**30)} GB")
        print(f"Used:  {used // (2**30)} GB")
        print(f"Free:  {free // (2**30)} GB")
        print(f"Free %: {(free/total*100):.1f}%")
        
        if free < 1 * (2**30):  # Less than 1GB
            print("⚠️  WARNING: Low disk space!")
            return False
        else:
            print("✅ Disk space OK")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_chroma_files():
    """Check ChromaDB file integrity."""
    print("\n📁 ChromaDB Files Check")
    print("=" * 60)
    try:
        chroma_dir = Path(os.environ.get("CHROMA_PERSIST_DIRECTORY", "/data/chroma_db/persist"))
        
        if not chroma_dir.exists():
            print(f"❌ ChromaDB directory doesn't exist: {chroma_dir}")
            return False
        
        # Check for SQLite files
        db_files = list(chroma_dir.rglob("*.sqlite*"))
        print(f"Found {len(db_files)} SQLite files")
        
        for db_file in db_files:
            size = db_file.stat().st_size
            print(f"  {db_file.name}: {size:,} bytes")
            
            # Check if file is accessible
            try:
                with open(db_file, 'rb') as f:
                    f.read(1)
                print(f"    ✅ Readable")
            except Exception as e:
                print(f"    ❌ Not readable: {e}")
                return False
        
        print("✅ All ChromaDB files accessible")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_chroma_connection():
    """Test basic ChromaDB connection."""
    print("\n🔌 ChromaDB Connection Test")
    print("=" * 60)
    try:
        import chromadb
        from chromadb.config import Settings
        
        persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY", "/data/chroma_db/persist")
        
        print(f"Connecting to: {persist_dir}")
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        print("✅ Client created")
        
        # Try to get collection
        collection = client.get_collection("historian_documents")
        count = collection.count()
        
        print(f"✅ Collection accessed: {count:,} vectors")
        
        # Try a simple query (without embeddings to avoid segfault)
        try:
            results = collection.get(limit=1, include=["metadatas"])
            print(f"✅ Read operation successful")
            return True
        except Exception as e:
            print(f"⚠️  Read operation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_memory():
    """Check available memory."""
    print("\n💾 Memory Check")
    print("=" * 60)
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        
        mem_info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':')
                mem_info[key.strip()] = value.strip()
        
        mem_total = int(mem_info['MemTotal'].split()[0]) // 1024  # MB
        mem_avail = int(mem_info['MemAvailable'].split()[0]) // 1024  # MB
        
        print(f"Total Memory: {mem_total} MB")
        print(f"Available:    {mem_avail} MB")
        print(f"Available %:  {(mem_avail/mem_total*100):.1f}%")
        
        if mem_avail < 500:  # Less than 500MB
            print("⚠️  WARNING: Low memory!")
            return False
        else:
            print("✅ Memory OK")
            return True
            
    except Exception as e:
        print(f"⚠️  Could not check memory: {e}")
        return True  # Don't fail on this


def suggest_fixes():
    """Suggest fixes based on common issues."""
    print("\n🔧 Suggested Fixes")
    print("=" * 60)
    print("""
1. Reset ChromaDB (if corrupted):
   docker compose down
   rm -rf /path/to/chroma_db/persist/*
   docker compose up -d
   docker compose exec app python scripts/setup_rag_database.py

2. If disk is full:
   - Clean up old Docker images: docker system prune -a
   - Free up disk space on host

3. If memory is low:
   - Reduce batch size in tests
   - Restart Docker containers: docker compose restart

4. If concurrent access issues:
   - Ensure only one process accesses ChromaDB at a time
   - Stop other running processes

5. Try minimal test:
   docker compose exec app python -c "
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(path='/data/chroma_db/persist')
collection = client.get_collection('historian_documents')
print(f'Count: {collection.count()}')
"
""")


def main():
    print("=" * 60)
    print("ChromaDB Health Diagnostic")
    print("=" * 60)
    
    checks = [
        ("Disk Space", check_disk_space),
        ("ChromaDB Files", check_chroma_files),
        ("Memory", check_memory),
        ("ChromaDB Connection", check_chroma_connection),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = all(results.values())
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    if all_ok:
        print("\n✅ All checks passed")
        print("Segfault may be due to other issues - see suggestions below")
    else:
        print("\n⚠️  Issues detected")
    
    suggest_fixes()


if __name__ == "__main__":
    main()