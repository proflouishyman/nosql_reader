# 2025-12-15 18:40
# Purpose: verify Chroma collection embedding dimension by reading one stored vector length.

import os
import chromadb

persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY", "/data/chroma_db/persist")
client = chromadb.PersistentClient(path=persist_dir)
col = client.get_collection("historian_documents")

res = col.get(limit=1, include=["embeddings"])

embs = res.get("embeddings")

if embs is None or len(embs) == 0:
    print("No embeddings found in collection.")
else:
    # embs[0] may be a numpy array or list
    print("Collection historian_documents embedding dim =", len(embs[0]))
