# DATA STRUCTURE DOCUMENTATION
## Historical Document Reader - MongoDB Schema Reference

This document provides the definitive schema for your MongoDB collections as they actually exist in production.

---

## 📊 Collections Overview

```
railroad_documents (database)
├── documents (9,642 docs)
└── document_chunks (chunks collection)
```

---

## 🗂️ Collection: `documents`

### Collection Name
```python
MONGO_DB_NAME = "railroad_documents"
DOCUMENTS_COLLECTION = "documents"
```

### Document Count
- **Total:** 9,642 documents

### Schema Structure

```javascript
{
  _id: ObjectId('69149fa1b533a660a69d8acc'),  // MongoDB ObjectId (NOT string)
  
  // Text Content
  ocr_text: "Full OCR extracted text...",      // PRIMARY text field (9,641/9,642 docs have this)
  summary: "AI-generated summary...",          // ALL docs have this
  
  // File Information
  filename: "tray_1_roll_5_page486_img1.png.json",
  relative_path: "path/to/file",
  file_path: "/full/path/to/file",
  file_hash: "sha256...",
  
  // Structured Data
  sections: [...],                             // Document sections/structure
  
  // Entity Extraction
  entities: [...],                             // Extracted entities
  entities_extracted: true,
  entities_extracted_at: ISODate("..."),
  entity_refs: [...]
}
```

### Key Fields for Retrieval

| Field | Type | Populated | Usage |
|-------|------|-----------|-------|
| `_id` | ObjectId | 100% | **Primary key** |
| `ocr_text` | String | 99.99% | **Primary text source** |
| `summary` | String | 100% | Fallback text source |
| `filename` | String | 100% | Display/citations |
| `content` | String | 0% | **NOT USED** |
| `text` | String | 0% | **NOT USED** |
| `title` | String | 0% | **NOT USED** |

### Critical Notes

1. **Text Storage:** Text is stored **directly in documents**, NOT in a separate full-text field
2. **Primary Text Field:** Always use `ocr_text` as the primary text source
3. **ObjectId Type:** `_id` is stored as MongoDB ObjectId, not string
4. **No Chunking at Document Level:** Documents contain full text, chunking happens in separate collection

---

## 🧩 Collection: `document_chunks`

### Collection Name
```python
CHUNKS_COLLECTION = "document_chunks"
```

### Schema Structure

```javascript
{
  _id: ObjectId('...'),                                    // MongoDB ObjectId
  
  // Parent Reference - CRITICAL TYPE MISMATCH
  document_id: "6939d8083334b77a9b7f12b7",                // STRING (not ObjectId!)
  
  // Chunk Identification
  chunk_id: "6939d8083334b77a9b7f12b7_chunk_000",         // String format: {doc_id}_chunk_{index}
  chunk_index: 0,                                          // Integer, 0-based
  
  // Content
  text: "B. & O. R. R. CO. DEPARTMENT No. 42792...",      // Chunk text content
  token_count: 312,                                        // Token count for this chunk
  
  // Overlap (for context preservation)
  overlap_previous: null,                                  // Usually null
  overlap_next: null,                                      // Usually null
  
  // Metadata
  metadata: {}                                             // Usually empty
}
```

### Key Fields for Retrieval

| Field | Type | Usage |
|-------|------|-------|
| `document_id` | **STRING** | Reference to parent `documents._id` (as string!) |
| `chunk_id` | String | Unique chunk identifier |
| `chunk_index` | Integer | Order within parent document |
| `text` | String | Chunk content |
| `token_count` | Integer | Size estimation |

---

## 🚨 CRITICAL TYPE MISMATCH

### The Problem

**Parent documents store IDs as ObjectId:**
```javascript
documents._id: ObjectId('6939d8083334b77a9b7f12b7')
```

**Chunks reference parents as STRING:**
```javascript
document_chunks.document_id: '6939d8083334b77a9b7f12b7'  // No ObjectId wrapper!
```

### Correct Query Pattern

**❌ WRONG - Will find nothing:**
```python
from bson import ObjectId

doc_id = "6939d8083334b77a9b7f12b7"
chunks = db.document_chunks.find({
    "document_id": ObjectId(doc_id)  # WRONG! Looking for ObjectId
})
```

**✅ CORRECT - Finds chunks:**
```python
doc_id = "6939d8083334b77a9b7f12b7"
chunks = db.document_chunks.find({
    "document_id": doc_id  # CORRECT! Use string directly
})
```

### Code Implementation

```python
def get_full_document_text(self, doc_ids: List[str]):
    """Fetch full text for documents."""
    
    for doc_id in doc_ids:
        # doc_id comes in as string like "6939d8083334b77a9b7f12b7"
        
        # Query chunks using STRING (not ObjectId)
        chunks = self.chunks_coll.find({
            "document_id": doc_id  # Use string directly!
        }).sort("chunk_index", 1)
        
        # Query parent document using ObjectId
        from bson import ObjectId
        parent = self.documents_coll.find_one({
            "_id": ObjectId(doc_id)  # Convert to ObjectId for parent lookup
        })
```

---

## 🔄 Retrieval Flow

### Small-to-Big Retrieval Process

```
1. Vector Search
   ↓
   Returns chunk IDs from ChromaDB
   
2. Hydrate Chunks from MongoDB
   ↓
   db.document_chunks.find({_id: {$in: chunk_ids}})
   ↓
   Extract document_id (STRING) from each chunk
   
3. Get Parent Document IDs
   ↓
   unique_doc_ids = ["6939d8083334b77a9b7f12b7", "6939d80a3334b77a9b7f1937", ...]
   
4. Fetch All Chunks for Each Parent
   ↓
   FOR EACH doc_id IN unique_doc_ids:
     chunks = db.document_chunks.find({
       "document_id": doc_id  # ← Use STRING, not ObjectId!
     }).sort("chunk_index", 1)
   
5. Combine Chunks into Full Text
   ↓
   full_text = "\n".join([chunk["text"] for chunk in chunks])
   
6. Fetch Parent Metadata (Optional)
   ↓
   parent = db.documents.find_one({
     "_id": ObjectId(doc_id)  # ← NOW convert to ObjectId
   })
```

---

## 🎯 Text Field Priority

When retrieving text from documents, try in this order:

```python
text = (
    doc.get("ocr_text") or      # 1. PRIMARY - 99.99% of docs
    doc.get("content") or        # 2. Not used in your data
    doc.get("text") or           # 3. Not used in your data  
    doc.get("summary") or        # 4. FALLBACK - 100% of docs
    ""                           # 5. Empty if nothing found
)
```

---

## 📋 Environment Variables

```bash
# MongoDB Connection
MONGO_URI=mongodb://admin:secret@mongodb:27017/admin
MONGO_DB_NAME=railroad_documents

# Collection Names
DOCUMENTS_COLLECTION=documents
CHUNKS_COLLECTION=document_chunks

# Connection Settings
MONGO_CONNECT_TIMEOUT_MS=5000
MONGO_SERVER_TIMEOUT_MS=5000
```

---

## 🧪 Testing Queries

### Test 1: Count Documents
```javascript
use railroad_documents
db.documents.countDocuments()
// Should return: 9642
```

### Test 2: Check Document Structure
```javascript
db.documents.findOne({}, {
  _id: 1,
  filename: 1,
  ocr_text: 1,
  summary: 1
})
// Returns ObjectId for _id, string for others
```

### Test 3: Count Chunks
```javascript
db.document_chunks.countDocuments()
// Returns total chunk count
```

### Test 4: Find Chunks for Document (CORRECT METHOD)
```javascript
// Get a document ID first
var doc = db.documents.findOne({}, {_id: 1})
var doc_id_str = doc._id.toString()  // Convert to string!

// Find chunks using STRING
db.document_chunks.find({
  document_id: doc_id_str
}).count()
// Should return > 0 if chunks exist
```

### Test 5: Verify Type Mismatch
```javascript
// This shows the type difference
db.document_chunks.findOne({}, {document_id: 1, _id: 0})
// Returns: { document_id: '6939d8083334b77a9b7f12b7' }  ← STRING

db.documents.findOne({}, {_id: 1, filename: 0})
// Returns: { _id: ObjectId('69149fa1b533a660a69d8acc') }  ← ObjectId
```

---

## 🐛 Common Debugging Issues

### Issue 1: "No chunks found for document"

**Symptom:**
```
[DEBUG] No chunks found for document 6939d8063334b77a9b7f0f45
```

**Cause:** Searching with ObjectId instead of string

**Fix:**
```python
# WRONG
chunks = db.document_chunks.find({"document_id": ObjectId(doc_id)})

# CORRECT
chunks = db.document_chunks.find({"document_id": doc_id})
```

### Issue 2: "Fetched 0 characters from 0 documents"

**Cause:** No chunks found (see Issue 1) OR documents have no text

**Check:**
```javascript
// Verify document has text
db.documents.findOne(
  {_id: ObjectId('6939d8063334b77a9b7f0f45')},
  {ocr_text: 1, summary: 1}
)
```

### Issue 3: "Collection not found"

**Cause:** Wrong collection name in config

**Verify:**
```javascript
db.getCollectionNames()
// Should show: ["documents", "document_chunks"]
```

**Fix:** Check `.env` file:
```bash
CHUNKS_COLLECTION=document_chunks  # Not "chunks"!
```

---

## 📊 Data Statistics

Based on production data as of 2025-12-28:

| Metric | Value |
|--------|-------|
| Total Documents | 9,642 |
| Documents with `ocr_text` | 9,641 (99.99%) |
| Documents with `summary` | 9,642 (100%) |
| Average `ocr_text` length | ~500-1,000 chars |
| Documents with `content` | 0 (0%) |
| Documents with `title` | 0 (0%) |

---

## 🔧 Code Reference

### Correct MongoDB Query Pattern

```python
from pymongo import MongoClient
from bson import ObjectId

# Connect
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
documents = db[DOCUMENTS_COLLECTION]
chunks = db[CHUNKS_COLLECTION]

# Get document by ID (requires ObjectId)
doc_id_str = "6939d8063334b77a9b7f0f45"
doc = documents.find_one({"_id": ObjectId(doc_id_str)})

# Get chunks for document (requires STRING)
chunk_list = list(chunks.find(
    {"document_id": doc_id_str}  # Use string directly!
).sort("chunk_index", 1))

# Combine chunk text
full_text = "\n".join([c["text"] for c in chunk_list])
```

---

## ✅ Summary

**Key Takeaways:**

1. **Documents** use ObjectId for `_id`
2. **Chunks** use STRING for `document_id` (not ObjectId!)
3. **Text** is in `ocr_text` field (99.99% of docs)
4. **Collections** are `documents` and `document_chunks`
5. **Query chunks** with string, NOT ObjectId
6. **Fallback** to embedded `ocr_text` if no chunks found

**Most Common Bug:**
Using `ObjectId(doc_id)` when querying chunks - always use the string directly!