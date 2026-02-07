# Tier 0 Corpus Exploration - Integration Guide

## Overview

Tier 0 is the **systematic corpus reading tier** that runs *before* question-answering. It discovers questions by reading the entire corpus in structured batches, accumulating findings in a persistent research notebook.

## Architecture

**New Files Created:**
- `app/historian_agent/research_notebook.py` - Persistent state manager
- `app/historian_agent/stratification.py` - Corpus stratification strategies  
- `app/historian_agent/corpus_explorer.py` - Main exploration orchestrator

**Dependencies:**
- Uses `rag_base.py` (DocumentStore, utilities)
- Uses `llm_abstraction.py` (LLMClient)
- Uses `config.py` (APP_CONFIG)
- Follows existing factory_refactor patterns

## How It Works

### 1. Stratification
Divides corpus into meaningful batches:
- **Temporal**: By year (e.g., "Year 1923")
- **Genre**: By document type (e.g., "Injury Reports")
- **Biographical**: By person (e.g., "Antonio Mancuso")
- **Collection**: By archival provenance
- **Spatial**: By physical box

### 2. Batch Reading
For each stratum:
1. Sample N documents from that stratum
2. Send to LLM with prior knowledge from notebook
3. LLM extracts: entities, patterns, contradictions, questions
4. Findings accumulate in notebook

### 3. Cumulative Synthesis
Research notebook tracks:
- **Entities**: People/orgs/places with document counts
- **Patterns**: Recurring themes with confidence scores
- **Contradictions**: Disagreements between sources
- **Questions**: Emerging research inquiries
- **Temporal map**: What happens when

### 4. Final Outputs
- **Corpus map**: Archive statistics + orientation notes
- **Research questions**: 8-12 questions to investigate with Tier 1/2
- **Pattern catalog**: High-confidence patterns found
- **Entity registry**: Top entities with contexts

## Integration with routes.py

### Step 1: Add import

```python
# In app/routes.py

# Add to AGENT HANDLER IMPORTS section
from historian_agent.corpus_explorer import CorpusExplorer

# Add to global instances
_corpus_explorer = None

def get_corpus_explorer():
    """Lazy initialization of CorpusExplorer"""
    global _corpus_explorer
    if _corpus_explorer is None:
        _corpus_explorer = CorpusExplorer()
    return _corpus_explorer
```

### Step 2: Add route

```python
# In app/routes.py

@app.route('/api/rag/explore_corpus', methods=['POST'])
def explore_corpus_endpoint():
    """
    Tier 0: Systematic corpus exploration.
    
    Request body:
    {
      "strategy": "balanced",  // temporal, biographical, genre, balanced
      "total_budget": 2000,    // Number of documents to read
      "year_range": [1920, 1930]  // Optional time filter
    }
    
    Returns:
    {
      "corpus_map": {...},
      "questions": [...],
      "patterns": [...],
      "entities": [...],
      "contradictions": [...],
      "notebook_path": "...",
      "exploration_metadata": {...}
    }
    """
    try:
        data = request.get_json()
        
        strategy = data.get('strategy', 'balanced')
        total_budget = data.get('total_budget', 2000)
        year_range = data.get('year_range')  # Optional tuple [start, end]
        
        # Convert year_range to tuple if provided
        if year_range and isinstance(year_range, list) and len(year_range) == 2:
            year_range = tuple(year_range)
        else:
            year_range = None
        
        # Get explorer
        explorer = get_corpus_explorer()
        
        # Run exploration
        report = explorer.explore(
            strategy=strategy,
            total_budget=total_budget,
            year_range=year_range,
            save_notebook=True
        )
        
        return jsonify({
            'status': 'success',
            'report': report
        })
        
    except Exception as e:
        logger.error(f"Corpus exploration failed: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/rag/exploration_status', methods=['GET'])
def exploration_status():
    """
    Get status of current exploration (if running).
    
    For now, returns simple status. Could be extended with
    progress tracking if exploration runs asynchronously.
    """
    try:
        # Check if explorer exists and has notebook
        explorer = get_corpus_explorer()
        
        if explorer.notebook.corpus_map['total_documents_read'] == 0:
            return jsonify({
                'status': 'not_started',
                'message': 'No exploration has been run yet'
            })
        
        return jsonify({
            'status': 'completed',
            'summary': explorer.notebook.get_summary()
        })
        
    except Exception as e:
        logger.error(f"Failed to get exploration status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
```

## Usage Examples

### Basic Exploration (2000 documents, balanced strategy)

```bash
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "balanced",
    "total_budget": 2000
  }'
```

### Temporal Focus (chronological reading, 1920-1930)

```bash
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "temporal",
    "total_budget": 1000,
    "year_range": [1920, 1930]
  }'
```

### Biographical Focus (people with substantial documentation)

```bash
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "biographical",
    "total_budget": 1500
  }'
```

## Workflow Integration

### Recommended User Flow

1. **First time**: Run Tier 0 exploration
   ```
   POST /api/rag/explore_corpus
   → Returns corpus map + research questions
   ```

2. **Review findings**: User examines questions and patterns
   ```
   {
     "questions": [
       {
         "question": "Why did injury rates spike during 1923-1925?",
         "why_interesting": "47 cases show correlation with labor disputes",
         "approach": "Compare injury rates by division and union presence"
       },
       ...
     ]
   }
   ```

3. **Investigate specific questions**: Use existing Tier 1/2 system
   ```
   POST /api/rag/investigate
   Body: { "question": "Why did injury rates spike during 1923-1925?" }
   → Uses existing tiered investigation
   ```

4. **Iterative refinement**: Re-run exploration with focused parameters
   ```
   POST /api/rag/explore_corpus
   Body: { 
     "strategy": "temporal",
     "year_range": [1923, 1925],  // Focus on spike period
     "total_budget": 500
   }
   ```

## Performance Characteristics

### Time Estimates (M4 Mac, 128GB RAM)

**Batch Size**: 50 documents per LLM call  
**LLM Time**: ~40s per batch  
**Total Time** = (total_budget / 50) * 40s

Examples:
- 500 docs: ~6.5 minutes (10 batches)
- 1000 docs: ~13 minutes (20 batches)
- 2000 docs: ~26 minutes (40 batches)

### Resource Usage

**Memory**: ~500MB (DocumentStore + LLMClient)  
**Disk**: ~1MB per saved notebook  
**Network**: None (uses local Ollama)  

### Optimization Options

**Reduce time**:
- Lower `total_budget` (500-1000 docs still effective)
- Use `profile="fast"` instead of `profile="quality"`
- Smaller `sample_size` per stratum

**Improve quality**:
- Increase `total_budget` (up to full corpus)
- Use `profile="quality"` or `profile="verifier"`
- Multiple passes with different strategies

## Storage and Persistence

### Notebook Files

Location: `/app/logs/corpus_exploration/`  
Format: JSON  
Naming: `notebook_YYYYMMDD_HHMMSS.json`

**Notebook contains**:
- All entities found (with contexts)
- All patterns (with evidence)
- All questions generated
- All contradictions noticed
- Complete corpus statistics
- Processing metadata

### Loading Previous Notebooks

```python
from historian_agent.research_notebook import ResearchNotebook

# Load previous exploration
notebook = ResearchNotebook.load('/app/logs/corpus_exploration/notebook_20251229_143022.json')

# Access findings
print(f"Entities found: {len(notebook.entities)}")
print(f"Patterns: {len(notebook.patterns)}")
print(f"Questions: {len(notebook.questions)}")

# Get summary
summary = notebook.get_summary()
```

## Configuration

### MongoDB Fields Used

Stratification depends on these document fields:
- `year` - Temporal stratification
- `document_type` - Genre stratification
- `person_id`, `person_name`, `person_folder` - Biographical
- `collection` - Collection stratification
- `archive_structure.physical_box` - Spatial stratification

Ensure these fields are populated during ingestion.

### LLM Profiles

Uses profiles from `config.py`:
- `quality` (default): `qwen2.5:32b` for batch analysis
- `verifier`: For high-accuracy question generation
- `fast`: For quick exploration (lower quality)

### Tuning Parameters

In `corpus_explorer.py`:
- `MAX_CHARS`: 60000 (batch size for LLM)
- `DOCS_PER_YEAR`: 50 (temporal sampling)
- `DOCS_PER_TYPE`: 100 (genre sampling)
- `MIN_DOCS_PER_PERSON`: 10 (biographical filter)

## Testing

### Simple Test

```python
# Test exploration with small budget
from historian_agent.corpus_explorer import explore_corpus

report = explore_corpus(
    strategy='balanced',
    total_budget=100,  # Small test
    year_range=(1920, 1925)
)

print(f"Documents read: {report['exploration_metadata']['documents_read']}")
print(f"Questions generated: {len(report['questions'])}")
print(f"Patterns found: {len(report['patterns'])}")
```

### Full Validation

```bash
# Run test script
python /home/claude/test_tier0.py
```

## Troubleshooting

### Issue: "No documents found"
**Cause**: MongoDB fields not populated  
**Fix**: Run ingestion to populate `year`, `document_type`, etc.

### Issue: "LLM timeout"
**Cause**: Batch too large  
**Fix**: Reduce `docs_per_year` or use `profile="fast"`

### Issue: "Invalid JSON from LLM"
**Cause**: LLM didn't follow JSON format  
**Fix**: Check prompt, try different model, or add retry logic

### Issue: "Exploration too slow"
**Cause**: Large batches, slow model  
**Fix**: Reduce `total_budget`, use faster model profile

## Next Steps

After Tier 0 exploration:

1. **Review corpus map**: Understand archive scope and biases
2. **Select questions**: Choose which to investigate with Tier 1/2
3. **Run investigations**: Use existing `POST /api/rag/investigate`
4. **Iterate**: Re-explore focused areas based on findings

The Tier 0 → Tier 1/2 pipeline replicates how historians actually work:
**Read broadly → Form questions → Chase specific evidence**
