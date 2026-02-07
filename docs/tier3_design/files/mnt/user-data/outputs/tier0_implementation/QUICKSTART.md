# Tier 0 Corpus Exploration - Quick Start

## 5-Minute Deployment

### Files to Move

From `/home/claude` to your project:

```bash
# Core implementation files
cp /home/claude/app/historian_agent/research_notebook.py \
   app/historian_agent/research_notebook.py

cp /home/claude/app/historian_agent/stratification.py \
   app/historian_agent/stratification.py

cp /home/claude/app/historian_agent/corpus_explorer.py \
   app/historian_agent/corpus_explorer.py

# Test and docs
cp /home/claude/test_tier0.py \
   scripts/test_tier0.py

cp /home/claude/TIER0_INTEGRATION.md \
   docs/TIER0_INTEGRATION.md

cp /home/claude/TIER0_IMPLEMENTATION_SUMMARY.md \
   docs/TIER0_IMPLEMENTATION_SUMMARY.md
```

### Add Route (2 minutes)

Edit `app/routes.py`:

**Step 1:** Add import at top (around line 15):
```python
from historian_agent.corpus_explorer import CorpusExplorer
```

**Step 2:** Add global instance (around line 25):
```python
_corpus_explorer = None

def get_corpus_explorer():
    """Lazy initialization of CorpusExplorer"""
    global _corpus_explorer
    if _corpus_explorer is None:
        _corpus_explorer = CorpusExplorer()
    return _corpus_explorer
```

**Step 3:** Add endpoint (anywhere after other /api/rag routes):
```python
@app.route('/api/rag/explore_corpus', methods=['POST'])
def explore_corpus_endpoint():
    """Tier 0: Systematic corpus exploration."""
    try:
        data = request.get_json()
        
        strategy = data.get('strategy', 'balanced')
        total_budget = data.get('total_budget', 2000)
        year_range = data.get('year_range')
        
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
```

### Test (1 minute)

```bash
# Quick validation (no LLM calls)
docker compose exec app python -c "
from historian_agent.stratification import CorpusStratifier
s = CorpusStratifier()
strata = s.temporal_stratification()
print(f'✅ Created {len(strata)} temporal strata')
"

# Full test with small budget
docker compose exec app python scripts/test_tier0.py --budget 50 --skip-full
```

### First Run (5-10 minutes)

```bash
# Start small exploration
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "balanced",
    "total_budget": 500
  }'

# Response will include:
# - corpus_map (archive statistics + notes)
# - questions (8-12 research questions)
# - patterns (recurring themes found)
# - entities (people/orgs found)
# - notebook_path (saved state)
```

---

## What You Get

After 5-10 minute exploration with budget=500:

### 1. Corpus Map
```
"This archive documents B&O Railroad employment 1900-1940.
Relief Records show detailed injury documentation.
Microfilm captures early employment records.
Notable gaps in 1910-1912..."
```

### 2. Research Questions
```
1. Why did injury rates spike during 1923-1925?
   → 47 cases correlate with labor disputes

2. How did disability approval rates vary by division?
   → Division A: 85%, Division C: 45%

3. What patterns exist in wage records vs injury reports?
   → Wage records systematically underreport injuries
```

### 3. High-Confidence Patterns
```
- Railroad firemen primarily suffered burns (83 docs, high confidence)
- Injury rates spike during labor disputes (47 docs, high confidence)
- Disability claims more likely approved if union member (23 docs, medium)
```

### 4. Top Entities
```
- Antonio Mancuso: 173 documents
- B&O Railroad: 1847 documents
- United Mine Workers: 156 documents
```

---

## Next Actions

### Use Generated Questions

Take questions from Tier 0 and investigate with Tier 1/2:

```bash
# From Tier 0 output
question="Why did injury rates spike during 1923-1925?"

# Investigate with existing system
curl -X POST http://localhost:5006/api/rag/investigate \
  -d "{\"question\": \"$question\"}"
```

### Run Focused Exploration

Based on Tier 0 findings, dive deeper:

```bash
# Tier 0 found spike in 1923-1925, explore that period
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -d '{
    "strategy": "temporal",
    "total_budget": 300,
    "year_range": [1923, 1925]
  }'
```

### Scale Up

Once validated, run full corpus exploration:

```bash
# Overnight run: full corpus
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -d '{
    "strategy": "balanced",
    "total_budget": 5000
  }'
```

---

## Troubleshooting

### "No documents found in stratum"

**Fix:** Check MongoDB fields are populated:
```bash
docker compose exec mongodb mongosh historical_documents --eval '
db.documents.findOne({}, {year: 1, document_type: 1, person_id: 1})
'
```

### "LLM timeout"

**Fix:** Use faster model:
```python
# In corpus_explorer.py, change:
profile="quality"  →  profile="fast"
```

### "Exploration too slow"

**Fix:** Reduce budget or use faster model:
```bash
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -d '{"total_budget": 200}'  # Much faster
```

---

## Full Documentation

- `TIER0_IMPLEMENTATION_SUMMARY.md` - Complete overview
- `TIER0_INTEGRATION.md` - Detailed integration guide
- `test_tier0.py` - Test suite with examples

---

## Summary

**You now have:**
✅ Tier 0 exploration system installed  
✅ Flask route ready  
✅ Test suite available  
✅ Documentation complete  

**Ready to use:**
```bash
# Start exploration
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -d '{"strategy": "balanced", "total_budget": 500}'

# Get research questions
# Investigate with Tier 1/2
# Iterate and refine
```

**This solves:** "Top-K myopia" - you now read systematically across the entire corpus, discovering patterns and questions you didn't know to ask.
