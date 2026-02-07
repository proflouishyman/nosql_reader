# CODE CONSOLIDATION GUIDE - Tier 0 Implementation

## Overview

This guide shows how to consolidate Tier 0 code with existing patterns from your codebase to avoid duplication and maintain "one true source" for configuration and utilities.

---

## 1. Configuration (.env file)

### Add to `.env` file:

```bash
# =========================================================
# TIER 0 CORPUS EXPLORATION CONFIGURATION
# =========================================================

# Question Generation Pipeline
QUESTION_TARGET_COUNT=12
QUESTION_MIN_SCORE=60
QUESTION_QUESTIONS_PER_TYPE=5
QUESTION_MAX_REFINEMENTS=2

# Question Validation Thresholds
QUESTION_SCORE_EXCELLENT=80
QUESTION_SCORE_GOOD=70
QUESTION_SCORE_ACCEPTABLE=60
QUESTION_SCORE_REFINE=50

# Corpus Exploration Settings
CORPUS_EXPLORATION_BUDGET=2000
CORPUS_EXPLORATION_STRATEGY=balanced
CORPUS_BATCH_SIZE=50
CORPUS_MAX_BATCH_CHARS=60000

# Notebook Storage
NOTEBOOK_SAVE_DIR=/app/logs/corpus_exploration
NOTEBOOK_AUTO_SAVE=1

# Debug and Logging
TIER0_DEBUG_MODE=1
TIER0_LOG_DIR=/app/logs/tier0
```

### Update `config.py`:

```python
# In config.py, add Tier 0 configuration section

@dataclass(frozen=True)
class Tier0Config:
    """Tier 0 corpus exploration configuration."""
    # Question generation
    target_count: int
    min_score: int
    questions_per_type: int
    max_refinements: int
    
    # Thresholds
    score_excellent: int
    score_good: int
    score_acceptable: int
    score_refine: int
    
    # Corpus exploration
    exploration_budget: int
    exploration_strategy: str
    batch_size: int
    max_batch_chars: int
    
    # Storage
    notebook_save_dir: str
    notebook_auto_save: bool
    
    # Debug
    tier0_debug_mode: bool
    tier0_log_dir: str

# In AppConfig dataclass, add:
@dataclass(frozen=True)
class AppConfig:
    # ... existing fields ...
    tier0: Tier0Config  # NEW

# In ConfigLoader.from_env(), add:
tier0 = Tier0Config(
    target_count=_env_int("QUESTION_TARGET_COUNT", 12),
    min_score=_env_int("QUESTION_MIN_SCORE", 60),
    questions_per_type=_env_int("QUESTION_QUESTIONS_PER_TYPE", 5),
    max_refinements=_env_int("QUESTION_MAX_REFINEMENTS", 2),
    score_excellent=_env_int("QUESTION_SCORE_EXCELLENT", 80),
    score_good=_env_int("QUESTION_SCORE_GOOD", 70),
    score_acceptable=_env_int("QUESTION_SCORE_ACCEPTABLE", 60),
    score_refine=_env_int("QUESTION_SCORE_REFINE", 50),
    exploration_budget=_env_int("CORPUS_EXPLORATION_BUDGET", 2000),
    exploration_strategy=_env("CORPUS_EXPLORATION_STRATEGY", "balanced"),
    batch_size=_env_int("CORPUS_BATCH_SIZE", 50),
    max_batch_chars=_env_int("CORPUS_MAX_BATCH_CHARS", 60000),
    notebook_save_dir=_env("NOTEBOOK_SAVE_DIR", "/app/logs/corpus_exploration"),
    notebook_auto_save=_env_bool("NOTEBOOK_AUTO_SAVE", True),
    tier0_debug_mode=_env_bool("TIER0_DEBUG_MODE", False),
    tier0_log_dir=_env("TIER0_LOG_DIR", "/app/logs/tier0"),
)

return AppConfig(
    # ... existing fields ...
    tier0=tier0  # NEW
)
```

---

## 2. Reusable Utilities

### JSON Parsing - REUSE from existing code

**Before (duplicated in each file):**
```python
# In question_validator.py
text = response.content.strip()
if text.startswith('```json'):
    text = text[7:]
# ... etc
```

**After (use consolidated utility):**
```python
from tier0_utils import parse_llm_json

# Clean pattern
data = parse_llm_json(response.content, default={})
```

**Pattern consolidated from:**
- `ner_processor_llm.py` - JSON cleaning for entity extraction
- `person_synthesis.py` - JSON parsing for synthesis
- `iterative_adversarial_agent.py` - JSON array extraction
- `json_validator.py` - Control character removal

---

### Logging - REUSE from adversarial_rag.py pattern

**Before (duplicated):**
```python
# In question_validator.py
_log_file = None

def _init_log_file():
    global _log_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ... etc
```

**After (use consolidated logger):**
```python
from tier0_utils import Tier0Logger

logger = Tier0Logger(
    log_dir=APP_CONFIG.tier0.tier0_log_dir,
    log_prefix="question_validation"
)

logger.log("Validate", "Starting validation", icon="⚖️")
logger.log_prompt("Validation Prompt", prompt)
logger.log_response("Validation Response", response)
logger.close()
```

**Pattern reused from:**
- `adversarial_rag.py` - Timestamped log files, debug_step()
- `iterative_adversarial_agent.py` - Prompt/response logging
- `person_synthesis.py` - Debug file saving

---

### File Saving - REUSE from person_synthesis.py pattern

**Before (duplicated):**
```python
# In research_notebook.py
def save(self, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(self.to_dict(), f, indent=2)
```

**After (use consolidated saver):**
```python
from tier0_utils import save_with_timestamp

# Save with automatic timestamping
filepath = save_with_timestamp(
    content=notebook.to_dict(),
    base_dir=APP_CONFIG.tier0.notebook_save_dir,
    filename_prefix="notebook",
    file_type="json"
)

# Returns: /app/logs/corpus_exploration/20251229_143022_notebook.json
```

**Pattern reused from:**
- `person_synthesis.py._save_debug_file()` - Timestamp + subdirectory logic
- `backup_db.py` - Timestamped directory creation

---

## 3. Adversarial Validation - REUSE existing infrastructure

### Instead of reimplementing validation, EXTEND existing adversarial_rag.py

**Current approach:**
```python
# question_validator.py - NEW validation logic
class QuestionValidator:
    def validate(self, question, notebook):
        # Build prompt
        # Call LLM
        # Parse response
```

**Better approach - REUSE adversarial pattern:**
```python
# question_validator.py - EXTENDS adversarial_rag.py
from adversarial_rag import AdversarialRAGHandler

class QuestionValidator:
    """Uses existing adversarial verification pattern."""
    
    def __init__(self):
        # Reuse existing adversarial handler components
        self.llm = LLMClient()  # Same client
        self.logger = Tier0Logger(...)  # Same logging pattern
    
    def validate(self, question, notebook):
        """
        Validation is just verification against corpus.
        Reuses verify_citations() pattern.
        """
        # Same structure as adversarial_rag.verify_citations():
        # 1. Build prompt
        # 2. Call verifier model
        # 3. Parse JSON
        # 4. Fallback on error
        
        # Use same timeout calculation
        token_count = count_tokens(prompt)
        timeout = max(30, min(180, int((token_count / 40) * 1.2)))
        
        # Use same LLM call pattern
        response = self.llm.generate(
            messages=[...],
            profile="verifier",  # SAME profile as adversarial
            temperature=0.0,      # SAME deterministic
            retry=True,           # SAME retry
            timeout=timeout       # SAME adaptive timeout
        )
```

**Benefits:**
- No duplicate validation code
- Same retry logic
- Same error handling
- Same logging pattern
- Consistent scoring (0-100 scale)

---

## 4. Updated File Structure

### Files to KEEP (with consolidation):

```
app/historian_agent/
├── tier0_utils.py           # NEW - Consolidated utilities
├── question_models.py       # KEEP - Data models
├── question_typology.py     # KEEP - Type-specific generators
├── question_validator.py    # UPDATE - Use tier0_utils
├── question_pipeline.py     # UPDATE - Use tier0_utils
├── corpus_explorer.py       # UPDATE - Use tier0_utils
├── research_notebook.py     # UPDATE - Use save_with_timestamp()
└── stratification.py        # KEEP - As is

Existing (reused):
├── rag_base.py             # REUSE - DocumentStore, debug_print
├── llm_abstraction.py      # REUSE - LLMClient
├── adversarial_rag.py      # REUSE - Validation pattern
└── config.py               # UPDATE - Add Tier0Config
```

---

## 5. Migration Checklist

### Step 1: Update configuration
- [ ] Add Tier 0 variables to `.env`
- [ ] Update `config.py` with `Tier0Config`
- [ ] Test config loading: `python -c "from config import APP_CONFIG; print(APP_CONFIG.tier0.target_count)"`

### Step 2: Add consolidated utilities
- [ ] Copy `tier0_utils.py` to `app/historian_agent/`
- [ ] Test utilities: `python -c "from tier0_utils import parse_llm_json; print(parse_llm_json('{\"test\": 1}'))"`

### Step 3: Update question modules
- [ ] `question_validator.py` - Import from tier0_utils
- [ ] `question_pipeline.py` - Import from tier0_utils
- [ ] `corpus_explorer.py` - Import from tier0_utils
- [ ] `research_notebook.py` - Use `save_with_timestamp()`

### Step 4: Update imports

**Replace:**
```python
# Old - duplicated code
def clean_json_response(text):
    text = text.strip()
    if text.startswith('```json'):
        # ... etc
```

**With:**
```python
# New - consolidated
from tier0_utils import parse_llm_json, Tier0Logger, save_with_timestamp
```

### Step 5: Test integration
- [ ] Test question generation pipeline
- [ ] Test notebook saving
- [ ] Test logging
- [ ] Verify all files go to correct directories

---

## 6. Benefits of Consolidation

### Before (duplicated):
- JSON parsing: 5 different implementations
- Logging: 3 different patterns
- File saving: 2 different approaches
- Config: Hardcoded values

**Lines of duplicated code:** ~300 lines

### After (consolidated):
- JSON parsing: 1 utility function
- Logging: 1 logger class
- File saving: 1 function
- Config: All from .env via APP_CONFIG

**Lines saved:** ~250 lines
**Maintenance:** Single source of truth

---

## 7. Example Usage

### Complete example showing consolidation:

```python
# question_validator.py (UPDATED)
from config import APP_CONFIG
from tier0_utils import Tier0Logger, parse_llm_json
from llm_abstraction import LLMClient

class QuestionValidator:
    def __init__(self):
        # Config from APP_CONFIG (not hardcoded)
        self.min_score = APP_CONFIG.tier0.min_score
        self.min_refine = APP_CONFIG.tier0.score_refine
        
        # Logger (reused pattern)
        self.logger = Tier0Logger(
            log_dir=APP_CONFIG.tier0.tier0_log_dir,
            log_prefix="question_validation"
        )
        
        # LLM (reused infrastructure)
        self.llm = LLMClient()
    
    def validate(self, question, notebook):
        self.logger.log("Validate", f"Question: {question.question_text[:60]}...", icon="⚖️")
        
        # Build prompt (same pattern as adversarial_rag)
        prompt = build_validation_prompt(question, notebook)
        self.logger.log_prompt("Validation", prompt)
        
        # Call LLM (same pattern as adversarial_rag)
        response = self.llm.generate(
            messages=[...],
            profile="verifier",  # Reused profile
            temperature=0.0,
            retry=True
        )
        
        # Parse JSON (consolidated utility)
        data = parse_llm_json(response.content, default={})
        
        self.logger.log_response("Validation", response.content)
        return create_validation(data)
```

---

## 8. Testing Consolidation

```bash
# Test config loading
python -c "
from config import APP_CONFIG
print(f'Target questions: {APP_CONFIG.tier0.target_count}')
print(f'Min score: {APP_CONFIG.tier0.min_score}')
print(f'Notebook dir: {APP_CONFIG.tier0.notebook_save_dir}')
"

# Test utilities
python -c "
from tier0_utils import parse_llm_json, save_with_timestamp, Tier0Logger
from pathlib import Path

# Test JSON parsing
json_str = '{\"test\": \"value\"}'
print(parse_llm_json(json_str))

# Test file saving
data = {'notebook': 'test'}
path = save_with_timestamp(data, Path('/tmp'), 'test_notebook')
print(f'Saved to: {path}')

# Test logging
logger = Tier0Logger(Path('/tmp'), 'test')
logger.log('Test', 'Hello', icon='✅')
logger.close()
"
```

---

## Summary

**Consolidation eliminates:**
- ❌ Duplicated JSON parsing (5 implementations → 1)
- ❌ Duplicated logging (3 patterns → 1)
- ❌ Hardcoded config (scattered → centralized .env)
- ❌ File saving duplication (2 approaches → 1)

**Consolidation provides:**
- ✅ Single source of truth (.env → APP_CONFIG)
- ✅ Reusable utilities (tier0_utils.py)
- ✅ Consistent patterns (same as existing code)
- ✅ Easy maintenance (change once, apply everywhere)
- ✅ Backward compatible (doesn't break existing code)

**Next step:** Update the 4 new files to use these consolidated patterns.
