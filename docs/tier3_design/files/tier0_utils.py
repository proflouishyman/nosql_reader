# app/historian_agent/tier0_utils.py
# Created: 2025-12-29
# Purpose: Reusable utilities for Tier 0 (extracted from existing codebase)

"""
Tier 0 Utilities - Consolidates common patterns from existing code.

Reuses patterns from:
- adversarial_rag.py (logging, debug)
- person_synthesis.py (file saving with timestamps)
- iterative_adversarial_agent.py (JSON parsing)
- ner_processor_llm.py (JSON cleaning)
"""

import os
import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# JSON Parsing (Reused from existing code)
# ============================================================================

def clean_json_response(response_text: str) -> str:
    """
    Clean LLM JSON response (reused pattern from multiple files).
    
    This consolidates the JSON cleaning logic found in:
    - ner_processor_llm.py
    - person_synthesis.py
    - iterative_adversarial_agent.py
    - batch_download.py
    
    Args:
        response_text: Raw LLM response that may contain markdown
        
    Returns:
        Cleaned JSON string ready for parsing
    """
    text = response_text.strip()
    
    # Remove markdown code blocks (pattern from multiple files)
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    
    if text.endswith('```'):
        text = text[:-3]
    
    text = text.strip()
    
    # Remove control characters (from json_validator.py)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    
    # Extract JSON object/array (from json_validator.py)
    if text.startswith('{'):
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end + 1]
    elif text.startswith('['):
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            text = text[start:end + 1]
    
    return text


def parse_llm_json(response_text: str, default: Any = None) -> Any:
    """
    Parse LLM JSON response with cleaning and error handling.
    
    Consolidates JSON parsing pattern used throughout codebase.
    
    Args:
        response_text: Raw LLM response
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON object or default
    """
    try:
        cleaned = clean_json_response(response_text)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        if default is not None:
            return default
        raise ValueError(f"Failed to parse JSON: {str(e)}")


# ============================================================================
# File Saving (Reused from person_synthesis.py)
# ============================================================================

def save_with_timestamp(
    content: Any,
    base_dir: Path,
    filename_prefix: str,
    file_type: str = "json",
    subdirectory: Optional[str] = None
) -> Path:
    """
    Save content with timestamp (pattern from person_synthesis.py).
    
    Args:
        content: Content to save (dict, string, etc.)
        base_dir: Base directory
        filename_prefix: Prefix for filename
        file_type: File extension (json, txt, md)
        subdirectory: Optional subdirectory name
        
    Returns:
        Path to saved file
    """
    # Create directory structure
    if subdirectory:
        # Clean subdirectory name for filesystem
        safe_subdir = subdirectory.replace('/', '_').replace(' ', '_')
        save_dir = base_dir / safe_subdir
    else:
        save_dir = base_dir
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename
    filename = f"{timestamp}_{filename_prefix}.{file_type}"
    filepath = save_dir / filename
    
    # Save content
    if isinstance(content, (dict, list)):
        # JSON serialization
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, default=str)
    else:
        # Text content
        filepath.write_text(str(content), encoding='utf-8')
    
    return filepath


# ============================================================================
# Logging (Reused from adversarial_rag.py pattern)
# ============================================================================

class Tier0Logger:
    """
    Tier 0 logger (reuses adversarial_rag.py pattern).
    
    Provides:
    - Timestamped log files
    - Debug mode control
    - Console + file output
    """
    
    def __init__(self, log_dir: Path = None, log_prefix: str = "tier0"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files (uses TIER0_LOG_DIR if None)
            log_prefix: Prefix for log filename
        """
        self.log_dir = log_dir or TIER0_LOG_DIR
        self.log_prefix = log_prefix
        self.log_file = None
        self.debug_mode = TIER0_DEBUG_MODE
        
        if self.debug_mode:
            self._init_log_file()
    
    def _init_log_file(self):
        """Initialize log file with timestamp."""
        if self.log_file is not None:
            return
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"{self.log_prefix}_{timestamp}.log"
        
        try:
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self.log_file.write(f"=== {self.log_prefix.upper()} Log ===\n")
            self.log_file.write(f"Started: {datetime.now().isoformat()}\n")
            self.log_file.write("="*60 + "\n\n")
            self.log_file.flush()
            
            sys.stderr.write(f"🔍 [{self.log_prefix.upper()}] Logging to: {log_path}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"⚠️ Failed to create log file: {e}\n")
            self.log_file = None
    
    def log(self, step: str, detail: str = "", icon: str = "⚙️", level: str = "INFO"):
        """
        Log message (same pattern as adversarial_rag.py debug_step).
        
        Args:
            step: Step name
            detail: Detail message
            icon: Icon for console output
            level: Log level (INFO, WARN, ERROR)
        """
        if not self.debug_mode:
            return
        
        timestamp = time.strftime("%H:%M:%S")
        
        # Console output
        console_msg = f"{icon} [{timestamp}] [{step.upper()}]"
        if detail:
            console_msg += f" {detail}"
        sys.stderr.write(console_msg + "\n")
        sys.stderr.flush()
        
        # File output
        if self.log_file is not None:
            try:
                log_msg = f"[{timestamp}] [{level}] [{step.upper()}]\n"
                if detail:
                    log_msg += f"  {detail}\n"
                self.log_file.write(log_msg)
                self.log_file.flush()
            except Exception:
                pass
    
    def log_prompt(self, stage: str, prompt: str):
        """Log full LLM prompt to file (from iterative_adversarial_agent.py)."""
        if not self.debug_mode or self.log_file is None:
            return
        
        try:
            self.log_file.write(f"\n{'='*60}\n")
            self.log_file.write(f"PROMPT: {stage}\n")
            self.log_file.write(f"{'='*60}\n")
            self.log_file.write(prompt)
            self.log_file.write(f"\n{'='*60}\n\n")
            self.log_file.flush()
        except Exception:
            pass
    
    def log_response(self, stage: str, response: str):
        """Log LLM response to file (from iterative_adversarial_agent.py)."""
        if not self.debug_mode or self.log_file is None:
            return
        
        try:
            self.log_file.write(f"\n{'-'*60}\n")
            self.log_file.write(f"RESPONSE: {stage}\n")
            self.log_file.write(f"{'-'*60}\n")
            self.log_file.write(response[:2000])  # First 2000 chars
            if len(response) > 2000:
                self.log_file.write(f"\n... (truncated {len(response) - 2000} chars)")
            self.log_file.write(f"\n{'-'*60}\n\n")
            self.log_file.flush()
        except Exception:
            pass
    
    def close(self):
        """Close log file."""
        if self.log_file is not None:
            try:
                self.log_file.write(f"\n{'='*60}\n")
                self.log_file.write(f"Ended: {datetime.now().isoformat()}\n")
                self.log_file.write(f"{'='*60}\n")
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None


# ============================================================================
# Tier 0 Configuration (loads from APP_CONFIG which loads from .env)
# ============================================================================

# Question generation (from .env or defaults)
QUESTION_TARGET_COUNT = int(os.environ.get("QUESTION_TARGET_COUNT", "12"))
QUESTION_MIN_SCORE = int(os.environ.get("QUESTION_MIN_SCORE", "60"))
QUESTION_QUESTIONS_PER_TYPE = int(os.environ.get("QUESTION_QUESTIONS_PER_TYPE", "5"))
QUESTION_MAX_REFINEMENTS = int(os.environ.get("QUESTION_MAX_REFINEMENTS", "2"))

# Thresholds
QUESTION_SCORE_EXCELLENT = int(os.environ.get("QUESTION_SCORE_EXCELLENT", "80"))
QUESTION_SCORE_GOOD = int(os.environ.get("QUESTION_SCORE_GOOD", "70"))
QUESTION_SCORE_ACCEPTABLE = int(os.environ.get("QUESTION_SCORE_ACCEPTABLE", "60"))
QUESTION_SCORE_REFINE = int(os.environ.get("QUESTION_SCORE_REFINE", "50"))

# Corpus exploration
CORPUS_EXPLORATION_BUDGET = int(os.environ.get("CORPUS_EXPLORATION_BUDGET", "2000"))
CORPUS_EXPLORATION_STRATEGY = os.environ.get("CORPUS_EXPLORATION_STRATEGY", "balanced")
CORPUS_BATCH_SIZE = int(os.environ.get("CORPUS_BATCH_SIZE", "50"))
CORPUS_MAX_BATCH_CHARS = int(os.environ.get("CORPUS_MAX_BATCH_CHARS", "60000"))

# Storage (reuses pattern from person_synthesis.py)
NOTEBOOK_SAVE_DIR = Path(os.environ.get("NOTEBOOK_SAVE_DIR", "/app/logs/corpus_exploration"))
NOTEBOOK_AUTO_SAVE = os.environ.get("NOTEBOOK_AUTO_SAVE", "1") == "1"

# Debug
TIER0_DEBUG_MODE = os.environ.get("TIER0_DEBUG_MODE", os.environ.get("DEBUG_MODE", "0")) == "1"
TIER0_LOG_DIR = Path(os.environ.get("TIER0_LOG_DIR", "/app/logs/tier0"))
