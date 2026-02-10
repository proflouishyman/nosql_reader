# app/config.py
# Complete configuration system for Historical Document Reader
# Created: 2025-12-28

"""
Configuration Management

Loads all environment variables once at module import.
Provides typed, immutable configuration objects.
No runtime env reads anywhere else in the application.

Usage:
    from config import APP_CONFIG
    
    # Access any config
    db_name = APP_CONFIG.database.db_name
    llm_model = APP_CONFIG.llm_profiles['quality']['model']
    provider_url = APP_CONFIG.providers['ollama'].base_url
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================================
# Helper Functions
# ============================================================================

def _env(key: str, default: str = "") -> str:
    """Get string environment variable."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_optional_int(key: str) -> Optional[int]:
    value = os.getenv(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(key: str, default: float) -> float:
    """Get float environment variable."""
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.environ.get(key, "").lower()
    if value in ("1", "true", "yes", "on"):
        return True
    elif value in ("0", "false", "no", "off"):
        return False
    return default


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass(frozen=True)
class DatabaseConfig:
    """MongoDB configuration."""
    uri: str
    db_name: str
    documents_collection: str
    chunks_collection: str
    connection_timeout_ms: int
    server_selection_timeout_ms: int


@dataclass(frozen=True)
class ChromaConfig:
    """ChromaDB configuration."""
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding service configuration."""
    provider: str  # "local", "openai", "ollama"
    model: str
    dimension: int
    batch_size: int


@dataclass(frozen=True)
class RetrieverConfig:
    """Retrieval configuration."""
    top_k: int
    retrieval_pool_size: int
    parent_retrieval_cap: int
    vector_weight: float
    keyword_weight: float
    rrf_k: int


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration (legacy - for backward compatibility)."""
    model: str
    base_url: str
    temperature: float
    num_predict: int
    timeout_s: float


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a specific LLM provider."""
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 120.0
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdversarialConfig:
    """Adversarial verification configuration."""
    confidence_threshold: float
    min_score_fallback: int
    max_retries: int


@dataclass(frozen=True)
class Tier0Config:
    """Tier 0 (corpus exploration) configuration."""
    exploration_budget: int
    exploration_strategy: str
    full_corpus: bool
    llm_cache_enabled: bool
    llm_cache_dir: str
    synthesis_checkpoint_dir: str
    extract_dates_strict: bool
    group_indicator_min_docs: int
    batch_max_chars: int
    sub_batch_docs: int
    llm_timeout: int
    heartbeat_seconds: int
    batch_profile: str
    semantic_chunking: bool
    block_max_chars: int
    max_blocks_per_doc: int
    min_entities_per_batch: int
    min_patterns_per_batch: int
    repair_attempts: int
    repair_min_docs: int
    strict_closed_world: bool
    notebook_save_dir: str
    notebook_auto_save: bool
    tier0_log_dir: str
    tier0_debug_mode: bool
    question_min_evidence_docs: int
    question_per_type: int
    question_min_score: int
    question_min_score_refine: int
    question_max_refinements: int
    question_target_count: int
    question_min_count: int
    question_enforce_type_diversity: bool
    question_min_types: int
    answerability_min_docs: int
    answerability_max_docs: int
    answerability_top_k: int
    synthesis_enabled: bool
    synthesis_dynamic: bool
    synthesis_semantic_assignment: bool
    synthesis_embed_provider: str
    synthesis_embed_model: str
    synthesis_embed_cache: str
    synthesis_embed_timeout: int
    synthesis_assign_min_sim: float
    synthesis_dedupe_threshold: float
    synthesis_cluster_threshold: float
    synthesis_theme_merge_threshold: float
    synthesis_theme_count: int
    synthesis_min_themes: int
    synthesis_max_question_sample: int
    synthesis_max_pattern_sample: int
    synthesis_narrative_enabled: bool
    synthesis_narrative_max_themes: int
    recursive_enabled: bool
    recursive_min_docs: int
    recursive_max_docs: int
    recursive_max_depth: int
    recursive_subquestion_count: int
    recursive_leaf_profile: str
    recursive_writer_profile: str
    essay_min_words: int
    recursive_theme_max_leaves: int
    leaf_answers_collection: str
    pattern_merge_threshold: float
    runs_collection: str
    runs_store_notebook: bool
    doc_cache_enabled: bool
    doc_cache_mode: str
    doc_cache_collection: str
    doc_cache_prompt_version: str


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""
    
    # Database
    database: DatabaseConfig
    chroma: ChromaConfig
    
    # Embeddings
    embedding: EmbeddingConfig
    
    # Retrieval
    retriever: RetrieverConfig
    
    # LLM (legacy - for backward compatibility)
    llm_generator: LLMConfig
    llm_verifier: LLMConfig
    
    # NEW: Provider configurations
    providers: Dict[str, ProviderConfig]
    
    # NEW: LLM profiles for easy selection
    llm_profiles: Dict[str, Dict[str, Any]]
    
    # Adversarial
    adversarial: AdversarialConfig

    # Tier 0 (corpus exploration)
    tier0: Tier0Config
    
    # Application
    debug_mode: bool
    flask_debug: bool
    secret_key: str
    session_dir: Optional[str]


# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    """Loads configuration from environment variables."""
    
    @staticmethod
    def from_env() -> AppConfig:
        """Load complete configuration from environment."""
        
        # --- Database Configuration ---
        database = DatabaseConfig(
            uri=_env("APP_MONGO_URI") or _env("MONGO_URI", "mongodb://localhost:27017"),
            db_name=_env("MONGO_DB_NAME", "railroad_documents"),
            documents_collection=_env("DOCUMENTS_COLLECTION", "documents"),
            chunks_collection=_env("CHUNKS_COLLECTION", "chunks"),
            connection_timeout_ms=_env_int("MONGO_CONNECT_TIMEOUT_MS", 5000),
            server_selection_timeout_ms=_env_int("MONGO_SERVER_TIMEOUT_MS", 5000)
        )
        
        # --- ChromaDB Configuration ---
        chroma = ChromaConfig(
            persist_directory=_env("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
            collection_name=_env("CHROMA_COLLECTION_NAME", "historian_documents")
        )
        
        # --- Embedding Configuration ---
        embedding = EmbeddingConfig(
            provider=_env("EMBEDDING_PROVIDER", "ollama"),
            model=_env("HISTORIAN_AGENT_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
            dimension=_env_int("EMBEDDING_DIMENSION", 1024),
            batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32)
        )
        
        # --- Retrieval Configuration ---
        retriever = RetrieverConfig(
            top_k=_env_int("HISTORIAN_AGENT_TOP_K", 5),
            retrieval_pool_size=_env_int("RETRIEVAL_POOL_SIZE", 40),
            parent_retrieval_cap=_env_int("PARENT_RETRIEVAL_CAP", 8),
            vector_weight=_env_float("VECTOR_WEIGHT", 0.7),
            keyword_weight=_env_float("KEYWORD_WEIGHT", 0.3),
            rrf_k=_env_int("RRF_K", 60)
        )
        
        # --- Legacy LLM Configuration (backward compatibility) ---
        ollama_base_url = _env("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        
        llm_generator = LLMConfig(
            model=_env("LLM_MODEL", "qwen2.5:32b"),
            base_url=ollama_base_url,
            temperature=_env_float("LLM_TEMPERATURE", 0.2),
            num_predict=_env_int("LLM_NUM_PREDICT", 4000),
            timeout_s=_env_float("LLM_TIMEOUT", 120.0)
        )
        
        llm_verifier = LLMConfig(
            model=_env("VERIFIER_MODEL", "qwen2.5:32b"),
            base_url=ollama_base_url,
            temperature=0.0,  # Verifier always uses 0 temperature
            num_predict=_env_int("VERIFIER_NUM_PREDICT", 500),
            timeout_s=_env_float("VERIFIER_TIMEOUT", 60.0)
        )
        
        # --- NEW: Provider Configurations ---
        providers = {
            "ollama": ProviderConfig(
                base_url=ollama_base_url,
                timeout=_env_float("OLLAMA_TIMEOUT", 120.0),
                options={
                    "num_ctx": _env_int("OLLAMA_NUM_CTX", 131072),
                    "repeat_penalty": _env_float("OLLAMA_REPEAT_PENALTY", 1.15),
                    "num_gpu": _env_optional_int("OLLAMA_NUM_GPU"),
                    "num_batch": _env_optional_int("OLLAMA_NUM_BATCH"),
                    "num_thread": _env_optional_int("OLLAMA_NUM_THREAD"),
                }
            ),
            "openai": ProviderConfig(
                api_key=_env("OPENAI_API_KEY", ""),
                timeout=_env_float("OPENAI_TIMEOUT", 30.0),
                options={}
            ),
            "lmstudio": ProviderConfig(
                base_url=_env("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                timeout=_env_float("LMSTUDIO_TIMEOUT", 120.0),
                options={}
            ),
        }
        
        # --- NEW: LLM Profiles ---
        llm_profiles = {
            # Fast local model for quick operations (multi-query generation, etc.)
            "fast": {
                "provider": "ollama",
                "model": _env("LLM_FAST_MODEL", "llama3.2:3b"),
                "temperature": 0.3,
                "timeout": 30.0,
            },
            
            # Quality local model for main generation
            "quality": {
                "provider": "ollama",
                "model": _env("LLM_MODEL", "qwen2.5:32b"),
                "temperature": 0.2,
                "timeout": 120.0,
            },
            
            # Verifier model for fact-checking (strict, deterministic)
            "verifier": {
                "provider": "ollama",
                "model": _env("VERIFIER_MODEL", "qwen2.5:32b"),
                "temperature": 0.0,
                "timeout": 60.0,
            },
            
            # Cloud fallback for when local models are unavailable
            "cloud": {
                "provider": "openai",
                "model": _env("OPENAI_MODEL", "gpt-4"),
                "temperature": 0.2,
                "timeout": 30.0,
            },
        }
        
        # --- Adversarial Configuration ---
        adversarial = AdversarialConfig(
            confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", 0.9),
            min_score_fallback=_env_int("MIN_SCORE_FALLBACK", 75),
            max_retries=_env_int("VERIFIER_MAX_RETRIES", 3)
        )

        # --- Tier 0 Configuration ---
        tier0 = Tier0Config(
            exploration_budget=_env_int("TIER0_EXPLORATION_BUDGET", 2000),
            exploration_strategy=_env("TIER0_EXPLORATION_STRATEGY", "balanced"),
            full_corpus=_env_bool("TIER0_FULL_CORPUS", False),
            llm_cache_enabled=_env_bool("TIER0_LLM_CACHE_ENABLED", True),
            llm_cache_dir=_env("TIER0_LLM_CACHE_DIR", "/app/logs/llm_cache"),
            synthesis_checkpoint_dir=_env("TIER0_SYNTHESIS_CHECKPOINT_DIR", "/app/logs/synthesis_checkpoints"),
            extract_dates_strict=_env_bool("TIER0_EXTRACT_DATES_STRICT", True),
            group_indicator_min_docs=_env_int("TIER0_GROUP_INDICATOR_MIN_DOCS", 3),
            batch_max_chars=_env_int("TIER0_BATCH_MAX_CHARS", 60000),
            sub_batch_docs=_env_int("TIER0_SUB_BATCH_DOCS", 10),
            llm_timeout=_env_int("TIER0_LLM_TIMEOUT", 300),
            heartbeat_seconds=_env_int("TIER0_HEARTBEAT_SECONDS", 60),
            batch_profile=_env("TIER0_BATCH_PROFILE", "fast"),
            semantic_chunking=_env_bool("TIER0_SEMANTIC_CHUNKING", True),
            block_max_chars=_env_int("TIER0_BLOCK_MAX_CHARS", 2000),
            max_blocks_per_doc=_env_int("TIER0_MAX_BLOCKS_PER_DOC", 12),
            min_entities_per_batch=_env_int("TIER0_MIN_ENTITIES_PER_BATCH", 5),
            min_patterns_per_batch=_env_int("TIER0_MIN_PATTERNS_PER_BATCH", 2),
            repair_attempts=_env_int("TIER0_REPAIR_ATTEMPTS", 1),
            repair_min_docs=_env_int("TIER0_REPAIR_MIN_DOCS", 6),
            strict_closed_world=_env_bool("TIER0_STRICT_CLOSED_WORLD", True),
            notebook_save_dir=_env("NOTEBOOK_SAVE_DIR", "/app/logs/corpus_exploration"),
            notebook_auto_save=_env_bool("NOTEBOOK_AUTO_SAVE", True),
            tier0_log_dir=_env("TIER0_LOG_DIR", "/app/logs/tier0"),
            tier0_debug_mode=_env_bool("TIER0_DEBUG_MODE", False),
            question_min_evidence_docs=_env_int("TIER0_QUESTION_MIN_EVIDENCE_DOCS", 5),
            question_per_type=_env_int("TIER0_QUESTION_PER_TYPE", 4),
            question_min_score=_env_int("TIER0_QUESTION_MIN_SCORE", 60),
            question_min_score_refine=_env_int("TIER0_QUESTION_MIN_SCORE_REFINE", 50),
            question_max_refinements=_env_int("TIER0_QUESTION_MAX_REFINEMENTS", 2),
            question_target_count=_env_int("TIER0_QUESTION_TARGET_COUNT", 12),
            question_min_count=_env_int("TIER0_QUESTION_MIN_COUNT", 8),
            question_enforce_type_diversity=_env_bool("TIER0_QUESTION_ENFORCE_TYPE_DIVERSITY", True),
            question_min_types=_env_int("TIER0_QUESTION_MIN_TYPES", 3),
            answerability_min_docs=_env_int("TIER0_ANSWERABILITY_MIN_DOCS", 5),
            answerability_max_docs=_env_int("TIER0_ANSWERABILITY_MAX_DOCS", 200),
            answerability_top_k=_env_int("TIER0_ANSWERABILITY_TOP_K", 50),
            synthesis_enabled=_env_bool("TIER0_SYNTHESIS_ENABLED", True),
            synthesis_dynamic=_env_bool("TIER0_SYNTHESIS_DYNAMIC", True),
            synthesis_semantic_assignment=_env_bool("TIER0_SYNTHESIS_SEMANTIC_ASSIGNMENT", True),
            synthesis_embed_provider=_env("TIER0_SYNTHESIS_EMBED_PROVIDER", "ollama"),
            synthesis_embed_model=_env("TIER0_SYNTHESIS_EMBED_MODEL", "qwen3-embedding:0.6b"),
            synthesis_embed_cache=_env("TIER0_SYNTHESIS_EMBED_CACHE", "/app/logs/embedding_cache.pkl"),
            synthesis_embed_timeout=_env_int("TIER0_SYNTHESIS_EMBED_TIMEOUT", 120),
            synthesis_assign_min_sim=_env_float("TIER0_SYNTHESIS_ASSIGN_MIN_SIM", 0.2),
            synthesis_dedupe_threshold=_env_float("TIER0_SYNTHESIS_DEDUPE_THRESHOLD", 0.86),
            synthesis_cluster_threshold=_env_float("TIER0_SYNTHESIS_CLUSTER_THRESHOLD", 0.78),
            synthesis_theme_merge_threshold=_env_float("TIER0_SYNTHESIS_THEME_MERGE_THRESHOLD", 0.84),
            synthesis_theme_count=_env_int("TIER0_SYNTHESIS_THEME_COUNT", 5),
            synthesis_min_themes=_env_int("TIER0_SYNTHESIS_MIN_THEMES", 4),
            synthesis_max_question_sample=_env_int("TIER0_SYNTHESIS_MAX_QUESTION_SAMPLE", 24),
            synthesis_max_pattern_sample=_env_int("TIER0_SYNTHESIS_MAX_PATTERN_SAMPLE", 12),
            synthesis_narrative_enabled=_env_bool("TIER0_SYNTHESIS_NARRATIVE_ENABLED", True),
            synthesis_narrative_max_themes=_env_int("TIER0_SYNTHESIS_NARRATIVE_MAX_THEMES", 5),
            recursive_enabled=_env_bool("TIER0_RECURSIVE_ENABLED", False),
            recursive_min_docs=_env_int("TIER0_RECURSIVE_MIN_DOCS", 5),
            recursive_max_docs=_env_int("TIER0_RECURSIVE_MAX_DOCS", 15),
            recursive_max_depth=_env_int("TIER0_RECURSIVE_MAX_DEPTH", 3),
            recursive_subquestion_count=_env_int("TIER0_RECURSIVE_SUBQUESTION_COUNT", 3),
            recursive_leaf_profile=_env("TIER0_RECURSIVE_LEAF_PROFILE", "quality"),
            recursive_writer_profile=_env("TIER0_RECURSIVE_WRITER_PROFILE", "quality"),
            essay_min_words=_env_int("TIER0_ESSAY_MIN_WORDS", 1200),
            recursive_theme_max_leaves=_env_int("TIER0_RECURSIVE_THEME_MAX_LEAVES", 20),
            leaf_answers_collection=_env("TIER0_LEAF_ANSWERS_COLLECTION", "tier0_leaf_answers"),
            pattern_merge_threshold=_env_float("TIER0_PATTERN_MERGE_THRESHOLD", 0.9),
            runs_collection=_env("TIER0_RUNS_COLLECTION", "tier0_runs"),
            runs_store_notebook=_env_bool("TIER0_RUNS_STORE_NOTEBOOK", True),
            doc_cache_enabled=_env_bool("TIER0_DOC_CACHE_ENABLED", True),
            doc_cache_mode=_env("TIER0_DOC_CACHE_MODE", "use"),
            doc_cache_collection=_env("TIER0_DOC_CACHE_COLLECTION", "tier0_doc_cache"),
            doc_cache_prompt_version=_env("TIER0_DOC_CACHE_PROMPT_VERSION", "v1"),
        )
        
        # --- Application Configuration ---
        debug_mode = _env_bool("DEBUG_MODE", False)
        flask_debug = _env_bool("FLASK_DEBUG", False)
        secret_key = _env("SECRET_KEY", "change-me-in-production")
        session_dir = _env("SESSION_FILE_DIR") or None
        
        return AppConfig(
            database=database,
            chroma=chroma,
            embedding=embedding,
            retriever=retriever,
            llm_generator=llm_generator,
            llm_verifier=llm_verifier,
            providers=providers,
            llm_profiles=llm_profiles,
            adversarial=adversarial,
            tier0=tier0,
            debug_mode=debug_mode,
            flask_debug=flask_debug,
            secret_key=secret_key,
            session_dir=session_dir
        )


# ============================================================================
# Config Merging Utilities
# ============================================================================

def merge_config(base: Any, overrides: Dict[str, Any]) -> Any:
    """
    Merge configuration overrides with base config.
    
    Args:
        base: Base configuration object (dataclass or dict)
        overrides: Dict of values to override
    
    Returns:
        New configuration object with overrides applied
    """
    if isinstance(base, dict):
        result = base.copy()
        result.update(overrides)
        return result
    
    # Dataclass - convert to dict, merge, return dict
    if hasattr(base, '__dataclass_fields__'):
        result = {k: getattr(base, k) for k in base.__dataclass_fields__}
        result.update(overrides)
        return result
    
    # Unknown type - return overrides
    return overrides


# ============================================================================
# Global Configuration Instance
# ============================================================================

# Load configuration once at module import
APP_CONFIG = ConfigLoader.from_env()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "APP_CONFIG",
    "AppConfig",
    "DatabaseConfig",
    "ChromaConfig",
    "EmbeddingConfig",
    "RetrieverConfig",
    "LLMConfig",
    "ProviderConfig",
    "AdversarialConfig",
    "Tier0Config",
    "merge_config",
]


# ============================================================================
# Configuration Validation (Optional - Run on Import)
# ============================================================================

def _validate_config():
    """Validate critical configuration at startup."""
    issues = []
    
    # Check MongoDB URI
    if not APP_CONFIG.database.uri:
        issues.append("MONGO_URI not set")
    
    # Check Ollama URL for default provider
    if APP_CONFIG.providers['ollama'].base_url is None:
        issues.append("OLLAMA_BASE_URL not set")
    
    # Check OpenAI key if using cloud profile
    if (APP_CONFIG.llm_profiles.get('cloud', {}).get('provider') == 'openai' 
        and not APP_CONFIG.providers['openai'].api_key):
        issues.append("OPENAI_API_KEY not set but 'cloud' profile uses OpenAI")
    
    # Warn about secret key
    if APP_CONFIG.secret_key == "change-me-in-production":
        issues.append("SECRET_KEY still set to default - should be changed in production")
    
    if issues and APP_CONFIG.debug_mode:
        import sys
        sys.stderr.write("\n⚠️  Configuration Issues:\n")
        for issue in issues:
            sys.stderr.write(f"   - {issue}\n")
        sys.stderr.write("\n")


# Run validation on import
_validate_config()
