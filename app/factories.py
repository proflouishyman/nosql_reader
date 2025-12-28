# app/factories.py
# Created: 2025-12-23 22:00 America/New_York
# Purpose: Factory pattern for RAG system components with smart caching
#          - Cache default configs to avoid re-initialization
#          - Support per-request overrides without polluting cache
#          - Clear initialization flow with config merging

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
from threading import Lock
import hashlib
import json

from config import (
    APP_CONFIG,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrieverConfig,
    DatabaseConfig,
    merge_config,
)


# ============================================================================
# LLM Factory
# ============================================================================

class LLMFactory:
    """
    Factory for creating LLM instances with smart caching.
    
    Caching Strategy:
        - Default configs are cached (read from APP_CONFIG)
        - Overridden configs are NOT cached (per-request instances)
        - Cache key = hash of (provider, base_url, model, temperature, timeout)
    """
    
    _cache: Dict[str, Any] = {}
    _lock = Lock()
    
    @staticmethod
    def _config_hash(cfg: LLMConfig) -> str:
        """Generate stable hash of LLM config for cache keying."""
        key_parts = (
            cfg.provider,
            cfg.base_url,
            cfg.model,
            f"{cfg.temperature:.3f}",
            f"{cfg.timeout_s:.1f}",
            str(cfg.max_retries),
            str(cfg.num_predict),
        )
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    @staticmethod
    def create(
        cfg: LLMConfig,
        overrides: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ) -> Any:
        """
        Create an LLM instance.
        
        Args:
            cfg: Base LLM configuration
            overrides: Optional config overrides (disables caching)
            cache: Whether to cache this instance (default: True)
                   Automatically set to False if overrides are provided
        
        Returns:
            LLM instance (ChatOllama or ChatOpenAI)
        
        Examples:
            # Use default generator LLM (cached)
            llm = LLMFactory.create(APP_CONFIG.llm_generator)
            
            # Override temperature for one request (NOT cached)
            hot_llm = LLMFactory.create(
                APP_CONFIG.llm_generator,
                overrides={"temperature": 0.8}
            )
            
            # Use verifier with different model (NOT cached)
            strict = LLMFactory.create(
                APP_CONFIG.llm_verifier,
                overrides={"model": "qwen2.5:72b", "temperature": 0.0}
            )
        """
        # Apply overrides if provided
        if overrides:
            cfg = merge_config(cfg, overrides)
            cache = False  # Never cache overridden instances
        
        # Check cache
        if cache:
            cache_key = LLMFactory._config_hash(cfg)
            with LLMFactory._lock:
                if cache_key in LLMFactory._cache:
                    return LLMFactory._cache[cache_key]
        
        # Create new instance
        if cfg.provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                from langchain_ollama import ChatOllama
            
            llm = ChatOllama(
                model=cfg.model,
                temperature=cfg.temperature,
                base_url=cfg.base_url,
                timeout=cfg.timeout_s,
                num_predict=cfg.num_predict,
            )
        
        elif cfg.provider == "openai":
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model=cfg.model,
                temperature=cfg.temperature,
                timeout=cfg.timeout_s,
                max_retries=cfg.max_retries,
                max_tokens=cfg.num_predict if cfg.num_predict > 0 else None,
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {cfg.provider}")
        
        # Cache if enabled
        if cache:
            with LLMFactory._lock:
                LLMFactory._cache[cache_key] = llm
        
        return llm
    
    @staticmethod
    def create_generator(overrides: Optional[Dict[str, Any]] = None) -> Any:
        """Create generator LLM using default config."""
        return LLMFactory.create(APP_CONFIG.llm_generator, overrides=overrides)
    
    @staticmethod
    def create_verifier(overrides: Optional[Dict[str, Any]] = None) -> Any:
        """Create verifier LLM using default config."""
        return LLMFactory.create(APP_CONFIG.llm_verifier, overrides=overrides)
    
    @staticmethod
    def clear_cache():
        """Clear the LLM cache (useful for testing)."""
        with LLMFactory._lock:
            LLMFactory._cache.clear()


# ============================================================================
# Embedding Factory
# ============================================================================

class EmbeddingFactory:
    """Factory for creating embedding service instances."""
    
    _instance: Optional[Any] = None
    _lock = Lock()
    
    @staticmethod
    def create(
        cfg: Optional[EmbeddingConfig] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create embedding service.
        
        Args:
            cfg: Embedding configuration (defaults to APP_CONFIG.embedding)
            overrides: Optional config overrides
        
        Returns:
            EmbeddingService instance
        """
        from historian_agent.embeddings import EmbeddingService
        
        if cfg is None:
            cfg = APP_CONFIG.embedding
        
        if overrides:
            cfg = merge_config(cfg, overrides)
        
        # Singleton pattern for default config
        if overrides is None:
            with EmbeddingFactory._lock:
                if EmbeddingFactory._instance is None:
                    EmbeddingFactory._instance = EmbeddingService(
                        provider=cfg.provider,
                        model=cfg.model,
                        openai_api_key=cfg.openai_api_key or None,
                    )
                return EmbeddingFactory._instance
        
        # Non-cached for overrides
        return EmbeddingService(
            provider=cfg.provider,
            model=cfg.model,
            openai_api_key=cfg.openai_api_key or None,
        )
    
    @staticmethod
    def reset():
        """Reset singleton instance (for testing)."""
        with EmbeddingFactory._lock:
            EmbeddingFactory._instance = None


# ============================================================================
# Vector Store Factory
# ============================================================================

class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    _instance: Optional[Any] = None
    _lock = Lock()
    
    @staticmethod
    def create(
        cfg: Optional[VectorStoreConfig] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create vector store.
        
        Args:
            cfg: Vector store configuration (defaults to APP_CONFIG.vector_store)
            overrides: Optional config overrides
        
        Returns:
            VectorStore instance
        """
        from historian_agent.vector_store import get_vector_store
        
        if cfg is None:
            cfg = APP_CONFIG.vector_store
        
        if overrides:
            cfg = merge_config(cfg, overrides)
        
        # Singleton pattern for default config
        if overrides is None:
            with VectorStoreFactory._lock:
                if VectorStoreFactory._instance is None:
                    VectorStoreFactory._instance = get_vector_store(
                        store_type=cfg.type,
                        persist_directory=cfg.persist_directory,
                    )
                return VectorStoreFactory._instance
        
        # Non-cached for overrides
        return get_vector_store(
            store_type=cfg.type,
            persist_directory=cfg.persist_directory,
        )
    
    @staticmethod
    def reset():
        """Reset singleton instance (for testing)."""
        with VectorStoreFactory._lock:
            VectorStoreFactory._instance = None


# ============================================================================
# Database Factory
# ============================================================================

class DatabaseFactory:
    """Factory for creating MongoDB client and collection bundles."""
    
    _client: Optional[Any] = None
    _lock = Lock()
    
    @staticmethod
    def get_client(
        cfg: Optional[DatabaseConfig] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get MongoDB client (singleton).
        
        Args:
            cfg: Database configuration (defaults to APP_CONFIG.database)
            overrides: Optional config overrides
        
        Returns:
            MongoClient instance
        """
        from pymongo import MongoClient
        
        if cfg is None:
            cfg = APP_CONFIG.database
        
        if overrides:
            cfg = merge_config(cfg, overrides)
        
        # Singleton pattern for default config
        if overrides is None:
            with DatabaseFactory._lock:
                if DatabaseFactory._client is None:
                    DatabaseFactory._client = MongoClient(
                        cfg.uri,
                        serverSelectionTimeoutMS=cfg.timeout_ms,
                        maxPoolSize=cfg.max_pool_size,
                    )
                    # Test connection
                    DatabaseFactory._client.admin.command('ping')
                return DatabaseFactory._client
        
        # Non-cached for overrides
        client = MongoClient(
            cfg.uri,
            serverSelectionTimeoutMS=cfg.timeout_ms,
            maxPoolSize=cfg.max_pool_size,
        )
        client.admin.command('ping')
        return client
    
    @staticmethod
    def get_collections(
        client: Optional[Any] = None,
        cfg: Optional[DatabaseConfig] = None
    ) -> Dict[str, Any]:
        """
        Get standard collection bundle.
        
        Args:
            client: MongoClient instance (uses default if None)
            cfg: Database configuration (defaults to APP_CONFIG.database)
        
        Returns:
            Dict mapping collection names to collection objects
        """
        if client is None:
            client = DatabaseFactory.get_client(cfg)
        
        if cfg is None:
            cfg = APP_CONFIG.database
        
        db = client[cfg.db_name]
        
        return {
            'documents': db['documents'],
            'chunks': db['document_chunks'],
            'entities': db['linked_entities'],
            'persons': db['persons'],
        }
    
    @staticmethod
    def reset():
        """Close client and reset (for testing)."""
        with DatabaseFactory._lock:
            if DatabaseFactory._client is not None:
                DatabaseFactory._client.close()
                DatabaseFactory._client = None


# ============================================================================
# Retriever Factory
# ============================================================================

class RetrieverFactory:
    """Factory for creating retriever instances."""
    
    @staticmethod
    def create_hybrid(
        cfg: Optional[RetrieverConfig] = None,
        overrides: Optional[Dict[str, Any]] = None,
        vector_store: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
        chunks_collection: Optional[Any] = None,
    ) -> Any:
        """
        Create hybrid retriever.
        
        Args:
            cfg: Retriever configuration (defaults to APP_CONFIG.retriever)
            overrides: Optional config overrides
            vector_store: VectorStore instance (creates default if None)
            embedding_service: EmbeddingService instance (creates default if None)
            chunks_collection: MongoDB chunks collection (uses default if None)
        
        Returns:
            HybridRetriever instance
        """
        from historian_agent.retrievers import HybridRetriever
        
        if cfg is None:
            cfg = APP_CONFIG.retriever
        
        if overrides:
            cfg = merge_config(cfg, overrides)
        
        # Create dependencies if not provided
        if vector_store is None:
            vector_store = VectorStoreFactory.create()
        
        if embedding_service is None:
            embedding_service = EmbeddingFactory.create()
        
        if chunks_collection is None:
            collections = DatabaseFactory.get_collections()
            chunks_collection = collections['chunks']
        
        return HybridRetriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
            mongo_collection=chunks_collection,
            top_k=cfg.top_k,
            alpha=cfg.hybrid_alpha,
        )
    
    @staticmethod
    def create_keyword(
        cfg: Optional[RetrieverConfig] = None,
        overrides: Optional[Dict[str, Any]] = None,
        chunks_collection: Optional[Any] = None,
    ) -> Any:
        """
        Create keyword-only retriever.
        
        Args:
            cfg: Retriever configuration (defaults to APP_CONFIG.retriever)
            overrides: Optional config overrides
            chunks_collection: MongoDB chunks collection (uses default if None)
        
        Returns:
            MongoKeywordRetriever instance
        """
        from historian_agent.retrievers import MongoKeywordRetriever
        
        if cfg is None:
            cfg = APP_CONFIG.retriever
        
        if overrides:
            cfg = merge_config(cfg, overrides)
        
        if chunks_collection is None:
            collections = DatabaseFactory.get_collections()
            chunks_collection = collections['chunks']
        
        return MongoKeywordRetriever(
            collection=chunks_collection,
            top_k=cfg.top_k,
        )


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    print("=== LLM Factory Examples ===\n")
    
    # Default generator (cached)
    gen1 = LLMFactory.create_generator()
    gen2 = LLMFactory.create_generator()
    print(f"Same instance? {gen1 is gen2}")  # True
    
    # Override temperature (not cached)
    hot = LLMFactory.create_generator(overrides={"temperature": 0.8})
    print(f"Hot temp same as gen1? {hot is gen1}")  # False
    
    # Default verifier (cached separately)
    ver = LLMFactory.create_verifier()
    print(f"Verifier same as gen? {ver is gen1}")  # False
    
    print("\n=== Database Factory Examples ===\n")
    
    # Get client (singleton)
    client1 = DatabaseFactory.get_client()
    client2 = DatabaseFactory.get_client()
    print(f"Same client? {client1 is client2}")  # True
    
    # Get collections
    collections = DatabaseFactory.get_collections()
    print(f"Available collections: {list(collections.keys())}")
    
    print("\n=== Embedding Factory Examples ===\n")
    
    # Default embedding service (singleton)
    emb1 = EmbeddingFactory.create()
    emb2 = EmbeddingFactory.create()
    print(f"Same embedding service? {emb1 is emb2}")  # True
    
    # Override provider (not cached)
    openai_emb = EmbeddingFactory.create(overrides={"provider": "openai"})
    print(f"OpenAI same as default? {openai_emb is emb1}")  # False
