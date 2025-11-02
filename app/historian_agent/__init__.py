"""Historian agent LangChain integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging  # Added to log vector fallback events for RAG mode.
import os
import threading

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSerializable
from langchain_core.retrievers import BaseRetriever  # Added to type annotate pluggable retrievers.

try:  # Import lazily to keep optional dependencies optional during runtime init
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - handled at runtime when provider is used
    ChatOpenAI = None  # type: ignore

try:
    from langchain_community.chat_models import ChatOllama
except Exception:  # pragma: no cover - handled dynamically
    ChatOllama = None  # type: ignore

from .embeddings import EmbeddingService  # Added to construct embedding client when semantic search is enabled.
from .retrievers import (  # Added hybrid/vector retrievers per RAG design.
    HybridRetriever,
    KeywordRetriever,
    MongoKeywordRetriever,
    VectorRetriever,
)
from .vector_store import get_vector_store  # Switched to factory helper to honour configuration defaults per design docs.


logger = logging.getLogger(__name__)  # Added module-level logger for diagnostics.


def _coerce_bool(value: Any) -> bool:
    """Return ``True`` for truthy string/int/bool values and ``False`` otherwise."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


@dataclass
class HistorianAgentConfig:
    """Configuration for the Historian Agent pipeline."""

    enabled: bool = True
    model_provider: str = "ollama"
    model_name: str = "llama3"
    temperature: float = 0.2
    max_context_documents: int = 4
    system_prompt: str = (
        "You are the Historian Agent, an expert archival researcher. "
        "Synthesize clear, sourced answers using the provided context."
    )
    context_fields: Tuple[str, ...] = field(
        default_factory=lambda: ("title", "content")
    )
    summary_field: str = "content"
    allow_general_fallback: bool = True
    ollama_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    use_vector_retrieval: bool = False  # Added to toggle hybrid/vector retrieval pipeline.
    embedding_provider: str = "local"  # Added to choose between local and OpenAI embeddings.
    embedding_model: str = "all-MiniLM-L6-v2"  # Added to configure embedding model selection.
    chunk_size: int = 1000  # Added to propagate chunking settings to dependent services.
    chunk_overlap: int = 200  # Added to keep overlap consistent with chunker defaults.
    vector_store_type: str = "chroma"  # Added to allow switching vector backend implementations.
    chroma_persist_directory: Optional[str] = None  # Added to configure Chroma persistence path.
    hybrid_alpha: float = 0.5  # Added to balance vector vs keyword contributions in hybrid search.

    @classmethod
    def from_env(
        cls,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "HistorianAgentConfig":
        env = overrides.copy() if overrides else {}
        defaults = cls()
        if "enabled" in env:
            env["enabled"] = _coerce_bool(env["enabled"])
        env.setdefault("enabled", os.environ.get("HISTORIAN_AGENT_ENABLED", "1") != "0")
        env.setdefault(
            "model_provider",
            os.environ.get("HISTORIAN_AGENT_MODEL_PROVIDER", "ollama"),
        )
        env.setdefault(
            "model_name",
            os.environ.get("HISTORIAN_AGENT_MODEL", "llama3"),
        )
        if "temperature" in env:
            env["temperature"] = float(env["temperature"])
        env.setdefault(
            "temperature",
            float(os.environ.get("HISTORIAN_AGENT_TEMPERATURE", "0.2")),
        )
        if "max_context_documents" in env:
            env["max_context_documents"] = int(env["max_context_documents"])
        env.setdefault(
            "max_context_documents",
            int(os.environ.get("HISTORIAN_AGENT_CONTEXT_K", "4")),
        )
        env.setdefault(
            "system_prompt",
            os.environ.get(
                "HISTORIAN_AGENT_SYSTEM_PROMPT",
                cls.system_prompt,
            ),
        )
        context_fields = env.get("context_fields")
        if context_fields is None:
            raw_fields = os.environ.get("HISTORIAN_AGENT_CONTEXT_FIELDS")
            if raw_fields:
                context_fields = [
                    field.strip() for field in raw_fields.split(",") if field.strip()
                ]
            else:
                context_fields = list(defaults.context_fields)
        elif isinstance(context_fields, str):
            context_fields = [
                field.strip() for field in context_fields.split(",") if field.strip()
            ]
        elif isinstance(context_fields, Iterable):
            context_fields = list(context_fields)
        else:
            context_fields = list(defaults.context_fields)
        env["context_fields"] = tuple(context_fields)
        env.setdefault(
            "summary_field",
            os.environ.get("HISTORIAN_AGENT_SUMMARY_FIELD", "content"),
        )
        env.setdefault(
            "allow_general_fallback",
            _coerce_bool(os.environ.get("HISTORIAN_AGENT_FALLBACK", "1")),
        )
        env["allow_general_fallback"] = _coerce_bool(env["allow_general_fallback"])
        if "ollama_base_url" in env and env["ollama_base_url"] == "":
            env["ollama_base_url"] = None
        env.setdefault(
            "ollama_base_url",
            os.environ.get("HISTORIAN_AGENT_OLLAMA_BASE_URL")
            or os.environ.get("OLLAMA_BASE_URL")
            or None,
        )  # Added legacy env fallback so older deployments using HISTORIAN_AGENT_OLLAMA_BASE_URL keep working.
        if "openai_api_key" in env and env["openai_api_key"] == "":
            env["openai_api_key"] = None
        env.setdefault(
            "openai_api_key",
            os.environ.get("OPENAI_API_KEY") or None,
        )
        env.setdefault(
            "use_vector_retrieval",
            _coerce_bool(
                os.environ.get("HISTORIAN_AGENT_USE_VECTOR_RETRIEVAL", "false")
            ),
        )
        env["use_vector_retrieval"] = _coerce_bool(env["use_vector_retrieval"])
        env.setdefault(
            "embedding_provider",
            os.environ.get("HISTORIAN_AGENT_EMBEDDING_PROVIDER", "local"),
        )
        env.setdefault(
            "embedding_model",
            os.environ.get("HISTORIAN_AGENT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )
        if "chunk_size" in env:
            env["chunk_size"] = int(env["chunk_size"])
        env.setdefault(
            "chunk_size",
            int(os.environ.get("HISTORIAN_AGENT_CHUNK_SIZE", "1000")),
        )
        if "chunk_overlap" in env:
            env["chunk_overlap"] = int(env["chunk_overlap"])
        env.setdefault(
            "chunk_overlap",
            int(os.environ.get("HISTORIAN_AGENT_CHUNK_OVERLAP", "200")),
        )
        env.setdefault(
            "vector_store_type",
            os.environ.get("HISTORIAN_AGENT_VECTOR_STORE", "chroma"),
        )
        env.setdefault(
            "chroma_persist_directory",
            os.environ.get("CHROMA_PERSIST_DIRECTORY") or None,
        )
        if "hybrid_alpha" in env:
            env["hybrid_alpha"] = float(env["hybrid_alpha"])
        env.setdefault(
            "hybrid_alpha",
            float(os.environ.get("HISTORIAN_AGENT_HYBRID_ALPHA", "0.5")),
        )
        return cls(**env)


class HistorianAgentError(RuntimeError):
    """Raised when the Historian Agent cannot respond."""


class HistorianAgent:
    """Wrapper around a LangChain Runnable that orchestrates retrieval + response."""

    def __init__(
        self,
        config: HistorianAgentConfig,
        retriever: BaseRetriever,  # Broadened to accept keyword/vector/hybrid retrievers.
        chain: RunnableSerializable,
    ) -> None:
        self._config = config
        self._retriever = retriever
        self._chain = chain

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def config(self) -> HistorianAgentConfig:
        return self._config

    def invoke(
        self,
        question: str,
        chat_history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> Dict[str, object]:
        """Run the Historian Agent chain and return the answer with cited sources."""

        if not self.enabled:
            raise HistorianAgentError("Historian agent is disabled via configuration.")
        if not question:
            raise HistorianAgentError("Question is required.")

        documents = self._retriever.get_relevant_documents(question)
        if not documents:
            raise HistorianAgentError("No relevant context was found for the request.")

        context = self._render_context(documents)
        history_messages = self._convert_history(chat_history or [])
        chain_input = {
            "system_prompt": self._config.system_prompt,
            "question": question,
            "context": context,
            "chat_history": history_messages,
        }
        answer = self._chain.invoke(chain_input)
        return {
            "answer": answer.strip(),
            "sources": self._serialise_sources(documents),
        }

    def _render_context(self, documents: Sequence[Document]) -> str:
        """Render retrieved documents into the prompt-friendly context block."""

        rendered_segments = []
        for index, doc in enumerate(documents, start=1):
            title = doc.metadata.get("title") or f"Document {index}"
            rendered_segments.append(f"[{index}] {title}\n{doc.page_content.strip()}")
        return "\n\n".join(rendered_segments)

    @staticmethod
    def _serialise_sources(documents: Sequence[Document]) -> List[Dict[str, str]]:
        """Convert ``Document`` objects to serialisable citation metadata."""

        serialised = []
        for index, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            snippet = doc.page_content[:400].strip()
            serialised.append(
                {
                    "reference": f"[{index}]",
                    "id": metadata.get("_id", ""),
                    "title": metadata.get("title", f"Document {index}"),
                    "snippet": snippet,
                }
            )
        return serialised

    @staticmethod
    def _convert_history(history: Sequence[Dict[str, str]]) -> List[BaseMessage]:
        """Transform chat history dicts into LangChain message objects."""

        messages: List[BaseMessage] = []
        for turn in history:
            role = (turn.get("role") or "").lower()
            content = turn.get("content") or ""
            if not content:
                continue
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages


_agent_lock = threading.Lock()
_cached_agent: Optional[HistorianAgent] = None
_cached_signature: Optional[tuple] = None


def _build_llm(config: HistorianAgentConfig) -> Runnable:
    """Instantiate the LangChain LLM client for the configured provider."""

    provider = config.model_provider.lower()
    if provider == "openai":
        if ChatOpenAI is None:
            raise HistorianAgentError(
                "langchain-openai is not available. Install it or choose another provider.")
        api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HistorianAgentError("OPENAI_API_KEY is required for the OpenAI provider.")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )
    if provider == "ollama":
        if ChatOllama is None:
            raise HistorianAgentError(
                "langchain-community with Ollama support is not available. Install it or choose another provider.")
        base_url = (
            config.ollama_base_url
            or os.environ.get("HISTORIAN_AGENT_OLLAMA_BASE_URL")
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        )  # Added HISTORIAN_AGENT_OLLAMA_BASE_URL fallback so runtime respects both env names documented elsewhere.
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=base_url,
        )
    raise HistorianAgentError(f"Unsupported model provider: {config.model_provider}")


def _build_chain(llm: Runnable) -> RunnableSerializable:
    """Assemble the Historian Agent runnable chain from prompt, LLM, and parser."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Using the following context, answer the question.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Provide a concise, well-structured answer and cite the provided references "
                "by their bracketed numbers when relevant.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def _config_signature(config: HistorianAgentConfig) -> tuple:
    return (
        config.enabled,
        config.model_provider,
        config.model_name,
        config.temperature,
        config.max_context_documents,
        config.system_prompt,
        tuple(config.context_fields),
        config.summary_field,
        config.allow_general_fallback,
        config.ollama_base_url,
        config.openai_api_key,
        config.use_vector_retrieval,  # Added to bust cache when semantic retrieval toggles.
        config.embedding_provider,  # Added to reflect embedding backend changes.
        config.embedding_model,  # Added to refresh agent when model swaps.
        config.chunk_size,  # Added to ensure chunking adjustments rebuild components.
        config.chunk_overlap,  # Added to cover overlap changes.
        config.vector_store_type,  # Added to capture backend selection.
        config.chroma_persist_directory,  # Added to reinitialise vector store on path change.
        config.hybrid_alpha,  # Added to account for retrieval weighting tweaks.
    )


def get_agent(collection, overrides: Optional[Dict[str, Any]] = None) -> HistorianAgent:
    global _cached_agent, _cached_signature
    with _agent_lock:
        config = HistorianAgentConfig.from_env(overrides)
        signature = _config_signature(config)
        if _cached_agent is not None and _cached_signature == signature:
            return _cached_agent
        fallback_retriever = MongoKeywordRetriever(collection, config)
        retriever: BaseRetriever = fallback_retriever  # Default to keyword search for safety.
        if config.enabled and config.use_vector_retrieval:
            try:
                provider = (config.embedding_provider or "").strip().lower()
                if provider in {"local", "huggingface"}:
                    provider_name = "huggingface"
                    provider_kwargs = {}
                elif provider == "openai":
                    provider_name = "openai"
                    provider_kwargs = {"api_key": config.openai_api_key}
                else:
                    raise HistorianAgentError(
                        f"Unsupported embedding provider for vector retrieval: {config.embedding_provider}"
                    )

                embedding_service = EmbeddingService(  # Build embedding client for semantic search.
                    provider=provider_name,
                    model_name=config.embedding_model,
                    **provider_kwargs,
                )

                if config.vector_store_type.lower() != "chroma":
                    raise HistorianAgentError(
                        f"Unsupported vector store type: {config.vector_store_type}"
                    )

                persist_directory = (
                    config.chroma_persist_directory
                    or os.environ.get("CHROMA_PERSIST_DIRECTORY")
                    or "/home/claude/chroma_db"
                )

                chunks_collection = collection.database.get_collection("document_chunks")  # Added to query chunk-level material for semantic retrieval.
                if chunks_collection.estimated_document_count() == 0:
                    raise HistorianAgentError("document_chunks collection is empty; run the embedding migration before enabling vector mode")  # Added guard so we fall back when migration has not been executed.

                vector_store = get_vector_store(  # Use shared factory to ensure consistent collection naming + caching.
                    store_type=config.vector_store_type,
                    persist_directory=persist_directory,
                    collection_name="historian_document_chunks",
                )

                vector_retriever = VectorRetriever(
                    vector_store=vector_store,
                    embedding_service=embedding_service,
                    mongo_collection=chunks_collection,
                    top_k=max(1, config.max_context_documents * 2),  # Increased recall for fusion by fetching extra chunk candidates.
                )
                keyword_retriever = KeywordRetriever(
                    mongo_collection=chunks_collection,
                    config=config,
                    top_k=max(1, config.max_context_documents * 2),  # Keep keyword pool size aligned with vector retriever for RRF.
                )
                alpha = max(0.0, min(1.0, config.hybrid_alpha))
                retriever = HybridRetriever(
                    vector_retriever=vector_retriever,
                    keyword_retriever=keyword_retriever,
                    vector_weight=alpha,
                    keyword_weight=1.0 - alpha,
                    top_k=config.max_context_documents,
                )
            except Exception as exc:
                logger.warning(
                    "Vector retrieval initialisation failed; reverting to keyword search: %s",
                    exc,
                )
                retriever = fallback_retriever
        if not config.enabled:
            chain = RunnableLambda(lambda _: "Historian agent is currently disabled.")
            _cached_agent = HistorianAgent(config=config, retriever=retriever, chain=chain)
            _cached_signature = signature
            return _cached_agent
        llm = _build_llm(config)
        chain = _build_chain(llm)
        _cached_agent = HistorianAgent(config=config, retriever=retriever, chain=chain)
        _cached_signature = signature
        return _cached_agent


def reset_agent() -> None:
    global _cached_agent, _cached_signature
    with _agent_lock:
        _cached_agent = None
        _cached_signature = None


__all__ = [
    "HistorianAgent",
    "HistorianAgentConfig",
    "HistorianAgentError",
    "MongoKeywordRetriever",
    "KeywordRetriever",  # Added to expose new keyword retriever implementation.
    "VectorRetriever",  # Added to expose semantic retriever for external callers.
    "HybridRetriever",  # Added to expose hybrid retriever for integrations.
    "get_agent",
    "reset_agent",
]
