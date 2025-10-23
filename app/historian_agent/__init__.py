"""Historian agent LangChain integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import os
import re
import threading

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSerializable

try:  # Import lazily to keep optional dependencies optional during runtime init
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - handled at runtime when provider is used
    ChatOpenAI = None  # type: ignore

try:
    from langchain_community.chat_models import ChatOllama
except Exception:  # pragma: no cover - handled dynamically
    ChatOllama = None  # type: ignore


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
            os.environ.get("OLLAMA_BASE_URL") or None,
        )
        if "openai_api_key" in env and env["openai_api_key"] == "":
            env["openai_api_key"] = None
        env.setdefault(
            "openai_api_key",
            os.environ.get("OPENAI_API_KEY") or None,
        )
        return cls(**env)


class HistorianAgentError(RuntimeError):
    """Raised when the Historian Agent cannot respond."""


class MongoKeywordRetriever:
    """Simple keyword-based retriever over a Mongo collection."""

    def __init__(self, collection, config: HistorianAgentConfig):
        self._collection = collection
        self._config = config

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Return a list of LangChain ``Document`` objects that match ``query``."""

        if not query:
            return []

        regex = re.compile(re.escape(query), re.IGNORECASE)
        filters = [
            {field: {"$regex": regex}}
            for field in self._config.context_fields
        ]
        mongo_query = {"$or": filters} if filters else {}
        cursor = self._collection.find(mongo_query).limit(self._config.max_context_documents)
        documents: List[Document] = []

        for record in cursor:
            metadata = {
                "_id": str(record.get("_id")),
                "title": record.get("title") or record.get("name") or record.get("document_title"),
            }
            content_segments: List[str] = []
            for field in self._config.context_fields:
                value = record.get(field)
                if isinstance(value, str):
                    content_segments.append(value)
                elif isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
                    # Flatten iterable values into joined text
                    content_segments.append(" ".join(map(str, value)))
                elif isinstance(value, dict):
                    content_segments.append(" ".join(map(str, value.values())))
            page_content = "\n".join(seg for seg in content_segments if seg)
            if not page_content and self._config.allow_general_fallback:
                summary_candidate = record.get(self._config.summary_field)
                if isinstance(summary_candidate, str):
                    page_content = summary_candidate
            if not page_content:
                continue
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents


class HistorianAgent:
    """Wrapper around a LangChain Runnable that orchestrates retrieval + response."""

    def __init__(
        self,
        config: HistorianAgentConfig,
        retriever: MongoKeywordRetriever,
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
        base_url = config.ollama_base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
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
    )


def get_agent(collection, overrides: Optional[Dict[str, Any]] = None) -> HistorianAgent:
    global _cached_agent, _cached_signature
    with _agent_lock:
        config = HistorianAgentConfig.from_env(overrides)
        signature = _config_signature(config)
        if _cached_agent is not None and _cached_signature == signature:
            return _cached_agent
        retriever = MongoKeywordRetriever(collection, config)
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
    "get_agent",
    "reset_agent",
]
