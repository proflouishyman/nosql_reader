# app/llm_abstraction.py
# Created: 2025-12-24
# Purpose: Provider-agnostic LLM interface with OpenAI-format messages

"""
LLM Abstraction Layer

This is the TOP layer that your RAG code interacts with.
Provides a clean, provider-agnostic interface using OpenAI message format.

Responsibilities:
- Input validation
- Generic logging
- Response standardization
- Optional retry logic
- Delegates to llm_layer.py for provider-specific implementation

Does NOT:
- Know about specific providers (Ollama, OpenAI, etc.)
- Handle provider-specific errors (llm_layer does this)
- Make HTTP calls directly (llm_layer does this)
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config import APP_CONFIG
from llm_layer import ProviderRouter, LLMProviderError


logger = logging.getLogger(__name__)


# ============================================================================
# Response Objects
# ============================================================================

@dataclass
class LLMResponse:
    """
    Standardized response from any LLM provider.
    
    All metrics flow through this object for consistency.
    """
    # Content
    content: str
    
    # Metadata
    provider_used: str
    model_name: str
    request_id: str
    
    # Metrics
    latency: float  # Seconds
    tokens: Dict[str, int] = field(default_factory=dict)  # prompt, completion, total
    
    # Status
    success: bool = True
    error: Optional[Exception] = None
    
    # Raw response (for debugging)
    raw_response: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation for logging."""
        status = "✓" if self.success else "✗"
        tokens_str = f"{self.tokens.get('total', 0)} tokens" if self.tokens else "unknown tokens"
        return (
            f"LLMResponse({status} {self.provider_used}/{self.model_name}, "
            f"{self.latency:.2f}s, {tokens_str})"
        )


# ============================================================================
# Main Abstraction Layer
# ============================================================================

class LLMClient:
    """
    Provider-agnostic LLM client.
    
    This is what your RAG code imports and uses.
    All provider complexity is hidden in llm_layer.py.
    
    Usage:
        llm = LLMClient()
        
        # Full control
        response = llm.generate(
            messages=[{"role": "user", "content": "Hello"}],
            provider="ollama",
            model="qwen2.5:32b"
        )
        
        # Simplified
        response = llm.generate_simple(
            prompt="Hello",
            system="You are helpful",
            provider="ollama"
        )
        
        # Using profiles
        response = llm.generate(
            messages=[...],
            profile="fast"  # Looks up from config
        )
    """
    
    def __init__(self):
        """Initialize LLM client with provider router."""
        self.router = ProviderRouter()
        logger.info("LLM abstraction layer initialized")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        profile: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        retry: bool = False,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using specified provider.
        
        Provider-agnostic interface - validates input, logs, handles retries.
        Delegates actual call to llm_layer.py.
        
        Args:
            messages: OpenAI-format messages list
            provider: Provider name ("ollama", "openai", "lmstudio")
            model: Model name (provider-specific)
            profile: Named profile from config (overrides provider/model)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            retry: Enable retry logic
            max_retries: Max retry attempts
            timeout: Request timeout in seconds
            **kwargs: Provider-specific options (passed through)
        
        Returns:
            LLMResponse with content and metrics
        
        Raises:
            ValueError: Invalid input
            LLMProviderError: Provider error after retries exhausted
        """
        # Validate input
        self._validate_messages(messages)
        
        # Resolve profile if specified
        if profile:
            profile_config = self._get_profile(profile)
            provider = provider or profile_config.get("provider")
            model = model or profile_config.get("model")
            temperature = profile_config.get("temperature", temperature)
            max_tokens = profile_config.get("max_tokens", max_tokens)
            timeout = timeout or profile_config.get("timeout")
        
        # Ensure provider and model are specified
        if not provider:
            raise ValueError("Must specify 'provider' or 'profile'")
        if not model:
            raise ValueError("Must specify 'model' or 'profile'")
        
        # Log request
        self._log_request(provider, model, messages, temperature)
        
        # Execute with optional retry
        if retry:
            return self._generate_with_retry(
                messages=messages,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs
            )
        else:
            return self._generate_once(
                messages=messages,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
    
    def generate_simple(
        self,
        prompt: str,
        system: str = "",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        profile: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """
        Simplified interface for basic prompt/response.
        
        Converts simple prompt/system to OpenAI message format internally.
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            provider: Provider name
            model: Model name
            profile: Named profile from config
            temperature: Sampling temperature
            **kwargs: Additional options
        
        Returns:
            LLMResponse
        """
        # Convert to messages format
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Delegate to main generate
        return self.generate(
            messages=messages,
            provider=provider,
            model=model,
            profile=profile,
            temperature=temperature,
            **kwargs
        )
    
    def _generate_once(
        self,
        messages: List[Dict],
        provider: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: Optional[float],
        **kwargs
    ) -> LLMResponse:
        """
        Single generation attempt.
        
        Delegates to llm_layer for provider-specific implementation.
        """
        request_id = self._generate_request_id()
        start_time = time.time()
        
        try:
            # Delegate to provider layer
            result = self.router.call(
                provider=provider,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
            
            latency = time.time() - start_time
            
            # Build response object
            response = LLMResponse(
                content=result["content"],
                provider_used=provider,
                model_name=model,
                request_id=request_id,
                latency=latency,
                tokens=result.get("tokens", {}),
                success=True,
                raw_response=result
            )
            
            # Log success
            logger.info(f"LLM call succeeded: {response}")
            
            return response
        
        except LLMProviderError as e:
            # Provider-specific error from llm_layer
            latency = time.time() - start_time
            
            logger.error(f"LLM call failed: {provider}/{model} - {str(e)}")
            
            return LLMResponse(
                content="",
                provider_used=provider,
                model_name=model,
                request_id=request_id,
                latency=latency,
                success=False,
                error=e
            )
    
    def _generate_with_retry(
        self,
        messages: List[Dict],
        provider: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: Optional[float],
        max_retries: int,
        **kwargs
    ) -> LLMResponse:
        """
        Generate with exponential backoff retry.
        
        Generic retry logic - provider-specific retries handled in llm_layer.
        """
        last_error = None
        
        for attempt in range(max_retries):
            response = self._generate_once(
                messages=messages,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
            
            if response.success:
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                return response
            
            last_error = response.error
            
            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} failed, "
                    f"waiting {wait_time}s: {str(last_error)}"
                )
                time.sleep(wait_time)
        
        # All retries exhausted
        logger.error(f"All {max_retries} retry attempts exhausted")
        raise last_error or LLMProviderError("Unknown error after retries")
    
    def _validate_messages(self, messages: List[Dict]) -> None:
        """Validate OpenAI message format."""
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dict, got {type(msg)}")
            
            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role' field")
            
            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content' field")
            
            if msg["role"] not in ("system", "user", "assistant"):
                raise ValueError(
                    f"Message {i} has invalid role '{msg['role']}', "
                    f"must be system/user/assistant"
                )
    
    def _get_profile(self, profile_name: str) -> Dict[str, Any]:
        """Get profile configuration from APP_CONFIG."""
        if not hasattr(APP_CONFIG, 'llm_profiles'):
            raise ValueError(
                f"No LLM profiles configured. "
                f"Add 'llm_profiles' section to config.py"
            )
        
        profiles = APP_CONFIG.llm_profiles
        
        if profile_name not in profiles:
            available = list(profiles.keys())
            raise ValueError(
                f"Profile '{profile_name}' not found. "
                f"Available profiles: {available}"
            )
        
        return profiles[profile_name]
    
    def _log_request(
        self,
        provider: str,
        model: str,
        messages: List[Dict],
        temperature: float
    ) -> None:
        """Log LLM request details."""
        if not APP_CONFIG.debug_mode:
            return
        
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        logger.debug(
            f"LLM request: {provider}/{model}, "
            f"temp={temperature}, "
            f"messages={len(messages)}, "
            f"chars={total_chars}"
        )
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracing."""
        import uuid
        return f"llm_{int(time.time())}_{uuid.uuid4().hex[:8]}"


# ============================================================================
# Convenience Functions
# ============================================================================

# Global instance for convenience
_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Get global LLM client instance (singleton)."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


__all__ = [
    "LLMClient",
    "LLMResponse",
    "get_client",
]
