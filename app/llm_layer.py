# app/llm_layer.py
# Created: 2025-12-24
# Purpose: Provider-specific LLM implementations and routing

"""
LLM Provider Layer

This is the BOTTOM layer that handles provider-specific details.
Each provider has its own class with specific HTTP calls and error handling.

Responsibilities:
- Provider-specific HTTP/SDK calls
- Format conversion (OpenAI messages ↔ provider format)
- Provider-specific error handling
- Provider-specific configuration

Does NOT:
- Generic retry logic (llm_abstraction does this)
- Input validation (llm_abstraction does this)
- Know about your RAG code (llm_abstraction is the interface)
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from config import APP_CONFIG


logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class LLMProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ProviderConnectionError(LLMProviderError):
    """Provider is unreachable."""
    pass


class ProviderAuthError(LLMProviderError):
    """Authentication failed (API key, etc.)."""
    pass


class ProviderRateLimitError(LLMProviderError):
    """Rate limit exceeded."""
    pass


class ProviderModelError(LLMProviderError):
    """Model not available or invalid."""
    pass


# ============================================================================
# Base Provider Interface
# ============================================================================

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement the call() method.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize provider with configuration.
        
        Args:
            config: Provider-specific configuration dict
        """
        self.config = config
    
    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make provider-specific call.
        
        Args:
            messages: OpenAI-format messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            timeout: Request timeout
            **kwargs: Provider-specific options
        
        Returns:
            Dict with:
                - content: Generated text
                - tokens: Dict with prompt_tokens, completion_tokens, total_tokens
                - raw: Raw provider response (optional)
        
        Raises:
            LLMProviderError: On provider-specific errors
        """
        pass


# ============================================================================
# Ollama Provider
# ============================================================================

class OllamaProvider(LLMProvider):
    """
    Ollama provider implementation.
    
    Uses Ollama's /api/chat endpoint with OpenAI-compatible format.
    """
    
    def __init__(self, config):
        """
        Initialize Ollama provider.
        
        Args:
            config: ProviderConfig dataclass from APP_CONFIG.providers['ollama']
        """
        super().__init__(config)
        
        # Handle ProviderConfig dataclass (frozen, no .get() method)
        if hasattr(config, 'base_url'):
            # It's a ProviderConfig dataclass
            self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")
            self.timeout = config.timeout
            self.options = config.options or {}
        else:
            # Fallback for dict (testing/compatibility)
            self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
            self.timeout = config.get("timeout", 120.0)
            self.options = config.get("options", {})
        
        self.chat_url = f"{self.base_url}/api/chat"
    
    def call(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """Call Ollama API."""
        
        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        # Add optional parameters
        if max_tokens and max_tokens > 0:
            payload["options"]["num_predict"] = max_tokens
        
        # Ollama-specific options from config
        num_ctx = kwargs.get("num_ctx") or self.options.get("num_ctx", 131072)
        repeat_penalty = kwargs.get("repeat_penalty") or self.options.get("repeat_penalty", 1.15)
        
        payload["options"]["num_ctx"] = num_ctx
        payload["options"]["repeat_penalty"] = repeat_penalty
        
        # Make HTTP request
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=timeout or self.timeout
            )
            
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            raise ProviderConnectionError(
                f"Ollama request timed out after {timeout}s"
            )
        
        except requests.exceptions.ConnectionError as e:
            raise ProviderConnectionError(
                f"Cannot connect to Ollama at {self.base_url}: {str(e)}"
            )
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise ProviderModelError(
                    f"Model '{model}' not found. "
                    f"Pull it with: ollama pull {model}"
                )
            else:
                raise LLMProviderError(
                    f"Ollama HTTP error {response.status_code}: {response.text}"
                )
        
        # Parse response
        try:
            data = response.json()
        except ValueError as e:
            raise LLMProviderError(f"Invalid JSON from Ollama: {str(e)}")
        
        # Extract content
        content = ""
        if "message" in data and "content" in data["message"]:
            content = data["message"]["content"]
        elif "response" in data:
            # Fallback for /api/generate format
            content = data["response"]
        else:
            raise LLMProviderError(f"Unexpected Ollama response format: {list(data.keys())}")
        
        # Extract token counts (if available)
        tokens = {}
        if "prompt_eval_count" in data:
            tokens["prompt_tokens"] = data["prompt_eval_count"]
        if "eval_count" in data:
            tokens["completion_tokens"] = data["eval_count"]
        if tokens:
            tokens["total_tokens"] = tokens.get("prompt_tokens", 0) + tokens.get("completion_tokens", 0)
        
        return {
            "content": content,
            "tokens": tokens,
            "raw": data
        }


# ============================================================================
# OpenAI Provider
# ============================================================================

class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation.
    
    Uses OpenAI Python SDK or direct HTTP to /v1/chat/completions.
    """
    
    def __init__(self, config):
        """
        Initialize OpenAI provider.
        
        Args:
            config: ProviderConfig dataclass from APP_CONFIG.providers['openai']
        """
        super().__init__(config)
        
        # Handle ProviderConfig dataclass
        if hasattr(config, 'api_key'):
            self.api_key = config.api_key
            self.timeout = config.timeout
        else:
            self.api_key = config.get("api_key")
            self.timeout = config.get("timeout", 30.0)
        
        if not self.api_key:
            raise ProviderAuthError("OpenAI API key not configured")
        
        # Try to use OpenAI SDK if available
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.use_sdk = True
        except ImportError:
            # Fall back to HTTP
            self.use_sdk = False
            self.base_url = "https://api.openai.com/v1"
    
    def call(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        
        if self.use_sdk:
            return self._call_sdk(messages, model, temperature, max_tokens, timeout, **kwargs)
        else:
            return self._call_http(messages, model, temperature, max_tokens, timeout, **kwargs)
    
    def _call_sdk(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """Call using OpenAI SDK."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout or 30,
                **kwargs
            )
            
            content = response.choices[0].message.content
            
            tokens = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return {
                "content": content,
                "tokens": tokens,
                "raw": response.model_dump()
            }
        
        except Exception as e:
            # OpenAI SDK exceptions
            error_msg = str(e)
            
            if "rate_limit" in error_msg.lower():
                raise ProviderRateLimitError(f"OpenAI rate limit: {error_msg}")
            elif "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                raise ProviderAuthError(f"OpenAI auth error: {error_msg}")
            elif "model" in error_msg.lower():
                raise ProviderModelError(f"OpenAI model error: {error_msg}")
            else:
                raise LLMProviderError(f"OpenAI error: {error_msg}")
    
    def _call_http(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """Call using direct HTTP."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        # Add optional kwargs
        payload.update(kwargs)
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout or 30
            )
            
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            raise ProviderConnectionError(f"OpenAI request timed out")
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise ProviderRateLimitError("OpenAI rate limit exceeded")
            elif response.status_code == 401:
                raise ProviderAuthError("OpenAI API key invalid")
            else:
                raise LLMProviderError(f"OpenAI HTTP {response.status_code}: {response.text}")
        
        data = response.json()
        
        content = data["choices"][0]["message"]["content"]
        tokens = {
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "total_tokens": data["usage"]["total_tokens"]
        }
        
        return {
            "content": content,
            "tokens": tokens,
            "raw": data
        }


# ============================================================================
# LM Studio Provider
# ============================================================================

class LMStudioProvider(LLMProvider):
    """
    LM Studio provider implementation.
    
    Uses OpenAI-compatible API (usually localhost:1234).
    """
    
    def __init__(self, config):
        """
        Initialize LM Studio provider.
        
        Args:
            config: ProviderConfig dataclass from APP_CONFIG.providers['lmstudio']
        """
        super().__init__(config)
        
        # Handle ProviderConfig dataclass
        if hasattr(config, 'base_url'):
            self.base_url = (config.base_url or "http://localhost:1234/v1").rstrip("/")
            self.timeout = config.timeout
        else:
            self.base_url = config.get("base_url", "http://localhost:1234/v1").rstrip("/")
            self.timeout = config.get("timeout", 120.0)
    
    def call(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """Call LM Studio API (OpenAI-compatible)."""
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout or 120
            )
            
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            raise ProviderConnectionError(f"LM Studio request timed out")
        
        except requests.exceptions.ConnectionError as e:
            raise ProviderConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}: {str(e)}"
            )
        
        except requests.exceptions.HTTPError as e:
            raise LLMProviderError(f"LM Studio HTTP {response.status_code}: {response.text}")
        
        data = response.json()
        
        # LM Studio response format (OpenAI-compatible)
        content = data["choices"][0]["message"]["content"]
        
        # Token counts may not always be available
        tokens = {}
        if "usage" in data:
            tokens = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0)
            }
        
        return {
            "content": content,
            "tokens": tokens,
            "raw": data
        }


# ============================================================================
# Provider Router
# ============================================================================

class ProviderRouter:
    """
    Routes calls to appropriate provider based on provider name.
    
    Manages provider instances and configuration.
    """
    
    def __init__(self):
        """Initialize router with provider configs from APP_CONFIG."""
        self.providers: Dict[str, LLMProvider] = {}
        self._init_providers()
    
    def _init_providers(self) -> None:
        """Initialize providers from configuration."""
        
        # Check if providers config exists
        if not hasattr(APP_CONFIG, 'providers'):
            logger.warning("No 'providers' section in config, using defaults")
            return
        
        provider_configs = APP_CONFIG.providers
        
        # Initialize each configured provider
        for name, config in provider_configs.items():
            # Skip OpenAI initialization unless an API key is configured.
            if name == "openai" and not getattr(config, "api_key", None):
                logger.info("Skipping provider openai: API key not configured")
                continue
            try:
                provider_class = self._get_provider_class(name)
                self.providers[name] = provider_class(config)
                logger.info(f"Initialized provider: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {name}: {e}")
    
    def _get_provider_class(self, name: str) -> type:
        """Get provider class by name."""
        providers = {
            "ollama": OllamaProvider,
            "openai": OpenAIProvider,
            "lmstudio": LMStudioProvider,
        }
        
        if name not in providers:
            raise ValueError(
                f"Unknown provider '{name}'. "
                f"Available: {list(providers.keys())}"
            )
        
        return providers[name]
    
    def call(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: Optional[float],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route call to appropriate provider.
        
        Args:
            provider: Provider name
            messages: OpenAI-format messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens
            timeout: Request timeout
            **kwargs: Provider-specific options
        
        Returns:
            Dict with content, tokens, raw
        
        Raises:
            LLMProviderError: Provider-specific errors
        """
        # Get or create provider instance
        if provider not in self.providers:
            # Lazy initialization if not pre-configured
            logger.warning(f"Provider '{provider}' not pre-configured, initializing with defaults")
            try:
                provider_class = self._get_provider_class(provider)
                self.providers[provider] = provider_class({})
            except Exception as e:
                raise LLMProviderError(f"Cannot initialize provider '{provider}': {e}")
        
        provider_instance = self.providers[provider]
        
        # Delegate to provider
        return provider_instance.call(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )


__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "ProviderConnectionError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderModelError",
    "ProviderRouter",
    "OllamaProvider",
    "OpenAIProvider",
    "LMStudioProvider",
]
