"""
Unified LLM Client for OpenAI and Anthropic APIs.

This module provides a consistent interface for calling different LLM providers,
making it easy to compare responses and switch between models.
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from config/.env
_project_root = Path(__file__).parent.parent.parent.parent
_env_path = _project_root / "config" / ".env"
load_dotenv(_env_path)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    raw_response: Any = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a completion for the given prompt."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the provider/model name."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client wrapper."""

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    def __init__(self, model: str = None, api_key: str = None):
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start_time = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate cost
        pricing = self.PRICING.get(self.model, {"input": 0, "output": 0})
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            raw_response=response,
        )

    def get_name(self) -> str:
        return f"OpenAI/{self.model}"


class AnthropicClient(LLMClient):
    """Anthropic API client wrapper."""

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    }

    def __init__(self, model: str = None, api_key: str = None):
        from anthropic import Anthropic

        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        start_time = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Calculate cost
        pricing = self.PRICING.get(self.model, {"input": 0, "output": 0})
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            raw_response=response,
        )

    def get_name(self) -> str:
        return f"Anthropic/{self.model}"


def get_client(provider: str = "openai", model: str = None) -> LLMClient:
    """Factory function to get an LLM client by provider name."""
    if provider.lower() == "openai":
        return OpenAIClient(model=model)
    elif provider.lower() in ("anthropic", "claude"):
        return AnthropicClient(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Quick test
if __name__ == "__main__":
    print("Testing LLM clients...")

    # Test OpenAI
    try:
        openai_client = OpenAIClient()
        response = openai_client.complete("Say 'Hello, World!' and nothing else.")
        print(f"\nOpenAI Response: {response.content}")
        print(f"  Tokens: {response.total_tokens}, Cost: ${response.cost_usd:.6f}, Latency: {response.latency_ms:.0f}ms")
    except Exception as e:
        print(f"OpenAI error: {e}")

    # Test Anthropic
    try:
        anthropic_client = AnthropicClient()
        response = anthropic_client.complete("Say 'Hello, World!' and nothing else.")
        print(f"\nAnthropic Response: {response.content}")
        print(f"  Tokens: {response.total_tokens}, Cost: ${response.cost_usd:.6f}, Latency: {response.latency_ms:.0f}ms")
    except Exception as e:
        print(f"Anthropic error: {e}")
