from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`."""
    val = value.strip().lower()
    if val in ("openai", "open_ai"):
        return "openai"
    if val in ("custom", "openai_compatible", "local_custom"):
        return "custom"
    if val in ("gemini", "google", "google-genai", "google_genai"):
        return "gemini"
    if val in ("anthropic", "claude", "anthorpic"):
        return "anthropic"
    if val in ("ollama", "local_ollama"):
        return "ollama"
    if val in ("openrouter", "open_router"):
        return "openrouter"
    return val


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider.

    Supported:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenAI` with OpenRouter URL
    """
    prov = normalize_provider(config.provider)
    if prov == "openai":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError:
            raise ImportError("langchain-openai is required for openai provider.")

    elif prov == "custom":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key or "EMPTY",
                base_url=config.base_url,
            )
        except ImportError:
            raise ImportError("langchain-openai is required for custom provider.")

    elif prov == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                google_api_key=config.api_key,
            )
        except ImportError:
            raise ImportError("langchain-google-genai is required for gemini provider.")

    elif prov == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError:
            raise ImportError("langchain-anthropic is required for anthropic provider.")

    elif prov == "ollama":
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url or "http://localhost:11434",
            )
        except ImportError:
            raise ImportError("langchain-ollama is required for ollama provider.")

    elif prov == "openrouter":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1",
            )
        except ImportError:
            raise ImportError("langchain-openai is required for openrouter provider.")

    raise ValueError(f"Unsupported provider: {config.provider} (normalized: {prov})")
