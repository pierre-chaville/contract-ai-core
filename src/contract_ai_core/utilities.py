from __future__ import annotations

"""Utilities for connecting to LangChain chat models across providers.

Supported providers:
- "openai" via langchain-openai ChatOpenAI
- "azure" via langchain-openai AzureChatOpenAI
- "anthropic" via langchain-anthropic ChatAnthropic

Environment variables (expected in .env):
- OPENAI_API_KEY for OpenAI
- AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION for Azure
- Optionally AZURE_OPENAI_DEPLOYMENT_NAME for Azure deployment name
- ANTHROPIC_API_KEY for Anthropic
"""

import os
from typing import Any


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # Optional dependency; ignore if unavailable
        pass


def get_langchain_chat_model(
    provider: str,
    model_name: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> Any:
    """Return a configured LangChain chat model for the given provider.

    Parameters
    - provider: one of {"openai", "azure", "anthropic"}
    - model_name: provider-specific model or deployment name
    - temperature: sampling temperature
    - max_tokens: optional max completion tokens (if supported)
    """
    _load_dotenv_if_available()

    normalized = (provider or "").strip().lower()

    if normalized == "openai":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "langchain-openai is required for provider 'openai'. Install with: pip install langchain-openai"
            ) from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        return ChatOpenAI(
            model=model_name or "gpt-4.1-mini",
            temperature=float(temperature),
            api_key=api_key,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )

    if normalized == "azure":
        try:
            from langchain_openai import AzureChatOpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "langchain-openai is required for provider 'azure'. Install with: pip install langchain-openai"
            ) from exc
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not api_key or not endpoint or not api_version or not deployment:
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and a deployment name."
            )
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            openai_api_version=api_version,
            azure_deployment=deployment,
            api_key=api_key,  # type: ignore[arg-type]
            temperature=float(temperature),
            max_tokens=max_tokens,
        )

    if normalized == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "langchain-anthropic is required for provider 'anthropic'. Install with: pip install langchain-anthropic"
            ) from exc
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
        return ChatAnthropic(
            model=model_name or "claude-3-5-sonnet-20240620",
            temperature=float(temperature),
            api_key=api_key,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )

    raise NotImplementedError(f"Unsupported provider: {provider}")
