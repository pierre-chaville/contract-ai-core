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
import re
from typing import Any

from .schema import Paragraph


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


def split_text_into_paragraphs(text: str) -> list[Paragraph]:
    """Split text into paragraphs with markdown cleanup and heuristic merging.

    Cleanup rules:
    - Collapse multiple consecutive empty lines into a single empty line
    - If a line contains '|' but neither the previous nor next line contains '|',
      replace '|' with a space (break stray table artifacts)
    - If a line contains only '|' and '-' characters and neither neighbor line
      contains '|', drop the line (remove markdown table separator rows)

    Paragraph merge rules (merge current line into the previous paragraph only if):
    - The previous line does not end with one of . : ! ?
    - The current line does not start with a list/enumeration marker like
      '-', '1)', '1.', 'a)', '(a)', '(1)', etc.
    - Neither the previous line nor the current line contains a '|'
    """

    if not text:
        return []

    # Normalize newlines and split
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    cleaned_lines: list[str] = []
    n = len(raw_lines)
    only_pipes_dashes_re = re.compile(r"^[\s\|\-]+$")
    only_stars_re = re.compile(r"^[\s\*]+$")
    star_rule_re = re.compile(r"^\*+(?:\s*\*+){2,}$")

    def remove_md_emphasis(s: str) -> str:
        # Remove strong/italic/strike emphasis markers: **text**, *text*, __text__, _text_, ~~text~~
        s = re.sub(r"(\*\*|__)(.*?)\1", r"\2", s)
        s = re.sub(r"(\*|_)(.*?)\1", r"\2", s)
        s = re.sub(r"~~(.*?)~~", r"\1", s)
        return s

    i = 0
    while i < n:
        line = raw_lines[i]
        stripped = line.strip()

        # Remove markdown emphasis markers inline
        line = remove_md_emphasis(line)

        # Remove lines that are only stars (e.g., markdown separators) or star rules like * * *
        if stripped and (only_stars_re.match(stripped) or star_rule_re.match(stripped)):
            i += 1
            continue

        # Remove any line consisting solely of '|' and '-' (table separators), unconditionally
        if stripped and only_pipes_dashes_re.match(stripped):
            i += 1
            continue

        # Convert all pipes to tabs to normalize table-like content into columns
        if "|" in line:
            line = line.replace("|", "\t")

        cleaned_lines.append(line)
        i += 1

    # Collapse multiple consecutive empty lines into a single empty line
    collapsed_lines: list[str] = []
    empty_streak = 0
    for line in cleaned_lines:
        if line.strip() == "":
            empty_streak += 1
            if empty_streak > 1:
                continue
        else:
            empty_streak = 0
        collapsed_lines.append(line)

    # Merge lines into paragraphs using heuristics
    paragraphs: list[str] = []
    buffer = ""
    prev_line_text: str | None = None

    list_marker_re = re.compile(r"^\s*(?:-+|\d+\)|\d+\.|[A-Za-z]\)|\([A-Za-z]\)|\(\d+\))\s+")

    for line in collapsed_lines:
        stripped = line.strip()
        if stripped == "":
            if buffer:
                paragraphs.append(buffer.strip())
                buffer = ""
                prev_line_text = None
            continue

        if not buffer:
            buffer = stripped
            prev_line_text = line
            continue

        # Decide merge vs start new paragraph
        assert prev_line_text is not None
        prev_last_char = buffer.rstrip()[-1] if buffer.rstrip() else ""
        prev_ends_sentence = prev_last_char in ".:!?"
        curr_starts_list = bool(list_marker_re.match(stripped))
        prev_has_pipe_now = "|" in prev_line_text
        curr_has_pipe_now = "|" in line

        can_merge = (
            (not prev_ends_sentence)
            and (not curr_starts_list)
            and (not prev_has_pipe_now)
            and (not curr_has_pipe_now)
        )

        if can_merge:
            buffer = f"{buffer.rstrip()} {stripped}"
        else:
            paragraphs.append(buffer.strip())
            buffer = stripped

        prev_line_text = line

    if buffer:
        paragraphs.append(buffer.strip())

    return [Paragraph(index=i, text=block) for i, block in enumerate(paragraphs) if block]
