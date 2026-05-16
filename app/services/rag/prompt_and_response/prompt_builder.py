"""Prompt builder utilities for assembling the LLM prompt."""

from __future__ import annotations

from typing import Dict, List

from ..schemas_and_settings.schemas import RetrievedChunk
from ..schemas_and_settings.settings import PROMPT_MAX_CONTEXT_TOKENS, PROMPT_TEMPLATE # type: ignore


def build_prompt(query: str, chunks: List[RetrievedChunk], max_tokens: int = PROMPT_MAX_CONTEXT_TOKENS) -> str:
    pieces: List[str] = []
    token_budget = max_tokens

    for chunk in chunks:
        # `RetrievedChunk` stores extra fields in `metadata`; prefer metadata title to avoid AttributeError
        title = None
        try:
            title = chunk.metadata.get("title") if isinstance(chunk.metadata, dict) else None
        except Exception:
            title = None
        piece = f"Source: {chunk.source_key}\nTitle: {title or ''}\nText: {chunk.text}\n---\n"
        pieces.append(piece)
        token_budget -= len(piece.split())
        if token_budget <= 0:
            break

    context = "\n".join(pieces)
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    return prompt


def build_chat_prompt(query: str, items: List[RetrievedChunk], history: List[Dict[str, str]] | None = None, extra_context: str | None = None, max_tokens: int = PROMPT_MAX_CONTEXT_TOKENS) -> str:
    # simple compatibility wrapper used by pipeline; includes recent history and extra context
    history = history or []
    header_parts: List[str] = []
    if extra_context:
        header_parts.append(str(extra_context).strip())
    if history:
        last_msgs = history[-6:]
        header_parts.append("\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in last_msgs))
    header = "\n\n".join(header_parts)
    core = build_prompt(query, items, max_tokens=max_tokens)
    return (header + "\n\n" + core).strip() if header else core
