"""LangChain-backed structured LLM calls.

``StructuredLLM.generate(system=..., user=..., schema=...)`` sends one system and
one user message and returns an instance of ``schema`` (a Pydantic model).

The provider is chosen with ``QQQ_LLM_MODEL`` (any ``init_chat_model`` string,
e.g. ``"anthropic:claude-haiku-5-5"``). Other providers need their LangChain
integration package installed (e.g. ``langchain-openai``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic:claude-haiku-5-5"

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The LLM call failed or never produced output matching the schema."""


class StructuredLLM:
    """Thin wrapper around a LangChain chat model that returns validated Pydantic objects."""

    def __init__(self, chat_model: BaseChatModel, *, max_attempts: int = 2):
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.chat_model = chat_model
        self.max_attempts = max_attempts

    def generate(self, *, system: str, user: str, schema: type[T]) -> T:
        # Messages are built directly (not via ChatPromptTemplate) so braces in the
        # user's text are never interpreted as template variables.
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        runnable = self.chat_model.with_structured_output(schema, include_raw=True)

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = runnable.invoke(messages)
            except Exception as exc:  # network/auth/rate limit; the client already retries
                raise LLMError(f"LLM call failed: {exc}") from exc

            try:
                return _to_schema(result, schema)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "LLM output failed %s validation (attempt %d/%d): %s",
                    schema.__name__, attempt, self.max_attempts, exc,
                )

        raise LLMError(
            f"LLM returned invalid {schema.__name__} after {self.max_attempts} attempts"
        ) from last_error


def _to_schema(result: Any, schema: type[T]) -> T:
    """Unpack an ``include_raw=True`` result into a validated ``schema`` instance."""
    if isinstance(result, dict) and "parsed" in result:
        if result.get("parsing_error") is not None:
            raise ValueError(f"parsing error: {result['parsing_error']}")
        result = result["parsed"]
    if result is None:
        raise ValueError("model returned no structured output")
    if isinstance(result, schema):
        return result
    if isinstance(result, BaseModel):
        result = result.model_dump()
    return schema.model_validate(result)


def get_llm(model: str | None = None, **kwargs: Any) -> StructuredLLM:
    """Build the default StructuredLLM (temperature 0) from ``QQQ_LLM_MODEL``."""
    from langchain.chat_models import init_chat_model

    model = model or os.getenv("QQQ_LLM_MODEL", DEFAULT_MODEL)
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("max_tokens", 1024)
    return StructuredLLM(init_chat_model(model, **kwargs))
