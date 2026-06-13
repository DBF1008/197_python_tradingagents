"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, Research Manager, and Sentiment Analyst all
follow the same canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.
3. **Sticky fallback**: once a structured invocation fails at runtime,
   the binding is marked as *poisoned* and all subsequent calls for the
   same agent skip the structured path entirely. This avoids repeatedly
   sending requests that are guaranteed to fail (and generating 400s,
   duplicate warnings, and stats noise) when the root cause is a
   provider/model incompatibility rather than a one-off glitch.

Centralising the pattern here keeps the agent factories small and ensures
all agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class _StructuredBinding:
    """Bundles a structured-LLM binding with its fallback state.

    Returned by :func:`bind_structured` and consumed by
    :func:`invoke_structured_or_freetext`. The ``_poisoned`` flag flips to
    ``True`` after the first structured-output invocation failure so later
    calls for the same agent go straight to the free-text path without
    retrying the broken structured round-trip.

    Each :func:`bind_structured` call produces an independent binding, so
    a runtime failure in one agent (e.g. Trader / ``TraderProposal``) does
    not affect another agent (e.g. Research Manager / ``ResearchPlan``),
    even when both share the same underlying LLM.
    """

    structured_llm: Optional[Any]
    plain_llm: Any
    agent_name: str
    _poisoned: bool = field(default=False, repr=False)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> _StructuredBinding:
    """Return a :class:`_StructuredBinding` for *llm* + *schema*.

    If the provider raises ``NotImplementedError`` or ``AttributeError``
    from ``with_structured_output``, the binding's ``structured_llm`` is
    ``None`` and every invocation uses free-text generation directly —
    no structured round-trip is ever attempted.

    The binding is intentionally *local* to this agent: a runtime failure
    here will not poison other agents' bindings, even if they share the
    same underlying LLM.
    """
    try:
        structured_llm = llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        structured_llm = None

    return _StructuredBinding(
        structured_llm=structured_llm,
        plain_llm=llm,
        agent_name=agent_name,
    )


def invoke_structured_or_freetext(
    structured_llm: Any,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``structured_llm`` is normally a :class:`_StructuredBinding` returned
    by :func:`bind_structured`. When the structured invocation raises, the
    binding is *poisoned* so every later call for the same agent skips
    straight to the free-text path — this is the sticky fallback that
    prevents the same provider/model incompatibility from generating
    repeated 400 errors, duplicate warnings, and wasted tokens across the
    rest of the run.

    For backward compatibility, a raw structured-LLM object (anything that
    is not a :class:`_StructuredBinding`) is also accepted: in that case
    ``plain_llm`` is used as the free-text fallback and no sticky state is
    tracked, which preserves the behavior of the original single-shot
    fallback for any caller that bypasses :func:`bind_structured`.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    binding = structured_llm if isinstance(structured_llm, _StructuredBinding) else None

    # --- fast path: binding exists, not poisoned, structured LLM available ---
    if binding is not None and binding.structured_llm is not None and not binding._poisoned:
        try:
            result = binding.structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            binding._poisoned = True
            logger.warning(
                "%s: structured-output invocation failed (%s); "
                "poisoning binding — subsequent calls will use free-text directly",
                binding.agent_name, exc,
            )
            response = binding.plain_llm.invoke(prompt)
            return response.content

    # --- binding exists but is either None or poisoned: skip structured ---
    if binding is not None:
        response = binding.plain_llm.invoke(prompt)
        return response.content

    # --- legacy path: caller passed a raw structured LLM, no sticky tracking ---
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content
