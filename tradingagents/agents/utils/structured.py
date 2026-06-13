"""Shared helper for invoking an agent with structured output and a graceful, *sticky* fallback.

The Portfolio Manager, Trader, Research Manager, and Sentiment Analyst all
follow the same canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output at all (rare; mostly older Ollama models),
   the wrap is skipped and the agent uses free-text generation for every
   call instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails at runtime — a genuine
   provider/model incompatibility such as a 400 for an unsupported
   ``response_format``, or malformed JSON from a weak model — fall back to a
   plain ``llm.invoke`` so the pipeline never blocks.

The subtlety this module handles is step 2's failure mode. A provider/model
incompatibility is not transient: it recurs on *every* structured call for
the lifetime of the agent. The previous implementation re-attempted the
structured path on each invocation, so a single incompatibility produced a
fresh failed request (often a 400), a fresh warning, and fresh statistics
noise on every node execution within an analysis round.

:class:`StructuredBinding` makes the fallback **sticky**: the first runtime
failure disables structured output for the rest of that binding's life and
logs exactly one warning, after which every call goes straight to free text.
Because each agent factory creates its own binding (one per agent, one per
schema), the sticky state is naturally isolated — one agent degrading does
not affect any other. The binding is used sequentially within a graph run,
so no locking is required.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredBinding:
    """An LLM wrapped for structured output, with a sticky free-text fallback.

    Construct one via :func:`bind_structured` and reuse it across every
    invocation of a single agent. ``structured_llm`` is the result of
    ``llm.with_structured_output(schema)``, or ``None`` when the provider does
    not support structured output at bind time (in which case the binding is
    in free-text mode from the start).

    The first time a structured invocation raises at runtime, the binding
    records that fact (:attr:`_runtime_fallback`) and never attempts the
    structured path again — subsequent calls go directly to ``plain_llm`` with
    no retry and no repeated warning.
    """

    def __init__(
        self,
        structured_llm: Optional[Any],
        plain_llm: Any,
        agent_name: str,
    ) -> None:
        self._structured_llm = structured_llm
        self._plain_llm = plain_llm
        self._agent_name = agent_name
        # Sticky flag: flipped on the first runtime structured-output failure,
        # after which every invoke() uses free text. Isolated per binding.
        self._runtime_fallback = False

    @property
    def uses_structured_output(self) -> bool:
        """Whether the next :meth:`invoke` will attempt the structured path.

        ``True`` only when the provider supported structured output at bind
        time *and* no runtime failure has disabled it yet.
        """
        return self._structured_llm is not None and not self._runtime_fallback

    def invoke(self, prompt: Any, render: Callable[[T], str]) -> str:
        """Run the structured call and render to markdown; fall back to free text.

        ``prompt`` is whatever the underlying LLM accepts (a string for chat
        invocations, a list of message dicts/objects for chat models that take
        that shape). The same value is forwarded to the free-text path so the
        fallback sees exactly the input the structured call did.

        On the first runtime failure the structured path is disabled for the
        remainder of this binding's life (sticky), so a recurring
        provider/model incompatibility is not re-triggered on every call.
        """
        if self.uses_structured_output:
            try:
                result = self._structured_llm.invoke(prompt)
                return render(result)
            except Exception as exc:
                # A provider/model incompatibility recurs on every call, so
                # disable structured output now and warn exactly once instead
                # of failing (and logging) on each subsequent invocation.
                self._runtime_fallback = True
                logger.warning(
                    "%s: structured-output invocation failed (%s); disabling "
                    "structured output and using free-text generation for the "
                    "remainder of this run",
                    self._agent_name, exc,
                )

        response = self._plain_llm.invoke(prompt)
        return response.content


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> StructuredBinding:
    """Wrap ``llm`` for structured output, returning a stateful :class:`StructuredBinding`.

    If the provider does not support ``with_structured_output`` at all, a
    warning is logged and the binding is created in free-text mode so every
    call uses plain generation. If the provider claims support but later fails
    at invocation time, the binding switches to free text on the first failure
    and stays there — see :meth:`StructuredBinding.invoke`.
    """
    try:
        structured_llm: Optional[Any] = llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        structured_llm = None
    return StructuredBinding(structured_llm, llm, agent_name)
