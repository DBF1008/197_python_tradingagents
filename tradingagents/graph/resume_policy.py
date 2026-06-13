"""Resume policy: decide whether a checkpoint is safe to resume.

Pure logic — no I/O, no side effects.  Callers (``propagate()``, the CLI)
load the manifest and pass it here alongside the *current* runtime config.

Compatibility rules
-------------------

* **Hard** (topology-breaking): ``selected_analysts``, ``asset_type``.
  A different set of analysts changes the graph structure; a different
  asset type switches the entire pipeline.  Default: **REJECT**.

* **Soft** (behavioural drift): ``llm_provider``, ``deep_think_llm``,
  ``quick_think_llm``, ``temperature``, ``backend_url``.  The graph
  topology is unchanged but results will differ.  Default: **WARN_RESUME**.

* **Legacy** (no manifest): the checkpoint predates the manifest system.
  Default: **LEGACY_RESUME** (allow with a warning).

Passing ``force=True`` downgrades **REJECT** to **WARN_RESUME** so users
can override at their own risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tradingagents.graph.checkpoint_manifest import RunManifest


class ResumeDecision(Enum):
    """Outcome of a resume compatibility check."""

    RESUME = "resume"  # all fields match — safe to continue
    WARN_RESUME = "warn_resume"  # soft drift only — proceed with warnings
    REJECT = "reject"  # hard incompatibility — refuse to resume
    LEGACY_RESUME = "legacy_resume"  # no manifest — allow with warning
    FRESH_START = "fresh_start"  # no checkpoint exists at all


# Fields whose mismatch is a hard rejection.
_HARD_FIELDS: Tuple[str, ...] = ("selected_analysts", "asset_type")

# Fields whose mismatch is a soft warning.
_SOFT_FIELDS: Tuple[str, ...] = (
    "llm_provider",
    "deep_think_llm",
    "quick_think_llm",
    "temperature",
    "backend_url",
)


class ResumeRejectedError(Exception):
    """Raised when a resume attempt is rejected due to configuration drift."""

    def __init__(self, verdict: "ResumeVerdict"):
        self.verdict = verdict
        super().__init__(verdict.reason)


@dataclass(frozen=True)
class ResumeVerdict:
    """The result of an :func:`evaluate_resume` call."""

    decision: ResumeDecision
    reason: str
    manifest: Optional[RunManifest] = None
    incompatible_fields: Tuple[str, ...] = ()
    drifted_fields: Tuple[str, ...] = ()


def _extract_current(manifest: RunManifest, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a comparison dict from current config, matching manifest field names."""
    return {
        "selected_analysts": tuple(manifest.selected_analysts),  # placeholder
        "asset_type": manifest.asset_type,  # placeholder
        "llm_provider": config.get("llm_provider", ""),
        "deep_think_llm": config.get("deep_think_llm", ""),
        "quick_think_llm": config.get("quick_think_llm", ""),
        "temperature": config.get("temperature"),
        "backend_url": config.get("backend_url"),
    }


def evaluate_resume(
    existing: Optional[RunManifest],
    current_analysts: List[str],
    current_asset_type: str,
    current_config: Dict[str, Any],
    force: bool = False,
) -> ResumeVerdict:
    """Evaluate whether it is safe to resume from an existing checkpoint.

    Parameters
    ----------
    existing:
        The manifest loaded from disk for the matching thread, or ``None``
        when no manifest exists (either no checkpoint or a legacy one).
    current_analysts:
        The analyst keys selected for the current run, in order.
    current_asset_type:
        ``"stock"`` or ``"crypto"``.
    current_config:
        The runtime config dict (same shape as ``DEFAULT_CONFIG``).
    force:
        When ``True``, downgrade hard rejections to warnings so the user
        can resume anyway (at their own risk).

    Returns
    -------
    ResumeVerdict with the decision, reason, and lists of incompatible /
    drifted fields.
    """
    # No checkpoint at all → fresh start.
    if existing is None:
        return ResumeVerdict(
            decision=ResumeDecision.FRESH_START,
            reason="No checkpoint found. Starting fresh.",
        )

    incompatible: List[str] = []
    drifted: List[str] = []

    # --- hard checks ---
    if tuple(current_analysts) != existing.selected_analysts:
        incompatible.append("selected_analysts")
    if current_asset_type != existing.asset_type:
        incompatible.append("asset_type")

    # --- soft checks ---
    for field in _SOFT_FIELDS:
        current_val = current_config.get(field)
        manifest_val = getattr(existing, field, None)
        if current_val != manifest_val:
            drifted.append(field)

    # --- decision ---
    if incompatible:
        fields_desc = ", ".join(incompatible)
        if force:
            return ResumeVerdict(
                decision=ResumeDecision.WARN_RESUME,
                reason=(
                    f"Forced resume despite incompatible fields: {fields_desc}. "
                    f"Graph topology may differ from the original run."
                ),
                manifest=existing,
                incompatible_fields=tuple(incompatible),
                drifted_fields=tuple(drifted),
            )
        return ResumeVerdict(
            decision=ResumeDecision.REJECT,
            reason=(
                f"Checkpoint incompatible with current config: {fields_desc} changed. "
                f"Use --force-resume to override or --clear-ticker-checkpoint to discard."
            ),
            manifest=existing,
            incompatible_fields=tuple(incompatible),
            drifted_fields=tuple(drifted),
        )

    if drifted:
        fields_desc = ", ".join(drifted)
        return ResumeVerdict(
            decision=ResumeDecision.WARN_RESUME,
            reason=f"Resuming with configuration drift: {fields_desc} changed.",
            manifest=existing,
            drifted_fields=tuple(drifted),
        )

    return ResumeVerdict(
        decision=ResumeDecision.RESUME,
        reason="Configuration matches. Safe to resume.",
        manifest=existing,
    )


def format_resume_summary(verdict: ResumeVerdict) -> str:
    """Render a human-readable multi-line resume summary.

    Intended for display in the CLI via a ``rich`` Panel.
    """
    lines: List[str] = []

    # Status line with emoji indicator.
    status_map = {
        ResumeDecision.RESUME: "✅ RESUME",
        ResumeDecision.WARN_RESUME: "⚠️  WARN_RESUME",
        ResumeDecision.REJECT: "🚫 REJECT",
        ResumeDecision.LEGACY_RESUME: "⚠️  LEGACY_RESUME (no manifest)",
        ResumeDecision.FRESH_START: "🆕 FRESH_START",
    }
    lines.append(f"Decision: {status_map.get(verdict.decision, str(verdict.decision))}")
    lines.append(f"Reason:   {verdict.reason}")

    m = verdict.manifest
    if m is not None:
        lines.append("")
        lines.append(f"Ticker:     {m.ticker}")
        lines.append(f"Date:       {m.trade_date}")
        lines.append(f"Thread ID:  {m.thread_id}")
        lines.append(f"Asset type: {m.asset_type}")
        lines.append(
            f"Analysts:   {', '.join(m.selected_analysts) if m.selected_analysts else '(none)'}"
        )
        lines.append(f"Last step:  {m.last_completed_step}")
        lines.append("")
        lines.append(f"LLM provider:     {m.llm_provider}")
        lines.append(f"Deep think model:  {m.deep_think_llm}")
        lines.append(f"Quick think model: {m.quick_think_llm}")
        temp = m.temperature if m.temperature is not None else "(default)"
        lines.append(f"Temperature:      {temp}")

    if verdict.incompatible_fields:
        lines.append("")
        lines.append(f"⛔ Incompatible: {', '.join(verdict.incompatible_fields)}")

    if verdict.drifted_fields:
        lines.append("")
        lines.append(f"⚠️  Drifted:     {', '.join(verdict.drifted_fields)}")

    return "\n".join(lines)
