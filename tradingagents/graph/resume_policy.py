"""Resume policy for checkpoint/resume runs.

This module is the *auditable* layer on top of the raw LangGraph checkpoint.
A :class:`RunManifest` records the identity of the run that produced a
checkpoint — the selected analysts, the asset type, and a fingerprint of the
key LLM configuration — so that before resuming we can decide whether the
saved partial state is actually compatible with the run we are about to start.

Design notes:

* This module is **pure** — it has no ``sqlite3``/``langgraph`` imports and does
  no I/O. Persistence and the live "last completed step" lookup live in
  :mod:`tradingagents.graph.checkpointer`, which bridges this policy to disk.
  Keeping it pure makes the compatibility logic trivially unit-testable.
* The ``last_step`` stored on a manifest is a best-effort breadcrumb only.
  LangGraph maintains its own per-node ``step`` in the checkpoint metadata, so
  the authoritative step is always read live from the checkpoint at resume time
  (see ``checkpointer.plan_resume``). ``evaluate_resume`` therefore never
  compares ``last_step``.
* Analyst-set isolation is handled upstream by folding the analyst set into the
  ``thread_id`` (see ``checkpointer.thread_id``); a manifest read back for a
  given thread will normally already agree on analysts. The analyst comparison
  here is a defensive cross-check, not the primary isolation mechanism.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Manifest schema version. Bump when the persisted payload shape changes so
# ``RunManifest.from_payload`` can migrate / tolerate older rows.
MANIFEST_VERSION = 1

# The config keys that define "which brain produced this checkpoint". A change
# in any of these means a resumed run would mix outputs from two different LLM
# configurations into a single report, so they gate compatibility. The two
# ``*_rounds`` keys are included despite the name because they change the
# debate/risk loop counts (conditional logic), which a half-finished checkpoint
# encodes implicitly.
FINGERPRINT_KEYS: Tuple[str, ...] = (
    "llm_provider",
    "deep_think_llm",
    "quick_think_llm",
    "backend_url",
    "temperature",
    "google_thinking_level",
    "openai_reasoning_effort",
    "anthropic_effort",
    "max_debate_rounds",
    "max_risk_discuss_rounds",
)


def _none_if_blank(value: Any) -> Optional[Any]:
    """Collapse ``None`` and empty string to ``None`` (leave other values)."""
    if value is None or value == "":
        return None
    return value


def normalize_analysts(analysts) -> Tuple[str, ...]:
    """Return analyst keys lower-cased, de-duped and sorted (order-independent)."""
    if not analysts:
        return tuple()
    return tuple(sorted({str(a).strip().lower() for a in analysts}))


def llm_fingerprint(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a normalized fingerprint of the run-defining config keys.

    Normalization mirrors how the values are actually consumed so that
    equivalent-but-differently-typed inputs do not register as drift:

    * ``llm_provider`` is lower-cased (the graph lower-cases it too).
    * ``temperature`` is ``None`` for unset/blank, else ``float`` — matching
      ``TradingAgentsGraph._get_provider_kwargs`` which does ``float(value)``
      so an env string ``"0.2"`` and a programmatic ``0.2`` compare equal.
    * ``max_debate_rounds`` / ``max_risk_discuss_rounds`` coerce to ``int``.
    * Empty strings collapse to ``None`` everywhere; missing keys are ``None``.

    The result always has exactly :data:`FINGERPRINT_KEYS`.
    """
    fp: Dict[str, Any] = {}
    for key in FINGERPRINT_KEYS:
        raw = _none_if_blank(config.get(key))
        if raw is None:
            fp[key] = None
        elif key == "llm_provider":
            fp[key] = str(raw).lower()
        elif key == "temperature":
            fp[key] = float(raw)
        elif key in ("max_debate_rounds", "max_risk_discuss_rounds"):
            fp[key] = int(raw)
        else:
            fp[key] = str(raw)
    return fp


@dataclass(frozen=True)
class RunManifest:
    """Auditable identity of the run that produced (or will produce) a checkpoint."""

    thread_id: str
    ticker: str
    trade_date: str
    asset_type: str
    analysts: Tuple[str, ...]
    llm_config: Dict[str, Any]
    last_step: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    manifest_version: int = MANIFEST_VERSION

    @classmethod
    def build(
        cls,
        *,
        thread_id: str,
        ticker: str,
        trade_date: str,
        asset_type: str,
        analysts,
        config: Dict[str, Any],
        last_step: Optional[int] = None,
        now: Optional[str] = None,
    ) -> "RunManifest":
        """Construct a manifest from raw run parameters + config.

        ``now`` (ISO-8601) is injectable for deterministic tests; when omitted
        the current UTC time is used for both ``created_at`` and ``updated_at``.
        """
        ts = now or datetime.now(timezone.utc).isoformat()
        return cls(
            thread_id=thread_id,
            ticker=ticker,
            trade_date=str(trade_date),
            asset_type=asset_type,
            analysts=normalize_analysts(analysts),
            llm_config=llm_fingerprint(config),
            last_step=last_step,
            created_at=ts,
            updated_at=ts,
            manifest_version=MANIFEST_VERSION,
        )

    def to_payload(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (analysts as a list)."""
        return {
            "thread_id": self.thread_id,
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "asset_type": self.asset_type,
            "analysts": list(self.analysts),
            "llm_config": dict(self.llm_config),
            "last_step": self.last_step,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "manifest_version": self.manifest_version,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RunManifest":
        """Reconstruct from a stored payload, tolerating older manifest versions."""
        return cls(
            thread_id=payload["thread_id"],
            ticker=payload.get("ticker", ""),
            trade_date=str(payload.get("trade_date", "")),
            asset_type=payload.get("asset_type", "stock"),
            analysts=normalize_analysts(payload.get("analysts") or []),
            llm_config=dict(payload.get("llm_config") or {}),
            last_step=payload.get("last_step"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            manifest_version=int(payload.get("manifest_version", 1)),
        )


class ResumeAction(enum.Enum):
    """What the resume policy decided to do."""

    FRESH = "fresh"            # no usable checkpoint — start from scratch
    RESUME = "resume"          # compatible checkpoint — resume from last step
    INCOMPATIBLE = "incompatible"  # checkpoint exists but config drifted


@dataclass
class ResumeDecision:
    """Outcome of comparing a stored manifest against the intended run."""

    action: ResumeAction
    thread_id: str
    intended: RunManifest
    stored: Optional[RunManifest] = None
    # Live step from the checkpoint (LangGraph metadata), filled by plan_resume.
    last_step: Optional[int] = None
    # (field, stored_value, intended_value) for each drifted field.
    mismatches: List[Tuple[str, Any, Any]] = field(default_factory=list)
    # Free-form notes (e.g. legacy checkpoint without a manifest).
    notes: List[str] = field(default_factory=list)

    @property
    def should_resume(self) -> bool:
        return self.action is ResumeAction.RESUME

    @property
    def requires_clear(self) -> bool:
        """True when there is incompatible state that must be discarded."""
        return self.action is ResumeAction.INCOMPATIBLE

    def summary(self) -> str:
        """Human-readable resume summary for CLI output and logs."""
        analysts = ",".join(self.intended.analysts) or "(none)"
        head = (
            f"Checkpoint for {self.intended.ticker} {self.intended.trade_date} "
            f"[analysts: {analysts}; asset: {self.intended.asset_type}] "
            f"(thread {self.thread_id})"
        )
        if self.action is ResumeAction.FRESH:
            body = "  -> starting fresh (no compatible checkpoint)."
        elif self.action is ResumeAction.RESUME:
            step = "unknown" if self.last_step is None else self.last_step
            body = f"  -> resuming from step {step} (config matches)."
        else:  # INCOMPATIBLE
            lines = [
                "  -> INCOMPATIBLE: saved run was produced with a different "
                "configuration; discarding its state and starting fresh."
            ]
            for fld, old, new in self.mismatches:
                lines.append(f"     - {fld}: {old!r} -> {new!r}")
            body = "\n".join(lines)
        if self.notes:
            body += "\n" + "\n".join(f"  note: {n}" for n in self.notes)
        return head + "\n" + body


def _diff_manifests(
    stored: RunManifest, intended: RunManifest
) -> List[Tuple[str, Any, Any]]:
    """Return the list of (field, stored, intended) that differ."""
    mismatches: List[Tuple[str, Any, Any]] = []
    if stored.asset_type != intended.asset_type:
        mismatches.append(("asset_type", stored.asset_type, intended.asset_type))
    if stored.analysts != intended.analysts:
        mismatches.append(
            ("analysts", list(stored.analysts), list(intended.analysts))
        )
    for key in FINGERPRINT_KEYS:
        # .get() so an older manifest missing a newer key is treated as drift
        # (safe: it forces a fresh start rather than a silent mismatch).
        sv = stored.llm_config.get(key)
        iv = intended.llm_config.get(key)
        if sv != iv:
            mismatches.append((key, sv, iv))
    return mismatches


def evaluate_resume(
    stored: Optional[RunManifest],
    intended: RunManifest,
    *,
    has_live_checkpoint: bool,
    strict_unverified: bool = False,
) -> ResumeDecision:
    """Decide whether to resume, start fresh, or treat as incompatible.

    Args:
        stored: the manifest previously persisted for this thread, or ``None``.
        intended: the manifest describing the run we are about to start.
        has_live_checkpoint: whether LangGraph actually has saved state for this
            thread (i.e. ``checkpoint_step`` returned a value).
        strict_unverified: when True, a live checkpoint with no manifest (legacy
            or interrupted-before-write) is treated as INCOMPATIBLE rather than
            resumed; also surfaced for config ``checkpoint_strict``.

    Branches:
        1. no manifest, no live state -> FRESH.
        2. live state but no manifest  -> RESUME (with a note) by default, or
           INCOMPATIBLE under ``strict_unverified``.
        3. manifest present -> compare fields; any drift -> INCOMPATIBLE,
           else RESUME if live state exists, else FRESH.
    """
    if stored is None:
        if not has_live_checkpoint:
            return ResumeDecision(
                action=ResumeAction.FRESH,
                thread_id=intended.thread_id,
                intended=intended,
            )
        # Live state with no manifest: legacy checkpoint or a run killed between
        # graph compile and manifest write.
        if strict_unverified:
            return ResumeDecision(
                action=ResumeAction.INCOMPATIBLE,
                thread_id=intended.thread_id,
                intended=intended,
                notes=["checkpoint has no manifest; strict mode rejects it"],
            )
        return ResumeDecision(
            action=ResumeAction.RESUME,
            thread_id=intended.thread_id,
            intended=intended,
            notes=["checkpoint has no manifest; resuming unverified"],
        )

    mismatches = _diff_manifests(stored, intended)
    if mismatches:
        return ResumeDecision(
            action=ResumeAction.INCOMPATIBLE,
            thread_id=intended.thread_id,
            intended=intended,
            stored=stored,
            mismatches=mismatches,
        )

    if has_live_checkpoint:
        return ResumeDecision(
            action=ResumeAction.RESUME,
            thread_id=intended.thread_id,
            intended=intended,
            stored=stored,
        )
    # Manifest matches but state was already cleared (e.g. previous run finished).
    return ResumeDecision(
        action=ResumeAction.FRESH,
        thread_id=intended.thread_id,
        intended=intended,
        stored=stored,
    )
