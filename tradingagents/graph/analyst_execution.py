from dataclasses import dataclass
from time import monotonic
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class AnalystNodeSpec:
    key: str
    agent_node: str
    clear_node: str
    tool_node: str
    report_key: str


@dataclass(frozen=True)
class AnalystExecutionPlan:
    specs: List[AnalystNodeSpec]
    concurrency_limit: int


ANALYST_NODE_SPECS: Dict[str, AnalystNodeSpec] = {
    "market": AnalystNodeSpec(
        key="market",
        agent_node="Market Analyst",
        clear_node="Msg Clear Market",
        tool_node="tools_market",
        report_key="market_report",
    ),
    "social": AnalystNodeSpec(
        # Wire key stays "social" for saved-config back-compat; the
        # user-facing label is "Sentiment Analyst" to match the rename
        # that landed in v0.2.5 (sentiment_analyst now ingests news +
        # StockTwits + Reddit, not just social media).
        key="social",
        agent_node="Sentiment Analyst",
        clear_node="Msg Clear Sentiment",
        tool_node="tools_social",
        report_key="sentiment_report",
    ),
    "news": AnalystNodeSpec(
        key="news",
        agent_node="News Analyst",
        clear_node="Msg Clear News",
        tool_node="tools_news",
        report_key="news_report",
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        agent_node="Fundamentals Analyst",
        clear_node="Msg Clear Fundamentals",
        tool_node="tools_fundamentals",
        report_key="fundamentals_report",
    ),
}


def build_analyst_execution_plan(
    selected_analysts: Iterable[str],
    concurrency_limit: int = 1,
) -> AnalystExecutionPlan:
    if concurrency_limit < 1:
        raise ValueError("analyst concurrency limit must be >= 1")

    specs: List[AnalystNodeSpec] = []
    for analyst_key in selected_analysts:
        spec = ANALYST_NODE_SPECS.get(analyst_key)
        if spec is None:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        specs.append(spec)

    if not specs:
        raise ValueError("at least one analyst must be selected")

    return AnalystExecutionPlan(specs=specs, concurrency_limit=concurrency_limit)


def get_initial_analyst_node(plan: AnalystExecutionPlan) -> str:
    return plan.specs[0].agent_node


class AnalystWallTimeTracker:
    def __init__(self, plan: AnalystExecutionPlan):
        self.plan = plan
        self._started_at: Dict[str, float] = {}
        self._wall_times: Dict[str, float] = {}

    def mark_started(self, analyst_key: str, started_at: Optional[float] = None) -> None:
        if analyst_key not in ANALYST_NODE_SPECS:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        self._started_at.setdefault(analyst_key, monotonic() if started_at is None else started_at)

    def mark_completed(
        self,
        analyst_key: str,
        completed_at: Optional[float] = None,
    ) -> None:
        if analyst_key not in ANALYST_NODE_SPECS:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        if analyst_key in self._wall_times:
            return
        started_at = self._started_at.get(analyst_key)
        if started_at is None:
            return
        finished_at = monotonic() if completed_at is None else completed_at
        self._wall_times[analyst_key] = max(0.0, finished_at - started_at)

    def mark_initial_active(self, started_at: Optional[float] = None) -> None:
        """Start the initial active window: the first ``concurrency_limit``
        analysts that run before the graph emits its first chunk.

        Anchoring these starts here (just before streaming begins) is more
        accurate than waiting for the first chunk, which only arrives after
        the first node has run.
        """
        start = monotonic() if started_at is None else started_at
        for spec in self.plan.specs[: self.plan.concurrency_limit]:
            self.mark_started(spec.key, started_at=start)

    def active_keys(self) -> List[str]:
        """Analysts that have started but not yet completed (currently running)."""
        return [
            key
            for key in self._started_at
            if key not in self._wall_times
        ]

    def get_wall_times(self) -> Dict[str, float]:
        return dict(self._wall_times)

    def format_summary(self) -> str:
        parts = []
        for spec in self.plan.specs:
            duration = self._wall_times.get(spec.key)
            if duration is not None:
                label = spec.agent_node.removesuffix(" Analyst")
                parts.append(f"{label} {duration:.2f}s")
        if not parts:
            return "Analyst wall time: pending"
        return "Analyst wall time: " + " | ".join(parts)


def sync_analyst_tracker_from_chunk(
    tracker: AnalystWallTimeTracker,
    chunk: Dict[str, str],
    now: Optional[float] = None,
) -> None:
    """Reconcile the wall-time tracker against a streamed graph chunk.

    Works for any ``concurrency_limit``. Two passes run per chunk:

    1. Completion pass — any analyst whose report is present in this chunk is
       marked completed at ``now``. Completion is idempotent and uses the
       analyst's *original* start time, so a chunk that merely re-carries an
       already-finished report (or omits a finished one entirely) leaves the
       recorded wall time untouched.
    2. Activation pass — the first ``concurrency_limit`` not-yet-completed
       analysts (in plan order) are marked started. This opens a slot for the
       next analyst the moment an earlier one finishes, so several analysts can
       be active at once and each accumulates real wall time from its own start.

    The tracker's own completion state (``get_wall_times``) drives the active
    window, so this stays correct even when the stream emits per-node deltas
    that do not repeat previously seen reports.
    """
    current_time = monotonic() if now is None else now

    # Pass 1: complete every analyst whose report landed in this chunk.
    for spec in tracker.plan.specs:
        if not bool(chunk.get(spec.report_key)):
            continue
        # Guarantee a start time so an analyst whose report we see before it was
        # ever activated still appears in the summary (collapsing to ~0s) rather
        # than vanishing; an already-active analyst keeps its real start.
        tracker.mark_started(spec.key, started_at=current_time)
        tracker.mark_completed(spec.key, completed_at=current_time)

    # Pass 2: keep up to concurrency_limit not-yet-completed analysts active.
    completed = tracker.get_wall_times()
    slots = tracker.plan.concurrency_limit
    for spec in tracker.plan.specs:
        if slots <= 0:
            break
        if spec.key in completed:
            continue
        tracker.mark_started(spec.key, started_at=current_time)
        slots -= 1
