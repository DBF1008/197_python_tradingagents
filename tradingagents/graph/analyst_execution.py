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

    def get_wall_times(self) -> Dict[str, float]:
        return dict(self._wall_times)

    def is_started(self, analyst_key: str) -> bool:
        return analyst_key in self._started_at

    def is_completed(self, analyst_key: str) -> bool:
        return analyst_key in self._wall_times

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
    current_time = monotonic() if now is None else now

    # 1. Mark all specs whose report appears in this chunk as started.
    #    setdefault inside mark_started is idempotent — only the earliest
    #    timestamp is kept, so repeated chunks never overwrite a prior start.
    for spec in tracker.plan.specs:
        if chunk.get(spec.report_key):
            tracker.mark_started(spec.key, started_at=current_time)

    # 2. Process completions BEFORE computing capacity, so that analysts
    #    finishing in this chunk free their slot for the next pending one.
    #    mark_completed is idempotent (no-op if already recorded).
    for spec in tracker.plan.specs:
        if chunk.get(spec.report_key):
            tracker.mark_completed(spec.key, completed_at=current_time)

    # 3. How many analysts are still in-flight (started but not completed)?
    active_count = sum(
        1 for spec in tracker.plan.specs
        if tracker.is_started(spec.key) and not tracker.is_completed(spec.key)
    )
    available_capacity = tracker.plan.concurrency_limit - active_count

    # 4. Fill available capacity with the next pending analysts (plan order).
    for spec in tracker.plan.specs:
        if available_capacity <= 0:
            break
        if not tracker.is_started(spec.key):
            tracker.mark_started(spec.key, started_at=current_time)
            available_capacity -= 1
