"""
Timing utilities for Agent runtime analysis.

Provides TimingStats for per-agent and per-paper timing aggregation,
and timed_step for wrapping function calls with timing.

Uses time.perf_counter() for high precision, thread-safe for parallel pipelines.
"""
import time
import json
import threading
from pathlib import Path
from typing import Callable, Any, Dict, List


class TimingStats:
    """
    Accumulates timing statistics per agent name.
    Supports per-paper breakdown (keyed by paper_idx) and aggregate totals.
    Thread-safe for parallel pipeline execution.
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._paper_timings: Dict[int, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def record(self, paper_idx: int, name: str, elapsed: float) -> None:
        """
        Record a single timing for an agent step.

        Args:
            paper_idx: Paper index (1-based) for per-paper breakdown
            name: Agent step name (e.g., "Extraction", "Grounding")
            elapsed: Elapsed time in seconds
        """
        with self._lock:
            if paper_idx not in self._paper_timings:
                self._paper_timings[paper_idx] = {}
            self._paper_timings[paper_idx][name] = elapsed
            if name not in self._data:
                self._data[name] = {"total": 0.0, "calls": 0, "per_call": []}
            self._data[name]["total"] += elapsed
            self._data[name]["calls"] += 1
            self._data[name]["per_call"].append(elapsed)

    def get_aggregate(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregate timing data for all agents."""
        return self._data.copy()

    def format_report(self, logger=None) -> str:
        """
        Format a human-readable timing report.

        Args:
            logger: Optional logger to also emit lines to

        Returns:
            Formatted report string
        """
        lines = []

        def emit(s: str) -> None:
            lines.append(s)
            if logger:
                logger.info(s)

        emit("=== Agent Timing Report ===")
        emit("")

        for p in sorted(self._paper_timings.keys()):
            paper_times = self._paper_timings[p]
            paper_total = sum(paper_times.values())
            emit(f"Per-Paper (Paper {p}):")
            for name, t in paper_times.items():
                emit(f"  {name}:  {t:.2f}s")
            emit(f"  Total:  {paper_total:.2f}s")
            emit("")

        emit("=== Aggregate ===")
        total_pipeline = 0.0
        agent_order = ("Extraction", "Grounding", "Validation", "Evolution")
        seen = set()
        for name in agent_order:
            if name not in self._data:
                continue
            seen.add(name)
            d = self._data[name]
            total = d["total"]
            calls = d["calls"]
            avg = total / calls if calls > 0 else 0
            total_pipeline += total
            emit(f"{name}:  {total:.2f}s ({calls} papers, avg {avg:.2f}s)")
        for name in sorted(self._data.keys()):
            if name not in seen:
                d = self._data[name]
                total = d["total"]
                calls = d["calls"]
                avg = total / calls if calls > 0 else 0
                total_pipeline += total
                emit(f"{name}:  {total:.2f}s ({calls} papers, avg {avg:.2f}s)")
        emit(f"Pipeline Total: {total_pipeline:.2f}s")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export timing data as a JSON-serializable dict."""
        return {
            "agents": self._data,
            "per_paper": {str(k): v for k, v in self._paper_timings.items()},
            "pipeline_total": sum(d["total"] for d in self._data.values()),
        }

    def save_json(self, path: str) -> None:
        """
        Save timing report to a JSON file.

        Args:
            path: File path to write (parent dirs created if needed)
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def timed_step(
    stats: TimingStats,
    paper_idx: int,
    name: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Execute a function and record its elapsed time.

    Args:
        stats: TimingStats instance to record into
        paper_idx: Paper index (1-based) for per-paper breakdown
        name: Agent step name for the record
        fn: Callable to execute (e.g., agent.process)
        *args: Positional arguments for fn
        **kwargs: Keyword arguments for fn

    Returns:
        Return value of fn(*args, **kwargs)
    """
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        return result
    finally:
        elapsed = time.perf_counter() - start
        stats.record(paper_idx, name, elapsed)
