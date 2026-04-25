"""Cost tracker with per-call records and budget enforcement."""
from __future__ import annotations

import threading
from dataclasses import dataclass

# SiliconFlow (.com) pricing, USD per 1M tokens (prompt, completion).
# Approximate — update from scripts/fetch_siliconflow_pricing.py when needed.
PRICING: dict[str, tuple[float, float]] = {
    "Qwen/Qwen2.5-72B-Instruct": (0.30, 0.45),
    "Qwen/Qwen3-32B": (0.27, 0.41),
    "Qwen/Qwen2.5-7B-Instruct": (0.0, 0.0),
    "Qwen/Qwen3-8B": (0.0, 0.0),
    "deepseek-ai/DeepSeek-R1": (0.55, 2.19),
    "deepseek-ai/DeepSeek-V3": (0.27, 1.09),
    "Qwen/Qwen3-Embedding-0.6B": (0.0, 0.0),
    "Qwen/Qwen3-Embedding-4B": (0.0, 0.0),
    "Qwen/Qwen3-Embedding-8B": (0.0, 0.0),
}


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_in, p_out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CallRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    phase: str
    agent_type: str
    latency_s: float


class CostTracker:
    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.spent_usd: float = 0.0
        self.calls: list[CallRecord] = []
        self._lock = threading.Lock()

    def record(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        phase: str,
        agent_type: str,
        latency_s: float,
    ) -> CallRecord:
        cost = compute_cost_usd(model, prompt_tokens, completion_tokens)
        with self._lock:
            rec = CallRecord(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                phase=phase,
                agent_type=agent_type,
                latency_s=latency_s,
            )
            self.calls.append(rec)
            self.spent_usd += cost
            if self.spent_usd > self.budget_usd:
                raise BudgetExceeded(
                    f"Spent ${self.spent_usd:.4f} exceeds budget ${self.budget_usd:.2f}"
                )
        return rec

    def summary(self) -> dict[str, object]:
        by_type: dict[str, float] = {}
        by_phase: dict[str, float] = {}
        by_model: dict[str, float] = {}
        for c in self.calls:
            by_type[c.agent_type] = by_type.get(c.agent_type, 0.0) + c.cost_usd
            by_phase[c.phase] = by_phase.get(c.phase, 0.0) + c.cost_usd
            by_model[c.model] = by_model.get(c.model, 0.0) + c.cost_usd
        return {
            "total_usd": round(self.spent_usd, 6),
            "calls": len(self.calls),
            "by_type": {k: round(v, 6) for k, v in by_type.items()},
            "by_phase": {k: round(v, 6) for k, v in by_phase.items()},
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
        }


# Module-level singleton — worker threads share a reference without having
# to thread the tracker through every function signature.
_tracker: CostTracker | None = None


def set_tracker(t: CostTracker | None) -> None:
    global _tracker
    _tracker = t


def get_tracker() -> CostTracker | None:
    return _tracker
