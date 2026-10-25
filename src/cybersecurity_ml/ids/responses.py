"""Recommendation-only response adapter interface for IDS actions."""
from typing import Protocol


class ResponseAdapter(Protocol):
    """A sink that receives policy actions and returns an execution record."""

    def dispatch(self, event_id: str | None, action: str) -> dict:
        """Handle an action without changing the model's decision."""


class RecommendationAdapter:
    """Safe default: report the action but never modify host/network controls."""

    def dispatch(self, event_id: str | None, action: str) -> dict:
        result = {"action": action, "mode": "recommendation_only"}
        if event_id is not None:
            result["event_id"] = event_id
        return result
