"""Tavily real-time web search wrapper.

Defaults bumped from ai-agent-demo (was max_results=5, depth=basic) — in
PreFlight, Tavily is central to the Ontology phase (competitors + market
facts grounding), not cosmetic.
"""
from __future__ import annotations

from typing import Any

from tavily import TavilyClient

from sim_config import settings

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=settings().tavily_api_key)
    return _client


def tavily_search(
    query: str, *, max_results: int = 10, depth: str = "advanced"
) -> list[dict[str, Any]]:
    """Return a list of {url, title, content, score} dicts."""
    q = query[:400]
    response = _get_client().search(query=q, max_results=max_results, search_depth=depth)
    return [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]
