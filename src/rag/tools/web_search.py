from os import environ as env

from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool
from llama_index.tools.tavily_research import TavilyToolSpec

load_dotenv()


_tavily_spec: TavilyToolSpec | None = None


def _spec() -> TavilyToolSpec | None:
    global _tavily_spec
    if _tavily_spec is not None:
        return _tavily_spec
    key = env.get("TAVILY_API_KEY")
    if not key:
        return None
    _tavily_spec = TavilyToolSpec(api_key=key)
    return _tavily_spec


def web_search(query: str, max_results: int = 7) -> list[dict]:
    """Search the web with Tavily and return the top results.

    Args:
        query: The search query.
        max_results: Maximum results to return (default 7).
    """
    spec = _spec()
    if spec is None:
        return [{"error": "TAVILY_API_KEY is not configured"}]
    documents = spec.search(query=query, max_results=max_results)
    return [
        {"url": doc.extra_info.get("url"), "content": doc.text}
        for doc in documents
    ]


web_search_tool = FunctionTool.from_defaults(
    fn=web_search,
    name="web_search",
)
