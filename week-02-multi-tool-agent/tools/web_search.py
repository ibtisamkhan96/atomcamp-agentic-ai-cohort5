"""Additional tool: general web search (DuckDuckGo).

For general or recent questions the other tools do not cover. No API key
needed. DuckDuckGo can rate-limit, so this fails soft and reports the error
rather than crashing the agent.
"""
from langchain.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for general or recent information not covered by the other tools."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        result = DuckDuckGoSearchRun().run(query)
        return result or "No web results found."
    except Exception as e:
        return f"Web search failed: {e}"
