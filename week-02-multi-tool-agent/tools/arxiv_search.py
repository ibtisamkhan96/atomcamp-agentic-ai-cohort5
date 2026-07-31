"""External API tool #2: arXiv.

Searches arXiv for recent scientific papers (materials science, physics,
machine learning) and returns the top few with titles, authors, dates and
short summaries. No API key needed. Fails soft.
"""
from langchain.tools import tool


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for recent scientific papers on materials science, physics or machine learning.
    Returns the top few papers with titles, authors, dates and summaries."""
    try:
        from langchain_community.utilities import ArxivAPIWrapper
        wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=1200)
        result = wrapper.run(query)
        return result or "No arXiv results found for that query."
    except Exception as e:
        return f"arXiv search failed: {e}"
