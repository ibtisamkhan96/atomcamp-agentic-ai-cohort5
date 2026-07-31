"""The LangChain (v1) multi-tool agent.

Built with the current LangChain 1.0 API: `create_agent` (which runs the core
agent loop on top of LangGraph) and `init_chat_model` (the unified model
initialiser). Registers the five tools and returns a ready-to-invoke agent.
"""
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from tools.converter import convert_units
from tools.materials_project import materials_project_lookup
from tools.arxiv_search import search_arxiv
from tools.web_search import web_search
from tools.rag import search_uploaded_documents

load_dotenv()

# Five tools: two external APIs, one RAG retrieval tool, one custom Python
# function, and one general web search.
TOOLS = [
    materials_project_lookup,   # external API 1
    search_arxiv,               # external API 2
    search_uploaded_documents,  # RAG document retrieval
    convert_units,              # custom Python function
    web_search,                 # additional tool
]

SYSTEM_PROMPT = (
    "You are a materials research assistant for a materials informatics engineer. "
    "Your tools are: materials_project_lookup (computed properties of inorganic materials), "
    "search_arxiv (recent scientific papers), search_uploaded_documents (the PDF the user uploaded), "
    "convert_units (materials unit conversions), and web_search (general or recent information). "
    "Read the question, decide which tool or tools it needs, call them, and combine the results "
    "into a clear, well-structured answer. Only state what the tools return. If a tool fails or has "
    "no data, say so honestly and never invent values. Use plain punctuation."
)


def _model_id() -> str:
    """Resolve the model string in the 'provider:model' form init_chat_model expects."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL")
    if model:
        return model if ":" in model else f"{provider}:{model}"
    return "groq:llama-3.3-70b-versatile" if provider == "groq" else "anthropic:claude-sonnet-4-6"


def build_agent():
    """Create the LangChain v1 agent (a compiled LangGraph)."""
    model = init_chat_model(_model_id(), temperature=0)
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def run_agent(agent, query: str):
    """Run one query. Returns (answer_text, list_of_tool_names_used)."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = result["messages"]

    # Which tools ran: collect the names of every tool call the model made.
    tools_used = []
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tools_used.append(name)

    # The final answer is the content of the last message.
    answer = messages[-1].content if messages else ""
    if isinstance(answer, list):  # some providers return a list of content blocks
        answer = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in answer
        )
    return answer, tools_used


if __name__ == "__main__":
    # Quick command-line smoke test (needs your .env keys set).
    agent = build_agent()
    for q in [
        "What is the band gap of mp-149?",
        "Convert 210 GPa to psi",
        "Find recent papers on solid state electrolytes",
    ]:
        answer, used = run_agent(agent, q)
        print("\nQ:", q)
        print("Tools:", used)
        print("A:", answer)
