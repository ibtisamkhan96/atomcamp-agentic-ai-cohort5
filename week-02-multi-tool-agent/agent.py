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
from tools.optimade_search import optimade_search
from tools.rapidapi_element import element_lookup
from tools.arxiv_search import search_arxiv
from tools.web_search import web_search
from tools.rag import search_uploaded_documents

load_dotenv()

# Seven tools: four external APIs (OPTIMADE across many databases, Materials
# Project, arXiv, and a periodic-table API via RapidAPI), one RAG retrieval tool,
# one custom Python function, and one general web search.
TOOLS = [
    optimade_search,            # external API: broad search across many databases
    materials_project_lookup,   # external API: deep properties from Materials Project
    element_lookup,             # external API via RapidAPI: chemical element properties
    search_arxiv,               # external API: recent papers
    search_uploaded_documents,  # RAG document retrieval
    convert_units,              # custom Python function
    web_search,                 # additional tool
]

SYSTEM_PROMPT = (
    "You are a materials research assistant for a materials informatics engineer. "
    "Your tools are: optimade_search (broad search for materials by element across many "
    "databases, Materials Project, OQMD, COD, Alexandria), materials_project_lookup (deep "
    "computed properties of one material from the Materials Project), element_lookup (properties "
    "of a single chemical element, atomic number, mass, symbol, via a periodic-table API), "
    "search_arxiv (recent scientific papers), search_uploaded_documents (the PDF the user "
    "uploaded), convert_units (materials unit conversions), and web_search (general or recent "
    "information). "
    "Workflow: for broad 'which materials contain these elements' questions use optimade_search; "
    "if the user then wants detailed properties of a specific Materials Project result, take its "
    "mp- id and call materials_project_lookup. "
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
    # gpt-oss-120b is used as the Groq default because it emits tool calls
    # reliably; llama-3.3-70b sometimes garbles them (Groq returns tool_use_failed).
    return "groq:openai/gpt-oss-120b" if provider == "groq" else "anthropic:claude-sonnet-4-6"


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
