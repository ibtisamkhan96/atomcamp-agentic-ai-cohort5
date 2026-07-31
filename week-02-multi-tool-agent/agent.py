"""The LangChain multi-tool agent.

Registers the five tools, builds an LLM (Groq by default, Anthropic optional),
and wires them into a tool-calling agent that decides which tool or tools each
question needs, calls them, and combines the results.
"""
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools.converter import convert_units
from tools.materials_project import materials_project_lookup
from tools.arxiv_search import search_arxiv
from tools.web_search import web_search
from tools.rag import search_uploaded_documents

load_dotenv()

# The five tools: two external APIs, one RAG retrieval tool, one custom Python
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


def _build_llm():
    """Build the chat model. LLM_PROVIDER selects groq (default, free) or anthropic."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest")
        return ChatAnthropic(model=model, temperature=0)
    from langchain_groq import ChatGroq
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0)


def build_agent() -> AgentExecutor:
    """Create the tool-calling agent executor."""
    llm = _build_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


def run_agent(executor: AgentExecutor, query: str):
    """Run one query. Returns (answer, list_of_tool_names_used)."""
    result = executor.invoke({"input": query})
    steps = result.get("intermediate_steps", [])
    tools_used = [step[0].tool for step in steps]
    return result.get("output", ""), tools_used


if __name__ == "__main__":
    # Quick command-line smoke test (needs your .env keys set).
    executor = build_agent()
    for q in [
        "What is the band gap of mp-149?",
        "Convert 210 GPa to psi",
        "Find recent papers on solid state electrolytes",
    ]:
        answer, used = run_agent(executor, q)
        print("\nQ:", q)
        print("Tools:", used)
        print("A:", answer)
