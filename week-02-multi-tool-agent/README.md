# Materials Research Assistant (Multi-Tool AI Agent)

Week 2 assignment for atomcamp's Agentic AI course (Cohort 5): a practical,
multi-tool AI agent built with LangChain, framed for my own field, materials
informatics.

## Project overview

**Problem.** A materials engineer's questions span different sources: property
databases, recent literature, quick unit calculations, and their own documents.
Normally that means juggling several tools and tabs.

**Solution.** A LangChain agent that reads the question, decides which tool or
tools it needs, calls them, and combines the results into one answer.

**Target users.** Materials scientists, informatics engineers, and students.

**Why an agent.** The questions genuinely need different tools, and the agent
selects and combines them dynamically. That is the point of a tool-using agent
rather than a single prompt.

## Architecture

Built with **LangChain v1** (`create_agent` + `init_chat_model`), the current agent API.

```
Gradio UI (app.py)
      |
      v
LangChain v1 agent, create_agent (agent.py)   <-- LLM: Groq (default) or Anthropic
      |
      +-- optimade_search            (external API: broad search across many databases)
      +-- materials_project_lookup   (external API: deep Materials Project properties)
      +-- search_arxiv               (external API: arXiv papers)
      +-- search_uploaded_documents  (RAG over an uploaded PDF)
      +-- convert_units              (custom Python function)
      +-- web_search                 (DuckDuckGo)
```

The agent chooses tools based on the question, feeds their outputs back into the
LLM, and the LLM writes the final answer. Every tool fails soft: on an error it
returns a message instead of crashing the loop.

Two of the tools are designed to work together: `optimade_search` finds materials
broadly across many databases and returns their ids, and for a Materials Project
hit (an mp- id) the agent then calls `materials_project_lookup` for the deep
properties. Broad discovery first, deep dive second.

## Tools and APIs used

| Tool | Type | What it does |
|---|---|---|
| `optimade_search` | External API | Broad search across many databases (Materials Project, OQMD, COD, Alexandria) via the OPTIMADE standard |
| `materials_project_lookup` | External API | Deep computed properties (band gap, density, formation energy, stability) by mp-id or formula |
| `search_arxiv` | External API | Recent papers on materials, physics, machine learning |
| `search_uploaded_documents` | RAG retrieval | Answers from a PDF the user uploads |
| `convert_units` | Custom Python | Materials unit conversions (stress, density, temperature) |
| `web_search` | Web | General or recent information (DuckDuckGo) |

**RAG pipeline:** PDF loading (`PyPDFLoader`) then text splitting
(`RecursiveCharacterTextSplitter`) then embeddings
(`sentence-transformers/all-MiniLM-L6-v2`, free and local) then a FAISS vector
store then relevant-passage retrieval, with the agent's LLM generating the
answer from the retrieved passages.

## Setup instructions

```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Configure your keys
cp .env.example .env        # then edit .env

#    - LLM_PROVIDER=groq and a free GROQ_API_KEY from https://console.groq.com
#      (or set LLM_PROVIDER=anthropic with an ANTHROPIC_API_KEY)
#    - MP_API_KEY: a free key from https://materialsproject.org/api

# 3. Run the app
python app.py               # opens the Gradio interface in your browser
```

The first run downloads the small embedding model, so it needs an internet
connection and takes a minute.

## Example queries

- Which binary titanium oxides appear across the materials databases?
- What is the band gap of mp-149?
- What is the density of SiO2 in the Materials Project?
- Convert 210 GPa to psi
- Convert 7.87 g/cm3 to kg/m3
- Find recent arXiv papers on lithium solid state electrolytes
- Upload a datasheet, then: what is the maximum service temperature in the document?

## Limitations

- Materials Project values are computed with DFT, so band gaps are underestimated
  and should be treated as lower bounds.
- The RAG store is in memory and holds one document at a time; uploading a new
  PDF replaces the previous one, and it is not saved between runs.
- DuckDuckGo search can rate-limit; the tool reports this rather than failing hard.
- Tool selection quality depends on the chosen LLM.

## About

Built by Ibtisam Ahmed Khan, a materials engineer working in data and AI.

- Materials Decoded: https://materialsdecoded.com
- GitHub: https://github.com/ibtisamkhan96
- LinkedIn: https://www.linkedin.com/in/ibtisam-ahmed-khan
