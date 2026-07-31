"""Gradio user interface for the Materials Research Assistant.

One question box, one Ask button. If you attach a PDF, the app indexes it for you
(no separate step), and the Ask button streams a live status so you always know
it is working. The agent is built once at startup.
"""
import os
import gradio as gr
from dotenv import load_dotenv

from agent import build_agent, run_agent
from tools import rag
from tools.rag import ingest_pdf

load_dotenv()

# Build the agent once. If keys are missing, capture the error and show it in the
# UI instead of crashing.
try:
    AGENT = build_agent()
    STARTUP_ERROR = None
except Exception as e:
    AGENT = None
    STARTUP_ERROR = str(e)


def _file_path(file):
    """Gradio may pass a path string or an object with .name; handle both."""
    if file is None:
        return None
    return file if isinstance(file, str) else getattr(file, "name", None)


def index_document(file):
    """Embed a PDF into the RAG store. Runs automatically when a file is uploaded."""
    path = _file_path(file)
    if not path:
        return "No file selected."
    try:
        n_chunks = ingest_pdf(path)
        return f"Indexed '{os.path.basename(path)}' into {n_chunks} chunks."
    except Exception as e:
        return f"Failed to index the document: {e}"


def ask(query, file):
    """One-click flow: index the PDF if needed, then answer. Streams status updates."""
    if not query or not query.strip():
        yield "Please type a question first.", ""
        return
    if AGENT is None:
        yield f"Agent not available: {STARTUP_ERROR}. Check your .env keys.", ""
        return

    # If a PDF is attached and not yet indexed, index it now (once).
    path = _file_path(file)
    if path and os.path.basename(path) != rag.STATE.source:
        yield f"Indexing {os.path.basename(path)}...", ""
        try:
            ingest_pdf(path)
        except Exception as e:
            yield f"Failed to index the document: {e}", ""
            return

    yield "Working...", ""
    try:
        answer, tools_used = run_agent(AGENT, query)
        note = ", ".join(tools_used) if tools_used else "none (answered directly)"
        yield answer, f"**Tools used:** {note}"
    except Exception as e:
        yield f"Something went wrong while answering: {e}", ""


with gr.Blocks(title="Materials Research Assistant") as demo:
    gr.Markdown(
        "# Materials Research Assistant\n"
        "A multi-tool AI agent for materials work. It routes your question to the right tool: "
        "search across materials databases (OPTIMADE), Materials Project properties, a chemical "
        "element lookup, arXiv papers, your uploaded document (RAG), unit conversions, or a web search."
    )
    if STARTUP_ERROR:
        gr.Markdown(f"**Startup warning:** {STARTUP_ERROR}. Set your keys in .env and restart.")

    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is the band gap of mp-149?  |  Convert 210 GPa to psi  |  Summarise the PDF I attached",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer = gr.Markdown(label="Answer")
            tools_note = gr.Markdown()
        with gr.Column(scale=1):
            gr.Markdown("### Attach a document (optional)\nDrop a PDF here to ask about it. It indexes automatically, then just click Ask.")
            pdf = gr.File(file_types=[".pdf"], label="PDF")
            index_status = gr.Markdown()
            gr.Markdown(
                "### Example queries\n"
                "- Which binary titanium oxides exist across the databases?\n"
                "- What is the atomic mass and symbol of iron?\n"
                "- What is the band gap of mp-149?\n"
                "- Convert 7.87 g/cm3 to kg/m3\n"
                "- Summarise the key results in the PDF I attached"
            )

    ask_btn.click(ask, inputs=[query, pdf], outputs=[answer, tools_note])
    query.submit(ask, inputs=[query, pdf], outputs=[answer, tools_note])
    # Index as soon as a PDF is uploaded, so it is ready before you even ask.
    pdf.upload(index_document, inputs=pdf, outputs=index_status)


if __name__ == "__main__":
    demo.launch()
