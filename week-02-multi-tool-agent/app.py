"""Gradio user interface for the Materials Research Assistant.

Provides a query box, an optional PDF upload (for the RAG tool), the agent's
answer, a panel showing which tools ran, and clear error messages when a call
fails. The agent is built once at startup.
"""
import os
import gradio as gr
from dotenv import load_dotenv

from agent import build_agent, run_agent
from tools.rag import ingest_pdf

load_dotenv()

# Build the agent once. If keys are missing, capture the error and show it in
# the UI instead of crashing.
try:
    AGENT = build_agent()
    STARTUP_ERROR = None
except Exception as e:
    AGENT = None
    STARTUP_ERROR = str(e)


def index_document(file):
    """Ingest an uploaded PDF into the RAG vector store.

    Gradio may hand us the file as a path string or as an object with a .name,
    so handle both.
    """
    if file is None:
        return "No file selected. Choose a PDF to index it."
    path = file if isinstance(file, str) else getattr(file, "name", None)
    if not path:
        return "Could not read the uploaded file."
    try:
        n_chunks = ingest_pdf(path)
        return f"Indexed '{os.path.basename(path)}' into {n_chunks} chunks. You can now ask about it."
    except Exception as e:
        return f"Failed to index the document: {e}"


def ask(query):
    """Run the agent on a query and return (answer, tools-used note)."""
    if AGENT is None:
        return f"Agent not available: {STARTUP_ERROR}. Check the keys in your .env file.", ""
    if not query or not query.strip():
        return "Please type a question first.", ""
    try:
        answer, tools_used = run_agent(AGENT, query)
        note = ", ".join(tools_used) if tools_used else "none (answered directly)"
        return answer, f"**Tools used:** {note}"
    except Exception as e:
        return f"Something went wrong while answering: {e}", ""


with gr.Blocks(title="Materials Research Assistant") as demo:
    gr.Markdown(
        "# Materials Research Assistant\n"
        "A multi-tool AI agent for materials work. It routes your question to the right tool: "
        "Materials Project properties, arXiv papers, your uploaded document (RAG), unit conversions, "
        "or a web search."
    )
    if STARTUP_ERROR:
        gr.Markdown(f"**Startup warning:** {STARTUP_ERROR}. Set your keys in .env and restart.")

    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(
                label="Your question",
                placeholder="e.g. What is the band gap of mp-149?  |  Convert 210 GPa to psi  |  What does my uploaded datasheet say about service temperature?",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")
            answer = gr.Markdown(label="Answer")
            tools_note = gr.Markdown()
        with gr.Column(scale=1):
            gr.Markdown("### Upload a document (optional)\nFor questions about your own PDF (a paper or datasheet). It indexes automatically on upload; wait for the 'Indexed ... chunks' message, then ask.")
            pdf = gr.File(file_types=[".pdf"], label="PDF")
            index_btn = gr.Button("Index document")
            index_status = gr.Markdown()
            gr.Markdown(
                "### Example queries\n"
                "- What is the band gap of mp-149?\n"
                "- Convert 7.87 g/cm3 to kg/m3\n"
                "- Recent arXiv papers on lithium solid electrolytes\n"
                "- Summarise the key results in the document I uploaded"
            )

    ask_btn.click(ask, inputs=query, outputs=[answer, tools_note])
    query.submit(ask, inputs=query, outputs=[answer, tools_note])
    index_btn.click(index_document, inputs=pdf, outputs=index_status)
    # Auto-index as soon as a PDF is uploaded, so the extra button click is optional.
    pdf.upload(index_document, inputs=pdf, outputs=index_status)


if __name__ == "__main__":
    demo.launch()
