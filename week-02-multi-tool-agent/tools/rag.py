"""RAG tool: question answering over an uploaded document.

This is the assignment's required RAG-based document retrieval tool. The full
pipeline is here: load a PDF, split it into chunks, embed the chunks with a
free HuggingFace model, store them in a FAISS vector database, and retrieve the
most relevant passages for a query. The agent's LLM then writes the answer from
those passages (the generation half of RAG).

The uploaded document lives in a small module-level state object so both the
Gradio app (which ingests the file) and the tool (which the agent calls) can
reach it.
"""
import os
from langchain_core.tools import tool


class _RagState:
    retriever = None
    source = None


STATE = _RagState()


def ingest_pdf(path: str) -> int:
    """Load a PDF, split, embed and build the in-memory FAISS store. Returns the chunk count."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    docs = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Free, local embedding model: no API key required.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    STATE.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    STATE.source = os.path.basename(path)
    return len(chunks)


@tool
def search_uploaded_documents(query: str) -> str:
    """Search the document the user uploaded and return the most relevant passages.
    Use this whenever the question refers to 'the document', 'the paper', 'the datasheet',
    or anything the user says they uploaded."""
    if STATE.retriever is None:
        return "No document has been uploaded yet. Ask the user to upload a PDF first."
    try:
        docs = STATE.retriever.invoke(query)
        if not docs:
            return "No relevant passages found in the uploaded document."
        passages = []
        for i, d in enumerate(docs, 1):
            page = d.metadata.get("page")
            tag = f" (page {page + 1})" if isinstance(page, int) else ""
            passages.append(f"[{i}]{tag} {d.page_content.strip()}")
        return "\n\n".join(passages)
    except Exception as e:
        return f"Document search failed: {e}"
