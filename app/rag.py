"""
rag.py — Core RAG engine
Alur:
  Upload  → load_document → split chunks → embed → simpan ke ChromaDB (per session)
  Ask     → embed pertanyaan → retrieve chunks → kirim ke Groq LLM + history → jawaban
"""

import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

from app.loaders import load_document

load_dotenv()

# ── Konfigurasi ──────────────────────────────────────────────────────────────

CHROMA_DIR   = os.getenv("CHROMA_DIR", "./chroma_db")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
RETRIEVAL_K  = int(os.getenv("RETRIEVAL_K", "4"))

# ── In-memory session store ──────────────────────────────────────────────────
# Format: { session_id: { "filename": str, "chunk_count": int, "history": [...] } }
_sessions: dict[str, dict[str, Any]] = {}

# ── Singleton embeddings (load sekali, reuse) ────────────────────────────────
_embeddings = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        # Model ringan, jalan lokal, tidak butuh API key
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY tidak ditemukan. "
            "Daftar gratis di https://console.groq.com lalu set di .env"
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        groq_api_key=api_key,
    )


def _get_vectorstore(session_id: str) -> Chroma:
    """Buka atau buat ChromaDB collection untuk session ini."""
    return Chroma(
        collection_name=f"session_{session_id}",
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


# ── Upload & Indexing ─────────────────────────────────────────────────────────

def process_file(file_path: Path, file_ext: str, filename: str) -> dict[str, Any]:
    """
    Load → split → embed → simpan ke ChromaDB.
    Return: { session_id, chunk_count }
    """
    # 1. Load teks
    raw_text = load_document(file_path, file_ext)

    # 2. Split jadi chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    docs = [
        Document(page_content=chunk, metadata={"source": filename, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    # 3. Buat session ID & simpan ke ChromaDB
    session_id = str(uuid.uuid4())
    vectorstore = _get_vectorstore(session_id)
    vectorstore.add_documents(docs)

    # 4. Simpan session info
    _sessions[session_id] = {
        "filename": filename,
        "chunk_count": len(docs),
        "history": [],  # list of LangChain message objects
    }

    return {"session_id": session_id, "chunk_count": len(docs)}


# ── Q&A ───────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Kamu adalah asisten yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.

Aturan:
- Jawab HANYA berdasarkan konteks dokumen yang diberikan
- Jika informasi tidak ada di dokumen, katakan dengan jujur: "Informasi ini tidak tersedia dalam dokumen."
- Gunakan bahasa yang sama dengan pertanyaan user (Indonesia atau Inggris)
- Jawaban harus jelas, terstruktur, dan informatif
- Jika ada data/angka dari dokumen, sebutkan dengan tepat"""


def ask(session_id: str, question: str) -> dict[str, Any]:
    """
    Retrieve relevan chunks → bangun prompt dengan history → tanya Groq → simpan ke history.
    Return: { answer, sources, history_length }
    """
    # Validasi session
    if session_id not in _sessions:
        raise ValueError(
            f"Session '{session_id}' tidak ditemukan. "
            "Upload file terlebih dahulu untuk mendapatkan session_id."
        )

    session = _sessions[session_id]

    # 1. Retrieve chunks relevan dari ChromaDB
    vectorstore = _get_vectorstore(session_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return {
            "answer": "Tidak ditemukan informasi relevan dalam dokumen untuk menjawab pertanyaan ini.",
            "sources": [],
            "history_length": len(session["history"]),
        }

    # 2. Bangun konteks dari chunks
    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    sources = [doc.page_content[:200] + "..." for doc in relevant_docs]

    # 3. Bangun messages: system + history + user baru
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Tambahkan history percakapan sebelumnya
    messages.extend(session["history"])

    # Tambahkan pertanyaan baru dengan konteks
    user_message_content = (
        f"Konteks dari dokumen '{session['filename']}':\n\n"
        f"{context}\n\n"
        f"Pertanyaan: {question}"
    )
    messages.append(HumanMessage(content=user_message_content))

    # 4. Tanya Groq
    llm = _get_llm()
    response = llm.invoke(messages)
    answer = response.content.strip()

    # 5. Simpan ke history (simpan versi clean tanpa konteks agar history tidak membengkak)
    session["history"].append(HumanMessage(content=question))
    session["history"].append(AIMessage(content=answer))

    # Batasi history maksimal 20 pesan (10 turn) agar tidak melebihi context window
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    return {
        "answer": answer,
        "sources": sources,
        "history_length": len(session["history"]),
    }


# ── Session Management ────────────────────────────────────────────────────────

def get_session_info(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        raise ValueError(f"Session '{session_id}' tidak ditemukan.")
    s = _sessions[session_id]
    return {
        "session_id": session_id,
        "filename": s["filename"],
        "chunk_count": s["chunk_count"],
        "history_length": len(s["history"]),
    }


def delete_session(session_id: str) -> None:
    """Hapus session dari memory dan ChromaDB."""
    if session_id not in _sessions:
        raise ValueError(f"Session '{session_id}' tidak ditemukan.")

    # Hapus dari ChromaDB
    try:
        vectorstore = _get_vectorstore(session_id)
        vectorstore.delete_collection()
    except Exception:
        pass  # Lanjutkan meski gagal hapus dari ChromaDB

    # Hapus dari memory
    del _sessions[session_id]


def list_sessions() -> list[dict[str, Any]]:
    return [
        {
            "session_id": sid,
            "filename": s["filename"],
            "chunk_count": s["chunk_count"],
            "history_length": len(s["history"]),
        }
        for sid, s in _sessions.items()
    ]
