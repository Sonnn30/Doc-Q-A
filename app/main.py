from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
import uvicorn

from app.rag import process_file, ask, get_session_info, delete_session, list_sessions
from app.schemas import (
    UploadResponse,
    AskRequest,
    AskResponse,
    SessionInfo,
    DeleteResponse,
)

app = FastAPI(
    title="RAG Q&A API",
    description=(
        "Upload file (PDF/DOCX/TXT/MD), lalu tanya jawab tentang isinya "
        "menggunakan Groq LLM + ChromaDB + session history."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_MB = 20


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "RAG Q&A API berjalan. Upload file di POST /upload, lalu tanya di POST /ask",
    }


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["RAG"])
async def upload_file(
    file: UploadFile = File(..., description="File PDF, DOCX, TXT, atau MD"),
):
    """
    Upload dokumen untuk diproses.
    Mengembalikan **session_id** yang dipakai untuk Q&A berikutnya.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: {ext}. Gunakan: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File terlalu besar ({size_mb:.1f} MB). Maksimal {MAX_FILE_SIZE_MB} MB.",
        )

    # Simpan sementara ke disk
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    tmp_path.write_bytes(content)

    try:
        result = process_file(
            file_path=tmp_path,
            file_ext=ext,
            filename=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses file: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResponse(
        session_id=result["session_id"],
        filename=file.filename,
        chunk_count=result["chunk_count"],
        message=f"File '{file.filename}' berhasil diproses menjadi {result['chunk_count']} chunk. Kamu bisa mulai bertanya!",
    )


# ── Ask ───────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(body: AskRequest):
    """
    Ajukan pertanyaan tentang dokumen yang sudah diupload.
    Gunakan **session_id** dari respons `/upload`.
    History percakapan disimpan otomatis per sesi.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

    try:
        result = ask(session_id=body.session_id, question=body.question)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menjawab: {str(e)}")

    return AskResponse(
        answer=result["answer"],
        session_id=body.session_id,
        sources=result["sources"],
        history_length=result["history_length"],
    )


# ── Session Management ────────────────────────────────────────────────────────

@app.get("/sessions", response_model=list[SessionInfo], tags=["Session"])
def get_all_sessions():
    """Lihat semua sesi aktif beserta info file dan history."""
    return list_sessions()


@app.get("/sessions/{session_id}", response_model=SessionInfo, tags=["Session"])
def get_session(session_id: str):
    """Lihat info sesi tertentu."""
    try:
        return get_session_info(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/sessions/{session_id}", response_model=DeleteResponse, tags=["Session"])
def remove_session(session_id: str):
    """Hapus sesi dan data vector dari ChromaDB."""
    try:
        delete_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return DeleteResponse(
        message="Sesi berhasil dihapus.",
        session_id=session_id,
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
