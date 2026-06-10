from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    session_id: str = Field(..., description="ID sesi unik untuk Q&A berikutnya")
    filename: str = Field(..., description="Nama file yang diupload")
    chunk_count: int = Field(..., description="Jumlah chunk yang disimpan ke ChromaDB")
    message: str = Field(..., description="Pesan status")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123",
                "filename": "laporan.pdf",
                "chunk_count": 24,
                "message": "File berhasil diproses. Kamu bisa mulai bertanya!",
            }
        }


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' atau 'assistant'")
    content: str = Field(..., description="Isi pesan")


class AskRequest(BaseModel):
    session_id: str = Field(..., description="Session ID dari respons upload")
    question: str = Field(..., description="Pertanyaan tentang dokumen")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123",
                "question": "Apa kesimpulan utama dari dokumen ini?",
            }
        }


class AskResponse(BaseModel):
    answer: str = Field(..., description="Jawaban dari LLM berdasarkan dokumen")
    session_id: str
    sources: list[str] = Field(default=[], description="Potongan teks sumber yang dipakai")
    history_length: int = Field(..., description="Jumlah pesan dalam history sesi ini")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Kesimpulan utama dokumen ini adalah...",
                "session_id": "abc123",
                "sources": ["Pada bab 3, disebutkan bahwa..."],
                "history_length": 4,
            }
        }



class SessionInfo(BaseModel):
    session_id: str
    filename: str
    chunk_count: int
    history_length: int


class DeleteResponse(BaseModel):
    message: str
    session_id: str
