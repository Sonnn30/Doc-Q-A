# RAG Q&A API

API tanya jawab dokumen menggunakan **Groq LLM** + **ChromaDB** + **LangChain**.

## Stack
| Komponen | Teknologi |
|---|---|
| API Framework | FastAPI |
| LLM | Groq (llama-3.3-70b) |
| Vector Store | ChromaDB (persisten ke disk) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (lokal, gratis) |
| Session History | In-memory per session |

---

## Cara Kerja

```
Upload File
    ↓
Load teks (PDF/DOCX/TXT/MD)
    ↓
Split jadi chunks (800 token per chunk)
    ↓
Embed pakai sentence-transformers → simpan ke ChromaDB
    ↓
Dapat session_id

Tanya (Ask)
    ↓
Embed pertanyaan → cari 4 chunk paling relevan di ChromaDB
    ↓
Kirim ke Groq: system prompt + history + konteks + pertanyaan
    ↓
Dapat jawaban + simpan ke history sesi
```

---

## Setup

### 1. Buat virtual environment
```bash
conda create -n rag-api python=3.11 -y
conda activate rag-api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Buat file .env
```bash
cp .env.example .env
```

Edit `.env` dan isi `GROQ_API_KEY`:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxx   # Daftar gratis di https://console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Jalankan server
```bash
uvicorn app.main:app --reload
```

Buka: http://localhost:8000/docs

---

## Cara Pakai

### Step 1 — Upload file
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@dokumen_saya.pdf"
```

Response:
```json
{
  "session_id": "abc-123-xxx",
  "filename": "dokumen_saya.pdf",
  "chunk_count": 24,
  "message": "File berhasil diproses..."
}
```

### Step 2 — Tanya jawab (pakai session_id)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123-xxx",
    "question": "Apa kesimpulan utama dokumen ini?"
  }'
```

### Step 3 — Tanya lagi (history otomatis tersimpan)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123-xxx",
    "question": "Bisa jelaskan lebih detail poin pertama?"
  }'
```

---

## Endpoints

| Method | URL | Deskripsi |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload & proses file |
| `POST` | `/ask` | Tanya jawab dokumen |
| `GET` | `/sessions` | Lihat semua sesi aktif |
| `GET` | `/sessions/{id}` | Info sesi tertentu |
| `DELETE` | `/sessions/{id}` | Hapus sesi & data vector |

---

## Model Groq yang Bisa Dipakai

| Model | Kecepatan | Kemampuan |
|---|---|---|
| `llama-3.3-70b-versatile` | Sedang | ⭐⭐⭐⭐⭐ Terbaik |
| `llama-3.1-8b-instant` | Sangat cepat | ⭐⭐⭐ |
| `mixtral-8x7b-32768` | Sedang | ⭐⭐⭐⭐ Context besar |

Ganti di `.env`:
```
GROQ_MODEL=llama-3.1-8b-instant
```

---

## Catatan

- **Session hilang** saat server restart (history in-memory). ChromaDB tetap ada di disk.
- **Maksimal file**: 20 MB
- **Format didukung**: PDF, DOCX, TXT, MD
- **History per sesi**: Disimpan maksimal 20 pesan (10 turn) untuk menghindari context overflow
