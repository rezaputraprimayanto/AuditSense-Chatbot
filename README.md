# AuditSense-Chatbot
Local LLM Based RAG System untuk PDF dan DOCX

## Deskripsi
AuditSense-Chatbot adalah aplikasi local RAG yang memungkinkan upload dokumen PDF dan DOCX, kemudian menjawab pertanyaan berdasarkan isi dokumen tersebut.

## Prasyarat
- Python 3.11+ (direkomendasikan)
- Node.js 20+ dan npm
- Git
- Ruang disk cukup untuk model `gemma-4-26B-A4B-it-Q4_K_M.gguf`

## Instalasi Backend
1. Buka terminal dan masuk ke folder backend:
   ```powershell
   cd backend
   ```
2. Buat virtual environment:
   ```powershell
   python -m venv .venv
   ```
3. Aktifkan virtual environment:
   ```powershell
   .\.venv\Scripts\Activate
   ```
4. Pasang dependensi Python:
   ```powershell
   pip install -r requirements.txt
   ```

## Download Model
1. Download model `gemma-4-26B-A4B-it-Q4_K_M.gguf` dari Hugging Face.
2. Letakkan file model di path berikut relatif ke root repository:
   ```text
   models/gemma4/gemma-4-26B-A4B-it-Q4_K_M.gguf
   ```
3. Jika folder `models` atau `models/gemma4` belum ada, buat folder tersebut terlebih dahulu.

> Catatan: konfigurasi default backend berada di `backend/src/config.py` pada variabel `LLAMA_GGUF_PATH`.

## Instalasi Frontend
1. Buka terminal dan masuk ke folder frontend:
   ```powershell
   cd frontend
   ```
2. Pasang dependensi JavaScript:
   ```powershell
   npm install
   ```

## Menjalankan Aplikasi
### Jalankan backend
Dari folder `backend`:
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Jalankan frontend
Dari folder `frontend`:
```powershell
npm run dev
```

## Menggunakan Aplikasi
1. Buka browser ke `http://localhost:3000`
2. Upload dokumen dengan format:
   - `.pdf`
   - `.docx`
3. Setelah upload berhasil, dokumen akan tersedia untuk diindeks dan dijadikan sumber jawaban.

## Catatan Penting
- Hanya file PDF dan DOCX yang didukung.
- Jika model tidak ditemukan, pastikan file `gemma-4-26B-A4B-it-Q4_K_M.gguf` berada di folder `models/gemma4`.
- Backend harus dijalankan sebelum frontend dapat menggunakan API upload dan preview dokumen.

