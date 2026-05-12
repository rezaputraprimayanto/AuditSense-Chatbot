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
1. Buat token Hugging Face
   - Login ke Hugging Face.
   - Buka `Settings` → `Access Tokens`.
   - Klik `Create new token`.
   - Pilih token tipe `Read` saja, karena hanya untuk download model.
   - Copy token yang muncul, formatnya biasanya seperti:
     ```text
     hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     ```
   > Hugging Face CLI menggunakan User Access Token untuk autentikasi saat login atau mengakses repo tertentu.

2. Install Hugging Face CLI
   Di terminal:
   ```powershell
   pip install -U "huggingface_hub[cli]"
   ```
   Cek sudah terinstall:
   ```powershell
   huggingface-cli --help
   ```

3. Login CLI
   Jalankan:
   ```powershell
   huggingface-cli login
   ```
   - Nanti akan muncul prompt untuk memasukkan token.
   - Paste token `hf_...`, lalu tekan Enter.
   - Jika ditanya `Add token as git credential? (Y/n)`, tekan Enter saja atau ketik `Y`.
   Cek login berhasil:
   ```powershell
   huggingface-cli whoami
   ```

4. Download model
   - Buat folder jika belum ada:
     ```powershell
     mkdir -p models/gemma4
     cd models/gemma4
     ```
   - Download file GGUF:
     ```powershell
     huggingface-cli download ggml-org/gemma-4-26B-A4B-it-GGUF \
       gemma-4-26B-A4B-it-Q4_K_M.gguf \
       --local-dir .
     ```
   - Setelah selesai, file akan ada di:
     ```text
     models/gemma4/gemma-4-26B-A4B-it-Q4_K_M.gguf
     ```

5. Versi satu blok
   ```powershell
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login
   mkdir -p models/gemma4
   cd models/gemma4
   huggingface-cli download ggml-org/gemma-4-26B-A4B-it-GGUF \
     gemma-4-26B-A4B-it-Q4_K_M.gguf \
     --local-dir .
   ```

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

