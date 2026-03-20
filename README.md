# 🤖 RAG Chatbot — Dynamic Knowledge Retrieval System

> Built for **Cerevyn Solutions Campus Drive · March 2026**

A production-grade Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF document and ask questions from it in real time. Supports text-based, image-based, and mixed PDFs with OCR.

---

## 🚀 Live Demo

Deployed on Railway: `https://rag-chatbot-production.up.railway.app`

---

## 📌 Problem Statement

**Problem Statement 6 — Dynamic Knowledge Retrieval Chatbot (RAG)**

Design a chatbot that retrieves answers from dynamic document sources using a retrieval pipeline, context-based answer generation, and graceful handling of unknown queries.

---

## ✨ Features

- 📄 Upload any PDF — text-based, image-based (scanned), or mixed
- 🧠 Automatic OCR for image-based PDFs using Tesseract
- ⚡ Fast answers powered by Groq's llama-3.3-70b model
- 📑 Built-in PDF viewer with page navigation
- 🔍 Source page tracking — see exactly which pages the answer came from
- 💾 Persistent storage — app restores after browser refresh
- 🔄 Full reset functionality
- 📊 Session stats — questions asked, pages indexed
- 🐳 Docker containerized
- ⚙️ CI/CD pipeline via GitHub Actions → Railway

---

## 🏗️ RAG Architecture

```
User uploads PDF
       ↓
PyPDFLoader reads document
       ↓
RecursiveCharacterTextSplitter
(500 char chunks, 50 char overlap)
       ↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
converts chunks → 384-dim vectors
       ↓
FAISS Vector Store
indexes all vectors
       ↓
User asks a question
       ↓
FAISS Similarity Search
retrieves top 4 matching chunks
       ↓
Groq LLM (llama-3.3-70b)
answers strictly from retrieved chunks
       ↓
Answer displayed with source pages
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM | Groq llama-3.3-70b-versatile | Fast inference, free tier |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | 384-dim dense vectors, 22MB |
| Vector Store | FAISS (Facebook AI Similarity Search) | Fast nearest-neighbor search |
| RAG Framework | LangChain + LangChain Classic | Pipeline orchestration |
| OCR | Tesseract 5.5 + pdf2image + Poppler | Image-based PDF support |
| PDF Viewer | PDF.js 3.11 | In-browser PDF rendering |
| UI Framework | Streamlit | Web interface |
| Containerization | Docker | Deployment packaging |
| CI/CD | GitHub Actions | Auto deploy on push |
| Deployment | Railway | Cloud hosting |

---

## 📁 Project Structure

```
rag_chatbot/
├── app.py                      # Streamlit UI — all frontend logic
├── rag_engine.py               # RAG pipeline — all AI/ML logic
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container definition
├── start.py                    # Production startup script
├── .env                        # API keys (never commit this)
├── .gitignore                  # Ignores venv, .env, cache files
├── static/
│   └── pdf_viewer.html         # PDF.js viewer
├── vectorstore_cache/          # FAISS index saved to disk
│   ├── index.faiss
│   ├── index.pkl
│   └── meta.json               # PDF name + page contents
└── .github/
    └── workflows/
        └── deploy.yml          # CI/CD pipeline
```

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.11+
- Tesseract OCR installed
- Poppler installed

### 1. Clone the repository

```bash
git clone https://github.com/Bharath-342/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (Windows)

Download from: https://github.com/UB-Mannheim/tesseract/wiki

Install to default path: `C:\Program Files\Tesseract-OCR`

### 5. Install Poppler (Windows)

Download from: https://github.com/oschwartz10612/poppler-windows/releases

Extract to: `C:\Users\<YourName>\poppler\`

Update path in `rag_engine.py`:
```python
POPPLER_PATH = r"C:\Users\<YourName>\poppler\poppler-25.12.0\Library\bin"
```

### 6. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Get your free Groq API key at: https://console.groq.com

Get your HuggingFace token at: https://huggingface.co/settings/tokens

### 7. Run the app

```bash
streamlit run app.py
```

Open your browser at: `http://localhost:8501`

---

## 🐳 Docker Setup

### Build and run locally

```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 -e GROQ_API_KEY=your_key rag-chatbot
```

---

## ⚙️ CI/CD Pipeline

Every push to the `main` branch triggers:

1. **Test job** — installs dependencies, verifies imports, loads `rag_engine`
2. **Deploy job** — deploys to Railway automatically (only if tests pass)

Pipeline defined in `.github/workflows/deploy.yml`

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLM inference |
| `HF_TOKEN` | Optional | HuggingFace token for faster downloads |

---

## 💬 How to Use

1. Open the app in your browser
2. Upload a PDF file (text, scanned, or mixed)
3. Wait for the processing steps to complete:
   - PDF chunking
   - Page content storage
   - Embedding generation
   - QA chain initialization
4. Ask any question about the document
5. Click **"📚 Source pages used"** to see which pages the answer came from
6. Click **"📄 Go to Page X"** to jump the PDF viewer to that page
7. Ask **"what is on page 3"** to get a summary of a specific page
8. Use **"🗑 Clear"** to clear chat history
9. Use **"🔄 Reset"** to remove the current PDF and start fresh

---

## 🧠 Models Used

### LLM — Groq llama-3.3-70b-versatile
- 70 billion parameter model
- Runs on Groq's LPU (Language Processing Unit) hardware
- Sub-second inference latency
- Free tier available
- Temperature set to 0 for factual, deterministic answers

### Embeddings — HuggingFace all-MiniLM-L6-v2
- 22MB model, runs fully locally — no API call needed
- Produces 384-dimensional dense vectors
- Trained on 1 billion sentence pairs
- Optimized for semantic similarity search
- Fast inference: ~14,000 sentences/second on CPU

### Vector Store — FAISS
- Facebook AI Similarity Search
- IndexFlatL2 — exact L2 distance search
- Persisted to disk so vectorstore survives app restarts
- Top 4 chunks retrieved per query (configurable)

### OCR — Tesseract 5.5
- Open source OCR engine by Google
- Used when PDF has less than 100 characters of extractable text
- Mixed mode: only image-based pages get OCR, text pages stay as-is
- DPI set to 200 for optimal accuracy vs speed

---

## 📊 Example Outputs

**Question:** "What is this document about?"

**Answer:** "This document is about a Resume Job Description Matching System — a Backend API project that uses Node.js, Express.js, PostgreSQL, Docker, and Render Cloud to automate resume screening and provide compatibility scores."

---

**Question:** "What is on page 3?"

**Answer:** "Page 3 covers the API authentication system. It describes JWT-based token authentication, the middleware structure, and how protected routes are secured using the `protect` middleware function."

---

**Question:** "What is the matching algorithm?"

**Answer:** "The matching algorithm compares the resume skill set against job description requirements and computes a percentage-based compatibility score using the formula: MatchingScore = (Number of Matched Skills / Total Skills Required by JD) × 100"

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Author

Built by **Bharath** for the Cerevyn Solutions Campus Drive, March 2026.
