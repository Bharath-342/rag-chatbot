import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import base64
import json
import shutil

from langchain_community.document_loaders import PyPDFLoader
from rag_engine import (
    load_and_split_pdf,
    build_vectorstore,
    build_qa_chain,
    ask_question,
    extract_page_number_from_question,
    save_vectorstore,
    load_vectorstore_from_disk
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
META_PATH       = os.path.join(BASE_DIR, "vectorstore_cache", "meta.json")
STATIC_DIR      = os.path.join(BASE_DIR, "static")
STATIC_PDF_PATH = os.path.join(STATIC_DIR, "current.pdf")

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "qa_chain": None,
    "chat_history": [],
    "pdf_name": None,
    "question_count": 0,
    "jump_to_page": 1,
    "page_contents": {}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auto restore on refresh ───────────────────────────────────────────────────
if st.session_state.qa_chain is None:
    _vs = load_vectorstore_from_disk()
    if _vs is not None:
        st.session_state.qa_chain = build_qa_chain(_vs)
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            st.session_state.pdf_name      = meta.get("pdf_name", None)
            st.session_state.page_contents = meta.get("page_contents", {})

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #12121a;
    --border:    #1e1e2e;
    --accent:    #7c3aed;
    --accent2:   #06b6d4;
    --accent3:   #f59e0b;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --success:   #10b981;
    --danger:    #ef4444;
    --glow:      rgba(124,58,237,0.4);
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(124,58,237,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(6,182,212,0.1) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 2px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Title ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.8rem !important;
    background: linear-gradient(135deg, #7c3aed, #06b6d4, #f59e0b) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -1px !important;
}

h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--accent) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent2) !important;
    box-shadow: 0 0 20px var(--glow) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: linear-gradient(135deg, #1e1e2e, #2d2d3f) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}

/* ── Chat messages ── */
/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    animation: slideIn 0.3s ease !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] div {
    color: #f1f5f9 !important;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
[data-testid="stChatInput"]:focus-within {
    box-shadow: 0 0 0 2px var(--glow) !important;
}

/* ── Status/expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Info/success/error boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border) !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent2) !important; }

/* ── Floating particle canvas ── */
#particle-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}
</style>

<!-- Floating particles background -->
<canvas id="particle-canvas"></canvas>
<script>
(function() {
    var canvas = document.getElementById('particle-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    var particles = [];
    for (var i = 0; i < 60; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.5 + 0.3,
            dx: (Math.random() - 0.5) * 0.3,
            dy: (Math.random() - 0.5) * 0.3,
            color: ['#7c3aed','#06b6d4','#f59e0b'][Math.floor(Math.random()*3)]
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(function(p) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.fill();
            p.x += p.dx;
            p.y += p.dy;
            if (p.x < 0 || p.x > canvas.width)  p.dx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
        });

        // Draw connecting lines between close particles
        for (var i = 0; i < particles.length; i++) {
            for (var j = i + 1; j < particles.length; j++) {
                var dist = Math.hypot(particles[i].x - particles[j].x, particles[i].y - particles[j].y);
                if (dist < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = 'rgba(124,58,237,' + (1 - dist/100) * 0.2 + ')';
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();

    window.addEventListener('resize', function() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
})();
</script>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem; margin-bottom:0.5rem;'>🤖</div>
        <div style='font-family: Syne, sans-serif; font-weight:800; font-size:1.3rem;
                    background: linear-gradient(135deg, #7c3aed, #06b6d4);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            RAG Chatbot
        </div>
        <div style='font-family: Space Mono, monospace; font-size:0.65rem; color:#64748b; margin-top:4px;'>
            Dynamic Knowledge Retrieval
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='font-family: Syne, sans-serif; font-weight:600; font-size:0.85rem;
                color:#7c3aed; margin-bottom:0.8rem; letter-spacing:2px;'>
        ⚡ HOW IT WORKS
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("📤", "Upload", "Any PDF — text, image or mixed"),
        ("✂️", "Chunk", "Split into 500-char pieces"),
        ("🧠", "Embed", "HuggingFace all-MiniLM-L6-v2"),
        ("📦", "Index", "Stored in FAISS vector store"),
        ("🔍", "Retrieve", "Top 4 chunks per question"),
        ("⚡", "Answer", "Groq llama-3.3-70b responds"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex; gap:10px; align-items:flex-start;
                    padding:8px; margin-bottom:6px;
                    background:#12121a; border-radius:8px;
                    border-left:2px solid #7c3aed;'>
            <span style='font-size:1.1rem;'>{icon}</span>
            <div>
                <div style='font-family:Syne,sans-serif; font-weight:600;
                            font-size:0.8rem; color:#e2e8f0;'>{title}</div>
                <div style='font-family:Space Mono,monospace; font-size:0.65rem;
                            color:#64748b;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='font-family:Syne,sans-serif; font-weight:600; font-size:0.85rem;
                color:#06b6d4; margin-bottom:0.8rem; letter-spacing:2px;'>
        🛠 TECH STACK
    </div>
    """, unsafe_allow_html=True)

    techs = [
        ("#7c3aed", "LangChain", "RAG Pipeline"),
        ("#06b6d4", "Groq", "LLM Inference"),
        ("#f59e0b", "FAISS", "Vector Store"),
        ("#10b981", "HuggingFace", "Embeddings"),
        ("#ef4444", "Tesseract", "OCR Engine"),
        ("#8b5cf6", "PDF.js", "PDF Viewer"),
        ("#ec4899", "Streamlit", "UI Framework"),
    ]
    cols = st.columns(2)
    for i, (color, name, role) in enumerate(techs):
        with cols[i % 2]:
            st.markdown(f"""
            <div style='background:#12121a; border:1px solid #1e1e2e;
                        border-top:2px solid {color};
                        border-radius:6px; padding:6px 8px; margin-bottom:6px;'>
                <div style='font-family:Syne,sans-serif; font-weight:600;
                            font-size:0.72rem; color:{color};'>{name}</div>
                <div style='font-family:Space Mono,monospace; font-size:0.6rem;
                            color:#64748b;'>{role}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='font-family:Syne,sans-serif; font-weight:600; font-size:0.85rem;
                color:#f59e0b; margin-bottom:0.8rem; letter-spacing:2px;'>
        📊 SESSION
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.pdf_name:
        st.markdown(f"""
        <div style='background:#12121a; border:1px solid #1e1e2e;
                    border-radius:8px; padding:12px; margin-bottom:8px;'>
            <div style='font-family:Space Mono,monospace; font-size:0.65rem;
                        color:#64748b; margin-bottom:4px;'>LOADED FILE</div>
            <div style='font-family:Syne,sans-serif; font-weight:600;
                        font-size:0.8rem; color:#10b981;
                        white-space:nowrap; overflow:hidden;
                        text-overflow:ellipsis;'>
                ✅ {st.session_state.pdf_name}
            </div>
        </div>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;'>
            <div style='background:#12121a; border:1px solid #1e1e2e;
                        border-radius:8px; padding:10px; text-align:center;'>
                <div style='font-family:Syne,sans-serif; font-weight:800;
                            font-size:1.4rem; color:#7c3aed;'>
                    {st.session_state.question_count}
                </div>
                <div style='font-family:Space Mono,monospace; font-size:0.6rem;
                            color:#64748b;'>QUESTIONS</div>
            </div>
            <div style='background:#12121a; border:1px solid #1e1e2e;
                        border-radius:8px; padding:10px; text-align:center;'>
                <div style='font-family:Syne,sans-serif; font-weight:800;
                            font-size:1.4rem; color:#06b6d4;'>
                    {len(st.session_state.page_contents)}
                </div>
                <div style='font-family:Space Mono,monospace; font-size:0.6rem;
                            color:#64748b;'>PAGES</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#12121a; border:1px dashed #1e1e2e;
                    border-radius:8px; padding:16px; text-align:center;'>
            <div style='font-size:1.5rem; margin-bottom:4px;'>📭</div>
            <div style='font-family:Space Mono,monospace; font-size:0.65rem;
                        color:#64748b;'>No PDF loaded yet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.6rem;
                color:#2d2d3f; text-align:center; line-height:1.6;'>
        Built for Cerevyn Solutions<br>Campus Drive · March 2026
    </div>
    """, unsafe_allow_html=True)

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.5rem;'>
    <h1 style='margin-bottom:0;'>🤖 RAG Chatbot</h1>
</div>
""", unsafe_allow_html=True)
st.caption("Dynamic Knowledge Retrieval — Upload any PDF and ask questions from it in real time.")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── PDF Upload ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;'>
    <div style='width:28px; height:28px; background:linear-gradient(135deg,#7c3aed,#5b21b6);
                border-radius:6px; display:flex; align-items:center; justify-content:center;
                font-size:0.9rem;'>📄</div>
    <span style='font-family:Syne,sans-serif; font-weight:600; font-size:1.1rem;'>
        Step 1 — Upload your PDF
    </span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Supports text-based, image-based, and mixed PDFs · Max 200MB",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.session_state.pdf_name != uploaded_file.name:
        try:
            pdf_bytes = uploaded_file.read()

            os.makedirs(STATIC_DIR, exist_ok=True)
            with open(STATIC_PDF_PATH, "wb") as f:
                f.write(pdf_bytes)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            with st.status("⚡ Processing your PDF...", expanded=True) as status:

                st.write("📖 Reading PDF and splitting into chunks...")
                chunks = load_and_split_pdf(tmp_path)
                st.write(f"✅ Done — **{len(chunks)}** chunks created.")

                st.write("📑 Storing page-level content...")
                raw_docs = PyPDFLoader(tmp_path).load()
                st.session_state.page_contents = {
                    i + 1: doc.page_content for i, doc in enumerate(raw_docs)
                }
                st.write(f"✅ Done — **{len(st.session_state.page_contents)}** pages stored.")

                st.write("🧠 Building embeddings and indexing into FAISS...")
                vectorstore = build_vectorstore(chunks)
                st.write("✅ Done — vectors stored in FAISS.")

                st.write("⚡ Connecting to Groq LLM...")
                st.session_state.qa_chain = build_qa_chain(vectorstore)
                st.write("✅ Done — QA chain ready.")

                st.write("💾 Saving to disk for refresh persistence...")
                save_vectorstore(vectorstore)
                os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
                with open(META_PATH, "w") as f:
                    json.dump({
                        "pdf_name": uploaded_file.name,
                        "page_contents": st.session_state.page_contents
                    }, f)
                st.write("✅ Done — data saved.")

                status.update(
                    label="✅ PDF indexed successfully!",
                    state="complete",
                    expanded=False
                )

            st.session_state.pdf_name       = uploaded_file.name
            st.session_state.chat_history   = []
            st.session_state.question_count = 0
            st.session_state.jump_to_page   = 1
            os.unlink(tmp_path)

        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.session_state.qa_chain = None
            st.session_state.pdf_name = None
    else:
        st.success(f"✅ **{uploaded_file.name}** is already loaded and ready.")

# ── Before upload ─────────────────────────────────────────────────────────────
if st.session_state.qa_chain is None:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;'>
        <div style='width:28px; height:28px; background:linear-gradient(135deg,#06b6d4,#0891b2);
                    border-radius:6px; display:flex; align-items:center; justify-content:center;
                    font-size:0.9rem;'>💬</div>
        <span style='font-family:Syne,sans-serif; font-weight:600; font-size:1.1rem;'>
            Step 2 — Ask questions
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.info("⬆️ Upload a PDF above to get started.")

# ── After upload: Chat + Viewer ───────────────────────────────────────────────
if st.session_state.qa_chain is not None:

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    chat_col, viewer_col = st.columns([1, 1])

    # ── LEFT: Chat ────────────────────────────────────────────────────────────
    with chat_col:
        st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;'>
            <div style='width:28px; height:28px; background:linear-gradient(135deg,#06b6d4,#0891b2);
                        border-radius:6px; display:flex; align-items:center;
                        justify-content:center; font-size:0.9rem;'>💬</div>
            <span style='font-family:Syne,sans-serif; font-weight:600; font-size:1.1rem;'>
                Step 2 — Ask questions
            </span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.caption(
                f"🗂 **{st.session_state.pdf_name}** · "
                f"💬 **{st.session_state.question_count}** questions asked"
            )
        with col2:
            if st.button("🗑 Clear"):
                st.session_state.chat_history   = []
                st.session_state.question_count = 0
                st.rerun()
        with col3:
            if st.button("🔄 Reset"):
                st.session_state.qa_chain       = None
                st.session_state.chat_history   = []
                st.session_state.pdf_name       = None
                st.session_state.question_count = 0
                st.session_state.jump_to_page   = 1
                st.session_state.page_contents  = {}
                cache_dir = os.path.join(BASE_DIR, "vectorstore_cache")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                if os.path.exists(STATIC_PDF_PATH):
                    os.remove(STATIC_PDF_PATH)
                st.rerun()

        st.divider()

        # Chat history
        for question, answer, sources in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)

                no_answer_phrases = [
                    "does not contain any information",
                    "does not contain a direct answer",
                    "i don't know based on the provided document",
                    "only has",
                    "does not exist"
                ]
                has_real_answer = not any(
                    phrase in answer.lower() for phrase in no_answer_phrases
                )

                if sources and has_real_answer:
                    with st.expander("📚 Source pages used"):
                        seen = set()
                        for doc in sources:
                            page = doc.metadata.get("page", None)
                            if page is not None and page not in seen:
                                seen.add(page)
                                if st.button(
                                    f"📄 Go to Page {page + 1}",
                                    key=f"page_{page}_{question[:10]}"
                                ):
                                    st.session_state.jump_to_page = page + 1
                                    st.rerun()

        user_question = st.chat_input("Ask anything about your PDF...")

        if user_question:
            with st.spinner("⚡ Thinking..."):
                answer = ask_question(
                    st.session_state.qa_chain,
                    user_question,
                    page_contents=st.session_state.page_contents
                )
                asked_page = extract_page_number_from_question(user_question)
                if asked_page:
                    sources = []
                    total_pages = len(st.session_state.page_contents)
                    if 1 <= asked_page <= total_pages:
                        st.session_state.jump_to_page = asked_page
                else:
                    result = st.session_state.qa_chain.invoke({"query": user_question})
                    sources = result.get("source_documents", [])

            st.session_state.chat_history.append((user_question, answer, sources))
            st.session_state.question_count += 1
            st.rerun()

    # ── RIGHT: PDF Viewer ─────────────────────────────────────────────────────
    with viewer_col:
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;'>
            <div style='width:28px; height:28px; background:linear-gradient(135deg,#f59e0b,#d97706);
                        border-radius:6px; display:flex; align-items:center;
                        justify-content:center; font-size:0.9rem;'>📑</div>
            <span style='font-family:Syne,sans-serif; font-weight:600; font-size:1.1rem;'>
                PDF Viewer — Page {st.session_state.jump_to_page}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if os.path.exists(STATIC_PDF_PATH):
            with open(STATIC_PDF_PATH, "rb") as f:
                pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

            page_num = st.session_state.jump_to_page

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        background: #0a0a0f;
        font-family: 'Space Mono', monospace;
    }}
    #controls {{
        background: linear-gradient(135deg, #12121a, #1a1a2e);
        border-bottom: 1px solid #1e1e2e;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    #controls button {{
        background: linear-gradient(135deg, #7c3aed, #5b21b6);
        color: white;
        border: none;
        padding: 5px 14px;
        cursor: pointer;
        border-radius: 6px;
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        transition: all 0.2s;
        box-shadow: 0 2px 8px rgba(124,58,237,0.4);
    }}
    #controls button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(124,58,237,0.6);
    }}
    #page-info {{
        color: #94a3b8;
        font-size: 12px;
        flex: 1;
        text-align: center;
        letter-spacing: 1px;
    }}
    #page-info span {{
        color: #7c3aed;
        font-weight: bold;
    }}
    #canvas-container {{
        display: flex;
        justify-content: center;
        padding: 16px;
        min-height: 700px;
        background: #0a0a0f;
    }}
    canvas {{
        box-shadow: 0 8px 32px rgba(0,0,0,0.8),
                    0 0 0 1px rgba(124,58,237,0.2);
        border-radius: 4px;
        max-width: 100%;
    }}
    #loading {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 600px;
        color: #64748b;
        font-size: 13px;
        gap: 16px;
    }}
    .spinner {{
        width: 36px; height: 36px;
        border: 3px solid #1e1e2e;
        border-top-color: #7c3aed;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono&display=swap" rel="stylesheet">
</head>
<body>
<div id="controls">
    <button onclick="prevPage()">◀ Prev</button>
    <div id="page-info">Page <span id="current-page">{page_num}</span> of <span id="total-pages">—</span></div>
    <button onclick="nextPage()">Next ▶</button>
</div>
<div id="canvas-container">
    <div id="loading">
        <div class="spinner"></div>
        <div>Loading PDF...</div>
    </div>
    <canvas id="pdf-canvas" style="display:none;"></canvas>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<script>
    pdfjsLib.GlobalWorkerOptions.workerSrc =
        'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

    var pdfDoc = null;
    var currentPage = {page_num};

    function renderPage(num) {{
        pdfDoc.getPage(num).then(function(page) {{
            var viewport = page.getViewport({{ scale: 1.4 }});
            var canvas = document.getElementById('pdf-canvas');
            var ctx = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;
            document.getElementById('loading').style.display = 'none';
            canvas.style.display = 'block';
            page.render({{ canvasContext: ctx, viewport: viewport }});
            document.getElementById('current-page').textContent = num;
            document.getElementById('total-pages').textContent = pdfDoc.numPages;
        }});
    }}

    function prevPage() {{
        if (currentPage <= 1) return;
        currentPage--;
        renderPage(currentPage);
    }}

    function nextPage() {{
        if (currentPage >= pdfDoc.numPages) return;
        currentPage++;
        renderPage(currentPage);
    }}

    var base64 = "{pdf_base64}";
    var binary = atob(base64);
    var pdfArray = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) {{
        pdfArray[i] = binary.charCodeAt(i);
    }}

    pdfjsLib.getDocument({{ data: pdfArray }}).promise.then(function(pdf) {{
        pdfDoc = pdf;
        renderPage(currentPage);
    }}).catch(function(err) {{
        document.getElementById('loading').innerHTML =
            '<div style="color:#ef4444;">Error loading PDF: ' + err.message + '</div>';
    }});
</script>
</body>
</html>
"""
            components.html(html_content, height=820, scrolling=True)
        else:
            st.info("PDF viewer will appear here after upload.")