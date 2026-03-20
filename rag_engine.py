import warnings
warnings.filterwarnings("ignore")

import os
import re
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

# Force load .env from the exact folder where rag_engine.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Immediately read and validate the key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your .env file in the rag_chatbot folder.")

# Folder where vectorstore is saved on disk
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore_cache")

# Tesseract and Poppler paths for Windows
import platform
if platform.system() == "Windows":
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH   = r"C:\Users\AMARSATISH\poppler\poppler-25.12.0\Library\bin"
else:
    # Linux — Docker / Railway server
    TESSERACT_PATH = "/usr/bin/tesseract"
    POPPLER_PATH   = "/usr/bin"
# Global variable to store total page count after PDF is loaded
TOTAL_PAGES = 0


def extract_text_with_ocr(pdf_path: str):
    """
    Extracts text from PDFs that are image-based or mixed (text + images).
    Uses pdf2image to convert pages to images, then pytesseract for OCR.
    Returns a list of Document objects with page content and metadata.
    """
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    # Convert all PDF pages to images
    pages = convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH)

    documents = []
    for i, page_image in enumerate(pages):
        # Run OCR on each page image
        text = pytesseract.image_to_string(page_image)
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"page": i, "source": pdf_path}
            ))
        else:
            # Even blank pages get a placeholder so page count stays correct
            documents.append(Document(
                page_content="",
                metadata={"page": i, "source": pdf_path}
            ))

    return documents


def load_and_split_pdf(pdf_path: str):
    """
    Step 1 of RAG pipeline.
    Tries normal text extraction first.
    If text is too little (image-based PDF), falls back to OCR.
    For mixed PDFs — merges text extraction + OCR per page.
    """
    global TOTAL_PAGES

    # ── Try normal text extraction first ─────────────────────────────────────
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    TOTAL_PAGES = len(documents)

    total_text = " ".join([doc.page_content for doc in documents]).strip()

    # If less than 100 characters total — likely image-based, use full OCR
    if len(total_text) < 100:
        print("Low text detected — switching to full OCR mode.")
        documents = extract_text_with_ocr(pdf_path)
        TOTAL_PAGES = len(documents)

    else:
        # Mixed PDF — some pages may be images
        # For pages with very little text, supplement with OCR
        try:
            import pytesseract
            from pdf2image import convert_from_path
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

            ocr_pages = convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH)

            for i, doc in enumerate(documents):
                if len(doc.page_content.strip()) < 50 and i < len(ocr_pages):
                    # This page has almost no text — run OCR on it
                    ocr_text = pytesseract.image_to_string(ocr_pages[i])
                    if ocr_text.strip():
                        documents[i] = Document(
                            page_content=ocr_text,
                            metadata=doc.metadata
                        )
        except Exception:
            # If OCR fails for any reason, continue with what we have
            pass

    # Filter out empty pages for chunking
    non_empty_docs = [doc for doc in documents if doc.page_content.strip()]

    if not non_empty_docs:
        raise ValueError(
            "Could not extract any content from this PDF. "
            "Please ensure it is a readable PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(non_empty_docs)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise ValueError("Could not extract any content from this PDF. Please try a different file.")

    return chunks


def build_vectorstore(chunks):
    """
    Step 2 of RAG pipeline.
    Convert chunks into vector embeddings and store in FAISS.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore):
    """
    Save the FAISS vectorstore to disk so it survives page refreshes.
    """
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)


def load_vectorstore_from_disk():
    """
    Load the FAISS vectorstore from disk if it exists.
    Returns vectorstore or None if not found.
    """
    index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")
    if not os.path.exists(index_file):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def build_qa_chain(vectorstore):
    """
    Step 3 of RAG pipeline.
    Build the QA chain connecting FAISS retriever + Groq LLM.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=GROQ_API_KEY
    )

    prompt_template = """
You are a helpful assistant that answers questions strictly based on the provided document context.

Follow these rules:
1. If the question is asking for a summary, overview, or description of the document
   (examples: "tell about pdf", "what is this pdf about", "what does this pdf tell",
   "summarize", "give overview", "describe the document"),
   then summarize the key points from the context in a clear and concise way.

2. If the question is specific and the exact answer exists in the context, answer it directly.

3. If the question is specific but the exact answer is not in the context,
   share the most related information available from the context that might help,
   and end with: "Note: The document does not contain a direct answer to this question."

4. If the context has absolutely nothing related to the question, respond with:
   "This document does not contain any information about that topic."

5. Never make up information. Never answer from outside the context.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def process_pdf(pdf_path: str):
    """
    Master function that runs the full pipeline.
    """
    chunks = load_and_split_pdf(pdf_path)
    vectorstore = build_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore)
    return qa_chain


def extract_page_number_from_question(question: str):
    """
    Detects if user is asking about a specific page number.
    Returns the page number (int) if found, else None.
    """
    match = re.search(r'page\s*(?:no\.?|number)?\s*(\d+)', question.lower())
    if match:
        return int(match.group(1))
    return None


def ask_question(qa_chain, question: str, page_contents: dict = None):
    """
    Handles page-specific questions separately before hitting the LLM.
    """
    global TOTAL_PAGES

    asked_page = extract_page_number_from_question(question)

    if asked_page is not None:
        total = TOTAL_PAGES if TOTAL_PAGES > 0 else len(page_contents) if page_contents else 0

        if asked_page > total:
            return f"This PDF only has {total} page(s). Page {asked_page} does not exist."

        if page_contents and asked_page in page_contents:
            page_text = page_contents[asked_page].strip()
            if not page_text:
                return f"Page {asked_page} exists but has no readable text content."

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                api_key=GROQ_API_KEY
            )
            focused_prompt = f"""
The user wants to know what is on page {asked_page} of a document.
Here is the exact content of page {asked_page}:

{page_text}

Summarize what this page contains in a clear and helpful way.
"""
            response = llm.invoke(focused_prompt)
            return response.content

    result = qa_chain.invoke({"query": question})
    return result["result"]