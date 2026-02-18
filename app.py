import streamlit as st
import os
import json
import requests
import tempfile
from dotenv import load_dotenv
from fpdf import FPDF
from typing import List, Dict, Tuple, Any, Optional

# LangChain & AI Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

# 1. ENVIRONMENT SETUP
load_dotenv()
API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
HISTORY_FILE: str = "chat_history.json"

st.set_page_config(page_title="Multi-Doc AI Resilience", layout="centered", page_icon="🛡️")

# 2. UI STYLING
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; border: 1px solid #e0e0e0; }
    .source-box { font-size: 0.8rem; color: #444; background: #f9f9f9; padding: 10px; border-radius: 8px; border-left: 5px solid #28a745; margin-top: 5px; }
    .stProgress > div > div > div > div { background-color: #28a745; }
    </style>
    """, unsafe_allow_html=True)

# 3. UTILITIES & SECURITY
def check_ollama() -> bool:
    """Checks if the local Ollama instance is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def save_history() -> None:
    """Persists chat history to a JSON file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=4)

def load_history() -> List[Dict[str, Any]]:
    """Loads chat history from JSON if available."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError): 
            return []
    return []

def generate_pdf_report(messages: List[Dict[str, Any]]) -> bytes:
    """Generates a PDF report from the chat history."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Conversation Report", ln=True, align="C")
    pdf.ln(10)
    
    for msg in messages:
        role = "USER" if msg.get("role") == "user" else "AI"
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"{role}:", ln=True)
        pdf.set_font("Arial", "", 10)
        # Safe encoding to handle special characters
        content = str(msg.get("content", "")).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, content)
        pdf.ln(4)
        
    return bytes(pdf.output())

# 4. CASCADING FALLBACK SYSTEM
def ask_ai_cascading(prompt: str, primary_model: str) -> Tuple[str, str]:
    """
    Orchestrates the fallback logic between Cloud APIs and Local Models.
    Returns a tuple: (Answer Content, Model Name Used)
    """
    models: List[str] = []
    if primary_model == "gemini-2.5-pro":
        models = ["gemini-2.5-pro", "gemini-2.5-flash", "ollama"]
    else:
        models = ["gemini-2.5-flash", "ollama"]

    for model_name in models:
        try:
            if model_name == "ollama":
                # Local Fallback using SLM (Small Language Model)
                llm = Ollama(model="llama3.2:1b")
                res = llm.invoke(prompt)
                return str(res), "Ollama (Local 🏠 - Llama 3.2 1B)"
            else:
                # Cloud API Call
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=API_KEY)
                res = llm.invoke(prompt)
                return str(res.content), model_name
                
        except Exception as e:
            # Detect Rate Limit Errors (429)
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                st.warning(f"⚠️ {model_name} quota exhausted. Scaling down infrastructure...")
                continue
            # Raise other unexpected errors
            raise e
            
    return "Critical Error: All models failed to respond.", "None"

# 5. SESSION STATE MANAGEMENT
if "messages" not in st.session_state: st.session_state.messages = load_history()
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "processed_files" not in st.session_state: st.session_state.processed_files = []
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

def reset_all() -> None:
    """Resets the application state and clears history."""
    if os.path.exists(HISTORY_FILE): 
        os.remove(HISTORY_FILE)
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()

# --- INITIAL SYSTEM CHECK ---
if not check_ollama():
    st.error("❌ Ollama service not detected. Please start the local inference server.")
    st.stop()

st.title("🛡️ Multi-Doc Insight AI")

# 6. SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox("Preferred Model:", ["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
    st.divider()
    
    uploaded_files = st.file_uploader(
        "Upload PDFs (Ctrl+Click)", 
        type="pdf", 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    if st.session_state.messages:
        pdf_bytes = generate_pdf_report(st.session_state.messages)
        st.download_button(
            label="📥 Export PDF Report", 
            data=pdf_bytes, 
            file_name="chat_summary.pdf", 
            mime="application/pdf", 
            use_container_width=True
        )

    if st.button("🗑️ Reset All", use_container_width=True, type="primary"):
        reset_all()

# 7. DOCUMENT PROCESSING
def process_docs(files: List[Any]) -> VectorStore:
    """
    Ingests raw PDF files, splits content, and indexes into ChromaDB.
    Optimized for low-memory environments (Intel i3 / 8GB RAM).
    """
    all_splits: List[Document] = []
    prog_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(files)
    
    for i, file in enumerate(files):
        status_text.caption(f"Processing: {file.name}")
        
        # Create temporary file to read PDF stream
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Add metadata for citation
        for d in docs: 
            d.metadata["source_file"] = file.name
        
        # Semantic Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, 
            chunk_overlap=30, 
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = splitter.split_documents(docs)
        
        # Memory Safety Filter: discard overly large chunks
        safe_splits = [s for s in splits if len(s.page_content) < 1200]
        all_splits.extend(safe_splits)
        
        # Cleanup temp file
        os.remove(tmp_path)
        prog_bar.progress((i + 1) / total_files)

    status_text.caption("Generating local embeddings (Ollama)...")
    
    # Vectorization
    emb = OllamaEmbeddings(model="all-minilm")
    vs = Chroma.from_documents(
        documents=all_splits, 
        embedding=emb, 
        collection_name=f"v_{st.session_state.uploader_key}"
    )
    
    prog_bar.empty()
    status_text.empty()
    return vs

# 8. AUTO-ANALYSIS LOGIC
current_files: List[str] = [f.name for f in uploaded_files] if uploaded_files else []

if uploaded_files and current_files != st.session_state.processed_files:
    try:
        st.session_state.vectorstore = process_docs(uploaded_files)
        st.session_state.processed_files = current_files
        st.success("✅ Documents indexed and ready for retrieval!")
    except Exception as e:
        st.error(f"Indexing Error: {e}")

# 9. CHAT INTERFACE
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "actual_model" in m:
            st.caption(f"Model used: {m['actual_model']}")
        if "src" in m:
            with st.expander("Sources"):
                for s in m["src"]:
                    st.markdown(f"<div class='source-box'><b>File:</b> {s['f']} | <b>Page:</b> {s['p']}<br>{s['t']}...</div>", unsafe_allow_html=True)

if query := st.chat_input("Ask your documents..."):
    if not st.session_state.vectorstore:
        st.warning("⚠️ Please upload documents first!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        save_history()
        with st.chat_message("user"): st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context & Generating response..."):
                try:
                    # 1. Retrieval (RAG)
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.invoke(query)
                    
                    # 2. Context Formatting
                    ctx = "\n".join([f"[{d.metadata.get('source_file')}, p.{d.metadata.get('page', 0)+1}]: {d.page_content}" for d in docs])
                    src = [{"f": d.metadata.get('source_file'), "p": d.metadata.get('page', 0)+1, "t": d.page_content[:150]} for d in docs]
                    
                    # 3. Cascading Generation
                    prompt_text = f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer strictly based on the provided context."
                    answer, used_model = ask_ai_cascading(prompt_text, selected_model)
                    
                    # 4. Output & Persist
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "actual_model": used_model,
                        "src": src
                    })
                    save_history()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Runtime Error: {e}")