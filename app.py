import streamlit as st
import os
import json
import requests
import tempfile
from dotenv import load_dotenv
from fpdf import FPDF

# LangChain & AI
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. ENVIRONMENT SETUP
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
HISTORY_FILE = "chat_history.json"

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
def check_ollama():
    try:
        return requests.get("http://localhost:11434", timeout=2).status_code == 200
    except:
        return False

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=4)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return []
    return []

def generate_pdf_report(messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Conversation Report", ln=True, align="C")
    pdf.ln(10)
    for msg in messages:
        role = "USER" if msg["role"] == "user" else "AI"
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"{role}:", ln=True)
        pdf.set_font("Arial", "", 10)
        content = msg["content"].encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 6, content)
        pdf.ln(4)
    return pdf.output()

# 4. CASCADING FALLBACK SYSTEM
def ask_ai_cascading(prompt, primary_model):
    """Attempt responses through a hierarchy if quotas fail"""
    if primary_model == "gemini-2.5-pro":
        models = ["gemini-2.5-pro", "gemini-2.5-flash", "ollama"]
    else:
        models = ["gemini-2.5-flash", "ollama"]

    for model_name in models:
        try:
            if model_name == "ollama":
                # Using 1B model for low-RAM compatibility
                llm = Ollama(model="llama3.2:1b")
                res = llm.invoke(prompt)
                return res, "Ollama (Local 🏠 - Llama 3.2 1B)"
            else:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=API_KEY)
                res = llm.invoke(prompt)
                return res.content, model_name
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.warning(f"⚠️ {model_name} quota exhausted. Scaling down...")
                continue
            raise e
    return "Critical Error: No models available.", "None"

# 5. SESSION STATE MANAGEMENT
if "messages" not in st.session_state: st.session_state.messages = load_history()
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "processed_files" not in st.session_state: st.session_state.processed_files = []
if "uploader_key" not in st.session_state: st.session_state.uploader_key = 0

def reset_all():
    if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
    st.session_state.clear()
    st.cache_resource.clear()
    st.rerun()

# --- INITIAL SYSTEM CHECK ---
if not check_ollama():
    st.error("❌ Ollama not detected. Please start Ollama to use local embeddings and fallback.")
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
        st.download_button("📥 Export PDF Report", data=bytes(pdf_bytes), file_name="chat_summary.pdf", use_container_width=True)

    if st.button("🗑️ Reset All", use_container_width=True, type="primary"):
        reset_all()

# 7. DOCUMENT PROCESSING

def process_docs(files):
    all_splits = []
    prog_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        status_text.caption(f"Processing: {file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for d in docs: d.metadata["source_file"] = file.name
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=30, separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = splitter.split_documents(docs)
        # Safety filter for low-RAM stability
        safe_splits = [s for s in splits if len(s.page_content) < 1200]
        all_splits.extend(safe_splits)
        os.remove(tmp_path)
        prog_bar.progress((i+1)/len(files))

    status_text.caption("Generating vector database...")
    emb = OllamaEmbeddings(model="all-minilm")
    vs = Chroma.from_documents(all_splits, emb, collection_name=f"v_{st.session_state.uploader_key}")
    prog_bar.empty()
    status_text.empty()
    return vs

# 8. AUTO-ANALYSIS LOGIC
current_files = [f.name for f in uploaded_files] if uploaded_files else []
if uploaded_files and current_files != st.session_state.processed_files:
    try:
        st.session_state.vectorstore = process_docs(uploaded_files)
        st.session_state.processed_files = current_files
        st.success("✅ Documents ready for querying!")
    except Exception as e:
        st.error(f"Analysis error: {e}")

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
        st.warning("Please upload documents first!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        save_history()
        with st.chat_message("user"): st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning..."):
                try:
                    # Retrieval
                    docs = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(query)
                    ctx = "\n".join([f"[{d.metadata['source_file']}, p.{d.metadata['page']+1}]: {d.page_content}" for d in docs])
                    src = [{"f": d.metadata['source_file'], "p": d.metadata['page']+1, "t": d.page_content[:150]} for d in docs]
                    
                    # Cascading Generation
                    prompt = f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer strictly based on context."
                    answer, used_model = ask_ai_cascading(prompt, selected_model)
                    
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
                    st.error(f"Error: {e}")