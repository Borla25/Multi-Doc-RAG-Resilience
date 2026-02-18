# 📚 Multi-Doc AI: Resilience RAG System

A sophisticated **Retrieval-Augmented Generation (RAG)** application designed to query multiple PDF documents simultaneously. This system features a hybrid architecture to balance privacy, performance, and cloud-computing costs.

## 🚀 Key Features
- **Hybrid Architecture:** Local embeddings via **Ollama** (`all-minilm`) and high-level reasoning via **Google Gemini 2.5** (Pro/Flash).
- **Multi-Document Ingestion:** Simultaneous upload and analysis of multiple PDFs with precise metadata tracking (file name and page number).
- **Resilience Logic (Auto-Scaling):** Automatically falls back to local models (Ollama/Llama 3.2) if cloud API quotas are exhausted (Error 429).
- **Resource Optimized:** Aggressive text chunking and filtering designed for consumer-grade hardware (tested on Intel i3, 8GB RAM).
- **Session Persistence:** Auto-saves chat history to JSON for continuous workflows.
- **Professional Export:** Generates downloadable PDF reports of the conversation.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **AI Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Local Embedding:** Ollama
- **LLM API:** Google Generative AI (Gemini)
- **Document Processing:** PyPDF & RecursiveCharacterTextSplitter

## 📋 Installation
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your `.env` file with your `GOOGLE_API_KEY`.
4. Ensure **Ollama** is installed and running locally.
5. Pull the local fallback model: `ollama pull llama3.2:1b`

## 🧠 Technical Challenges Solved
- **API Rate Limiting:** Implemented a cascading fallback mechanism to ensure 100% uptime.
- **Memory Management:** Optimized chunk sizes to prevent 500 errors on low-RAM systems.
- **Data Privacy:** Localized vector indexing to keep document structure off-cloud.