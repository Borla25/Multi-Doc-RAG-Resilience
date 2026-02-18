# 🛡️ Multi-Doc AI: Resilience RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** application designed to query multiple PDF documents simultaneously. This system features a hybrid architecture to balance privacy, performance, and cloud-computing costs, engineered with **strict type hinting** and **fail-safe mechanisms**.

## 🏗️ Architecture Flow

```mermaid
graph TD
    User[User] -->|Uploads PDF| UI[Streamlit UI]
    UI -->|Extract Text| PDF[PyPDF Loader]
    PDF -->|Chunking| Splitter[Recursive Splitter]
    Splitter -->|Embeddings| OllamaEmb[Ollama Local]
    OllamaEmb -->|Store Vectors| Chroma[ChromaDB]
    
    User -->|Asks Question| UI
    UI -->|Retrieve Context| Chroma
    
    UI -->|Prompt + Context| Logic{Quota Check}
    Logic -->|Quota OK| Cloud[Gemini 2.5 Pro/Flash]
    Logic -->|Quota Exceeded 429| Fallback[Llama 3.2 Local]
    
    Cloud -->|Response| UI
    Fallback -->|Response| UI
🚀 Key Features
Hybrid Intelligence
Orchestrates local embeddings via Ollama (all-minilm) and cloud reasoning via Google Gemini 2.5.

Resilience Logic (Auto-Scaling)
Implements a Cascading Fallback System. If cloud API quotas are exhausted (Error 429), the system automatically switches to a local SLM (Llama 3.2 1B) without interrupting the user session.

Resource Optimized
Aggressive text chunking and filtering designed for consumer-grade hardware (tested on Intel i3, 8GB RAM).

Engineering Standards
Codebase utilizes Strict Type Hinting (PEP 484) and comprehensive docstrings for maintainability and scalability.

Session Persistence
Auto-saves chat history to JSON and supports PDF report export.

🛠️ Tech Stack
Frontend: Streamlit

AI Orchestration: LangChain

Vector Database: ChromaDB

Local Embedding/Inference: Ollama

LLM API: Google Generative AI (Gemini)

Quality Assurance: Type Hinting (typing), Error Handling

📋 Installation
1. Clone the repository
Bash
git clone [https://github.com/Borla25/Multi-Doc-RAG-Resilience.git](https://github.com/Borla25/Multi-Doc-RAG-Resilience.git)
cd Multi-Doc-RAG-Resilience
2. Install dependencies
Bash
pip install -r requirements.txt
3. Setup Environment
Create a .env file in the root directory and add your API key:

Plaintext
GOOGLE_API_KEY=your_api_key_here
4. Prepare Local Models (Ollama)
Ensure Ollama is installed and running, then pull the required models:

Bash
ollama pull all-minilm    # For Embeddings
ollama pull llama3.2:1b   # For Local Fallback
5. Run the Application
Bash
streamlit run app.py
🧠 Technical Challenges Solved
API Rate Limiting
Implemented a hierarchy of models to ensure 100% service uptime.

Memory Constraints
Optimized chunk sizes to prevent OOM (Out of Memory) errors on low-RAM systems during vectorization.

Data Privacy
Localized vector indexing keeps document structure off-cloud.

Built with ❤️ by Borla25
