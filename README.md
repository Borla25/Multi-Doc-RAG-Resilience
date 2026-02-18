# 📚 Multi-Doc Insight AI: Hybrid RAG System

Un'applicazione avanzata di **Retrieval-Augmented Generation (RAG)** che permette di interrogare più documenti PDF contemporaneamente. Il sistema utilizza un'architettura ibrida per bilanciare privacy, prestazioni locali e potenza di calcolo in cloud.

## 🚀 Caratteristiche Principali
- **Architettura Ibrida:** Embeddings generati localmente via **Ollama** (`all-minilm`) e ragionamento affidato a **Google Gemini 2.5** (Pro/Flash) tramite API.
- **Multi-Document Ingestion:** Caricamento e analisi simultanea di più file PDF con tracking dei metadati (nome file e pagina).
- **Resource Optimization:** Algoritmo di chunking ottimizzato per hardware con risorse limitate (testato su i3, 8GB RAM).
- **Persistenza dei Dati:** Salvataggio automatico della cronologia chat in formato JSON per sessioni di lavoro continue.
- **Export Professionale:** Generazione di report della conversazione in formato PDF.
- **UI/UX Avanzata:** Barra di avanzamento dinamica durante l'indicizzazione e gestione dei limiti di quota (Rate Limiting).

## 🛠️ Stack Tecnologico
- **Frontend:** Streamlit
- **Orchestrazione AI:** LangChain
- **Vector Database:** ChromaDB
- **Embedding Locale:** Ollama
- **LLM:** Google Generative AI (Gemini)
- **Document Processing:** PyPDF & RecursiveCharacterTextSplitter

## 📋 Installazione
1. Clonare il repository.
2. Installare le dipendenze: `pip install -r requirements.txt`
3. Configurare il file `.env` con la propria `GOOGLE_API_KEY`.
4. Assicurarsi che Ollama sia installato e in esecuzione localmente.

## 🧠 Sfide Tecniche Risolte
- Gestione dei crash di memoria tramite filtri di sicurezza sui chunk di testo.
- Implementazione di logiche di fallback manuale per la gestione delle quote API gratuite.
- Ottimizzazione del contesto per minimizzare la latenza su processori dual-core.