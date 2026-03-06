# RAG Based Project

A Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and Ollama (Llama 3.2 LLM).

---

## Architecture

```
User Query
    ↓
FastAPI (/ask)
    ↓
Deterministic Guardrail   # Banned keyword check
    ↓
ChromaDB Retriever        # Fetch top-k similar documents
    ↓
Llama 3.2 (Ollama)        # Generate answer from context
    ↓
Inline Model Guardrail    # Safety check on answer
    ↓
Response
```

---

## Project Structure

```
RAG_BASED_PROJECT/
│
├── data/                             # All raw input documents
│   ├── faqs/
│   │   └── faqs.txt                  # Frequently asked questions (text format)
│   ├── html/
│   │   └── about.html                # Company/product info in HTML format
│   ├── pdfs/
│   │   └── TechCorp_Sample_Data.pdf  # Sample business data in PDF format
│   └── vector_store/                 # ChromaDB embeddings storage (auto-generated)
│
├── logs/
│   └── app.log                       # All application logs saved here (auto-generated)
│
├── src/                              # Core business logic
│   ├── config.py                     # Centralized app settings (chunk size, model, paths)
│   ├── logger.py                     # Logger setup with console + file handlers
│   ├── loaders.py                    # Load PDF, HTML, TXT documents from data folder
│   ├── chunker.py                    # Split documents into smaller chunks
│   ├── embedding.py                  # Generate embeddings using SentenceTransformer
│   ├── vectorstore.py                # Store and manage embeddings in ChromaDB
│   ├── retriever.py                  # Search similar documents from ChromaDB
│   ├── rag_pipeline.py               # Core RAG logic - retrieval + LLM answer generation
│   ├── guardrails.py                 # Safety checks on query and LLM answer
│   └── data_ingestion.py             # Orchestrates full pipeline (load → chunk → embed → store)
│
├── app.py                            # FastAPI entry point - defines API endpoints
├── requirements.txt                  # All Python dependencies
└── README.md                         # Project documentation
```

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Llama 3.2 model pulled in Ollama

```bash
ollama pull llama3.2
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/JaspinderKaurWalia26/projects.git
cd projects
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Ingestion

Documents are already placed in the `data/` folder:
- FAQs → `data/faqs/faqs.txt`
- HTML → `data/html/about.html`
- PDFs → `data/pdfs/TechCorp_Sample_Data.pdf`

To add more documents, place them in the relevant folder and run:

```bash
python -m src.data_ingestion
```

This will:
1. Load all documents from `data/`
2. Chunk them into smaller pieces
3. Generate embeddings using `all-MiniLM-L6-v2`
4. Store in ChromaDB at `data/vector_store/`

---

## Run the API

```bash
uvicorn app:app --reload
```

API will be available at: `http://localhost:8000`

Swagger docs at: `http://localhost:8000/docs`

---

## API Endpoints

### Health Check
```
GET /health
```
```json
{"status": "ok"}
```

### Ask a Question
```
POST /ask
```
```json
// Request
{
    "question": "What is your refund policy?"
}

// Response
{
    "answer": "Our refund policy allows returns within 30 days..."
}
```

---

## Guardrails

The system has 2 safety layers:

**1. Deterministic Check** — Runs before query reaches LLM:
- Blocks banned keywords (hack, exploit, malware, etc.)
- Rejects empty queries

**2. Inline Model Guardrail** — Runs after LLM generates answer:
- Checks for PII leaks
- Checks for confidential information
- Checks for harmful or inappropriate content
- Returns safe fallback if unsafe
---
