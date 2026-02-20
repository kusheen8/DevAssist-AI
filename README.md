# 🚀 DevAssist AI — Developer Documentation Assistant

🔗 **Live Demo:** https://devassist-ai.streamlit.app/

DevAssist AI is a **Retrieval-Augmented Generation (RAG)** based developer documentation assistant that allows users to upload technical PDFs (PRDs, specs, API docs, assignments) and ask grounded questions.
The system retrieves relevant content from the uploaded document and generates accurate answers using an LLM — while preventing hallucinations through strict prompt guardrails.

---

## ✨ Features

* 📂 Upload developer documentation PDFs
* 🔎 Semantic search using embeddings + vector database
* 🤖 Context-aware AI answers powered by Llama 3.1 (Groq)
* 🧠 Guardrails to avoid hallucinations
* 💬 Chat-style interface with history
* 📚 Developer Mode to view source pages
* ⚡ Fast deployment with Streamlit Cloud

---

## 🧠 Architecture Overview

DevAssist AI follows a **RAG (Retrieval Augmented Generation)** pipeline:

```
Upload PDF
   ↓
Text Extraction (PyPDFLoader)
   ↓
Chunking (RecursiveCharacterTextSplitter)
   ↓
Embeddings (MiniLM)
   ↓
Chroma Vector Database
   ↓
Retriever
   ↓
Groq LLM (Llama 3.1)
   ↓
Grounded Answer
```

The model only answers from retrieved document context, ensuring reliable and controlled responses.

---

## 🛠️ Tech Stack

### Frontend / Interface

* Streamlit

### Backend / AI Pipeline

* LangChain
* Groq (Llama-3.1-8B-Instant)
* HuggingFace Embeddings (all-MiniLM-L6-v2)

### Vector Database

* ChromaDB

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

---

## 🔐 Guardrails & Safety

* Answers are restricted to uploaded document context
* Non-developer or unrelated queries are blocked
* Temperature set to `0` for deterministic responses
* Custom prompt template prevents hallucinated outputs

---

## 📦 Installation (Local Setup)

Clone the repository:

```bash
git clone https://github.com/your-username/devassist-ai.git
cd devassist-ai
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your API key in `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_api_key_here"
```

Run the app:

```bash
streamlit run app.py
```

---

## 🚀 Deployment

The project is deployed using:

* **Streamlit Cloud** for hosting
* Groq API for LLM inference
* Local persistent ChromaDB for embeddings storage

👉 Live App: https://devassist-ai.streamlit.app/

---

## 🎯 Use Cases & Example Questions

DevAssist AI helps developers quickly understand technical documentation.
Some example questions you can ask after uploading a PDF:

* “What is the project overview?”
* “Explain the frontend architecture.”
* “What backend technologies are used?”
* “Describe the system workflow.”
* “What APIs are implemented?”
* “List the main features mentioned in the document.”
* “Explain the deployment strategy.”
* “What database is used?”
* “Summarize the assignment requirements.”

---

## 👩‍💻 Author

**Kusheen Dhar**
CS Engineering Student | Full Stack & AI Developer

---

## ⭐ Acknowledgements

* LangChain
* HuggingFace
* Groq
* Streamlit
* ChromaDB

---

> ⚙️ Built as a placement-ready AI project demonstrating RAG architecture, LLM integration, and full-stack deployment.
