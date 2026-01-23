# 📚 PDF Chat App (RAG + LLM)

A web-based application that allows users to upload PDF documents and ask questions using natural language.  
This project is built using the **Retrieval-Augmented Generation (RAG)** approach with embeddings, vector search, and a Large Language Model (LLM).

🔗 **Live Deployment:**  
https://rag-app-app-mg8aeuyx9wakshr8kr5o9v.streamlit.app/

---

## 🧠 Project Overview

This application enables intelligent question-answering over PDF documents.  
Instead of traditional keyword search, it performs **semantic search** and uses an **LLM** to generate meaningful answers grounded in document content.

---

## ✨ Features

- Upload one or multiple PDF files
- Extracts readable text from PDFs
- Splits text into semantic chunks
- Generates embeddings using Sentence Transformers
- Stores embeddings in FAISS vector database
- Retrieves relevant chunks using similarity search
- Uses an LLM to generate final answers
- Simple, clean Streamlit UI
- Fully deployed on Streamlit Cloud

---

## 🚀 Live Demo

Try the application here:  
👉 https://rag-app-app-mg8aeuyx9wakshr8kr5o9v.streamlit.app/

Example questions:
- *Explain the skills mentioned in the document*
- *Summarize this PDF*
- *What is the document about?*

---

## 🛠️ Tech Stack

| Layer | Technology |
|-----|-----------|
| Frontend | Streamlit |
| PDF Parsing | PyPDF |
| Text Chunking | LangChain Text Splitters |
| Embeddings | sentence-transformers |
| Vector Store | FAISS |
| LLM | HuggingFace Inference API (FLAN-T5) |
| Language | Python |

---

## 🧩 System Architecture (RAG Pipeline)

PDF Upload
↓
Text Extraction
↓
Text Chunking
↓
Embedding Generation
↓
FAISS Vector Store
↓
Relevant Context Retrieval
↓
LLM-based Answer Generation



## 📋 Assessment Alignment

This project satisfies the following assessment requirements:

- ✅ Uses a **Large Language Model (LLM)**
- ✅ Implements **Retrieval-Augmented Generation (RAG)**
- ✅ Embedding-based semantic search
- ✅ Vector database integration (FAISS)
- ✅ End-to-end working AI application
- ✅ Cloud deployment with public access

---

## 💻 Run Locally (Optional)

```bash
git clone <your-repository-link>
cd rag-streamlit-app
pip install -r requirements.txt
streamlit run app.py
🔐 Environment Variables
The application requires a HuggingFace API token.

Streamlit Secrets
toml
Copy code
HUGGINGFACEHUB_API_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxx"
⚠️ Never commit API keys directly to GitHub.

🔮 Future Enhancements
Conversational chat history

OCR support for scanned PDFs

Improved answer formatting

Multi-language PDF support

Dedicated LLM endpoint for faster responses

👤 Author
Kushal Pandey

📄 License
This project is created for educational and assessment purposes.
