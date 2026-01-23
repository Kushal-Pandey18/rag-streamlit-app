import streamlit as st
from pypdf import PdfReader
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- PDF TEXT ----------
def get_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


# ---------- CHUNKING ----------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# ---------- VECTOR STORE ----------
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


# ---------- HF REST CALL (NO LANGCHAIN LLM) ----------
def generate_answer(context, question):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {
        "Authorization": f"Bearer {st.secrets['HUGGINGFACEHUB_API_TOKEN']}"
    }

    prompt = f"""
Answer the question using the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": prompt},
        timeout=60
    )

    data = response.json()

    # ✅ SAFE HANDLING
    if isinstance(data, list):
        return data[0].get("generated_text", "No answer generated.")

    if isinstance(data, dict):
        if "error" in data:
            return f"⚠️ Model loading / busy. Try again in 30 seconds.\n\n{data['error']}"
        return data.get("generated_text", "No answer generated.")

    return "Unexpected response from model."



if __name__ == "__main__":
    main()

