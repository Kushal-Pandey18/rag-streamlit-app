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
            extracted = page.extract_text()
            if extracted:
                text += extracted
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


# ---------- HF GENERATION (NO LANGCHAIN LLM) ----------
def generate_answer(context, question):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {
        "Authorization": f"Bearer {st.secrets['HUGGINGFACEHUB_API_TOKEN']}"
    }

    prompt = f"""
Use the following context to answer the question.
If the answer is not present, say "I don't know".

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

    return response.json()[0]["generated_text"]


# ---------- MAIN ----------
def main():
    st.set_page_config(page_title="PDF Chat (FREE)", page_icon="📚")
    st.title("📚 PDF Chat App (FREE)")
    st.write("Upload PDFs and ask questions")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    user_question = st.text_input("Ask a question from your PDFs")

    if user_question and st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(user_question, k=3)
        context = "\n".join([d.page_content for d in docs])

        with st.spinner("Generating answer..."):
            answer = generate_answer(context, user_question)

        st.markdown("### ✅ Answer")
        st.write(answer)

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdfs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not pdfs:
                st.warning("Upload at least one PDF")
                return

            with st.spinner("Processing PDFs..."):
                raw_text = get_text_from_pdf(pdfs)
                chunks = chunk_text(raw_text)
                st.session_state.ve_
