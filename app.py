import streamlit as st
import requests
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- PDF TEXT ----------
def get_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text.strip()


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


# ---------- LLM (HF REST API) ----------
def generate_answer(context, question):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {
        "Authorization": f"Bearer {st.secrets['HUGGINGFACEHUB_API_TOKEN']}"
    }

    prompt = f"""
Use the context below to answer the question.
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

    data = response.json()

    if isinstance(data, list):
        return data[0].get("generated_text", "No answer generated.")

    if isinstance(data, dict) and "error" in data:
        return "⚠️ Model is loading or busy. Please try again."

    return "Unexpected response from LLM."


# ---------- MAIN ----------
def main():
    st.set_page_config(page_title="PDF Chat App (RAG + LLM)", page_icon="📚")
    st.title("📚 PDF Chat App (RAG + LLM)")
    st.write("Upload PDFs and ask questions using an LLM-powered RAG system.")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    with st.sidebar:
        st.subheader("Upload PDFs")
        docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing PDFs..."):
                raw_text = get_text_from_pdf(docs)

                if not raw_text:
                    st.error("No readable text found.")
                    return

                chunks = chunk_text(raw_text)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.success("PDFs processed successfully!")

    question = st.text_input("Ask a question from your PDFs")

    if question and st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        with st.spinner("Generating answer using LLM..."):
            answer = generate_answer(context, question)

        st.markdown("### ✅ Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
