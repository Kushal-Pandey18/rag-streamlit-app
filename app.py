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


# ---------- HF REST CALL ----------
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

    if isinstance(data, list):
        return data[0].get("generated_text", "No answer generated.")

    if isinstance(data, dict):
        if "error" in data:
            return "⚠️ Model is loading or busy. Please try again in 30 seconds."
        return data.get("generated_text", "No answer generated.")

    return "Unexpected response from model."


# ---------- MAIN APP ----------
def main():
    st.set_page_config(page_title="📚 PDF Chat App (FREE)", page_icon="📚")
    st.title("📚 PDF Chat App (FREE)")
    st.write("Upload PDFs and ask questions")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    with st.sidebar:
        st.subheader("Upload PDFs")
        docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_text_from_pdf(docs)
                chunks = chunk_text(raw_text)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.success("PDFs processed successfully!")

    st.subheader("Ask a question from your PDFs")
    question = st.text_input("Enter your question")

    if question and st.session_state.vectorstore:
        with st.spinner("Generating answer..."):
            docs = st.session_state.vectorstore.similarity_search(question, k=3)
            context = "\n".join([d.page_content for d in docs])
            answer = generate_answer(context, question)

        st.markdown("### ✅ Answer")
        st.write(answer)


# ---------- RUN ----------
if __name__ == "__main__":
    main()
