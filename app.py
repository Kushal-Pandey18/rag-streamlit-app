import streamlit as st
from pypdf import PdfReader
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="wide")


# ---------------- FUNCTIONS ----------------
def get_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text.strip()


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


def generate_answer(context, question, api_key):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {api_key}"}

    prompt = f"""
Answer the question using the context below.
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

    if isinstance(data, dict):
        if "error" in data:
            return "⚠️ Model is loading or busy. Please try again in 30 seconds."
        return data.get("generated_text", "No answer generated.")

    return "Unexpected response."


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🔐 HuggingFace API Key")
    api_key = st.text_input("Enter your HuggingFace API Token", type="password")

    st.info("Your key is used only for this session and not stored.")

    st.divider()

    st.subheader("📂 Upload PDFs")
    pdf_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_btn = st.button("🚀 Process Documents")

    st.divider()
    st.caption("Built by Kushal Pandey")


# ---------------- MAIN UI ----------------
st.title("📄 Retrieval Augmented Generation (RAG) Engine")
st.write("Upload PDFs and ask questions using HuggingFace LLM.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


if process_btn:
    if not api_key:
        st.warning("⚠️ Please enter your HuggingFace API token.")
    elif not pdf_files:
        st.warning("⚠️ Please upload at least one PDF.")
    else:
        with st.spinner("Processing documents..."):
            raw_text = get_text_from_pdf(pdf_files)

            if not raw_text:
                st.error("No readable text found in PDFs.")
            else:
                chunks = chunk_text(raw_text)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.success("✅ Documents processed successfully!")


st.divider()
st.subheader("💬 Ask a question")

question = st.text_input("Ask a question about your documents")

if question and st.session_state.vectorstore and api_key:
    docs = st.session_state.vectorstore.similarity_search(question, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    with st.spinner("Generating answer..."):
        answer = generate_answer(context, question, api_key)

        st.markdown("### ✅ Answer")
        st.write(answer)

elif question:
    st.warning("⚠️ Please upload PDFs and enter API key first.")
