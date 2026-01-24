import streamlit as st
from pypdf import PdfReader
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- CSS ----------------
def load_css():
    st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: white;
    }
    .chat-user {
        background-color: #1e293b;
        color: white;
        padding: 12px;
        border-radius: 10px;
        margin: 8px 0;
    }
    .chat-bot {
        background-color: #020617;
        color: #38bdf8;
        padding: 12px;
        border-radius: 10px;
        margin: 8px 0;
        border-left: 4px solid #22c55e;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------- PDF TEXT ----------
def get_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
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


# ---------- HF API CALL ----------
def generate_answer(context, question, api_token):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    prompt = f"""
Answer the question using only the context below.
If answer is not found, say "I don't know".

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


# ---------- MAIN ----------
def main():
    st.set_page_config(page_title="PDF Chat", page_icon="📚")
    load_css()

    st.markdown("<h1 style='color:black;'>📚 PDF Chat App </h1>", unsafe_allow_html=True)
    st.write("Upload PDFs and ask questions using HuggingFace LLM")

    # Session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        st.subheader("🔑 HuggingFace API Key")
        user_api_key = st.text_input("Enter your HuggingFace API key", type="password")

        if not user_api_key:
            st.info("Using Streamlit secrets if available.")

        st.subheader("📂 Upload PDFs")
        docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("📥 Process PDFs"):
            if not docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing PDFs..."):
                raw_text = get_text_from_pdf(docs)

                if not raw_text:
                    st.error("No readable text found in PDFs.")
                    return

                chunks = chunk_text(raw_text)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.success("PDFs processed successfully!")

        show_chunks = st.checkbox("Show retrieved chunks")

        if st.button("🔄 Reset session"):
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.experimental_rerun()

    # Choose API key
    if user_api_key:
        api_token = user_api_key
    else:
        api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")

    if not api_token:
        st.warning("⚠️ Please enter HuggingFace API key to enable answers.")
        return

    # Chat input
    user_question = st.text_input("Ask a question from your PDFs")

    if user_question and st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(user_question, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        with st.spinner("Generating answer..."):
            answer = generate_answer(context, user_question, api_token)

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("bot", answer))

        if show_chunks:
            st.markdown("### 🔍 Retrieved Chunks")
            for i, d in enumerate(docs):
                st.info(d.page_content)

    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-user'>👨‍💻 You: {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>🤖 Bot: {msg}</div>", unsafe_allow_html=True)


# ---------- RUN ----------
if __name__ == "__main__":
    main()

