import streamlit as st
from pypdf import PdfReader
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="wide")


# ---------------- DARK THEME CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.chat-user {
    background-color: #1e293b;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.chat-bot {
    background-color: #020617;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    border-left: 3px solid #38bdf8;
}
.card {
    background-color: #020617;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


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


# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/337/337946.png", width=80)
    st.title("RAG Engine")

    api_key = st.text_input("🔑 HuggingFace API Token", type="password")
    st.caption("Used only for this session")

    show_chunks = st.toggle("📄 Show retrieved chunks")

    st.divider()

    pdf_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process Documents"):
        if not api_key:
            st.warning("Enter HuggingFace API token")
        elif not pdf_files:
            st.warning("Upload PDFs")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text = get_text_from_pdf(pdf_files)
                chunks = chunk_text(raw_text)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.session_state.chat_history = []
                st.success("✅ Documents processed!")

    if st.button("🧹 Reset Session"):
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.experimental_rerun()

    st.divider()
    st.caption("Built by Kushal Pandey")


# ---------------- MAIN UI ----------------
st.markdown("<h1 style='text-align:center;'>📄 Retrieval Augmented Generation (RAG)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload PDFs and chat with them using HuggingFace LLM</p>", unsafe_allow_html=True)

st.divider()

question = st.text_input("💬 Ask a question from your PDFs")

if question and st.session_state.vectorstore and api_key:
    docs = st.session_state.vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    with st.spinner("Thinking..."):
        answer = generate_answer(context, question, api_key)

    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("bot", answer))

# ---------------- CHAT UI ----------------
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='chat-user'>🧑 <b>You:</b> {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>🤖 <b>Bot:</b> {msg}</div>", unsafe_allow_html=True)

# ---------------- SHOW CHUNKS ----------------
if show_chunks and st.session_state.vectorstore and question:
    st.divider()
    st.subheader("📄 Retrieved Chunks")
    for i, d in enumerate(docs):
        st.markdown(f"<div class='card'><b>Chunk {i+1}</b><br>{d.page_content}</div>", unsafe_allow_html=True)
