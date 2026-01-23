import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA


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


# ---------- QA CHAIN ----------
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={
            "temperature": 0.3,
            "max_length": 512
        }
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )


# ---------- MAIN APP ----------
def main():
    st.set_page_config(page_title="PDF Chat (FREE)", page_icon="📚")
    st.title("📚 PDF Chat App (FREE)")
    st.write("Upload PDFs and ask questions")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    user_question = st.text_input("Ask a question from your PDFs")

    if user_question and st.session_state.conversation:
        answer = st.session_state.conversation.run(user_question)
        st.markdown("### ✅ Answer")
        st.write(answer)

    with st.sidebar:
        st.subheader("Upload PDFs")
        docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process PDFs"):
            if not docs:
                st.warning("Please upload at least one PDF")
                return

            with st.spinner("Processing PDFs..."):
                raw_text = get_text_from_pdf(docs)
                chunks = chunk_text(raw_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("PDFs processed. Ask questions now!")


# ---------- RUN ----------
if __name__ == "__main__":
    main()
