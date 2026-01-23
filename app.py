import streamlit as st
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
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text.strip()


# ---------- CHUNKING ----------
def chunk_text(text):
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


# ---------- VECTOR STORE ----------
def get_vectorstore(chunks):
    if not chunks:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


# ---------- MAIN APP ----------
def main():
    st.set_page_config(page_title="PDF Chat (FREE)", page_icon="📚")
    st.title("📚 PDF Chat App (FREE)")
    st.write("Upload PDFs and ask questions")

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
                    st.error("No readable text found in PDFs (scanned PDFs not supported).")
                    return

                chunks = chunk_text(raw_text)

                if not chunks:
                    st.error("Failed to split text into chunks.")
                    return

                st.session_state.vectorstore = get_vectorstore(chunks)

                if st.session_state.vectorstore is None:
                    st.error("Vector store creation failed.")
                    return

                st.success("PDFs processed successfully!")

    user_question = st.text_input("Ask a question from your PDFs")

    if user_question and st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(
            user_question, k=3
        )

        if not docs:
            st.info("No relevant content found.")
            return

        context = "\n\n".join([d.page_content for d in docs])

        st.markdown("### ✅ Relevant Answer (from PDF)")
        st.write(context)


# ---------- RUN ----------
if __name__ == "__main__":
    main()
