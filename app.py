import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


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


# ---------- QA CHAIN (STABLE) ----------
def get_qa_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
        temperature=0.3,
        max_new_tokens=512,
    )

    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate.from_template(
        """Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain


# ---------- MAIN APP ----------
def main():
    st.set_page_config(page_title="PDF Chat (FREE)", page_icon="📚")
    st.title("📚 PDF Chat App (FREE)")
    st.write("Upload PDFs and ask questions")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    user_question = st.text_input("Ask a question from your PDFs")

    if user_question and st.session_state.qa_chain:
        answer = st.session_state.qa_chain.invoke(user_question)
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
                st.session_state.qa_chain = get_qa_chain(vectorstore)
                st.success("PDFs processed. Ask questions now!")


if __name__ == "__main__":
    main()
