import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory



def get_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


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


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


def main():
    st.set_page_config(page_title="PDF Chat (FREE)", page_icon="📚")
    st.title("📚 PDF Chat App (FREE)")
    st.write("Upload PDFs and ask questions")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    user_question = st.text_input("Ask a question")

    if user_question and st.session_state.conversation:
        response = st.session_state.conversation({"question": user_question})
        st.write("**Answer:**")
        st.write(response["answer"])

    with st.sidebar:
        st.subheader("Upload PDFs")
        docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_pdf(docs)
                chunks = chunk_text(raw_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Ready to chat!")


if __name__ == "__main__":
    main()

