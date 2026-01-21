import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# ---------- PDF TEXT ----------
def get_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        file = PdfReader(pdf)
        for page in file.pages:
            text += page.extract_text()
    return text


# ---------- CHUNKING ----------
def chunk_text(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(raw_text)



# ---------- VECTOR STORE ----------
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


# ---------- CONVERSATION CHAIN ----------
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


# ---------- CHAT HANDLER ----------
def handle_user_input(user_query):
    response = st.session_state.conversation({"question": user_query})
    st.write("**Answer:**")
    st.write(response["answer"])


# ---------- MAIN APP ----------
def main():
    st.set_page_config(page_title="PDF Chat", page_icon="📚")
    load_dotenv()

    st.title("📚 PDF Chat App")
    st.write("Upload PDFs and ask questions")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Ask your PDF")
    user_query = st.text_input("Enter your question:")

    if user_query and st.session_state.conversation:
        handle_user_input(user_query)

    with st.sidebar:
        st.subheader("Upload PDFs")
        docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_text_from_pdf(docs)
                chunks = chunk_text(raw_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("PDFs processed successfully!")


# ---------- RUN ----------
if __name__ == "__main__":
    main()

