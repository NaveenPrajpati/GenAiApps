import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import LanceDB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables
load_dotenv()
st.set_page_config(
    page_title="Agentic RAG with LangChain", page_icon="🧠", layout="wide"
)
st.title("🧠 Agentic RAG with LangChain + LanceDB")
st.markdown(
    """
This app demonstrates RAG using LangChain and LanceDB.
Enter your OpenAI API key in the sidebar to get started!
"""
)


    st.header("🔧 Configuration")
    openai_key = st.text_input(
        "OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", "")
    )
    new_url = st.text_input(
        "Add Knowledge Source URL",
        placeholder="https://docs.langchain.com/docs/introduction",
    )
    if st.button("➕ Add URL", type="primary"):
        if new_url:
            st.session_state.urls_to_add = new_url
            st.success(f"URL added: {new_url}")
        else:
            st.error("Please enter a valid URL.")


    @st.cache_resource(show_spinner="📚 Loading knowledge base...")
    def load_knowledge(urls):
        docs = []
        for url in urls:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        vectordb = LanceDB.from_documents(
            split_docs,
            embedding=embeddings,
            uri="tmp/lancedb",
            table_name="agentic_rag_docs",
        )
        return vectordb

    if "knowledge_urls" not in st.session_state:
        st.session_state.knowledge_urls = [
            "https://docs.langchain.com/docs/introduction"
        ]
    # Add new URLs
    if hasattr(st.session_state, "urls_to_add"):
        if st.session_state.urls_to_add not in st.session_state.knowledge_urls:
            st.session_state.knowledge_urls.append(st.session_state.urls_to_add)
        del st.session_state.urls_to_add
        st.rerun()

    # Load knowledge
    vectordb = load_knowledge(st.session_state.knowledge_urls)
    retriever = vectordb.as_retriever()

    # LangChain QA Chain
    llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4o")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        verbose=True,
    )

    st.sidebar.markdown("### 📚 Current Knowledge Sources")
    for i, url in enumerate(st.session_state.knowledge_urls, 1):
        st.sidebar.markdown(f"{i}. {url}")

    st.divider()
    st.subheader("🤔 Ask a Question")
    query = st.text_area(
        "Your question:",
        value=st.session_state.get("query", "What is LangChain?"),
        height=100,
    )
    if st.button("🚀 Get Answer", type="primary"):
        if query:
            st.markdown("### 💡 Answer")
            with st.spinner("🔍 Searching and generating answer..."):
                result = qa_chain({"query": query})
                st.markdown(result["result"])
                if result.get("source_documents"):
                    st.markdown("#### 📖 Sources")
                    for doc in result["source_documents"]:
                        st.markdown(f"- {doc.metadata.get('source', '')}")
        else:
            st.error("Please enter a question")


