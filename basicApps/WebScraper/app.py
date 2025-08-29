import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import validators
from typing import List, Dict, Any, Optional
import time
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Set page config
st.set_page_config(page_title="LangChain Web Scraper", page_icon="🌐", layout="wide")


class ExtractionResult(BaseModel):
    """Structured extraction result."""

    summary: Optional[str] = Field(
        default=None,
        description="Short summary of the page relevant to the instruction.",
    )
    items: List[str] = Field(
        ..., description="List of items/information extracted per the instruction."
    )
    evidence: Optional[List[str]] = Field(
        default=None,
        description="Optional quotes/snippets from the page that support each item.",
    )


def load_website_content(url: str) -> List[Document]:
    """Load content from the website using LangChain WebBaseLoader"""
    try:
        # Validate URL
        if not validators.url(url):
            st.error("Please enter a valid URL")
            return []

        with st.spinner("Loading website content..."):
            loader = WebBaseLoader(url)
            documents = loader.load()

            if not documents:
                st.error("No content could be loaded from the URL")
                return []

            # Split documents into smaller chunks if they're too large
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000, chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(documents)

            return split_docs

    except Exception as e:
        st.error(f"Error loading website: {str(e)}")
        return []


def extract_information(documents: List[Document], extraction_query: str) -> str:
    """Extract specific information from documents using LLM"""
    try:
        # Combine all document content
        combined_text = "\n\n".join([doc.page_content for doc in documents])

        # Create prompt template for extraction
        extraction_prompt = PromptTemplate.from_template(
            """
            You are an AI assistant specialized in extracting specific information from web content.
            
            Website Content:
            {text}
            
            Extraction Request: {query}
            
            Instructions:
            1. Carefully analyze the provided website content
            2. Extract only the information that directly relates to the user's request
            3. If the requested information is not found, clearly state that it's not available
            4. Present the extracted information in a clear, organized format
            5. Include relevant details and context when available
            
            Extracted Information:
            """,
        )
        llm = ChatOpenAI(temperature=0.1, max_tokens=2000).with_structured_output(
            ExtractionResult
        )

        with st.spinner("Extracting information..."):
            result = llm.invoke(
                [
                    {"role": "system", "content": "You are a careful extractor..."},
                    {
                        "role": "user",
                        "content": f"Instruction:\n{extraction_query}\n\nContent:\n{combined_text}",
                    },
                ]
            )

        return result

    except Exception as e:
        st.error(f"Error extracting information: {str(e)}")
        return ""


def display_extraction_results(result: str):
    """Display the extraction results in a formatted way"""
    if result:
        st.success("✅ Information extracted successfully!")
        st.markdown("### Extracted Information")
        st.markdown(result)
    else:
        st.warning(
            "No information could be extracted. Please try a different query or URL."
        )


def webScraper():
    """Main Streamlit app"""

    # Title and description
    st.title("🌐 LangChain Web Scraper")
    st.markdown(
        """
    This app uses LangChain to scrape websites and extract specific information using AI.
    Simply enter a URL and describe what you want to extract!
    """
    )

    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.markdown("---")
        st.markdown("### 📝 Example Queries")
        st.markdown(
            """
        - "Extract all contact information"
        - "Find product names and prices"
        - "Get all email addresses"
        - "Extract article headlines and summaries"
        - "Find company information and address"
        - "Get all links and their descriptions"
        """
        )

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        # URL input
        url = st.text_input(
            "🔗 Website URL",
            placeholder="https://example.com",
            help="Enter the full URL of the website you want to scrape",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        load_button = st.button("🔄 Load Website", type="secondary")

    # Extraction query input
    extraction_query = st.text_area(
        "🎯 What do you want to extract?",
        placeholder="Describe what information you want to extract from the website...",
        height=100,
        help="Be specific about what you want to extract (e.g., 'contact information', 'product prices', 'article titles')",
    )

    # Extract button
    extract_button = st.button(
        "🚀 Extract Information",
        type="primary",
        disabled=not (url and extraction_query),
    )

    # Store documents in session state
    if "documents" not in st.session_state:
        st.session_state.documents = []

    # Load website content
    if load_button or (extract_button and not st.session_state.documents):
        if url:
            documents = load_website_content(url)
            st.session_state.documents = documents

            if documents:
                st.success(
                    f"✅ Successfully loaded {len(documents)} document chunks from the website"
                )

                # Show preview of loaded content
                with st.expander("📄 Preview of loaded content"):
                    preview_text = (
                        documents[0].page_content[:500] + "..."
                        if len(documents[0].page_content) > 500
                        else documents[0].page_content
                    )
                    st.text(preview_text)
        else:
            st.error("Please enter a valid URL")

    # Extract information
    if extract_button and st.session_state.documents and extraction_query:

        result = extract_information(st.session_state.documents, extraction_query)
        display_extraction_results(result)
