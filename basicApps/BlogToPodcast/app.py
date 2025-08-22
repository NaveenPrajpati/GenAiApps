import os
import textwrap
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

try:
    from murf import Murf

    murf_client = Murf(api_key=os.getenv("MURF_API_KEY"))
except ImportError:
    murf_client = None


def blogToPodcast():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY; set it via .env or Streamlit secrets.")
        return

    st.title("Blog → Podcast  👋")
    url = st.text_input("Enter your blog URL", placeholder="https://...")
    max_chars = st.slider("Max summary length (chars)", 500, 4000, 2000, step=100)
    do_tts = st.checkbox("Generate audio (Murf)", value=True)

    if st.button("Generate Podcast", disabled=not url or len(url) < 5):
        try:
            with st.spinner("Loading and summarizing…"):
                loader = WebBaseLoader(url)
                docs = loader.load()
                if not docs:
                    st.error("Could not load the page. Please check the URL.")
                    return

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000, chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)

                llm = ChatOpenAI(temperature=0.3)
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = chain.run(chunks)
                summary = textwrap.shorten(
                    summary.strip().replace("\n\n", "\n"),
                    width=max_chars,
                    placeholder="…",
                )

            st.subheader("Blog Summary")
            st.markdown(summary)

            if do_tts:
                if murf_client is None:
                    st.warning("Murf SDK not installed; skipping audio.")
                else:
                    with st.spinner("Generating audio…"):
                        tts = murf_client.text_to_speech.generate(
                            text=summary, voice_id="en-US-terrell"
                        )
                        audio_bytes = tts.audio_file
                        st.subheader("Audio Podcast")
                        st.audio(audio_bytes, format="audio/mp3")

        except Exception as e:
            st.error(f"Error: {e}")
