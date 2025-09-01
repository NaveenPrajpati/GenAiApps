# app.py
import os
import logging
import streamlit as st

from typing import List

# === Logging ===
logging.basicConfig(level=logging.INFO)

# === LangChain core / models / tools / loaders ===
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import NewsURLLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- UI ----------
st.title("AI Startup Trend Analysis Agent 📈")
st.caption(
    "Get the latest trend analysis and startup opportunities based on your topic of interest in a click!."
)

topic = st.text_input("Enter the area of interest for your Startup:")
max_results = st.sidebar.slider("Max news links", min_value=3, max_value=15, value=6)
recent_hint = st.sidebar.selectbox(
    "Recency hint for search",
    ["past week", "past month", "past 3 months", "none"],
    index=1,
)

if st.button("Generate Analysis"):
    with st.spinner("Processing your request..."):
        try:
            # === Model ===
            llm = ChatOpenAI(
                temperature=0.2,
                max_tokens=1500,
            )

            # === 1) Collect news links with DuckDuckGo ===
            # Use structured results so we can easily extract URLs
            ddg = DuckDuckGoSearchResults(output_format="list", max_results=max_results)
            recency_q = f" from the {recent_hint}" if recent_hint != "none" else ""
            query = f"latest news articles about {topic}{recency_q}"
            results: List[dict] = ddg.invoke(
                query
            )  # list of dicts (title, link, snippet, etc.)

            # Extract URLs and display the hits
            urls = []
            st.subheader("Collected Articles")
            if not isinstance(results, list):
                st.info("No structured results returned from search.")
            else:
                for r in results:
                    link = r.get("link") or r.get("href") or r.get("url")
                    title = r.get("title") or "(no title)"
                    snippet = r.get("snippet") or r.get("body") or ""
                    if link and link.startswith("http"):
                        urls.append(link)
                        st.markdown(
                            f"- [{title}]({link})  \n  <small>{snippet}</small>",
                            unsafe_allow_html=True,
                        )

            if not urls:
                st.error(
                    "No valid article links found. Try broadening your topic or changing recency."
                )
                st.stop()

            # === 2) Load article content ===
            loader = NewsURLLoader(
                urls=urls,
                text_mode=True,
                continue_on_failure=True,
                show_progress_bar=False,
            )
            docs = loader.load()

            if not docs:
                st.error("Failed to load article contents.")
                st.stop()

            # Optional: chunk long docs for map-reduce summarization
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000, chunk_overlap=300
            )
            doc_chunks = splitter.split_documents(docs)

            # === 3) Summarize each chunk (map), then combine (reduce) ===
            map_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a concise news analyst. Extract the key facts, numbers, entities, "
                        "and any signals relevant to startups, markets, users, regulations, and funding.",
                    ),
                    (
                        "human",
                        "Article chunk:\n\n{chunk}\n\nReturn a 5-8 bullet point summary.",
                    ),
                ]
            )
            map_chain = map_prompt | llm | StrOutputParser()

            summaries = []
            for d in doc_chunks:
                summaries.append(map_chain.invoke({"chunk": d.page_content}))

            reduce_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an elite research synthesizer for startup founders. "
                        "Combine notes across multiple articles into a succinct brief.",
                    ),
                    (
                        "human",
                        "Here are bullet notes from many recent articles about {topic}.\n\n"
                        "{notes}\n\n"
                        "Create a consolidated set of 8-12 bullets capturing news highlights, "
                        "market/context shifts, user behavior signals, competitive moves, "
                        "regulatory items, and funding/deal activity.",
                    ),
                ]
            )
            reduce_chain = reduce_prompt | llm | StrOutputParser()
            consolidated_summary = reduce_chain.invoke(
                {"topic": topic, "notes": "\n".join(summaries)}
            )

            # === 4) Trend Analysis & Opportunities ===
            analysis_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a veteran startup strategist. Based on the news brief, "
                        "identify emerging trends and concrete startup opportunities.",
                    ),
                    (
                        "human",
                        "News brief about {topic}:\n\n{brief}\n\n"
                        "Now produce a founder-facing **Trend Analysis & Opportunity Report** with:\n"
                        "1) 3-5 named trends (each with 2–3 crisp proof points from the news),\n"
                        "2) Opportunity theses (who is the ICP, wedge, business model, GTM),\n"
                        "3) Risks & regulatory considerations,\n"
                        "4) A 30/60/90-day action plan for a new startup exploring this space.\n"
                        "Keep it pragmatic and specific.",
                    ),
                ]
            )
            analysis_chain = analysis_prompt | llm | StrOutputParser()
            report = analysis_chain.invoke(
                {"topic": topic, "brief": consolidated_summary}
            )

            st.subheader("Trend Analysis and Potential Startup Opportunities")
            st.write(report)

            with st.expander("Show consolidated news brief"):
                st.write(consolidated_summary)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Enter the topic and API key, then click 'Generate Analysis' to start.")
