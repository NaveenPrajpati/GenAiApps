# app.py
from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain (latest patterns)
from langchain_openai import (
    ChatOpenAI,
)  # optional (not used below, but keep if you want OpenAI fallback)
from langchain_core.messages import HumanMessage, SystemMessage

# DuckDuckGo tool (correct import per docs)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_react_agent  # modern Agent API

load_dotenv()


def breakupRecovery():
    def _img_to_data_url(uploaded) -> str:
        """Convert an uploaded image (Streamlit UploadedFile) to a data URL for Gemini image input."""
        tmp_path = Path(tempfile.gettempdir()) / f"st_{uploaded.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getvalue())
        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # Keep MIME simple; Streamlit guards types. You can branch on uploaded.type if needed.
        return f"data:image/png;base64,{b64}"

    def _gemini_multimodal_parts(
        text: str, uploads: List[st.runtime.uploaded_file_manager.UploadedFile]
    ):
        """Return a list of message parts for ChatOpenAI with optional images."""
        parts: List[dict] = [{"type": "text", "text": text}] if text else []
        for f in uploads:
            parts.append({"type": "image_url", "image_url": _img_to_data_url(f)})
        return parts

    # ──────────────────────────────
    # App UI
    # ──────────────────────────────
    st.set_page_config(
        page_title="💔 Breakup Recovery Squad", page_icon="💔", layout="wide"
    )
    st.title("💔 Breakup Recovery Squad")
    st.markdown(
        "### Your AI-powered breakup recovery team is here to help!\n"
        "Share your feelings and (optionally) chat screenshots."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Share Your Feelings")
        user_input = st.text_area(
            "How are you feeling? What happened?",
            height=160,
            placeholder="Tell us your story…",
        )

    with col2:
        st.subheader("Upload Chat Screenshots (optional)")
        uploaded_files = st.file_uploader(
            "Images: JPG/PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
        if uploaded_files:
            for f in uploaded_files:
                st.image(f, caption=f.name, use_container_width=True)

    go = st.button("Get Recovery Plan 💝", type="primary")

    # ──────────────────────────────
    # Run
    # ──────────────────────────────
    if go:

        if not (user_input or uploaded_files):
            st.warning("Please share your feelings or upload at least one screenshot.")
            st.stop()

        # Build the Gemini chat model (correct class + model name per docs)
        gemini = ChatOpenAI(
            temperature=0.7,
        )

        # Prepare multimodal parts once
        parts = _gemini_multimodal_parts(user_input, uploaded_files or [])

        # ── 1) Therapist (empathetic)
        with st.spinner("🤗 Getting empathetic support…"):
            therapist_msg = [
                SystemMessage(
                    content=(
                        "You are an empathetic therapist who:\n"
                        "1) validates feelings, 2) uses gentle humor, 3) shares relatable experiences,\n"
                        "4) offers comfort & encouragement. Keep it kind, warm, concise."
                    )
                ),
                HumanMessage(content=parts),
            ]
            therapist_resp = gemini.invoke(therapist_msg)
            st.subheader("🤗 Emotional Support")
            st.markdown(therapist_resp.content)

        # ── 2) Closure (unsent messages, release)
        with st.spinner("✍️ Crafting closure messages…"):
            closure_msg = [
                SystemMessage(
                    content=(
                        "You specialize in emotional closure. Produce:\n"
                        "• A heartfelt unsent message template\n"
                        "• A short release exercise\n"
                        "• A gentle ritual to mark closure\n"
                        "Tone: authentic, kind, non-judgmental."
                    )
                ),
                HumanMessage(content=parts),
            ]
            closure_resp = gemini.invoke(closure_msg)
            st.subheader("✍️ Finding Closure")
            st.markdown(closure_resp.content)

        # ── 3) Routine planner (7-day plan)
        with st.spinner("📅 Creating your recovery plan…"):
            routine_msg = [
                SystemMessage(
                    content=(
                        "You design practical 7-day recovery plans with daily activities,\n"
                        "self-care, light social guidance, and 6–10 song suggestions overall.\n"
                        "Ensure activities are doable in 15–45 minutes."
                    )
                ),
                HumanMessage(content=parts),
            ]
            routine_resp = gemini.invoke(routine_msg)
            st.subheader("📅 Your Recovery Plan")
            st.markdown(routine_resp.content)

        # ── 4) Brutal honesty (optionally uses DDG tool)
        with st.spinner("💪 Getting honest perspective…"):
            # Set up DuckDuckGo search tool (correct import + usage)
            ddg = DuckDuckGoSearchRun()

            # Create a simple ReAct agent that can call DuckDuckGo when useful
            honesty_system = (
                "You are blunt but constructive. If helpful, call the DuckDuckGo search tool "
                "to pull a stat, reputable article, or definition that supports your advice. "
                "Be direct. Avoid cruelty. End with 3 actionable steps."
            )
            honesty_prompt = (
                "Give a tough-love analysis of the situation below. Use the search tool only "
                "if it will add concrete facts or perspective.\n\n"
                f"User context:\n{user_input[:4000]}"
            )

            # Build & run the agent
            honesty_agent = create_react_agent(
                gemini, tools=[ddg], state_modifier=honesty_system
            )
            honesty_exec = AgentExecutor(
                agent=honesty_agent, tools=[ddg], verbose=False
            )
            honesty_out = honesty_exec.invoke({"input": honesty_prompt})

            st.subheader("💪 Honest Perspective")
            st.markdown(honesty_out["output"])

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center'>"
            "<p>Made with ❤️ by the Breakup Recovery Squad</p>"
            "<p>Share your journey with <b>#BreakupRecoverySquad</b></p>"
            "</div>",
            unsafe_allow_html=True,
        )
