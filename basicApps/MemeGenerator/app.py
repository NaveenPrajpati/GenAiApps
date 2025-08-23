# app.py
import os
import re
import asyncio
import streamlit as st
from dotenv import load_dotenv

# Browser-use (Python 3.11+)
# pip install browser-use
from browser_use import Agent

# LLMs (choose one at runtime)
from langchain_openai import ChatOpenAI  # OpenAI (e.g., gpt-4o)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)  # Gemini (e.g., gemini-1.5-flash)
from langchain.chat_models import init_chat_model
from browser_use.llm import ChatOpenAI, ChatGoogle
from browser_use import Agent

load_dotenv()


# -----------------------------
# Async worker that drives the Agent
# -----------------------------
async def generate_meme(query: str, model_choice: str) -> str | None:
    """Use browser-use Agent to generate a meme on Imgflip and return a direct image URL."""

    # pick one model at runtime
    def make_llm(model_choice: str):
        if model_choice == "Gemini":
            # GOOGLE_API_KEY must be set in env
            return ChatGoogle(model="gemini-2.0-flash-exp")  # vision-capable
        else:
            # OPENAI_API_KEY must be set in env
            return ChatOpenAI(model="gpt-4o")  # or "o3", "gpt-4o-mini", etc.

    llm = make_llm(model_choice)

    task_description = (
        "You are a meme generator expert. You are given a query and you need to generate a meme for it.\n"
        "Steps:\n"
        "1) Open https://imgflip.com/memetemplates\n"
        "2) Use the Search bar to search for ONE MAIN ACTION VERB extracted from this topic: '{0}'\n"
        "3) Pick a template that fits '{0}' metaphorically and click 'Add Caption'\n"
        "4) Fill Top Text (setup/context) and Bottom Text (punchline/outcome) relevant to '{0}'\n"
        "5) Preview the meme; if it doesn't make sense, adjust texts and retry\n"
        "6) Click 'Generate meme'\n"
        "7) Copy the generated meme's public image link and return ONLY that link\n"
    ).format(query)

    # 3) Start the browser-use Agent
    # See docs for options like controller/hooks, etc.  [oai_citation:5‡Browser Use](https://docs.browser-use.com/customize/agent-settings?utm_source=chatgpt.com)
    agent = Agent(
        task=task_description,
        llm=llm,
        use_vision=True,  # recommended for page understanding
        max_actions_per_step=5,
        max_failures=25,
    )

    # 4) Run the Agent (limit steps defensively)
    # Examples in issues/docs show .run(max_steps=...) and then .final_result()  [oai_citation:6‡GitHub](https://github.com/browser-use/browser-use/issues/2513?utm_source=chatgpt.com) [oai_citation:7‡Medium](https://medium.com/%40michael.rhema/automated-browsing-browser-use-is-a-free-alternative-to-anthropics-computer-use-fff444b8b631?utm_source=chatgpt.com)
    history = await agent.run(max_steps=40)
    final_text = history.final_result()

    # 5) Extract a usable meme URL
    # Case A: direct image link (preferred)
    m = re.search(
        r"https?://i\.imgflip\.com/([\w]+)\.(?:jpg|png|gif)", final_text, re.I
    )
    if m:
        return m.group(0)

    # Case B: page link like https://imgflip.com/i/XXXX  -> convert to i.imgflip.com/XXXX.jpg
    m = re.search(r"https?://imgflip\.com/i/(\w+)", final_text, re.I)
    if m:
        meme_id = m.group(1)
        return f"https://i.imgflip.com/{meme_id}.jpg"

    return None


# -----------------------------
# Streamlit UI
# -----------------------------
def generateMeme():
    st.title("🥸 AI Meme Generator Agent (browser-use)")
    st.info(
        "This app drives a real browser via **browser-use** to create a meme on Imgflip "
        "based on your idea. Make sure your API key for the chosen model is set."
    )

    # Model + keys
    model_choice = st.selectbox(
        "Choose LLM", ["OpenAI GPT-4o", "Gemini 1.5 Flash"], index=0
    )

    # if model_choice == "OpenAI GPT-4o":
    #     openai_key = st.text_input(
    #         "OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", "")
    #     )
    #     if openai_key:
    #         os.environ["OPENAI_API_KEY"] = openai_key
    # else:
    #     google_key = st.text_input(
    #         "GOOGLE_API_KEY", type="password", value=os.getenv("GOOGLE_API_KEY", "")
    #     )
    #     if google_key:
    #         os.environ["GOOGLE_API_KEY"] = google_key

    query = st.text_input(
        "Describe your meme idea",
        placeholder="e.g., 'When the build passes locally but CI says nope'",
    )

    # Env checks
    need_key_env = (
        "OPENAI_API_KEY" if model_choice == "OpenAI GPT-4o" else "GOOGLE_API_KEY"
    )

    if st.button("Generate Meme 🚀"):
        if not os.getenv(need_key_env):
            st.warning(f"Please provide your {need_key_env}")
            st.stop()
        if not query.strip():
            st.warning("Please enter a meme idea")
            st.stop()

        with st.spinner(f"Launching browser agent with {model_choice}…"):
            try:
                # Run the async task; Streamlit main is sync
                meme_url = asyncio.run(generate_meme(query.strip(), model_choice))

                if meme_url:
                    st.success("✅ Meme Generated!")
                    st.image(
                        meme_url, caption="Generated Meme", use_container_width=True
                    )
                    st.markdown(f"**Direct Link:** {meme_url}")
                else:
                    st.error(
                        "❌ Could not find a generated meme link. Try a simpler idea."
                    )
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.caption(
                    "Tips: Ensure **Python 3.11+** and Playwright Chromium are installed. "
                    "You can install Chromium via: `uvx playwright install chromium --with-deps`."
                )  #  [oai_citation:8‡GitHub](https://github.com/browser-use/browser-use?utm_source=chatgpt.com)
