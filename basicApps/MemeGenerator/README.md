# AI Meme Generator Agent (Browser-Use + LLM)

A Streamlit app that leverages the **browser-use** agent and multimodal LLMs (OpenAI GPT-4o or Gemini) to automate meme creation on Imgflip, all from natural-language input.

---

## Features

- 🎨 Describe your own meme concept in plain English.
- 🤖 The app spins up a real browser agent to search Imgflip, select a template, add captions, and generate a meme.
- 🖼 Displays the final meme image and provides a direct link.
- 💡 Works with either **OpenAI GPT-4o** or **Google Gemini 1.5 Flash**.
- 🔧 Built with **browser-use**, requiring **Python 3.11+** and Playwright’s Chromium.

---

## Setup & Installation

```bash
# 1) Use Python 3.11+ virtual environment
python3 -m venv venv
source venv/bin/activate

# 2) Install dependencies
pip install streamlit browser-use langchain-openai langchain-google-genai python-dotenv playwright

# 3) Install Chromium for browser-use
# If your system has 'uvx':
uvx playwright install chromium --with-deps
# Otherwise:
playwright install chromium --with-deps --no-shell

# 4) Set API keys in `.env`
# For OpenAI (GPT-4o):
OPENAI_API_KEY=sk-...
# For Gemini (Gemini 1.5 Flash):
GOOGLE_API_KEY=...

# 5) Run the app
streamlit run app.py
```
