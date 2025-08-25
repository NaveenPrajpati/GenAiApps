# 💔 Breakup Recovery Squad

An empathetic **Streamlit** app powered by **LangChain** and **Google Gemini** that offers support after a breakup. It accepts **text + screenshots** (multimodal), runs four focused “personas,” and (optionally) uses web search for grounded, tough-love advice.

---

## ✨ What it does

- **Therapist** — validates feelings with warmth and encouragement.
- **Closure** — drafts unsent messages + simple release/closure exercises.
- **Recovery Plan** — a practical 7-day routine (15–45 min tasks).
- **Brutal Honesty** — direct, fact-backed advice; can call DuckDuckGo search.

---

## 🚀 Features

- 🧠 **Multi-Agent Team:**
  - **Therapist Agent:** Offers empathetic support and coping strategies.
  - **Closure Agent:** Writes emotional messages users shouldn't send for catharsis.
  - **Routine Planner Agent:** Suggests daily routines for emotional recovery.
  - **Brutal Honesty Agent:** Provides direct, no-nonsense feedback on the breakup.
- 📷 **Chat Screenshot Analysis:**
  - Upload screenshots for chat analysis.
- 🔑 **API Key Management:**
  - Store and manage your Gemini API keys securely via Streamlit's sidebar.
- ⚡ **Parallel Execution:**
  - Agents process inputs in coordination mode for comprehensive results.
- ✅ **User-Friendly Interface:**
  - Simple, intuitive UI with easy interaction and display of agent responses.

---

## 🛠️ Usage

1. **Enter Your Feelings:**
   - Describe how you're feeling in the text area.
2. **Upload Screenshot (Optional):**
   - Upload a chat screenshot (PNG, JPG, JPEG) for analysis.
3. **Execute Agents:**
   - Click **"Get Recovery Support"** to run the multi-agent team.
4. **View Results:**
   - Individual agent responses are displayed.
   - A final summary is provided by the Team Leader.

---

## 🧑‍💻 Agents Overview

- **Therapist Agent**
  - Provides empathetic support and coping strategies.
  - Uses **Gemini 2.0 Flash (Google Vision Model)** and DuckDuckGo tools for insights.
- **Closure Agent**

  - Generates unsent emotional messages for emotional release.
  - Ensures heartfelt and authentic messages.

- **Routine Planner Agent**

  - Creates a daily recovery routine with balanced activities.
  - Includes self-reflection, social interaction, and healthy distractions.

- **Brutal Honesty Agent**
  - Offers direct, objective feedback on the breakup.
  - Uses factual language with no sugar-coating.

---

## 📄 License

This project is licensed under the **MIT License**.

---
