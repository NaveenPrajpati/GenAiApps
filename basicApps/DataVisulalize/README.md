# Ask Your Data via SQL – Streamlit + LangChain SQL Agent

**This application allows you to upload CSV/XLSX files, load them into a local SQL database (DuckDB or SQLite), and use a Large Language Model (LLM) to generate SQL queries in response to natural-language questions. The app executes those SQL queries—no arbitrary Python code execution—providing a safer, transparent data-QA interface.**

---

## Features

- **Upload multiple CSV/XLSX files** as underlying tables.
- Choose between **DuckDB (recommended)** or **SQLite** as the SQL backend.
- Uses **LangChain’s SQL chain** to convert natural-language questions into SQL only (no Python execution).
- Displays:
  - Generated SQL query
  - Query results in a table
  - Quick plot if the results look like label-value pairs
- Built with **Streamlit** for easy deployment and interactive UI.

---

## Why This Approach?

- **Security:** Avoids LLM execution of arbitrary Python. Only SQL is generated and executed.
- **Performance:** DuckDB is optimized for analytics, especially on CSV input.
- **Maintainability:** Clear separation between query generation (LLM) and execution (SQL engine).
- **Modern Design:** Streamlit + LangChain integration with SQLDatabase and `create_sql_query_chain`.

---

## Screenshots

_(Feel free to add a screenshot of the UI here.)_

---

## Setup & Installation

```bash
# 1) Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 2) Install dependencies
pip install streamlit pandas openpyxl matplotlib python-dotenv duckdb duckdb-engine sqlalchemy langchain langchain-community langchain-openai

# 3) Set your OpenAI key
export OPENAI_API_KEY="sk-..."

# 4) Run the app
streamlit run app.py
```
