# app.py
import io
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Headless plotting so Streamlit can capture figures reliably
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain_openai import ChatOpenAI
import re, tempfile
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

fig, ax = plt.subplots()
ax.scatter([1, 2, 3], [1, 2, 3])

load_dotenv()


def dataVisualize():
    st.set_page_config(page_title="Visualize Data", page_icon="📊", layout="wide")
    st.title("🛡️📊 Ask question to Your Data via SQL (DuckDB/SQLite + LangChain)")

    st.markdown(
        "Upload **CSV/XLSX** → we load into a local DB → the LLM **generates SQL only**, "
        "we run it, and show the results (no arbitrary Python execution)."
    )

    backend = st.radio(
        "Choose backend", ["DuckDB (recommended)", "SQLite"], horizontal=True
    )
    uploads = st.file_uploader(
        "Upload one or more files", type=["csv", "xlsx"], accept_multiple_files=True
    )
    question = st.text_area(
        "Your question",
        "Total rows per category in a descending table; also show a simple bar chart.",
    )

    def _sanitize(name: str) -> str:
        base = os.path.splitext(os.path.basename(name))[0]
        return re.sub(r"[^0-9a-zA-Z_]+", "_", base).strip("_") or "table"

    if st.button("Ask"):
        if not uploads:
            st.error("Please upload at least one CSV or XLSX file.")
            st.stop()

        import sqlite3  # local import for the SQLite branch

        tmpdir = tempfile.mkdtemp()
        created = []

        # ---------- Build engine FIRST, then load data with the SAME engine ----------
        if backend.startswith("DuckDB"):
            db_path = os.path.join(tmpdir, "session.duckdb")
            engine = create_engine(
                f"duckdb:///{db_path}"
            )  # requires `duckdb-engine`   [oai_citation:4‡MotherDuck](https://motherduck.com/docs/integrations/language-apis-and-drivers/python/sqlalchemy/?utm_source=chatgpt.com)
            db_uri = f"duckdb:///{db_path}"
        else:
            db_path = os.path.join(tmpdir, "session.sqlite")
            engine = create_engine(f"sqlite:///{db_path}")
            db_uri = f"sqlite:///{db_path}"

        # Ingest each upload into a table using the SAME engine
        with engine.begin() as conn:
            for f in uploads:
                tbl = _sanitize(f.name)
                raw = f.read()
                if f.name.lower().endswith(".csv"):  # ← fixed typo (no space)
                    df = pd.read_csv(io.BytesIO(raw))
                else:
                    df = pd.read_excel(io.BytesIO(raw))
                df.to_sql(
                    tbl, conn, if_exists="replace", index=False
                )  # pandas→SQLAlchemy   [oai_citation:5‡pandas.pydata.org](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html?utm_source=chatgpt.com)
                created.append((tbl, df.shape))

        with st.expander("📁 Tables created"):
            for tbl, (r, c) in created:
                st.write(f"**{tbl}** — {r} rows × {c} cols")

        # ---------- LangChain: SQL-only chain (no Python REPL) ----------
        llm = ChatOpenAI(temperature=0)
        db = SQLDatabase.from_uri(
            db_uri
        )  # SQLAlchemy URI wrapper   [oai_citation:6‡LangChain](https://python.langchain.com/api_reference/community/utilities/langchain_community.utilities.sql_database.SQLDatabase.html?utm_source=chatgpt.com)
        sql_chain = create_sql_query_chain(
            llm, db
        )  # generates SQL only      [oai_citation:7‡LangChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sql_database.query.create_sql_query_chain.html?utm_source=chatgpt.com)

        allowed_tables = ", ".join(t for t, _ in created)
        prompt_hint = (
            "Use ONLY these tables: "
            + allowed_tables
            + ". Prefer straightforward SELECTs with GROUP BY/ORDER BY. "
            "Do not guess column names; rely on table info."
        )  # Prompting tips for SQL QA                                      [oai_citation:8‡LangChain](https://python.langchain.com/docs/how_to/sql_prompting/?utm_source=chatgpt.com)

        with st.spinner("Generating SQL…"):
            sql = sql_chain.invoke(
                {"question": f"{prompt_hint}\n\nUser question: {question}"}
            )

        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        # ---------- Execute SQL ----------
        try:
            df_out = pd.read_sql_query(sql, engine)
        except Exception as e:
            st.error(f"SQL failed: {e}")
            st.stop()

        st.subheader("Result")
        st.dataframe(df_out, use_container_width=True)

        # ---------- Optional quick chart ----------
        if df_out.shape[1] >= 2:
            xcol, ycol = df_out.columns[0], df_out.columns[1]
            if pd.api.types.is_numeric_dtype(df_out[ycol]):
                fig, ax = plt.subplots()
                ax.bar(df_out[xcol].astype(str).head(20), df_out[ycol].head(20))
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.set_title("Quick view")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(
                    fig
                )  # always pass figure (no global-figure deprecation)    [oai_citation:9‡docs.streamlit.io](https://docs.streamlit.io/develop/api-reference/charts/st.pyplot?utm_source=chatgpt.com)

    st.info(
        "This app uses LangChain’s SQL chain (generates SQL only) over DuckDB/SQLite for safer analysis. "
        "We also pass a Matplotlib Figure into st.pyplot() to avoid deprecated global-figure usage."
    )
