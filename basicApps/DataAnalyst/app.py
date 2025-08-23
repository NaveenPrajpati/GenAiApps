# app.py
import io
import os
import textwrap
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Optional: safer SQL path via DuckDB (for big joins / sheet selection)
import duckdb

# Headless plotting so Streamlit can capture figures reliably
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


load_dotenv()


def dataAnalyst():
    st.set_page_config(page_title="Ask your data", page_icon="📊", layout="wide")
    show_code = False
    use_duckdb = True
    st.markdown(
        "Upload **CSV** or **XLSX**. Then ask a question like *“Show top 10 products by     revenue this year as a bar chart.”*"
    )

    uploads = st.file_uploader(
        "Upload one or more files", type=["csv", "xlsx"], accept_multiple_files=True
    )
    question = st.text_area(
        "Your question", "Show total rows and basic summary statistics."
    )

    if st.button("Ask"):
        if not uploads:
            st.error("Please upload at least one CSV or XLSX file.")
            st.stop()

        # ---- Load files to DataFrames ----
        dataframes = {}
        for f in uploads:
            name = os.path.splitext(os.path.basename(f.name))[0]
            try:
                if f.name.lower().endswith(".csv"):
                    df = pd.read_csv(f)
                else:
                    # If multiple sheets, read the first; you can extend to read all    sheets
                    df = pd.read_excel(f)
                dataframes[name] = df
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
                st.stop()

        # Preview
        with st.expander("📁 Preview loaded tables"):
            for name, df in dataframes.items():
                st.write(f"**{name}**  —  {df.shape[0]} rows × {df.shape[1]} cols")
                st.dataframe(df.head(10), use_container_width=True)

        # ---- Optional: Register DuckDB views for internal SQL (joins, xlsx sheet  control, etc.) ----
        if use_duckdb:
            con = duckdb.connect()
            for name, df in dataframes.items():
                con.register(name, df)
            st.info(
                "DuckDB views registered. The agent can also query via pandas; "
                "use this mode if you plan to join multiple tables or run SQL   internally."
            )

        # ---- Build agent over all DataFrames ----
        llm = ChatOpenAI()
        # You can pass a list of DFs; the agent will know variable names
        dfs_list = list(dataframes.values())

        # Security warning: this agent executes Python via REPL under the hood.
        # Only run for trusted users or within a sandbox!
        agent = create_pandas_dataframe_agent(
            llm,
            dfs_list,
            agent_type="openai-tools",  # good default with tool calling
            verbose=bool(show_code),
        )

        # Helpful system instructions to nudge behavior
        system_nudges = textwrap.dedent(
            """
            You are a careful data analyst.
            - Prefer pandas operations and matplotlib for plots.
            - Always print final numeric answers.
            - If plotting, call matplotlib and ensure a figure is displayed.
            - If multiple DataFrames exist, print which table(s) you used.
            - If something is ambiguous (e.g., which date column), ask a clarifying     question.
        """
        )

        try:
            with st.spinner("Thinking… crunching numbers…"):
                # Provide both the user question and a reminder/nudge
                result = agent.invoke({"input": f"{system_nudges}\n\nQ: {question}"})
        except Exception as e:
            st.error(f"Agent failed: {e}")
            st.stop()

        # ---- Display answer ----
        st.subheader("Answer")
        output = result.get("output") or result
        if isinstance(output, dict):
            st.write(output)
        else:
            st.write(output)

        # If the agent created plots with matplotlib, Streamlit will auto-capture via   pyplot
        st.pyplot(clear_figure=False)
