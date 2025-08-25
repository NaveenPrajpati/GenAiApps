from typing import Optional
import os
import math

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()


prompt = PromptTemplate.from_template(
    """You are a financial analysis assistant.
        - Always use tables for numbers (e.g., stock prices, market caps, financial ratios).
        - Use bullet points for qualitative insights (e.g., news summaries).
        - Keep responses clear, concise, and structured for analysts.
        """
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [DuckDuckGoSearchRun()]
agent = create_react_agent(llm, tools, prompt)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    result = executor.invoke(
        {
            "input": "Provide a financial analysis of Tesla's stock performance this week."
        }
    )

    print(result["output"])
