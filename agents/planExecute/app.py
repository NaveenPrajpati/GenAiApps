import getpass
import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from typing import Union, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display
from dotenv import load_dotenv
import asyncio
from langchain_tavily import TavilySearch

load_dotenv()

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("OPENAI_API_KEY")
# _set_env("TAVILY_API_KEY")


tools = [TavilySearch(max_results=3)]  # Tavily tool via langchain_community

# Use a smaller, fast model for execution and keep stronger models for plan/replan
exec_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = "You are a helpful assistant."
agent_executor = create_react_agent(exec_llm, tools, prompt=prompt)

# -----------------------------
# State definition for LangGraph
# -----------------------------
# input:      original user request
# plan:       list of remaining steps to execute (strings)
# past_steps: running log [(step, result_text)] with an accumulator so each node appends
# response:   final answer string; when set, the graph will terminate


# Define the State
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


# Planning Step
class Plan(BaseModel):
    """Plan to follow in future"""

    # NOTE: Pydantic v2 model; requires langchain-core >= 0.3 per docs
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


# Step 1: Planner — produce a minimal step-by-step plan using structured output
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)


# Re-Plan Step
class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


# Step 3: Re-planner — update remaining steps or return the final response
replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)


# Step 2: Execute the next task in the plan using a ReAct tool-calling agent.
# - Renders the current plan as a numbered list
# - Executes only the first remaining task via the prebuilt agent
# - Appends (task, result_text) to past_steps
# - If the plan is empty, short-circuits with a friendly message
async def execute_step(state: PlanExecute):
    """Step 2: Execute the next task in the plan using a ReAct tool-calling agent.
    - Renders the current plan as a numbered list
    - Executes only the first remaining task via the prebuilt agent
    - Appends (task, result_text) to past_steps
    - If the plan is empty, short-circuits with a friendly message
    """
    plan = state.get("plan", [])
    if not plan:
        return {
            "response": "No steps left to execute. (Planner returned an empty plan.)"
        }

    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:\n{plan_str}\n\nYou are tasked with executing step 1, {task}."""

    # Call the prebuilt ReAct agent (tools + LLM) to perform the step
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )

    # The prebuilt agent returns a `messages` list; take the last AI message content
    result_text = agent_response["messages"][-1].content

    # Accumulate executed step and its result; do not mutate state in place
    return {"past_steps": [(task, result_text)]}


# Small helper: call the planner LLM to generate the initial plan
async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    """Re-plan after each execution:
    - If the model returns a `Response`, we set `response` and end the graph.
    - Otherwise, we overwrite `plan` with only the *remaining* steps to do.
    """
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


# Gate: stop when a final response is present; otherwise loop back to agent
def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


graph = StateGraph(PlanExecute)

# Add the plan node
graph.add_node("planner", plan_step)

# Add the execution step
graph.add_node("agent", execute_step)

# Add a replan node
graph.add_node("replan", replan_step)

# Start → Plan
graph.add_edge(START, "planner")

# Plan → Execute first step
graph.add_edge("planner", "agent")

# Execute → Re-plan
graph.add_edge("agent", "replan")

# Re-plan → (Agent | End)
graph.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# Enable persistence so we can resume conversations and use human-in-the-loop interrupts later
checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

# (Optional) Visualize the graph when running in notebooks; safe to ignore on CLI
try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    pass


config = {
    "recursion_limit": 50,
    "configurable": {"thread_id": "plan-exec-demo"},
}  # thread_id enables persistence
# You can change the input; the loop will plan → execute → re-plan until a final response is set
inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}


async def main():
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    asyncio.run(main())
