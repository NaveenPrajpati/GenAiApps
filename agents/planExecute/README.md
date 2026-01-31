# Plan–Execute Agent (LangGraph + LangChain)

> A minimal, production‑ready Plan→Execute→Re‑plan agent built with **LangGraph** and **LangChain**, using **Tavily** for web search and **OpenAI** models for planning and execution. Includes short‑term persistence via LangGraph checkpointers and streaming.

---

## Why

Natural language questions often require **planning** (break a goal into steps), **tool use** (search, fetch, compute), and **adaptation** (re‑plan based on results). Traditional single‑prompt solutions struggle with multi‑step tasks and transparency.

This repo demonstrates a **Plan–Execute** pattern that:

- **Separates concerns**: one model plans, a different (faster) model executes via tools.
- **Loops with feedback**: after each step, we re‑plan based on what was learned.
- **Persists state**: a checkpointer preserves conversation/thread context (useful for HITL and restarts).
- **Streams events**: see every state update as it flows through the graph.

Use it as a starter for production agent workflows that need reliability, cost control, and clarity.

---

## What

### Features

- **Planner (Step 1):** Generates a minimal, ordered step list via structured output (`Plan` pydantic model).
- **Executor (Step 2):** Runs the **next** step with a prebuilt ReAct agent (`create_react_agent`) wired to **Tavily** search.
- **Re‑planner (Step 3):** Either finalizes a response or updates the remaining plan (`Act` with `Response | Plan`).
- **State machine:** Implemented with **LangGraph** (`StateGraph`) and type‑annotated `TypedDict` state.
- **Persistence:** `InMemorySaver` checkpointer + `thread_id` in `config.configurable`.
- **Streaming:** `app.astream(...)` yields incremental events for observability.
- **Batteries included docs/comments:** The code is heavily commented line‑by‑line.

### Tech stack

- **LangGraph**: State graphs, nodes, edges, persistence, streaming
- **LangChain**: LLMs, prompts, tools (Tavily), structured output
- **OpenAI**: `gpt-4o` (planner/re‑planner), `gpt-4o-mini` (executor)
- **Tavily**: Web search tool via `langchain_community`

---

## How

### 1) Install

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U \
  langgraph \
  langchain-openai \
  langchain-community \
  tavily-python
```

> If you plan to visualize the graph in notebooks, also: `pip install ipython`.

### 2) Set environment variables

The script prompts if they’re missing, but it’s cleaner to export them:

```bash
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
```

### 3) Run

```bash
python agents/planExecute/app.py
```

What you’ll see:

- A **plan** (list of steps)
- One or more executed **past\_steps**
- Either an updated **plan** (loop continues) or a final **response** (graph ends)

### 4) Change the input

In `app.py` near the bottom:

```python
inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}
```

Replace the question with your own. The graph will plan → execute → re‑plan until a final `response` is set.

---

## Architecture

```text
┌────────┐   Start    ┌──────────┐   Plan →    ┌──────────┐   Execute →   ┌──────────┐
│ START  │──────────▶│ Planner  │────────────▶│  Agent   │──────────────▶│ Re‑plan  │
└────────┘            └──────────┘             └──────────┘                └──────────┘
                                                                ┌─────────────────────┐
                                                                │  should_end?       │
                                                                │  yes → END         │
                                                                │  no  → Agent again │
                                                                └─────────────────────┘
```

- **Planner**: Produces a minimal step list (`Plan.steps`).
- **Agent**: Executes only the **first** remaining step using ReAct + tools.
- **Re‑plan**: Returns either a final `Response` or the **remaining** steps.

---

## State schema

```python
class PlanExecute(TypedDict):
    input: str             # Original user request
    plan: List[str]        # Steps still to execute (strings)
    past_steps: Annotated[List[Tuple], operator.add]  # Log of (step, result)
    response: str          # Final answer; when set, graph ends
```

> `Annotated[..., operator.add]` makes `past_steps` an **accumulator** — each node appends without mutating state in place.

---

## Models & Tools

- **Planner / Re‑planner**: `ChatOpenAI(model="gpt-4o", temperature=0)` with `.with_structured_output(...)` for `Plan` and `Act` models.
- **Executor (Agent)**: `ChatOpenAI(model="gpt-4o-mini", temperature=0)` via `create_react_agent(exec_llm, tools, prompt=...)`.
- **Tools**: `TavilySearchResults(max_results=3)` from `langchain_community.tools.tavily_search`.

Why this split? Planning benefits from stronger reasoning (`gpt-4o`), while execution is more frequent → use a **faster, cheaper** model (`gpt-4o-mini`).

---

## Persistence & Streaming

- **Persistence**: `checkpointer = InMemorySaver()` and `app = workflow.compile(checkpointer=checkpointer)`.
- **Threading**: Pass a stable thread id to group events:

```python
config = {
  "recursion_limit": 50,
  "configurable": {"thread_id": "plan-exec-demo"}
}
```

- **Streaming**: Iterate `async for event in app.astream(inputs, config=config): ...` to receive state deltas in real time.

---

## Extending

1. **Human‑in‑the‑loop (HITL)**

   - With a checkpointer in place, you can add `interrupt_before=["agent"]` or programmatic `interrupt()` calls to pause before tool use.

2. **Typed final answers**

   - If you need structured JSON at the end, define a `pydantic` schema and pass `response_format=YourSchema` to `create_react_agent(...)`; read `response["structured_response"]`.

3. **More tools**

   - Add calculators, code interpreters, vector search retrievers, custom APIs, etc., to the agent’s tool list.

4. **Observability**

   - In notebooks, the code attempts to render a Mermaid graph with `app.get_graph(xray=True).draw_mermaid_png()`.

---

## Troubleshooting

- **Import errors**
  - Ensure you’re using up‑to‑date packages: `langgraph`, `langchain-openai`, `langchain-community`, `tavily-python`.
- **Auth issues**
  - Confirm `OPENAI_API_KEY` and `TAVILY_API_KEY` are set in your shell.
- **No steps left to execute**
  - The planner may return an empty plan for extremely simple/ambiguous inputs. Try a more specific prompt.
- **Too many loops**
  - Reduce `recursion_limit` or add guardrails in `replan_step` (e.g., max iterations or confidence checks).

---

## File layout

```
agents/
└─ planExecute/
   └─ app.py       # Main graph, tools, models, and streaming runner
```

---

## License

MIT (or adapt to your project’s license).

---

## Acknowledgements

- LangGraph team for the Plan‑and‑Execute pattern and prebuilt ReAct agent
- LangChain community tools (Tavily)
- OpenAI models powering planning and execution

