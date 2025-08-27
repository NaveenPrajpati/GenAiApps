from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import convert_to_messages
from typing import Literal, List
from langchain_core.messages import convert_to_messages, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


def build_retriever():
    # Load pages
    docs_nested: List[List] = [WebBaseLoader(u).load() for u in urls]
    docs = [d for batch in docs_nested for d in batch]

    # Split into semantically coherent chunks
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=150
    )
    splits = splitter.split_documents(docs)

    # Build vector store (you can switch to FAISS/Chroma in prod)
    embeddings = OpenAIEmbeddings()  # set OPENAI_API_KEY in env
    vs = InMemoryVectorStore.from_documents(splits, embedding=embeddings)

    # Return a LC retriever
    return vs.as_retriever(search_kwargs={"k": 6})


retriever = build_retriever()

# Wrap as a tool the model can call
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    description=(
        "Search and return relevant context chunks from Lilian Weng's blog. "
        "Use this when the user asks anything about her posts or concepts discussed there."
    ),
)

retriever_tool.invoke({"query": "types of reward hacking"})


# -----------------------------
# 2) Models
# -----------------------------
# Keep model configurable; default here, but you can override via config.
# See docs for dynamic switching. (You can also use "gpt-4.1-mini" etc.)
response_model = ChatGoogleGenerativeAI()


# Grader for retrieval relevance (structured output)
class GradeDocuments(BaseModel):
    """Binary relevance for retrieved context vs user question."""

    binary_score: str = Field(description="Use 'yes' if relevant, else 'no'.")


grader = ChatGoogleGenerativeAI()


# -----------------------------
# 3) Nodes (graph functions)
# -----------------------------
def generate_query_or_respond(state: MessagesState):
    """
    Let the LLM decide: call the retriever tool or answer directly.
    We bind the retriever tool; if tools are requested, ToolNode will run them.
    """
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# After ToolNode runs, the last message(s) will be ToolMessages with retrieved text.
# We grade whether that retrieved text is relevant to the user’s question.
def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    messages = state["messages"]
    # Find the latest user message as the question
    question = None
    for m in reversed(messages):
        if m.type == "human":
            question = m.content
            break
    question = question or messages[0].content

    # Find latest tool message (the retrieval output). If none, ask for rewrite.
    tool_context = None
    for m in reversed(messages):
        if m.type == "tool":
            tool_context = m.content
            break
    if not tool_context:
        return "rewrite_question"

    prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        f"Here is the retrieved document:\n\n{tool_context}\n\n"
        f"Here is the user question: {question}\n"
        "If the document contains keyword(s) or semantic meaning related to the user question, "
        "grade it as relevant.\n"
        "Answer with a single token: 'yes' or 'no'."
    )

    result = grader.with_structured_output(GradeDocuments).invoke(
        [HumanMessage(content=prompt)]
    )
    return (
        "generate_answer"
        if result.binary_score.strip().lower() == "yes"
        else "rewrite_question"
    )


def rewrite_question(state: MessagesState):
    """
    Rewrite the original user question to improve retrieval (query expansion / clarification).
    """
    # Pull the latest user query
    messages = state["messages"]
    user_q = None
    for m in reversed(messages):
        if m.type == "human":
            user_q = m.content
            break
    user_q = user_q or messages[0].content

    prompt = (
        "Look at the input and reason about the underlying semantic intent.\n"
        "Original question:\n-------\n"
        f"{user_q}\n"
        "-------\n"
        "Reformulate a better search query (concise, keywords + entities):"
    )
    reformulated = response_model.invoke([HumanMessage(content=prompt)]).content
    # Re-insert as a new user message so the loop can try tools again
    return {
        "messages": convert_to_messages([{"role": "user", "content": reformulated}])
    }


def generate_answer(state: MessagesState):
    """
    Generate a concise answer grounded in retrieved context.
    """
    messages = state["messages"]
    # recover user question
    question = None
    for m in reversed(messages):
        if m.type == "human":
            question = m.content
            break
    question = question or messages[0].content

    # recover latest tool context
    context = ""
    for m in reversed(messages):
        if m.type == "tool":
            context = m.content
            break

    prompt = (
        "You are an assistant for question-answering tasks.\n"
        "Use the retrieved context to answer the question.\n"
        "If the answer isn't in the context, say you don't know.\n"
        "Limit to ~3 sentences.\n\n"
        f"Question: {question}\n"
        f"Context:\n{context}"
    )
    answer = response_model.invoke([HumanMessage(content=prompt)])
    return {"messages": [answer]}


# -----------------------------
# 4) Graph wiring
# -----------------------------
graph = StateGraph(MessagesState)

graph.add_node("generate_query_or_respond", generate_query_or_respond)
graph.add_node("retrieve", ToolNode([retriever_tool]))
graph.add_node("rewrite_question", rewrite_question)
graph.add_node("generate_answer", generate_answer)

graph.add_edge(START, "generate_query_or_respond")

# Route based on whether the model decided to call tools
graph.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,  # prebuilt condition that routes when tools were requested
    {
        "tools": "retrieve",  # go run tools
        END: END,  # otherwise, if no tools were requested, finish with the LLM reply
    },
)


# After retrieval, decide: good enough? => answer, else rewrite and try again
graph.add_conditional_edges("retrieve", grade_documents)
graph.add_edge("generate_answer", END)
graph.add_edge("rewrite_question", "generate_query_or_respond")

# No persistence:
app = graph.compile()


# Quick smoke test
print("--- Direct greeting (no retrieval) ---")
for event in app.stream({"messages": [{"role": "user", "content": "hello!"}]}):
    for node, update in event.items():
        print("node:", node)


print("\n--- RAG question ---")
for event in app.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What types of reward hacking does Lilian Weng describe?",
            }
        ]
    }
):
    for node, update in event.items():
        last_msg = update["messages"][-1]
        print("node:", node, "|", getattr(last_msg, "type", ""))
        # print bodies if you want:
        print(getattr(last_msg, "content", ""))
