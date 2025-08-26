from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

PDF_URL = "https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"

# 1) Load the PDF (use PyPDFLoader for local files)
loader = WebBaseLoader(PDF_URL)  # requires langchain-community + pypdf
docs = loader.load()

# 2) Split into chunks (good defaults; tweak for your corpus)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""],
)
splits = splitter.split_documents(docs)

# 3) Embed & index with FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4) Prompt + “stuff” doc chain
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful culinary assistant. Use ONLY the provided context to answer.\n"
            "If the answer isn't in the context, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini")  # pick any chat model you have access to
doc_chain = create_stuff_documents_chain(llm, prompt)  # formats docs into {context}

# 5) Retrieval chain (LCEL). Also return the source docs.
rag_chain = (
    RunnablePassthrough.assign(
        # fetch docs and pass them through under the key "context"
        context=lambda x: retriever.invoke(x["question"])
    )
    # run the doc_chain which expects {"context": List[Document], "question": str}
    | doc_chain
)

# Optional: if you want a plain string instead of a Message
final_chain = rag_chain | StrOutputParser()

if __name__ == "__main__":
    question = "Find a simple, easy-to-make Thai recipe."
    answer = final_chain.invoke({"question": question})
    print("\n=== Answer ===\n", answer)

    # If you also want to see sources:
    result_with_sources = (
        RunnablePassthrough.assign(context=lambda x: retriever.invoke(x["question"]))
        | (lambda x: {"answer": doc_chain.invoke(x), "sources": x["context"]})
    ).invoke({"question": question})

    print("\n=== Sources ===")
    for i, d in enumerate(result_with_sources["sources"], 1):
        print(
            f"{i}. {d.metadata.get('source', 'unknown')} p.{d.metadata.get('page', '?')}"
        )
