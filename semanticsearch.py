from dotenv import load_dotenv
import getpass
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.runnables import chain
from typing import List

file = "./rn-jd.pdf"
loader = PyPDFLoader(file_path=file)
doc = loader.load()

print(len(doc))
print(doc[0].metadata)
print(f"{doc[0].page_content[:200]}\n")
load_dotenv()

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

splited_doc = splitter.split_documents(doc)

# print(len(splited_doc))
# print(f"{splited_doc[0].page_content[:200]}\n")


embeder = OpenAIEmbeddings(model="text-embedding-3-large")

embeded_doc1 = embeder.embed_query(splited_doc[0].page_content)


vector_store = InMemoryVectorStore(embedding=embeder)

ids = vector_store.add_documents(documents=splited_doc)


# print(f"Generated vectors1 of length {len(embeded_doc1)}\n")

query1 = vector_store.similarity_search_with_score(
    "how many years of experience required for this job"
)
query2 = vector_store.similarity_search_by_vector(
    embeder.embed_query("how many years of experience required for this job")
)
doc = query2[0]
print(f" vector search--- {doc}")


@chain
def retriever1(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


result = retriever1.batch(
    ["how many years of experience required for this job", "What is Job role"]
)


retriever2 = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

result2 = retriever2.batch(
    ["how many years of experience required for this job", "What is Job role"]
)

print(f" retriver result-- {result}")
