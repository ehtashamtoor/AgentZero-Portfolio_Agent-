# from langchain_google_vertexai import VertexAIEmbeddings
from langchain_qdrant.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_core.tools import create_retriever_tool
from dotenv import load_dotenv

load_dotenv()

#  testing the retreiver tool
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

resume_vectorstore = QdrantVectorStore(
    embedding=embeddings,
    client=qdrant,
    collection_name="resume_docs"
)

resume_retriever = resume_vectorstore.as_retriever()


docs = resume_retriever.invoke("List all job titles and companies he has worked at with duration.")
# result = "\n\n".join([doc.page_content for doc in docs])

print("result\n", docs)

# vectorstore = Qdrant(
#     client=qdrant,
#     collection_name="agentzero-docs",
#     embeddings=embeddings,
#     distance_strategy="cosine",
# )

# retriever = vectorstore.as_retriever(
#     search_kwargs={
#         "k": 5,
#         "filter": {
#             "section": "work_experiences"
#         }
#     }
# )

# docs = retriever.invoke(input="What are all of his work experiences?")

# print("Found %d documents" % len(docs))

# for doc in docs:
#   print(doc.page_content)
#   print("============= chunk ends here =============")