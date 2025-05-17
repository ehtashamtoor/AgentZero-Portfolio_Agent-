from langchain_qdrant import Qdrant

from qdrant_client import QdrantClient
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()



#  testing the retreiver tool
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),  # e.g. "https://YOUR_PROJECT.qdrant.io"
    api_key=os.getenv("QDRANT_API_KEY")
)

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

vectorstore = Qdrant(
    client=qdrant,
    collection_name="agentzero-docs",
    embeddings=embeddings,
    distance_strategy="cosine",
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {
            "section": "work_experiences"
        }
    }
)



docs = retriever.invoke(input="What are all of his work experiences?")

print("Found %d documents" % len(docs))

for doc in docs:
  print(doc.page_content)
  print("============= chunk ends here =============")