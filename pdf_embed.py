
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain_qdrant import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os, re
from dotenv import load_dotenv

load_dotenv()

# Load PDF content
loader = PyMuPDFLoader("All_about_ehtasham_toor.pdf")
raw_docs = loader.load()
full_text = "\n".join([doc.page_content for doc in raw_docs])


predefined_headings = [
    "About",
    "Work Experiences",
    "Education",
    "Skills & Technologies Overview",
    "Frontend technologies",
    "Backend technologies",
    "Databases",
    "Tools",
    "Testing Tools",
    "AI SKILLS ",
    "Interests",
    "Other Skills",
    "Projects",
    "Links",
]

escaped_headings = sorted(predefined_headings, key=len, reverse=True)
pattern = "|".join([re.escape(h) for h in escaped_headings])

def split_by_known_headings(text):
    matches = list(re.finditer(pattern, text))
    chunks = []

    for idx, match in enumerate(matches):
        heading = match.group(0).strip()
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        chunks.append(Document(page_content=section_text, metadata={"section": heading.lower().replace(" ", "_")}))
    
    return chunks

tagged_docs = split_by_known_headings(full_text)
print(f"✅ Created {len(tagged_docs)} custom heading-based chunks.")

# Split into chunks
# splits = text_splitter.split_documents(docs)

# print(f"Split the pdf into {len(splits)} sub-documents.")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

if qdrant.collection_exists("agentzero-docs"):
    print("Collection exists. Deleting...")
    qdrant.delete_collection("agentzero-docs")
    
# Create collection (if not already created)
if not qdrant.collection_exists("agentzero-docs"):
    print("Creating collection...")
    qdrant.create_collection(
        collection_name="agentzero-docs",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# embeddings = VertexAIEmbeddings(model="text-embedding-004")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# all-mpnet-base-v2

# Add to Qdrant
vectorstore = Qdrant(
    client=qdrant,
    collection_name="agentzero-docs",
    embeddings=embeddings,
)

vectorstore.add_documents(tagged_docs)
# vectorstore.add_documents(splits)



