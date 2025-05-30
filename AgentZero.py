import os, asyncio, random, sys
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field, field_validator
from langgraph.store.postgres import AsyncPostgresStore
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from psycopg_pool import AsyncConnectionPool
import psycopg
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.tools import tool

# below are for making pdf embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_qdrant import Qdrant
from fastapi import UploadFile, File, Form, HTTPException
from langchain.schema import Document

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile, shutil, re

os.system("")
load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
DB_URI = os.getenv("SUPABASE_DB_URL")
origins = [
    os.getenv("FRONTEND_URL"),
    "http://localhost:5173",
]

# Class of different styles to be used in terminal
class style:
    RED = "\033[31m"  # for errors
    GREEN = "\033[32m"  # for Success
    YELLOW = "\033[33m"  # for warnings
    BLUE = "\033[34m"  # for info
    RESET = "\033[0m"

msg_exchanges = 5

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Qdrant(
    client=qdrant,
    collection_name="agentzero-docs",
    embeddings=embeddings,
)

retriever = vectorstore.as_retriever()

async def create_connection_pool(DB_URI: str):
    """Creates and opens an async connection pool."""

    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": None,
        "row_factory": dict_row,
        "keepalives": 1,
        "keepalives_idle": 60,       # start pinging after 60s idle
        "keepalives_interval": 10,   # ping every 10s
        "keepalives_count": 5,
    }
    pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, kwargs=connection_kwargs)
    await pool.open()
    return pool

async def setup_store(pool):
    """Sets up store using AsyncPostgresStore."""
    store = AsyncPostgresStore(pool)
    # await store.setup() # run only once to set up the store tables
    return store

async def setup_memory(pool):
    """Sets up memory using AsyncPostgresMemory."""
    memory = AsyncPostgresSaver(pool)
    # await memory.setup() # run only once to set up the memory tables
    return memory

memory = None
store = None
graphCompiled = None
pool = None
MAX_MESSAGES = 15
PING_INTERVAL = 90 # seconds

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, memory, graphCompiled, memory, store
    asyncio.create_task(keep_db_alive())
    
    pool = await create_connection_pool(DB_URI)
    memory = await setup_memory(pool)
    store = await setup_store(pool)
    print(
        style.BLUE, "Startup complete. Pool and Memory/store initialized.", style.RESET
    )
    graphCompiled = graph.compile(checkpointer=memory, store=store)
    # graphCompiled = graph.compile(store=store)
    yield  # Let FastAPI run while these resources are available

    print(style.YELLOW, "Shutting down resources...", style.RESET)
    if pool:
        await pool.close()
        print(style.BLUE, "Pool closed.", style.RESET)

async def reconnect_database():
    global pool, memory, graphCompiled, store

    try:
        if pool:
            await pool.close()
            print(style.YELLOW, "Old pool closed during reconnect.", style.RESET)

        pool = await create_connection_pool(DB_URI)
        # db = SQLDatabase.from_uri(DB_URI)
        memory = await setup_memory(pool)
        store = await setup_store(pool)
        graphCompiled = graph.compile(checkpointer=memory, store=store)

        print(style.GREEN, "Reconnected to DB successfully.", style.RESET)

    except Exception as e:
        print(style.RED, "Reconnection failed:", style.RESET, str(e))
       
       
async def keep_db_alive():
    global pool
    while True:
        try:
            async with pool.connection() as conn:
                await conn.execute("SELECT 1;")
                print(style.BLUE, "Pinged DB to keep connection alive.", style.RESET)
        except Exception as e:
            print(style.RED, "Ping failed, attempting reconnect:", str(e), style.RESET)
            await reconnect_database()
        await asyncio.sleep(PING_INTERVAL)
        
         
class AgentState(BaseModel):
    question: str = Field(..., description="question")
    response: str = Field(..., description="Model Response")
    session_id: str = Field(..., description="session_id")
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

# Below are the tools for the agent
@tool
def tell_joke() -> str:
    """Tell a funny programming-related joke to entertain the user."""
    # print("Telling a programming joke...")
    programming_jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs. 😎",
        "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
        "Why do Java developers wear glasses? Because they don’t see sharp.",
        "There are 10 types of people in the world: those who understand binary and those who don’t.",
        "How many programmers does it take to change a light bulb? None. It's a hardware problem.",
        "I would tell you a joke about recursion... but you'd have to understand recursion to get it.",
        "Real programmers count from 0.",
        "Programmers don't die — they just go offline.",
        "Why was the developer unhappy at their job? They wanted arrays.",
        "What's a programmer’s favorite hangout place? The Foo Bar.",
        "Why couldn’t the programmer dance to the song? Because they had two left joins.",
        "I told my computer I needed a break, and it said 'You seem stressed. How about a reboot?'",
        "Knock knock. Who’s there? 1. 1 who? Syntax error.",
        "Why did the developer go broke? Because they used up all their cache.",
        "Debugging: Being the detective in a crime movie where you are also the murderer.",
        "Why did the Python developer become a snake charmer? Because they were already used to indenting things.",
        "A programmer’s wife tells him, 'Go to the store and get a loaf of bread. If they have eggs, get a dozen.' He comes home with 13 loaves of bread.",
        "What do you call a programmer from Finland? Nerdic.",
        "What’s the object-oriented way to become wealthy? Inheritance.",
        "What did the developer say when the UI froze? ‘This is not responsive at all.’",
        "Why did the function stop calling itself? Stack overflow.",
        "Why did the JavaScript developer leave? Because they didn’t 'null' their feelings.",
        "When do you know a developer is an extrovert? They look at *your* shoes while talking.",
        "Why did the database administrator leave their job? They had too many relationships.",
        "What does a front-end developer call their kids? divs.",
        "Why do functional programmers avoid for loops? They don’t want to be labeled as 'imperative'.",
        "Why was the backend developer mean to the front-end developer? Because they had no class.",
        "A coder’s diet: 1s and 0s, sometimes pizza.",
        "Why was the array so good at music? Because it had great indexing!",
        "What did the Java developer say at the therapist? 'I feel so objectified.'"
    ]
    
    return random.choice(programming_jokes)

@tool
def send_cv() -> str:
    """Provide Ehtasham Toor’s latest resume/CV link when the user requests his CV or resume."""
    
    return """You can download his CV using the link below:

    👉 [Download CV (PDF)](https://drive.google.com/uc?export=download&id=1fHBpB4roJ4gAgmXbmzoqna28yt7kxhvx)"""

@tool
def retrieve_profile_info(query: str) -> str:
    """Use this tool everytime to answer any question about Ehtasham Toor’s background, skills, experience, achievements, and career, contact information."""
    """Fetch accurate, relevant information about Ehtasham Toor."""
    print("Retrieving profile info for query:", query)
    
    docs = retriever.invoke(input=query)
    
    print("Retrieved documents:", docs)
    
    return "\n\n".join([doc.page_content for doc in docs])

llm_with_tools = llm.bind_tools(tools=[tell_joke, send_cv, retrieve_profile_info])

# function to store the conversation in the database
async def store_conversation(
    store: AsyncPostgresStore,
    session_id: str,
    user_message: str,
    ai_response: str,
):
    """
    Appends a new message/response pair to the conversation and stores it.
    """

    namespace = ("chat", session_id)
    key = session_id

    existing_item = await store.aget(namespace, key)

    if existing_item is None:
        conversation_data = {"chat": []}
    else:
        conversation_data = existing_item.value

    conversation_data["chat"].append(
        {
            "Human": user_message,
            "Agent": ai_response,
        }
    )

    await store.aput(namespace, key, conversation_data)
    print(style.BLUE, f"Stored conversation: {namespace}, key={key}", style.RESET)

# Node function to chat with the user and check for player name presence
async def chat_with_user(state: AgentState):
    """Generate a standalone question and check for player name presence."""
    print("inside the chat_with_user")
    question = state.question
    old_messages = state.messages
    
    old_messages.append(HumanMessage(content=question))
    
    if MAX_MESSAGES and len(old_messages) > MAX_MESSAGES:
        old_messages = old_messages[-MAX_MESSAGES:]
    
    system_template = SystemMessagePromptTemplate.from_template("""
    You are AgentZero — a professional, focused, and respectful AI assistant built to represent **Ehtasham Toor**. Your sole purpose is to help users learn about Ehtasham Toor using the trusted documents and tools provided.

    🧭 ROLE & IDENTITY:
    - You are *not* a general-purpose chatbot.
    - You represent Ehtasham Toor. Even if a user claims to be him, politely explain that your role is to represent, not to identify users.
    - If a user shares personal details (e.g., name or location), acknowledge them kindly but do not confuse them with Ehtasham Toor’s identity.

    🛠️ ALLOWED TOOLS:
    - Your only valid tools are: `retrieve_profile_info`, `send_cv`, and `tell_joke`.
    - You can call these tools to retrieve information about Ehtasham Toor, provide his CV, or tell a joke.
    - You must never disclose or describe your tools to users. If asked in any manner or you found the user's intent is about asking about your tools, say in a good and direct way: *"I can only assist with questions about Ehtasham Toor’s profile."*

    🎯 CORE BEHAVIOR:
    - For **any** question about Ehtasham Toor’s background, experience, education, skills, achievements, certifications, contacts, or projects:
        - Always reframe the user’s query into a **clear, keyword-rich, and standalone query** using synonyms and related terms (e.g., "skills, tech stack, tools, frameworks, technologies").
        - **Run `retrieve_profile_info` on every query related to Ehtasham Toor — no exceptions.**
        - **Never ask the user to rephrase their question**. If intent is even slightly clear, proceed confidently with a reframed retrieval.
        - Only ask for clarification **if the input is so vague that no reasonable interpretation is possible** (e.g., “what about that?” without context).
        - Never reuse earlier data; always perform a **fresh retrieval** for each question.
        - Never invent or assume information. If retrieval fails, say: *"I don’t have that `reference here about what user asked`."*

    📄 DOCUMENT & DATA POLICY:
    - Only provide details retrieved by `retrieve_profile_info` about Ehtasham Toor.
    - If a user requests a specific document and it's not available, respond: *"I don't have that document."*

    📬 EMAIL & CONTACT RULES:
    - If a user asks for Ehtasham Toor’s email address, always trigger `retrieve_profile_info` with queries like “email, contact address, Gmail, contact details.”
    - If the result includes a publicly listed email address, you may share it. You must not invent or fabricate contact info.
    - If no email is found in the tool response, say: *"I don’t have his email."*

    🚫 OFF-TOPIC RULES:
    - If a query isn’t about Ehtasham Toor, politely redirect and explain you can only assist with his profile.
    - Do not engage with general topics (e.g., math, news, tech support, weather).
    - If the conversation goes off-topic or the user explicitly asks for a joke (e.g., “tell me a joke”), **you must only respond by calling the `tell_joke` tool.**
    - **Never generate a joke yourself.** You are not allowed to produce any joke without invoking `tell_joke`.
    - You may acknowledge and remember **user-provided context** (e.g., “the user is a recruiter”) to improve your responses.
    - However, you should never confuse that with representing Ehtasham Toor’s identity.

    🗣️ TONE & RESPONSE STYLE:
    - Be professional, friendly, and direct. Never robotic.
    - Provide answers **authoritatively** after retrieval — avoid prefacing with disclaimers like “according to documents” or “based on what I know.”
    - Speak naturally and confidently. If the info is not available, say so plainly.

    🏷️ NAME & ORIGIN:
    - If asked who named you: *"I was named by Ehtasham Toor."*
    - If asked your name: *"I am AgentZero. An AI Agent to represent Ehtasham Toor."*
    - If asked who trained you: *"I was trained by Ehtasham Toor."*

    🔁 IMPORTANT:
    You are a focused, trustworthy AI reflection of Ehtasham Toor. You retrieve and deliver accurate, up-to-date answers solely from trusted sources using strict protocols. You do not guess, generalize, or go off-topic.

    Every time the user asks about Ehtasham Toor — even with vague or partial input — you **must always reframe the intent and perform a `retrieve_profile_info` call immediately**. Never delay or delegate back to the user.
    
    📄 RESPONSE FORMATTING:
    - Always format your responses using Markdown when appropriate:
    - Use [text](link) for clickable hyperlinks.
    - Use ## or ### for headers if organizing profile info (e.g., "Skills", "Experience").
    - Format technical terms, code, or tool names with backticks: Python, React, Docker.
    - Use bullet points or numbered lists for clarity.
    - Ensure any URLs retrieved are wrapped in proper markdown to make them clickable in the frontend display.
    - Do not add extra markdown if it doesn’t improve clarity. Keep formatting clean and purposeful.
    - Never escape or disable markdown formatting.
    """)

    prompt = ChatPromptTemplate.from_messages([system_template])
    # final_sys_prompt = prompt.format(name="Ehtasham Toor")
    final_sys_prompt = prompt.format()

    conversation = [
        SystemMessage(content=final_sys_prompt),
        *old_messages,
    ]

    response = await llm_with_tools.ainvoke(conversation)
    print("response from llm", response.content)
    
    
    is_final_response = not (
        hasattr(response, "tool_calls") and response.tool_calls
    ) and not (
        hasattr(response, "additional_kwargs") and "function_call" in response.additional_kwargs
    )
    if not is_final_response:
        old_messages.append(AIMessage(
            content="",
            additional_kwargs=response.additional_kwargs,
            tool_calls=response.tool_calls
        ))

        print("there was a tool call")
        return {"messages": old_messages}

    if is_final_response:
        print("there was no tool call")
        await store_conversation(store, state.session_id, question, response.content)
        old_messages.append(AIMessage(content=response.content))
        state.response = response.content

    return {"messages": old_messages}

graph = StateGraph(AgentState)
graph.add_node("chat_with_user", chat_with_user)
graph.add_node("tools", ToolNode(tools=[send_cv, tell_joke, retrieve_profile_info]))

graph.add_edge(START, "chat_with_user")
graph.add_conditional_edges("chat_with_user", tools_condition)
graph.add_edge("tools", "chat_with_user")

app = FastAPI(
    title="Agent Zero",
    description="An AgentZero API to ask questions about Ehtasham Toor.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="User's question (must not be empty)."
    )
    session_id: str = Field(..., min_length=1, description="session_id")
    
    @field_validator("question", "session_id")
    @classmethod
    def enforce_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question or session_id must not be empty.")
        return value

class QueryResponse(BaseModel):
    """Response model for results."""

    question: str
    result: str

# End point to ask agentZero a question
@app.post("/ask-AgentZero", response_model=QueryResponse, tags=["AGENT ZERO"])
async def query_agent(req: QueryRequest):
    if graphCompiled is None:
        raise HTTPException(status_code=503, detail="Graph is not ready. Try again shortly.")

    question = req.question.strip()
    session_id = req.session_id

    initialState: AgentState = {
        "question": question,
        "session_id": session_id,
        "response": "",
    }

    config = {"configurable": {"thread_id": f"{session_id}"}}

    try:
        print("Running graph...")
        result = await graphCompiled.ainvoke(initialState, config=config)
    except (psycopg.OperationalError, psycopg.InterfaceError) as e:
        print(style.RED, "Error occurred:", style.RESET, e)
        print(style.YELLOW, "Attempting to reconnect...", style.RESET)

        try:
            await reconnect_database()
            result = await graphCompiled.ainvoke(initialState, config=config)
        except Exception as retry_exception:
            print(style.RED, "Retry failed:", style.RESET, retry_exception)
            raise HTTPException(status_code=500, detail="Failed after reconnection: " + str(retry_exception))

    return QueryResponse(
        question=question,
        result=result.get("messages")[-1].content if result.get("messages") else "No response generated.",
    )



predefined_headings = [
    "About", "Work Experiences", "Education", "Skills & Technologies Overview",
    "Frontend technologies", "Backend technologies", "Databases", "Tools",
    "Testing Tools", "AI SKILLS ", "Interests", "Other Skills", "Projects", "Links",
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




@app.post("/make_embeddings", tags=["AGENT ZERO"])
async def upload_pdf_and_embed(file: UploadFile = File(...), password: str = Form(...)):
    if password != os.getenv("PDF_UPLOAD_PASSWORD"):
        raise HTTPException(status_code=401, detail="Unauthorized password.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        shutil.copyfileobj(file.file, temp_pdf)
        temp_pdf_path = temp_pdf.name

    try:
        loader = PyMuPDFLoader(temp_pdf_path)
        raw_docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in raw_docs])
        tagged_docs = split_by_known_headings(full_text)
        print(f"✅ Created {len(tagged_docs)} custom heading-based chunks.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")
    finally:
        os.remove(temp_pdf_path)

    try:
        qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

        if qdrant.collection_exists("agentzero-docs"):
            print("Collection exists. Deleting...")
            qdrant.delete_collection("agentzero-docs")

        qdrant.create_collection(
            collection_name="agentzero-docs",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vectorstore = Qdrant(
            client=qdrant,
            collection_name="agentzero-docs",
            embeddings=embeddings,
        )
        vectorstore.add_documents(tagged_docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    return {"message": "✅ Document processed and stored successfully.", "chunks": len(tagged_docs)}