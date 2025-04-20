import os, asyncio, datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from pydantic import BaseModel, Field
from langgraph.store.postgres import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

os.system("")
load_dotenv()

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

class AgentState(BaseModel):
    # question: str = Field(..., description="User's question")
    response: str = Field(..., description="Model Response")
    session_id: str = Field(..., description="session_id")
    messages: Annotated[list[BaseMessage], add_messages] = []
    
# @tool
def tell_joke() -> str:
    """Tell a funny programming-related joke to entertain the user."""
    return "Why do programmers prefer dark mode? Because light attracts bugs. 😎"

# @tool
def send_cv() -> str:
    """Provide Ehtasham Toor’s latest resume link when the user requests his CV or resume."""
    return "Here’s Ehtasham Toor’s CV: [Download CV](https://yourdomain.com/cv.pdf)"

def add_two_numbers(a:int, b :int) -> int:
    """ Add a and b
    Args: 
        a (int): first number
        b (int): second number
    Returns:
        int: sum of a and b
    """
    return a + b


llm_with_tools = llm.bind_tools([tell_joke, send_cv, add_two_numbers])

async def create_connection_pool(DB_URI: str):
    """Creates and opens an async connection pool."""

    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    }
    pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, kwargs=connection_kwargs)
    await pool.open()
    return pool

    # async with AsyncConnectionPool(
    #     conninfo=DB_URI,
    #     max_size=20,
    #     kwargs=connection_kwargs,
    # ) as pool:
    #     store = AsyncPostgresStore(pool)
    #     # await store.setup()
    #     return store

async def setup_memory(pool):
    """Sets up memory using AsyncPostgresSaver."""
    memory = AsyncPostgresStore(pool)
    # await memory.setup()
    return memory

async def store_conversation(
    store: AsyncPostgresStore,
    user_id: str,
    conversation_id: str,
    user_message: str,
    ai_response: str,
    game: str,
    userType: str,
):
    """
    Appends a new message/response pair to the conversation and stores it.
    """

    namespace = ("chat", user_id)
    key = conversation_id

    existing_item = await store.aget(namespace, key)

    if existing_item is None:
        conversation_data = {"chat": []}
    else:
        conversation_data = existing_item.value

    conversation_data["chat"].append(
        {
            "Human": user_message,
            "AI": ai_response,
        }
    )

    conversation_data["game"] = game
    conversation_data["userType"] = userType

    await store.aput(namespace, key, conversation_data)
    print(style.BLUE, f"Stored conversation: {namespace}, key={key}", style.RESET)

    # await summarization_node()

async def summarize_messages(messages):
    """
    Summarizes the provided messages while maintaining context for follow-up questions.
    """

    conversation_text = "\n".join(
        [f"Human: {msg['Human']}\nAI: {msg['AI']}" for msg in messages]
    )

    prompt = PromptTemplate.from_template(
        "Summarize the following conversation while preserving key details and making it concise:\n\n"
        "{conversation}\n\n"
        "Rules:\n"
        "1. Do not rephrase facts or add new details.\n"
        "2. Keep the summary in a structured 'Human: ... AI: ...' format.\n"
        "3. Preserve entities like player names, seasons, and key numbers.\n"
        "4. If a follow-up question refers to a previous entity, maintain that reference.\n\n"
        "Summary:"
    )

    summary = await llm.ainvoke(prompt.format(conversation=conversation_text))

    print(style.BLUE, "===============Summary made============== ", style.RESET)
    return summary.content

# Node function to summarize the messages
async def summarization_node(state: AgentState):
    """
    Summarizes messages every 5 exchanges and updates summary tracking.
    """
    print("into the summarization node ===========================================")
    conversation_id = state.configId
    user_id = state.userId

    namespace = ("chat", user_id)
    key = conversation_id

    existing_item = await store.aget(namespace, key)

    if existing_item is None:
        print("no item found")
        return state  # No conversation exists for now

    print(
        style.BLUE,
        "Item found.. messages length",
        style.RESET,
        len(existing_item.value["chat"]),
    )

    conversation_data = existing_item.value
    total_messages = len(conversation_data["chat"])
    summary_made = conversation_data.get("summaryMade", 0)

    print(
        style.BLUE,
        "total_messages N summary_node==================",
        style.RESET,
        total_messages,
        summary_made + msg_exchanges,
    )

    # If we haven't summarized yet, or more messages have arrived
    if total_messages >= summary_made + msg_exchanges:

        print("====== going to summarize")
        new_messages_to_summarize = conversation_data["chat"][
            summary_made : summary_made + msg_exchanges
        ]
        new_summary = await summarize_messages(new_messages_to_summarize)

        # Update the stored summary
        previous_summary = conversation_data.get("summary", "")
        updated_summary = (
            f"{previous_summary}\n\n{new_summary}" if previous_summary else new_summary
        )

        # Update storage
        conversation_data["summary"] = updated_summary
        conversation_data["summaryMade"] = summary_made + msg_exchanges

        await store.aput(namespace, key, conversation_data)

        print(
            f"✅ Updated summary for {namespace}, key={key}, summaryMade={summary_made + msg_exchanges}"
        )

    return state

    # system = """
    #     You are a chat assistant expert that determines whether a user's question is relevant to the {sport} database.
    #     Your goal is to classify the question, extract key details, and refine it into a standalone query.
    #     You will also use previous chat history to check if the question is a follow-up before forming the standalone question.

    #     Instructions: Follow these strict rules:
    #     4. Handle abbreviations, nicknames, and partial names for players and teams:
    #      - Recognize common nicknames and abbreviations for players.
    #      - Recognize common abbreviations and short forms for teams.
    #      - If a name is ambiguous, assume the most well-known player or team in the context of

    #     ## Step 1: Verify Sport and Database Relevance
    #     - The user's question must be related to {sport}.
    #     - Check if the query is relevant to the database schema:

    #     **Schema Information:**
    #     {schema_info}

    #     - If the question is NOT related to {sport} or the database, mark `"isRelevant": false`.

    #     ## Step 2: Use Context for Reference Resolution
    #     The user may refer to past queries using words like "he," "they," "last time," etc.
    #     Use the **conversation history** to resolve ambiguous references:

    #     If a user’s question lacks details (e.g., “What about Player X?”), use the context to infer the missing information.

    #     ## Step 3: Classify the Query
    #     1. **General Query:**
    #     - Greetings: “Hi”, “Hello”
    #     - Out-of-context: “What’s the weather?”
    #     - Random definitions: “What is {sport}?”

    #     2. **Database Query:**
    #     - Asking for statistics: “Player X goals in 2022”
    #     - Mentioning specific data points

    #     3. **Comparison Query:**
    #     - Includes: “compare”, “vs”, “versus”
    #     - Explicitly mentions two or more entities

    #     ## Step 4: Extract Key Details
    #     - **Players:** Extract player names. Return `[]` if none are mentioned.
    #     - **Year:** Identify the year.
    #     - If **not mentioned** but can be inferred from context, **use the last mentioned year/season**.
    #     - Default to `2023` if no other information is available.

    #     ## Step 5: Handle Follow-Up Queries with Context Awareness
    #     - If the previous query was about a **league-wide leader** (e.g., "Who had the most points in 2021?"),
    #     and the new question asks about a different stat (e.g., "What about assists?"),
    #     **do not assume the user is asking about the same player.**
    #     - Instead, maintain the original question structure and **apply it to the new stat**.
    #     -  If the previous query was about a **specific statistic** (e.g., "How many three-pointers did Player X make in 2021-22?"),
    #     and the new question refers to another player (e.g., "What about Player Y?"),
    #     **assume the user is referring to the same statistic and season.**

    #     - If the previous query was about a **specific player** (e.g., "How many games did Player X play?"),
    #     and the new question asks about a different stat (e.g., "What about assists?"),
    #     **assume the user is referring to the same player.**

    #     ### Example Context Handling:
    #     **Human:** Who had the most points in the 2021 season?
    #     **AI:** PlayerZ led the NBA with 2,015 points in 2021.
    #     **Human:** What about assists?
    #     **Standalone Question:** `"Who had the most assists in the 2021 NBA season?"`

    #     **Human:** How many games did Player B play in 2022?
    #     **AI:** 20 games.
    #     **Human:** What about Player C?
    #     **Standalone Question:** `"How many games did Player C play in the 2022 season?"`

    #     **Human:** Which team had the highest average points per game in 2023?
    #     **Standalone Question:** `"Which team had the highest average points per game in the 2023 {sport} season?"`

    #     (NOT **"What is the highest points for <last player>?"**)

    #     ## Step 6: Return JSON Output
    #     Respond strictly in this JSON format:

    #     ```json
    #     {{{{
    #     "isRelevant": true or false,
    #     "isVsQuery": true or false,
    #     "players": [<string>, ...],
    #     "year": <number>,
    #     "standaloneQuestion": "<string>"
    #     }}}}""".format(
    #     schema_info=schema_info,
    #     sport=game,
    #     context=(f"\n{context}\n" if context else ""),
    # )

# Node function to get the context of the conversation for the LLM
async def get_context(conversation_id: str, user_id: str):
    """
    Retrieves summary and unsummarized messages to send to LLM.
    """

    namespace = ("chat", user_id)
    key = conversation_id

    existing_item = await store.aget(namespace, key)

    if existing_item is None:
        conversation_data = {"chat": [], "summary": "", "summaryMade": 0}
    else:
        conversation_data = existing_item.value

    summary = conversation_data.get("summary", "")
    summary_made = conversation_data.get("summaryMade", 0)
    chat_history = conversation_data.get("chat", [])

    unsummarized_messages = chat_history[summary_made:]

    formatted_messages = "\n".join(
        [f"Human: {msg['Human']}\nAI: {msg['AI']}" for msg in unsummarized_messages]
    )

    if summary:
        context = f"\nSummary: {summary}\nRecent Messages:\n{formatted_messages}"
    else:
        context = f"\nRecent Messages:\n{formatted_messages}"

    return context

# Node function to chat with the user and check for player name presence
def chat_with_user(state: AgentState):
    """Generate a standalone question and check for player name presence."""

    user_message = state.messages[-1] if state.messages else HumanMessage(content="Hi, I need your help.")  # Default if no message

    # Retrieve conversation history
    # context = await get_context(state.session_id, "12312312")

    system = f"""
        You are an AI assistant. you can chat with the user
        You may use tools like 'send_cv' or 'tell_joke', 'add_two_numbers' to help users.
    """

    # Construct prompt with system instructions and user input
    conversation = [
        SystemMessage(content=system),
        # SystemMessage(content=f"Conversation Context: {context}"),
        user_message,
    ]

    check_prompt = ChatPromptTemplate.from_messages(conversation)

    LLM_call = check_prompt | llm_with_tools
    response = llm_with_tools.invoke({"messages": state.messages},
        config={"configurable": {"thread_id": state.session_id}}
    )

    # Extract and parse model response
    result = response.content if hasattr(response, "content") else str(response)
    print(style.BLUE, "result", style.RESET, result)
    
    state.response = result

    # Store conversation for future reference
    # await store_conversation(
    #     store, "12312312", state.session_id, state.question, result, "no", "user"
    # )
    
    state.messages.append(AIMessage(content=result))  

    # updated_messages = add_messages(
    #     state.messages,
    #     [AIMessage(content=result)]
    # )

    return {
        "response": result,
        "session_id": state.session_id,
        "messages": state.messages,
    }

# Node function to check if summarization is needed
async def should_summarize(state: AgentState):
    """
    Checks if summarization is needed.
    Returns "summarization_node" if needed, else "END".
    """

    namespace = ("chat", state.userId)
    key = state.configId

    existing_item = await store.aget(namespace, key)

    if existing_item is None:
        return "END"

    conversation_data = existing_item.value
    total_messages = len(conversation_data["chat"])
    summary_made = conversation_data.get("summaryMade", 0)

    print(
        style.BLUE,
        "total_messages N summary_node==================",
        style.RESET,
        total_messages,
        summary_made + msg_exchanges,
    )

    # If at least msg_exchanges (5 or 10) new messages since last summarization, trigger summarization
    if total_messages >= summary_made + msg_exchanges and state.userType == "user":
        return "summarization_node"

    return "END"


DB_URI = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}?sslmode=disable"
)

memory = None
store = None
graphCompiled = None
pool = None


# @asynccontextmanager
# async def lifespan(app: FastAPI):
    # global pool, memory, graphCompiled, store
    # pool = await create_connection_pool(DB_URI)
    # store = await setup_memory(pool)
    # print(
    #     style.BLUE, "Startup complete. Pool and Memory/store initialized.", style.RESET
    # )
    # # graphCompiled = graph.compile(checkpointer=memory)
    # graphCompiled = graph.compile(store=store)
    # yield  # Let FastAPI run while these resources are available

    # print(style.YELLOW, "Shutting down resources...", style.RESET)
    # if pool:
    #     await pool.close()
    #     print(style.BLUE, "Pool closed.", style.RESET)

app = FastAPI(
    title="Agent Zero",
    description="A FastAPI service for Agent Zero.",
    # lifespan=lifespan,
)

# db = SQLDatabase.from_uri(DB_URI)
# graph.add_node("summarization_node", summarization_node)

graph: StateGraph = StateGraph(AgentState)

graph.add_node("chat_with_user", chat_with_user)
graph.add_node("tools", ToolNode([send_cv, tell_joke]))

graph.add_edge(START, "chat_with_user")
graph.add_conditional_edges("chat_with_user", tools_condition)
graph.add_edge("tools", END)
graph.add_edge("chat_with_user", END) 

initialState: AgentState = {
    "messages": [HumanMessage(content="Can u add 4 and 5")],
    "session_id": '12',
    "response": ""
}

graphCompiled = graph.compile()

# config = {"configurable": {"thread_id": "12"}}

# result =  graphCompiled.invoke(initialState, config=config)

# print(style.GREEN, "result", style.RESET, result)

print(style.BLUE, "graph made but not compiled yet", style.RESET)


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="User's question (must not be empty)."
    )
    session_id: str = Field(..., min_length=1, description="session_id")


class QueryResponse(BaseModel):
    """Response model for agentZero results."""

    question: str
    result: str


# End point to ask agentZero
@app.post("/ask-agentZero", response_model=QueryResponse, tags=["agentZero"])
async def query_agent(req: QueryRequest):
    """API endpoint to send a query to the agent.

    - **question**: The user's question.
    - **user_id**: Unique user identifier.

    Returns:
    - `original_question`: The original question.
    - `AiResponse`: The AI-generated response (if successful).
    - `error`: Any error message (if an exception occurs).
    """

    if graphCompiled is None:
        return {"error": "Graph is not ready. Try again in a few seconds."}

    question = req.question
    session_id = req.session_id

    initialState: AgentState = {
        "question": question,
        "session_id": req.session_id,
    }

    try:
        config = {"configurable": {"thread_id": f"{session_id}"}}

        # now run the graph
        result = await graphCompiled.ainvoke(initialState, config=config)

        # print("result", result)

        return QueryResponse(
            question=result.get("question"),
            result=result.get("result"),
        )

    except Exception as e:
        print(style.RED, "exception error", style.RESET, e)
        raise HTTPException(status_code=500, detail=str(e))
