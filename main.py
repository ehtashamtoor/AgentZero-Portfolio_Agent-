import os, datetime, time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts import get_conversion_prompt, NBA_SQL_GENERATION
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List, Optional, Annotated, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import QuerySQLDataBaseTool
from operator import itemgetter
from vector_store import VectorStore

os.system("")


# Class of different styles to be used in terminal
class style:
    RED = "\033[31m"  # for errors
    GREEN = "\033[32m"  # for Success
    YELLOW = "\033[33m"  # for warnings
    BLUE = "\033[34m"  # for info
    RESET = "\033[0m"


load_dotenv()

msg_exchanges = 5


class AgentState(BaseModel):
    question: str = Field(..., description="User's standalone question")
    original_question: str = Field(..., description="User's question")
    question_category: str = Field(..., description="User's question category")
    sql_query: str = Field(default="", description="SQL query to be executed")
    query_result: str = Field(default="", description="Result of the SQL query")
    query_rows: List = Field(
        default_factory=list, description="Rows returned from the query"
    )
    attempts: int = Field(default=0, description="Number of query execution attempts")
    relevance: Optional[bool] = Field(
        default=None, description="Indicates if the query result is relevant"
    )
    sql_error: Optional[bool] = Field(
        default=None, description="Indicates if there was an SQL error"
    )
    isGeneral: Optional[bool] = Field(
        default=None, description="Indicates if the response is general or from DB"
    )
    game: str = Field(..., description="Game name associated with the query")
    top_k: str = Field(default="2", description="top_k results to get")
    configId: str = Field(..., description="Configuration ID for the session")
    userId: str = Field(..., description="User ID")
    userType: str = Field(..., description="User Type")
    schema_info: str = Field(default="", description="Database schema information")
    messages: Annotated[
        List[AnyMessage],
        add_messages,
        Field(default_factory=list, description="List of messages in the session"),
    ]


class CheckRelevance(BaseModel):
    isRelevant: bool = Field(
        description="Indicates whether the question is related to the database schema. True or False."
    )
    isVsQuery: bool = Field(
        description="Indicates whether the question is a comparison query. True or False."
    )
    players: list[str] = Field(
        description="List of player names mentioned in the query."
    )
    year: int = Field(description="Year mentioned in the query.")
    standaloneQuestion: str = Field(
        description="A standalone version of the query, with pronouns removed and context added."
    )


class ConvertToSQL(BaseModel):
    sql_query: str = Field(
        description="The SQL query corresponding to the user's question."
    )


class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question.")


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


async def setup_memory(pool):
    """Sets up memory using AsyncPostgresSaver."""
    memory = AsyncPostgresStore(pool)
    # memory = AsyncPostgresSaver(pool)
    await memory.setup()
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


# old function to get relevant tables from the database
# def get_relevant_tables(sport: str, db: SQLDatabase, category: str = "") -> List[str]:
#     """Get relevant tables from the database."""
#     all_tables = [t for t in db.get_usable_table_names()]
#     relevant_tables = [
#         name for name in all_tables if name.lower().startswith(f"{sport.lower()}_")
#     ]
#     print(style.BLUE, "relevant_tables", style.RESET, relevant_tables)
#     return relevant_tables


#  new function to get relevant tables from the database based on sport, category
def get_relevant_tables(sport: str, db: SQLDatabase, category: str = "") -> List[str]:
    """Get relevant tables from the database based on sport, category."""
    all_tables = [t for t in db.get_usable_table_names()]

    if sport.lower() == "nba":
        if category.lower() == "playoffs":
            # Include playoffs tables and always include nba_player_info
            relevant_tables = [
                name for name in all_tables if name.lower().startswith("nba_playoffs")
            ]
            if "nba_player_info" in all_tables:
                relevant_tables.append("nba_player_info")
        else:
            # Default case for NBA — exclude playoffs tables
            relevant_tables = [
                name
                for name in all_tables
                if name.lower().startswith("nba_")
                and not name.lower().startswith("nba_playoffs")
            ]
    else:
        # Default for all other sports
        relevant_tables = [
            name for name in all_tables if name.lower().startswith(f"{sport.lower()}_")
        ]

    print(style.BLUE, "relevant_tables", style.RESET, relevant_tables)
    return relevant_tables


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

    # summary = conversation_data.get("summary", "")
    # summary_made = conversation_data.get("summaryMade", 0)
    chat_history = conversation_data.get("chat", [])

    # unsummarized_messages = chat_history[summary_made:]

    # formatted_messages = "\n".join(
    #     [f"Human: {msg['Human']}\nAI: {msg['AI']}" for msg in unsummarized_messages]
    # )

    # if summary:
    #     context = f"\nSummary: {summary}\nRecent Messages:\n{formatted_messages}"
    # else:
    #     context = f"\nRecent Messages:\n{formatted_messages}"

    #  get only last 5 messages to set in context of agent
    messagess = chat_history[-5:]

    formatted_messages = "\n".join(
        [f"Human: {msg['Human']}\nAI: {msg['AI']}" for msg in messagess]
    )

    # Set the context
    context = f"\nRecent Messages:\n{formatted_messages}"

    return context


# node function to check for the relevance of the question, if the question is relevant to the database
async def check_relevance(state: AgentState):
    """Check if the question is relevant to the database."""

    # rel_tables = get_relevant_tables(state.game, db)
    # schema_info = db.get_table_info(rel_tables)
    # state.schema_info = schema_info

    question = state.question
    schema_info = state.schema_info
    game = state.game
    conversation_id = state.configId
    user_id = state.userId

    #  when context got added in the prompt, omit the flow for guest user, cause we donot need any memory for guest users

    should_include_context = state.userType == "user"
    context = (
        await get_context(conversation_id, user_id) if should_include_context else ""
    )

    # print(style.BLUE, "context is ================\n", style.RESET, context)

    system = """
        You are a chat assistant expert that determines whether a user's question is relevant to the {sport} database.  
        Your goal is to classify the question, extract key details, and refine it into a standalone query. standalone query must be according to user intent.  
        Use the provided conversation context to resolve references.

        Instructions: Follow these strict rules:
         - Handle abbreviations, nicknames, and partial names for players and teams:
         - Recognize common nicknames and abbreviations for players.
         - Recognize common abbreviations and short forms for teams.
         - If a name is ambiguous, assume the most well-known player or team in the context of {sport}

        ## Step 1: Verify Sport and Database Relevance  
        - The user's question must be related to {sport}.  
        - Check if the query is relevant to the database schema:  

        **Schema Information:**  
        {schema_info}

        - If the question is NOT related to {sport} or the database, mark `"isRelevant": false`.

        ## Step 2: Use Context for Reference Resolution  
        The user may refer to past queries using words like "he," "they," "last time," etc.  
        Use the **conversation history** to resolve ambiguous references:  

        If a user’s question lacks details (e.g., “What about Player X?”), use the context to infer the missing information.  

        ## Step 3: Classify the Query  
        1. **General Query:**  
        - Greetings: “Hi”, “Hello”  
        - Out-of-context: “What’s the weather?”  
        - Random definitions: “What is {sport}?”  

        2. **Database Query:**  
        - Asking for statistics: “Player X goals in 2022”  
        - Mentioning specific data points  

        3. **Comparison Query:**  
        - Includes: “compare”, “vs”, “versus”  
        - Explicitly mentions two or more entities

        ## Step 4: Extract Key Details  
        - **Players:** Extract player names. Return `[]` if none are mentioned.   

        ## Step 5: Return JSON Output  
        Respond strictly in this JSON format:  

        ```json
        {{{{
        "isRelevant": true or false,
        "isVsQuery": true or false,
        "players": [<string>, ...],
        "year": <number>,
        "standaloneQuestion": "<string>"
        }}}}
        
        """.format(
        schema_info=schema_info,
        sport=game,
    )

    conversation = [
        SystemMessage(content=system),
    ]

    # Add the conversation context only if should_include_context is True
    if should_include_context and context:
        conversation.append(SystemMessage(content=f"Conversation Context: {context}"))

    conversation.append(HumanMessage(content=f"Question: {question}"))

    check_prompt = ChatPromptTemplate.from_messages(conversation)

    structured_llm = llm.with_structured_output(CheckRelevance)

    relevance_checker = check_prompt | structured_llm
    is_relevant: CheckRelevance = await relevance_checker.ainvoke(
        {"question": question}, config={"configurable": {"thread_id": state.configId}}
    )

    print(
        style.BLUE, "==========is_relevant===============\n ", style.RESET, is_relevant
    )

    state.relevance = is_relevant.isRelevant
    state.question = is_relevant.standaloneQuestion

    return state


# node function by which a general response will be given to user query
async def general_response(state: AgentState):
    question = state.question
    game = state.game
    """Generates a general response to the user query."""

    system = """You are an {game} sports assistant focused on answering questions and providing {game} stats. 
    Use clear, structured responses with data presented in paragraphs or bullet points for readability.

    - Don't tell the user to search on a relevant website; instead, try to answer the question first. 
    You can mention external sources at the end if necessary.

    - For real-time data requests, prioritize web search and provide relevant, up-to-date information 
    without mentioning data limitations.
    
    ### Important
        - Use Your Reasoning Power to analyze what user is asking. If users asks a general question which is not relevant to {game}, then give user a polite answer that you are a sport assistant and you are here to help with {game} related stats.


    Ensure all player, team, or game references are accurate and up-to-date. Do not mention that data access is limited, but summarize available information as needed."

    """

    human = f"Question: {question}"

    gen_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)]
    )

    gen_response = gen_prompt | llm | StrOutputParser()
    gen_response_message = await gen_response.ainvoke({"game": game})

    print(style.GREEN, "gen_response_message==== ", style.RESET, gen_response_message)

    await store_conversation(
        store,
        user_id=state.userId,
        conversation_id=state.configId,
        user_message=state.original_question,
        ai_response=gen_response_message,
        game=state.game,
        userType=state.userType,
    )

    state.query_result = gen_response_message
    state.isGeneral = True

    return state


async def get_table_columns_for_tables(db: SQLDatabase, table_names: List[str]) -> str:
    schema_info = ""

    for table in table_names:
        query = f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{table}';
        """

        result = db.run(query)

        if isinstance(result, str):
            result = eval(result)

        if isinstance(result, list) and all(
            isinstance(col, tuple) and len(col) == 2 for col in result
        ):
            schema_info += f"Table: {table}\n"
            for col in result:
                schema_info += f"  Column Name: {col[0]}, Data Type: {col[1]}\n"
        else:
            schema_info += f"Unexpected format for table {table}: {result}\n"

    return schema_info


# function to classify type of question according to game
def classify_query_type_llm(question: str, game: str) -> str:
    valid_categories = {
        "nba": ["regular", "playoffs"],
        "afl": ["regular", "fantasy"],
        "aflw": ["regular"],
        "cric": ["regular"],
        "ufc": ["regular"],
    }

    game = game.lower()
    categories = valid_categories.get(game, ["regular"])
    allowed_labels = "', '".join(categories)

    system_prompt = (
        f"You are a classification assistant. Given a user question and the name of a game, "
        f"classify the question into one of the allowed categories for that game. "
        f"Respond with only one of these: '{allowed_labels}'.\n\n"
        f"Examples:\n"
        f"- NBA: 'Most assists in 2022 playoffs?' → 'playoffs'\n"
        f"- NBA: 'Top scorer in 2021 season?' → 'regular'\n"
        f"- AFL: 'Best fantasy performer in round 2?' → 'fantasy'\n"
        f"- UFC: 'Who won the last lightweight title fight?' → 'regular'\n\n"
        f"If the user mentions something that doesn't match the allowed categories (e.g., 'fantasy' for NBA), "
        f"choose the best match among the allowed ones."
    )

    user_prompt = f"Game: {game}\nQuestion: {question}\n\nCategory:"

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    category = response.content.strip().lower()

    if category not in categories:
        return "regular"
    return category


# node function to make SQL query
async def convert_to_sql(state: AgentState):
    """Converts the user query to a syntactically correct SQL query."""

    question = state.question
    schema_info = state.schema_info
    game = state.game
    top_k = int(state.top_k)
    category = state.question_category

    start = time.perf_counter()
    print(style.BLUE, "========= attempts ============", style.RESET, state.attempts)

    # now we need to get relevant examples based on game, category
    vector_store = VectorStore()

    db_examples = vector_store.search_examples_vectors(question, game, category)

    allRelevantExamples = "\n\n".join(
        f"human: {ex['input']}\n ai: {ex['query']}" for ex in db_examples
    )

    print(
        style.GREEN, "All Relevant Examples:::::::\n", style.RESET, allRelevantExamples
    )
    db_time = time.perf_counter() - start

    print(f"🐢 DB time: {db_time * 1000:.2f} ms")
    current_time = datetime.datetime.now()

    SQL_conversion_prompt = get_conversion_prompt(game, category)

    SYS_PROMPT = SQL_conversion_prompt.format(
        table_info=schema_info,
        top_k=top_k,
        current_time=current_time,
        rel_examples=allRelevantExamples,
    )

    sql_prompt = ChatPromptTemplate.from_messages(
        [("system", SYS_PROMPT), ("human", "{input}")]
    ).partial(table_info=schema_info, top_k=top_k)

    if state.attempts > 0:
        system_instruction = "Your task is to refine or regenerate SQL queries based on errors or unsatisfactory results. Ensure the query correctly retrieves relevant information for the given question."

        if state.sql_error:
            instruction = (
                "The previous query resulted in an error. Analyze the issue and generate a "
                "corrected SQL query that avoids the error while preserving the intent."
            )
        elif not state.query_result or state.query_result.strip() == "":
            instruction = (
                "The previous query executed successfully but returned no relevant results. "
                "Modify the query to better match the intent of the question."
            )
        else:
            instruction = (
                "The previous query executed successfully, but ensure it is optimal and efficient. "
                "If improvements can be made, suggest a refined version."
            )

        SYS_PROMPT = (
            SYS_PROMPT
            + f"Below is the system instructions for you: {system_instruction}"
        )

        # human = (
        #     "human",
        #     """Original Question: {question}\n\nPrevious SQL Query:\n{state.sql_query}\n\nQuery Result / Error:\n{state.query_result}\n\n{instruction}""",
        # )

        sql_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYS_PROMPT
                    + """Original Question: {question}

                    Previous SQL Query:
                    {prev_sql_query}

                    Query Result / Error:
                    {query_result}

                    {instruction}

                    Provide only the corrected SQL query without any explanation.
                    """,
                ),
                (
                    "human",
                    "{input}",
                ),
            ]
        ).partial(table_info="schema_info", top_k=top_k, question=question)

        # sql_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             SYS_PROMPT,
        #         ),
        #         (
        #             "human",
        #             """Original Question: {question}

        #             Previous SQL Query:
        #             {prev_sql_query}

        #             Query Result / Error:
        #             {query_result}

        #             {instruction}

        #             Provide only the corrected SQL query without any explanation.
        #             """,
        #         ),
        #     ]
        # )

    execute_query = QuerySQLDataBaseTool(db=db)

    write_query = create_sql_query_chain(llm, db, prompt=sql_prompt, k=top_k)

    chain = RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )

    if state.attempts > 0:
        sql_query = await chain.ainvoke(
            {
                "question": question,
                "top_k": top_k,
                "table_info": "schema_info",
                "prev_sql_query": state.sql_query,
                "query_result": state.query_result,
                "instruction": instruction,
            }
        )
        # sql_query: ConvertToSQL = sql_converter.invoke(
        #     {
        #         "question": question,
        #         "prev_sql_query": state.sql_query,
        #         "query_result": state.query_result,
        #         "instruction": instruction,
        #     }
        # )
    else:
        sql_query = await chain.ainvoke(
            {"question": question, "top_k": top_k, "table_info": ""}
        )
    state.sql_query = sql_query.get("query")

    return state


def sql_syntax_query(state: AgentState):
    """Converts the generated query into a syntactically correct SQL query."""

    sqlquery = state.sql_query

    current_time = datetime.datetime.now()

    system = NBA_SQL_GENERATION.format(sqlquery=sqlquery, current_time=current_time)

    sql_prompt = ChatPromptTemplate.from_messages([("system", system)])

    structured_llm = llm.with_structured_output(ConvertToSQL)

    sql_converter = sql_prompt | structured_llm
    sql_query: ConvertToSQL = sql_converter.invoke({"sqlquery": sqlquery})

    state.sql_query = str(sql_query.sql_query)

    return state


#  node function to execute the SQL query
def execute_sql_query(state: AgentState):
    """Executes the SQL query and returns the result from the database"""
    print(style.BLUE, "state.sql_error====", style.RESET, state.sql_error)
    try:
        query = state.sql_query
        result = db._execute(query)
        print(
            style.GREEN, "result after executing SQL query====\n", style.RESET, result
        )

        if len(result) == 0:
            result = "No data found for the given query"
        # if result:
        #     VectorStore.add_gpt_vector(question=state.question, message=result)
        state.query_result = str(result)
        state.sql_error = False
    except Exception as e:
        state.sql_error = True
        print(style.BLUE, "SQL Query Error in executing \n", style.RESET, str(e))
        state.query_result = str(e)

    return state


# node function to format the result of sql query into human readable form
async def human_readable_query_result(state: AgentState):
    """Formats the SQL query result into a response"""

    print(
        style.BLUE,
        "================== responding to the user readable response =============",
        style.RESET,
    )

    query_result = state.query_result
    sql_query = state.sql_query

    system_instruction = """
        ### **User-Friendly Instruction Update**  

        You are an assistant that converts SQL query results into **clear, natural language responses**. When presenting information:  

        - **Do not include any identifiers like player IDs.**  
        - Always use **player or team names** to make responses more intuitive.  
        - Instead of just stating facts, provide context or comparisons when relevant.  
        - If the result contains multiple rows, return it as a **Markdown table** so it can be displayed properly in chat UI.  
        - If the result contains multiple rows, return it as a **Markdown table** so it can be displayed properly in the chat UI.  
        - **Display a maximum of 20 rows** from the SQL result. If the result contains more than 20 rows, include only the first 20.  

        #### **Table Formatting Example:**  
        If the result contains multiple rows, return it like this: Below is just an example

        
        | head 1       | head 2         | head 3 | head 4 | head 5 |
        |-----------|----------------|--------|---------|---------|
        | 2024-01-12 | Miami Heat      | 25     | 6       | 12      |
        | 2024-01-15 | Boston Celtics  | 30     | 5       | 9       |
        

        - **Ensure column headers are human-friendly** (e.g., "Games Played" instead of "games_played").  
        - **Do not add extra explanations** when using a table—just return the table directly, with one two lines of context if needed.  
     """

    instruction = "Format the response accordingly based on the query result."

    generate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            (
                "human",
                """SQL Query:
                {sql_query}

                Result:
                {query_result}

                {instruction}
                """,
            ),
        ]
    )

    format_response = generate_prompt | llm | StrOutputParser()
    format_response_message = await format_response.ainvoke(
        {
            "sql_query": sql_query,
            "query_result": query_result,
            "instruction": instruction,
        }
    )

    # Store conversation message
    await store_conversation(
        store,
        user_id=state.userId,
        conversation_id=state.configId,
        user_message=state.original_question,
        ai_response=format_response_message,
        game=state.game,
        userType=state.userType,
    )

    state.query_result = format_response_message
    state.isGeneral = False

    print(style.BLUE, "format_response_message\n", style.RESET, format_response_message)

    return state


# node function to rewrite the user question to resolve errors in the query
async def rewrite_question(state: AgentState):
    """Rewrites the user question to resolve errors in the SQL query."""

    question = state.question

    system = """You are an assistant that rewrites user question to resolve errors in SQL queries. Your task is to generate a clear, concise, and unambiguous question that can be used to generate a correct postgreSQL query. Follow these guidelines to ensure accurate query generation:

    1. **Pronoun Removal**:
    - Remove pronouns and ambiguous terms from the question to create a clear and unambiguous query.
    2. **Context Addition**:
    - Add relevant context or player names to the query to make it self-contained and complete.
    3. **Query Structure**:
    - Ensure the rewritten query is a complete question that can be used to generate a valid SQL query.
    4. **Error Resolution**:
    - Resolve any errors or ambiguities present in the original query to generate a correct SQL query.

    ### Example Inputs:
    - Original Query: "How many points did he score?"
    - Rewritten Query: "How many points did LeBron James score?"

    ### Important:
    Return only the rewritten query in your response, ensuring compliance with the above rules.

    Output Format:
    Strictly return the result in the following JSON format:
    {{{{
      "question": "<string>"
    }}}}
    """

    conversation = [SystemMessage(content=system)]
    conversation.append(HumanMessage(content=f"Question: {question}"))

    rewrite_prompt = ChatPromptTemplate.from_messages(conversation)

    structured_llm = llm.with_structured_output(RewrittenQuestion)

    rewrite_query = rewrite_prompt | structured_llm
    rewritten_question: RewrittenQuestion = await rewrite_query.ainvoke(
        {"question": question}
    )

    state.question = rewritten_question.question

    state.attempts += 1
    print(style.BLUE, "rewritten_question==== ", style.RESET, rewritten_question)

    return state


# node function to regenerate the sql query using old sql query and query result, question
async def regenerate_sql_query(state: AgentState):
    """Attempts to refine or regenerate the SQL query based on errors or unsatisfactory results."""

    question = state.question
    prev_sql_query = state.sql_query
    query_result = state.query_result
    sql_error = state.sql_error

    system_instruction = (
        "You are an expert postgresSQL assistant. Your task is to refine or regenerate SQL queries "
        "based on errors or unsatisfactory results. Ensure the query correctly retrieves "
        "relevant information for the given question."
        "Below is the relevant tables schema info:"
    )

    if sql_error:
        instruction = (
            "The previous query resulted in an error. Analyze the issue and generate a "
            "corrected SQL query that avoids the error while preserving the intent."
        )
    elif not query_result or query_result.strip() == "":
        instruction = (
            "The previous query executed successfully but returned no relevant results. "
            "Modify the query to better match the intent of the question."
        )
    else:
        instruction = (
            "The previous query executed successfully, but ensure it is optimal and efficient. "
            "If improvements can be made, suggest a refined version."
        )

    generate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction + state.schema_info),
            (
                "human",
                """Original Question: {question}

                Previous SQL Query:
                {prev_sql_query}

                Query Result / Error:
                {query_result}

                {instruction}

                Provide only the corrected SQL query without any explanation.
                """,
            ),
        ]
    )

    refine_sql = generate_prompt | llm | StrOutputParser()
    new_sql_query = await refine_sql.ainvoke(
        {
            "question": question,
            "prev_sql_query": prev_sql_query,
            "query_result": query_result,
            "instruction": instruction,
        }
    )

    print(
        style.GREEN,
        "==============new SQL query=============\n",
        style.RESET,
        new_sql_query,
    )

    state.sql_query = new_sql_query.strip()
    # state.attempts += 1

    return state


def end_max_iterations(state: AgentState):
    state.query_result = "Please try again."
    print(style.GREEN, "Maximum attempts reached. Ending the workflow.", style.RESET)
    return state


#  function to route the agent to the next node based on the relevance of the question
def relevance_router(state: AgentState):
    if state.relevance:
        return "convert_to_sql"
    else:
        return "general_response"


def check_attempts_router(state: AgentState):
    if state.attempts >= 3:
        return "end_max_iterations"
    else:
        return "rewrite_question"


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


def sql_error_router(state: AgentState):
    if state.sql_error:
        state.sql_query = ""
        state.query_result = ""
        state.sql_error = False
        return "rewrite_question"
    else:
        return "human_readable_query_result"


DB_URI = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}?sslmode=disable"
)

memory = None
store = None
graphCompiled = None
pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool, memory, graphCompiled, store
    pool = await create_connection_pool(DB_URI)
    store = await setup_memory(pool)
    print(
        style.BLUE, "Startup complete. Pool and Memory/store initialized.", style.RESET
    )
    # graphCompiled = graph.compile(checkpointer=memory)
    graphCompiled = graph.compile(store=store)
    yield  # Let FastAPI run while these resources are available

    print(style.YELLOW, "Shutting down resources...", style.RESET)
    if pool:
        await pool.close()
        print(style.BLUE, "Pool closed.", style.RESET)


llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), temperature=0, model="gpt-4o-mini"
)

app = FastAPI(
    title="SQL Agent API",
    description="A FastAPI service for SQL-based question answering using LangGraph",
    lifespan=lifespan,
)

db = SQLDatabase.from_uri(DB_URI)

top_k = 2

graph: StateGraph = StateGraph(AgentState)

graph.add_node("check_relevance", check_relevance)

graph.add_node("general_response", general_response)
graph.add_node("summarization_node", summarization_node)
graph.add_node(
    "convert_to_sql",
    convert_to_sql,
)
graph.add_node("execute_sql_query", execute_sql_query)
graph.add_node("human_readable_query_result", human_readable_query_result)
graph.add_node("rewrite_question", rewrite_question)
graph.add_node("end_max_iterations", end_max_iterations)

graph.add_edge(START, "check_relevance")
graph.add_conditional_edges(
    "check_relevance",
    relevance_router,
    {
        "convert_to_sql": "convert_to_sql",
        "general_response": "general_response",
    },
)
graph.add_edge("convert_to_sql", "execute_sql_query")
graph.add_conditional_edges(
    "execute_sql_query",
    sql_error_router,
    {
        "rewrite_question": "rewrite_question",
        "human_readable_query_result": "human_readable_query_result",
    },
)
graph.add_conditional_edges(
    "rewrite_question",
    check_attempts_router,
    {
        "end_max_iterations": "end_max_iterations",
        "rewrite_question": "convert_to_sql",
    },
)
graph.add_edge("end_max_iterations", "general_response")

graph.add_conditional_edges(
    "human_readable_query_result",
    should_summarize,
    {
        "summarization_node": "summarization_node",
        "END": END,
    },
)
graph.add_conditional_edges(
    "general_response",
    should_summarize,
    {
        "summarization_node": "summarization_node",
        "END": END,
    },
)

graph.add_edge("summarization_node", END)
print(style.BLUE, "graph made but not compiled yet", style.RESET)


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="User's question (must not be empty)."
    )
    game: str = Field(..., description="Game name (must be in uppercase)")
    session_id: str = Field(
        ..., min_length=1, description="Session ID (must not be empty)."
    )
    user_id: str = Field(..., min_length=5, description="user ID (must not be empty).")
    userType: Literal["guest", "user"] = Field(
        ...,
        description="User type: guest or logged-in user (must be 'guest' or 'user').",
    )

    @field_validator("game")
    @classmethod
    def enforce_uppercase(cls, value: str) -> str:
        """Ensure 'game' is always uppercase."""
        if not value.isupper():
            raise ValueError("GAME must be in uppercase")
        return value

    @field_validator("question", "session_id", "user_id")
    @classmethod
    def enforce_non_empty(cls, value: str) -> str:
        """Ensure 'question', 'session_id', 'user_id are not empty."""
        if not value or not value.strip():
            raise ValueError("Field must not be empty")
        return value


class QueryResponse(BaseModel):
    """Response model for STATGPT results."""

    question: str
    original_question: str
    AiResponse: Optional[str] = ""
    error: Optional[str] = ""
    sql_query: Optional[str] = ""
    isGeneral: Optional[bool] = None


# End point to ask statGPT
@app.post("/ask-statgpt", response_model=QueryResponse, tags=["STATGPT"])
async def query_agent(req: QueryRequest):
    """API endpoint to send a query to the agent.

    - **question**: The user's question.
    - **session_id**: Unique session identifier.
    - **user_id**: Unique user identifier.
    - **game**: Game context for query.(Must be uppercase)
    - **userType**: userType of user (guest or user)

    Returns:
    - `question`: The standalone question.
    - `original_question`: The original question.
    - `AiResponse`: The AI-generated response (if successful).
    - `error`: Any error message (if an exception occurs).
    - `isGeneral`: Indicates if the response is general one or from the database.
    """

    if graphCompiled is None:
        return {"error": "Graph is not ready. Try again in a few seconds."}

    question = req.question
    session_id = req.session_id
    user_id = req.user_id
    game = req.game

    category = classify_query_type_llm(question, game)
    print(style.BLUE, "category of the question", style.RESET, category, game)

    #  now on the basis of game and category, we need to get relevant tables from the database
    rel_tables = get_relevant_tables(game, db, category)

    schema_info = await get_table_columns_for_tables(db, rel_tables)

    initialState: AgentState = {
        "question": question,
        "original_question": question,
        "question_category": category,
        "sql_query": "",
        "query_result": "",
        "query_rows": [],
        "attempts": 0,
        "relevance": None,
        "sql_error": False,
        "isGeneral": None,
        "game": game,
        "top_k": "2",
        "configId": session_id,
        "userId": user_id,
        "userType": req.userType,
        "schema_info": schema_info,
        "messages": [],
    }

    try:
        config = {"configurable": {"thread_id": f"{session_id}"}}

        # now run the graph
        result = await graphCompiled.ainvoke(initialState, config=config)

        return QueryResponse(
            question=result.get("question"),
            original_question=result.get("original_question"),
            AiResponse=result.get("query_result"),
            sql_query=result.get("sql_query"),
            isGeneral=result.get("isGeneral"),
        )

    except Exception as e:
        print(style.RED, "exception error", style.RESET, e)
        raise HTTPException(status_code=500, detail=str(e))
