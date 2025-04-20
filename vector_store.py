import os
import asyncio
from typing import Dict, List
from dotenv import load_dotenv
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class VectorStore:
    """Handles vector storage and retrieval using PostgreSQL + pgvector."""

    def __init__(self):
        """Initialize database connection."""
        db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        self.engine = create_engine(db_url)  # ✅ Prevents async issues
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def add_gpt_vector(self, question: str, message: str):
        """Inserts a new embedding into the database."""
        embeddings = OpenAIEmbeddings()
        embedding = embeddings.embed_query(question)
        embedding_json = json.dumps(embedding)  # Store as JSON
        message_json = json.dumps(message)
        query = """
            INSERT INTO gpt_items (question, embedding, message) 
            VALUES (:question, :embedding, :message)
        """

        with self.SessionLocal() as session:
            session.execute(
                text(query),
                {
                    "question": question,
                    "embedding": embedding_json,
                    "message": message_json,
                },
            )
            session.commit()

        print("✅ Added vector:", {"question": question, "message": message})

    def search_gpt_vectors(self, question: str):
        """Searches for the most relevant message using cosine similarity."""
        embeddings = OpenAIEmbeddings()
        query_embedding = embeddings.embed_query(question)
        embedding_json = json.dumps(query_embedding)
        query = """SELECT * 
            FROM gpt_items
            WHERE embedding <-> $1 >= 0.8
            ORDER BY embedding <-> $1
            LIMIT 1;"""

        with self.SessionLocal() as session:
            result = session.execute(text(query), {"embedding": embedding_json})
            row = result.fetchone()

            if row:
                return {"question": row[0], "message": row[1]}

        return None

    #  new function for searching examples with game and category filtering
    def search_examples_vectors(
        self, question: str, game: str, category: str, k: int = 2
    ) -> List[Dict[str, str]]:
        """Returns top-k examples filtered by game and category."""

        if not game.isidentifier() or not category.isidentifier():
            raise ValueError("Invalid game or category")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        query_embedding = embeddings.embed_query(question)

        query = text(
            """
            SELECT input, query
            FROM game_examples
            WHERE game = :game AND category = :category
            ORDER BY embedding <-> :embedding
            LIMIT :k;
        """
        )

        with self.SessionLocal() as session:
            result = session.execute(
                query,
                {
                    "embedding": json.dumps(query_embedding),
                    "k": k,
                    "game": game.lower(),
                    "category": category.lower(),
                },
            )
            rows = result.fetchall()
            return [{"input": row[0], "query": row[1]} for row in rows] if rows else []

    # old function for searching examples without game and category filtering
    # def search_examples_vectors(self, question: str, table_name: str, k: int = 2):
    #     """Returns top-k examples as a list of dicts in the format: {input, query}"""

    #     if not table_name.isidentifier():
    #         raise ValueError("Invalid table name")

    #     embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #     query_embedding = embeddings.embed_query(question)

    #     query = text(
    #         f"""
    #         SELECT input, query
    #         FROM {table_name}
    #         ORDER BY embedding <-> :embedding
    #         LIMIT :k;
    #         """
    #     )

    #     with self.SessionLocal() as session:
    #         result = session.execute(
    #             query, {"embedding": json.dumps(query_embedding), "k": k}
    #         )
    #         rows = result.fetchall()

    #         return [{"input": row[0], "query": row[1]} for row in rows] if rows else []
