import os
import sys
import requests
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# =========================
# LANGCHAIN CORE
# =========================
from langsmith import traceable
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool

# =========================
# RETRIEVAL
# =========================
from langchain_core.tools.retriever import create_retriever_tool

# =========================
# MODELS & VECTOR STORE
# =========================
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# =========================
# AGENT
# =========================
from langchain.agents import create_agent

# =========================
# LOAD ENVIRONMENT
# =========================
load_dotenv()


# =========================
# ALPHA VANTAGE NEWS TOOL
# =========================
@tool
def get_financial_news(ticker: str) -> str:
    """
    Fetch real-time financial news and sentiment from Alpha Vantage.
    Input MUST be stock ticker symbol only (example: AAPL, NVDA)
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    clean_ticker = ticker.strip().upper().replace("(", "").replace(")", "")

    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT"
        f"&tickers={clean_ticker}"
        f"&apikey={api_key}"
    )

    print(f"\n[TOOL DEBUG] Alpha Vantage request: {clean_ticker}")

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "Note" in data:
            return "Alpha Vantage rate limit reached. Try again later."

        feed = data.get("feed", [])

        # fallback topic if no ticker news
        if not feed:
            fallback_url = (
                "https://www.alphavantage.co/query"
                f"?function=NEWS_SENTIMENT"
                f"&topics=technology"
                f"&apikey={api_key}"
            )
            feed = requests.get(fallback_url).json().get("feed", [])[:3]

        if not feed:
            return f"No financial news found for {clean_ticker}"

        formatted_news = []
        for item in feed[:5]:
            title = item.get("title", "No Title")
            sentiment = item.get("overall_sentiment_label", "Neutral")
            formatted_news.append(
                f"Headline: {title}\n"
                f"Sentiment: {sentiment}\n"
                f"Source: Alpha Vantage\n"
                "---"
            )

        return "\n".join(formatted_news)

    except Exception as e:
        return f"Financial news error: {str(e)}"


# =========================
# SYSTEM PROMPT
# =========================
AGENT_SYSTEM_PROMPT = """
You are a strict financial RAG assistant.

You have ONLY two tools:

1. search_knowledge_base → internal financial documents, filings, and company data
2. get_financial_news → real-time stock news and sentiment from Alpha Vantage

STRICT RULES:

- ALWAYS use search_knowledge_base for:
  - revenue
  - profit
  - filings
  - financial statements
  - company fundamentals
  - historical financial data

- ALWAYS use get_financial_news for:
  - recent stock news
  - sentiment
  - current financial events

- NEVER use external web search
- NEVER invent financial data
- ALWAYS cite your sources

Internal knowledge base is the primary and most trusted source.

Answer professionally and accurately.
"""


# =========================
# CHAT ENGINE
# =========================
class ChatEngine:

    def __init__(
        self,
        provider: str = "OpenAI",
        model_name: str = "gpt-4o-mini",
        api_key: str = None
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        print("Loading embedding model...")
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
        )

        self.vector_store = None
        self.agent_executor = None
        self.history_store = {}

    # =========================
    # LOAD KNOWLEDGE BASE
    # =========================
    def load_pinecone_knowledge_base(self):
        print(f"Connecting to Pinecone index: {self.index_name}")
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )

    # =========================
    # SESSION MEMORY
    # =========================
    def get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    # =========================
    # CREATE AGENT
    # =========================
    def create_conversational_chain(self):
        if not self.vector_store:
            self.load_pinecone_knowledge_base()

        retriever_tool = create_retriever_tool(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 10}),
            name="search_knowledge_base",
            description="Search internal financial documents, filings, and company data."
        )

        tools = [
            retriever_tool,
            get_financial_news
        ]

        llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=0
        )

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT
        )

        self.agent_executor = RunnableWithMessageHistory(
            agent,
            self.get_session_history,
            input_messages_key="messages",
            history_messages_key="chat_history",
        )

    # =========================
    # ASK QUESTION
    # =========================
    @traceable(name="Financial_RAG_Process")
    async def ask_with_sources_async(
        self,
        question: str,
        session_id: str = "default_session"
    ):
        if not self.agent_executor:
            self.create_conversational_chain()

        inputs = {"messages": [HumanMessage(content=question)]}

        response = await self.agent_executor.ainvoke(
            inputs,
            config={"configurable": {"session_id": session_id}}
        )

        messages = response.get("messages", [])
        final_answer = messages[-1].content if messages else "Error generating response."

        sources = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    tool_name = call.get("name")
                    if tool_name == "search_knowledge_base":
                        docs_info = call.get("metadata", [])
                        if docs_info:
                            for doc in docs_info:
                                source_label = doc.get("source") or doc.get("title") or "Internal KB"
                                sources.append(f"Internal KB: {source_label}")
                        else:
                            sources.append("Internal Knowledge Base (Pinecone)")
                    elif tool_name == "get_financial_news":
                        sources.append("Alpha Vantage Financial News")

        # Append sources to answer
        final_answer_with_sources = final_answer
        if sources:
            final_answer_with_sources += "\n\nSources:\n- " + "\n- ".join(list(set(sources)))

        return {
            "answer": final_answer_with_sources,
            "sources": list(set(sources)) if sources else ["General Knowledge"]
        }