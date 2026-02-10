import os
import sys
import requests
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# 1. LangChain Core & Lifecycle
from langsmith import traceable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool

# 2. Modern Tooling & Retrieval (Updated 2026 Paths)
from langchain_core.tools.retriever import create_retriever_tool

# 3. Model & Vector Store Integrations
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# 4. Agent Execution (2026 Standard)
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun

# Duck duck GO FALLBACK Initialize the tool directly
ddg_tool = DuckDuckGoSearchRun()
# (Optional) If you want to customize the name/description for the agent:
ddg_tool.name = "global_web_search"
ddg_tool.description = "Use this for general news, current events, or when financial tools return no results."

# --- ALPHA VANTAGE NEWS ---

@tool
def get_financial_news(ticker: str) -> str:
    """
    Mandatory for live market news and sentiment. 
    Input MUST be a single stock ticker symbol ONLY (e.g., 'AAPL', 'NVDA').
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    # 1. Clean the input strictly
    clean_ticker = ticker.strip().upper().replace("(", "").replace(")", "")
    
    # 2. Build the URL
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={clean_ticker}&apikey={api_key}'
    
    print(f"\n[TOOL DEBUG] Requesting Alpha Vantage for: {clean_ticker}")
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        # Handle API Notes (Rate Limits)
        if "Note" in data:
            print(f"[TOOL DEBUG] Rate limit hit.")
            return "Note: Alpha Vantage rate limit reached. Please wait 60 seconds."
        
        feed = data.get("feed", [])
        
        # 3. If no ticker-specific news, try a broader topic search as a fallback
        if not feed:
            print(f"[TOOL DEBUG] No specific news for {clean_ticker}, trying broad search...")
            fallback_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=technology&apikey={api_key}'
            feed = requests.get(fallback_url).json().get("feed", [])[:3]

        if not feed:
            return f"No news found for {clean_ticker} even in broad search."

        formatted_news = []
        for item in feed[:5]:
            title = item.get("title", "No Title")
            sentiment = item.get("overall_sentiment_label", "Neutral")
            formatted_news.append(f"Headline: {title}\nSentiment: {sentiment}\n---")
            
        return "\n".join(formatted_news)

    except Exception as e:
        return f"Tool error: {str(e)}"

# --- IMPROVED SYSTEM PROMPT ---

AGENT_SYSTEM_PROMPT = """
You are a precise financial assistant.
1. Use 'search_knowledge_base' for internal documents, filings, and historical data.
2. Use 'get_financial_news' for real-time market events and stock sentiment.
3. Use 'global_web_search' ONLY if the other tools return no results or for non-financial current events.
Always cite your sources. If you use news, mention Alpha Vantage.
Answer in English unless requested otherwise.
"""

# --- MAIN ENGINE ---

class ChatEngine:
    def __init__(self, provider: str = "OpenAI", model_name: str = "gpt-4o-mini", api_key: str = None):
        load_dotenv()
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        print("Loading HuggingFace embedding model...")
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
        )
        
        self.vector_store = None
        self.agent_executor = None
        self.history_store = {} 

    def load_pinecone_knowledge_base(self):
        print(f"🌲 Connecting to Pinecone index: {self.index_name}...")
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
            
    def get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def create_conversational_chain(self):
        if not self.vector_store:
            self.load_pinecone_knowledge_base()
        
        retriever_tool = create_retriever_tool(
            self.vector_store.as_retriever(search_kwargs={"k": 4}),
            "search_knowledge_base",
            "Searches internal company documents and historical filings."
        )

        tools = [retriever_tool, get_financial_news,ddg_tool]
        llm = ChatOpenAI(model=self.model_name, api_key=self.api_key, temperature=0)

        # 2026 Agent Abstraction
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT
        )

        # Wrap with history logic
        self.agent_executor = RunnableWithMessageHistory(
            agent,
            self.get_session_history,
            input_messages_key="messages",
            history_messages_key="chat_history",
        )
    
    @traceable(name="RAG_Brain_Process")
    async def ask_with_sources_async(self, question: str, session_id: str = "default_session"):
        if not self.agent_executor:
            self.create_conversational_chain()
        
        # In 2026, agents expect a list of messages as input
        inputs = {"messages": [HumanMessage(content=question)]}
        
        response = await self.agent_executor.ainvoke(
            inputs,
            config={"configurable": {"session_id": session_id}}
        )
    
        # Logic to extract answer and tools from the State messages
        messages = response.get("messages", [])
        final_answer = messages[-1].content if messages else "I encountered an error."
        
        sources = []
        for msg in messages:
            # Check for tool calls made by the AI
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    name = call.get("name")
                    if name == "get_financial_news":
                        sources.append("Alpha Vantage Real-time News")
                    elif name == "search_knowledge_base":
                        sources.append("Internal Knowledge Base (Pinecone)")

        return {
            "answer": final_answer,
            "sources": list(set(sources)) if sources else ["General Knowledge"]
        }