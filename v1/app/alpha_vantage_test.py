import asyncio
import langchain
from engine_deploy import ChatEngine

# Enables the 2026 detailed trace logs
langchain.debug = True 

async def test_agent_routing():
    engine = ChatEngine()
    
    # Test 1: Should use Pinecone (Knowledge Base)
    print("\n--- TEST: INTERNAL DOCS ---")
    # We must 'await' the async function here
    result1 = await engine.ask_with_sources_async("What is APPLE's company policy on remote work?")
    print(f"Answer: {result1['answer']}")
    print(f"Sources: {result1['sources']}")
    
    # Test 2: Should use Alpha Vantage (News API)
    print("\n--- TEST: LIVE NEWS ---")
    result2 = await engine.ask_with_sources_async("What is the latest market sentiment for Nvidia (NVDA)?")
    print(f"Answer: {result2['answer']}")
    print(f"Sources: {result2['sources']}")

    # Test 3: Should use DuckDuckGo (Global Web Search)
    print("\n--- TEST: GENERAL WEB SEARCH ---")
    result3 = await engine.ask_with_sources_async("Who is currently leading the governemt of Serbia?")
    print(f"Answer: {result3['answer']}")
    print(f"Sources: {result3['sources']}")

if __name__ == "__main__":
    # Use asyncio.run to start the event loop
    asyncio.run(test_agent_routing())