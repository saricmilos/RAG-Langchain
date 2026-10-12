import requests
import time
from langsmith import traceable
from langchain.tools import tool

@tool
def rag_chat(message: str, session_id: str = "apple_simple_eval") -> dict:
    """Queries the RAG endpoint for simple Apple facts."""
    url = "https://rag-langchain-red-river-2523.fly.dev/chat"
    payload = {"message": message, "session_id": session_id}
    try:
        response = requests.post(url, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": []}

# SIMPLIFIED QUESTIONS: Targeted at text sections, not tables
TEST_SUITE = [
    {
        "label": "2025_CEO_Quote",
        "q": "What did Tim Cook say about Apple's commitment to American innovation in the February 2025 $500 billion investment announcement?"
    },
    {
        "label": "Houston_Facility_Purpose",
        "q": "According to the documentation, what is the primary purpose of the new 250,000-square-foot facility in Houston?"
    },
    {
        "label": "iPhone_17_Launch",
        "q": "Does the 2025 documentation mention the launch of the iPhone 17 series and Apple Watch Series 11?"
    },
    {
        "label": "Legal_DMA_Simple",
        "q": "What was the reason the European Commission issued a fine against Apple in April 2025?"
    },
    {
        "label": "Share_Repurchase_Note",
        "q": "Does the 2025 report mention a share repurchase program, and what was the amount of the new authorization mentioned?"
    },
    {
        "label": "Environmental_Goal",
        "q": "What percentage of carbon emissions has Apple cut since 2015, according to the 2025 Environmental Progress Report?"
    }
]

@traceable(name="Apple_Simple_Text_Check")
def run_eval():
    session_id = f"simple_audit_{int(time.time())}"
    for test in TEST_SUITE:
        print(f"\n[Testing: {test['label']}]")
        result = rag_chat.invoke({"message": test['q'], "session_id": session_id})
        print(f"🤖 Answer: {result.get('answer')}")
        print(f"📚 Sources: {result.get('sources')}")

if __name__ == "__main__":
    run_eval()