import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Setup Environment and Path Logic
current_file_path = Path(__file__).resolve()
app_dir = current_file_path.parent
root_dir = app_dir.parent

load_dotenv(dotenv_path=root_dir / ".env")

if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

# Import your custom ChatEngine
# Ensure this matches the file name where ChatEngine is defined (e.g., engine_deploy or engine)
from rag_engine.ingestion_pipeline import ChatEngine 

async def run_comprehensive_apple_test():
    """
    Expanded Test Suite for Apple 10-K analysis.
    Covers: Risks, Financials, Math, and Multi-turn Reasoning.
    """
    print("🌲 Initializing Apple 10-K Stress Test (Pinecone Mode)...")
    
    engine = ChatEngine(provider="OpenAI", model_name="gpt-4o-mini")
    
    # --- Step 1: Loading Knowledge Base (Mirroring main.py logic) ---
    print("\n--- Step 1: Connecting to Pinecone ---")
    try:
        engine.load_pinecone_knowledge_base()
        engine.create_conversational_chain()
        print("✅ Connection and Chain ready.")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # --- EXPANDED TEST QUERIES (CLEANED FOR APPLE) ---
    test_categories = {
        "Strategic Risks": [
            "What are the primary risks Apple identifies related to its reliance on third-party components and manufacturing?",
            "How does Apple describe risks associated with intense competition in the smartphone and services markets?",
            "What supply chain and fulfillment risks does Apple highlight, particularly regarding its concentration in China?",
            "How does Apple assess the impact of global geopolitical tensions and trade tariffs on its international operations?",
            "What risks does Apple identify related to labor practices and human rights within its global supply chain?",
            "How does Apple describe regulatory risks tied to App Store antitrust scrutiny in the U.S. and EU?",
            "What are the risks associated with data security and the protection of customer privacy across Apple's ecosystem?"
        ],
        "Financial Performance": [
            "What was Apple’s total net sales for the fiscal year, and how did it change year over year?",
            "How did Services revenue growth compare to iPhone and Wearables segment growth?",
            "What were the primary drivers of gross margin changes this year?",
            "How much did Apple spend on Research and Development (R&D) this year compared to last year?",
            "What portion of total revenue came from the Greater China region?",
            "How did 'Wearables, Home and Accessories' revenue change compared to the prior year?",
            "What were the main factors affecting changes in operating expenses this year?"
        ],
        "Math & Tabular Data": [
            "What is Apple’s current ratio (Current Assets / Current Liabilities) for the fiscal year end?",
            "What was the effective income tax rate reported by Apple for the year?",
            "How much cash, cash equivalents, and marketable securities did Apple hold at year-end?",
            "What was the year-over-year percentage change in net income?",
            "How did capital expenditures compare to the prior year?",
            "What was the percentage contribution of 'Services' to total net sales?"
        ],
        "Segment & Geographic Analysis": [
            "Break down net sales by reportable segment: iPhone, Mac, iPad, Wearables, and Services.",
            "Which geographic region (Americas, Europe, Greater China, Japan, Rest of Asia) grew the fastest?",
            "How does Apple explain the profitability of its Services segment vs. Products segment?",
            "What are the key cost structures unique to Apple's Services business?",
            "How does the strengthening or weakening of the U.S. dollar impact Apple’s international results?"
        ],
        "Legal & Regulatory": [
            "Summarize any material legal proceedings disclosed in Item 3 or the Commitments and Contingencies note.",
            "What antitrust or competition-related investigations does Apple disclose regarding the App Store?",
            "How does Apple describe compliance risks related to the Digital Markets Act (DMA) in the EU?",
            "Are there any disclosed risks related to intellectual property infringement or patent litigation?",
            "What regulatory risks does Apple associate with government requests for user data?"
        ],
        "Future Outlook & Forward-Looking Statements": [
            "What does management identify as the biggest uncertainties affecting future growth in Services?",
            "How does Apple describe expected trends in consumer demand for the iPhone over the next year?",
            "What are Apple’s expectations for continued investment in silicon development and AI?",
            "How does management discuss inflation and component cost fluctuations affecting margins?",
            "What forward-looking statements are made about 'Apple Intelligence' or generative AI adoption?"
        ],
        "Capital Allocation & Liquidity": [
            "How does Apple plan to manage its 'Net Cash Neutral' goal over time?",
            "What are Apple’s priorities regarding share repurchases and dividends?",
            "How much debt did Apple issue or repay during the fiscal year?",
            "How does management assess its ability to generate sufficient cash flow for operations?",
            "What credit facilities or commercial paper programs does Apple rely on?"
        ],
        "Multi-Turn Reasoning (Context Stress Test)": [
            "What does Apple say about the seasonality of its business and the holiday quarter?",
            "How does seasonality specifically affect the timing of iPhone launches and manufacturing ramps?",
            "What operational strategies does Apple use to manage inventory during peak-season demand?",
            "How do these peak-season trends impact short-term inventory turnover ratios?",
            "What long-term investments in manufacturing capacity are justified by this seasonal behavior?"
        ],
        "Nuance & Intent": [
            "Does Apple frame its privacy features primarily as a competitive advantage or a regulatory burden?",
            "How does Apple describe the strategic importance of its vertical integration (Apple Silicon) to the company?",
            "What language does Apple use to discuss potential margin pressure from hardware commoditization?",
            "How does Apple define 'Environmental Social and Governance' (ESG) goals within the filing?",
            "Are there areas where Apple emphasizes long-term ecosystem loyalty while downplaying short-term hardware cycles?"
        ]
    }

    session_id = "apple_pinecone_stress_test_001"
    
    print(f"\n--- Step 2: Executing Tests ---")

    for category, queries in test_categories.items():
        print(f"\n>>> CATEGORY: {category}")
        print("="*40)
        
        for query in queries:
            try:
                # Using the async method from your engine
                result = await engine.ask_with_sources_async(query, session_id=session_id)
                
                print(f"Q: {query}")
                print(f"A: {result.get('answer', 'No answer returned')}")
                
                # Format sources for clean printing
                sources = result.get('sources', [])
                source_names = ', '.join([s.split('/')[-1] for s in sources]) if sources else "None"
                print(f"Sources: {source_names}")
                print("-" * 20)
                
            except Exception as e:
                print(f"Error on query: {query}\nDetails: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_comprehensive_apple_test())
    except KeyboardInterrupt:
        print("\nTest cancelled.")