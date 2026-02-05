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
from engine import ChatEngine

async def run_comprehensive_apple_test():
    """
    Expanded Test Suite for Apple 10-K analysis.
    Covers: Risks, Financials, Math, and Multi-turn Reasoning.
    """
    print("Initializing Comprehensive Apple 10-K Stress Test...")
    
    engine = ChatEngine(provider="OpenAI", model_name="gpt-4o-mini")
    
    # Load the Knowledge Base
    print("\n--- Step 1: Loading Knowledge Base ---")
    engine.build_knowledge_base(force_rebuild=False)
    
    # --- EXPANDED TEST QUERIES ---
    test_categories = {
        "Strategic Risks": [
            "What are the primary risks Amazon identifies related to its reliance on third-party sellers?",
            "How does Amazon describe risks associated with AWS competition and pricing pressure?",
            "What supply chain and fulfillment risks does Amazon highlight, particularly during peak seasons?",
            "How does Amazon assess the impact of global geopolitical tensions on its international operations?",
            "What risks does Amazon identify related to labor availability, unionization, and workforce costs?",
            "How does Amazon describe regulatory risks tied to antitrust scrutiny in the U.S. and EU?",
            "What are the risks associated with data security and customer trust across Amazon’s platforms?"
        ],
        "Financial Performance": [
            "What was Amazon’s total net sales for the fiscal year, and how did it change year over year?",
            "How did AWS revenue growth compare to North America and International segment growth?",
            "What were the primary drivers of operating income (or loss) in each reportable segment?",
            "How much did Amazon spend on Technology and Content this year, and how does it compare to last year?",
            "What portion of total revenue came from subscription services such as Prime?",
            "How did advertising services revenue change compared to the prior year?",
            "What were the main factors affecting changes in operating expenses this year?"
        ],
        "Math & Tabular Data": [
            "What is Amazon’s current ratio (Current Assets / Current Liabilities) for the fiscal year?",
            "What was the effective income tax rate reported for the year?",
            "How much cash, cash equivalents, and marketable securities did Amazon hold at year-end?",
            "What was the year-over-year percentage change in operating cash flow?",
            "How did capital expenditures compare to the prior year?",
            "What was the percentage contribution of AWS to total operating income?"
        ],
        "Segment & Geographic Analysis": [
            "Break down net sales by reportable segment: North America, International, and AWS.",
            "Which geographic region experienced the fastest revenue growth?",
            "How does Amazon explain differences in profitability between North America and International segments?",
            "What are the key cost structures unique to AWS compared to retail segments?",
            "How does currency fluctuation impact Amazon’s international results?"
        ],
        "Legal & Regulatory": [
            "Summarize any material legal proceedings disclosed in Item 3 of the 10-K.",
            "What antitrust or competition-related investigations does Amazon disclose?",
            "How does Amazon describe compliance risks related to data privacy laws such as GDPR?",
            "Are there any disclosed risks related to product liability or third-party seller misconduct?",
            "What regulatory risks does Amazon associate with cloud computing and government contracts?"
        ],
        "Future Outlook & Forward-Looking Statements": [
            "What does management identify as the biggest uncertainties affecting future growth?",
            "How does Amazon describe expected trends in consumer demand over the next year?",
            "What are Amazon’s expectations for continued investment in AWS infrastructure?",
            "How does management discuss inflation and interest rates affecting costs and pricing?",
            "What forward-looking statements are made about automation and AI adoption?"
        ],
        "Capital Allocation & Liquidity": [
            "How does Amazon plan to manage its liquidity over the next 12 months?",
            "What are Amazon’s priorities regarding capital expenditures going forward?",
            "Does Amazon discuss share repurchases or debt repayment strategies?",
            "How does management assess its ability to generate sufficient cash flow?",
            "What credit facilities or debt instruments does Amazon rely on?"
        ],
        "Governance & Compensation": [
            "Who are Amazon’s named executive officers?",
            "How is executive compensation structured to align with long-term performance?",
            "What role does the Board play in overseeing risk management?",
            "Does Amazon have a committee overseeing technology, AI, or innovation?",
            "What policies govern executive stock ownership and trading?"
        ],
        "Multi-Turn Reasoning (Context Stress Test)": [
            "What does Amazon say about the seasonality of its business?",
            "How does seasonality specifically affect fulfillment costs and margins?",
            "What operational strategies does Amazon use to manage peak-season demand?",
            "How do these strategies impact short-term profitability?",
            "What long-term investments are justified by this seasonal behavior?"
        ],
        "Nuance & Intent": [
            "Does Amazon frame Generative AI primarily as a growth opportunity or a competitive risk?",
            "How does Amazon describe the strategic importance of AWS to the broader company?",
            "What language does Amazon use to discuss potential margin pressure versus long-term value creation?",
            "How does Amazon define ‘customer obsession’ in operational terms within the filing?",
            "Are there areas where Amazon downplays risks while emphasizing long-term strategy?"
        ]
    }

    session_id = "apple_stress_test_001"
    
    print(f"\n--- Step 2: Executing Tests ---")

    for category, queries in test_categories.items():
        print(f"\n>>> CATEGORY: {category}")
        print("="*40)
        
        for query in queries:
            try:
                result = await engine.ask_with_sources_async(query, session_id=session_id)
                
                print(f"Q: {query}")
                print(f"A: {result['answer']}")
                print(f"Sources: {', '.join([s.split('/')[-1] for s in result['sources']])}")
                print("-" * 20)
                
            except Exception as e:
                print(f"Error on query: {query}\nDetails: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_comprehensive_apple_test())
    except KeyboardInterrupt:
        print("\nTest cancelled.")