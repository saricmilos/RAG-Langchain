from sec_edgar_downloader import Downloader
import time

# SEC requirement: "Name Email"
dl = Downloader("MilosResearchBot", "milos@example.com", "data/docs")

# Top 10 High-Value Tickers for 2026 Financial RAG
tickers = [
    "NVDA",  # NVIDIA
    "AAPL",  # Apple
    "GOOGL", # Alphabet/Google
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "META",  # Meta
    "TSLA",  # Tesla
    "AVGO",  # Broadcom
    "BRK-B", # Berkshire Hathaway
    "LLY"    # Eli Lilly
]

filing_types = ["10-K", "10-Q", "8-K", "DEF 14A"]

for ticker in tickers:
    print(f"--- Starting Download for {ticker} ---")
    for f_type in filing_types:
        try:
            # Set a 'limit' or 'after' date to prevent downloading 30 years of history
            print(f"  Fetching {f_type}...")
            dl.get(f_type, ticker, after="2025-01-01", download_details=False)
            
            # SEC fair access: stay under 10 requests per second
            time.sleep(0.2) 
        except Exception as e:
            print(f"  Error fetching {f_type} for {ticker}: {e}")
            
print("\n✅ All downloads complete. Check data/docs/sec-edgar-filings/")