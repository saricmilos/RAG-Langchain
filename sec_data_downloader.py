from sec_edgar_downloader import Downloader

# SEC requires: "Name Email"
dl = Downloader("MilosResearchBot", "milos@example.com", "data/docs")
ticker = "AAPL"

# 1. Annual and Quarterly (The ones you already have)
print(f"Downloading 10-K and 10-Q for {ticker}...")
dl.get("10-K", ticker, after="2019-01-01")
dl.get("10-Q", ticker, after="2019-01-01")

# 2. Current Reports (The 'Breaking News' of finance)
print(f"Downloading 8-K for {ticker}...")
dl.get("8-K", ticker, after="2023-01-01") # Just last 3 years for relevance

# 3. Proxy Statements (Governance & Executive Pay)
print(f"Downloading DEF 14A for {ticker}...")
dl.get("DEF 14A", ticker, after="2020-01-01")

print("Done! Check your 'data/docs/sec-edgar-filings' folder.")