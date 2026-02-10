import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Path Resolution
current_file_path = Path(__file__).resolve()
rag_engine_dir = current_file_path.parent
root_dir = rag_engine_dir.parent 

# Add rag_engine to sys.path so Python can find ingestion_pipeline.py
if str(rag_engine_dir) not in sys.path:
    sys.path.append(str(rag_engine_dir))

from ingestion_pipeline import SECIngestionPipeline

def run_pinecone_sync():
    load_dotenv(dotenv_path=root_dir / ".env")
    docs_dir = root_dir / "data" / "docs"

    print(f"--- Syncing from: {docs_dir} ---")

    if not docs_dir.exists():
        print(f"❌ Error: {docs_dir} not found.")
        return

    # Initialize and Run
    pipeline = SECIngestionPipeline(data_path=str(docs_dir))
    
    try:
        index_name = os.getenv("PINECONE_INDEX_NAME", "sec-filings-index")
        pipeline.build_pinecone_cloud(index_name=index_name)
        print(f"\n✅ Sync Complete.")
    except Exception as e:
        print(f"\n❌ Sync Failed: {e}")

if __name__ == "__main__":
    run_pinecone_sync()