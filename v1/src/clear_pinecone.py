from pinecone import Pinecone
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Path Resolution
# Current file is in /root/rag_engine/sync_to_pinecone.py
current_file_path = Path(__file__).resolve()
rag_engine_dir = current_file_path.parent
root_dir = rag_engine_dir.parent  # This points to the Root Folder

# Load .env from the root folder
load_dotenv(dotenv_path=root_dir / ".env")

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

index_name = "sec-filings-index"

if pc.has_index(index_name):
    pc.delete_index(index_name)
    print(f"✅ Index '{index_name}' deleted. You can start fresh now.")
else:
    print(f"Index '{index_name}' does not exist.")