import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Vector Stores
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings

# Custom Functions
from langchain_functions import (
    langchain_document_loader,
    create_advanced_sec_splitter
)

load_dotenv()

class SECIngestionPipeline:
    def __init__(self, data_path="./data/docs"):
        self.data_path = Path(data_path)
        print("Loading local embedding model: sentence-transformers/all-MiniLM-L6-v2...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.splitter = create_advanced_sec_splitter()

    def _get_company_folders(self, processed_companies=None):
        """Standardizes pathing and filters out already processed companies."""
        edgar_path = self.data_path
        if (edgar_path / "sec-edgar-filings").exists():
            edgar_path = edgar_path / "sec-edgar-filings"
        
        all_folders = [f for f in edgar_path.iterdir() if f.is_dir()]
        
        if processed_companies:
            return [f for f in all_folders if f.name not in processed_companies]
        return all_folders

    def build_pinecone_cloud(self, index_name="sec-filings-index"):
        """Syncs local files to Pinecone with Resuming Logic and Progress Bars."""
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # 1. Initialize/Check Index
        if not pc.has_index(index_name):
            tqdm.write(f"Creating new Pinecone index: {index_name}...")
            pc.create_index(
                name=index_name, dimension=384, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        index = pc.Index(index_name)
        vstore = PineconeVectorStore(index_name=index_name, embedding=self.embeddings)

        # 2. Resuming Logic: Fetch existing companies from Pinecone metadata
        processed_companies = set()
        stats = index.describe_index_stats()
        if stats['total_vector_count'] > 0:
            tqdm.write("Checking Pinecone for existing companies to skip...")
            # Querying metadata to find unique company names already present
            results = index.query(vector=[0.0]*384, top_k=10000, include_metadata=True)
            for match in results['matches']:
                if 'company_name' in match['metadata']:
                    processed_companies.add(match['metadata']['company_name'])
            tqdm.write(f"Found {len(processed_companies)} companies already in the cloud.")

        folders = self._get_company_folders(processed_companies)
        if not folders:
            tqdm.write("✨ All companies are already synced!")
            return vstore

        tqdm.write(f"🚀 Found {len(folders)} new folders to process.")

        # 3. Main Loop with Progress Bars
        for folder in tqdm(folders, desc="Overall Progress", unit="company"):
            tqdm.write(f"\n--- Processing & Uploading: {folder.name} ---")
            
            docs = langchain_document_loader(folder)
            if not docs:
                tqdm.write(f"⚠️ No documents found for {folder.name}, skipping.")
                continue
            
            # Enrich metadata
            for d in docs:
                d.metadata["company_name"] = folder.name
                d.metadata["source"] = Path(d.metadata.get("source", "unknown")).name

            chunks = self.splitter.split_documents(docs)
            if not chunks:
                continue

            # 4. Batch Upload with Inner Progress Bar
            batch_size = 50
            for i in tqdm(range(0, len(chunks), batch_size), 
                          desc=f" └─ Uploading {folder.name}", 
                          leave=False):
                batch = chunks[i : i + batch_size]
                vstore.add_documents(batch)
            
            tqdm.write(f"✅ {folder.name} synced. Total vectors in index: {index.describe_index_stats()['total_vector_count']}")

        return vstore

    def build_local_faiss(self, index_path="./data/faiss_index"):
        """Builds local FAISS with resume logic and progress bars."""
        persist_path = Path(index_path)
        processed_companies = set()
        vector_store = None

        if persist_path.exists():
            vector_store = FAISS.load_local(str(persist_path), self.embeddings, allow_dangerous_deserialization=True)
            processed_companies = {
                doc.metadata.get("company_name") 
                for doc in vector_store.docstore._dict.values() 
                if "company_name" in doc.metadata
            }
            tqdm.write(f"Detected {len(processed_companies)} companies in local FAISS.")

        folders = self._get_company_folders(processed_companies)
        if not folders:
            tqdm.write("✨ Local FAISS is already up to date!")
            return vector_store

        for folder in tqdm(folders, desc="Local FAISS Build"):
            docs = langchain_document_loader(folder)
            if not docs: continue
            for d in docs: d.metadata["company_name"] = folder.name
            
            chunks = self.splitter.split_documents(docs)
            
            # Inner Progress for Embedding
            batch_size = 50
            temp_db = FAISS.from_documents(chunks[:batch_size], self.embeddings)
            if len(chunks) > batch_size:
                for i in tqdm(range(batch_size, len(chunks), batch_size), desc=" └─ Embedding", leave=False):
                    temp_db.add_documents(chunks[i:i+batch_size])

            if vector_store is None:
                vector_store = temp_db
            else:
                vector_store.merge_from(temp_db)
            
            vector_store.save_local(str(persist_path))
        
        return vector_store

if __name__ == "__main__":
    pipeline = SECIngestionPipeline()
    # pipeline.build_local_faiss()
    pipeline.build_pinecone_cloud()