import os
import sys
from pathlib import Path

# Add the /app directory to sys.path so engine.py imports work
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from rag_engine.ingestion_pipeline import ChatEngine

def test_knowledge_base(force_rebuild: bool = False):
    """
    Tests the ChatEngine knowledge base building process.
    
    Args:
        force_rebuild (bool): If True, re-processes documents and re-embeds.
                             If False, attempts to load from existing FAISS index.
    """
    # 1. Path Resolution
    BASE_DIR = current_dir.parent
    DOCS_DIR = BASE_DIR / "data" / "docs"
    # Note: We point to the folder containing the index files
    STORE_DIR = BASE_DIR / "data" / "vector_stores" / "faiss_index"
    
    print(f"--- Configuration ---")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Target Docs:    {DOCS_DIR}")
    print(f"Rebuild Mode:   {force_rebuild}")

    if not DOCS_DIR.exists():
        print(f"FAILURE: Directory not found at {DOCS_DIR}")
        return

    # 2. Initialize Engine
    engine = ChatEngine(data_path=str(DOCS_DIR))

    print("\n--- Starting build_knowledge_base Test ---")
    try:
        # Pass the dynamic rebuild flag to your engine method
        vector_store = engine.build_knowledge_base(force_rebuild=force_rebuild)
        
        if vector_store:
            # Check 1: Vector Count
            total_vectors = vector_store.index.ntotal
            print(f"STATUS: Success. {total_vectors} chunks indexed.")
            
            # Check 2: Persistence verification
            index_file = STORE_DIR / "index.faiss"
            if index_file.exists():
                print(f"STATUS: Index verified on disk at {index_file}")
            else:
                print("WARNING: Persistence file not detected on disk.")

            # Check 3: Metadata Integrity & Retrieval
            # Performing a generic search to verify the structure of the data
            results = vector_store.similarity_search("context", k=1)
            if results:
                source = results[0].metadata.get('source', 'No Source Found')
                print(f"VERIFICATION: Retrieval functional. Sample source: {source}")
                
                # Verify hierarchical headers (metadata check)
                headers = {k: v for k, v in results[0].metadata.items() if 'Header' in k or 'header' in k.lower()}
                if headers:
                    print(f"VERIFICATION: Hierarchical headers found: {headers}")
                else:
                    print("INFO: No specific header metadata found in this chunk.")
        else:
            print("FAILURE: build_knowledge_base returned None.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # SET THIS TO TRUE: If you added new files or changed splitting logic.
    # SET THIS TO FALSE: For fast testing and to verify the "load from disk" logic.
    test_knowledge_base(force_rebuild=True)