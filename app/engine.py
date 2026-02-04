import os
import sys
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

# 1. Setup Environment and Path Logic
# Get the directory where engine.py resides (main_folder/app)
current_file_path = Path(__file__).resolve()
app_dir = current_file_path.parent
root_dir = app_dir.parent

# Load .env from the root directory
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("CRITICAL: OPENAI_API_KEY not found in .env file!")

# 2. LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# Import your custom functions
# Ensure app_dir is in sys.path so it can find langchain_functions
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

from langchain_functions import (
    langchain_document_loader,
    create_advanced_markdown_splitter,
    create_vectorstore
)

class ChatEngine:
    """
    A conversational AI engine that combines document retrieval with LLM generation.
    """
    
    def __init__(self, data_path: str = None, model_name: str = "gpt-4o-mini"):
        # Default to root_dir/data/docs if no path provided
        self.data_path = data_path or str(root_dir / "data" / "docs")
        self.model_name = model_name
        
        # Initialize Embeddings
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.chat_chain = None
        
        # Initialize Memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
    
    def build_knowledge_base(self, force_rebuild: bool = False):
        """
        Builds or loads a vector store with persistence and error handling.
        """
        # 1. Define persistence path (using your VECTOR_STORE_DIR logic)
        persist_path = root_dir / "data" / "vector_stores" / "faiss_index"

        # 2. Check if we can just load the existing index
        if not force_rebuild and persist_path.exists():
            print(f"Loading existing vector store from {persist_path}")
            self.vector_store = FAISS.load_local(
                str(persist_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            return self.vector_store

        # 3. Ingestion logic
        print(f"Reading documents from: {self.data_path}")
        documents = langchain_document_loader(self.data_path)
        
        if not documents:
            print(f"Warning: No documents found in {self.data_path}")
            return None

        # Reporting
        counts = Counter(doc.metadata.get('source', '').split('.')[-1] for doc in documents)
        print(f"\n--- Knowledge Base Load Summary ---")
        print(f"Total Documents: {len(documents)}")
        for ext, count in counts.items():
            print(f"  * {ext.upper() or 'UNKNOWN'}: {count} files")

        # 4. Optimized Hierarchical Splitting
        md_splitter, rec_splitter = create_advanced_markdown_splitter()
        final_chunks = []

        for doc in documents:
            # Skip empty docs to prevent vector store errors
            if not doc.page_content.strip():
                continue

            # Stage A: Header-aware split
            header_docs = md_splitter.split_text(doc.page_content)
            
            # Transfer metadata
            for h_doc in header_docs:
                h_doc.metadata.update(doc.metadata)
                
            # Stage B: Recursive chunking
            sub_chunks = rec_splitter.split_documents(header_docs)
            final_chunks.extend(sub_chunks)
        
        # 5. Build and Save
        if final_chunks:
            self.vector_store = FAISS.from_documents(final_chunks, self.embeddings)
            
            # Save locally so the next run is instant
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(persist_path))
            
            print(f"Successfully indexed and saved {len(final_chunks)} chunks.")
        else:
            print("No valid text chunks were created. Vector store not updated.")

        return self.vector_store
    
    def create_conversational_chain(self):
        """
        Step 2: Reasoning & Generation
        """
        if not self.vector_store:
            self.build_knowledge_base()
        
        llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        
        self.chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory,
            verbose=True
        )
    
    def ask(self, question: str) -> str:
        """
        Interface for asking questions.
        """
        if not self.chat_chain:
            self.create_conversational_chain()
        
        response = self.chat_chain.invoke({"question": question})
        return response["answer"]
    
    def reset_memory(self):
        self.memory.clear()

    def get_chat_history(self):
        return self.memory.chat_memory.messages

if __name__ == "__main__":
    # Test block
    engine = ChatEngine()
    
    # Verify build
    engine.build_knowledge_base()
    
    # Run a test query if documents exist
    if engine.vector_store:
        answer = engine.ask("Who is Milos Saric?")
        print(f"Answer: {answer}")