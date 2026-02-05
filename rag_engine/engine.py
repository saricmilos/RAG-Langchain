import os
import sys
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import asyncio

# 1. Setup Environment and Path Logic
current_file_path = Path(__file__).resolve()
app_dir = current_file_path.parent
root_dir = app_dir.parent

load_dotenv(dotenv_path=root_dir / ".env")

# 2. LangChain Imports
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Import your custom functions
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

from langchain_functions import (
    langchain_document_loader,
    create_advanced_sec_splitter,
    get_vectorstore_retriever,
    create_compression_retriever,
    instantiate_llm
)

# --- PROMPT TEMPLATES ---

REWRITER_SYSTEM_PROMPT = """
<task_context>
You are an expert Query Transformation Engine. Your goal is to convert a conversational follow-up into a context-independent Standalone Question for a RAG pipeline.
</task_context>

<rules>
1. CORE_TASK: Use 'Chat History' to resolve ambiguity in the 'Follow-up Input'.
2. NO_CHAT: Do NOT answer the question. Do NOT include pleasantries (e.g., "Sure, here it is").
3. PRONOUN_RESOLUTION: Replace 'it', 'they', 'that process', etc., with the specific entities mentioned in history.
4. LANGUAGE_PARITY: Always output in the same language as the Follow-up Input.
5. NO_CHANGE_REQUIRED: If the Follow-up Input is already a clear standalone question, return it exactly as-is.
</rules>

<output_format>
Return ONLY the rephrased string. No additional text.
</output_format>
"""

def get_answer_template(language: str = "English") -> ChatPromptTemplate:
    """Factory function for the final generation prompt."""
    system_instruction = (
        "You are a precise assistant. Answer the user's question using ONLY the "
        "provided context delimited by <context></context> tags.\n\n"
        "If the answer is not in the context, say you don't know. "
        f"Your response must be written in {language}."
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Context:\n<context>\n{context}\n</context>\n\nQuestion: {input}")
    ])

# --- MAIN ENGINE ---

class ChatEngine:
    """
    State-of-the-art Conversational RAG Engine using 
    History-Aware Retrieval and Document Stuffing.
    """
    
    def __init__(
        self, 
        data_path: str = None, 
        provider: str = "OpenAI", 
        model_name: str = "gpt-4o-mini",
        api_key: str = None
    ):
        self.data_path = data_path or str(root_dir / "data" / "docs")
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API Key for {provider} not found!")

        print("Loading local embedding model: sentence-transformers/all-MiniLM-L6-v2...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        self.vector_store = None
        self.chat_chain = None
        self.history_store = {} # Session ID -> InMemoryChatMessageHistory

    def get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def build_knowledge_base(self, force_rebuild: bool = False):
            persist_path = root_dir / "data" / "vector_stores" / "faiss_index"
            processed_companies = set()
            
            # 1. Load existing and identify what is already indexed
            if not force_rebuild and persist_path.exists():
                tqdm.write(f"Loading main index from {persist_path}")
                self.vector_store = FAISS.load_local(
                    str(persist_path), self.embeddings, allow_dangerous_deserialization=True
                )
                
                # Extract unique companies from existing metadata
                # docstore contains all the documents currently in the index
                if self.vector_store.docstore:
                    processed_companies = {
                        doc.metadata.get("company_name") 
                        for doc in self.vector_store.docstore._dict.values()
                        if "company_name" in doc.metadata
                    }
                    tqdm.write(f"Detected {len(processed_companies)} companies already in index.")
            else:
                tqdm.write("Starting a fresh build...")
                self.vector_store = None

            edgar_path = Path(self.data_path)
            if (edgar_path / "sec-edgar-filings").exists():
                edgar_path = edgar_path / "sec-edgar-filings"
                
            # Filter out folders that are already in the processed_companies set
            all_folders = [f for f in edgar_path.iterdir() if f.is_dir()]
            company_folders = [f for f in all_folders if f.name not in processed_companies]
            
            if not company_folders:
                tqdm.write("✨ All companies are already indexed. Nothing to do!")
                return self.vector_store

            tqdm.write(f"Queue: {len(company_folders)} new companies to process.")
            sec_splitter = create_advanced_sec_splitter()

            # 2. Main Loop
            for folder in tqdm(company_folders, desc="🚀 Overall Progress", unit="company"):
                tqdm.write(f"\n--- Processing: {folder.name} ---")
                
                company_docs = langchain_document_loader(folder)
                if not company_docs:
                    continue

                company_chunks = []
                for doc in company_docs:
                    doc.metadata["source"] = Path(doc.metadata.get("source", "unknown")).name
                    doc.metadata["company_name"] = folder.name # Critical for resume logic
                    
                    chunks = sec_splitter.split_documents([doc])
                    company_chunks.extend(chunks)

                if not company_chunks:
                    continue

                # 3. Batch Embedding with Inner Progress Bar
                batch_size = 50
                total_chunks = len(company_chunks)
                
                # Use small slice for initial DB creation
                temp_db = FAISS.from_documents(company_chunks[:batch_size], self.embeddings)
                
                if total_chunks > batch_size:
                    for i in tqdm(range(batch_size, total_chunks, batch_size), 
                                desc=f"   └─ Embedding {folder.name}", 
                                leave=False):
                        temp_db.add_documents(company_chunks[i : i + batch_size])

                # 4. Merge and Save
                if self.vector_store is None:
                    self.vector_store = temp_db
                else:
                    self.vector_store.merge_from(temp_db)
                
                persist_path.parent.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(persist_path))
                
                tqdm.write(f"✅ Indexed {folder.name}. Total vectors: {self.vector_store.index.ntotal}")

            return self.vector_store

    def create_conversational_chain(self, strategy: str = "similarity"):
        if not self.vector_store:
            self.build_knowledge_base()
        
        # 2. Critical Check: If still None, the data directory was likely empty
        if not self.vector_store:
            raise ValueError(
                f"Vector store could not be initialized. Please check if '{self.data_path}' "
                "contains valid documents or if the index exists."
            )
        
        # 1. Selection of Retriever
        if strategy == "mmr":
            base_retriever = get_vectorstore_retriever(
                self.vector_store, search_type="mmr", k=4, fetch_k=20
            )
        elif strategy == "compression":
            standard_retriever = get_vectorstore_retriever(self.vector_store, k=10)
            base_retriever = create_compression_retriever(
                self.embeddings, standard_retriever, k=4
            )
        else: # Default: Similarity
            base_retriever = get_vectorstore_retriever(self.vector_store, k=4)

        # 2. Setup Specialized LLMs
        llm_rephraser = instantiate_llm(
            provider=self.provider, api_key=self.api_key, 
            model_name="gpt-4o", temperature=0.0
        )
        llm_generator = instantiate_llm(
            provider=self.provider, api_key=self.api_key, 
            model_name=self.model_name, temperature=0.5
        )

        # 3. History-Aware Retrieval (Wraps your chosen base_retriever)
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "<follow_up_input>\n{input}\n</follow_up_input>"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm=llm_rephraser,
            retriever=base_retriever, # Plug in the strategy-based retriever here
            prompt=rephrase_prompt
        )

        # 4. Final RAG Pipeline Integration
        qa_prompt = get_answer_template(language="English")
        question_answer_chain = create_stuff_documents_chain(
            llm=llm_generator,
            prompt=qa_prompt
        )

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 5. Wrap with State Management
        self.chat_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
# New Async version of your function
    async def ask_with_sources_async(self, question: str, session_id: str = "default_session"):
        if not self.chat_chain:
            self.create_conversational_chain()
        
        # We use ainvoke here for non-blocking performance
        response = await self.chat_chain.ainvoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
    
        return {
            "answer": response["answer"],
            "sources": list(set(doc.metadata.get("source", "Unknown") for doc in response["context"]))
        }
    
async def main():
    engine = ChatEngine()
    engine.build_knowledge_base()
    
    # 1. First Turn
    print(f"--- TURN 1 ---")
    query1 = "How did a?"
    # You MUST 'await' the async function
    res1 = await engine.ask_with_sources_async(query1, 'user_123')
    
    print(f"User: {query1}")
    print(f"AI: {res1['answer']}")
    print(f"Sources: {res1['sources']}") 
    
    # 2. Second Turn
    print(f"\n--- TURN 2 ---")
    query2 = "Who can build me one?"
    res2 = await engine.ask_with_sources_async(query2, 'user_123')
    
    print(f"User: {query2}")
    print(f"AI: {res2['answer']}")
    print(f"Sources: {res2['sources']}")

if __name__ == "__main__":
    # Use asyncio.run to execute the async main function
    asyncio.run(main())