import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Path & Env Setup
current_file_path = Path(__file__).resolve()
app_dir = current_file_path.parent
root_dir = app_dir.parent

# 2. LangChain & LangSmith Imports
from langsmith import traceable  # <--- Added for LangSmith
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

# Import helper functions
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

from functions_deploy import (
    get_vectorstore_retriever,
    create_compression_retriever,
    instantiate_llm
)

# --- PROMPT TEMPLATES ---

REWRITER_SYSTEM_PROMPT = """
You are an expert Query Transformation Engine. 
Convert the conversational follow-up into a context-independent Standalone Question.
Use 'Chat History' to resolve pronouns (it, they, etc.).
Return ONLY the rephrased string. No extra text.
"""

def get_answer_template(language: str = "English") -> ChatPromptTemplate:
    system_instruction = (
        "You are a precise financial assistant. Answer the user's question using ONLY the "
        "provided context inside <context> tags.\n\n"
        "If the answer is not in the context, say you don't know.\n"
        f"Answer in {language}."
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Context:\n<context>\n{context}\n</context>\n\nQuestion: {input}")
    ])

# --- MAIN ENGINE ---

class ChatEngine:
    def __init__(
        self, 
        provider: str = "OpenAI", 
        model_name: str = "gpt-4o-mini",
        api_key: str = None
    ):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        if not self.api_key:
            raise ValueError(f"API Key for {provider} not found!")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in .env!")

        print("Loading HuggingFace embedding model...")
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
        )
        
        self.vector_store = None
        self.chat_chain = None
        self.history_store = {} 

    def load_pinecone_knowledge_base(self):
        """Connects to the existing Pinecone index."""
        print(f"🌲 Connecting to Pinecone index: {self.index_name}...")
        try:
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                pinecone_api_key=self.pinecone_api_key
            )
            print("✅ Success: Connected to Pinecone.")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pinecone: {e}")
            
    def get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def create_conversational_chain(self, strategy: str = "similarity"):
        """Builds the full RAG pipeline."""
        if not self.vector_store:
            self.load_pinecone_knowledge_base()
        
        # 1. Retriever Selection
        if strategy == "mmr":
            base_retriever = get_vectorstore_retriever(self.vector_store, search_type="mmr", k=4)
        elif strategy == "compression":
            standard_retriever = get_vectorstore_retriever(self.vector_store, k=10)
            base_retriever = create_compression_retriever(self.embeddings, standard_retriever, k=4)
        else: 
            base_retriever = get_vectorstore_retriever(self.vector_store, k=4)

        # 2. Setup LLMs
        llm_rephraser = instantiate_llm(self.provider, self.api_key, "gpt-4o", 0.0)
        llm_generator = instantiate_llm(self.provider, self.api_key, self.model_name, 0.5)

        # 3. History-Aware Retrieval
        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm=llm_rephraser,
            retriever=base_retriever,
            prompt=rephrase_prompt
        )

        # 4. Final Answer Chain
        qa_prompt = get_answer_template(language="English")
        question_answer_chain = create_stuff_documents_chain(llm=llm_generator, prompt=qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 5. History Wrapper
        self.chat_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    # --- ADDED TRACEABLE HERE ---
    @traceable(name="RAG_Brain_Process")
    async def ask_with_sources_async(self, question: str, session_id: str = "default_session"):
        """Async entry point for FastAPI."""
        if not self.chat_chain:
            self.create_conversational_chain()
        
        # LangChain automatically traces child runs when environment variables are set.
        # This will show 'RunnableWithMessageHistory' and its nested 'RetrievalChain'.
        response = await self.chat_chain.ainvoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
    
        # Extract unique sources from context documents
        context_docs = response.get("context", [])
        sources = sorted(list(set(doc.metadata.get("source", "Unknown") for doc in context_docs)))

        return {
            "answer": response["answer"],
            "sources": sources
        }
