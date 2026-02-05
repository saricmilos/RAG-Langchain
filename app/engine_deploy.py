import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Path & Env Setup
# Ensures paths work regardless of where the script is executed
current_file_path = Path(__file__).resolve()
app_dir = current_file_path.parent
root_dir = app_dir.parent

# 2. LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Import your helper functions from the 'app' directory
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API Key not found. Check your .env file.")

        # Switch back to the model that matches your 29,802 saved vectors
        print("Loading HuggingFace embedding model to match existing index...")
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
        )
        
        self.vector_store = None
        self.chat_chain = None
        self.history_store = {} 

    def get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def load_offline_knowledge_base(self):
        """Loads the pre-built FAISS index from the specific folder."""
        persist_path = root_dir / "data" / "vector_stores" / "faiss_index"
        
        if (persist_path / "index.faiss").exists():
            print(f"📦 Loading pre-built index from: {persist_path}")
            # allow_dangerous_deserialization is required for loading local pickle files (.pkl)
            self.vector_store = FAISS.load_local(
                str(persist_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"✅ Success: Loaded {self.vector_store.index.ntotal} vectors.")
        else:
            raise FileNotFoundError(f"FAISS index files missing at: {persist_path}")
    
    def create_conversational_chain(self, strategy: str = "similarity"):
        """Builds the full RAG pipeline."""
        if not self.vector_store:
            self.load_offline_knowledge_base()
        
        # 1. Retriever Selection
        if strategy == "mmr":
            base_retriever = get_vectorstore_retriever(self.vector_store, search_type="mmr", k=4)
        elif strategy == "compression":
            standard_retriever = get_vectorstore_retriever(self.vector_store, k=10)
            base_retriever = create_compression_retriever(self.embeddings, standard_retriever, k=4)
        else: 
            base_retriever = get_vectorstore_retriever(self.vector_store, k=4)

        # 2. Setup LLMs
        # We use a lower temperature for rephrasing to keep it precise
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
    
    async def ask_with_sources_async(self, question: str, session_id: str = "default_session"):
        """Async entry point for FastAPI."""
        if not self.chat_chain:
            self.create_conversational_chain()
        
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