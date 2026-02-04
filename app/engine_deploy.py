import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Setup Environment and Path Logic
# __file__ is /root/app/engine_deploy.py
current_file_path = Path(__file__).resolve()
app_dir = current_file_path.parent
root_dir = app_dir.parent

# Load .env from root folder
load_dotenv(dotenv_path=root_dir / ".env")

# LangChain Imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Ensure langchain_functions can be imported from the app folder
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

from langchain_functions import (
    get_vectorstore_retriever,
    create_compression_retriever,
    instantiate_llm
)

REWRITER_SYSTEM_PROMPT = """
<task_context>
You are an expert Query Transformation Engine. Your goal is to convert a conversational follow-up into a context-independent Standalone Question for a RAG pipeline.
</task_context>

<rules>
1. CORE_TASK: Use 'Chat History' to resolve ambiguity in the 'Follow-up Input'.
2. NO_CHAT: Do NOT answer the question. 
3. PRONOUN_RESOLUTION: Replace 'it', 'they', etc., with specific entities from history.
4. LANGUAGE_PARITY: Output in the same language as the Follow-up Input.
5. NO_CHANGE_REQUIRED: If the input is already a clear standalone question, return it as-is.
</rules>

<output_format>
Return ONLY the rephrased string. No additional text.
</output_format>
"""

def get_answer_template(language: str = "English") -> ChatPromptTemplate:
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
        
        if not self.api_key:
            raise ValueError(f"API Key for {provider} not found in .env.")

        self.embeddings = OpenAIEmbeddings() 
        self.vector_store = None
        self.chat_chain = None
        self.history_store = {} 

    def get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def load_offline_knowledge_base(self):
        # Look for data folder in the root directory
        persist_path = root_dir / "data" / "vector_stores" / "faiss_index"
        
        if persist_path.exists():
            print(f"Loading index from: {persist_path}")
            self.vector_store = FAISS.load_local(
                str(persist_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            raise FileNotFoundError(f"FAISS index folder not found at: {persist_path}")
    
    def create_conversational_chain(self, strategy: str = "similarity"):
        if not self.vector_store:
            self.load_offline_knowledge_base()
        
        if strategy == "mmr":
            base_retriever = get_vectorstore_retriever(self.vector_store, search_type="mmr", k=4, fetch_k=20)
        elif strategy == "compression":
            standard_retriever = get_vectorstore_retriever(self.vector_store, k=10)
            base_retriever = create_compression_retriever(self.embeddings, standard_retriever, k=4)
        else: 
            base_retriever = get_vectorstore_retriever(self.vector_store, k=4)

        llm_rephraser = instantiate_llm(provider=self.provider, api_key=self.api_key, model_name="gpt-4o", temperature=0.0)
        llm_generator = instantiate_llm(provider=self.provider, api_key=self.api_key, model_name=self.model_name, temperature=0.5)

        rephrase_prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "<follow_up_input>\n{input}\n</follow_up_input>"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm=llm_rephraser,
            retriever=base_retriever,
            prompt=rephrase_prompt
        )

        qa_prompt = get_answer_template(language="English")
        question_answer_chain = create_stuff_documents_chain(llm=llm_generator, prompt=qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        self.chat_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    async def ask_with_sources_async(self, question: str, session_id: str = "default_session"):
        if not self.chat_chain:
            self.create_conversational_chain()
        
        response = await self.chat_chain.ainvoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
    
        context_docs = response.get("context", [])
        sources = list(set(doc.metadata.get("source", "Unknown") for doc in context_docs))

        return {
            "answer": response["answer"],
            "sources": sources
        }