from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Import the ChatEngine class from engine_deploy.py in the same app folder
from app.engine_deploy import ChatEngine

#LangSmith Test
from langchain.tools import tool
import requests

# --- API SCHEMAS ---

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"
    strategy: Optional[str] = "similarity"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str

# --- GLOBAL ENGINE INSTANCE ---

engine = ChatEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup logic to connect to Pinecone once.
    This runs before the server starts accepting requests.
    """
    try:
        # 1. Connect to Pinecone Cloud (Updated from load_offline_knowledge_base)
        engine.load_pinecone_knowledge_base() 
        
        # 2. Prepare the RAG chain
        engine.create_conversational_chain()
        
        print(f"✅ Initialization successful: Connected to Pinecone index '{engine.index_name}'.")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        raise e
    yield
    print("Shutting down server.")

app = FastAPI(
    title="Conversational RAG API",
    lifespan=lifespan
)

# Add this after your app instance
origins = [
    "https://cassiopeiai.com",  # allow your frontend
    "http://localhost:3000",    # if testing locally
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # or ["*"] for any origin (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Returns the status of the engine and vector store."""
    return {
        "status": "online",
        "vector_store_loaded": engine.vector_store is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Processes a chat message through the RAG pipeline with detailed error reporting.
    """
    # 1. Validation check
    if not request.message or request.message.strip() == "":
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # 2. Call the asynchronous RAG brain
        # This will trigger create_conversational_chain() if it hasn't run yet
        result = await engine.ask_with_sources_async(
            question=request.message,
            session_id=request.session_id
        )
        
        # 3. Safety check on the result dictionary
        if not result or "answer" not in result:
            raise ValueError("RAG Engine returned an empty or invalid response.")

        # 4. Return the validated Pydantic response
        return ChatResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            session_id=request.session_id
        )

    except FileNotFoundError as fnf:
        print(f"DATA ERROR: {fnf}")
        raise HTTPException(
            status_code=500, 
            detail=f"Vector Database missing: {str(fnf)}"
        )
        
    except Exception as e:
        # This prints the FULL error stack trace in your server terminal
        print(f"RAG SYSTEM ERROR: {type(e).__name__}")
        import traceback
        traceback.print_exc() 
        
        # This returns the specific error message to Postman
        error_msg = str(e) if str(e) else "Internal Engine Error (Check Server Logs)"
        raise HTTPException(
            status_code=500, 
            detail=f"RAG Engine Error: {error_msg}"
        )

if __name__ == "__main__":
    import uvicorn
    # When running from the root folder: python app/main.py
    # This matches your specified folder structure.
    uvicorn.run(app, host="0.0.0.0", port=8000)