from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

# Import the ChatEngine class from engine_deploy.py in the same app folder
from app.engine_deploy import ChatEngine

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
    Startup logic to load the pre-built FAISS index once.
    This runs before the server starts accepting requests.
    """
    try:
        # load_offline_knowledge_base handles the path to the root data folder
        engine.load_offline_knowledge_base()
        engine.create_conversational_chain()
        print("Initialization successful: FAISS index loaded and chain created.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        raise e
    yield
    print("Shutting down server.")

app = FastAPI(
    title="Conversational RAG API",
    lifespan=lifespan
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
    Processes a chat message through the RAG pipeline.
    Expects JSON input: {"message": "...", "session_id": "..."}
    """
    try:
        # Call the asynchronous RAG brain
        result = await engine.ask_with_sources_async(
            question=request.message,
            session_id=request.session_id
        )
        
        # Return the validated Pydantic response
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=request.session_id
        )
    except Exception as e:
        # Log the error for debugging and return a 500 error
        print(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG Engine Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # When running from the root folder: python app/main.py
    # This matches your specified folder structure.
    uvicorn.run(app, host="0.0.0.0", port=8000)