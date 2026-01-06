"""
Sunjos - Your Cute Knowledge Q&A API
FastAPI backend for RAG-based document Q&A
"""

import os
import shutil
import uuid
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import get_rag_engine, RAGEngine

# Load environment variables
load_dotenv()

# Verify Groq API key
if not os.getenv("GROQ_API_KEY"):
    print("⚠️  Warning: GROQ_API_KEY not set. Create a .env file with your API key.")

# Initialize FastAPI app
app = FastAPI(
    title="Sunjos",
    description="Your Cute Knowledge Q&A Assistant",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# ─────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 4  # Number of chunks to retrieve


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    has_context: bool


class DocumentResponse(BaseModel):
    id: str
    filename: str
    chunks: int
    characters: int
    added_at: str


class ConversationItem(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    documents_count: int


# ─────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Sunjos",
        "description": "Your Cute Knowledge Q&A Assistant",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "POST /upload",
            "query": "POST /query",
            "documents": "GET /documents",
            "conversations": "GET /conversations"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and status"""
    engine = get_rag_engine()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        documents_count=len(engine.get_documents())
    )


@app.post("/upload", response_model=DocumentResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the knowledge base
    Supports: PDF, DOCX, TXT, MD
    """
    # Validate file extension
    filename = file.filename or "unknown"
    extension = os.path.splitext(filename)[1].lower()
    
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{extension}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Save file temporarily
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{extension}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process with RAG engine
        engine = get_rag_engine()
        doc_meta = engine.add_document(temp_path, filename)
        
        return DocumentResponse(**doc_meta)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query_documents(request: QueryRequest):
    """
    Ask a question about your documents
    Returns an answer with source citations
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        engine = get_rag_engine()
        result = engine.query(request.question, k=request.k or 4)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            has_context=result["has_context"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/documents", response_model=List[DocumentResponse], tags=["Documents"])
async def list_documents():
    """Get list of all documents in the knowledge base"""
    engine = get_rag_engine()
    documents = engine.get_documents()
    return [DocumentResponse(**doc) for doc in documents]


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Remove a document from the knowledge base"""
    engine = get_rag_engine()
    success = engine.remove_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully", "id": doc_id}


@app.get("/conversations", response_model=List[ConversationItem], tags=["Conversations"])
async def get_conversations(limit: int = 50):
    """Get conversation history"""
    engine = get_rag_engine()
    history = engine.get_conversation_history(limit)
    return [ConversationItem(**item) for item in history]


@app.delete("/conversations", tags=["Conversations"])
async def clear_conversations():
    """Clear conversation history"""
    engine = get_rag_engine()
    engine.clear_conversation_history()
    return {"message": "Conversation history cleared"}


# ─────────────────────────────────────────────────────────────
# Run server
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"""
    ╔══════════════════════════════════════════════╗
    ║      ✿ Sunjos - Cute Knowledge Q&A ✿        ║
    ║──────────────────────────────────────────────║
    ║  Server: http://{host}:{port}                   ║
    ║  Docs:   http://{host}:{port}/docs              ║
    ╚══════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=host, port=port)
